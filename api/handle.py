"""The lifetime of one companion child process; no server imports in the kernel."""
from __future__ import annotations

import atexit
from collections import deque
import os
import queue
import selectors
import shutil
import subprocess
import threading
import time
import warnings

from .. import locate, portable
from . import protocol, window
from .result import CompanionError, CompanionStartError, CompanionWarning

_APPS = set()
_APPS_LOCK = threading.Lock()
OPEN_WAIT_S = 20.0           # a browser opened for the app: a link is shown too when no page signed in within this
SIGN_IN_LOOK_S = 2.0         # how often the engine is asked meanwhile (GET /health presence)
LINKS_KEPT = 16              # the links shown in the notebook that the handle can still replace


def _windows(health: dict) -> int:
    """The pages of browsers the app opened that signed in so far, as the engine's /health says (presence.windows: the
    window launch codes traded; another tab connecting is none of them); 0 when it does not say."""
    windows = (health.get("presence") or {}).get("windows") if isinstance(health, dict) else None
    return windows if type(windows) is int else 0


def _reap(process) -> None:
    """`process` waited for: a second to end by itself, then asked to (SIGTERM), then made to (SIGKILL)."""
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _shutdown():
    # EOF asks the engine to discard; never run SpatialData code during interpreter shutdown.
    with _APPS_LOCK:
        apps = tuple(_APPS)
    for app in apps:
        app._shutdown.set()
        if app._process is not None:
            app._process.stdin.close()


atexit.register(_shutdown)


class CompanionApp:
    """An open app. Saves go into your SpatialData automatically, between notebook cells.

    ``close()`` discards work you did not save in the app; ``close(pending="add")`` adds it to your SpatialData first.
    ``open()`` opens it again when its window or tab went away. ``added`` is a tuple of the element names successfully
    added to the object so far.
    """

    def __init__(self, *, sdata, folders, config, binding, snapshot, schedule, timeout, settings=None):
        self._sdata, self._folders, self._config = sdata, folders, config
        self._binding, self._snapshot, self._schedule = binding, snapshot, schedule
        self._timeout = timeout
        self._settings = dict(settings or {})    # the user's [desktop] table: where the app opens, Chrome's flags
        self._state, self._url, self._added = "starting", None, []
        self._lock = threading.RLock()
        self._outgoing = queue.Queue()
        self._shutdown, self._done = threading.Event(), threading.Event()
        self._process = self._thread = self._delivery = None
        self._failure = None
        self._failed_at = None
        self._owner_thread = threading.get_ident()
        self._events_fd = None
        self._ended = False
        self._port = None
        self._started = time.monotonic()
        self._links = []                   # the links shown in the notebook (window.Shown), oldest first
        self._chrome = []                  # the session's own Chrome windows started (their processes; None: unknown)
        # A browser opened and its page not signed in yet: {since, before: window sign-ins then, chrome, shown: when the
        # link to open by hand was shown (None: not yet)}.
        self._opened_at = None
        self._pages = []                   # [(opening page, when it is removed: its code no longer signs in)]
        self._open_once = False            # the app was opened once (the engine was ready): open() may open it again

    @property
    def url(self) -> str | None:
        """The one-use opening link, once the engine is listening."""
        with self._lock:
            return self._url

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def added(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._added)

    def __repr__(self) -> str:
        return f"<CompanionApp {self.state}; {len(self.added)} elements added>"

    def _send(self, op: str | dict, **fields) -> None:
        """For api/deliver.py: queue a validated control message; never block the kernel on a full pipe."""
        value = protocol.validate_control(op) if isinstance(op, dict) else protocol.build_control(op, **fields)
        if not self._done.is_set():
            self._outgoing.put(protocol.encode_line(value).encode())

    def _record_added(self, keys) -> None:
        """For api/deliver.py, called only after an atomic apply succeeded on the owning thread."""
        if threading.get_ident() != self._owner_thread:
            raise CompanionError("Results must be added on the thread that opened the app")
        if not isinstance(keys, (list, tuple)) or not all(isinstance(key, str) and key for key in keys):
            raise ValueError("Added element names must be text")
        with self._lock:
            self._added.extend(key for key in keys if key not in self._added)

    def _start(self, python: str) -> None:
        # On Linux the engine gets SIGTERM should the kernel go without closing it (setpriv's death signal). macOS and
        # Windows have no such signal: there the end of its control pipe, which a kernel's end closes whatever ends it,
        # tells the engine to close (engine/session.py), as it does on Linux too.
        launcher = []
        if portable.LINUX:
            if not (setpriv := shutil.which("setpriv")):
                raise CompanionError("The app needs the Linux setpriv command")
            launcher = [setpriv, "--pdeathsig", "TERM"]
        # The engine's interpreter imports this very copy by itself (installed in its environment, as the kernel's
        # is): no PYTHONPATH, and the kernel's own never reaches it. Otherwise the session's pylib, made now.
        environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        environment.pop("PYTHONPATH", None)
        if not locate.imports_this_copy(python, environment, cwd=self._folders.state):
            environment["PYTHONPATH"] = str(self._folders.pylib())
        events_r, events_w = os.pipe()
        self._events_fd = events_r
        environment[protocol.ENV_EVENTS_FD], passing = portable.pass_fd(events_w)
        command = [*launcher, python, "-m", "sopa_companion", "serve",
                   "--config", str(self._folders.state / "session.toml")]
        try:
            with (self._folders.state / "logs/server.log").open("ab") as log:
                self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                                 cwd=self._folders.state, env=portable.child_environment(environment),
                                                 close_fds=True, bufsize=0, **passing, **portable.new_session())
            self._folders.engine_started(self._process.pid)
            os.set_blocking(events_r, False)
            os.set_blocking(self._process.stdin.fileno(), False)
            self._send("hello", protocol=protocol.PROTOCOL_VERSION)
            self._send("bind", **self._binding)
            self._send("object", **self._snapshot())
            with _APPS_LOCK:
                _APPS.add(self)
            self._thread = threading.Thread(target=self._io, name="companion-io-" + self._folders.id, daemon=True)
            self._thread.start()
        except BaseException:
            if self._process is not None:
                self._process.kill()
                self._process.wait()
                self._process.stdin.close()
            os.close(events_r)
            self._events_fd = None
            with _APPS_LOCK:
                _APPS.discard(self)
            raise
        finally:
            os.close(events_w)

    def _redact(self, text: str) -> str:
        return text.replace(self._config["server"]["token"], "[private]")

    def _fail(self, message: str, *, start=False) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._state = "failed"
            self._failed_at = time.monotonic()
            tail = ""
            try:
                with (self._folders.state / "logs/server.log").open(encoding="utf-8", errors="replace") as log:
                    tail = "\n".join(line.rstrip() for line in deque(log, maxlen=30))
            except OSError:
                pass
            error = CompanionStartError if start else CompanionError
            self._failure = error(self._redact(message + ("\n" + tail if tail else "")))

    def _event(self, event: dict) -> None:
        name = event["ev"]
        if name == "ready":
            if not self._is_engine(event["pid"]) or self._port is not None:
                raise ValueError("The app sent an unexpected engine identity")
            self._port = event["port"]
        elif name == "session":
            if event["state"] == "failed":
                self._fail(event["message"] or "The app could not open the sample", start=self._url is None)
                self._send("close", pending=protocol.DEFAULT_CLOSE_POLICY)
            else:
                with self._lock:
                    if self._state not in ("failed", "closing", "closed"):
                        self._state = "closing" if event["state"] == "closed" else event["state"]
        elif name == "deliver":
            if self._delivery is None:
                self._send("receive_failed", id=event["id"], message="This build could not add the result to your object")
            else:
                self._delivery.receive(event)
        elif name == "closing":
            with self._lock:
                if self._state != "failed":
                    self._state = "closing"
        elif name == "ended":
            # The ended event's `added` holds delivery ids, not element names: api/deliver.py records the keys.
            self._ended = True
        elif name == "link":
            # A one-use link signed a browser in. The engine cannot say which: the newest link not said to be used yet
            # says so (the one still showing a code, if any), so no code is left in the notebook and a use that was not
            # the user's shows.
            with self._lock:
                link = next((shown for shown in reversed(self._links) if shown.state != "used"), None)
            if link is not None and event["state"] == "used":
                link.used(time.time())

    def _is_engine(self, pid) -> bool:
        """Whether the ready event's pid is the engine this handle started: the process itself, or on Windows the
        interpreter that a virtual environment's python.exe (a launcher, which waits for it and passes it the pipes)
        starts as its only child."""
        if type(pid) is not int:
            return False
        if pid == self._process.pid:
            return True
        if not portable.WINDOWS:
            return False
        try:
            import psutil
            return psutil.Process(pid).ppid() == self._process.pid
        except Exception:
            return False

    def _show(self, url: str, *, why: str | None = None) -> None:
        """The link `url` (with `why`) shown in the notebook, the one to open by hand from now on (app.url): every link
        shown before it that still shows its code is taken out, so the notebook holds one live code at most."""
        shown = window.show_link(url, self._port, why=why, schedule=self._schedule)
        with self._lock:
            earlier, self._links = self._links, [*self._links, shown][-LINKS_KEPT:]
            self._url = url
        for link in earlier:
            link.take_out(window.TAKEN_OUT["newer"])

    def _open(self) -> None:
        port, token, timeout = self._port, self._config["server"]["token"], min(0.5, self._timeout)
        before = _windows(window.request(port, token, "/api/v1/health", timeout=timeout))
        mode, session = self._config["session"]["open"], self._config["session"]["id"]
        # A window opened now takes a code that lives seconds; a link a person opens by hand (printed, app.url) one the
        # engine keeps for ten minutes. Either names the session, whose tab then keeps its storage for it.
        url = window.launch_url(port, token, link=mode != "window", session=session, timeout=timeout)
        if mode == "window":
            # The browser opens a private page in the session's folder (or a snap browser's own) that sends it on to
            # `url`: a code on its command line could be read, and used first, by any account on this computer.
            with (self._folders.state / "logs/browser.log").open("ab") as log:
                opened = window.open_window(url, folder=self._folders.state, settings=self._settings, log=log)
            if opened.notice:
                print(opened.notice, flush=True)
            if opened.how == "link":       # no browser after all: a link to open by hand, never the window's code
                self._show(window.launch_url(port, token, link=True, session=session, timeout=timeout), why=opened.why)
                return
            with self._lock:
                self._url = url
                if opened.how == "chrome":
                    self._chrome.append(opened.process)
                if opened.page is not None:
                    self._pages.append((opened.page, time.monotonic() + window.PAGE_KEEP_S))
                # Its page's sign-in is looked for (_not_yet): a browser that took the page may never show it, and a
                # link shown before is not needed once it signed in.
                self._opened_at = {"since": time.monotonic(), "before": before, "chrome": opened.how == "chrome",
                                   "shown": None}
        elif mode == "link":
            self._show(url)
        else:
            with self._lock:
                self._url = url

    def _not_yet(self, now: float) -> None:
        """A browser was opened and its page has not signed in since (presence.windows: its launch code traded; another
        tab connecting is no sign of it). After OPEN_WAIT_S without, a link to open by hand (none for the app's own
        Chrome window, which is not waited for). Once it signs in, every link still shown is taken out: none is needed
        any more. The looking ends then, at OPEN_WAIT_S for the Chrome window, or once the link shown here expired."""
        with self._lock:
            waiting = self._opened_at
        if waiting is None:
            return
        try:
            health = window.request(self._port, self._config["server"]["token"], "/api/v1/health",
                                    timeout=min(0.5, self._timeout))
            signed = _windows(health) > waiting["before"]
        except (OSError, ValueError):
            signed = False
        if signed:
            ahead = None
        elif waiting["shown"] is not None:
            if now - waiting["shown"] < self._config["session"].get("link_code_ttl_s", 600):
                return                     # the link shown can still sign a browser in
            ahead = None
        elif now - waiting["since"] < OPEN_WAIT_S:
            return
        else:
            ahead = None if waiting["chrome"] else {**waiting, "shown": now}
        with self._lock:
            if self._opened_at is not waiting:
                return                     # open() opened it again meanwhile
            self._opened_at = ahead
            links = list(self._links) if signed else []
        for link in links:
            link.take_out(window.TAKEN_OUT["opened"])
        if ahead is not None:
            self._show(window.launch_url(self._port, self._config["server"]["token"], link=True,
                                         session=self._config["session"]["id"], timeout=min(0.5, self._timeout)),
                       why=window.NOT_YET)

    def _remove_pages(self, now: float | None = None) -> None:
        """The opening pages whose codes no longer sign in (all of them without `now`), wherever they are: the
        session's folder, or a snap browser's own."""
        with self._lock:
            done = [page for page, until in self._pages if now is None or now >= until]
            self._pages = [(page, until) for page, until in self._pages if page not in done]
        for page in done:
            try:
                page.unlink(missing_ok=True)
            except OSError:
                pass
        if done:
            window.remove_pages(self._folders.state, older_than=None if now is None else window.PAGE_KEEP_S)

    def _io(self) -> None:
        process, pending, received = self._process, bytearray(), bytearray()
        last_object, last_poll, last_look = None, 0.0, 0.0
        opened = False
        selector = portable.pipe_selector()
        writing, eof, exited_at, ended_at = False, False, None, None
        try:
            selector.register(self._events_fd, selectors.EVENT_READ)
            control_fd = process.stdin.fileno()
            while True:
                if self._shutdown.is_set():
                    break
                now = time.monotonic()
                if self._failed_at is not None and now - self._failed_at > 1:
                    break
                if self._ended:
                    ended_at = ended_at or now
                    if now - ended_at > 2:
                        self._fail("The app did not exit after closing")
                        break
                if not opened and self.state not in ("failed", "closing") and now - self._started >= self._timeout:
                    self._fail("The app did not start in time", start=True)
                    break
                if process.poll() is not None:
                    exited_at = exited_at or now
                    if eof or now - exited_at > 0.2:
                        break
                if self._port is not None and not opened and self.state not in ("failed", "closing"):
                    try:
                        self._open()
                        opened = True
                        with self._lock:
                            self._open_once = True
                    except (OSError, ValueError):
                        pass  # ready can precede the first healthy HTTP response
                if self._pages:
                    self._remove_pages(now)                        # their codes are used or expired by now
                if self._opened_at is not None and now - last_look >= SIGN_IN_LOOK_S and self.state not in (
                        "failed", "closing"):
                    last_look = now
                    try:
                        self._not_yet(now)
                    except (OSError, ValueError):
                        pass  # the engine is going: nothing to open any more
                if now - last_poll >= 2 and process.poll() is None:
                    value = self._snapshot()
                    if value != last_object:
                        self._send("object", **value)
                        last_object = value
                    last_poll = now
                if not pending:
                    try:
                        pending.extend(self._outgoing.get_nowait())
                    except queue.Empty:
                        pass
                if pending and not writing:
                    selector.register(control_fd, selectors.EVENT_WRITE)
                    writing = True
                for key, _ in selector.select(0.05):
                    if key.fd == control_fd:
                        try:
                            del pending[:os.write(control_fd, pending)]
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            pending.clear()
                        except OSError:
                            if not portable.WINDOWS:
                                raise
                            pending.clear()                # Windows' word for a pipe whose reader is gone
                        if not pending:
                            selector.unregister(control_fd)
                            writing = False
                    else:
                        try:
                            chunk = os.read(self._events_fd, 65536)
                        except BlockingIOError:            # Windows: the pipe looked ready and was not
                            continue
                        if not chunk:
                            selector.unregister(self._events_fd)
                            eof = True
                            continue
                        received.extend(chunk)
                        while b"\n" in received:
                            line, _, rest = received.partition(b"\n")
                            received[:] = rest
                            if len(line) > protocol.MAX_EVENT_BYTES:
                                raise ValueError("The app sent an oversized message")
                            event = protocol.decode_event_line(bytes(line))
                            if event is not None:
                                self._event(event)
                        if len(received) > protocol.MAX_EVENT_BYTES:
                            raise ValueError("The app sent an unfinished oversized message")
                if eof and process.poll() is None:
                    if self._ended:
                        continue
                    raise ValueError("The app closed its messages before ending")
            if not self._ended and self._failure is None and not self._shutdown.is_set():
                self._fail(f"The app stopped unexpectedly (exit code {process.poll()})", start=not opened)
        except Exception as exc:
            self._fail(f"The app connection failed: {exc}", start=not opened)
        finally:
            try:
                self._end(process, selector)
            except Exception as exc:       # whatever it was, the handle is done: close() never waits for this thread
                self._fail(f"The app did not close cleanly: {exc}")
            with self._lock:
                if self._failure is None:
                    self._state = "closed"
            with _APPS_LOCK:
                _APPS.discard(self)
            self._done.set()
            if self._schedule is not None:
                try:
                    self._schedule(self._finish_delivery)
                except RuntimeError:
                    pass  # the kernel's loop may already be shutting down
                if self._failure is not None:
                    warnings.warn(str(self._failure), CompanionWarning, stacklevel=1)

    def _end(self, process, selector) -> None:
        """The session's end, as its thread goes: the engine's pipes closed and the engine reaped, the session's own
        Chrome window closed when one was opened (whatever opened since: a tab of the user's own browser is the
        user's), the opening pages and the session's folders removed."""
        selector.close()
        process.stdin.close()
        os.close(self._events_fd)
        self._events_fd = None
        if process.poll() is None:
            _reap(process)
        try:
            with self._lock:
                chrome = list(self._chrome)
            if chrome:
                window.close_profile(self._folders.state / "chrome-profile")
                for started in chrome:
                    if started is not None:
                        _reap(started)
            self._remove_pages()
            self._folders.remove()
        except OSError as exc:
            self._fail(f"The app could not remove its temporary folders: {exc}")

    def _apply_pending(self) -> None:
        if self._delivery is not None:
            self._delivery.apply_pending()

    def _finish_delivery(self) -> None:
        if self._delivery is not None:
            self._delivery.apply_pending()
            self._delivery.close()
            self._delivery = None

    def _run_until_closed(self) -> None:
        """The script's main-thread event pump; deliberately not a public wait method."""
        try:
            while not self._done.wait(0.02):
                self._apply_pending()
            self._thread.join()
            self._finish_delivery()
        except BaseException:
            self.close(pending=protocol.DEFAULT_CLOSE_POLICY)
            raise
        if self._failure is not None:
            raise self._failure

    def open(self) -> None:
        """Open the app again, for a window or tab that went away (a crash, a closed laptop); the session goes on
        meanwhile. A new window when it opens one, else a new link (with open="none", a new ``url``)."""
        if threading.get_ident() != self._owner_thread:
            raise CompanionError("Open the app on the thread that opened it")
        with self._lock:
            state, once = self._state, self._open_once
        if state not in ("ready", "starting", "preparing") or self._done.is_set():
            raise CompanionError("The app is not open")
        if not once:
            return                         # the engine is starting: the app opens by itself once it is ready
        try:
            self._open()
        except (OSError, ValueError) as exc:
            raise CompanionError(f"The app could not be opened again: {exc}") from None

    def close(self, pending: str = protocol.DEFAULT_CLOSE_POLICY, timeout: float | None = None) -> None:
        """Close the app and remove its folders.

        Work you did not save in the app is discarded, as when the window is closed without an answer; pass
        ``pending="add"`` to add it to your SpatialData first. A timeout discards work still in progress; results
        already saved stay in your SpatialData.
        """
        protocol.build_control("close", pending=pending)
        if threading.get_ident() != self._owner_thread:
            raise CompanionError("Close the app on the thread that opened it")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout < float("inf")):
            raise ValueError("timeout must be a finite number of seconds, or None")
        if not self._done.is_set():
            with self._lock:
                if self._state != "failed":
                    self._state = "closing"
            self._send("close", pending=pending)
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._done.wait(0.02):
                self._apply_pending()
                if deadline is not None and time.monotonic() >= deadline:
                    self._shutdown.set()
                    deadline = None
        if self._thread is not None:
            self._thread.join()
        self._finish_delivery()
