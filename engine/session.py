"""One notebook's API session and its private JSON-line pipes."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
import select
import sys
import threading
import time
import tomllib

from .. import portable
from ..api import protocol

LOG = logging.getLogger(__name__)
MAX_CONTROL_BYTES = 1024 * 1024
# How long one event line may wait for room in the events pipe while the notebook reads (it reads every 50 ms, and
# only stops while it loads a result): the line is written whole, in as many parts as the pipe takes.
EVENTS_WRITE_S = 30.0


def _release(fd: int) -> None:
    """Let go of whatever `fd` leads to while keeping its number taken (by /dev/null, closed on exec): nothing opened
    later gets that number, so nothing can write into a stranger's file through it."""
    null = os.open(os.devnull, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.dup2(null, fd, inheritable=False)
    finally:
        os.close(null)


def read_session_config(path) -> dict:
    """Read session.toml before an ephemeral listener replaces server.port with its actual port."""
    with Path(path).open("rb") as stream:
        return protocol.validate_session_config(tomllib.load(stream))


class Session:
    """The session as Runs and the server use it. start() owns duplicate descriptors; stop() leaves the caller's open.
    The events descriptor the handle passes in SOPA_COMPANION_EVENTS_FD is this Session's alone, let go of as it is made
    (_release).

    bind(path, *, name, points_key) returns the inspected sample detail. It runs on its own thread, so a slow
    inspection never holds up receipt acknowledgements or the kernel-gone notification. on_close() only schedules
    shutdown; the control reader must remain available throughout the app's teardown.
    """

    mode = "api"

    def __init__(self, config, *, control_fd=None, events_fd=None, bind=None, on_close=None, clock=time.monotonic):
        self.config = protocol.validate_session_config(config)
        self.settings = self.config["session"]
        self.bound, self.view_path, self.notebook_busy = None, None, False
        self.version = 0
        self.kernel_gone = False
        self.bind_sample, self.on_close, self.clock = bind, on_close, clock
        self._control_source = control_fd
        inherited = events_fd is None and protocol.ENV_EVENTS_FD in os.environ
        value = os.environ.pop(protocol.ENV_EVENTS_FD) if inherited else events_fd
        if inherited and portable.WINDOWS:
            # Windows passes the pipe's handle (portable.pass_fd): a descriptor of it, copied, and the inherited one
            # closed, so the copy is the pipe's only end here, as below.
            passed = portable.take_fd(value, write=True)
            self._events_fd = os.dup(passed)
            os.close(passed)
            inherited = False
        else:
            self._events_fd = os.dup(int(value)) if value is not None else None
        if self._events_fd is not None:
            os.set_inheritable(self._events_fd, False)
            os.set_blocking(self._events_fd, False)
        if inherited:
            # The descriptor the handle passed is this Session's alone: its pipe is let go of here (its number now
            # leads to /dev/null), so once the Session closes its copy the notebook reads the end of the channel,
            # never a channel that stays open with nobody writing.
            _release(int(value))
        self._events_lock = threading.Lock()
        self._condition = threading.Condition(threading.RLock())
        self._stopped = threading.Event()
        self._reader = self._binder = None
        self._wake = None
        self._binding = False
        self._names, self._object_zarr, self._auto_save = set(), None, False
        self._answer, self._answer_at, self._policy = None, None, None
        # The runs the kept answer is for (None: every run pending), the runs the closing policy is for (None: every
        # run: the handle's close, the kernel gone, no answer), and the last revision taken from each window (None: a
        # page that names none), as server/app.py ClosingAnswer keeps them.
        self._answer_runs, self._policy_runs, self._revisions = frozenset(), None, {}
        self._policy_from_answer, self._answers_taken = False, 0
        self._ending = False
        self._received, self._applied, self._deliveries = {}, {}, {}
        self._send_failed, self._not_added = set(), []
        self._sample = {"id": None, "name": None, "backed": False, "zarr_name": None, "view": None,
                        "state": "opening", "progress": {"done": 0, "total": 0}, "message": None,
                        "memory_only": [], "tools": []}

    def _changed(self):
        self.version += 1
        self._condition.notify_all()

    def public(self) -> dict:
        with self._condition:
            writes = "zarr" if self._object_zarr and self._auto_save else (
                "memory_autosave_off" if self._object_zarr else "memory")
            return protocol.build_health_session(
                self.mode, id=self.settings["id"], sample=self._sample, tool=self.settings["tool"] or None,
                writes=writes, notebook={"busy": self.notebook_busy}, patches=bool(self.settings["patches_dir"]),
                link=self.settings["open"] == "link", state="ending" if self._ending else "running")

    def event(self, since=-1) -> tuple[int, dict | None]:
        """A detached health.session push, once per version; the session event to the kernel is a different shape."""
        with self._condition:
            return self.version, {"ev": "session", **self.public()} if since != self.version else None

    def target(self) -> str | None:
        with self._condition:
            return self._object_zarr if self._auto_save else None

    def names(self) -> set[str]:
        with self._condition:
            return set(self._names)

    def deliver(self, event: dict) -> None:
        """Send one event to the kernel as one whole line (_write_line), in the order deliver was called."""
        event = protocol.validate_event(event)
        payload = protocol.encode_line(event).encode("utf-8")
        with self._condition:
            if event["ev"] == "deliver":
                self._deliveries[event["id"]] = event
        failed = False
        with self._events_lock:
            try:
                if self._events_fd is None:
                    raise BrokenPipeError("events pipe is closed")
                if len(payload) > protocol.MAX_EVENT_BYTES:
                    raise ValueError(f"the message is {len(payload):,} bytes, more than the notebook reads")
                self._write_line(payload)
            except (OSError, ValueError) as exc:
                failed = True
                LOG.warning("API event %s was not sent: %s", event["ev"], exc)
        if failed and event["ev"] == "deliver":
            with self._condition:
                self._send_failed.add(event["id"])
                self._condition.notify_all()

    def _write_line(self, payload: bytes) -> None:
        """Write `payload` whole into the non-blocking events pipe, waiting for room as the notebook reads, for up to
        EVENTS_WRITE_S. A line of which nothing was written by then is not sent (BlockingIOError), and the channel stays
        as it was; one cut off part-way ends the channel, so the notebook reads its end after the unfinished line rather
        than another message run into it. Called with the events lock held."""
        view, deadline = memoryview(payload), time.monotonic() + EVENTS_WRITE_S
        while view:
            try:
                view = view[os.write(self._events_fd, view):]
                continue
            except BlockingIOError:
                pass
            left = deadline - time.monotonic()
            if left > 0 and portable.writable(self._events_fd, left):
                continue
            if len(view) == len(payload):
                raise BlockingIOError(f"the notebook made no room for it in {EVENTS_WRITE_S:g} s")
            os.close(self._events_fd)
            self._events_fd = None
            raise BrokenPipeError(f"the notebook stopped reading part-way through it for {EVENTS_WRITE_S:g} s; the "
                                  "channel is closed")

    def received(self, id, timeout) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: id in self._received or id in self._send_failed or self.kernel_gone or
                                     self._stopped.is_set(), timeout=max(0, timeout))
            return self._received.get(id, False)

    def applied(self, id) -> dict | None:
        with self._condition:
            return copy.deepcopy(self._applied.get(id))

    def closing_policy(self) -> str | None:
        with self._condition:
            return self._policy if self._ending else self._answer

    def closing(self) -> dict:
        """{answer, age_s, applies, covers} as the desktop's ClosingAnswer.public gives them: `applies` while closing
        would carry the answer out for some run (`covers`; null: every run pending), never for an answer that covers
        none."""
        with self._condition:
            age = None if self._answer_at is None else round(max(0, self.clock() - self._answer_at), 1)
            runs = self._policy_runs if self._ending else self._answer_runs if self._answer is not None else frozenset()
            return {"answer": self._policy if self._ending else self._answer, "age_s": age,
                    "applies": (self._ending or self._answer is not None) and runs != frozenset(),
                    "covers": None if runs is None else sorted(runs)}

    def closing_covers(self) -> frozenset | None:
        """The runs the closing policy is for once closing began; None: every run."""
        with self._condition:
            return self._policy_runs

    def answer_closing(self, body) -> int:
        """The page's answer (server/api_serve.py read_closing's {answer, runs, revision, page, end}): kept for
        the runs it names (none: every run pending; cover_closing narrows it to those), unless a later choice of its
        window was taken first; "withdraw" takes the kept one back. With end it starts closing (by="page"). -> the
        taken answer's number for cover_closing, 0 when it changed nothing."""
        if body.get("answer") != "withdraw":
            body = {**body, **protocol.validate_closing({k: v for k, v in body.items() if v is not None})}
        answer, runs, revision, page = body["answer"], body.get("runs"), body.get("revision"), body.get("page")
        with self._condition:
            if self._ending:
                return 0
            last = self._revisions.get(page)
            if revision is not None and last is not None and revision < last:
                return 0
            if revision is not None:
                self._revisions.pop(page, None)
                self._revisions[page] = revision
                while len(self._revisions) > 64:
                    self._revisions.pop(next(iter(self._revisions)))
            if answer == "withdraw":
                self._answer, self._answer_runs = None, frozenset()
            else:
                self._answer, self._answer_at = answer, self.clock()
                self._answer_runs = None if runs is None else frozenset(runs)
            self._answers_taken += 1
            taken = self._answers_taken
        if answer != "withdraw" and body.get("end", False):
            self.begin_close(by="page")
        return taken

    def cover_closing(self, taken, pending) -> None:
        """Answer number `taken` (answer_closing's), if it is still the kept one, is for the runs pending when it came
        (their ids) that it named, or every one of them when it named none: never for a run started later."""
        pending = frozenset(pending)
        with self._condition:
            if self._answer is None or self._answers_taken != taken:
                return
            self._answer_runs = pending if self._answer_runs is None else self._answer_runs & pending
            if self._ending and self._policy_from_answer:
                self._policy_runs = self._answer_runs

    def withdraw_closing(self) -> None:
        """A page connected again (a reload): an answer given before no longer counts."""
        with self._condition:
            if not self._ending:
                self._answer, self._answer_runs = None, frozenset()
                self._revisions.pop(None, None)

    def begin_close(self, pending=None, *, by="window") -> None:
        """The first close decides. A page's answer (with end, or kept from a beacon for the window watcher) or none
        (discard); the handle's close is its caller's own answer and wins over a kept beacon's (the popup, then the
        handle, then the beacon, then the window watcher); the kernel gone discards."""
        policy = protocol.DEFAULT_CLOSE_POLICY if pending is None else pending
        protocol.build_event("closing", pending=policy, by=by)
        with self._condition:
            first = not self._ending
            if by == "kernel":
                self.kernel_gone = True
                self._policy, self._policy_runs = protocol.DEFAULT_CLOSE_POLICY, None
                self._policy_from_answer = False
            elif first and by == "handle":
                self._policy, self._policy_runs = policy, None
            elif first:
                self._policy_from_answer = self._answer is not None
                self._policy = self._answer or policy
                self._policy_runs = self._answer_runs if self._answer is not None else None
            else:
                return
            self._ending = True
            self._changed()
            policy = self._policy
        self.deliver(protocol.build_event("closing", pending=policy, by=by))
        self.deliver(protocol.build_event("session", state="closing", message=None))
        if first and self.on_close is not None:
            self.on_close()

    def handle(self, message) -> None:
        """Handle a validated control message (also useful to callers with an in-process fake)."""
        message = protocol.validate_control(message, allow_unknown=True)
        if message is None:
            return
        op = message["op"]
        if op == "close":
            self.begin_close(message["pending"], by="handle")
        elif op == "bind":
            self._start_bind(message)
        elif op == "prepare_failed":
            self._sample_status("failed", message["message"])
        elif op == "prepare":
            with self._condition:
                if self._ending or self._binding or self.bound is not None:
                    return
                self._sample["progress"] = {"done": message["done"], "total": message["total"]}
            self._sample_status("preparing")
        elif op == "object":
            with self._condition:
                self._names = set(message["names"])
                self._object_zarr, self._auto_save = message["zarr"], message["auto_save"]
                if self._sample["backed"]:
                    self._sample["zarr_name"] = Path(message["zarr"]).name if message["zarr"] else None
                self._changed()
        elif op == "notebook":
            with self._condition:
                if self.notebook_busy != message["busy"]:
                    self.notebook_busy = message["busy"]
                    self._changed()
        elif op in ("received", "receive_failed", "applied"):
            with self._condition:
                if op == "applied":
                    self._applied[message["id"]] = message
                else:
                    self._received[message["id"]] = op == "received"
                self._condition.notify_all()
        # hello is validated by the protocol; it needs no reply.

    def _sample_status(self, state, message=None):
        with self._condition:
            if self._ending:
                return
            self._sample.update(state=state, message=message)
            self._changed()
        status = "starting" if state == "opening" else state
        self.deliver(protocol.build_event("session", state=status, message=message))

    def _start_bind(self, message):
        with self._condition:
            if self._ending or self._binding or self.bound is not None:
                LOG.warning("Ignoring a second bind or a bind while the API session is ending")
                return
            self._binding = True
            self.view_path = message["path"]
            self._sample.update(name=message["name"], backed=message["backed"], view=message["view"],
                                zarr_name=Path(message["zarr"]).name if message["zarr"] else None,
                                memory_only=message["memory_only"])
        self._sample_status("opening")

        def inspect_sample():
            try:
                if self.bind_sample is None:
                    raise RuntimeError("The sample opener is not attached")
                detail = self.bind_sample(message["path"], name=message["name"], points_key=message["points_key"])
                with self._condition:
                    if self._ending or self._stopped.is_set():
                        return
                    self.bound = detail["id"]
                    self._sample.update(id=self.bound, tools=list(detail["tools"]))
                self._sample_status("ready")
            except Exception as exc:
                LOG.warning("Opening the API session's sample failed: %s", exc)
                lines = str(exc).splitlines()
                self._sample_status("failed", lines[0] if lines else "The sample could not be opened")
            finally:
                with self._condition:
                    self._binding = False
                    self._condition.notify_all()

        self._binder = threading.Thread(target=inspect_sample, name="companion-bind", daemon=True)
        self._binder.start()

    def start(self) -> None:
        if self._reader is not None:
            return
        source = sys.stdin.fileno() if self._control_source is None else self._control_source
        control = os.dup(source)
        self._wake = os.pipe()
        self._reader = threading.Thread(target=self._read, args=(control,), name="companion-control", daemon=True)
        self._reader.start()

    def _read(self, control):
        buffer, dropping = b"", False
        try:
            while not self._stopped.is_set():
                if portable.WINDOWS:
                    # Windows' select takes sockets only: the pipe is looked at every 10 ms, and stop() is seen
                    # within 50 ms (the wake pipe is for select).
                    if not portable.readable(control, 0.05):
                        continue
                else:
                    readable, _, _ = select.select([control, self._wake[0]], [], [])
                    if self._wake[0] in readable:
                        break
                chunk = os.read(control, 65536)
                if not chunk:
                    if buffer and not dropping:
                        self._read_line(buffer)
                    self.begin_close(by="kernel")
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not dropping and len(line) <= MAX_CONTROL_BYTES:
                        self._read_line(line)
                    elif not dropping:
                        LOG.warning("Ignoring an API control line larger than %s bytes", MAX_CONTROL_BYTES)
                    dropping = False
                if len(buffer) > MAX_CONTROL_BYTES:
                    LOG.warning("Ignoring an API control line larger than %s bytes", MAX_CONTROL_BYTES)
                    buffer, dropping = b"", True
        except OSError as exc:
            if not self._stopped.is_set():
                LOG.warning("API control pipe closed: %s", exc)
                self.begin_close(by="kernel")
        finally:
            os.close(control)

    def _read_line(self, line):
        try:
            message = protocol.decode_control_line(line)
            if message is not None:
                self.handle(message)
        except (ValueError, TypeError) as exc:
            LOG.warning("Ignoring an invalid API control message: %s", exc)

    def note_not_added(self, run, reason, message) -> None:
        """The runs discarded or failed before a delivery existed (Runs.close_session), for the ended event."""
        item = {"run": run, "reason": reason, "message": message}
        protocol.build_event("ended", added=[], not_added=[item])
        with self._condition:
            self._not_added.append(item)

    def finish(self) -> None:
        with self._condition:
            added, not_added = [], copy.deepcopy(self._not_added)
            for id, event in self._deliveries.items():
                applied = self._applied.get(id)
                if applied is not None and applied["state"] == "added":
                    added.append(id)
                elif applied is not None:
                    not_added.append({"run": event["run"], "reason": "clash" if applied["clash"] else "failed",
                                      "message": applied["message"] or "The result could not be added"})
                elif not self._received.get(id, False):
                    not_added.append({"run": event["run"], "reason": "incomplete",
                                      "message": "The notebook did not receive the result"})
        self.deliver(protocol.build_event("session", state="closed", message=None))
        self.deliver(protocol.build_event("ended", added=added, not_added=not_added))

    def stop(self) -> None:
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        if self._wake is not None:
            os.write(self._wake[1], b"x")
            self._reader.join()
            for fd in self._wake:
                os.close(fd)
            self._wake = None
        if self._binder is not None:
            self._binder.join(timeout=1)
            if self._binder.is_alive():
                LOG.warning("The API sample opener did not stop after its helpers closed")
        with self._events_lock:
            if self._events_fd is not None:
                os.close(self._events_fd)
                self._events_fd = None
