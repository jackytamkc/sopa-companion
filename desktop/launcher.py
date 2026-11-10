"""Desktop commands with injectable OS boundaries for headless testing."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlencode

import httpx

from ..engine.config import ROOT
from ..server.runtime_file import (OTHER_ENGINE, ForeignRuntime, healthy_runtime, other_engine, pid_alive,
                                   private_directory, read_runtime, remove_runtime, runtime_path)
from . import browser, chrome, migrate
from .logs import log_path, log_tail, open_log, redact
from .processes import DesktopError, active_metadata, foreign_engine, owns_listener, stop_process, verify_process
from .window import chrome_profile

# What status and stop say when this copy's engine is not running and another copy's engine holds its state folder.
OTHER_ENGINE_STATUS = ("SoPA companion: not running (its state folder's runtime file belongs to another engine: pid "
                       "{pid}, data folder {data_dir})")

# How long `stop --force` waits for the engine to exit: it asks the engine to stop without the dialog (POST /shutdown):
# runs nobody answered for are discarded, but a run being saved (a leave or a save under way, or one the page answered
# Save for) is finished and saved first, and proseg needs time to write a whole sample's results; only then does it
# kill the engine (proseg dies with it). A plain `stop` (refused while a run is not saved to its sample) waits 30 s.
STOP_TIMEOUT_S = 900


def stopping_words(discarding: list, saving: list, timeout: float) -> str:
    """What `stop --force` says the engine does with its runs (POST /shutdown's {discarding, saving})."""
    def runs(n: int) -> str:
        return f"{n} run" if n == 1 else f"{n} runs"
    parts = []
    if discarding:
        parts.append(f"{runs(len(discarding))} not saved to "
                     f"{'its sample is' if len(discarding) == 1 else 'their samples are'} discarded")
    if saving:
        parts.append(f"{runs(len(saving))} {'is' if len(saving) == 1 else 'are'} being saved first "
                     f"(up to {timeout:g} s)")
    return "SoPA companion is stopping: " + "; ".join(parts) + "."


def api_request(record: dict, method: str, path: str) -> dict | list:
    """Credentials travel only in an HTTP header, never in a subprocess argument."""
    try:
        with httpx.Client(base_url=record["base_url"], timeout=2, trust_env=False,
                          headers={"Authorization": "Bearer " + record["token"], "Origin": record["base_url"]}) as client:
            response = client.request(method, path)
            if response.status_code != 200:
                raise DesktopError(f"Engine request failed (HTTP {response.status_code}).")
            return response.json()
    except (httpx.HTTPError, ValueError):
        raise DesktopError("The local engine is not responding.") from None


def stdout(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def stderr(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def stdout_terminal() -> bool:
    """Whether what launch prints reaches a person at a terminal: not when the desktop icon started it (Terminal=false:
    the desktop session's own log, which others may be able to read), nor a file or a pipe."""
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


@dataclass
class Services:
    spawn: Callable = subprocess.Popen
    run: Callable = subprocess.run
    which: Callable = shutil.which
    alive: Callable = pid_alive
    discover: Callable = foreign_engine
    stopper: Callable = stop_process
    request: Callable = api_request
    clock: Callable = time.monotonic
    sleep: Callable = time.sleep
    output: Callable = stdout
    error: Callable = stderr
    environ: dict = field(default_factory=lambda: dict(os.environ))
    opener: Callable = browser.open_app
    terminal: Callable = stdout_terminal

    def healthy(self, record: dict | None) -> bool:
        def probe(value):
            # Never send the token to a listener we have not identified: a stale record's pid or port may have been
            # reused by another process (or another user).
            try:
                verify_process(value["pid"], record=value)
            except Exception:
                return False
            if not owns_listener(value["pid"], value["port"]):
                return False
            try:
                self.request(value, "GET", "/api/v1/health")
                self.request(value, "GET", "/api/v1/runs")
                return True
            except DesktopError:
                return False
        return healthy_runtime(record, probe, alive=self.alive)

    def has_display(self) -> bool:
        return browser.desktop_session(self.environ)

    def request_health(self, record: dict) -> dict:
        answer = self.request(record, "GET", "/api/v1/health")
        return answer if isinstance(answer, dict) else {}

    def home(self) -> Path:
        """The home folder of the user this launcher runs for (a snap browser's folders are in it)."""
        return Path(self.environ.get("HOME") or Path.home())


def _record(config):
    try:
        return read_runtime(config)
    except (OSError, ValueError):
        # Discovery must still check the lock/owner before replacing broken metadata.
        return None


def positive_timeout(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise DesktopError("Timeout must be a positive, finite number of seconds.")
    return value


def _busy_launch(config):
    try:
        with (runtime_path(config).parent / "launch.lock").open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return False


@contextmanager
def _launch_lock(config, services, deadline):
    directory = private_directory(runtime_path(config).parent)
    fd = os.open(directory / "launch.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        os.fchmod(stream.fileno(), 0o600)
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if services.clock() >= deadline:
                    raise DesktopError("Timed out waiting for another launcher.")
                services.sleep(.2)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def _progress(services):
    child = None
    if services.has_display() and (zenity := services.which("zenity")):
        try:
            child = services.spawn([zenity, "--progress", "--pulsate", "--auto-close", "--no-cancel",
                                    "--title=SoPA companion", "--text=Starting the engine…"],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, close_fds=True)
        except OSError:
            pass
    try:
        yield
    finally:
        if child is not None:
            try:
                child.stdin.write(b"100\n")
                child.stdin.close()
                child.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                child.terminate()


def _handle_foreign(config, record, services, *, wait_for_launcher=False):
    foreign = services.discover(config, record)
    if not foreign:
        return True
    # The lock's holder may have appeared between our initial lock check and discovery.
    if wait_for_launcher and _busy_launch(config):
        return False
    pid = foreign.get("pid")
    if pid and foreign.get("mine") is False:
        # Another user's engine holds this data folder: never offered a restart.
        raise DesktopError(f"Another user's SoPA companion engine (pid {pid}) is using {config['paths']['data_dir']}. "
                           "It cannot be stopped from here: ask them to stop it, or give this app its own data folder "
                           "(paths.data_dir in a --config file).", 3)
    from ..cli import command_words
    instructions = (f"SoPA companion already has an engine owner (pid {pid or 'unknown'}) but no healthy runtime file. "
                    "In its terminal, wait for active runs to finish, stop serve with Ctrl+C, then run "
                    f"{command_words()} launch with the same --config file.")
    zenity = services.which("zenity") if services.has_display() else None
    if not pid or foreign.get("data_dir") != config["paths"]["data_dir"] or not zenity:
        raise DesktopError(instructions, 3)
    question = f"SoPA companion is already running from a terminal (pid {pid}). Restart it so the icon can manage it?"
    try:
        answer = services.run([zenity, "--question", "--no-markup", "--title=SoPA companion", "--text=" + question],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        raise DesktopError(instructions, 3) from None
    if answer.returncode != 0:
        raise DesktopError("Engine restart declined. " + instructions, 3)
    services.stopper(config, pid, record=record if record and record["pid"] == pid else None)
    return True


class Refused(DesktopError):
    """A launch refused by a rule of its own (no display for the window, another copy's engine holding the state folder):
    its message says why, and no engine log lines are shown with it (command), as none would be about it."""


class SharedStateFolder(Refused):
    """Another copy's engine holds this state folder (refuse_shared): exit 3. The engine log in that folder is that
    engine's, never shown as this copy's."""

    def __init__(self, message: str):
        super().__init__(message, 3)


def refuse_shared(config: dict) -> None:
    """Another copy's engine holds this state folder (its runtime file names another data folder, and it runs:
    server/runtime_file.py other_engine): refused with exit 3, nothing of that engine touched."""
    if other := other_engine(config):
        raise SharedStateFolder(OTHER_ENGINE.format(pid=other["pid"], data_dir=other["data_dir"],
                                                    state_dir=config["paths"]["state_dir"]))


def spawn_engine(config: dict, config_path: str | Path | None, services: Services) -> subprocess.Popen:
    command = [os.path.abspath(sys.executable), "-m", "sopa_companion"]
    if config_path:
        command.extend(["--config", str(Path(config_path).resolve())])
    command.extend(["serve", "--host", config["server"]["host"], "--port", str(config["server"]["port"])])
    env = dict(services.environ)
    env.update(SOPA_COMPANION_DESKTOP_LOG=str(log_path(config)), PYTHONUNBUFFERED="1")
    with open_log(log_path(config)) as log:
        return services.spawn(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)


def ensure_engine(config: dict, *, config_path: str | Path | None = None,
                  timeout: float | None = None, services: Services | None = None) -> dict:
    services = services or Services()
    timeout = positive_timeout(timeout if timeout is not None else config.get("desktop", {}).get("start_timeout_s", 120))
    deadline = services.clock() + timeout
    refuse_shared(config)
    record = _record(config)
    if services.healthy(record):
        return record
    with _progress(services):
        # An overlapping double-click waits for the first launcher without writing.
        while True:
            while _busy_launch(config):
                if services.clock() >= deadline:
                    raise DesktopError("Timed out waiting for another launcher.")
                services.sleep(.2)
            record = _record(config)
            if services.healthy(record):
                return record
            if _handle_foreign(config, record, services, wait_for_launcher=True):
                break
        # Time spent answering the restart question is not engine startup time.
        deadline = services.clock() + timeout
        with _launch_lock(config, services, deadline):
            record = _record(config)
            if services.healthy(record):
                return record
            _handle_foreign(config, record, services)
            if record:
                remove_runtime(config, record["pid"])
            refuse_shared(config)                  # another copy's engine may have started meanwhile
            deadline = services.clock() + timeout
            child = spawn_engine(config, config_path, services)
            while services.clock() < deadline:
                code = child.poll()
                if code is not None:
                    # Exit 3: the engine refused to start. When another copy's engine took the state folder as it
                    # started (it lost the race at its runtime file), that is what launch says, with launch's exit 3.
                    if code == 3:
                        refuse_shared(config)
                    raise DesktopError(f"Engine startup failed (exit {code}).", 3 if code == 3 else 1)
                record = _record(config)
                if record and record["pid"] == child.pid and services.healthy(record):
                    return record
                services.sleep(.2)
    raise DesktopError(f"Engine startup timed out after {timeout:g} seconds. Check status before retrying.")


# What launch prints with a link when no browser opened (its why first), or when none signed in after it opened one.
OPEN_BY_HAND = "Open this link in your browser. It signs one browser in, once, within 10 minutes."
NOT_YET = ("SoPA companion has not opened in a browser yet. If no tab shows it, open this link in your browser "
           "(once, within 10 minutes):")
# The same with no terminal to print a link to (the icon's launch): never a live code, the address and how to get one.
NOT_OPENED = "SoPA companion has not opened in a browser yet."
NO_TERMINAL = "SoPA companion: {url}\nFor a private URL: {command} launch --print-url"
OPEN_WAIT_S = 20.0          # [desktop] open_wait_s: how long launch waits for a page of the browser it opened to sign in


def browser_command(config: dict, url: str, *, which=shutil.which) -> list[str]:
    """The Chrome app window's command on `url`, whatever [desktop] browser says: the harness's Chrome is the app
    window's, from this one place. The app's own launch opens the browser through desktop/browser.py open_app instead,
    on a private opening page (never an address with a launch code, which any account could read on a command line)."""
    executable = chrome.find(which)
    if executable is None:
        raise DesktopError("No Chrome or Chromium was found for the app window. Use launch --print-url.")
    flags = config.get("desktop", {}).get("chrome_flags", [])
    if problem := chrome.flags_problem(flags):
        raise DesktopError(problem + ".")
    # The engine recognises the app's own Chrome by its profile flag (engine/stain_patches.py: the Cellpose GUI opens
    # on that window's desktop).
    return chrome.command(executable, url, chrome_profile(config), flags, window_class="SoPACompanion")


def open_wait(config: dict) -> float:
    """[desktop] open_wait_s: how long launch waits for a sign-in in the browser it opened (0: it does not wait)."""
    value = config.get("desktop", {}).get("open_wait_s", OPEN_WAIT_S)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise DesktopError("desktop.open_wait_s must be a number of seconds (0: launch does not wait).")
    return float(value)


def signed_in(record: dict, services: Services) -> int | None:
    """How many pages of browsers the app opened have signed in to the engine so far (GET /health presence.windows:
    the window launch codes traded; another tab connecting is none of them); None without an answer."""
    try:
        presence = services.request_health(record).get("presence")
    except DesktopError:
        return None
    windows = presence.get("windows") if isinstance(presence, dict) else None
    return windows if isinstance(windows, int) and not isinstance(windows, bool) else None


def launch_code(record: dict, services: Services, *, link: bool = False) -> str:
    """A one-use launch code, which only the engine's bearer token may ask for: for the window opened now (45 s), or
    with `link` for a link printed for a person to open by hand (10 minutes: time for an SSH tunnel)."""
    response = services.request(record, "POST", "/api/v1/session/launch-code" + ("?link=1" if link else ""))
    code = response.get("code") if isinstance(response, dict) else None
    if not isinstance(code, str) or not code:
        raise DesktopError("The engine did not return a launch code. Restart the engine.")
    return code


def launch(config: dict, args, *, services: Services | None = None) -> int:
    services = services or Services()
    # The window needs a display: without one (an SSH session), say so before any engine starts, rather than after.
    if not args.print_url and not args.no_browser and not services.has_display():
        raise Refused("No display is available. Use launch --no-browser or launch --print-url for an SSH tunnel.")
    wait = 0.0
    if not (args.print_url or args.no_browser):
        # Where the app opens, checked before any engine starts: a bad [desktop] browser is named in plain words now.
        try:
            mode, _ = browser.choice(config, args.browser)
        except ValueError as exc:
            raise DesktopError(str(exc)) from None
        if mode == "chrome" and (problem := chrome.flags_problem(config.get("desktop", {}).get("chrome_flags", []))):
            raise DesktopError(problem + ".")
        wait = open_wait(config)
    # The default instance's first start takes the previous version's state folder and window profile, before anything
    # makes its own state folder (ensure_engine's launch lock): said here and in the engine's log.
    if moved := migrate.migrate(config):
        services.output(moved)
        with open_log(log_path(config)) as log:
            log.write((moved + "\n").encode())
    record = ensure_engine(config, config_path=args.config, timeout=args.timeout, services=services)
    # A one-use link, never the bearer token: a browser keeps the URLs it opened in its history, and the token issues
    # launch codes and stops the engine for as long as it runs.
    link = lambda: record["base_url"] + "/?" + urlencode([("code", launch_code(record, services, link=True))])

    def by_hand(lead: str, bare: str | None = None) -> None:
        """A link to open by hand, after `lead`, printed to a terminal only. Anywhere else (the icon's launch prints into
        the desktop session's own log) a live code would sit where nobody sees it: `bare` instead, then the address
        and how to ask for a link."""
        if services.terminal():
            services.output(lead)
            services.output(link())
            return
        if bare:
            services.output(bare)
        from ..cli import command_words
        services.output(NO_TERMINAL.format(url=record["base_url"], command=command_words()))
    if args.print_url:
        services.output(link())
    elif args.no_browser:
        services.output(f"SoPA companion running (pid {record['pid']}) at {record['base_url']}")
    else:
        # The browser opens a private page in the engine's private run folder (0700, where its runtime file is), or in a
        # snap browser's own folder, that sends it on to `url`: a launch code on a command line could be read, and used
        # first, by any account on this computer. The pages earlier launches no longer need go first; this one goes once
        # its code has expired (at the next launch or stop), or now should the browser not start.
        folder = private_directory(runtime_path(config).parent)
        browser.remove_pages([folder, *browser.confined_folders(services.home())], older_than=browser.PAGE_KEEP_S)
        before = signed_in(record, services) or 0
        url = record["base_url"] + "/?" + urlencode([("code", launch_code(record, services))])
        try:
            opened = services.opener(url, folder=folder, config=config, profile=chrome_profile(config),
                                     choice=args.browser, environ=services.environ, which=services.which,
                                     spawn=services.spawn, run=services.run, home=services.home())
        except ValueError as exc:
            raise DesktopError(str(exc)) from None
        if opened.notice:
            services.output(opened.notice)
        if opened.how == "link":
            if opened.why:
                services.output(opened.why)
            by_hand(OPEN_BY_HAND)
            return 0
        # A browser took the page but may never show it (a URL forwarder as the desktop's https handler, a sandbox the
        # launcher could not tell, a browser that fails without a word): no sign-in within the wait, and a link is
        # printed too. The app's own Chrome window is not waited for.
        if opened.how != "chrome" and wait > 0:
            deadline = services.clock() + wait
            while services.clock() < deadline:
                services.sleep(.5)
                if (signed_in(record, services) or 0) > before:
                    return 0
            by_hand(NOT_YET, NOT_OPENED)
    return 0


def stop(config: dict, args, *, services: Services | None = None) -> int:
    services = services or Services()
    timeout = positive_timeout(args.timeout if args.timeout is not None else STOP_TIMEOUT_S if args.force else 30)
    try:
        record = read_runtime(config)
    except ForeignRuntime:
        record = None                           # another copy's engine's record: none of this copy's
    if record is None:
        if services.discover(config):
            raise DesktopError("No runtime file is available for the existing engine. Run launch on the desktop to restart it, or stop serve in its terminal.", 3)
        if other := other_engine(config):       # nothing of this copy runs, and the other one is never signalled
            services.output(OTHER_ENGINE_STATUS.format(pid=other["pid"], data_dir=other["data_dir"]))
            return 0
        services.output("SoPA companion is not running.")
        return 0
    if not services.alive(record["pid"]):
        remove_runtime(config, record["pid"])
        services.output("SoPA companion is not running; removed its stale runtime file.")
        return 0
    # (healthy() identifies the listener first: the token is never sent to another process.)
    healthy = services.healthy(record)
    if args.force and healthy:
        # Stop without the desktop dialog: the runs nobody answered for are discarded, a save under way is finished
        # first; the engine then exits by itself.
        try:
            answer = services.request(record, "POST", "/api/v1/shutdown")
            discarding, saving = answer.get("discarding") or [], answer.get("saving") or []
            if discarding or saving:
                services.output(stopping_words(discarding, saving, timeout))
        except (DesktopError, AttributeError):
            pass      # an engine that is already shutting down (its save dialog may be open): SIGTERM, as before
    pending = engine_pending(record, services) if healthy and not args.force else None
    services.stopper(config, record["pid"], record=record, force=args.force, timeout=timeout,
                     **({} if pending is None else {"active_reader": lambda _config: {"runs": pending}}))
    # The opening pages, here and in a snap browser's folder: their codes went with the engine.
    browser.remove_pages([runtime_path(config).parent, *browser.confined_folders(services.home())])
    services.output("SoPA companion stopped.")
    return 0


def engine_pending(record: dict, services: Services) -> list[dict] | None:
    """The runs a plain `stop` waits for, as the engine has them (GET /closing: Runs.pending, the runs its window can
    save or discard), as stop_process lists them: [{id, state}]. A folder the engine left alone at its start, as the
    last session's save still writes it, is none of them. None when the engine cannot say (one
    shutting down, or an older one): stop_process then reads the runs' run.json files."""
    try:
        answer = services.request(record, "GET", "/api/v1/closing")
    except DesktopError:
        return None
    runs = answer.get("runs") if isinstance(answer, dict) else None
    if not isinstance(runs, list):
        return None
    return [{"id": run.get("id"), "state": run.get("state") or "unknown"} for run in runs if isinstance(run, dict)]


def directory_size(path: str | Path) -> int:
    total = 0
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(directory) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    return total


def status(config: dict, args, *, services: Services | None = None) -> int:
    services = services or Services()
    record = _record(config)
    running = services.healthy(record)
    active = active_metadata(config)
    result = {"running": running, "healthy": running, "pid": record["pid"] if record else None,
              "port": record["port"] if record else None, "base_url": record["base_url"] if record else None,
              "uptime_s": max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(record["started_at"])).total_seconds()) if running else None,
              "version": record["version"] if record else None, "active_runs": active["runs"],
              "data_dir": config["paths"]["data_dir"], "log_path": str(log_path(config)),
              "sizes_bytes": {"var": directory_size(config["paths"]["data_dir"])},
              # Where launch would open the app, and why not when it would print a link (a bad setting included).
              "browser": browser.describe(config, services.environ, run=services.run, which=services.which,
                                          home=services.home())}
    if running:
        # The pages signed in to the running engine (GET /health presence: here, leaving, closed or away).
        try:
            result["window"] = services.request_health(record).get("presence")
        except DesktopError:
            result["window"] = None
    if not running:
        foreign = services.discover(config, record)
        if foreign:
            result.update(running=True, pid=foreign["pid"], unmanaged=True)
            if not record or record["pid"] != foreign["pid"]:
                result.update(port=None, base_url=None, version=None)
    # Another copy's engine whose runtime file is in this state folder: {pid, data_dir, port}, or None.
    result["other_engine"] = other = other_engine(config)
    if args.json:
        services.output(json.dumps(result, indent=2))
    else:
        state = "running" if running else ("running without a healthy runtime file" if result["running"] else "not running")
        services.output(OTHER_ENGINE_STATUS.format(pid=other["pid"], data_dir=other["data_dir"])
                        if other and not result["running"] else "SoPA companion: " + state)
        services.output(json.dumps(result, indent=2))
    return 0


def command(config: dict, args, *, services: Services | None = None) -> int:
    services = services or Services()
    try:
        return {"launch": launch, "stop": stop, "status": status}[args.command](config, args, services=services)
    except (DesktopError, OSError, ValueError) as exc:
        record = _record(config)
        secrets = [record["token"]] if record else []
        if config["server"].get("token") != "auto":
            secrets.append(config["server"]["token"])
        message = redact(str(exc), secrets)
        if args.command == "launch" and not isinstance(exc, Refused):
            message += "\n\nLast 30 engine log lines:\n" + log_tail(log_path(config), secrets=secrets)
        services.error(message)
        if services.has_display() and args.command != "status" and (zenity := services.which("zenity")):
            try:
                services.run([zenity, "--error", "--no-markup", "--title=SoPA companion", "--width=640", "--text=" + message],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
        return getattr(exc, "exit_code", 1)
