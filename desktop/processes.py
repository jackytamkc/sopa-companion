"""Read-only engine discovery and identity-checked shutdown."""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import tomllib

import psutil

from ..names import MODULES
# serve_command lives with the runtime file (the engine's own refusal of a second engine needs it); its name stays here.
from ..server.runtime_file import module_args, pid_alive, remove_runtime, serve_command  # noqa: F401
from .window import AppWindow, chrome_profile  # noqa: F401 (the app window lives in window.py)


class DesktopError(Exception):
    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


def verify_process(pid: int, *, record: dict | None = None, process_factory=psutil.Process) -> psutil.Process:
    try:
        process = process_factory(pid)
        owners = process.uids()
        if owners.real != os.getuid() or owners.effective != os.getuid():
            raise DesktopError("Refusing to stop an engine owned by another user.")
        if not serve_command(process.cmdline()):
            raise DesktopError("Refusing to signal a pid whose command is not the app's serve.")
        if record and process.create_time() > datetime.fromisoformat(record["started_at"]).timestamp():
            raise DesktopError("Refusing to signal a reused pid; the runtime file is stale.")
        return process
    except psutil.AccessDenied:
        raise DesktopError("Cannot verify ownership of the engine process.") from None


def _app_folder(folder: Path) -> bool:
    """A copy of the app: its code, and its settings (config.toml, or a fresh clone's config.example.toml)."""
    return ((folder / "config.toml").is_file() or (folder / "config.example.toml").is_file()) and \
        (folder / "cli.py").is_file() and (folder / "engine").is_dir()


def _install_root(process) -> Path | None:
    """The app folder a `python -m` serve process runs from: its working folder when that is one (the launcher starts
    the engine there), else the folder of the package it runs (`-m sopa_companion`, or the previous version's `-m
    proseg_live`) that its Python finds first from its working folder or its PYTHONPATH. None when neither says (this
    install's own folder was assumed, so another install's engine, or another user's, counted as this one's), and for
    the console script's engine (`sopa-companion serve`): its working folder is wherever it was typed, and its package
    is the script's environment's, which its command line does not tell. Such an engine is still found by the engine
    lock of the data folder it holds (foreign_engine)."""
    try:
        cwd = Path(process.cwd())
        module = module_args(process.cmdline())
    except (psutil.Error, OSError):
        return None
    if module is None:
        return None
    try:
        search = (process.environ() or {}).get("PYTHONPATH", "")
    except (psutil.Error, OSError):
        search = ""
    package = module[0] if module and module[0] in MODULES else MODULES[0]
    found_from = [cwd / entry / package for entry in search.split(os.pathsep) if entry]   # an absolute one: as is
    for folder in (cwd, cwd / package, *found_from):
        try:
            if _app_folder(folder):
                return folder.resolve()
        except OSError:
            continue
    return None


def _data_dir(root: Path | None, config_path: Path | None = None) -> Path | None:
    """The data folder an engine of the app folder `root` uses, from the settings files load_config merges for that
    folder (engine/config.py sources: config.toml, else config.example.toml, then config.local.toml, an installed
    copy's user settings file, then the --config file), as load_config reads it (data_dir_of: "auto" by that copy's own
    rule, "~" expanded, a relative one inside `root`). Without `root`, only an absolute one in the --config file tells;
    else None."""
    from ..engine.config import data_dir_of, sources
    files = sources(config_path, root) if root is not None else [config_path] if config_path is not None else []
    paths: dict = {}
    for file in files:
        try:
            paths = {**paths, **tomllib.loads(Path(file).read_text(encoding="utf-8")).get("paths", {})}
        except FileNotFoundError:
            continue
    if root is not None:
        return data_dir_of(paths.get("data_dir"), root)
    data_dir = Path(str(paths.get("data_dir", "var"))).expanduser()
    return data_dir.resolve() if data_dir.is_absolute() else None


def _process_data_dir(process: psutil.Process) -> Path | None:
    """The data folder of a `serve` process (its app folder's, with its --config file): None when it cannot be told,
    and such a process is never taken for this install's engine (the engine lock still finds this one's)."""
    args = process.cmdline()
    config_path = None
    for index, arg in enumerate(args):
        if arg == "--config" and index + 1 < len(args):
            config_path = Path(args[index + 1])
        elif arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1])
        else:
            continue
        if not config_path.is_absolute():
            config_path = Path(process.cwd()) / config_path
        break
    return _data_dir(_install_root(process), config_path)


def _lock_owner(path: Path) -> int | None:
    """None means unlocked, 0 means held but its owner cannot be identified."""
    try:
        with path.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                info = os.fstat(stream.fileno())
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
                return None
    except FileNotFoundError:
        return None
    try:
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            if len(fields) < 6 or fields[1] == "->":
                continue
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino):
                return max(0, int(fields[4]))
    except (OSError, ValueError):
        pass
    return 0


def _mine(pid: int | None, uid: int) -> bool:
    """Whether a process is this user's (an unknown one is not)."""
    try:
        return pid is not None and psutil.Process(pid).uids().real == uid
    except (psutil.Error, OSError):
        return False


def foreign_engine(config: dict, record: dict | None = None, *, processes=psutil.process_iter,
                   uid: int | None = None) -> dict | None:
    """Detect a legacy/startup owner without creating a lock or any data files: {pid, data_dir, mine}: a `serve` of
    this user whose data folder is this one, or the engine lock's holder (the authority: every engine holds its data
    folder's lock). Another user's process is never this install's engine, nor one of another app folder, nor whatever
    listens on the port (the icon offered to stop them, or refused to start, although serve
    takes the next free port); `mine` is False for a lock held by another user's process, never offered a restart."""
    uid = os.getuid() if uid is None else uid
    data_dir = Path(config["paths"]["data_dir"])
    owner = _lock_owner(data_dir / ".engine.lock")
    for process in processes(["pid", "cmdline", "uids"]):
        try:
            owners = process.info.get("uids")
            if owners is None or owners.real != uid or not serve_command(process.info["cmdline"] or []):
                continue
            if _process_data_dir(process) == data_dir:
                return {"pid": process.pid, "data_dir": str(data_dir), "mine": True}
        except (psutil.Error, OSError, ValueError):
            continue
    if owner is not None:
        return {"pid": owner or None, "data_dir": str(data_dir), "mine": _mine(owner or None, uid)}
    if record and pid_alive(record["pid"]):
        try:
            verify_process(record["pid"], record=record)
            return {"pid": record["pid"], "data_dir": str(data_dir), "mine": True}
        except (DesktopError, psutil.NoSuchProcess):
            pass
    return None


def owns_listener(pid: int, port: int) -> bool:
    """Whether this pid is the process listening on the loopback port (checked before any credential is sent)."""
    try:
        connections = psutil.Process(pid).net_connections(kind="inet")
    except (psutil.Error, OSError):
        return False
    return any(c.status == psutil.CONN_LISTEN and c.laddr.port == port for c in connections)


def active_metadata(config: dict) -> dict:
    """The runs a plain `stop` waits for an answer about, from their run.json, when the engine cannot say itself
    (desktop/launcher.py engine_pending: no runtime credentials, or no answer): this version's runs (the one-run marker)
    not saved to their samples. Runs from earlier versions never block it (the app lists them at
    its next start), nor a run an engine kept on disk as it closed (`closed`: its save failed, or its save helper still
    used it; the next start removes it): neither is one the app can save or discard."""
    from ..engine.runs import leave_kind, stored_meta
    result = {"runs": []}
    for kind, filename in (("runs", "run.json"),):
        for path in sorted((Path(config["paths"]["data_dir"]) / kind).glob("*/" + filename)):
            try:
                value = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                raise DesktopError("Cannot inspect active work; retry, or use stop --force.", 2) from None
            if not isinstance(value, dict) or value.get("session_model") != 2 or value.get("closed"):
                continue
            if leave_kind(stored_meta(path.parent, value))["kind"] in ("ask", "saving"):
                result[kind].append({"id": value.get("id", path.parent.name), "state": value.get("state", "unknown")})
    return result


def stop_process(config: dict, pid: int, *, force: bool = False, timeout: float = 30,
                 record: dict | None = None, process_factory=psutil.Process, active_reader=active_metadata) -> None:
    try:
        process = verify_process(pid, record=record, process_factory=process_factory)
    except psutil.NoSuchProcess:
        if record:
            remove_runtime(config, pid)
        return
    if not force:
        active = active_reader(config)
        pending = [f"run {item['id']} ({item['state']})" for item in active["runs"]]
        if pending:
            raise DesktopError("A run is not saved to its sample yet: save or discard it in the app, or use stop --force to discard it.\n" + "\n".join(pending), 2)
    # psutil's signal methods recheck process identity to avoid pid-reuse races.
    try:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            if not force:
                raise DesktopError("Engine is still shutting down; use stop --force to allow SIGKILL.") from None
            verify_process(pid, record=record, process_factory=process_factory)
            process.kill()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                raise DesktopError("Engine did not exit after SIGKILL.") from None
    except psutil.NoSuchProcess:
        pass
    if record:
        remove_runtime(config, pid)
