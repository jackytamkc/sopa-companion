"""Private, atomic discovery of the engine that owns a data directory.

One state folder holds one engine's runtime file. Two copies of the app with data folders of their own can share a
state folder (an installed copy's default and a source tree whose settings name the same one): whichever engine holds
it first keeps it, and the other's launch and start are refused (other_engine, refuse_other), so its record is never
replaced while it runs.
"""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

import psutil

from .. import PROTOCOL, __version__, portable
from ..names import MODULES, command_script

# The refusal of a second engine on a state folder whose runtime file names another data folder and an engine that
# still runs (desktop/launcher.py says it on launch, exit 3; write_runtime refuses with it).
OTHER_ENGINE = ("Another SoPA companion engine (pid {pid}) uses this state folder, {state_dir}, for another data folder, "
                "{data_dir}. Stop that engine first (with the stop command of the copy it belongs to), or give one of "
                "the two a state folder of its own: [paths] state_dir.")


class ForeignRuntime(ValueError):
    """The runtime file names another data folder: another copy's engine wrote it (other_engine says whether it runs)."""


class OtherEngine(RuntimeError):
    """Another copy's engine holds this state folder (OTHER_ENGINE): an engine's start stops with exit 3, as launch's."""
    exit_code = 3


# The interpreter's own options that take the next argument as their value (`python -X dev -m ...`).
VALUE_OPTIONS = ("-X", "-W", "--check-hash-based-pycs")


def module_args(argv: list[str]) -> list[str] | None:
    """What follows `-m <module>` when the interpreter's own options end there; None when they end at code (`-c`), the
    standard input (`-`) or a script: a `-m` after those is that program's argument, not the interpreter's."""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "-m":
            return argv[index + 1:]
        if arg == "-" or arg.startswith("-c") or not arg.startswith("-"):
            return None
        index += 2 if arg in VALUE_OPTIONS else 1
    return None


def serve_command(argv: list[str]) -> bool:
    """Recognise the app's engine command, not a substring inside another command: `-m sopa_companion [--config FILE]
    serve`, the previous version's `-m proseg_live ...` (names.MODULES: stop and status know an engine it started), or
    the console script's `<python> <...>/sopa-companion [--config FILE] serve` (names.command_script). The `-m` is the
    interpreter's own (`python -c CODE -m sopa_companion serve` runs CODE). The script's form needs a Python interpreter
    first, so a wrapper (`timeout 600 sopa-companion serve`, `bash -c ...`) is never the engine: its python child is."""
    if (module := module_args(argv)) is not None:
        if not module or module[0] not in MODULES:
            return False
        args = module[1:]
    elif len(argv) >= 2 and re.split(r"[\\/]", argv[0])[-1].lower().startswith("python") and command_script(argv[1]):
        args = argv[2:]
    else:
        return False
    while args and args[0].startswith("--config"):
        if args[0] == "--config":
            args = args[2:]
        elif args[0].startswith("--config="):
            args = args[1:]
        else:
            return False
    return bool(args and args[0] == "serve")


def acquire_engine_lock(config: dict) -> int:
    """Exclusive ownership of the data directory for the engine's whole lifetime (released when the process exits).
    A second engine on the same runs would mark them interrupted and let them be deleted while proseg writes them."""
    path = Path(config["paths"]["data_dir"]) / ".engine.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | portable.O_BINARY, 0o600)
    try:
        portable.lock(fd, block=False)
    except BlockingIOError:
        os.close(fd)
        from ..cli import command_words
        command = command_words()
        raise RuntimeError(f"Another SoPA companion engine already owns {path.parent}. Use `{command} status`, "
                           f"or open it with `{command} launch`.") from None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def runtime_path(config: dict) -> Path:
    return Path(config["paths"]["state_dir"]) / "run/server.json"


def private_directory(path: str | Path) -> Path:
    """Create a user-only directory, refusing symlinks and foreign ownership."""
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
        raise ValueError("Private directory must be a directory owned by the current user")
    path.chmod(0o700)
    return path


def atomic_private_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    private_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            portable.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        portable.replace(temporary, path)
        portable.fsync_folder(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_runtime(value: dict, config: dict, *, any_data_dir: bool = False) -> dict:
    """Validate structure and origin before ever sending credentials to it. `any_data_dir`: a record of another data
    folder too (ForeignRuntime otherwise), to tell whose engine holds the state folder."""
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("Unsupported engine runtime file")
    for key in ("pid", "port"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("Invalid engine runtime identity")
    if value["port"] > 65535 or value.get("host") not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Engine runtime must use a loopback address")
    authority = "[::1]" if value["host"] == "::1" else value["host"]
    if value.get("base_url") != f"http://{authority}:{value['port']}":
        raise ValueError("Invalid engine runtime origin")
    if not isinstance(value.get("data_dir"), str) or not value["data_dir"]:
        raise ValueError("Incomplete engine runtime file")
    if not any_data_dir and value["data_dir"] != str(Path(config["paths"]["data_dir"]).resolve()):
        raise ForeignRuntime("Engine runtime belongs to another data directory")
    for key in ("token", "started_at", "version", "protocol", "python"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("Incomplete engine runtime file")
    if any(c in value["token"] for c in "\r\n") or not Path(value["python"]).is_absolute():
        raise ValueError("Invalid engine runtime credentials or interpreter")
    try:
        stamp = datetime.fromisoformat(value["started_at"])
        if stamp.tzinfo is None:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid engine start time") from None
    if not isinstance(value.get("argv"), list) or not all(isinstance(a, str) for a in value["argv"]):
        raise ValueError("Invalid engine runtime command")
    return value


@contextmanager
def _runtime_lock(config):
    directory = private_directory(runtime_path(config).parent)
    fd = os.open(directory / ".runtime.lock", os.O_CREAT | os.O_RDWR | portable.O_NOFOLLOW | portable.O_BINARY, 0o600)
    with os.fdopen(fd, "r+") as stream:
        portable.fchmod(stream.fileno(), 0o600)
        portable.lock(stream)
        try:
            yield
        finally:
            portable.unlock(stream)


def write_runtime(config: dict, token: str, host: str, port: int, *, pid: int | None = None,
                  argv: list[str] | None = None) -> dict:
    authority = "[::1]" if host == "::1" else host
    value = {"schema": 1, "pid": os.getpid() if pid is None else pid,
             "host": host, "port": port, "base_url": f"http://{authority}:{port}",
             "token": token, "started_at": datetime.now(timezone.utc).isoformat(),
             "version": __version__, "protocol": PROTOCOL,
             "data_dir": str(Path(config["paths"]["data_dir"]).resolve()),
             "python": os.path.abspath(sys.executable),
             "argv": list(argv if argv is not None else psutil.Process().cmdline())}
    validate_runtime(value, config)
    with _runtime_lock(config):
        refuse_other(config)             # another copy's engine keeps its record: a hand-started serve, a launcher race
        atomic_private_json(runtime_path(config), value)
    return value


def read_runtime(config: dict, *, any_data_dir: bool = False) -> dict | None:
    """This state folder's runtime file, validated (validate_runtime); None when there is none. ForeignRuntime when it
    names another data folder, unless `any_data_dir`."""
    path = runtime_path(config)
    try:
        directory = path.parent.lstat()
        if not stat.S_ISDIR(directory.st_mode) or not portable.private_info(directory, 0o700):
            raise ValueError("Engine runtime directory must be owned by this user with permissions 0700")
        if portable.WINDOWS and portable.linked(path.lstat()):
            raise ValueError("Engine runtime file must be owned by this user with permissions 0600")
        fd = os.open(path, os.O_RDONLY | portable.O_NOFOLLOW | portable.O_BINARY)
        with os.fdopen(fd, encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not portable.private_info(info, 0o600):
                raise ValueError("Engine runtime file must be owned by this user with permissions 0600")
            value = json.load(stream)
        return validate_runtime(value, config, any_data_dir=any_data_dir)
    except FileNotFoundError:
        return None


def engine_running(record: dict, *, process_factory=psutil.Process) -> bool:
    """Whether the engine a runtime record names still runs: its pid is alive and no zombie, this user's (real and
    effective), the app's `serve` (serve_command), and started no later than the record says (else the pid was reused).
    Any psutil.Error or OSError: False."""
    try:
        process = process_factory(record["pid"])
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if not portable.owned_by_me(process) or not serve_command(process.cmdline()):
            return False
        return process.create_time() <= datetime.fromisoformat(record["started_at"]).timestamp()
    except (psutil.Error, OSError, KeyError, TypeError, ValueError):
        return False


def other_engine(config: dict, *, process_factory=psutil.Process) -> dict | None:
    """Another copy's engine that holds this state folder: the runtime file here names another data folder and an
    engine that still runs (engine_running) -> {pid, data_dir, port}. None for no file, this data folder's own record, a
    broken file, or a dead or reused pid."""
    try:
        record = read_runtime(config, any_data_dir=True)
    except (OSError, ValueError):
        return None
    if record is None or record["data_dir"] == str(Path(config["paths"]["data_dir"]).resolve()):
        return None
    if not engine_running(record, process_factory=process_factory):
        return None
    return {"pid": record["pid"], "data_dir": record["data_dir"], "port": record["port"]}


def refuse_other(config: dict) -> None:
    """OtherEngine, a RuntimeError (OTHER_ENGINE), when another copy's engine holds this state folder (other_engine)."""
    if other := other_engine(config):
        raise OtherEngine(OTHER_ENGINE.format(pid=other["pid"], state_dir=config["paths"]["state_dir"],
                                              data_dir=other["data_dir"]))


def pid_alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def healthy_runtime(value: dict | None, probe, *, alive=pid_alive) -> bool:
    """A pid alone is insufficient: probe health and a protected API endpoint."""
    return bool(value and alive(value["pid"]) and probe(value))


def remove_runtime(config: dict, pid: int) -> bool:
    """Only the recorded owner (or its verified stopper) can remove this file."""
    if not runtime_path(config).exists():
        return False
    with _runtime_lock(config):
        value = read_runtime(config, any_data_dir=True)     # another data folder's record never raises here
        if value is None or value["pid"] != pid:
            return False
        portable.unlink(runtime_path(config), missing_ok=True)
        return True


def install_lifecycle(app, config, host, port):
    """Publish after bind/startup; compose with the engine's lifespan cleanup."""
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        record = None
        try:
            async with original(application):
                try:
                    record = write_runtime(config, app.state.token, host, port)
                except OtherEngine as exc:       # its start fails; the command says why once the server has stopped
                    app.state.refused = exc
                    raise
                yield
        finally:
            if record is not None:
                remove_runtime(config, record["pid"])

    app.router.lifespan_context = lifespan
