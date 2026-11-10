"""Private session folders, ownership records and conservative crash cleanup."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import tempfile
import time

from .. import dirs, portable
from .result import CompanionError

SCRATCH_ROOTS_KEPT = 32         # the scratch folders remember_scratch keeps for sweep_scratch, most recent first
_DATA = re.compile(r"sopa-companion-([0-9a-f]{8})")


def process_identity(pid: int) -> dict:
    """A process as a session's records name it: pid, computer, boot and start (clock ticks after boot, from
    /proc/<pid>/stat; on macOS and Windows, which have no /proc, psutil's start time in seconds since the epoch: a record
    is only ever compared on the computer that wrote it)."""
    if not portable.LINUX:
        return {"pid": pid, "host": socket.gethostname(), "boot": portable.boot_id(), "started": portable.started(pid)}
    started = None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        started = int(fields[19])
    except (OSError, ValueError, IndexError):
        pass
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = None
    return {"pid": pid, "host": socket.gethostname(), "boot": boot, "started": started}


def alive(identity: dict | None) -> bool | None:
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 0:
        return None
    if identity.get("host") != socket.gethostname():
        return None
    now = process_identity(identity["pid"])
    if identity.get("boot") and now["boot"] and identity["boot"] != now["boot"]:
        return False
    if not portable.LINUX:
        # No /proc: whether the pid runs (never os.kill on Windows) and, when it does, whether it is the same process.
        running = portable.alive(identity["pid"])
        if running is not True:
            return running
        if identity.get("started") is None or now["started"] is None:
            return None
        return abs(identity["started"] - now["started"]) < 0.01
    try:
        fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return False
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        return None
    if identity.get("started") is None or now["started"] is None:
        return None
    return identity["started"] == now["started"]


def private_directory(path: Path) -> Path:
    """`path`, a folder of this user's alone (made when missing): 0700 on POSIX; on Windows a real folder (no link or
    junction) in the user's own profile, whose access list keeps others out (private.py)."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
        raise CompanionError("The app needs a private folder owned by you")
    if portable.WINDOWS:
        return path
    path.chmod(0o700)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise CompanionError("The app needs a private folder on a Linux filesystem")
    return path


def state_root() -> Path:
    candidates = []
    if runtime := os.environ.get("XDG_RUNTIME_DIR"):
        candidates.append(Path(runtime) / "sopa-companion")
    candidates.append(dirs.user_cache() / "sessions")
    for path in candidates:
        try:
            # Resolved: the kernel, owner.json, the sweep and the engine (engine/config.py resolves state_dir) all name
            # the session's folders, and its Chrome profile, by one string even when the way there has a link in it.
            return private_directory(path.expanduser().resolve())
        except (OSError, CompanionError):
            continue
    raise CompanionError("The app could not make a private session folder")


def private_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        portable.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_config(path: Path, config: dict) -> None:
    def toml(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (str, int, list)):
            return json.dumps(value, ensure_ascii=False)
        raise TypeError("Unsupported session setting")

    def tables(name: str, table: dict) -> list[str]:
        """A table's own keys, then each table inside it as [name.key] ([locate.analysis])."""
        own = [f"{key} = {toml(value)}" for key, value in table.items() if not isinstance(value, dict)]
        inner = [part for key, value in table.items() if isinstance(value, dict)
                 for part in tables(f"{name}.{key}", value)]
        return ([f"[{name}]\n" + "\n".join(own) + "\n"] if own or not inner else []) + inner

    text = "\n".join(part for section, table in config.items() for part in tables(section, table))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | portable.O_NOFOLLOW | portable.O_BINARY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    if not portable.WINDOWS and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise CompanionError("The app could not protect its session settings")


def _remove(path: Path) -> None:
    """`path` removed, whatever it holds, tried again for a moment: on Windows a folder goes only once every process
    that had a file open in it (the engine and its helpers, a virus scanner) has let go, so it is tried for longer
    there (about 6 s)."""
    attempts = 12 if portable.WINDOWS else 4
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _marker(id: str, state: Path) -> dict:
    """What a session's data folder says of it (.sopa-companion-session.json): its id, its state folder, and the
    computer it runs on: a scratch folder may be shared between computers, each with state folders of its own."""
    return {"id": id, "state_dir": str(state), "host": socket.gethostname()}


@dataclass(frozen=True)
class SessionFolders:
    id: str
    state: Path
    data: Path

    @classmethod
    def create(cls, scratch: Path, zarr: str, *, root: Path | None = None):
        root = state_root() if root is None else private_directory(root)
        scratch.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            id = secrets.token_hex(4)
            state, data = root / id, scratch / ("sopa-companion-" + id)
            try:
                state.mkdir(mode=0o700)
            except FileExistsError:
                continue
            try:
                data.mkdir(mode=0o700)
            except FileExistsError:
                state.rmdir()
                continue
            except BaseException:
                state.rmdir()
                raise
            folders = cls(id, state, data)
            try:
                private_directory(state)
                (state / "logs").mkdir(mode=0o700)
                private_json(data / ".sopa-companion-session.json", _marker(id, state))
                private_json(state / "owner.json", {"schema": 1, "id": id, "kernel": process_identity(os.getpid()),
                             "engine": None, "data_dir": str(data), "state_dir": str(state), "zarr": zarr})
                return folders
            except BaseException:
                _remove(data)
                _remove(state)
                raise
        raise CompanionError("The app could not make a fresh session folder")

    def pylib(self) -> Path:
        """<state>/pylib, a folder holding one link, sopa_companion -> this package, for an interpreter that does not
        import this copy by itself (the engine's, api/handle.py): made when first needed, kept as it is after."""
        folder = self.state / "pylib"
        folder.mkdir(mode=0o700, exist_ok=True)
        link = folder / "sopa_companion"
        if not portable.folder_link(link):
            portable.link_folder(link, Path(__file__).resolve().parents[1])
        return folder

    def engine_started(self, pid: int) -> None:
        path = self.state / "owner.json"
        value = json.loads(path.read_text())
        value["engine"] = process_identity(pid)
        private_json(path, value)

    def remove(self) -> None:
        _remove(self.data)
        _remove(self.state)


@contextmanager
def _unlocked(path: Path):
    """Hold an existing lock while removing its session; never replace or create it."""
    fd = None
    try:
        try:
            fd = os.open(path, os.O_RDONLY | portable.O_NOFOLLOW)
        except FileNotFoundError:
            yield True
            return
        try:
            portable.lock(fd, block=False)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        if fd is not None:
            if portable.WINDOWS:
                try:
                    portable.unlock(fd)            # Windows lets go of a lock at close only "when it can"
                except OSError:
                    pass
            os.close(fd)


def sweep(root: Path | None = None) -> tuple[str, ...]:
    """Remove only verified sessions of dead kernels, with no live engine or held lock."""
    from .window import close_profile

    root = state_root() if root is None else private_directory(root)
    removed = []
    for state in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{8}", state.name):
            continue
        try:
            info = state.lstat()
            if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
                continue
            owner_path = state / "owner.json"
            if owner_path.is_symlink():
                continue
            owner = json.loads(owner_path.read_text())
            if not isinstance(owner, dict) or owner.get("schema") != 1 or owner.get("id") != state.name or owner.get("state_dir") != str(state):
                continue
            if alive(owner.get("kernel")) is not False or alive(owner.get("engine")) is True:
                continue
            data = Path(owner["data_dir"])
            if not data.is_absolute() or data.name != "sopa-companion-" + state.name or data.is_symlink():
                continue
            if data.exists():
                if not portable.mine(data.stat()):
                    continue
                marker = data / ".sopa-companion-session.json"
                value = None if marker.is_symlink() else json.loads(marker.read_text())
                # Its owner record says whose computer it is (alive, above); a marker from before markers named their
                # computer counts too.
                if not isinstance(value, dict) or {k: v for k, v in value.items() if k != "host"} != \
                        {"id": state.name, "state_dir": str(state)}:
                    continue
            with _unlocked(data / ".engine.lock") as free:
                if free:
                    close_profile(state / "chrome-profile")
                    SessionFolders(state.name, state, data).remove()
                    removed.append(state.name)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return tuple(removed)


def _scratch_roots_file() -> Path:
    return dirs.user_cache() / "scratch-roots.json"


def _scratch_roots() -> list[str]:
    try:
        roots = json.loads(_scratch_roots_file().read_text()).get("roots")
    except (OSError, ValueError, AttributeError):
        return []
    return [root for root in roots if isinstance(root, str) and Path(root).is_absolute()] if isinstance(roots, list) else []


def remember_scratch(scratch: Path) -> None:
    """Keep `scratch` among the folders sweep_scratch looks in (a list in the user's cache, which outlives the state
    root): a session's data folder there outlives a state root that a reboot or a logout cleared while it was open."""
    roots = [str(scratch)] + [root for root in _scratch_roots() if root != str(scratch)]
    try:
        private_directory(_scratch_roots_file().parent)
        private_json(_scratch_roots_file(), {"roots": roots[:SCRATCH_ROOTS_KEPT]})
    except (OSError, CompanionError):
        pass                    # only the later sweep is lost: the session itself goes on


def _own_state_roots() -> set[Path]:
    """Every state root a session of this user's can have had (state_root's places, and the runtime folder's usual
    one): a marker naming a state folder elsewhere is another account's, whatever a shared disk says owns its files."""
    places = [dirs.user_cache() / "sessions"]
    if not portable.WINDOWS:
        places.insert(0, Path(f"/run/user/{os.getuid()}") / "sopa-companion")
    if runtime := os.environ.get("XDG_RUNTIME_DIR"):
        places.append(Path(runtime) / "sopa-companion")
    return {place.expanduser().resolve() for place in places}


def sweep_scratch(scratch: Path | None = None) -> tuple[str, ...]:
    """Remove the data folders that sessions left in `scratch` and in the remembered scratch folders (remember_scratch)
    when their state folder went (a reboot or a logout cleared the state root, kernel and engine with it), which sweep
    can no longer find: only this user's own folders (never a link), whose marker names this computer and a state
    folder of the same id, in one of this user's state roots, that is gone, and whose engine lock nobody holds. A
    scratch folder shared with another computer holds that computer's live sessions too: their state folders are in its
    own runtime folder, and a file lock may not reach across computers."""
    removed, own = [], _own_state_roots()
    roots = dict.fromkeys(([str(scratch)] if scratch is not None else []) + _scratch_roots())
    for root in roots:
        try:
            folders = [path for path in Path(root).iterdir() if _DATA.fullmatch(path.name)]
        except OSError:
            continue
        for data in folders:
            try:
                info = data.lstat()
                if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
                    continue
                marker = data / ".sopa-companion-session.json"
                if not stat.S_ISREG(marker.lstat().st_mode):
                    continue
                value = json.loads(marker.read_text())
                id = _DATA.fullmatch(data.name)[1]
                state = Path(value["state_dir"]) if isinstance(value, dict) and set(value) == {"id", "state_dir", "host"} \
                    and value["id"] == id and isinstance(value["state_dir"], str) \
                    and value["host"] == socket.gethostname() else None
                if state is None or not state.is_absolute() or state.name != id or state.parent not in own:
                    continue
                try:
                    state.lstat()
                    continue                         # still there: sweep's to judge, by its owner record
                except FileNotFoundError:
                    pass
                with _unlocked(data / ".engine.lock") as free:
                    if free:
                        _remove(data)
                        removed.append(id)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return tuple(removed)
