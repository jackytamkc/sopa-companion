"""The previous version's state folder, moved once into this version's.

The app's package was called proseg_live before it was sopa_companion (names.py). Its default desktop instance kept its
state (runtime file, logs, crash reports) in ~/.local/state/proseg_live, and its window's Chrome profile (the page's
remembered settings, the window's size) apart, in ~/.local/share/proseg_live/chrome-profile. The first start of this
version's default instance (`serve`, `launch`) makes both its own: the profile goes into the old state folder, which
then becomes ~/.local/state/sopa-companion in one rename, so another start sees no new folder or a whole one. Renamed,
never copied: a move that cannot be done in place (another file system, no permission) is refused, nothing moved, and
says how to do it by hand. Refused too while anything of the previous version may still use either folder: its engine
or its window running, a launch of it under way, or the previous version installed in another folder, which its icon
would start again on new, empty folders (its desktop entries say where: _previous_start). A notebook's session, a
test's or the harness's engine (state folders of their own), `status` and `stop` never move anything.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import time
import tomllib

import psutil

from .. import dirs
from ..engine.config import ROOT, sources
from ..names import (COMMAND, LEGACY_ENTRY, LEGACY_INSTALL_RECORD, LEGACY_PROFILE, LEGACY_STATE_DIR, LOG_PREFIX,
                     MODULES, STATE_DIR, home_path)
from ..server.runtime_file import pid_alive
from .processes import DesktopError, _install_root, serve_command
from .window import AppWindow

PROFILE = "chrome-profile"          # the profile's name inside a state folder (desktop/window.py chrome_profile)
WAIT_S = 5.0                        # another start moving the folder at this moment: how long its move may take


class MigrationRefused(DesktopError):
    """The previous version's folders cannot be moved now: the start stops (exit code 3) and says why."""

    def __init__(self, message: str):
        super().__init__(message, 3)


def default_instance(config: dict, home: str | Path | None = None) -> bool:
    """Whether these settings are the default desktop instance's: its state folder is this version's default
    (names.STATE_DIR in `home`), whether a settings file names it or none does."""
    state = (config.get("paths") or {}).get("state_dir")
    if not state:
        return False
    return Path(state).expanduser().resolve() == home_path(STATE_DIR, home).resolve()


def _legacy_paths(home: str | Path | None) -> tuple[Path, Path, Path]:
    return home_path(LEGACY_STATE_DIR, home), home_path(LEGACY_PROFILE, home), home_path(LEGACY_ENTRY, home)


# A desktop entry's string escapes (the Desktop Entry Specification); any other backslash is kept as it is.
_ESCAPES = {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}


def _entry_keys(text: str) -> dict[str, str]:
    """The keys of a desktop entry's own group, [Desktop Entry] (its actions' groups left out), their values unescaped."""
    keys, group = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]
        elif group == "Desktop Entry" and "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            keys.setdefault(key.strip(), re.sub(r"\\(.)", lambda m: _ESCAPES.get(m[1], m[0]), value.strip()))
    return keys


def _previous_start(entry: Path) -> tuple[Path | None, Path | None] | None:
    """How a desktop entry starts the previous version: (the app folder it starts, or None when the entry does not tell,
    and the interpreter it runs, None when it runs no module); None when the entry is not there or runs another module.
    The folder is its Path= key (the installers that wrote one), else the first folder of the `env PYTHONPATH=...`
    before the interpreter (joined with the module's name), else the folder whose own .venv the interpreter is
    (`<folder>/.venv/bin/python`, as the README installs the app; the first installers' entries say no more than
    that). An entry that runs no module says where only by its Path= key."""
    try:
        keys = _entry_keys(entry.read_text(errors="replace"))
        argv = [arg.replace("%%", "%") for arg in shlex.split(keys.get("Exec", ""))]
    except (OSError, ValueError):
        return None
    pythonpath = ""
    if argv and Path(argv[0]).name == "env":
        argv.pop(0)
        while argv and "=" in argv[0] and not argv[0].startswith("-"):
            name, _, value = argv.pop(0).partition("=")
            pythonpath = value if name == "PYTHONPATH" else pythonpath
    module = argv[argv.index("-m") + 1] if "-m" in argv[1:-1] else None
    if module is None:
        return (Path(keys["Path"]), None) if keys.get("Path") else None
    if module not in MODULES[1:]:
        return None
    python = Path(argv[0])
    first = next((folder for folder in pythonpath.split(os.pathsep) if folder), None)
    if keys.get("Path"):
        return Path(keys["Path"]), python
    if first:
        return Path(first) / module, python
    if python.parent.name == "bin" and python.parent.parent.name == ".venv":
        return python.parent.parent.parent, python
    return None, python


def _previous_entries(entry: Path, record: Path | None) -> list[Path]:
    """The previous version's desktop entries: its applications menu's, and every other one its installer recorded (the
    launcher on the desktop)."""
    entries = [entry]
    try:
        recorded = json.loads(record.read_text())["paths"] if record is not None else []
        entries += [Path(path) for path in recorded if isinstance(path, str) and path.endswith(".desktop")]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return list(dict.fromkeys(entries))


def _config_file(args: list[str], cwd: Path | None) -> Path | None:
    for index, arg in enumerate(args):
        value = args[index + 1] if arg == "--config" and index + 1 < len(args) else \
            arg.split("=", 1)[1] if arg.startswith("--config=") else None
        if value is not None:
            path = Path(value)
            return path if path.is_absolute() or cwd is None else cwd / path
    return None


def _engine_state(process, module: str) -> Path | None:
    """The state folder a `serve` process uses, from its app folder's settings and its --config file merged as
    load_config merges them (its version's default when none names one); None when that cannot be told."""
    try:
        args, cwd = process.cmdline(), Path(process.cwd())
    except (psutil.Error, OSError):
        return None
    root, config = _install_root(process), _config_file(args, cwd)
    files = sources(config, root) if root is not None else [config] if config is not None else []
    if not files:
        return None
    paths: dict = {}
    for file in files:
        try:
            paths = {**paths, **(tomllib.loads(Path(file).read_text()).get("paths") or {})}
        except FileNotFoundError:
            continue
        except (OSError, ValueError, AttributeError):
            return None
    setting = paths.get("state_dir")
    if module == MODULES[0] and setting in (None, "", "auto"):
        return dirs.state_dir().resolve()            # this version's "auto" (engine/config.py state_dir_of)
    value = Path(str(setting or LEGACY_STATE_DIR)).expanduser()
    return (value if value.is_absolute() else cwd / value).resolve()


def _launch_held(path: Path) -> bool:
    """Whether a launcher holds this launch lock now (desktop/launcher.py _launch_lock), tried without waiting."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _profile_in_use(profile: Path) -> int | None:
    """The pid Chrome's SingletonLock in the profile names when it is this computer's and running, else None."""
    try:
        target = os.readlink(profile / "SingletonLock")
    except OSError:
        return None
    host, _, pid = target.rpartition("-")
    try:
        pid = int(pid)
    except ValueError:
        return None
    return pid if host == socket.gethostname() and pid_alive(pid) else None


def old_users(state: Path, profile: Path, *, entry: Path, record: Path | None = None, root: Path = ROOT,
              processes=psutil.process_iter, uid: int | None = None) -> list[str]:
    """What may still be using the previous version's state folder or profile, in plain words ([] when nothing does):
    its runtime file naming a running engine, its engine marker's owner running (engine/crashlog.py marker_owner), its
    launch lock held, a Chrome on its profile, a `serve` of this user (of either version) whose state folder it is,
    and a desktop entry of it (`entry`, and those its installation `record` lists) starting it from a folder that is
    there and is not this app's, or with an interpreter that is there when the entry does not say from which folder.
    Reads only: nothing is created or written, and a lock is only tried, without waiting."""
    from ..engine.crashlog import marker_owner
    uid = os.getuid() if uid is None else uid
    found = []
    run = state / "run"
    try:
        pid = json.loads((run / "server.json").read_text()).get("pid")
        if type(pid) is int and pid > 0 and pid_alive(pid):
            found.append(f"its engine (pid {pid}) is running ({run / 'server.json'})")
    except (OSError, ValueError, AttributeError):       # none, or not readable: the checks below see a running engine
        pass
    try:
        marker = json.loads((run / "engine.alive").read_text())
        if marker_owner(marker) == "running" and not any(f"pid {marker['pid']})" in said for said in found):
            found.append(f"its engine (pid {marker['pid']}) is running ({run / 'engine.alive'})")
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        pass
    if _launch_held(run / "launch.lock"):
        found.append(f"it is being started ({run / 'launch.lock'} is held)")
    if AppWindow(profile, processes=processes, uid=uid)() or _profile_in_use(profile) is not None:
        found.append(f"its app window is open (a Chrome on {profile})")
    for process in processes(["pid", "cmdline", "uids"]):
        try:
            owners, args = process.info.get("uids"), process.info.get("cmdline") or []
            if owners is None or owners.real != uid or not serve_command(args) or process.pid == os.getpid():
                continue
            # The console script's engine (sopa-companion serve) is this version's.
            module = args[args.index("-m") + 1] if "-m" in args else MODULES[0]
            used = _engine_state(process, module)
            if used is None or used == state.resolve():
                how = f"python -m {module}" if "-m" in args else COMMAND
                found.append(f"an engine (pid {process.pid}, {how}) may be using it")
        except (psutil.Error, OSError, ValueError, IndexError):
            continue
    for desktop in _previous_entries(entry, record):
        start = _previous_start(desktop)
        if start is None:
            continue
        elsewhere, python = start
        if elsewhere is not None:
            if elsewhere.is_dir() and elsewhere.resolve() != Path(root).resolve():
                found.append(f"the previous version is installed in {elsewhere} (its desktop entry {desktop}), whose "
                             "icon would start it again on new, empty folders")
        elif python is not None and (python.is_file() if python.is_absolute() else
                                     shutil.which(str(python)) is not None):
            # An interpreter that is still there, from a folder the entry does not name: it may be another one.
            found.append(f"the previous version's desktop entry {desktop} still starts it with {python}, from a "
                         "folder it does not name, and its icon would start it again on new, empty folders")
    return found


def _refused(why: str, state: Path, profile: Path, new: Path) -> MigrationRefused:
    return MigrationRefused(
        f"SoPA companion keeps its state in {new}, and the previous version's state folder {state} and window profile "
        f"{profile} are moved there at its first start, but not now: {why}. Nothing was moved or copied. Close the "
        "previous version, then start this one again; to run this copy beside it, give it its own [paths] state_dir "
        "in a --config file.")


def migrate(config: dict, *, root: Path = ROOT, home: str | Path | None = None, processes=psutil.process_iter,
            wait_s: float = WAIT_S) -> str | None:
    """Move the previous version's state folder and window profile into this default instance's state folder, once
    (module docstring). Returns the line saying what moved, or None when nothing had to (not the default instance; its
    state folder exists already; nothing of the previous version here; another start moved them meanwhile).
    MigrationRefused, with nothing moved, when they cannot be moved now."""
    if not default_instance(config, home):
        return None
    new = Path(config["paths"]["state_dir"])
    state, profile, entry = _legacy_paths(home)
    if os.path.lexists(new):
        return None
    has_state, has_profile = state.is_dir(), profile.is_dir()
    if not has_state and not has_profile:
        return None
    if reasons := old_users(state, profile, entry=entry, record=home_path(LEGACY_INSTALL_RECORD, home), root=root,
                            processes=processes):
        raise _refused("; ".join(reasons), state, profile, new)
    if has_profile and has_state and os.path.lexists(state / PROFILE):
        raise _refused(f"{state / PROFILE} is there already", state, profile, new)
    holder, made = (state, False) if has_state else (new.parent / f".{new.name}-{os.getpid()}.tmp", True)
    moved_profile = None
    try:
        if made:
            holder.mkdir(mode=0o700, parents=True)
        if has_profile:
            os.rename(profile, holder / PROFILE)
            moved_profile = holder / PROFILE
        os.rename(holder, new)                         # the commit: the new folder appears whole, profile included
    except FileNotFoundError:
        # A folder went while we moved it: another start of this version is moving them. Wait for its new folder.
        _undo(moved_profile, profile, holder if made else None)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if os.path.lexists(new):
                return None
            time.sleep(0.05)
        raise _refused("another start is moving them at this moment; start again", state, profile, new) from None
    except OSError as exc:
        left = _undo(moved_profile, profile, holder if made else None)
        steps = [f"mv {state} {new}"] if has_state else [f"mkdir -p {new}"]
        if has_profile:
            steps.append(f"mv {profile} {new / PROFILE}")
        raise _refused(f"they could not be moved in place ({exc}){left}. By hand, with nothing of either version "
                       f"running: {' && '.join(steps)}", state, profile, new) from None
    moved = [f"state folder {state}"] if has_state else []
    if has_profile:
        moved.append(f"window profile {profile}")
    return f"{LOG_PREFIX}: moved the previous version's {' and '.join(moved)} into {new}"


def _undo(moved: Path | None, back: Path, made: Path | None) -> str:
    """Put a profile moved for a move that failed back where it was, and remove the folder made for it. "" when all is
    as it was, else where the profile now is."""
    left = ""
    if moved is not None:
        try:
            os.rename(moved, back)
        except OSError as exc:
            left = f"; the window profile could not be put back ({exc}): it is in {moved}"
    if made is not None:
        try:
            made.rmdir()
        except OSError:
            pass
    return left
