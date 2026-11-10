"""Training patches in a folder the user chooses at every export: the checks a folder passes, the export dialog's
listing and suggestion, the folders exported into, and the export's own writes into such a folder.

A chosen folder is a training folder: the Cellpose GUI trains one model on every patch in the folder of the image it
has open, so an export goes straight into it, beside what is there, under names of its own (<sample>-<YYYYMMDD-HHMMSS>
-NN.tif and -NN_seg.npy), and never replaces a file. The app's records there are in the hidden .sopa-companion folder,
which Cellpose's listing skips: one manifest per export (sets/<set>.json, written last: the set exists once it does),
the GUI's record (gui.json) and its log (cellpose.log).

The path is typed by the user and the folder may be on a disk other people can write to. A path is checked when it is
listed and again at export, with every link resolved: the real path is the one shown, written to and remembered. The
helper writes an export into the app's own staging folder; the engine then copies the files in under hidden part names,
each created exclusively and never through a link, through folder handles opened without following links, renames them
to names it has just found free, and writes the manifest last. Names read from such a folder become paths only after
they are validated, and nothing the app starts runs in it or imports from it; the Cellpose GUI itself still loads the
folder's masks with pickle, so whoever can write to the folder can make the GUI run code.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import threading
import time
from typing import NamedTuple

from . import stain_levels
from .. import dirs, portable
from .config import ROOT
from .runs import RunError, atomic_text, now, user_cache

META = ".sopa-companion"
# The default place of new training folders: SOPA_COMPANION_PATCHES_HOME when set (the tests' and the harness's own
# folders, as SOPA_COMPANION_ZARR_LOCK_DIR names their locks), else this folder in the home folder.
HOME_ENV = "SOPA_COMPANION_PATCHES_HOME"
DEFAULT_HOME = "~/sopa-companion/training-patches"
# An export's file names start <stem>-: a sample's safe name and the export's local time (old sets: "patch").
STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
SET = re.compile(r"[0-9a-f]{32}")
HASH = re.compile(r"[0-9a-f]{64}")
# A notebook session's data folder (api/folders.py SessionFolders): removed when its session ends.
SESSION_DATA = re.compile(r"sopa-companion-[0-9a-f]{8}")
ZARR_MARKERS = ("zarr.json", ".zgroup", ".zarray")
# Cellpose lists a folder with an unescaped glob (cellpose/io.py get_image_files): these would make it list another one.
GLOB = frozenset("[]*?")
MAX_PATH, MAX_PART = 4096, 255          # bytes in UTF-8
MAX_MANIFEST = 1024 * 1024              # a manifest, a GUI record or the folders' list: read at most this
MAX_PATCHES = 64
MAX_ENTRIES = 2000                      # folders a listing shows
MAX_SETS = 200                          # manifests read in one folder
MAX_FOLDERS, MAX_KEYS = 100, 20         # the folders exported into, and the samples' keys kept with each
NAMES_TRIED = 50                        # <sample>-scale<k>, then -2 ... -50

# What a refusal says (one entry a line: frontend/test/patch-folder.test.mjs reads them for the banned words).
WORDS = {
    "empty": "Choose a folder for the patches.",
    "long": "This path is too long.",
    "relative": "Type the folder's whole path, starting with / or ~.",
    "control": "A folder path cannot hold line breaks or other invisible characters.",
    "glob": "The Cellpose GUI cannot open a folder whose path holds [ ] * or ?. Choose or make another folder.",
    "file": "{name} is a file, not a folder.",
    "top": "Choose or make a folder of its own for the patches, not {name}.",
    "hidden": "{name} is a hidden folder. Choose a folder you can see in a file browser.",
    "zarr": "{name} is a zarr store: patches inside it would mix with its data. Choose a folder outside it.",
    "session": "This folder belongs to a notebook session of the app and is removed when that session ends. Choose a folder of yours.",
    "app": "{name} belongs to the app or its Python environment. Choose a folder of yours.",
    "old": "This folder holds patches exported by an earlier version of the app. Choose another folder.",
    "records": "{name} holds a .sopa-companion entry that is not a folder. Choose another folder.",
    "write": "You cannot write to {name}.",
    "make": "{name} could not be made: {reason}",
    "read": "{name} could not be read: {reason}",
    "written": "The patches could not be written into {name}: {reason}",
    "clash": "A file named {file} appeared in {name} while the patches were made. Try again.",
    "gone": "The exported patches are no longer in {folder}. Export them again to go on.",
    "gui_missing": "The Cellpose GUI is not installed where Cellpose runs ({python}). Install it there, then open it again: {command}",
}


def words(key: str, **values) -> str:
    return WORDS[key].format(**values)


def refused(key: str, status: int = 422, **values) -> RunError:
    """A refusal about the folder: the page's export dialog keeps itself open on these and outlines the path."""
    return RunError(words(key, **values), status, field="folder")


def safe_name(text) -> str:
    """A sample's name as it can start a file or folder name: letters, digits, ".", "_" and "-", at most 40 of them,
    starting and ending with a letter or digit."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "")).strip("-._")[:40].strip("-._") or "sample"


def default_home() -> Path:
    return Path(os.path.expanduser(os.environ.get(HOME_ENV) or DEFAULT_HOME))


def stamp() -> str:
    """An export's local time as its file names carry it."""
    return time.strftime("%Y%m%d-%H%M%S")


def export_channels(settings: dict) -> list:
    """The channels an export's patches hold, in their order, as the patch helper writes them in its manifest
    (zarr_bridge/stain_patches.py Result.export): Cellpose's padded to three with nulls."""
    settings = settings if isinstance(settings, dict) else {}
    channels = settings.get("channels", [settings.get("nucleus"), settings.get("cytoplasm"), None])
    channels = list(channels) if isinstance(channels, list) else []
    if settings.get("method") != "instanseg":
        channels += [None] * (3 - len(channels))
    return channels


def read_small(path, limit: int = MAX_MANIFEST, dir_fd: int | None = None):
    """The JSON in a regular file of at most `limit` bytes, never through a link (OSError or ValueError otherwise): a
    FIFO or a device is refused before a byte is read."""
    flags = os.O_RDONLY | portable.O_NOFOLLOW | portable.O_NONBLOCK | portable.O_CLOEXEC | portable.O_BINARY
    fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("not a small file")
        data = b""
        while len(data) <= limit:
            chunk = os.read(fd, limit + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ValueError("not a small file")
    try:
        return json.loads(data)
    except RecursionError:                     # nested deeper than the parser goes: not a manifest either
        raise ValueError("not a manifest") from None


def valid_manifest(data, set: str | None = None, stem: str | None = None) -> dict | None:
    """`data` when it is an export's manifest whose names may become paths, else None: its set (the one expected, when
    given), its stem (an old set's is "patch") and at most 64 patches, each numbered once from 1 to 64, with a box of
    four whole numbers, files named <stem>-NN.tif and <stem>-NN_seg.npy, the exported mask's hash and its cell count."""
    if not isinstance(data, dict):
        return None
    name, own = data.get("set"), data.get("stem", "patch")
    if not isinstance(name, str) or not SET.fullmatch(name) or set is not None and name != set:
        return None
    if not isinstance(own, str) or not STEM.fullmatch(own) or stem is not None and own != stem:
        return None
    patches = data.get("patches")
    if not isinstance(patches, list) or len(patches) > MAX_PATCHES:
        return None
    seen = []                                   # the numbers so far (`set` is the expected set's name here)
    for patch in patches:
        if not isinstance(patch, dict):
            return None
        number, box = patch.get("id"), patch.get("box")
        if type(number) is not int or not 1 <= number <= MAX_PATCHES or number in seen:
            return None
        seen.append(number)
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            return None
        if patch.get("image") != f"{own}-{number:02d}.tif" or patch.get("mask") != f"{own}-{number:02d}_seg.npy":
            return None
        if not isinstance(patch.get("export_hash"), str) or not HASH.fullmatch(patch["export_hash"]):
            return None
        if type(patch.get("cells")) is not int or patch["cells"] < 0:
            return None
    return data


def _when(created, fallback: float) -> float:
    """A manifest's time (its `created`) as seconds, for ordering sets; `fallback` when it has none."""
    try:
        return datetime.fromisoformat(created).timestamp()
    except (TypeError, ValueError):
        return fallback


def legacy(root) -> list[tuple[float, Path, dict]]:
    """The sets of the earlier layout (<root>/<run>/<set>/patches.json), oldest first: (time, set folder, manifest).
    Their manifests are read as leniently as they always were (they name no path here); one that cannot be read is
    left out."""
    found = []
    for path in Path(root).glob("*/*/patches.json"):
        try:
            manifest = read_small(path)
            found.append((_when(manifest.get("created") if isinstance(manifest, dict) else None, path.stat().st_mtime),
                          path.parent, manifest if isinstance(manifest, dict) else {}))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda item: item[0])


def legacy_sets(root) -> list[Path]:
    """The set folders of the earlier layout under `root`, oldest first."""
    return [folder for _, folder, _ in legacy(root)]


def _real(path) -> Path:
    return Path(os.path.realpath(path))


def _name(path: Path) -> str:
    return path.name or "/"


def _folders(base: Path) -> list[str]:
    """The names of the folders in `base` that are not hidden, sorted; none when it cannot be read."""
    names = []
    try:
        with os.scandir(base) as listed:
            for entry in listed:
                try:
                    if not entry.name.startswith(".") and entry.is_dir():
                        names.append(entry.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(names)


def _nearest(path: Path) -> Path:
    """The deepest folder on the way to `path` that is there (`path` itself when it is)."""
    for folder in (path, *path.parents):
        if os.path.lexists(folder):
            return folder
    return Path("/")


def zarr_store(real: Path) -> str | None:
    """The zarr store `real` is in or is (its name), else None: a part named *.zarr, or a folder on the way that holds a
    zarr marker, the outermost one."""
    part = next((p for p in real.parts[1:] if p.lower().endswith(".zarr")), None)
    if part is not None:
        return part
    folder = Path(real.parts[0])
    for part in real.parts[1:]:
        folder = folder / part
        if not os.path.isdir(folder):
            break
        if any(os.path.lexists(folder / marker) for marker in ZARR_MARKERS):
            return folder.name
    return None


class Target(NamedTuple):
    real: Path
    exists: bool


class Source(NamedTuple):
    """An exported set as the panel reads it: where its patches are, its validated manifest (None when it is gone or
    unreadable), where its GUI's record and log are, and whether it is in a chosen folder (else the earlier layout)."""
    folder: Path
    manifest: dict | None
    meta: Path
    flat: bool


# Windows has none of these flags and opens no folder (an export there is refused: StainPatches.export).
OPEN_DIR = os.O_RDONLY | portable.O_DIRECTORY | portable.O_NOFOLLOW | portable.O_CLOEXEC
CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | portable.O_NOFOLLOW | portable.O_CLOEXEC | portable.O_BINARY


class Folder:
    """A chosen folder opened for one export: handles to it, its records and their sets folder, every later call made
    through them, so a folder swapped for a link after it was checked is never followed. The export's own files are
    tracked: undo() removes those it put in place, remove() its part files left."""

    def __init__(self, real: Path, fd: int, meta: int, sets: int):
        self.real, self.fd, self.meta, self.sets = real, fd, meta, sets
        self.parts: list[tuple[str, str, tuple[int, int]]] = []    # (part name, final name, (device, inode))
        self.placed: list[tuple[str, tuple[int, int]]] = []        # final names renamed into place
        self.manifest: tuple[str, tuple[int, int]] | None = None

    def failed(self, exc: OSError) -> RunError:
        self.undo()
        self.remove()
        return refused("written", name=_name(self.real), reason=exc.strerror or str(exc))

    def copy_in(self, staging: Path, manifest: dict, name: str) -> None:
        """The validated manifest's files, from the app's staging folder into hidden part files of the folder:
        .<final name>.<set[:8]>.part, each created exclusively."""
        try:
            for patch in manifest["patches"]:
                for key in ("image", "mask"):
                    final = patch[key]
                    part = f".{final}.{name[:8]}.part"
                    source = os.open(staging / final, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                    try:
                        target = os.open(part, CREATE, 0o666, dir_fd=self.fd)
                        try:
                            info = os.fstat(target)
                            self.parts.append((part, final, (info.st_dev, info.st_ino)))
                            while chunk := os.read(source, 1 << 20):
                                view = memoryview(chunk)
                                while view:
                                    view = view[os.write(target, view):]
                        finally:
                            os.close(target)
                    finally:
                        os.close(source)
        except OSError as exc:
            raise self.failed(exc) from None

    def publish(self) -> None:
        """Each part file renamed to its final name, which must be free: a name someone else took meanwhile undoes
        this export's own renames and is refused (409); nothing is replaced."""
        try:
            for part, final, ident in list(self.parts):
                try:
                    os.lstat(final, dir_fd=self.fd)
                except FileNotFoundError:
                    os.rename(part, final, src_dir_fd=self.fd, dst_dir_fd=self.fd)
                    self.placed.append((final, ident))
                    self.parts.remove((part, final, ident))
                    continue
                self.undo()
                self.remove()
                raise refused("clash", 409, file=final, name=_name(self.real))
        except OSError as exc:
            raise self.failed(exc) from None

    def write_manifest(self, name: str, manifest: dict) -> None:
        """sets/<set>.json, created exclusively: the export's commit point."""
        try:
            fd = os.open(f"{name}.json", CREATE, 0o666, dir_fd=self.sets)
            info = os.fstat(fd)
            self.manifest = (f"{name}.json", (info.st_dev, info.st_ino))
            with os.fdopen(fd, "w") as stream:
                json.dump(manifest, stream)
        except OSError as exc:
            raise self.failed(exc) from None

    def undo(self) -> None:
        """Take back what this export put in place: a file only while it is still the one it made (same inode)."""
        for handle, (final, ident) in [(self.fd, item) for item in self.placed] + \
                ([(self.sets, self.manifest)] if self.manifest else []):
            try:
                info = os.lstat(final, dir_fd=handle)
                if (info.st_dev, info.st_ino) == ident:
                    os.unlink(final, dir_fd=handle)
            except OSError:
                pass
        self.placed, self.manifest = [], None

    def remove(self) -> None:
        """The export's part files still there, by their own names."""
        for part, _, _ in self.parts:
            try:
                os.unlink(part, dir_fd=self.fd)
            except OSError:
                pass
        self.parts = []

    def close(self) -> None:
        for fd in (self.sets, self.meta, self.fd):
            os.close(fd)


def _open_or_make(name: str, parent: int) -> int:
    """A folder in `parent` (a handle), made when missing, opened without following a link."""
    try:
        os.mkdir(name, dir_fd=parent)
    except FileExistsError:
        pass
    return os.open(name, OPEN_DIR, dir_fd=parent)


class PatchFolders:
    """The chosen folders of one engine (engine/stain_patches.py StainPatches.folders): their checks, the export
    dialog's listing, the suggestion, the list of folders exported into, the sets found in them, and the stems of
    exports under way."""

    def __init__(self, patches):
        self.patches, self.runs = patches, patches.runs
        # The folders exported into, newest first: the data folder's (a notebook session's goes with it).
        self.path = self.runs.root.parent / "patch-folders.json"
        self.lock = threading.Lock()
        self.reserved: set[tuple[str, str]] = set()          # (folder, stem) of exports under way
        self.manifests: dict[tuple, dict | None] = {}        # (path, mtime_ns, size) -> manifest or None

    # ---- where the app is, and what a folder may not be ------------------------------------------------------------
    def notebook(self) -> bool:
        return self.runs.session is not None

    def base(self) -> Path:
        """Where new training folders are suggested: a notebook's patches_dir when it was given one, else the default
        place."""
        given = self.runs.patches_dir
        return Path(os.path.expanduser(given)) if given else default_home()

    def session_roots(self) -> list[Path]:
        """The places notebook sessions keep their state folders (api/folders.py state_root)."""
        places = [] if portable.WINDOWS else [Path(f"/run/user/{os.getuid()}") / "sopa-companion"]
        if not portable.LINUX:                       # their only place there: the user's cache
            places.append(dirs.user_cache() / "sessions")
        if runtime := os.environ.get("XDG_RUNTIME_DIR"):
            places.append(Path(runtime) / "sopa-companion")
        return [_real(place) for place in places]

    def app_roots(self) -> list[Path]:
        """The app's own folders: its data, state and code folders, the user's cache of it, and the Python environments
        it runs in (one holding the home folder, or /, is left out: a refusal of the whole home is no rule)."""
        roots = [_real(root) for root in (self.runs.root.parent, self.runs.state_dir, ROOT, user_cache())]
        home = _real(Path.home())
        environments = [Path(sys.prefix), Path(sys.base_prefix)]
        python = self.patches.stain.python
        if python and os.path.isabs(python):            # [stain] python's environment: <env>/bin/python
            environments += [path.parents[1] for path in (Path(python), Path(python).resolve())
                             if len(path.parents) > 1]
        return roots + [root for root in map(_real, environments) if not home.is_relative_to(root)]

    # ---- checking a path ---------------------------------------------------------------------------------------------
    def resolve(self, value) -> tuple[Path | None, str | None]:
        """(the real path, None), or (None, why) for a path refused before it is resolved: every link in it is resolved,
        in its last part and in its parents alike, and only the real path is used from then on."""
        if not isinstance(value, str) or not value:
            return None, words("empty")
        try:
            encoded = value.encode()
        except UnicodeEncodeError:
            return None, words("control")
        if len(encoded) > MAX_PATH or any(len(part.encode()) > MAX_PART for part in value.split("/")):
            return None, words("long")
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            return None, words("control")
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            return None, words("relative")
        return _real(expanded), None

    def problem(self, real: Path) -> str | None:
        """Why a resolved path may not take patches, the first rule that fails; None when it may. Nothing is written."""
        if any(c in GLOB for c in str(real)):
            return words("glob")
        home, temporary = _real(Path.home()), _real(tempfile.gettempdir())
        if real == Path("/") or real == home or real == temporary:
            return words("top", name="/" if real == Path("/") else "your home folder" if real == home else str(real))
        hidden = next((part for part in real.parts[1:] if part.startswith(".")), None)
        if hidden is not None:
            return words("hidden", name=hidden)
        if any(SESSION_DATA.fullmatch(part) for part in real.parts[1:]) or \
                any(real.is_relative_to(root) for root in self.session_roots()):
            return words("session")
        if any(real.is_relative_to(root) for root in self.app_roots()):
            return words("app", name=_name(real))
        store = zarr_store(real)
        if store is not None:
            return words("zarr", name=store)
        try:
            info = os.stat(real)
        except (FileNotFoundError, NotADirectoryError):          # not there yet: made at export
            folder = _nearest(real)
            if not os.path.isdir(folder):
                return words("file", name=_name(folder))
            return None if os.access(folder, os.W_OK | os.X_OK) else words("write", name=_name(folder))
        except OSError as exc:
            return words("read", name=_name(real), reason=exc.strerror or str(exc))
        if not stat.S_ISDIR(info.st_mode):
            return words("file", name=_name(real))
        if os.path.lexists(real / "patches.json"):
            return words("old")
        try:
            if not stat.S_ISDIR(os.lstat(real / META).st_mode):
                return words("records", name=_name(real))
        except FileNotFoundError:
            pass
        except OSError as exc:
            return words("read", name=_name(real), reason=exc.strerror or str(exc))
        return None if os.access(real, os.W_OK | os.X_OK) else words("write", name=_name(real))

    def check(self, value) -> Target:
        """The folder an export goes into, checked as the dialog's listing checks it (422 with field "folder")."""
        real, why = self.resolve(value)
        if why is None:
            why = self.problem(real)
        if why is not None:
            raise RunError(why, 422, field="folder")
        return Target(real, os.path.isdir(real))

    # ---- what the dialog shows ---------------------------------------------------------------------------------------
    def key(self, run, draft) -> list:
        """What a training folder is suggested for: the run's sample, the level its patches are cut at and their
        channels (as stain/job.json gives them)."""
        try:
            settings = json.loads((run.dir / "stain" / "job.json").read_text()).get("settings")
        except (OSError, ValueError, AttributeError):
            settings = None
        return [run.meta.get("sample"), draft.get("level", 0), export_channels(settings)]

    def named(self, run, key) -> str:
        return f"{safe_name(run.meta.get('sample_name') or run.meta.get('sample'))}-{stain_levels.level_name(key[1])}"

    def listing(self, run, path=None) -> dict:
        """The export dialog's view of a folder (`path`; none: the suggestion): its real path, its parent, its folders
        (none hidden, no zarr store), the shortcuts, a free name for a new folder, the patches it holds, whether they
        were cut at another resolution or of other channels, and why it may not take patches. Nothing is written."""
        with run.lock:
            key = self.key(run, self.patches.draft(run))
        if not path:
            path = str(self.suggest(run, key))
        real, why = self.resolve(path)
        if real is None:
            return {"path": path, "exists": False, "parent": None, "entries": [], "shortcuts": self.shortcuts(None),
                    "new": None, "holds": None, "mixes": None, "refused": why}
        why = self.problem(real)
        exists, entries, sets = os.path.lexists(real), [], []
        if os.path.isdir(real):
            try:
                entries = self.entries(real)
            except OSError as exc:
                why = why or words("read", name=_name(real), reason=exc.strerror or str(exc))
            sets = self.flat_sets(real)
        holds = mixes = None
        if sets:
            samples = sorted({str(m.get("sample_name") or m.get("sample") or "") for m in sets}, key=str.lower)
            holds = {"patches": sum(len(m["patches"]) for m in sets), "sets": len(sets), "samples": samples[:3],
                     "more": max(0, len(samples) - 3)}
            mixes = "resolution" if any(m.get("level", 0) != key[1] for m in sets) else \
                "channels" if any(m.get("channels") != key[2] for m in sets) else None
        return {"path": str(real), "exists": exists, "parent": str(real.parent) if real.parent != real else None,
                "entries": entries, "shortcuts": self.shortcuts(real), "new": self.free(real, self.named(run, key)),
                "holds": holds, "mixes": mixes, "refused": why}

    def entries(self, real: Path) -> list[dict]:
        """The folders in `real` (links to folders too), in case-insensitive order: none hidden, no zarr store; at most
        MAX_ENTRIES. `patches`: the folder holds an export of this app."""
        names = []
        with os.scandir(real) as listed:
            for entry in listed:
                if entry.name.startswith(".") or entry.name.lower().endswith(".zarr"):
                    continue
                try:
                    if entry.is_dir():
                        names.append(entry.name)
                except OSError:
                    continue
        entries = []
        for name in sorted(names, key=lambda n: (n.lower(), n)):
            child = real / name
            if any(os.path.lexists(child / marker) for marker in ZARR_MARKERS):
                continue
            entries.append({"name": name, "path": str(child), "patches": self.has_sets(child)})
            if len(entries) == MAX_ENTRIES:
                break
        return entries

    @staticmethod
    def has_sets(folder: Path) -> bool:
        try:
            with os.scandir(folder / META / "sets") as listed:
                return any(entry.name.endswith(".json") for entry in listed)
        except OSError:
            return False

    def shortcuts(self, shown: Path | None) -> list[dict]:
        """Home, and up to four folders exported into, newest first, that are there and are not the folder shown."""
        home = _real(Path.home())
        found = [{"kind": "home", "name": _name(home), "path": str(home)}]
        for entry in self.registry():
            if len(found) == 5:
                break
            folder = Path(entry["folder"])
            if folder != shown and os.path.isdir(folder):
                found.append({"kind": "earlier", "name": _name(folder), "path": str(folder)})
        return found

    @staticmethod
    def free(real: Path, name: str) -> str | None:
        """The first of `name`, `name`-2, ... that nothing in `real` is called."""
        for n in range(1, NAMES_TRIED + 1):
            candidate = name if n == 1 else f"{name}-{n}"
            if not os.path.lexists(real / candidate):
                return candidate
        return None

    def suggest(self, run, key=None) -> Path:
        """Where the dialog opens: the folder this sample's patches of this resolution and channels went to last, while
        it is there and may take patches; else a folder named after the sample and its resolution in the base place
        (<name>, <name>-2, ...: the first not there yet, or holding no export of another sample, resolution or
        channels); else the home folder, which the dialog shows refused."""
        if key is None:
            with run.lock:
                key = self.key(run, self.patches.draft(run))
        for entry in self.registry():
            folder = Path(entry["folder"])
            if key in entry["keys"] and os.path.isdir(folder) and self.problem(folder) is None:
                return folder
        base, name = self.base(), self.named(run, key)
        for n in range(1, NAMES_TRIED + 1):
            real = _real(base / (name if n == 1 else f"{name}-{n}"))
            if self.problem(real) is None and (not os.path.isdir(real) or all(
                    self.set_key(m) == key for m in self.flat_sets(real))):
                return real
        return _real(Path.home())

    @staticmethod
    def set_key(manifest: dict) -> list:
        return [manifest.get("sample"), manifest.get("level", 0), manifest.get("channels")]

    # ---- the sets in chosen folders ----------------------------------------------------------------------------------
    def flat_sets(self, folder) -> list[dict]:
        """The valid manifests of the exports in `folder` (at most MAX_SETS read), each read once while unchanged."""
        sets = Path(folder) / META / "sets"
        try:
            with os.scandir(sets) as listed:
                names = sorted(e.name for e in listed if e.name.endswith(".json") and SET.fullmatch(e.name[:-5]))
        except OSError:
            return []
        found = []
        for name in names[:MAX_SETS]:
            manifest = self.manifest(sets / name, name[:-5])
            if manifest is not None:
                found.append(manifest)
        return found

    def manifest(self, path: Path, name: str) -> dict | None:
        try:
            info = os.lstat(path)
        except OSError:
            return None
        key = (str(path), info.st_mtime_ns, info.st_size)
        with self.lock:
            if key in self.manifests:
                return self.manifests[key]
        try:
            value = valid_manifest(read_small(path), set=name)
        except (OSError, ValueError):
            value = None
        with self.lock:
            if len(self.manifests) > 4096:
                self.manifests.clear()
            self.manifests[key] = value
        return value

    def places(self) -> list[Path]:
        """Where sets in chosen folders are looked for: the folders exported into, the default place and the folders in
        it, and a notebook's patches_dir and the folders in it."""
        places = [Path(entry["folder"]) for entry in self.registry()]
        for base in [default_home(), *([self.base()] if self.runs.patches_dir else [])]:
            places.append(base)
            places += [base / name for name in _folders(base)[:MAX_ENTRIES]]
        return list(dict.fromkeys(_real(place) for place in places))

    def sources(self, sample=None) -> list[tuple[float, dict]]:
        """The sets in chosen folders, as (time, {folder, manifest, set, run}), of `sample` only when given."""
        found = []
        for folder in self.places():
            for manifest in self.flat_sets(folder):
                if sample is None or manifest.get("sample") == sample:
                    found.append((_when(manifest.get("created"), 0.), {
                        "folder": folder, "manifest": manifest, "set": manifest["set"], "run": manifest.get("run")}))
        return found

    # ---- the folders exported into -----------------------------------------------------------------------------------
    def read_registry(self) -> tuple[list[dict], bool]:
        """The folders exported into, newest first, and whether a file there could not be read (it counts as empty)."""
        try:
            data = read_small(self.path)
        except FileNotFoundError:
            return [], False
        except (OSError, ValueError):
            return [], True
        if not isinstance(data, dict) or data.get("v") != 1 or not isinstance(data.get("folders"), list):
            return [], True
        entries = []
        for entry in data["folders"][:MAX_FOLDERS]:
            if (isinstance(entry, dict) and isinstance(entry.get("folder"), str) and os.path.isabs(entry["folder"])
                    and isinstance(entry.get("keys"), list)):
                entries.append({"folder": entry["folder"], "used": str(entry.get("used") or ""),
                                "keys": [k for k in entry["keys"] if isinstance(k, list) and len(k) == 3][:MAX_KEYS]})
        return entries, False

    def registry(self) -> list[dict]:
        return self.read_registry()[0]

    def used(self, folder: Path, key: list) -> None:
        """Note an export into `folder` for its key, first in the list: the next export of that key suggests it. A list
        that could not be read starts again (said once in the engine log); one that cannot be written is said there
        too, and the export stands."""
        with self.patches.lock:
            entries, unreadable = self.read_registry()
            old = next((e for e in entries if e["folder"] == str(folder)), None)
            keys = [key] + [k for k in (old or {}).get("keys", []) if k != key]
            entries = [{"folder": str(folder), "used": now(), "keys": keys[:MAX_KEYS]}] + \
                [e for e in entries if e["folder"] != str(folder)]
            try:
                atomic_text(self.path, json.dumps({"v": 1, "folders": entries[:MAX_FOLDERS]}))
            except OSError as exc:
                print(f"SoPA companion: the list of training patch folders could not be written: {exc}", flush=True)
                return
        if unreadable:
            print(f"SoPA companion: {self.path} could not be read; it was written again with the folder exported "
                  "into now", flush=True)

    # ---- an export under way -----------------------------------------------------------------------------------------
    def reserve(self, real: Path, sample_name, count: int) -> str:
        """A stem of the export's own in `real`: <sample>-<YYYYMMDD-HHMMSS> in local time, with -2, -3, ... when another
        export under way here has it or a file of its names is already in the folder."""
        base = f"{safe_name(sample_name)}-{stamp()}"
        with self.lock:
            for n in range(1, 1000):
                stem = base if n == 1 else f"{base}-{n}"
                if (str(real), stem) in self.reserved or any(
                        os.path.lexists(real / f"{stem}-{i:02d}{end}") for i in range(1, count + 1)
                        for end in (".tif", "_seg.npy")):
                    continue
                self.reserved.add((str(real), stem))
                return stem
        raise RunError("The patch names could not be made. Try again.", 503)

    def release(self, real: Path, stem: str) -> None:
        with self.lock:
            self.reserved.discard((str(real), stem))

    @contextmanager
    def opened(self, real: Path):
        """The chosen folder for an export (Folder): its missing parts made, each opened without following a link from
        the deepest folder that is there, and its records folders made the same way. A part that cannot be made gives
        `make`, records that are not folders `records`. On leaving, the export's part files left are removed."""
        missing, folder = [], real
        while not os.path.lexists(folder) and folder != folder.parent:
            missing.append(folder.name)
            folder = folder.parent
        fd = meta = sets = None
        try:
            current = _name(folder)
            try:
                fd = os.open(folder, OPEN_DIR)
                for current in reversed(missing):
                    child = _open_or_make(current, fd)
                    os.close(fd)
                    fd = child
            except OSError as exc:
                raise refused("make", name=current, reason=exc.strerror or str(exc)) from None
            try:
                meta = _open_or_make(META, fd)
                sets = _open_or_make("sets", meta)
            except OSError as exc:
                if exc.errno in (errno.ENOTDIR, errno.ELOOP):     # a file, or a link, where a records folder goes
                    raise refused("records", name=_name(real)) from None
                raise refused("written", name=_name(real), reason=exc.strerror or str(exc)) from None
        except BaseException:
            for handle in (sets, meta, fd):
                if handle is not None:
                    os.close(handle)
            raise
        handle = Folder(real, fd, meta, sets)
        try:
            yield handle
        finally:
            handle.remove()
            handle.close()
