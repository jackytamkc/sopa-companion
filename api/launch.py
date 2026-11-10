"""Open one SpatialData sample in an isolated companion session."""
from __future__ import annotations

import asyncio
import builtins
from collections.abc import Mapping
import importlib
import importlib.util
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
import threading
import tomllib
import warnings

from .. import dirs, locate, portable
from ..desktop import browser
from .folders import SessionFolders, remember_scratch, state_root, sweep, sweep_scratch, write_config
from .handle import CompanionApp
from .protocol import build_session_config
from .result import CompanionError, CompanionWarning
from .window import check_flags, open_mode

ROOT = Path(__file__).resolve().parents[1]
_ENGINE_MODULES = locate.ENGINE_MODULES


def _settings() -> dict:
    """Read settings without importing the server or an engine environment: the app folder's config.toml (or
    config.example.toml), config.local.toml, then the user's own ~/.config/sopa-companion/config.toml ($XDG_CONFIG_HOME:
    dirs.user_config_file).

    launch() takes from them [api] engine_python, scratch_dir and model_cache, [desktop] browser and chrome_flags (where
    the kernel opens the app: the engine never needs them), [proseg] binary, [analysis] python and gpu, [stain] python
    and gpu, and [qc] python and gpu, and hands the engine what it uses in session.toml; [stain] instanseg_python only
    while the internal switch [stain] instanseg_enabled is on (InstanSeg is dormant), which session.toml then carries
    too. An interpreter's "auto" (or empty) is this kernel's own Python when it has the feature's packages (locate.py),
    resolved here: session.toml carries a full path or "off". The engine reads the app folder's files and then
    session.toml, never the user's file: any other setting there does not reach it."""
    def merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    user = dirs.user_config_file()
    paths = [ROOT / ("config.toml" if (ROOT / "config.toml").is_file() else "config.example.toml"),
             ROOT / "config.local.toml", user]
    result = {}
    for path in paths:
        if path.is_file():
            result = merge(result, tomllib.loads(path.read_text(encoding="utf-8")))
    return result


def _running_loop():
    """Capture a main-thread loop, without importing IPython in an ordinary script."""
    try:
        loop = asyncio.get_running_loop()
        return loop.call_soon_threadsafe
    except RuntimeError:
        pass
    get_shell = getattr(builtins, "get_ipython", None)
    if get_shell is None and (ipython := sys.modules.get("IPython")) is not None:
        get_shell = getattr(ipython, "get_ipython", None)
    shell = get_shell() if get_shell is not None else None
    loop = getattr(getattr(shell, "kernel", None), "io_loop", None)
    asyncio_loop = getattr(loop, "asyncio_loop", None)
    if asyncio_loop is not None and asyncio_loop.is_running():
        return loop.add_callback
    return None


def _names(sdata) -> list[str]:
    names = set()
    for kind in ("images", "labels", "points", "shapes", "tables"):
        collection = getattr(sdata, kind)
        names.update(collection.data.copy() if hasattr(collection, "data") else collection.copy())
    return sorted(names)


class _ObjectZarr:
    """The object's zarr as a full path. spatialdata keeps sdata.path as it was given, a relative path included, so it is
    resolved when a path is first seen and again only when the object's path changes (a write to another folder):
    resolved at every look, it would follow the kernel's working folder (a %cd) to another sample's zarr of that name."""

    def __init__(self):
        self.raw, self.full = None, None

    def __call__(self, path) -> Path | None:
        if path is None or "://" in str(path):
            return None
        if path != self.raw:
            self.raw, self.full = path, Path(path).expanduser().resolve()
        return self.full


def _snapshot(sdata, where: _ObjectZarr) -> dict:
    from sopa import settings

    auto_save = bool(settings.auto_save_on_disk)
    path = where(sdata.path)
    target = str(path) if auto_save and path is not None and path.is_dir() else None
    return {"names": _names(sdata), "zarr": target, "auto_save": auto_save}


def _path(value):
    """An interpreter's path, absolute, its symlinks kept: a venv's bin/python is a symlink to its base interpreter,
    which, started by the resolved path, finds none of the venv's packages."""
    return "" if not value else os.path.abspath(os.path.expanduser(str(value)))


def _python(feature: str, config: dict, lacking: dict | None = None) -> str:
    """What session.toml says for `feature`'s interpreter: a full path, or "off" (never "auto": the session's engine
    may run in another interpreter, [api] engine_python, and must not look again there). An "auto" that found the
    feature's modules missing in this kernel is "off" with what it found kept in `lacking` (session.toml's [locate]),
    so the session's engine names what the kernel lacks."""
    section, key, _ = locate.FEATURES[feature]
    found = locate.python_for(feature, (config.get(section) or {}).get(key))
    if found.how == "setting":
        if found.problem:
            warnings.warn("An optional Python environment could not be found: " + found.path, CompanionWarning,
                          stacklevel=3)
            return locate.OFF
        return _path(found.path)
    if found.how == locate.AUTO and found.missing and lacking is not None:
        lacking[feature] = found.report()
    return found.path or locate.OFF


def _attach_delivery(app, sdata):
    """The delivery's seam: attach(app, sdata) returns receive/apply_pending/close methods.

    The adapter schedules its own idle applies with the loop captured in app._schedule;
    it sends control lines through app._send and records successful names with app._record_added.
    """
    from .deliver import attach

    return attach(app, sdata)


def _trainer_device(index: int) -> str:
    """The GPU the mixing check's learning process would get for the engine's qc `index`: the index-th entry of this
    kernel's CUDA_VISIBLE_DEVICES, read as CUDA reads it (up to its first invalid entry), which the engine inherits, or
    the index itself when that is unset; "" when there is no such entry (engine/children.py gpu)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    entries = []
    for entry in (part.strip() for part in visible.split(",")):
        if not entry or entry.startswith("-"):
            break
        entries.append(entry)
    return entries[index] if index < len(entries) else ""


def _in_zarr(path: str) -> bool:
    """Whether `path` (as given, or with its links resolved) is in a zarr store: a part named *.zarr, or a folder on the
    way that holds a zarr marker. Training patches there would mix with the store's data."""
    for given in (Path(path), Path(os.path.realpath(path))):
        if any(part.lower().endswith(".zarr") for part in given.parts[1:]):
            return True
        folder = Path(given.parts[0])
        for part in given.parts[1:]:
            folder = folder / part
            if not folder.is_dir():
                break
            if any(os.path.lexists(folder / marker) for marker in ("zarr.json", ".zgroup", ".zarray")):
                return True
    return False


def _patches_folder(value) -> str:
    """patches_dir as a full path, made when missing; ValueError inside a zarr store (checked before anything is made)
    or unless it is a folder of yours (not a link)."""
    path = _path(value)
    if path:
        if _in_zarr(path):
            raise ValueError("patches_dir must not be inside a zarr store")
        folder = Path(path)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            info = folder.lstat()
        except OSError as exc:
            raise ValueError(f"patches_dir could not be made: {exc}") from None
        if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
            raise ValueError("patches_dir must be a folder of yours, not a link")
    return path


def launch(sdata, *, tool: str | None = None, points_key: str | None = None, open: str = "auto",
           copy: bool = False, gpu: int | Mapping[str, int] | None = None,
           scratch_dir: str | os.PathLike | None = None, patches_dir: str | os.PathLike | None = None,
           model_cache: str | os.PathLike | None = None, mixing_check: bool = False,
           timeout: float = 120.0) -> CompanionApp:
    """Open the SoPA companion on a SpatialData saved to a local zarr.

    In a notebook with a running event loop this returns at once; you can keep
    working while the app is open. In a script it runs until the app closes.
    Every save in the app is added to this object automatically on its main thread.

    ``tool`` opens "proseg" or "stain" directly. ``points_key`` chooses transcripts
    when there are several point elements. ``open`` is "auto" (with a display, the
    app opens where [desktop] browser says: your default browser, or with "chrome"
    an app window of its own; otherwise a link), "window", "link", or "none".
    ``open()`` on what this returns opens the app again.
    ``scratch_dir`` holds temporary run files; they are removed at close.
    ``patches_dir`` is where Export patches suggests a folder for training patches
    (one per sample and resolution) and where earlier ones are found again.
    ``model_cache`` keeps learned models. ``gpu`` is an index or a mapping with analysis, stain and qc indices,
    counted within this kernel's CUDA_VISIBLE_DEVICES; the mixing check learns on
    GPU 0 only. ``mixing_check`` starts the mixing check on. ``timeout`` limits
    engine startup. Private copies and objects without a local zarr are not
    available yet. Settings ([api], [desktop] browser and chrome_flags,
    [proseg] binary, and the Python environments and GPUs of [analysis], [stain]
    and [qc]) come from the app's settings files and your own settings file
    (~/.config/sopa-companion/config.toml); each helper runs in this kernel's
    Python when it has what it needs, unless your settings name another.
    """
    if not (sys.platform.startswith("linux") or sys.platform in ("darwin", "win32")):
        raise CompanionError("The companion needs Linux, macOS or Windows")
    if threading.current_thread() is not threading.main_thread():
        raise CompanionError("Open the companion on the main thread")
    if tool not in (None, "proseg", "stain"):
        raise ValueError('tool must be "proseg", "stain" or None')
    if type(copy) is not bool or type(mixing_check) is not bool:
        raise ValueError("copy and mixing_check must be True or False")
    if copy:
        raise ValueError("Private copies are not available yet; open the saved zarr instead")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive, finite number of seconds")
    mode = open_mode(open)
    from spatialdata import SpatialData

    if not isinstance(sdata, SpatialData):
        raise TypeError("sdata must be a SpatialData object")
    where = _ObjectZarr()
    bound = where(sdata.path)
    if bound is None or not bound.is_dir():
        raise ValueError("Please write it to a zarr first, then open it here")
    if points_key is not None and (not isinstance(points_key, str) or points_key not in sdata.points):
        raise ValueError("points_key must name a points element in your SpatialData")
    if points_key is None:
        preferred = sdata.attrs.get("transcripts_dataframe")
        if isinstance(preferred, str) and preferred in sdata.points:
            points_key = preferred
        elif len(sdata.points) == 1:
            points_key = next(iter(sdata.points))
        elif len(sdata.points) > 1:
            raise ValueError("Choose the transcripts with points_key")
    if points_key is None and not sdata.images:
        raise ValueError("The sample needs transcripts or an image")
    if tool == "proseg" and points_key is None:
        raise ValueError("Proseg needs transcripts in a points element")
    config = _settings()
    python = _path(config.get("api", {}).get("engine_python") or sys.executable)
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise CompanionError("The configured engine Python could not be found")
    if Path(python).samefile(sys.executable):
        missing = [name for name in _ENGINE_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise CompanionError("The engine needs " + ", ".join(missing) + "; install the app into this environment as "
                                 "the README's Install section says, or set api.engine_python")
    found = locate.proseg_live((config.get("proseg") or {}).get("binary"), root=ROOT)
    if found.how == "missing":
        raise CompanionError("The companion's proseg-live program is missing (" + locate.fix_words(found.fix, found.path)
                             + ")")
    binary = Path(found.path)
    indices = {kind: config.get(kind, {}).get("gpu", 0) for kind in ("analysis", "stain", "qc")}
    if gpu is not None:
        if isinstance(gpu, Mapping):
            if set(gpu) - indices.keys():
                raise ValueError("gpu keys must be analysis, stain or qc")
            indices.update(gpu)
        else:
            indices = dict.fromkeys(indices, gpu)
    if any(type(index) is not int or index < 0 for index in indices.values()):
        raise ValueError("gpu indices must be non-negative whole numbers")
    if (isinstance(gpu, Mapping) and "qc" in gpu or mixing_check) and (device := _trainer_device(indices["qc"])) != "0":
        # The mixing check's learning process runs on GPU 0 only (engine/qc.py TRAINER_GPU): a qc GPU named, or the
        # mixing check asked to start, that cannot be GPU 0 here is refused now rather than failing in the app.
        seen = f"GPU {device}" if device else "no GPU"
        raise ValueError(f"The mixing check learns on GPU 0 only, and qc GPU {indices['qc']} is {seen} in this "
                         "notebook. Choose the qc GPU that is GPU 0, or leave the mixing check off.")
    # Where the app opens is this kernel's to say (the session's engine never reads it): a bad [desktop] browser or
    # chrome_flags is named now, before anything starts.
    desktop = config.get("desktop", {})
    check_flags(desktop.get("chrome_flags", []))
    browser.choice(config)
    patches = _patches_folder(patches_dir)
    memory_only, _ = sdata._symmetric_difference_with_zarr_store()
    memory_only = sorted(path.split("/", 1)[-1] for path in memory_only)
    snapshot = _snapshot(sdata, where)
    if memory_only:
        warnings.warn("Only saved elements can be shown for now. Not shown: " + ", ".join(memory_only),
                      CompanionWarning, stacklevel=2)
    if not snapshot["auto_save"]:
        warnings.warn("Automatic saving is off in sopa. Saves will reach this object only.", CompanionWarning, stacklevel=2)
    # InstanSeg is dormant: only its internal switch on (exactly true) makes its environment count. Off, a configured
    # [stain] instanseg_python is left out without a word, and the session's engine offers Cellpose alone.
    instanseg = config.get("stain", {}).get("instanseg_enabled") is True
    root = state_root()
    scratch = Path(scratch_dir or config.get("api", {}).get("scratch_dir") or tempfile.gettempdir()).expanduser().resolve()
    if removed := sweep(root) + sweep_scratch(scratch):
        warnings.warn(f"Removed {len(removed)} temporary app session(s) left after a crash.", CompanionWarning, stacklevel=2)
    path = str(bound)
    session = SessionFolders.create(scratch, path, root=root)
    remember_scratch(scratch)
    app = None
    lacking = {}          # what "auto" found missing in the kernel, by feature: session.toml's [locate]
    try:
        settings = build_session_config(
            id=session.id, token=secrets.token_urlsafe(32), data_dir=str(session.data), state_dir=str(session.state),
            python=sys.executable, kernel_pid=os.getpid(), tool=tool, open=mode,
            # The mixing check learns in [qc] python ("auto": the kernel's own environment when it has scvi-tools,
            # scviva-tools and torch; or another environment's Python, as [analysis] python may name RAPIDS).
            qc_python=_python("qc", config, lacking), mixing_check=mixing_check,
            qc_gpu=indices["qc"], model_cache=_path(model_cache or config.get("api", {}).get("model_cache")),
            # Stain segmentation, likewise ("auto": the kernel's own environment when it has Cellpose).
            stain_python=_python("stain", config, lacking),
            stain_gpu=indices["stain"], instanseg_enabled=instanseg,
            instanseg_python=_python("instanseg", config) if instanseg else locate.OFF,
            analysis_python=_python("analysis", config, lacking), analysis_gpu=indices["analysis"],
            patches_dir=patches, proseg_binary=str(binary),       # the proseg-live checked above is the one that runs
            locate=lacking)
        write_config(session.state / "session.toml", settings)
        binding = {"path": path, "view": "in_place", "name": Path(path).name.removesuffix(".zarr"),
                   "backed": True, "zarr": path, "memory_only": memory_only,
                   "points_key": points_key, "linked": [], "written": []}
        app = CompanionApp(sdata=sdata, folders=session, config=settings, binding=binding,
                           snapshot=lambda: _snapshot(sdata, where), schedule=_running_loop(), timeout=float(timeout),
                           settings=desktop)
        app._delivery = _attach_delivery(app, sdata)
        app._start(python)
    except BaseException:
        if app is not None and app._delivery is not None:
            app._delivery.close()
        session.remove()
        raise
    if app._schedule is None:
        app._run_until_closed()
    return app
