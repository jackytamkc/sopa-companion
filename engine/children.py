"""How the engine starts its helper processes: the GPU a helper sees, its
environment, the folder it starts in and its launcher. Every place that starts a helper takes these from here.

GPUs. A setting's GPU index ([analysis] gpu, [stain] gpu, [qc] gpu) counts within the engine's own
CUDA_VISIBLE_DEVICES: index i is that list's i-th entry (a GPU number or a GPU's UUID), so a notebook that hides or
picks GPUs gives the app it starts the same view. With the variable unset, the index is the GPU's own number, as
nvidia-smi numbers them (PCI bus order: every helper gets CUDA_DEVICE_ORDER=PCI_BUS_ID), as before. An index past the
list's end, or any index of an empty list (no GPU visible to the engine), gives "": the helper then sees no GPU at all
(said once in the engine's log). gpu() is also what `nvidia-smi --id=` takes (engine/qc.py gpu_probe).

Environments. A helper gets this process's environment without the desktop's variables (a helper never draws on the
user's screen; the Cellpose GUI's launcher adds the desktop back) or the API engine's events pipe, and the app on its
PYTHONPATH only where its interpreter needs it:
- desktop, a source tree (a clone, this computer's production): config.helper_env_path(), the app's own entry first,
  then whatever the engine was given; the stain helpers (inherit=False) the app's entry alone, config.helper_path(), as
  they have always had;
- desktop, an installed copy: nothing in its own environment ("same": the package is installed there; the engine's own
  PYTHONPATH with inherit, never its site-packages, none without), and a link folder of the user's cache in another
  one (config.helper_path(), never the environment's site-packages; should the link not be made, no entry, said once);
- API mode (configure() with [session] mode = "api"): a helper in the engine's own environment ("same") keeps the
  engine's PYTHONPATH; a helper in another environment gets PYTHONPATH = <state>/pylib, a folder holding one link to the
  package, made when first needed: should it not be made, no PYTHONPATH (said once; the helper then cannot import the
  app, and fails saying so). Both get MPLBACKEND=Agg: ipykernel exports its inline backend.
Helpers start in a folder that holds no Python file (app_path): a source tree's entry as before, else the engine's state
folder, never site-packages, which `python -m` would put ahead of the standard library.
The session token is never in a helper's command line or environment (it is not in os.environ).
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Literal

from .. import dirs, portable
from .config import PACKAGE, ROOT, helper_env_path, helper_link, helper_path, inherited_path

Kind = Literal["same", "other"]
# Helper commands that only read the sample, or write only into the engine's own folders (prepare's transcripts, a
# stain run's tiles, a patch export): started under setpriv --pdeathsig TERM, so they end when the engine dies. Never
# the helper's `save`, nor a command not named here: a journalled write into a sample finishes or rolls back by itself
# when the engine dies.
READ_ONLY = frozenset({"inspect", "shapes", "prepare", "count", "tissue", "segmentations", "stain", "export"})
# The API engine's own events pipe (api/protocol.py ENV_EVENTS_FD, not imported: the desktop engine never loads api/):
# a helper never sees it.
EVENTS_FD = "SOPA_COMPANION_EVENTS_FD"

_lock = threading.Lock()
_api_state: Path | None = None          # API mode: the session's state folder (its pylib); None: the desktop
_state: Path | None = None              # the engine's state folder (either mode): an installed copy's helpers start there
_said: set = set()                      # what the engine's log was told already (_say_once)


def configure(config: dict) -> None:
    """Take the mode from an engine's effective configuration: API mode when [session] mode is "api" (its [paths]
    state_dir holds pylib), else the desktop; and, either way, the engine's state folder (app_path). engine/analysis.py
    AnalysisManager and engine/qc.py QcModels call it as an engine is made (server/app.py create_app makes both before
    anything starts a helper): the engine made last in a process sets it."""
    global _api_state, _state
    session = config.get("session") or {}
    state = (config.get("paths") or {}).get("state_dir")
    with _lock:
        _api_state = Path(state).expanduser() if session.get("mode") == "api" and state else None
        _state = Path(state).expanduser() if state else None


def gpu(index: int) -> str:
    """CUDA_VISIBLE_DEVICES for a helper on the engine's GPU `index` (module docstring); "" when the engine sees no
    such GPU."""
    return _device(index)


def _device(index: int) -> str:
    if type(index) is not int or index < 0:
        raise ValueError("A GPU index is a whole number from 0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    entries = []
    for entry in visible.split(","):
        entry = entry.strip()
        if not entry or entry.startswith("-"):
            break                       # CUDA reads the list up to its first invalid entry ("-1": no GPU at all)
        entries.append(entry)
    if index < len(entries):
        return entries[index]
    _say_once(("gpu", index, visible), f"GPU {index} is not one of the GPUs this engine sees (CUDA_VISIBLE_DEVICES="
              f"{visible!r}): a helper set to use it sees no GPU")
    return ""


def kind(python) -> Kind:
    """"same" when the interpreter `python` belongs to the engine's own environment (its bin folder's parent is
    sys.prefix; on Windows also a conda environment's python.exe, at the environment's top), else "other"."""
    if not python:
        return "other"
    path = Path(python) if os.sep in str(python) else Path(shutil.which(str(python)) or python)
    try:
        prefix, folder = Path(sys.prefix).resolve(), path.absolute().parent
        if portable.WINDOWS and folder.resolve() == prefix:
            return "same"
        return "same" if folder.parent.resolve() == prefix else "other"
    except OSError:
        return "other"


def env(kind: Kind, *, gpu: int | None = None, inherit: bool = True) -> dict[str, str]:
    """A helper's environment (module docstring) for a helper in the engine's environment ("same") or another one
    ("other"). `gpu`: the engine's GPU index the helper sees as its only one, numbered as nvidia-smi numbers GPUs;
    None leaves the GPUs as the engine sees them. `inherit` (the desktop's only): the engine's own PYTHONPATH after the
    app's entry; False, the app's entry alone (the stain helpers', as before)."""
    if kind not in ("same", "other"):
        raise ValueError('kind is "same" or "other"')
    result = dict(os.environ)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", EVENTS_FD):
        result.pop(name, None)
    with _lock:
        state = _api_state
    if state is None and kind == "same" and not dirs.source_tree(ROOT):
        # The package is installed in this environment: nothing to add, and never its site-packages from the engine's
        # own PYTHONPATH (inherited_path).
        value = inherited_path() if inherit else ""
        if value:
            result["PYTHONPATH"] = value
        else:
            result.pop("PYTHONPATH", None)
    elif state is None:
        entry = helper_path()
        if not entry:                           # an installed copy whose link could not be made: never site-packages
            _say_once(("helper_path",), f"{helper_link()} could not be made: helpers in other environments cannot "
                      "import the app")
        value = helper_env_path() if inherit else entry
        if value:
            result["PYTHONPATH"] = value
        else:
            result.pop("PYTHONPATH", None)
    else:
        result["MPLBACKEND"] = "Agg"
        if kind == "other":
            pylib = _pylib(state)
            if pylib is None:                         # never the engine's path: the helper cannot import the app
                result.pop("PYTHONPATH", None)
            else:
                result["PYTHONPATH"] = pylib
    if gpu is not None:
        result.update(CUDA_VISIBLE_DEVICES=_device(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return portable.child_environment(result)       # Windows: Python's UTF-8 mode, as the engine's own


def app_entry(kind: Kind) -> str | None:
    """What a helper's PYTHONPATH starts with (env), for a command shown to the user: in API mode the engine's own
    PYTHONPATH for a helper in its environment ("same"), the session's pylib for another; on the desktop nothing for an
    installed copy's own environment, else the app's entry (helper_path). None when there is nothing to name."""
    with _lock:
        state = _api_state
    if state is not None:
        return (os.environ.get("PYTHONPATH") or None) if kind == "same" else _pylib(state)
    if kind == "same" and not dirs.source_tree(ROOT):
        return None
    return helper_path() or None


def app_path() -> str:
    """The folder a helper starts in (`python -m` puts it first on the helper's sys.path, ahead of the standard
    library): one that holds no Python file but the app. A source tree's own entry on the desktop (helper_path: the app
    folder's parent, as before, or a clone's link folder, so a clone kept under another name imports itself); in API
    mode the session's state folder; an installed copy's desktop engine its state folder (the user's cache before an
    engine configured this module). Never an environment's site-packages."""
    with _lock:
        api, state = _api_state, _state
    if api is not None:
        return str(api)
    if dirs.source_tree(ROOT):
        return helper_path()
    if state is not None:
        return str(state)
    folder = dirs.user_cache()
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        pass                                    # the helper's start then says the folder is missing
    return str(folder)


def helper_prefix(command: str) -> list[str]:
    """The launcher a helper command starts under: setpriv --pdeathsig TERM for a READ_ONLY one (it ends when the
    engine dies); nothing for any other (the helper's `save`), nor where setpriv is missing. The signal follows the
    THREAD that started the helper: only for helpers that thread waits for (engine/stain_worker.py's does not)."""
    return ["setpriv", "--pdeathsig", "TERM"] if command in READ_ONLY and shutil.which("setpriv") else []


def _pylib(state: Path) -> str | None:
    """<state>/pylib, one link sopa_companion -> this package, made when a helper first needs it (or when it points
    elsewhere); None, said once, should it not be made: never the desktop's helper_path(), which for an installed
    package is the engine environment's site-packages."""
    folder = state / "pylib"
    link = folder / PACKAGE
    try:
        if not (portable.folder_link(link) and link.resolve() == ROOT):
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            portable.link_folder(link, ROOT)          # in one step (POSIX): a helper may be starting from it
    except OSError as exc:
        _say_once(("pylib", str(folder)), f"{link} could not be made ({exc}): helpers in other environments cannot "
                  "import the app")
        return None
    return str(folder)


def _say_once(key, message: str) -> None:
    """One line in the engine's log, the first time `key` comes up in this process."""
    with _lock:
        if key in _said:
            return
        _said.add(key)
    print(f"SoPA companion: {message}", flush=True)
