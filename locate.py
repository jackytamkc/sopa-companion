"""What the app runs, found: the Python each helper feature runs in, and the proseg-live program.

An interpreter setting ([zarr] python, [stain] python, [qc] python, [analysis] python; [stain] instanseg_python) is
"auto", the default (an empty value or none means the same): this Python, when it has every module the feature needs
(importlib.util.find_spec: nothing is imported, so neither the engine nor a notebook loads torch or RAPIDS to look),
else the feature is off; "off": the feature is off; anything else: another environment's Python, "~" expanded, kept as
given even when it is not there, so the Log can name it. The desktop engine resolves its settings once (engine/config.py
load_config) with its own interpreter; a notebook's launch() (api/launch.py) with the kernel's, and writes the answers
into session.toml as a full path or "off", never "auto". A notebook imports this module before anything of the engine
(the import fence), so it imports the standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import sysconfig

from . import dirs, portable
from .analysis.backend import CPU_MODULES

AUTO, OFF = "auto", "off"
PROGRAM = "proseg-live"
# What the program's --version says: the proseg it is made from. The app pins proseg-live>=3.2.0.1,==3.2.0.*
# (pyproject.toml), and the patch on the same proseg (3.2.0.1) still says this: whether a program keeps proseg's own
# count is asked of its --help (engine/runs.py counts_capable), never of this line.
PROGRAM_VERSION = "proseg 3.2.0"
# What the engine imports itself: a Python without one of these cannot run the engine (cli.py, api/launch.py).
ENGINE_MODULES = ("fastapi", "uvicorn", "httpx", "psutil", "numpy", "pyarrow", "tifffile", "imagecodecs", "zarr",
                  "PIL", "zstandard", "orjson")
# feature: (settings table, key, the modules its helper imports). The analysis worker ("analysis") needs what its CPU
# path imports (analysis/backend.py, standard library only): what `pip install sopa` brings; RAPIDS on a GPU is its
# faster path where it is installed, and an environment with RAPIDS alone is named by its path.
FEATURES = {
    "zarr": ("zarr", "python", ("sopa", "spatialdata")),
    "stain": ("stain", "python", ("cellpose",)),
    "instanseg": ("stain", "instanseg_python", ("instanseg",)),
    "qc": ("qc", "python", ("torch", "scvi", "scviva")),
    "analysis": ("analysis", "python", CPU_MODULES),
}
PROBE = ("import importlib.util as u, sys\n"
         "s = u.find_spec('sopa_companion')\n"
         "sys.stdout.write(s.origin if s is not None and s.origin else '')\n")
# How the program was found, in the words of the engine's log and the Log.
WHERE = {"source": "this source folder's build", "environment": "this Python environment",
         "user": "your user scripts folder", "path": "found on PATH", "setting": "set in the settings"}
# What makes a missing program, in the same words everywhere (the engine's log, the Log, --version, a notebook's
# error): never an index command (the two packages come from the release folder alone, README Install).
FIX_WORDS = {"build": "native/build.sh builds it", "install": "install it as the README's Install section says",
             "setting": "{path} is not there"}
# What makes a newer program in place of one found (an older proseg-live, which cannot keep proseg's own count): a
# source folder's build is built again, one installed (in this environment, the user's scripts folder or on PATH) is
# installed again as the README says, and one a setting names is replaced by pointing the setting at a newer one.
UPDATE_FIX = {"source": "build", "environment": "install", "user": "install", "path": "install", "setting": "setting"}
# What the engine's log calls each helper feature, in its order (the dormant InstanSeg is not named).
LOG_NAMES = (("zarr", "samples and saving"), ("stain", "Cellpose"), ("qc", "the mixing check's learning"),
             ("analysis", "Clusters & UMAP"))


@dataclass(frozen=True)
class Found:
    path: str                       # an interpreter: its path, "" when off or not found; the program: its path, and
                                    # when it is missing the place the fix puts it, never ""
    how: str                        # interpreters: "auto" | "setting" | "off";
                                    # the program: "source" | "environment" | "user" | "path" | "setting" | "missing"
    missing: tuple[str, ...] = ()   # "auto": the feature's modules this Python lacks
    problem: str = ""               # "missing": a path that is not a runnable file
    fix: str = ""                   # the program, when not usable: "build" | "install" | "setting"
    passed: tuple[str, ...] = ()    # the program: "<path>: <why>" for each one found outside this Python's own folders
                                    # that is not proseg-live as the app runs it (outside)

    def report(self) -> dict:
        return {"path": self.path, "how": self.how, "missing": list(self.missing), "problem": self.problem,
                "fix": self.fix, "passed": list(self.passed)}


def program(name: str) -> str:
    """A program's file name on this system ("proseg-live", "proseg-live.exe" on Windows)."""
    return name + ".exe" if os.name == "nt" else name


def runnable(path) -> bool:
    """A file this user may run (POSIX: os.X_OK; Windows: a file). False on OSError."""
    try:
        return os.path.isfile(path) and (os.name == "nt" or os.access(path, os.X_OK))
    except (OSError, TypeError, ValueError):
        return False


def _automatic(value) -> bool:
    return value is None or value == "" or value == AUTO


def _found(find, name: str) -> bool:
    try:
        return find(name) is not None
    except (ImportError, ValueError):
        return False


def python_for(feature: str, value=None, *, find=importlib.util.find_spec) -> Found:
    """The interpreter `feature` (FEATURES) runs in, from its setting `value` (module docstring). A module whose spec
    lookup raises (ImportError, ValueError) counts as missing."""
    section, key, modules = FEATURES[feature]
    if _automatic(value):
        missing = tuple(name for name in modules if not _found(find, name))
        return Found("", AUTO, missing) if missing else Found(os.path.abspath(sys.executable), AUTO)
    if value == OFF:
        return Found("", OFF)
    if not isinstance(value, str):
        raise ValueError(f'{section}.{key} must be "auto", "off" or the path of a Python')
    path = os.path.expanduser(value)
    return Found(path, "setting", problem="" if runnable(path) else "missing")


def _user_scripts() -> Path | None:
    """The user scheme's scripts folder (pip install --user: ~/.local/bin), when this Python reads the user scheme's
    packages (site.ENABLE_USER_SITE: not in a virtual environment that keeps the system's packages out, the usual
    kind, where pip cannot install there for it). None otherwise, or for a scheme this Python lacks."""
    if not site.ENABLE_USER_SITE:
        return None
    try:
        return Path(sysconfig.get_path("scripts", sysconfig.get_preferred_scheme("user")))
    except KeyError:
        return None


def outside(path) -> str:
    """Why a proseg-live found outside this Python's own folders (the user scheme's scripts folder, PATH) is not taken;
    "" when it is: a program (never a script: on some computers another app's launcher has that name, and it is not
    run) whose --version says PROGRAM_VERSION. A stale build or another app's command is passed over, and the engine
    keeps the place the fix puts the program, as when none is found."""
    try:
        with open(path, "rb") as stream:
            if stream.read(2) == b"#!":
                return "a script, not the program"
    except OSError as exc:
        return f"not readable ({exc.strerror or exc})"
    try:
        done = subprocess.run([str(path), "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=10, check=False, **portable.no_window())
    except (OSError, subprocess.SubprocessError):
        return "it did not answer --version"
    said = done.stdout.decode("utf-8", "replace").strip()
    return "" if done.returncode == 0 and said == PROGRAM_VERSION else \
        f"its --version says {said[:80]!r}, not {PROGRAM_VERSION!r}"


def _same_place(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def proseg_live(value=None, *, root=None) -> Found:
    """The proseg-live program from [proseg] binary `value`: a path, relative to `root` (default the app's
    folder) unless absolute; or "auto"/""/None: <root>/native/bin in a source tree, the environment's scripts folder,
    then outside this Python's own folders, only a program that is proseg-live as the app runs it (outside): the user
    scheme's scripts folder (_user_scripts), then PATH. fix: "build" for a source tree (auto, or a path to its
    native/bin/proseg-live), "install" for an installed copy, "setting" for any other path that is not there. Missing,
    `path` is where that fix puts the program (<root>/native/bin/proseg-live, <this Python's scripts folder>/proseg-live,
    or the setting's path), so the caller's later is_file() sees a build or an install without a restart."""
    root = Path(dirs.ROOT if root is None else root)
    tree = dirs.source_tree(root)
    built = root / "native" / "bin" / program(PROGRAM)
    if _automatic(value):
        own = Path(sysconfig.get_path("scripts")) / program(PROGRAM)
        for how, path in ((("source", built),) if tree else ()) + (("environment", own),):
            if runnable(path):
                return Found(str(path), how)
        user = _user_scripts()
        on_path = shutil.which(PROGRAM)
        seen, passed = [], []
        for how, path in (("user", user / program(PROGRAM) if user else None),
                          ("path", Path(os.path.abspath(on_path)) if on_path else None)):
            if path is None or not runnable(path) or any(_same_place(path, other) for other in seen):
                continue                        # ~/.local/bin is often on PATH too: one look at each program
            seen.append(path)
            if not (why := outside(path)):
                return Found(str(path), how, passed=tuple(passed))
            passed.append(f"{path}: {why}")
        if tree:
            return Found(str(built), "missing", fix="build", passed=tuple(passed))
        return Found(str(own), "missing", fix="install", passed=tuple(passed))
    if not isinstance(value, str):
        raise ValueError('proseg.binary must be "auto" or the path of proseg-live')
    path = Path(os.path.expanduser(value))
    path = path if path.is_absolute() else root / path
    if runnable(path):
        return Found(str(path), "setting")
    return Found(str(path), "missing", fix="build" if tree and _same_place(path, built) else "setting")


def fix_words(fix: str, path: str) -> str:
    """What to do about a missing program, in FIX_WORDS' words ("build" when the fix is not known)."""
    return FIX_WORDS.get(fix or "build", FIX_WORDS["build"]).format(path=path)


def program_fix(report: dict) -> str:
    """What makes the program the app needs, from proseg_live()'s report: a missing one's fix, else what updates the one
    found (UPDATE_FIX): "build", "install" or "setting" (the page's words for an older proseg-live, /health
    proseg_fix)."""
    report = report if isinstance(report, dict) else {}
    if report.get("how") == "missing":
        return report.get("fix") or "build"
    return UPDATE_FIX.get(report.get("how"), "build")


def imports_this_copy(python: str, env, *, cwd, timeout: float = 20.0) -> bool:
    """Whether `python`, started with `env` minus PYTHONPATH in `cwd` (a folder of the app's own, with no Python
    files), imports this very copy of the app: PROBE's answer and dirs.ROOT/__init__.py are the same file
    (os.path.samefile). False when it imports another copy or none, exits non-zero, or takes longer than `timeout`."""
    environment = {key: value for key, value in dict(env).items() if key != "PYTHONPATH"}
    try:
        done = subprocess.run([str(python), "-c", PROBE], env=environment, cwd=cwd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, check=False,
                              **portable.no_window())
        lines = done.stdout.decode("utf-8", "replace").strip().splitlines() if done.returncode == 0 else []
        return bool(lines) and os.path.samefile(lines[-1].strip(), dirs.ROOT / "__init__.py")
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _interpreter_words(report: dict) -> str:
    how, path = report.get("how"), report.get("path") or ""
    if how == AUTO:
        return "this Python" if path else "off (this Python has no " + ", ".join(report.get("missing") or ()) + ")"
    if how == "setting":
        return f"{path} (not there)" if report.get("problem") else path
    return "off"


def program_words(report: dict) -> str:
    """The proseg-live part of the engine's log line: where the program is and how it was found, or what to do; then
    any program of that name that was passed over, and why."""
    path = report.get("path") or ""
    if report.get("how") == "missing":
        words = f"proseg-live: missing ({fix_words(report.get('fix') or '', path)})"
    else:
        words = f"proseg-live: {path} ({WHERE.get(report.get('how'), report.get('how'))})"
    if passed := report.get("passed"):
        words += "; passed over: " + ", ".join(passed)
    return words


def summary(located: dict) -> str:
    """The engine log's line from load_config's config["locate"]: where each helper runs, then proseg-live."""
    parts = [f"{name}: {_interpreter_words(located.get(feature) or {})}" for feature, name in LOG_NAMES]
    return "; ".join([*parts, program_words(located.get("proseg") or {})])
