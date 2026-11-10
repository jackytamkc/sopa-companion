"""Effective configuration and confined writable paths.

The settings are read from these files, each merged over the one before (merge below). A source tree (a clone, this
computer's production: pyproject.toml beside the code; dirs.source_tree):
  config.toml          this computer's own settings; when there is none (a fresh clone), config.example.toml, the
                       defaults that ship
  config.local.toml    optional: only the settings that differ on this computer, kept apart so the file above can be
                       replaced by a newer copy without losing them
An installed copy (a wheel in a Python environment) reads the package's config.example.toml, then your own settings
file, ~/.config/sopa-companion/config.toml (dirs.user_config_file), which a source tree never reads, nor a notebook
session's engine (sources). Then, in either:
  --config FILE        the command line's (a second engine, the harness, a notebook session's session.toml)
  overrides            the caller's (tests)
"auto" finds things for you: [paths] data_dir and state_dir by the copy's own rule (dirs.py), [proseg] binary and each
helper's Python (locate.py); an empty value or none means the same. config.toml and config.local.toml are listed in
.gitignore (each computer keeps its own); config.example.toml lists every setting.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import tomllib

from .. import dirs, locate, portable

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "sopa_companion"            # the name the helpers import the app by (python -m sopa_companion.zarr_bridge.helper ...)
CONFIG, EXAMPLE, LOCAL = "config.toml", "config.example.toml", "config.local.toml"
# Environments and folders a setting may give from the home folder ("~/miniforge3/envs/..."): expanded here, so every
# engine gets an absolute path. The environments' "auto" and "off" are then resolved by locate.py.
HOME_PATHS = {"analysis": ("python",), "zarr": ("python", "browse_start"),
              "qc": ("python",), "stain": ("python", "instanseg_python", "patch_python")}
# proseg threads = "auto": half the cores this engine may use, at most 32, for a run on a region; three quarters for a
# whole-sample (tiled) run, which scales further. The rest keep the desktop and the app's window responsive. On a
# 64-core computer that gives 32 and 48.
AUTO_THREADS = {"threads": lambda cores: max(1, min(32, cores // 2)),
                "tiled_threads": lambda cores: max(1, cores * 3 // 4)}
# Proseg's own count ([analysis]; none of them is an option on screen): each run's spool cap, MB
# (engine/counts_spool.py: the engine's is twice it), the share of proseg's time its counting may take, and a byte
# budget per second (0: none; for a spool on a slow disk). engine/runs.py reads the first, engine/analysis.py the other
# two, each with its default.
COUNTS_SETTINGS = {"counts_spool_mb": 1024, "counts_extract_share": 0.10, "counts_mb_s": 0}


def merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def cores() -> int:
    """The cores this process may run on (its CPU affinity, which a container or taskset can narrow)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def helper_link(root=None, cache=None) -> Path:
    """Where helper_path puts its link for the app folder `root`: <cache>/<the folder's hash>/sopa_companion, the cache
    being ~/.cache/sopa-companion/pythonpath unless given."""
    root = Path(ROOT if root is None else root).resolve()
    folder = Path(cache or Path.home() / ".cache" / "sopa-companion" / "pythonpath") / \
        hashlib.sha1(str(root).encode()).hexdigest()[:12]
    return folder / PACKAGE


def helper_path(root=None, cache=None) -> str:
    """The folder a helper's PYTHONPATH starts with, so that `import sopa_companion` finds this copy of the app in the
    other environments (sopa, RAPIDS, InstanSeg), which never install it: a source tree's parent when its folder is
    called sopa_companion; else (a clone kept under its repository's name, or an installed copy, whose parent is its
    environment's site-packages) a folder holding one link, sopa_companion -> the app folder, made once under
    ~/.cache/sopa-companion/pythonpath/<the app folder's hash> (`cache` for the parent of those; helper_link). Should
    the link not be made, a source tree's parent all the same (the helpers then say they cannot find the app), and for
    an installed copy "": no entry, never its environment's site-packages, which would go ahead of another environment's
    own packages (engine/children.py says so in the engine's log, once)."""
    root = Path(ROOT if root is None else root).resolve()
    if root.name == PACKAGE and dirs.source_tree(root):
        return str(root.parent)
    link = helper_link(root, cache)
    folder = link.parent
    try:
        if not (portable.folder_link(link) and link.resolve() == root):
            folder.mkdir(parents=True, exist_ok=True)
            portable.link_folder(link, root)          # in one step (POSIX): another engine may be reading it
    except OSError:
        return str(root.parent) if dirs.source_tree(root) else ""
    return str(folder)


def helper_env_path(root=None) -> str:
    """PYTHONPATH for a helper: helper_path first, then whatever this process was given (inherited_path)."""
    root = Path(ROOT if root is None else root)
    return os.pathsep.join(p for p in (helper_path(root), inherited_path(root)) if p)


def inherited_path(root=None) -> str:
    """This process's own PYTHONPATH as a helper inherits it. An installed copy leaves out an entry that is its own
    site-packages (an earlier desktop entry put it there, or a user by hand): first on a helper's path it would go
    ahead of the standard library, and in another environment ahead of that environment's own packages. A source
    tree's is passed on as it is."""
    root = Path(ROOT if root is None else root)
    inherited = os.environ.get("PYTHONPATH", "")
    if inherited and not dirs.source_tree(root):
        site = root.resolve().parent
        inherited = os.pathsep.join(entry for entry in inherited.split(os.pathsep) if entry and not _is_folder(entry, site))
    return inherited


def _is_folder(entry: str, folder: Path) -> bool:
    try:
        return Path(entry).resolve() == folder
    except (OSError, RuntimeError, ValueError):
        return False


def api_session_file(path) -> bool:
    """Whether a --config file is a notebook session's session.toml ([session] mode = "api"); None or an empty path is
    not one."""
    if not path:
        return False
    try:
        with open(path, "rb") as stream:
            return (tomllib.load(stream).get("session") or {}).get("mode") == "api"
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def sources(path=None, root=None, *, user=None) -> list[Path]:
    """The settings files load_config merges, in order (the module docstring); `path` is the --config file. `user`:
    whether your own settings file is read when it is there; None: for an installed copy, unless `path` is a notebook
    session's session.toml (its engine takes what launch() read, never your file again; load_config passes False for a
    session's settings given as overrides)."""
    root = Path(ROOT if root is None else root)
    files = [root / CONFIG if (root / CONFIG).is_file() else root / EXAMPLE]
    if (root / LOCAL).is_file():
        files.append(root / LOCAL)
    if user is None:
        user = not dirs.source_tree(root) and not api_session_file(path)
    if user and (mine := dirs.user_config_file()).is_file():
        files.append(mine)
    if path and Path(path).resolve() not in [file.resolve() for file in files]:
        files.append(Path(path))
    return files


def _automatic(value) -> bool:
    return value is None or value == "" or value == "auto"


def data_dir_of(value, root) -> Path:
    """[paths] data_dir for the app folder `root`: "auto", "" or none, that copy's own rule (dirs.data_dir: a source
    tree's var/, an installed copy's per-user data folder); else "~" expanded, a relative one inside `root`; resolved."""
    if _automatic(value):
        return dirs.data_dir(root).resolve()
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def state_dir_of(value) -> Path:
    """[paths] state_dir: "auto", "" or none, ~/.local/state/sopa-companion (dirs.state_dir); else "~" expanded, a
    relative one from the working folder; resolved."""
    if _automatic(value):
        return dirs.state_dir().resolve()
    return Path(str(value)).expanduser().resolve()


def _resolved(report, value) -> bool:
    """Whether `value` is what an earlier load_config's report for it resolved to."""
    return isinstance(report, dict) and isinstance(value, str) and value == report.get("path")


def _lacking(report, value) -> bool:
    """Whether `value` is the "off" a notebook's launch() writes into session.toml for an "auto" that found the
    feature's modules missing in the kernel, with what it found beside it ([locate.<feature>]: no path, how "auto")."""
    return isinstance(report, dict) and value == locate.OFF and report.get("how") == locate.AUTO \
        and not report.get("path") and bool(report.get("missing"))


def load_config(path=None, overrides=None, *, root=None) -> dict:
    """The effective settings: `root` (tests) stands for the app's folder, ROOT, which relative paths are taken from.
    Overrides that are a notebook session's settings ([session] mode = "api": create_app loads the settings again over
    the session engine's own) never bring your settings file in, as its session.toml does not (sources)."""
    root = Path(ROOT if root is None else root)
    session = ((overrides or {}).get("session") or {}).get("mode") == "api"
    files = sources(path, root, user=False if session else None)
    if not files[0].is_file():
        raise FileNotFoundError(f"No settings in {root}: neither {CONFIG} nor {EXAMPLE} is there")
    config: dict = {}
    for file in files:
        config = merge(config, tomllib.loads(file.read_text(encoding="utf-8")))
    config = merge(config, overrides or {})
    data_dir = data_dir_of(config["paths"].get("data_dir"), root)
    config["paths"]["data_dir"] = str(data_dir)
    config["paths"]["cache_dir"] = str(data_dir / "cache")
    config["paths"]["state_dir"] = str(state_dir_of(config["paths"].get("state_dir")))
    config["paths"]["sample_roots"] = [os.path.expanduser(r) if isinstance(r, str) else r
                                       for r in config["paths"].get("sample_roots", [])]
    for section, keys in HOME_PATHS.items():
        for key in keys:
            if isinstance(value := (config.get(section) or {}).get(key), str):
                config[section][key] = os.path.expanduser(value)
    # What each helper runs in and the proseg-live program, resolved once here (locate.py), with what was found for
    # the logs (config["locate"]). Settings this function resolved before come back in as overrides (create_app loads
    # the settings again over the command's): a value that is the very answer of that report keeps it, so "off" never
    # turns into "auto" and a missing program's fix is not taken for a setting's. A notebook session's session.toml
    # carries what launch() found missing in the kernel for the features it wrote "off" for ([locate]): that report is
    # kept, and the feature reads as "auto" leaves it (""), so the engine says what the kernel lacks without looking.
    api = (config.get("session") or {}).get("mode") == "api"
    earlier = (overrides or {}).get("locate")
    if not isinstance(earlier, dict) and api:
        earlier = config.get("locate")
    earlier = earlier if isinstance(earlier, dict) else {}
    located = {}
    for feature, (section, key, _) in locate.FEATURES.items():
        table = config.get(section) or {}
        if key in table and (_resolved(earlier.get(feature), table.get(key))
                             or api and _lacking(earlier.get(feature), table.get(key))):
            located[feature] = copy.deepcopy(earlier[feature])
            config[section][key] = located[feature]["path"]
            continue
        found = locate.python_for(feature, table.get(key))
        located[feature] = found.report()
        if key in table or found.path:      # a missing key resolved to off stays missing, as before
            config.setdefault(section, {})[key] = found.path
    binary = config["proseg"].get("binary")
    if _resolved(earlier.get("proseg"), binary) and binary:
        located["proseg"] = copy.deepcopy(earlier["proseg"])
    else:
        found = locate.proseg_live(binary, root=root)
        config["proseg"]["binary"] = found.path      # never "": a missing program's is where its fix puts it
        located["proseg"] = found.report()
    config["locate"] = located
    for key, rule in AUTO_THREADS.items():
        if config["proseg"].get(key) == "auto":
            config["proseg"][key] = rule(cores())
    if config["server"]["host"] not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("SoPA companion requires a loopback bind address")
    if not config["server"]["token"]:
        raise ValueError("The session token cannot be empty")
    if type(config["proseg"]["threads"]) is not int or config["proseg"]["threads"] < 1:
        raise ValueError('proseg.threads must be a positive integer or "auto"')
    check_counts(config.get("analysis") or {})
    return config


def check_counts(analysis: dict) -> None:
    """[analysis]'s settings for proseg's own count (COUNTS_SETTINGS), when given: ValueError for one out of range. The
    settings are left as they are (each reader has its default)."""
    def number(key):
        value = analysis.get(key, COUNTS_SETTINGS[key])
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return value if ok else None

    spool = analysis.get("counts_spool_mb", COUNTS_SETTINGS["counts_spool_mb"])
    if type(spool) is not int or spool < 64:
        raise ValueError("analysis.counts_spool_mb must be a whole number of at least 64 (MB)")
    share = number("counts_extract_share")
    if share is None or not 0 < share <= 1:
        raise ValueError("analysis.counts_extract_share must be a number above 0 and at most 1")
    rate = number("counts_mb_s")
    if rate is None or rate < 0:
        raise ValueError("analysis.counts_mb_s must be a number of at least 0 (0: no limit)")
