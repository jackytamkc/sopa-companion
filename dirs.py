"""Where the app keeps its per-user folders and finds your settings file, in one place.

A copy of the app is a source tree (a clone, the dev copy, this computer's production: pyproject.toml beside the
code, and its own config.toml) or installed (a wheel in a Python environment). On Linux every folder that exists today
keeps its exact path: an engine of the previous release and one of this release must meet in the same state folder
and save locks (moved, they would no longer keep each other out). An installed copy's data folder is new, and fixed in
the home folder as the state folder is (data_dir). Every folder on macOS and Windows follows platformdirs, imported
only there. The save locks are not here: names.ZARR_LOCK_FOLDERS and FIT_LOCKS.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

from . import names

APP = "sopa-companion"
ROOT = Path(__file__).resolve().parent
# An installed copy's runs and caches on Linux (data_dir).
DATA_DIR = "~/.local/share/" + APP


def source_tree(root: str | os.PathLike | None = None) -> bool:
    """Whether `root` (default: this copy) is a source tree: it holds pyproject.toml beside the code. A wheel never
    ships pyproject.toml inside the package."""
    return (Path(ROOT if root is None else root) / "pyproject.toml").is_file()


def xdg_layout() -> bool:
    """Linux and the other systems that are neither macOS nor Windows keep the XDG folders."""
    return sys.platform not in ("darwin", "win32", "cygwin")


def _xdg(variable: str, fallback: str) -> Path:
    value = os.environ.get(variable, "")
    return Path(value) if os.path.isabs(value) else Path.home() / fallback


def _platformdirs():
    import platformdirs                        # macOS and Windows only: production's venv on Linux has none
    return platformdirs


def user_config_file() -> Path:
    """Your settings file: $XDG_CONFIG_HOME/sopa-companion/config.toml, else ~/.config/sopa-companion/config.toml;
    elsewhere in platformdirs' user config folder."""
    if xdg_layout():
        return _xdg("XDG_CONFIG_HOME", ".config") / APP / "config.toml"
    return Path(_platformdirs().user_config_dir(APP, appauthor=False, roaming=False)) / "config.toml"


def state_dir() -> Path:
    """[paths] state_dir = "auto": ~/.local/state/sopa-companion (names.STATE_DIR, whatever $XDG_STATE_HOME says:
    desktop/migrate.py knows the default instance by it); elsewhere platformdirs' user state folder."""
    if xdg_layout():
        return names.home_path(names.STATE_DIR)
    return Path(_platformdirs().user_state_dir(APP, appauthor=False))


def data_dir(root: str | os.PathLike | None = None) -> Path:
    """[paths] data_dir = "auto": a source tree's own var/ (today's default); an installed copy's per-user data folder,
    ~/.local/share/sopa-companion (platformdirs' elsewhere), never inside the Python environment, which an upgrade or
    an uninstall would remove. Whatever $XDG_DATA_HOME says, as the state folder: the engine a state folder holds is
    told by its data folder (server/runtime_file.py), so the same copy started with the variable and without it (a
    terminal that sets it, a desktop session that does not) must come to the same pair of folders."""
    root = Path(ROOT if root is None else root)
    if source_tree(root):
        return root / "var"
    if xdg_layout():
        return names.home_path(DATA_DIR)
    return Path(_platformdirs().user_data_dir(APP, appauthor=False))


def user_cache() -> Path:
    """The user's own cache of the app's tools and notebook sessions: $XDG_CACHE_HOME/sopa-companion, else
    ~/.cache/sopa-companion; elsewhere platformdirs' user cache folder."""
    if xdg_layout():
        return _xdg("XDG_CACHE_HOME", ".cache") / APP
    return Path(_platformdirs().user_cache_dir(APP, appauthor=False))
