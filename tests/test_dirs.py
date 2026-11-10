"""Where the app keeps its per-user folders (dirs.py): a copy holding pyproject.toml is a source tree, anything else an
installed copy; on Linux every folder keeps today's path (the state folder whatever $XDG_STATE_HOME says, so engines of
two releases still meet there), an installed copy's runs go to the user's data folder, never into its environment, and
that folder is fixed in the home folder as the state folder is; on macOS and Windows platformdirs gives them, and Linux
never imports it."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from sopa_companion import dirs

from conftest import api_python

XDG = ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")


def copies(tmp_path) -> tuple[Path, Path]:
    """A source tree (a clone: pyproject.toml beside the code) and an installed copy (a wheel's package folder)."""
    tree, installed = tmp_path / "clone", tmp_path / "env" / "site-packages" / "sopa_companion"
    for folder in (tree, installed):
        folder.mkdir(parents=True)
        (folder / "cli.py").write_text("")
    (tree / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    return tree, installed


def test_a_source_tree_is_told_by_its_pyproject(tmp_path):
    tree, installed = copies(tmp_path)
    assert dirs.source_tree(tree) and dirs.source_tree(str(tree)) and not dirs.source_tree(installed)
    assert dirs.source_tree() == (dirs.ROOT / "pyproject.toml").is_file() and dirs.ROOT == Path(dirs.__file__).parent
    (installed / "pyproject.toml").mkdir()                                  # a folder of that name is not the file
    assert not dirs.source_tree(installed)


def test_linux_keeps_todays_folders_whatever_xdg_says(tmp_path, monkeypatch):
    monkeypatch.setattr(dirs, "sys", types.SimpleNamespace(platform="linux"))
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    tree, installed = copies(tmp_path)
    for name in XDG:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    # The state folder never moves: desktop/migrate.py knows the default instance by this very path.
    assert dirs.state_dir() == home / ".local/state/sopa-companion"
    assert dirs.user_config_file() == tmp_path / "xdg_config_home" / "sopa-companion" / "config.toml"
    assert dirs.user_cache() == tmp_path / "xdg_cache_home" / "sopa-companion"
    assert dirs.data_dir(tree) == tree / "var"
    # Nor does an installed copy's data folder: the same copy started with $XDG_DATA_HOME and without it (a terminal
    # that sets it, a desktop session that does not) keeps one data folder for its one state folder, where the engine
    # it holds is told by its data folder (two would count the copy's own engine as another copy's).
    assert dirs.data_dir(installed) == home / ".local" / "share" / "sopa-companion"
    # A relative value is no XDG folder (the XDG rule says to ignore it), and an unset one: the home folder's.
    for relative in (True, False):
        for name in XDG:
            if relative:
                monkeypatch.setenv(name, "relative/folder")
            else:
                monkeypatch.delenv(name)
        assert dirs.user_config_file() == home / ".config" / "sopa-companion" / "config.toml"
        assert dirs.user_cache() == home / ".cache" / "sopa-companion"
        assert dirs.data_dir(installed) == home / ".local" / "share" / "sopa-companion"
        assert dirs.data_dir(tree) == tree / "var" and dirs.state_dir() == home / ".local/state/sopa-companion"
    assert dirs.xdg_layout()


def test_on_linux_dirs_imports_no_platformdirs(tmp_path):
    script = ("import sys\n"
              "from sopa_companion import dirs\n"
              "dirs.source_tree(); dirs.user_config_file(); dirs.state_dir(); dirs.data_dir(); dirs.user_cache()\n"
              "print('platformdirs' in sys.modules)\n")
    environment = {**os.environ, "PYTHONPATH": str(dirs.ROOT.parent), "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", script], env=environment, cwd=tmp_path, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "False"


# On macOS: dirs.py with platformdirs' own macOS rules, the folders printed. Run in a Python that has platformdirs.
MACOS = """
import json, sys, types
from platformdirs.macos import MacOS
from sopa_companion import dirs
dirs.sys = types.SimpleNamespace(platform="darwin")
def folder(kind):
    return lambda appname=None, appauthor=None, **kwargs: getattr(MacOS(appname, appauthor, **kwargs), kind)
kinds = ("user_config_dir", "user_state_dir", "user_data_dir", "user_cache_dir")
dirs._platformdirs = lambda: types.SimpleNamespace(**{kind: folder(kind) for kind in kinds})
tree, installed = sys.argv[1:]
print(json.dumps({"xdg": dirs.xdg_layout(), "data": str(dirs.data_dir(installed)), "state": str(dirs.state_dir()),
                  "config": str(dirs.user_config_file()), "cache": str(dirs.user_cache()),
                  "tree": str(dirs.data_dir(tree))}))
"""


def test_macos_takes_platformdirs(tmp_path):
    """On macOS the folders are platformdirs' (imported only there), checked against its own macOS rules: in this Python
    when it has platformdirs, else in the API environment's, which the app's requirements give it (a source folder's
    engine environment on Linux never needs it)."""
    python = sys.executable if importlib.util.find_spec("platformdirs") else api_python()
    tree, installed = copies(tmp_path)
    environment = {key: value for key, value in os.environ.items() if key not in XDG}   # newer platformdirs read them
    environment.update(PYTHONPATH=str(dirs.ROOT.parent), HOME=str(tmp_path / "home"))
    done = subprocess.run([python, "-c", MACOS, str(tree), str(installed)], env=environment, cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    found = {key: value if key == "xdg" else Path(value) for key, value in json.loads(done.stdout).items()}
    support = tmp_path / "home" / "Library" / "Application Support" / "sopa-companion"
    assert found["xdg"] is False
    assert found["data"] == support and found["state"] == support and found["config"] == support / "config.toml"
    assert found["cache"] == tmp_path / "home" / "Library" / "Caches" / "sopa-companion"
    assert found["tree"] == tree / "var"                                 # a source tree keeps its own var/ everywhere
