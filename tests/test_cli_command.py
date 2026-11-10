"""The command (cli.py): `sopa-companion` alone is `launch`; `--version` names the app and the proseg-live it would run;
no command ever starts another interpreter instead of this one: a Python without the engine's packages is told how to
fix it, by the copy's layout; and the usage names the command the user typed."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

from sopa_companion import cli, dirs


@pytest.fixture
def commands(monkeypatch):
    """What main() hands on: the settings it loads (by their --config) and the command it runs, with nothing started."""
    seen = []
    monkeypatch.setattr(cli, "engine_missing", lambda command: None)
    monkeypatch.setattr("sopa_companion.engine.config.load_config",
                        lambda path=None, *args, **kwargs: seen.append(("settings", path)) or {"server": {}})
    monkeypatch.setattr("sopa_companion.desktop.launcher.command",
                        lambda config, args: seen.append((args.command, args.config)) or 0)
    return seen


def test_no_command_is_launch(commands):
    assert cli.main([]) == 0
    assert commands == [("settings", None), ("launch", None)]
    commands.clear()
    assert cli.main(["--config", "/x/engine.toml"]) == 0
    assert commands == [("settings", "/x/engine.toml"), ("launch", "/x/engine.toml")]
    commands.clear()
    assert cli.main(["status"]) == 0 and commands[-1] == ("status", None)


def test_version_prints_the_app_and_its_proseg(tmp_path, monkeypatch, capsys):
    from sopa_companion import __version__, locate
    program = tmp_path / "bin" / "proseg-live"
    found = {"path": str(program), "how": "environment", "fix": ""}
    loaded = []
    monkeypatch.setattr("sopa_companion.engine.config.load_config",
                        lambda path=None, *args, **kwargs: loaded.append(path) or {"locate": {"proseg": dict(found)}})
    program.parent.mkdir()
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"sopa-companion {__version__}\nproseg-live: {program}\n"
    assert cli.main(["--config", "/x/engine.toml", "--version"]) == 0 and loaded[-1] == "/x/engine.toml"
    capsys.readouterr()
    # Not there now: what makes it, in the engine log's words; never an index command.
    for fix, words in (("build", "native/build.sh builds it"),
                       ("install", "install it as the README's Install section says"),
                       ("setting", f"{tmp_path / 'gone'} is not there")):
        found.update(path=str(tmp_path / "gone"), how="missing", fix=fix)
        assert cli.main(["--version"]) == 0
        assert capsys.readouterr().out.splitlines()[1] == f"proseg-live: missing ({words})"
    # Settings that do not load: the automatic lookup alone.
    def broken(*args, **kwargs):
        raise ValueError("SoPA companion requires a loopback bind address")
    monkeypatch.setattr("sopa_companion.engine.config.load_config", broken)
    monkeypatch.setattr(locate, "proseg_live", lambda value=None, root=None: locate.Found(str(program), "path"))
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"sopa-companion {__version__}", f"proseg-live: {program}"]


def test_the_desktop_never_re_executes(tmp_path, monkeypatch, commands):
    """An earlier version started its own .venv's Python in this one's place whenever another one ran it: an installed
    copy's command must run in the environment it was installed in, whatever folder holds a .venv."""
    def no_exec(*args, **kwargs):
        raise AssertionError(f"another interpreter was started: {args}")
    for name in ("execve", "execv", "execl", "execvp"):
        monkeypatch.setattr(os, name, no_exec)
    root = tmp_path / "sopa_companion"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (root / ".venv" / "bin" / "python").chmod(0o755)
    (root / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    monkeypatch.setattr(dirs, "ROOT", root)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "another-environment"))
    for argv in (["launch", "--no-browser"], ["stop"], ["status"], []):
        assert cli.main(argv) == 0, argv
    assert [command for command, _ in commands if command != "settings"] == ["launch", "stop", "status", "launch"]


def test_the_prog_is_the_command_users_typed(monkeypatch, capsys):
    for argv0, prog in (("/env/bin/sopa-companion", "sopa-companion"), ("C:/env/Scripts/sopa-companion.exe",
                                                                         "sopa-companion"),
                        ("/app/sopa_companion/__main__.py", "python -m sopa_companion"),
                        ("-c", "python -m sopa_companion")):
        monkeypatch.setattr(sys, "argv", [argv0])
        assert cli.parser().prog == prog, argv0
    monkeypatch.setattr(sys, "argv", ["/env/bin/sopa-companion"])
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: sopa-companion ") and "--version" in help_text


def test_an_engine_without_its_packages_says_how_to_install(tmp_path, monkeypatch, capsys):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name in ("fastapi", "orjson") else
                        real(name, *args))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                    # where a clone's link folder goes
    first = f"The app's engine needs fastapi, orjson in this Python ({os.path.abspath(sys.executable)})."

    def said(root: Path, argv=("launch",)) -> str:
        monkeypatch.setattr(dirs, "ROOT", root)
        assert cli.main(list(argv)) == 1
        out, err = capsys.readouterr()
        assert out == "" and err.endswith("\n")
        return err.rstrip("\n")
    installed = tmp_path / "env" / "site-packages" / "sopa_companion"
    installed.mkdir(parents=True)
    assert said(installed) == first + "\nInstall the app into this environment as the README's Install section says."
    # A source tree with its own environment: that environment's command, with the PYTHONPATH that runs this copy
    # (its .venv may know another copy through a .pth file): the parent of a folder called sopa_companion, ...
    named = tmp_path / "srv" / "sopa_companion"
    (named / ".venv" / "bin").mkdir(parents=True)
    (named / ".venv" / "bin" / "python").write_text("")
    (named / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    assert said(named, ["status"]) == first + (f"\nThis source folder has its own environment: env "
                                                f"PYTHONPATH={named.parent} {named}/.venv/bin/python -m sopa_companion status")
    # ... else a folder holding one link to it.
    other = tmp_path / "srv" / "sopa-companion"
    (other / ".venv" / "bin").mkdir(parents=True)
    (other / ".venv" / "bin" / "python").write_text("")
    (other / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    from sopa_companion.engine.config import helper_link
    link = helper_link(other)
    assert said(other) == first + (f"\nThis source folder has its own environment: env PYTHONPATH={link.parent} "
                                   f"{other}/.venv/bin/python -m sopa_companion launch")
    assert link.resolve() == other.resolve() and str(link).startswith(str(tmp_path / "home"))
    # A source tree without one: how to make it.
    (named / ".venv" / "bin" / "python").unlink()
    assert said(named, []) == first + ('\nMake this source folder\'s environment as docs/DEVELOPING.md\'s "From a source '
                                       'folder" section says.')


def test_hints_name_the_command_as_the_user_types_it(tmp_path, monkeypatch):
    """What the engine and the command tell the user to type: `sopa-companion` when that is what they typed, or for an
    installed copy (its environment has the command, and the engine its launch starts runs as `python -m`); a source
    folder's `python -m sopa_companion`."""
    from sopa_companion.server import runtime_file
    installed = tmp_path / "env" / "site-packages" / "sopa_companion"
    installed.mkdir(parents=True)
    tree = tmp_path / "clone" / "sopa_companion"
    tree.mkdir(parents=True)
    (tree / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    for argv0, root, words in (("/env/bin/sopa-companion", tree, "sopa-companion"),
                               ("/app/sopa_companion/__main__.py", installed, "sopa-companion"),
                               ("/app/sopa_companion/__main__.py", tree, "python -m sopa_companion")):
        monkeypatch.setattr(sys, "argv", [argv0])
        monkeypatch.setattr(dirs, "ROOT", root)
        assert cli.command_words() == words, (argv0, root)
        assert cli.opening_lines("http://127.0.0.1:8811/", None, False) == (
            f"SoPA companion: http://127.0.0.1:8811/\nFor a private URL: {words} launch --print-url")
        # A second engine on a data folder another engine holds.
        config = {"paths": {"data_dir": str(tmp_path / "var")}}
        held = runtime_file.acquire_engine_lock(config)
        try:
            with pytest.raises(RuntimeError) as refused:
                runtime_file.acquire_engine_lock(config)
        finally:
            os.close(held)
        assert str(refused.value) == (f"Another SoPA companion engine already owns {tmp_path / 'var'}. Use `{words} "
                                      f"status`, or open it with `{words} launch`.")
