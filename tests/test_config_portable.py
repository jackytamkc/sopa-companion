"""Setting the app up on another computer: requirements.txt lists everything the engine imports, `pip install -e .`
makes the venv import this copy from any folder (pyproject.toml), the desktop entry starts this copy wherever the
desktop starts it from, and git keeps build products, local settings and the native/target link out."""
from __future__ import annotations

import ast
import importlib.metadata as metadata
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest

from sopa_companion.engine.config import ROOT

# The engine's own code, run with the venv's interpreter (analysis/ runs in the RAPIDS environment, zarr_bridge/ in
# the sopa one: neither is the venv's).
ENGINE_CODE = ("engine", "server", "desktop", "cli.py", "__main__.py", "__init__.py", "names.py", "dirs.py", "locate.py",
               "portable.py")
CODE_FOLDERS = ("analysis", "api", "desktop", "engine", "server", "zarr_bridge")
# The package's own top-level modules (the notebook's entry point, `import sopa_companion`, is its __init__.py).
TOP_MODULES = ("__init__.py", "__main__.py", "cli.py", "names.py", "dirs.py", "locate.py", "portable.py")


def third_party_imports(paths) -> dict[str, set[str]]:
    """Top-level modules imported anywhere in these files (inside functions too) that are not the standard library's or
    the app's own: {module: {file}}."""
    found: dict[str, set[str]] = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "sopa_companion":
                    found.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return found


def requirement_names() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[\[=<>~! ;]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_requirements_list_every_package_the_engine_imports():
    files = [p for entry in ENGINE_CODE for p in ((ROOT / entry).rglob("*.py") if (ROOT / entry).is_dir() else [ROOT / entry])]
    imported = third_party_imports(files)
    distributions = metadata.packages_distributions()
    listed = requirement_names()
    missing = {}
    for module, where in imported.items():
        names = {d.lower().replace("_", "-") for d in distributions.get(module, [module])}
        if not names & listed:
            missing[module] = sorted(where)
    assert not missing, f"imported by the engine but not in requirements.txt: {missing}"


def test_pyproject_declares_this_folder_as_the_sopa_companion_package():
    """The folder is the sopa_companion package and its code folders its subpackages, API mode's api/ among them; the
    notebook's entry point is the package itself (`import sopa_companion`, its __init__.py). Its dependencies are
    ranges that admit requirements.txt's pins (tests/test_packaging.py checks them)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    setuptools = project["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"sopa_companion": "."}
    # The app's code folders are its subpackages (engine/ is a namespace one: it has no __init__.py).
    assert set(setuptools["packages"]) == {"sopa_companion", *(f"sopa_companion.{folder}" for folder in CODE_FOLDERS)}
    assert all(next((ROOT / folder).glob("*.py"), None) for folder in CODE_FOLDERS)
    assert all((ROOT / name).is_file() for name in TOP_MODULES)
    assert project["tool"]["setuptools"]["dynamic"] == {"version": {"attr": "sopa_companion.__version__"}}
    assert project["project"]["dynamic"] == ["version"] and project["project"]["dependencies"]
    python = project["project"]["requires-python"]
    assert python.startswith(">=3.") and sys.version_info[:2] >= tuple(map(int, python[2:].split(".")))


def setuptools_python() -> str | None:
    """An interpreter whose setuptools can build an editable install and reads pyproject.toml's licence expression (77
    or newer); the venv's may have none."""
    for candidate in (sys.executable, shutil.which("python3"), "/usr/bin/python3"):
        if not candidate or not Path(candidate).is_file():
            continue
        probe = subprocess.run([candidate, "-c", "import setuptools; print(setuptools.__version__)"],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode == 0 and int(probe.stdout.split(".")[0]) >= 77:
            return candidate
    return None


def test_pip_install_e_imports_this_copy_from_any_folder_whatever_it_is_called(tmp_path):
    """What `pip install -e .` installs (the editable wheel setuptools builds from pyproject.toml), unpacked into a
    folder of its own: `python -m sopa_companion`, and the command `sopa-companion`, then run this copy from another
    folder, though the clone's folder is not called sopa_companion."""
    python = setuptools_python()
    if python is None:
        pytest.skip("no interpreter with setuptools 77+ to build the editable install with")
    clone = tmp_path / "sopa-companion"
    for name in ("pyproject.toml", "requirements.txt", *TOP_MODULES):
        (clone / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, clone / name)
    for folder in CODE_FOLDERS:
        for path in (ROOT / folder).rglob("*.py"):
            target = clone / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    built = subprocess.run([python, "-c", "import sys, setuptools.build_meta as b; print(b.build_editable(sys.argv[1]))",
                            str(tmp_path / "wheel")], cwd=clone, capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr[-2000:]
    wheel = tmp_path / "wheel" / built.stdout.strip().splitlines()[-1]
    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)
    requires = [line for line in next(site.glob("*.dist-info/METADATA")).read_text().splitlines()
                if line.startswith("Requires-Dist:")]
    assert any("orjson" in line for line in requires) and any("fastapi" in line for line in requires)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # -I -S: no PYTHONPATH, no current folder, no site-packages, so only the unpacked install can provide sopa_companion.
    run = subprocess.run([python, "-I", "-S", "-c", "import runpy, site, sys; site.addsitedir(sys.argv[1]); "
                          "import sopa_companion, sopa_companion.engine.config as c; print(sopa_companion.__file__, c.ROOT); "
                          "sys.argv = ['sopa_companion', '--help']; runpy.run_module('sopa_companion', run_name='__main__')",
                          str(site)], cwd=elsewhere, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.splitlines()[0] == f"{clone / '__init__.py'} {clone}"
    assert "usage: python -m sopa_companion" in run.stdout
    # The install also carries the command `sopa-companion`, which pip puts on the environment's path: found and run
    # as that command finds and runs it, it takes the same commands, and its usage names it as it was typed.
    commands = next(site.glob("*.dist-info/entry_points.txt")).read_text()
    assert "[console_scripts]\nsopa-companion = sopa_companion.cli:main\n" in commands
    command = subprocess.run([python, "-I", "-S", "-c", "import site, sys; site.addsitedir(sys.argv[1]); "
                              "from importlib.metadata import entry_points; "
                              "(command,) = entry_points(group='console_scripts', name='sopa-companion'); "
                              "sys.argv = ['sopa-companion', '--help']; sys.exit(command.load()())", str(site)],
                             cwd=elsewhere, capture_output=True, text=True, timeout=120)
    assert command.returncode == 0, command.stderr[-2000:]
    assert "usage: sopa-companion" in command.stdout and "install-desktop" in command.stdout


def exec_argv(line: str) -> list[str]:
    """An Exec= value as the desktop runs it: the value's own escapes first, then the command line's quoting."""
    value = line.split("=", 1)[1]
    value = re.sub(r"\\(.)", lambda m: {"s": " ", "n": "\n", "t": "\t", "r": "\r"}.get(m.group(1), m.group(1)), value)
    return [arg.replace("%%", "%") for arg in shlex.split(value)]


def test_the_desktop_entry_starts_this_copy_from_any_folder(tmp_path):
    from sopa_companion.desktop import install
    entry = install.entry_text(sys.executable, tmp_path / "server.log")
    lines = entry.splitlines()
    assert f"Path={ROOT}".replace(" ", "\\s") in lines
    launch = exec_argv(next(line for line in lines if line.startswith("Exec=")))
    assert launch[-3:] == ["-m", "sopa_companion", "launch"]
    stop = exec_argv(next(line for line in lines if line.startswith("Exec=") and line.endswith(" stop")))
    assert stop[:len(launch) - 1] == launch[:-1]
    # The command up to the module, run from another folder with no PYTHONPATH of its own: it imports this copy (not
    # another one that the interpreter's site-packages may point to).
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    command = launch[:-3] + ["-c", "import sopa_companion; print(sopa_companion.__file__)"]
    found = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert found.returncode == 0, found.stderr[-2000:]
    assert Path(found.stdout.strip()) == ROOT / "__init__.py"


def test_a_desktop_entry_for_a_clone_with_another_name_relies_on_the_editable_install(tmp_path):
    """Source trees (pyproject.toml beside the code): one called sopa_companion puts its parent on PYTHONPATH, one of
    another name relies on its editable install. An installed copy relies on its environment, which holds the package:
    never `env PYTHONPATH=` its site-packages (the engine would hand it to every helper), and its launcher starts in
    the folder given (the engine's state folder), not in site-packages."""
    from sopa_companion.desktop import install
    python = "/opt/envs/app env/bin/python"
    roots = [tmp_path / "srv" / "apps" / "sopa_companion", tmp_path / "srv" / "my apps" / "sopa-companion"]
    for root in roots:
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    site = tmp_path / "env" / "site-packages" / "sopa_companion"
    site.mkdir(parents=True)
    state = tmp_path / "state folder"
    named = install.entry_text(python, "/tmp/server.log", root=roots[0])
    other = install.entry_text(python, "/tmp/server.log", root=roots[1])
    installed = install.entry_text(python, "/tmp/server.log", root=site, workdir=state)
    assert exec_argv(next(line for line in named.splitlines() if line.startswith("Exec=")))[:4] == \
        ["env", f"PYTHONPATH={roots[0].parent}", python, "-m"]
    assert exec_argv(next(line for line in other.splitlines() if line.startswith("Exec=")))[:3] == [python, "-m", "sopa_companion"]
    assert "Path=" + str(roots[1]).replace(" ", "\\s") in other.splitlines()
    assert "TryExec=/opt/envs/app\\senv/bin/python" in other.splitlines()
    for line in (line for line in installed.splitlines() if line.startswith("Exec=") and "xdg-open" not in line):
        assert exec_argv(line)[:3] == [python, "-m", "sopa_companion"] and "PYTHONPATH" not in line, line
        assert "site-packages" not in line
    assert "Path=" + str(state).replace(" ", "\\s") in installed.splitlines()
    if validator := shutil.which("desktop-file-validate"):
        for i, text in enumerate((named, other, installed)):
            (tmp_path / f"e{i}.desktop").write_text(text)
            checked = subprocess.run([validator, str(tmp_path / f"e{i}.desktop")], capture_output=True, text=True)
            assert checked.returncode == 0, checked.stdout + checked.stderr


def ignored(repository: Path, *paths: str) -> set[str]:
    out = subprocess.run(["git", "-C", str(repository), "check-ignore", "--no-index", *paths],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode in (0, 1), out.stderr
    return set(out.stdout.split())


def test_helpers_find_the_app_by_its_package_name_in_a_folder_of_any_name(tmp_path):
    """The helpers in the other environments (sopa, RAPIDS, InstanSeg) import the app as sopa_companion from PYTHONPATH,
    and those environments never install it: a clone kept under its repository's name (sopa-companion) was not found
    there, so every sample open, stain run and mixing check failed. Their PYTHONPATH starts with a folder holding one
    link, sopa_companion -> the clone; a folder called sopa_companion gives its parent, as before."""
    from sopa_companion.engine.config import helper_path
    assert helper_path() == str(ROOT.parent)
    clone = tmp_path / "sopa-companion"
    (clone / "engine").mkdir(parents=True)
    (clone / "__init__.py").write_text("WHERE = 'the clone'\n")
    path = helper_path(clone, cache=tmp_path / "cache")
    assert Path(path, "sopa_companion").resolve() == clone.resolve() and Path(path).parent == tmp_path / "cache"
    assert helper_path(clone, cache=tmp_path / "cache") == path               # made once, then used as it is
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": path, "HOME": str(tmp_path)}
    found = subprocess.run([sys.executable, "-c", "import sopa_companion; print(sopa_companion.WHERE)"], env=env,
                           cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert found.stdout.strip() == "the clone", found.stderr


def test_an_installed_copy_never_hands_its_site_packages_to_a_helper(tmp_path, monkeypatch):
    """An installed copy (no pyproject.toml) is called sopa_companion, and its parent is its environment's
    site-packages: helpers in other environments get the link folder instead, and an inherited PYTHONPATH entry naming
    that site-packages is left out. Should the link not be made, an installed copy gives no entry at all; a source tree
    its parent, as before."""
    from sopa_companion.engine.config import helper_env_path, helper_path
    site = tmp_path / "env" / "site-packages"
    installed = site / "sopa_companion"
    (installed / "engine").mkdir(parents=True)
    cache = tmp_path / "cache"
    path = helper_path(installed, cache=cache)
    assert Path(path, "sopa_companion").resolve() == installed.resolve() and Path(path).parent == cache
    assert str(site) not in path
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                    # where the default link folder goes
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), "/kept/one", str(site) + "/", "", "/kept/two"]))
    entries = helper_env_path(installed).split(os.pathsep)
    assert entries[1:] == ["/kept/one", "/kept/two"] and str(site) not in entries
    # The link's place blocked (a file where its folder goes): no entry for an installed copy, never site-packages.
    blocked = tmp_path / "blocked"
    blocked.write_text("")
    assert helper_path(installed, cache=blocked) == ""
    tree = tmp_path / "srv" / "sopa_companion"
    (tree / "engine").mkdir(parents=True)
    other = tmp_path / "srv" / "sopa-companion"
    (other / "engine").mkdir(parents=True)
    for root in (tree, other):
        (root / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    assert helper_path(tree, cache=blocked) == str(tree.parent)          # named as the package: its parent, as before
    assert helper_path(other, cache=blocked) == str(other.parent)        # the link failed: the parent all the same
    # A source tree's inherited PYTHONPATH is handed on as it was given.
    assert helper_env_path(tree) == os.pathsep.join([str(tree.parent), os.environ["PYTHONPATH"]])


def test_git_keeps_build_products_local_settings_and_the_target_link_out(tmp_path):
    """In a scratch repository with the app's .gitignore: native/target is ignored as a symlink (a "native/target/"
    rule never matched one) and as a folder; the tree native/build.sh makes, local settings and pip's leftovers too, and
    the wheels python -m build leaves in dist/ and the weekly resolver's output; the CI's helpers, their accepted
    answers, the workflows and proseg-live's package are kept."""
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True, timeout=60)
    shutil.copy(ROOT / ".gitignore", repository / ".gitignore")
    (repository / "native").mkdir()
    (repository / "native/target").symlink_to(tmp_path / "elsewhere/cargo-target")     # dangling, as in a clone
    out = ["native/target", "native/proseg/Cargo.lock", "native/proseg/src/main.rs", "native/bin/proseg-live",
           "native/.cache/proseg.made", "config.toml", "config.local.toml", "sopa_companion.egg-info/PKG-INFO",
           "var/runs/x", "dist/sopa_companion-0.3.0-py3-none-any.whl", "ci/resolve/out/base.in"]
    kept = ["native/build.sh", "native/proseg-live.patch", "config.example.toml", "pyproject.toml", "requirements.txt",
            "engine/config.py", "tests/test_config.py", "native/package/pyproject.toml", "ci/requirements.py",
            "ci/resolve/expected/base.in", ".github/workflows/release.yml"]
    assert ignored(repository, *out, *kept) == set(out)
    (repository / "native/target").unlink()
    (repository / "native/target").mkdir()
    (repository / "native/target/release").mkdir()
    assert ignored(repository, "native/target") == {"native/target"}
