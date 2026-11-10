"""Wheel metadata, the proseg-live package beside it, the README's install lines and the optional sopa entry point,
without installing packages or building proseg-live."""
from __future__ import annotations

import ast
import builtins
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import types

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
import pytest

from conftest import api_python

ROOT = Path(__file__).resolve().parents[1]
# The last version shipped as a wheel. pip leaves an installed wheel of the same version in place (it says so and
# installs nothing, --upgrade or not), so a wheel whose files differ must carry a higher version: 0.2.0 still carried
# proseg-live inside it (0.1.0 held the previous version's package and a sopa_companion that only forwarded to it).
LAST_SHIPPED = "0.2.0"
# The one requirement for one system: PyTorch 2.14 on macOS 14 with Apple silicon (pyproject.toml says why).
MACOS_14_TORCH = 'torch>=2.14,<2.15; sys_platform == "darwin" and platform_machine == "arm64" and "23." in platform_release'


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def sopa_environment():
    # The environment API mode's users have ($SOPA_COMPANION_API_PYTHON: `pip install sopa` and the companion's needs).
    python = api_python()
    probe = subprocess.run([python, "-I", "-B", "-c", """
import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path
print(json.dumps({
    'python': '.'.join(map(str, sys.version_info[:3])),
    'versions': {d.metadata['Name']: d.version for d in metadata.distributions()},
    'sopa_init': Path(importlib.util.find_spec('sopa').origin).read_text(),
}))
"""], capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    data = json.loads(probe.stdout)
    data["versions"] = {canonicalize_name(name): version for name, version in data["versions"].items()}
    return data


def test_dependencies_are_ranges_and_keep_desktop_requirements(project):
    """Ranges that admit a sopa environment, the desktop's pins inside them; the one exact pin is proseg-live's: the
    proseg it patches (3.2.0.*: a fourth number is a new patch on the same proseg), from 3.2.0.1 on, the first patch
    that keeps proseg's own count. sopa itself is a dependency, and the sopa-companion command is the console script."""
    metadata = project["project"]
    assert metadata["name"] == "sopa-companion"
    assert metadata["requires-python"] == ">=3.12"           # zarr 3.4.1, the zarr floor, needs Python 3.12
    assert metadata["dynamic"] == ["version"]
    groups = [metadata["dependencies"], *metadata["optional-dependencies"].values()]
    for group in groups:
        for text in group:
            req = Requirement(text)
            assert req.url is None and req.specifier, text
            if canonicalize_name(req.name) == "proseg-live":
                assert req.specifier == SpecifierSet(">=3.2.0.1,==3.2.0.*"), text
            elif req.marker is not None:
                assert text == MACOS_14_TORCH, text
            else:
                assert all(spec.operator == ">=" for spec in req.specifier), text
    base = {canonicalize_name(req.name): req for req in map(Requirement, metadata["dependencies"])}
    assert str(base["sopa"].specifier) == ">=2.2.10" and str(base["platformdirs"].specifier) == ">=4"
    assert "proseg-live" in base
    assert project["project"]["scripts"] == {"sopa-companion": "sopa_companion.cli:main"}
    desktop = [Requirement(line) for text in (ROOT / "requirements.txt").read_text().splitlines()
               if (line := text.split("#", 1)[0].strip())]
    for req in desktop:
        if canonicalize_name(req.name) == "pytest":
            continue
        current = base[canonicalize_name(req.name)]
        assert req.extras <= current.extras
        for spec in req.specifier:
            if spec.operator == "==":
                assert spec.version in current.specifier, (req, current)
    assert "pytest" not in base
    assert set(map(str, map(Requirement, metadata["optional-dependencies"]["api"]))) <= set(metadata["dependencies"])


def test_pytorch_2_14_on_macos_14_with_apple_silicon_only_and_a_marker_every_pip_reads(project):
    """PyTorch 2.15 calls Apple's GPU available on macOS 14 but aborts in every linear layer there, so [stain] and [qc]
    (and [all], the README's install) take PyTorch 2.14 on macOS 14 with Apple silicon, and only there: macOS 13 has no
    wheel of either, later macOS takes the newest PyTorch, and so do Intel Macs, Linux and Windows. pip with packaging 22
    to 25 raises InvalidVersion on any version comparison with platform_release where it is no version (a Linux
    kernel's "6.8.0-45-generic"), so the marker matches the Darwin version as text and no requirement compares it."""
    optional = project["project"]["optional-dependencies"]
    assert MACOS_14_TORCH in optional["stain"] and MACOS_14_TORCH in optional["qc"]
    marker = Requirement(MACOS_14_TORCH).marker
    for system, machine, release, pinned in [("darwin", "arm64", "23.6.0", True), ("darwin", "arm64", "23.0.0", True),
                                             ("darwin", "arm64", "22.6.0", False), ("darwin", "arm64", "24.6.0", False),
                                             ("darwin", "arm64", "25.1.0", False), ("darwin", "x86_64", "23.6.0", False),
                                             ("linux", "x86_64", "6.8.0-45-generic", False),
                                             ("linux", "x86_64", "5.15.153.1-microsoft-standard-WSL2", False),
                                             ("win32", "AMD64", "11", False)]:
        environment = {"sys_platform": system, "platform_machine": machine, "platform_release": release}
        assert marker.evaluate(environment) is pinned, environment
    for text in [*project["project"]["dependencies"], *(text for group in optional.values() for text in group)]:
        for comparison in re.split(r"\band\b|\bor\b|[()]", str(Requirement(text).marker or "")):
            if re.search(r"platform_(release|version)", comparison):
                assert re.search(r"\bin\b", comparison), (text, comparison)


def test_the_apps_own_code_is_bsd_3_clause_and_the_citation_names_this_version(project):
    """The app's own code is under the BSD 3-Clause licence: its text at the top (LICENSE), named as an SPDX expression
    with that file in pyproject.toml (setuptools 77 or newer reads both), and said in the README beside proseg-live's
    GPL v3, which covers native/ alone. CITATION.cff (what GitHub and an archive of a release cite) names the same
    licence and this version, so a release cannot leave it behind."""
    from sopa_companion import __version__
    metadata = project["project"]
    assert metadata["license"] == "BSD-3-Clause" and metadata["license-files"] == ["LICENSE"]
    assert Requirement(project["build-system"]["requires"][0]).specifier == SpecifierSet(">=77")
    assert not [c for c in metadata["classifiers"] if c.startswith("License ::")]      # the expression says it
    licence = (ROOT / "LICENSE").read_text()
    assert licence.startswith("BSD 3-Clause License\n\nCopyright (c) ") and "Neither the name of the copyright holder" in licence
    readme = (ROOT / "README.md").read_text()
    assert ("is under the BSD 3-Clause licence: see\n  "
            "[LICENSE](https://github.com/jackytamkc/sopa-companion/blob/main/LICENSE)") in readme and "no licence chosen" not in readme
    citation = (ROOT / "CITATION.cff").read_text()
    assert f"\nversion: {__version__}\n" in citation and "\nlicense: BSD-3-Clause\n" in citation
    assert re.search(r"\ndate-released: \d{4}-\d{2}-\d{2}\n", citation)


def test_the_pin_replaces_a_proseg_live_that_cannot_keep_proseg_s_own_count(project):
    """pip replaces an installed proseg-live 3.2.0 (it cannot keep proseg's own count: the engine says old_proseg while
    it is still there) with 3.2.0.1 or a later patch of proseg 3.2.0; never another proseg (3.2.1, 3.3.0). The
    squatter's 3.2.0.99 that the release job installs beside (ci/squat_index.py) is inside the pin, as before."""
    pin = next(Requirement(text) for text in project["project"]["dependencies"]
               if canonicalize_name(Requirement(text).name) == "proseg-live")
    for version, taken in (("3.2.0", False), ("3.2.0.1", True), ("3.2.0.2", True), ("3.2.0.99", True), ("3.2.1", False),
                           ("3.3.0", False), ("3.1.9.9", False)):
        assert pin.specifier.contains(Version(version)) is taken, version


def test_ranges_admit_the_installed_sopa_environment(project, sopa_environment):
    versions = sopa_environment["versions"]
    assert {"sopa", "spatialdata", "numpy", "pyarrow", "zarr"} <= versions.keys()
    assert sopa_environment["python"] in SpecifierSet(project["project"]["requires-python"])
    groups = [project["project"]["dependencies"], project["project"]["optional-dependencies"]["sopa"]]
    checked = set()
    for group in groups:
        for req in map(Requirement, group):
            name = canonicalize_name(req.name)
            if name in versions:
                assert versions[name] in req.specifier, f"{req} excludes installed {name} {versions[name]}"
                checked.add(name)
    assert {"sopa", "numpy", "pyarrow", "zarr"} <= checked


def test_packages_name_every_runtime_package_folder(project):
    """One package, sopa_companion: the repository's top folder (no second package beside it, no proseg_live)."""
    settings = project["tool"]["setuptools"]
    assert settings["package-dir"] == {"sopa_companion": "."}
    declared = set(settings["packages"])
    assert {"sopa_companion.api", "sopa_companion"} <= declared
    assert all(name == "sopa_companion" or name.startswith("sopa_companion.") for name in declared)
    assert not (ROOT / "facade").exists()
    # The harness, the tests and the CI's helpers are repository tools, not runtime packages (native/package/ is
    # proseg-live's own). engine is a namespace package.
    excluded = {"harness", "tests", "docs", "native", "frontend", "packaging", "build", "ci"}
    discovered = {"sopa_companion"}
    for folder in ROOT.iterdir():
        if not folder.is_dir() or folder.is_symlink() or folder.name.startswith(".") or folder.name in excluded:
            continue
        for path in folder.rglob("*.py"):
            if "__pycache__" not in path.parts:
                discovered.add("sopa_companion." + ".".join(path.parent.relative_to(ROOT).parts))
    assert discovered <= declared, f"runtime packages missing from pyproject.toml: {sorted(discovered - declared)}"
    assert "sopa_companion.harness" not in declared
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "sopa_companion.__version__"}


def test_the_console_command_is_the_function_python_m_sopa_companion_calls(project):
    """An install (the wheel, or `pip install -e .`) gives the command `sopa-companion`: loaded as the installed
    command loads it, it is a callable, the very one `python -m sopa_companion` calls (__main__.py), so both take the
    same commands."""
    scripts = project["project"]["scripts"]
    assert scripts == {"sopa-companion": "sopa_companion.cli:main"}
    command = importlib.metadata.EntryPoint(name="sopa-companion", value=scripts["sopa-companion"],
                                            group="console_scripts").load()
    assert callable(command)
    tree = ast.parse((ROOT / "__main__.py").read_text())
    imported = {}                                       # what __main__.py imports from the package, by its name there
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            module = importlib.import_module(f"sopa_companion.{node.module}")
            imported.update({alias.asname or alias.name: getattr(module, alias.name) for alias in node.names})
    called = [imported[node.func.id] for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported]
    assert called == [command]


def proseg_live_package() -> dict:
    return tomllib.loads((ROOT / "native/package/pyproject.toml").read_text())


def check_companion():
    """ci/check_companion.py as a module (standard library only), as tests/test_ci.py imports it."""
    sys.path.insert(0, str(ROOT / "ci"))
    try:
        import check_companion
    finally:
        sys.path.remove(str(ROOT / "ci"))
    return check_companion


def test_the_version_is_above_the_last_shipped_wheel():
    """A wheel built from this tree replaces an installed one of the last shipped version (pip installs a local
    wheel only when its version differs from the installed one), and PyPI takes each version's files once. The README
    installs both packages from PyPI with one line."""
    from sopa_companion import __version__
    from test_config_docs import PYPI_INSTALL
    assert Version(__version__) > Version(LAST_SHIPPED)
    assert str(Version(__version__)) == __version__                    # a plain release number, as pip reads it
    lines = (ROOT / "README.md").read_text().splitlines()
    assert any(text.strip() == PYPI_INSTALL for text in lines)


def test_proseg_live_is_its_own_gpl_package(project):
    """proseg-live is proseg (GPL v3) with the app's patch, and a package of its own (native/package/): maturin builds
    the one program proseg-live (not proseg, which sopa installs, nor proseg-to-baysor), stripped and from the patch's
    Cargo.lock, with proseg's version (a fourth number for a new patch on it), and no software bill of materials, which
    would name the folder it was built in. The companion pins it by its first three numbers. Its page says where its
    complete source is (crates.io's crate, the patch, Cargo.lock) and its licence; the patch adds the program's
    target; native/ keeps proseg's own licence file and a notice about the patched proseg. The pin's floor is the
    oldest patch the companion needs (NEEDS, ci/check_companion.py proseg_live_pin)."""
    package = proseg_live_package()
    version = next(line.split("=", 1)[1] for line in (ROOT / "native/build.sh").read_text().splitlines()
                   if line.startswith("VERSION="))
    assert package["project"]["name"] == "proseg-live" and package["project"]["version"].startswith(version)
    assert package["build-system"]["build-backend"] == "maturin"
    # The sbom setting below is read from maturin 1.12 on: pip, building from the sdist, never picks an older one.
    assert Requirement(package["build-system"]["requires"][0]).specifier == SpecifierSet(">=1.12,<2")
    maturin = package["tool"]["maturin"]
    assert maturin["bindings"] == "bin" and maturin["targets"] == [{"name": "proseg-live", "kind": "bin"}]
    assert maturin["strip"] is True and maturin["locked"] is True and maturin["sbom"] == {"rust": False}
    assert package["project"]["license"] == {"file": "LICENSE.md"} and package["project"]["readme"] == "PROSEG-LIVE.md"
    # Both packages are on PyPI, which refuses a classifier that forbids the upload.
    for classifiers in (package["project"]["classifiers"], project["project"]["classifiers"]):
        assert not [c for c in classifiers if c.startswith("Private ::")]
    pin = next(text for text in project["project"]["dependencies"] if canonicalize_name(Requirement(text).name) == "proseg-live")
    assert Requirement(pin).specifier == Requirement(check_companion().proseg_live_pin()).specifier
    page = (ROOT / "native/package/PROSEG-LIVE.md").read_text()
    for said in (f"https://crates.io/crates/proseg/{version}", "`proseg-live.patch`", "`Cargo.lock`",
                 "GNU General Public License v3", "proseg_live-<version>.tar.gz"):
        assert said in page, said
    patch = (ROOT / "native/proseg-live.patch").read_text()
    cargo = patch.split("diff -ruN a/Cargo.toml b/Cargo.toml\n", 1)[1].split("\ndiff ", 1)[0]
    assert '+name = "proseg-live"' in cargo.splitlines()
    licence = (ROOT / "native/LICENSE.md").read_text()
    assert licence.lstrip().startswith("Proseg is Copyright (C) 2024 by Daniel C. Jones")
    assert "GNU GENERAL PUBLIC LICENSE" in licence and "Version 3, 29 June 2007" in licence
    notice = (ROOT / "native/README.md").read_text()
    assert f"[proseg](https://github.com/dcjones/proseg) {version}" in notice
    assert f"https://crates.io/crates/proseg/{version}" in notice
    for name in ("proseg-live.patch", "build.sh", "LICENSE.md", "GNU General Public License v3"):
        assert name in notice, name


def test_package_data_covers_the_page_config_and_icon(project):
    """The companion's wheel is pure Python: the page, the settings' example and the icon, and nothing of native/
    (proseg-live, its patch and licence are proseg-live's own package)."""
    settings = project["tool"]["setuptools"]
    assert settings["include-package-data"] is False
    patterns = settings["package-data"]["sopa_companion"]
    assert not [pattern for pattern in patterns if pattern.startswith("native")]
    included = {path.relative_to(ROOT).as_posix() for pattern in patterns for path in ROOT.glob(pattern)
                if path.is_file()}
    needed = {"frontend/index.html", "config.example.toml", "desktop/sopa-companion.svg"}
    for folder in ("frontend/src", "frontend/styles", "frontend/demos"):
        needed.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).rglob("*") if path.is_file())
    assert needed <= included, f"runtime assets missing from the wheel: {sorted(needed - included)}"
    assert not any(path.startswith(("tests/", "harness/", "frontend/test/", "var/")) or
                   path in {"config.toml", "config.local.toml"} for path in included)


def test_build_wheel_sh_is_gone():
    """The companion's wheel no longer carries a program, so the script that tagged it for one platform went; wheels
    come from python -m build (the companion) and native/build.sh --wheel (proseg-live), or from CI."""
    assert not (ROOT / "build_wheel.sh").exists()
    assert "build_wheel" not in (ROOT / "README.md").read_text() + (ROOT / "pyproject.toml").read_text()


def fork_module_source():
    patch = (ROOT / "packaging/sopa-fork/sopa.patch").read_text()
    section = patch.split("diff --git a/sopa/_companion.py b/sopa/_companion.py\n", 1)[1].split("diff --git ", 1)[0]
    assert "--- /dev/null\n" in section
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")) + "\n"


@pytest.fixture
def companionapp(monkeypatch):
    original_import = builtins.__import__

    def no_dependencies(name, *args, **kwargs):
        if name.split(".")[0] in {"sopa_companion", "sopa_companion", "spatialdata", "numpy", "fastapi", "uvicorn"}:
            pytest.fail(f"the sopa wrapper imported {name} before it was called")
        return original_import(name, *args, **kwargs)

    namespace = {"__name__": "_companion_test"}
    with monkeypatch.context() as fence:
        fence.setattr(builtins, "__import__", no_dependencies)
        exec(compile(fork_module_source(), "sopa/_companion.py", "exec"), namespace)
    return namespace["companionapp"]


def test_sopa_wrapper_import_is_lazy(companionapp):
    assert callable(companionapp)


def test_sopa_wrapper_forwards_the_object_options_and_handle(monkeypatch, companionapp):
    module = types.ModuleType("sopa_companion")
    module.SOPA_API = 1
    sample, handle = object(), object()
    seen = []

    def launch(sdata, **kwargs):
        seen.append((sdata, kwargs))
        return handle

    module.launch = launch
    monkeypatch.setitem(sys.modules, "sopa_companion", module)
    assert companionapp(sample, open="none", tool="proseg", timeout=10) is handle
    assert seen == [(sample, {"open": "none", "tool": "proseg", "timeout": 10})]


def test_sopa_wrapper_missing_companion_has_the_install_hint(monkeypatch, companionapp):
    original_import = builtins.__import__
    missing = ModuleNotFoundError("No module named 'sopa_companion'", name="sopa_companion")

    def unavailable(name, *args, **kwargs):
        if name == "sopa_companion":
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ModuleNotFoundError, match="pip install 'sopa\\[companion\\]'") as caught:
        companionapp(object())
    assert caught.value.name == "sopa_companion"
    assert caught.value.__cause__ is missing


@pytest.mark.parametrize("error", [ModuleNotFoundError("missing numpy", name="numpy"),
                                  ModuleNotFoundError("broken companion", name="sopa_companion._internal"),
                                  ImportError("incompatible dependency")])
def test_sopa_wrapper_does_not_hide_import_errors(monkeypatch, companionapp, error):
    original_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "sopa_companion":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(type(error)) as caught:
        companionapp(object())
    assert caught.value is error


@pytest.mark.parametrize("api", [None, 0])
def test_sopa_wrapper_rejects_an_old_api(monkeypatch, companionapp, api):
    module = types.ModuleType("sopa_companion")
    if api is not None:
        module.SOPA_API = api
    monkeypatch.setitem(sys.modules, "sopa_companion", module)
    with pytest.raises(ImportError, match="pip install --upgrade 'sopa\\[companion\\]'"):
        companionapp(object())


def test_sopa_wrapper_keeps_launch_errors(monkeypatch, companionapp):
    module = types.ModuleType("sopa_companion")
    module.SOPA_API = 1
    error = RuntimeError("no room for the session")

    def launch(sdata, **kwargs):
        raise error

    module.launch = launch
    monkeypatch.setitem(sys.modules, "sopa_companion", module)
    with pytest.raises(RuntimeError) as caught:
        companionapp(object())
    assert caught.value is error


def test_sopa_patch_applies_to_the_installed_init_and_extra_context(tmp_path, sopa_environment):
    if sopa_environment["versions"]["sopa"] != "2.2.11":
        pytest.skip("the fork patch targets sopa 2.2.11; select that environment with SOPA_COMPANION_API_PYTHON")
    if not shutil.which("git"):
        pytest.skip("git is needed to check application of the sopa patch")
    # Wheels omit pyproject.toml: this checks its extra hunk's context, not a full upstream checkout.
    (tmp_path / "pyproject.toml").write_text('''[project]
name = "sopa"
version = "2.2.11"

[project.optional-dependencies]
cellpose = ["cellpose>=3.0.5", "opencv-python>=4.8.0"]
stardist = ["tensorflow>=2.18.0", "stardist>=0.9.2"]
baysor = ["toml>=0.10.2", "loompy>=3.0.7"]
''')
    (tmp_path / "sopa").mkdir()
    init = tmp_path / "sopa/__init__.py"
    init.write_text(sopa_environment["sopa_init"])
    patch = ROOT / "packaging/sopa-fork/sopa.patch"
    for flags in (["--check"], []):
        run = subprocess.run(["git", "apply", *flags, str(patch)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
    assert init.read_text() == sopa_environment["sopa_init"] + "    from ._companion import companionapp\n"
    assert (tmp_path / "sopa/_companion.py").read_text() == fork_module_source()
    extras = tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["companion"] == ["sopa-companion>=0.3"]
    # The extra admits this version and not the last shipped one, so it replaces that one where it is installed.
    from sopa_companion import __version__
    floor = Requirement(extras["companion"][0]).specifier
    assert __version__ in floor and LAST_SHIPPED not in floor
    assert set(extras) == {"companion", "cellpose", "stardist", "baysor"}
