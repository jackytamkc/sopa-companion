"""What the app runs, found (locate.py): each helper feature's Python ("auto": this Python when it has the feature's
modules, looked up without importing them; "off"; or a path kept as given), the proseg-live program (a source tree's
build, this Python's scripts folder, then, only when it is proseg-live 3.2.0 and no script, the user scheme's and
PATH's; missing, the place its fix puts it), whether another Python imports this very copy of the app, and the engine
log's line that says it all. Fake interpreters stand in for real ones: what this computer's Pythons hold must not decide
a test."""
from __future__ import annotations

import os
from pathlib import Path
import re
import site
import subprocess
import sys
import tomllib
import types

import pytest

from sopa_companion import dirs, locate


def everything(name):
    return object()                       # a spec for every module: "this Python has them all"


def nothing(name):
    return None


def executable(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def copies(tmp_path) -> tuple[Path, Path]:
    """A source tree (pyproject.toml beside the code) and an installed copy (a wheel's package folder)."""
    tree, installed = tmp_path / "clone", tmp_path / "env" / "site-packages" / "sopa_companion"
    for folder in (tree, installed):
        folder.mkdir(parents=True)
    (tree / "pyproject.toml").write_text('[project]\nname = "sopa-companion"\n')
    return tree, installed


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    """This Python's scripts folder and the user scheme's, as tmp folders, and a PATH of one tmp folder: no program of
    this computer's is ever found. This Python reads the user scheme (as a Python outside a virtual environment does)."""
    folders = types.SimpleNamespace(environment=tmp_path / "env" / "bin", user=tmp_path / "user" / "bin",
                                    path=tmp_path / "on-path")
    for folder in vars(folders).values():
        folder.mkdir(parents=True, exist_ok=True)
    stand_in = types.SimpleNamespace(get_preferred_scheme=lambda kind: "posix_user",
                                     get_path=lambda name, scheme=None: str(folders.user if scheme else folders.environment))
    monkeypatch.setattr(locate, "sysconfig", stand_in)
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setenv("PATH", str(folders.path))
    return folders


def program_like(path: Path) -> Path:
    """A program that is not a script (this Python, linked under the given name): what a proseg-live build is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.realpath(sys.executable))
    return path


def answering(said: str, asked: list):
    """A stand-in for locate's subprocess: every program asked for its --version says `said` (and is noted)."""
    def run(command, **kwargs):
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, said.encode() + b"\n", b"")
    return types.SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE,
                                 SubprocessError=subprocess.SubprocessError)


# ---- each helper feature's Python -----------------------------------------------------------------------------------
def test_auto_is_this_python_when_it_has_the_features_modules():
    for value in ("auto", "", None):
        found = locate.python_for("qc", value, find=everything)
        assert found == locate.Found(os.path.abspath(sys.executable), "auto"), value
        assert found.report() == {"path": os.path.abspath(sys.executable), "how": "auto", "missing": [],
                                  "problem": "", "fix": "", "passed": []}
    asked = []
    found = locate.python_for("qc", find=lambda name: asked.append(name) or (None if name == "scviva" else object()))
    assert asked == ["torch", "scvi", "scviva"] and found.path == "" and found.how == "auto"
    assert found.missing == ("scviva",) and found.report()["missing"] == ["scviva"]
    # Clusters & UMAP: what the analysis worker's CPU path imports (analysis/backend.py), which `pip install sopa`
    # brings; a Python with RAPIDS alone is named by its path.
    cpu = ("numcodecs", "scipy", "sklearn", "igraph", "umap", "numba")
    assert locate.python_for("analysis", "", find=nothing).missing == cpu
    assert {feature: modules for feature, (_, _, modules) in locate.FEATURES.items()} == {
        "zarr": ("sopa", "spatialdata"), "stain": ("cellpose",), "instanseg": ("instanseg",),
        "qc": ("torch", "scvi", "scviva"), "analysis": cpu}

    for error in (ValueError("cellpose.__spec__ is None"), ImportError("cellpose")):
        def broken(name, error=error):
            raise error
        assert locate.python_for("stain", find=broken).missing == ("cellpose",)    # a lookup that raises: missing


def test_off_wins_and_a_path_is_kept_as_given(tmp_path, monkeypatch):
    for feature in locate.FEATURES:
        assert locate.python_for(feature, "off", find=everything) == locate.Found("", "off")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    found = locate.python_for("analysis", "~/x/bin/python", find=everything)
    assert found.path == str(tmp_path / "home" / "x" / "bin" / "python") and found.how == "setting"
    assert found.problem == "missing"                                     # kept as given, so the Log can name it
    executable(tmp_path / "home" / "x" / "bin" / "python")
    assert locate.python_for("analysis", "~/x/bin/python") == locate.Found(found.path, "setting")
    # Its spelling is kept: a venv's bin/python is a link to its base interpreter, which finds none of the venv's.
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(sys.executable)
    assert locate.python_for("zarr", str(link)).path == str(link)
    plain = tmp_path / "plain"
    plain.write_text("")                                                   # a file nobody may run
    assert locate.python_for("zarr", str(plain)).problem == "missing"
    with pytest.raises(ValueError, match='qc.python must be "auto", "off" or the path of a Python'):
        locate.python_for("qc", 3)


def test_python_for_imports_nothing(tmp_path):
    before = set(sys.modules)
    for feature in locate.FEATURES:
        for value in (None, "auto", "", "off", str(tmp_path / "python")):
            locate.python_for(feature, value)
    assert set(sys.modules) == before


# ---- the proseg-live program ----------------------------------------------------------------------------------------
def test_proseg_live_auto_order(tmp_path, scripts, monkeypatch):
    # Programs outside this Python's own folders all pass for proseg-live here (the next test checks them).
    monkeypatch.setattr(locate, "outside", lambda path: "")
    tree, installed = copies(tmp_path)
    built = executable(tree / "native" / "bin" / "proseg-live")
    in_environment = executable(scripts.environment / "proseg-live")
    in_user = executable(scripts.user / "proseg-live")
    on_path = executable(scripts.path / "proseg-live")
    assert locate.proseg_live("auto", root=tree) == locate.Found(str(built), "source")
    assert locate.proseg_live(None, root=installed) == locate.Found(str(in_environment), "environment")
    built.chmod(0o644)                                                      # a file nobody may run is skipped
    assert locate.proseg_live("", root=tree) == locate.Found(str(in_environment), "environment")
    in_environment.unlink()
    assert locate.proseg_live(root=tree) == locate.Found(str(in_user), "user")
    in_user.chmod(0o644)
    assert locate.proseg_live(root=installed) == locate.Found(str(on_path), "path")
    on_path.chmod(0o644)
    assert locate.proseg_live(root=installed).how == "missing"
    # Only a source tree's own build counts: an installed copy has no native/bin.
    executable(installed / "native" / "bin" / "proseg-live")
    assert locate.proseg_live(root=installed).how == "missing"
    # A Python without a user scheme skips it.
    def no_user(name, scheme=None):
        if scheme:
            raise KeyError(scheme)
        return str(scripts.environment)
    locate.sysconfig.get_path = no_user
    in_user.chmod(0o755)
    assert locate.proseg_live(root=installed).how == "missing"
    # A Python in a virtual environment does not read the user scheme (pip never installs there for it): a program in
    # ~/.local/bin is not this Python's, and is not taken for this environment's own.
    locate.sysconfig.get_path = lambda name, scheme=None: str(scripts.user if scheme else scripts.environment)
    assert locate.proseg_live(root=installed) == locate.Found(str(in_user), "user")
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    assert locate.proseg_live(root=installed).how == "missing"
    on_path.chmod(0o755)
    assert locate.proseg_live(root=installed) == locate.Found(str(on_path), "path")
    on_path.chmod(0o644)
    # A path setting: as given (relative to the app's folder), whatever "auto" would find.
    assert locate.proseg_live(str(in_user), root=tree) == locate.Found(str(in_user), "setting")
    executable(tree / "bin" / "proseg-live")
    assert locate.proseg_live("bin/proseg-live", root=tree) == locate.Found(str(tree / "bin" / "proseg-live"), "setting")


def test_a_program_outside_this_python_is_taken_only_when_it_is_proseg_live(tmp_path, scripts, monkeypatch):
    """Outside this Python's own folders (the user scheme's scripts folder, PATH) a proseg-live may be another app's
    command (on one computer ~/.local/bin/proseg-live starts the previous version's app) or another proseg's build: it
    is taken only when it is a program, never a script (which is not even run), whose --version says proseg 3.2.0.
    Passed over, the engine keeps the place the fix puts the program and its log says what it passed over and why."""
    tree, installed = copies(tmp_path)
    ran = tmp_path / "ran"
    launcher = executable(scripts.user / "proseg-live", f"#!/bin/sh\ntouch {ran}\necho proseg 3.2.0\n")
    found = locate.proseg_live(root=installed)
    assert (found.path, found.how, found.fix) == (str(scripts.environment / "proseg-live"), "missing", "install")
    assert found.passed == (f"{launcher}: a script, not the program",) and not ran.exists()
    assert locate.program_words(found.report()) == (
        f"proseg-live: missing (install it as the README's Install section says); passed over: {launcher}: a script, "
        "not the program")
    # Another program of that name: asked for its version (here this Python, which says it is Python).
    other = program_like(scripts.path / "proseg-live")
    found = locate.proseg_live(root=tree)
    assert (found.path, found.how, found.fix) == (str(tree / "native" / "bin" / "proseg-live"), "missing", "build")
    assert found.passed[0] == f"{launcher}: a script, not the program"
    assert re.fullmatch(re.escape(f"{other}: its --version says ") + r"'Python 3\.\d+\.\d+.*', not 'proseg 3\.2\.0'",
                        found.passed[1]), found.passed
    # proseg-live as the app runs it (its --version says proseg 3.2.0): taken, after the launcher passed over.
    asked = []
    monkeypatch.setattr(locate, "subprocess", answering("proseg 3.2.0", asked))
    assert locate.proseg_live(root=installed) == locate.Found(str(other), "path",
                                                              passed=(f"{launcher}: a script, not the program",))
    assert asked == [[str(other), "--version"]]
    launcher.unlink()
    in_user = program_like(scripts.user / "proseg-live")
    assert locate.proseg_live(root=installed) == locate.Found(str(in_user), "user")
    # The same program in the user scheme's folder and on PATH (~/.local/bin often is): asked once.
    in_user.unlink()
    other.unlink()
    in_user = program_like(scripts.user / "proseg-live")
    monkeypatch.setenv("PATH", str(scripts.user))
    asked.clear()
    monkeypatch.setattr(locate, "subprocess", answering("proseg 3.3.0", asked))
    found = locate.proseg_live(root=installed)
    assert found.how == "missing" and found.passed == (f"{in_user}: its --version says 'proseg 3.3.0', not "
                                                       "'proseg 3.2.0'",)
    assert asked == [[str(in_user), "--version"]]
    # This Python's own folders are never asked: the proseg-live wheel installed what the app pins.
    asked.clear()
    executable(scripts.environment / "proseg-live")
    assert locate.proseg_live(root=installed).how == "environment" and asked == []
    # A program that cannot answer is passed over too.
    assert locate.outside(tmp_path / "gone").startswith("not readable")


def test_what_makes_a_newer_proseg_live_follows_how_it_was_found():
    """/health's proseg_fix, the page's words for an older proseg-live (one that cannot keep proseg's own count): a
    missing program's fix; else a source folder's build is built again, one installed (this environment, the user's
    scripts, PATH) installed again as the README says, one a setting names pointed at a newer one."""
    assert locate.program_fix({"how": "missing", "fix": "install", "path": "/x"}) == "install"
    assert locate.program_fix({"how": "missing", "fix": "", "path": "/x"}) == "build"
    for how, fix in (("source", "build"), ("environment", "install"), ("user", "install"), ("path", "install"),
                     ("setting", "setting")):
        assert locate.program_fix(locate.Found("/x", how).report()) == fix, how
    assert locate.program_fix(None) == locate.program_fix({}) == "build"


def test_the_program_version_is_the_pinned_prosegs():
    """What a proseg-live outside this Python must say: the proseg the app's pin and the proseg-live package name."""
    project = (dirs.ROOT / "pyproject.toml").read_text()
    pin = re.search(r'"proseg-live(?:>=[\d.]+,)?==(\d+\.\d+\.\d+)\.\*"', project).group(1)
    package = tomllib.loads((dirs.ROOT / "native/package/pyproject.toml").read_text())["project"]["version"]
    assert locate.PROGRAM_VERSION == f"proseg {pin}" == "proseg " + ".".join(package.split(".")[:3])


def test_proseg_live_says_how_to_fix_what_is_missing(tmp_path, scripts, monkeypatch):
    tree, installed = copies(tmp_path)
    expected = {
        ("auto", tree): (tree / "native" / "bin" / "proseg-live", "build"),          # native/build.sh makes it there
        ("auto", installed): (scripts.environment / "proseg-live", "install"),       # the release's install puts it here
        ("/nowhere/proseg-live", tree): (Path("/nowhere/proseg-live"), "setting"),
        ("bin/proseg-live", installed): (installed / "bin" / "proseg-live", "setting"),
        ("native/bin/proseg-live", tree): (tree / "native" / "bin" / "proseg-live", "build"),
        (str(tree / "native" / "bin" / "proseg-live"), tree): (tree / "native" / "bin" / "proseg-live", "build"),
        ("native/bin/proseg-live", installed): (installed / "native" / "bin" / "proseg-live", "setting"),
    }
    for (value, root), (path, fix) in expected.items():
        found = locate.proseg_live(value, root=root)
        assert (found.path, found.how, found.fix) == (str(path), "missing", fix), (value, root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert locate.proseg_live("~/bin/proseg-live", root=tree).path == str(tmp_path / "home" / "bin" / "proseg-live")
    assert locate.fix_words("setting", "/x/proseg-live") == "/x/proseg-live is not there"
    assert locate.fix_words("install", "") == "install it as the README's Install section says"
    assert locate.fix_words("", "") == locate.fix_words("build", "") == "native/build.sh builds it"
    with pytest.raises(ValueError, match="proseg.binary"):
        locate.proseg_live(["proseg-live"], root=tree)


def test_program_names_a_windows_exe(tmp_path, monkeypatch):
    assert locate.program("proseg-live") == "proseg-live"
    # Windows as locate.py sees it (os.name alone: the rest of this process stays POSIX).
    monkeypatch.setattr(locate, "os", types.SimpleNamespace(name="nt", path=os.path, access=os.access, X_OK=os.X_OK))
    assert locate.program("proseg-live") == "proseg-live.exe"
    plain = tmp_path / "proseg-live.exe"
    plain.write_text("")
    assert locate.runnable(plain)                                           # Windows: a file is enough
    monkeypatch.undo()
    assert not locate.runnable(plain) and not locate.runnable(tmp_path / "gone") and not locate.runnable(tmp_path)


# ---- whether another Python imports this very copy ------------------------------------------------------------------
def test_imports_this_copy_asks_the_interpreter(tmp_path):
    folder = tmp_path / "state"
    folder.mkdir()
    record = tmp_path / "seen.txt"
    other = tmp_path / "other" / "sopa_companion" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("")

    def interpreter(name: str, body: str) -> Path:
        return executable(tmp_path / name, "#!/bin/sh\n"
                          f'printf "%s|%s\\n" "${{PYTHONPATH-none}}" "$(pwd)" > "{record}"\n' + body)
    this = interpreter("this", f"printf '%s' '{dirs.ROOT / '__init__.py'}'\n")
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(tmp_path / "engine-only")}
    assert locate.imports_this_copy(str(this), environment, cwd=folder) is True
    pythonpath, where = record.read_text().rstrip("\n").split("|")
    assert pythonpath == "none" and Path(where).resolve() == folder.resolve()     # no PYTHONPATH, in the folder given
    assert environment["PYTHONPATH"] == str(tmp_path / "engine-only")      # the caller's own is left as it was
    cases = {"another": f"printf '%s' '{other}'\n", "none": "exit 0\n",
             "failing": f"printf '%s' '{dirs.ROOT / '__init__.py'}'\nexit 3\n", "slow": "exec sleep 30\n"}
    for name, body in cases.items():
        assert locate.imports_this_copy(str(interpreter(name, body)), environment, cwd=folder, timeout=1) is False, name
    assert locate.imports_this_copy(str(tmp_path / "no-such-python"), environment, cwd=folder) is False


def test_the_probe_names_the_package_a_python_would_import(tmp_path):
    """The real probe, run by this Python with a folder holding one link to this copy first on its path (as a session's
    pylib would be): it names this copy's __init__.py, and nothing is imported to tell."""
    link = tmp_path / "pylib" / "sopa_companion"
    link.parent.mkdir()
    link.symlink_to(dirs.ROOT, target_is_directory=True)
    script = "import sys\nsys.path.insert(0, sys.argv[1])\n" + locate.PROBE + "assert 'sopa_companion' not in sys.modules\n"
    done = subprocess.run([sys.executable, "-I", "-c", script, str(link.parent)], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    assert os.path.samefile(done.stdout, dirs.ROOT / "__init__.py")


# ---- the engine log's line ------------------------------------------------------------------------------------------
def test_summary_says_where_each_helper_runs():
    def report(path="", how="auto", missing=(), problem="", fix=""):
        return {"path": path, "how": how, "missing": list(missing), "problem": problem, "fix": fix}
    located = {"zarr": report("/opt/sopa/bin/python"), "stain": report("/opt/sopa/bin/python"),
               "instanseg": report(missing=["instanseg"]), "qc": report(missing=["scvi", "scviva"]),
               "analysis": report(missing=["cupy", "cuml", "cugraph", "cudf"]),
               "proseg": report("/opt/sopa/bin/proseg-live", "environment")}
    assert locate.summary(located) == (
        "samples and saving: this Python; Cellpose: this Python; the mixing check's learning: off (this Python has no "
        "scvi, scviva); Clusters & UMAP: off (this Python has no cupy, cuml, cugraph, cudf); proseg-live: "
        "/opt/sopa/bin/proseg-live (this Python environment)")
    located.update(zarr=report(how="off"), stain=report("/opt/other/bin/python", "setting"),
                   qc=report("/opt/gone/bin/python", "setting", problem="missing"),
                   proseg=report("/app/native/bin/proseg-live", "missing", fix="build"))
    assert locate.summary(located) == (
        "samples and saving: off; Cellpose: /opt/other/bin/python; the mixing check's learning: /opt/gone/bin/python "
        "(not there); Clusters & UMAP: off (this Python has no cupy, cuml, cugraph, cudf); proseg-live: missing "
        "(native/build.sh builds it)")
    words = {("source", ""): "/p (this source folder's build)", ("path", ""): "/p (found on PATH)",
             ("user", ""): "/p (your user scripts folder)",
             ("setting", ""): "/p (set in the settings)", ("missing", "install"):
             "missing (install it as the README's Install section says)", ("missing", "setting"): "missing (/p is not there)"}
    for (how, fix), said in words.items():
        assert locate.program_words(report("/p", how, fix=fix)) == "proseg-live: " + said


def test_locate_and_dirs_keep_the_import_fence():
    script = ("import sys\n"
              "sys.path.insert(0, sys.argv[1])\n"
              "import sopa_companion.locate, sopa_companion.dirs\n"
              "loaded = [name for name in ('fastapi', 'uvicorn', 'httpx', 'psutil', 'numpy') if name in sys.modules]\n"
              "loaded += [name for name in sys.modules if name.startswith(('sopa_companion.engine', "
              "'sopa_companion.server', 'sopa_companion.api', 'sopa_companion.desktop'))]\n"
              "print(loaded)\n")
    done = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(dirs.ROOT.parent)], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
