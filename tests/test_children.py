"""How the engine starts its helpers (engine/children.py): the GPU each sees, counted
within the engine's own CUDA_VISIBLE_DEVICES; the app on PYTHONPATH only where the helper's interpreter needs it (a
source tree's desktop as before, an installed copy nothing in its own environment and a link folder in another, API
mode's pylib for other environments, made when first needed) with MPLBACKEND=Agg in API mode; the folder each starts in,
never site-packages; setpriv --pdeathsig on the helper's read-only commands, never on save; and the helpers that use
them."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sopa_companion import dirs
from sopa_companion.engine import children
from sopa_companion.engine.config import ROOT, helper_env_path, helper_link, helper_path


@pytest.fixture(autouse=True)
def desktop_after(monkeypatch):
    """Every test leaves the module as an engine made with a desktop configuration would."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield
    children.configure({})


def api(tmp_path) -> Path:
    state = tmp_path / "state"
    children.configure({"session": {"mode": "api"}, "paths": {"state_dir": str(state)}})
    return state


# ---- GPUs ----------------------------------------------------------------------------------------------------------
def test_a_gpu_index_counts_within_the_engines_own_gpus(monkeypatch):
    assert [children.gpu(i) for i in (0, 1, 3)] == ["0", "1", "3"]          # unset: the GPU's own number, as before
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert [children.gpu(i) for i in (0, 1, 2)] == ["2", "3", ""]           # past the end: no GPU at all
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " GPU-5d2c1f , GPU-77aa ")
    assert [children.gpu(i) for i in (0, 1)] == ["GPU-5d2c1f", "GPU-77aa"]    # UUIDs pass through
    for hidden in ("", "-1", "-1,0", ",1"):                                   # none visible: CUDA stops at a bad entry
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
        assert children.gpu(0) == ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,-1,0")
    assert children.gpu(0) == "1" and children.gpu(1) == ""
    for bad in (-1, True, 1.0, "0", None):
        with pytest.raises(ValueError):
            children.gpu(bad)


def test_a_gpu_the_engine_does_not_see_is_said_once_in_its_log(monkeypatch, capsys):
    """A kernel that pinned its GPUs (CUDA_VISIBLE_DEVICES=1) maps the config's [stain] gpu = 1 to no GPU: the helper
    runs without one (Cellpose on the processor), and the engine's log says why, once."""
    monkeypatch.setattr(children, "_said", set())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert children.gpu(0) == "1" and capsys.readouterr().out == ""
    assert children.gpu(1) == children.gpu(1) == children.env("other", gpu=1)["CUDA_VISIBLE_DEVICES"] == ""
    said = capsys.readouterr().out.splitlines()
    assert said == ["SoPA companion: GPU 1 is not one of the GPUs this engine sees (CUDA_VISIBLE_DEVICES='1'): a helper "
                    "set to use it sees no GPU"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert children.gpu(1) == "" and "CUDA_VISIBLE_DEVICES=''" in capsys.readouterr().out    # another view: said too


def test_the_environment_shows_one_gpu_in_pci_order_and_never_the_desktop(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":77")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("SOPA_COMPANION_EVENTS_FD", "9")                     # the API engine's own pipe
    env = children.env("other", gpu=1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env and "DISPLAY" in os.environ
    assert "SOPA_COMPANION_EVENTS_FD" not in env
    from sopa_companion.api import protocol
    assert children.EVENTS_FD == protocol.ENV_EVENTS_FD
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert children.env("same", gpu=1)["CUDA_VISIBLE_DEVICES"] == "1"
    assert children.env("same")["CUDA_VISIBLE_DEVICES"] == "3,1"                # gpu None: as the engine sees them
    with pytest.raises(ValueError):
        children.env("elsewhere")


# ---- the app on PYTHONPATH -----------------------------------------------------------------------------------------
def test_the_desktop_keeps_todays_pythonpath_and_working_folder(monkeypatch):
    """Today's, helper by helper: the GPU worker, the trainer, the shape workers and the sample helper have the app's
    entry and then the engine's own PYTHONPATH (helper_env_path); the stain helpers (inherit=False) the app's alone."""
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.delenv("MPLBACKEND", raising=False)
    for kind in ("same", "other"):
        env = children.env(kind)
        assert env["PYTHONPATH"] == helper_env_path() and env["PYTHONPATH"].split(os.pathsep)[-1] == "/somewhere/else"
        assert "MPLBACKEND" not in env
        assert children.env(kind, inherit=False)["PYTHONPATH"] == helper_path()
    assert children.app_path() == helper_path() == str(ROOT.parent)        # the app folder is called sopa_companion
    assert not hasattr(children, "api_mode")              # app_path() and env() follow the mode; nothing else asks it


def test_api_mode_gives_other_environments_the_package_alone(tmp_path, monkeypatch):
    """The installed package's parent is the sopa env's site-packages: on a RAPIDS helper's path it would leak the whole
    env into it. Other environments get <state>/pylib, one link to the package; the engine's own keeps its path."""
    monkeypatch.setenv("PYTHONPATH", "/kernel/site-packages")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")     # what ipykernel exports
    state = api(tmp_path)
    state.mkdir()
    assert children.app_path() == str(state) and not (state / "pylib").exists()   # made when a helper first needs it
    same = children.env("same")
    assert same["PYTHONPATH"] == "/kernel/site-packages" and not (state / "pylib").exists()
    other = children.env("other")
    assert other["PYTHONPATH"] == str(state / "pylib") and children.app_path() == str(state)
    assert other["MPLBACKEND"] == "Agg"
    link = state / "pylib" / "sopa_companion"
    assert link.is_symlink() and link.resolve() == ROOT and sorted(p.name for p in (state / "pylib").iterdir()) == \
        ["sopa_companion"]
    same = children.env("same")
    assert same["PYTHONPATH"] == "/kernel/site-packages" and same["MPLBACKEND"] == "Agg"
    # A link the handle made (api/folders.py) is used as it is; a wrong one is pointed at this package.
    link.unlink()
    link.symlink_to(tmp_path, target_is_directory=True)
    assert children.env("other")["PYTHONPATH"] == str(state / "pylib") and link.resolve() == ROOT
    assert children.env("other", inherit=False)["PYTHONPATH"] == str(state / "pylib")    # the desktop's choice only
    children.configure({"session": {"mode": "desktop"}, "paths": {"state_dir": str(state)}})
    assert children.env("other")["PYTHONPATH"] == helper_env_path()


def test_without_its_pylib_another_environment_gets_no_path_never_the_engines(tmp_path, monkeypatch, capsys):
    """Should <state>/pylib not be made, a helper in another environment gets no PYTHONPATH (it cannot import the app,
    and fails saying so) and helpers start in the state folder: never the desktop's helper_path(), which for an
    installed package is the engine's site-packages. Said once in the engine's log."""
    monkeypatch.setattr(children, "_said", set())
    monkeypatch.setenv("PYTHONPATH", "/kernel/site-packages")
    state = api(tmp_path)
    state.mkdir()
    (state / "pylib").write_text("")                                        # in the way: pylib cannot be made
    other = children.env("other")
    assert "PYTHONPATH" not in other and other["MPLBACKEND"] == "Agg"
    assert children.env("same")["PYTHONPATH"] == "/kernel/site-packages"   # the engine's own: the package is there
    assert children.app_path() == str(state) and helper_path() not in (children.app_path(), other.get("PYTHONPATH"))
    said = [line for line in capsys.readouterr().out.splitlines() if "could not be made" in line]
    assert len(said) == 1 and said[0].startswith(f"SoPA companion: {state / 'pylib' / 'sopa_companion'} could not be made")


def test_a_helper_in_another_environment_imports_this_package_and_nothing_of_the_engines_path(tmp_path, monkeypatch):
    # What the engine itself was given: the app's parent and a folder only the engine's path names. Only that folder is
    # looked for in the helper's path: an interpreter whose environment has the app installed editable adds the app's
    # parent by its own .pth file (the app's installed copy does), which the helper's environment does not decide.
    engine_only = tmp_path / "engine-only"
    engine_only.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT.parent), str(engine_only)]))
    api(tmp_path)
    found = subprocess.run([sys.executable, "-c", "import json, sys, sopa_companion; "
                            "print(json.dumps([sopa_companion.__file__, sys.path]))"],
                           cwd=children.app_path(), env=children.env("other"), capture_output=True, text=True,
                           timeout=60)
    assert found.returncode == 0, found.stderr[-500:]
    module, path = json.loads(found.stdout)
    assert Path(module).resolve() == ROOT / "__init__.py"
    assert str(engine_only) not in path and str(tmp_path / "state" / "pylib") in path


def test_same_or_other_environment_by_the_interpreters_prefix(tmp_path):
    assert children.kind(sys.executable) == "same"
    other = tmp_path / "env" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("")
    assert children.kind(str(other)) == "other" and children.kind("") == "other" and children.kind(None) == "other"
    # Two venvs made from one base interpreter: each bin/python links to it, yet neither is this environment (a link
    # followed would make them "the same", and a RAPIDS venv's worker would go without the app).
    for name in ("a", "b"):
        python = tmp_path / name / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.symlink_to(os.path.realpath(sys.executable))
        (tmp_path / name / "pyvenv.cfg").write_text(f"home = {Path(os.path.realpath(sys.executable)).parent}\n")
        assert children.kind(str(python)) == "other", name


# ---- launchers -----------------------------------------------------------------------------------------------------
def test_read_only_helper_commands_die_with_the_engine_but_save_never_does(monkeypatch):
    for command in ("inspect", "shapes", "prepare", "count", "tissue", "segmentations", "stain", "export"):
        assert children.helper_prefix(command) == ["setpriv", "--pdeathsig", "TERM"], command
    for command in ("save", "something-new", ""):
        assert children.helper_prefix(command) == []
    monkeypatch.setattr(children.shutil, "which", lambda name: None)       # no setpriv on this computer
    assert children.helper_prefix("inspect") == []


# ---- the helpers that use them -------------------------------------------------------------------------------------
def test_the_stain_helpers_take_their_gpu_and_path_from_here(tmp_path, monkeypatch):
    from sopa_companion.engine.stain import StainRuns
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    stain = SimpleNamespace(python=sys.executable, gpu=1, instanseg_python=str(tmp_path / "is" / "bin" / "python"),
                            instanseg_models=tmp_path / "instanseg-models")
    env = StainRuns.helper_env(stain, tmp_path / "cache")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-b" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["NUMBA_CACHE_DIR"] == str(tmp_path / "cache" / "numba") and env["OMP_NUM_THREADS"] == "4"
    assert env["PYTHONPATH"] == helper_path()                              # the desktop: the app alone, as before
    stain.gpu = ""                                                          # tests: the helpers see no GPU
    assert StainRuns.helper_env(stain, tmp_path / "cache")["CUDA_VISIBLE_DEVICES"] == ""
    state = api(tmp_path)
    stain.gpu = 0
    same = StainRuns.helper_env(stain, tmp_path / "cache")                # Cellpose in the engine's own environment
    assert same.get("PYTHONPATH") == os.environ.get("PYTHONPATH") and same["MPLBACKEND"] == "Agg"
    # An InstanSeg run's helper starts InstanSeg's own environment, which inherits it: the package alone.
    pythons = StainRuns.run_pythons(stain, {"method": "instanseg"})
    assert pythons == (stain.python, stain.instanseg_python)
    assert StainRuns.helper_env(stain, tmp_path / "cache", *pythons)["PYTHONPATH"] == str(state / "pylib")
    assert StainRuns.run_pythons(stain, {"method": "cellpose"}) == (stain.python,)


def test_the_patch_worker_keeps_the_app_alone_on_the_desktop(tmp_path, monkeypatch):
    from sopa_companion.engine import stain_worker
    started = []

    class Popen:
        def __init__(self, command, **kwargs):
            started.append(kwargs)
            read, write = os.pipe()
            os.write(write, b'{"ok": true, "result": 7}\n')
            os.close(write)
            self.stdin, self.stdout = open(os.devnull, "wb"), os.fdopen(read, "rb")

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(stain_worker.subprocess, "Popen", Popen)
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setenv("DISPLAY", ":77")
    worker = stain_worker.StainWorker(sys.executable)
    assert worker.call(SimpleNamespace(dir=tmp_path, meta={}), "boxes") == 7
    worker.close()
    (kwargs,) = started
    assert kwargs["env"]["PYTHONPATH"] == helper_path() and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "DISPLAY" not in kwargs["env"] and kwargs["cwd"] == tmp_path


def test_the_gpu_worker_and_the_trainer_take_their_gpu_from_here(tmp_path, monkeypatch):
    from sopa_companion.engine.analysis import AnalysisManager
    from sopa_companion.engine.qc import QcModels
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    config = {"analysis": {"python": str(tmp_path / "rapids" / "bin" / "python"), "gpu": 1},
              "qc": {"python": sys.executable, "gpu": 0},
              "paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state")}}
    probed = []
    monkeypatch.setattr("sopa_companion.engine.analysis.gpu_probe", lambda index: probed.append(index) or None)
    manager = AnalysisManager(config, runs=None, start=False)
    assert manager._worker_env()["CUDA_VISIBLE_DEVICES"] == "5" and manager.probe() is None and probed == ["5"]
    assert manager._worker_env()["PYTHONPATH"] == helper_env_path()        # the desktop: as before
    models = QcModels(config, start=False)
    monkeypatch.setattr("sopa_companion.engine.qc.gpu_probe", lambda index: probed.append(index) or None)
    # GPU "4" is not the trainer's "0" (engine/qc.py TRAINER_GPU): refused without asking nvidia-smi.
    assert models._trainer_env()["CUDA_VISIBLE_DEVICES"] == "4" and "GPU 4 here" in models.probe()["refused"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5")
    assert models._trainer_env()["CUDA_VISIBLE_DEVICES"] == "0" and models.probe() is None and probed == ["5", "0"]
    models.close()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    state = tmp_path / "state"
    api_config = {**config, "session": {"mode": "api"}}
    manager = AnalysisManager(api_config, runs=None, start=False)           # an engine configures the module itself
    env = manager._worker_env()
    assert env["PYTHONPATH"] == str(state / "pylib") and env["MPLBACKEND"] == "Agg"


def test_the_shape_workers_start_from_the_apps_own_entry(tmp_path, monkeypatch):
    from sopa_companion.engine import shape_query
    started = []

    class Popen:
        def __init__(self, command, **kwargs):
            started.append((command, kwargs))
            self.stdout = open(os.devnull, "rb")                           # not a pipe: its size stays as it is
    monkeypatch.setattr(shape_query.subprocess, "Popen", Popen)
    monkeypatch.setenv("DISPLAY", ":77")
    shape_query.ShapeQueries()._spawn().stdout.close()
    (command, kwargs), = started
    assert command[1:] == ["-m", "sopa_companion.engine.shape_query"] and kwargs["cwd"] == helper_path()
    assert kwargs["env"]["PYTHONPATH"] == helper_env_path() and "DISPLAY" not in kwargs["env"]
    state = api(tmp_path)
    shape_query.ShapeQueries()._spawn().stdout.close()
    assert started[-1][1]["cwd"] == str(state) and started[-1][1]["env"]["MPLBACKEND"] == "Agg"


# ---- an installed copy (a wheel in a Python environment) ------------------------------------------------------------
@pytest.fixture
def installed_copy(tmp_path, monkeypatch):
    """This copy taken for an installed one (no pyproject.toml beside it), with a home folder of the test's own: the
    helpers' link folder goes under it."""
    monkeypatch.setattr(dirs, "source_tree", lambda root=None: False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(children, "_said", set())
    state = tmp_path / "state"
    state.mkdir()
    children.configure({"paths": {"state_dir": str(state)}})
    return state


def test_an_installed_copy_adds_nothing_for_its_own_environment(installed_copy, monkeypatch):
    """The package is installed in the engine's own environment: a helper there needs nothing on PYTHONPATH (the
    engine's own is kept with inherit, none without), and starts in the engine's state folder, never in site-packages."""
    monkeypatch.setenv("PYTHONPATH", "/engine/own")
    assert children.env("same")["PYTHONPATH"] == "/engine/own"
    assert "PYTHONPATH" not in children.env("same", inherit=False)
    # Never the installed copy's own site-packages (an earlier desktop entry put it on the engine's PYTHONPATH): first
    # on the helper's path it would go ahead of the standard library.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT.parent), "/opt/other"]))
    assert children.env("same")["PYTHONPATH"] == "/opt/other"
    monkeypatch.setenv("PYTHONPATH", str(ROOT.parent))
    assert "PYTHONPATH" not in children.env("same")
    monkeypatch.setenv("PYTHONPATH", "/engine/own")
    assert children.app_path() == str(installed_copy)
    assert children.app_entry("same") is None                                 # a command shown names no PYTHONPATH
    children.configure({})                                                    # no engine configured it yet
    cache = Path(os.environ["XDG_CACHE_HOME"]) / "sopa-companion"
    assert children.app_path() == str(dirs.user_cache()) == str(cache) and cache.is_dir()


def test_an_installed_copy_links_other_environments_never_its_site_packages(installed_copy, tmp_path, monkeypatch,
                                                                             capsys):
    """A helper in another environment gets the link folder in the user's cache, and an engine PYTHONPATH entry that
    names the installed copy's parent (its site-packages, put there by an older desktop entry) is left out, the others
    kept. Should the link not be made: no PYTHONPATH at all, said once in the engine's log."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/kept/one", str(ROOT.parent), "/kept/two"]))
    other = children.env("other")
    entries = other["PYTHONPATH"].split(os.pathsep)
    assert entries == [str(helper_link().parent), "/kept/one", "/kept/two"] and str(ROOT.parent) not in entries
    assert helper_link().resolve() == ROOT and children.env("other", inherit=False)["PYTHONPATH"] == entries[0]
    assert children.app_entry("other") == entries[0] and children.app_path() == str(installed_copy)
    blocked = tmp_path / "home2"
    blocked.mkdir()
    (blocked / ".cache").write_text("")                                       # in the way: the link cannot be made
    monkeypatch.setenv("HOME", str(blocked))
    monkeypatch.setenv("PYTHONPATH", str(ROOT.parent))
    assert "PYTHONPATH" not in children.env("other") and "PYTHONPATH" not in children.env("other", inherit=False)
    assert children.app_entry("other") is None
    said = [line for line in capsys.readouterr().out.splitlines() if "could not be made" in line]
    assert said == [f"SoPA companion: {helper_link()} could not be made: helpers in other environments cannot import "
                    "the app"]


def test_the_sample_helper_takes_its_path_and_folder_from_here(tmp_path, monkeypatch):
    """The sopa helper (engine/zarr_samples.py) gets its environment and working folder from here: no display, the
    app's entry as before in a source tree (helper_env_path), started in app_path() rather than the engine's own
    folder, which `python -m` would put first on its path."""
    from sopa_companion.engine import zarr_samples
    started = []

    class Popen:
        def __init__(self, command, **kwargs):
            started.append((command, kwargs))
            raise OSError("not started in this test")
    monkeypatch.setattr(zarr_samples.subprocess, "Popen", Popen)
    monkeypatch.setenv("DISPLAY", ":77")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    other = tmp_path / "sopa" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)
    samples = zarr_samples.ZarrSamples({"zarr": {"python": str(other)}, "proseg": {"max_region_um": 1000.0},
                                        "paths": {"data_dir": str(tmp_path / "var"), "cache_dir": str(tmp_path / "cache")}})
    with pytest.raises(OSError, match="not started"):
        samples.helper("inspect", {"path": str(tmp_path / "sample.zarr")})
    (command, kwargs), = started
    assert command[-5:] == [str(other), "-m", "sopa_companion.zarr_bridge.helper", "inspect", "-"]
    env = kwargs["env"]
    assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env and env["OMP_NUM_THREADS"] == "16"
    assert env["PYTHONPATH"] == helper_env_path() and kwargs["cwd"] == children.app_path() == helper_path()
    assert kwargs["cwd"] != str(ROOT)


def test_the_gui_command_shown_names_the_path_only_when_needed(tmp_path, monkeypatch):
    """The Cellpose GUI's command a user can copy (engine/stain_patches.py open_gui's fallback): a source tree's
    PYTHONPATH as before; an installed copy's GUI in its own environment needs none, in another the link folder."""
    from sopa_companion.engine.patch_folders import META, Source
    from sopa_companion.engine.stain_patches import StainPatches
    assert children.app_entry("same") == children.app_entry("other") == helper_path()      # a source tree, as before
    folder = tmp_path / "patches"
    folder.mkdir(parents=True)
    (folder / "sample-1-01.tif").write_bytes(b"")
    patches = StainPatches.__new__(StainPatches)
    patches.lock, patches.processes, patches.launcher = __import__("threading").RLock(), {}, "x"
    patches.window = None
    patches.stain = SimpleNamespace(python=sys.executable, gpu=0)
    patches.require = lambda run: None
    patches.draft = lambda run: {}
    patches.selected = lambda run, name, draft=None: Source(folder, {"patches": [{"image": "sample-1-01.tif"}]},
                                                            folder / META, True)
    patches.gui_record = lambda source: None
    patches.running = lambda source, record: False
    run = SimpleNamespace(lock=__import__("threading").RLock())
    shown = patches.open_gui(run, {})["command"]
    assert shown.startswith(f"env PYTHONPATH={helper_path()} PYTHONSAFEPATH=1 CUDA_VISIBLE_DEVICES=0 ")
    monkeypatch.setattr(dirs, "source_tree", lambda root=None: False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    shown = patches.open_gui(run, {})["command"]
    assert shown.startswith("env PYTHONSAFEPATH=1 CUDA_VISIBLE_DEVICES=0 ") and "PYTHONPATH=" not in shown   # its own
    patches.stain = SimpleNamespace(python=str(tmp_path / "other" / "bin" / "python"), gpu=0)
    shown = patches.open_gui(run, {})["command"]
    assert shown.startswith(f"env PYTHONPATH={helper_link().parent} PYTHONSAFEPATH=1 ")      # another one: the link
