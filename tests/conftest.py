"""Shared fixtures: a tiny synthetic Xenium bundle, a fake proseg-live, and an isolated engine configuration; and where
the few tests that need more than the app's venv find it (the sopa environment, the measuring harness's folder)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # the folder containing the sopa_companion package


# ---- what only some computers have -----------------------------------------------------------------------------------
# A few tests run a real helper, fit or Cellpose in the sopa environment, and a few read the measuring harness's own
# folder (its crop of a sample, the measured calibrations, models and runs). Both are found here, once,
# when the tests start (a test may point HOME elsewhere); a test that needs one that is not there is skipped and says
# what to set. Everything else needs only the app's venv. On a new computer:
#   SOPA_COMPANION_SOPA_PYTHON   the sopa environment's python; else the one this computer's settings give the app
#                                ([zarr], [stain] or [qc] python in config.toml or config.local.toml)
#   SOPA_COMPANION_HARNESS_DIR   the harness's folder (fixtures/, resolvi-prep/, var/); nothing else stands for it
#   SOPA_COMPANION_API_PYTHON    an environment made as API mode's users make theirs (`pip install sopa`, then the
#                                companion's requirements): the packaging checks read its versions and sopa itself
SOPA_VAR, HARNESS_VAR, API_VAR = "SOPA_COMPANION_SOPA_PYTHON", "SOPA_COMPANION_HARNESS_DIR", "SOPA_COMPANION_API_PYTHON"


def configured(*keys: tuple[str, str]) -> str:
    """The first of these settings, as (section, key), that this computer's settings fill in with a file that is there;
    "" when none is (a new clone's config.example.toml leaves every environment empty)."""
    try:
        from sopa_companion.engine.config import load_config
        config = load_config()
    except (OSError, ValueError, KeyError, TypeError):    # settings that do not load: none given
        return ""
    values = ((config.get(section) or {}).get(key) for section, key in keys)
    return next((value for value in values if isinstance(value, str) and value and Path(value).is_file()), "")


SOPA = (os.path.expanduser(os.environ.get(SOPA_VAR) or "") or
        configured(("zarr", "python"), ("stain", "python"), ("qc", "python")))
NO_SOPA = (f"no sopa environment here: {SOPA_VAR} names no file" if os.environ.get(SOPA_VAR) else
           f"no sopa environment here: set {SOPA_VAR}, or [zarr] python in config.toml")
INSTANSEG = configured(("stain", "instanseg_python"))      # the opt-in InstanSeg smoke's own environment
HARNESS = os.path.expanduser(os.environ.get(HARNESS_VAR) or "")
API = os.path.expanduser(os.environ.get(API_VAR) or "")


def sopa_python(skip: bool = True) -> str | None:
    """The sopa environment's python (spatialdata, sopa, Cellpose, the trainer's torch). Not on this computer: the test
    is skipped, saying so; with skip=False, None (a check made only where it can be)."""
    if SOPA and Path(SOPA).is_file():
        return SOPA
    if skip:
        pytest.skip(NO_SOPA)
    return None


def api_python(skip: bool = True) -> str | None:
    """The python of an environment made as API mode's users make theirs ($SOPA_COMPANION_API_PYTHON): not given, or not
    there, the test is skipped, saying what to set; with skip=False, None. Never the sopa environment the app runs its
    helpers in (that one can hold another sopa, even an editable checkout)."""
    if API and Path(API).is_file():
        return API
    if skip:
        pytest.skip(f"no API environment here: set {API_VAR} to the python of an environment made with `pip install sopa`")
    return None


def instanseg_python(skip: bool = True) -> str | None:
    """InstanSeg's own environment, as this computer's settings give it ([stain] instanseg_python); else skipped."""
    if INSTANSEG:
        return INSTANSEG
    if skip:
        pytest.skip("no InstanSeg environment here: set [stain] instanseg_python in config.toml")
    return None


# A mark for the tests that need the sopa environment from their first line: skipped before their fixtures run.
needs_sopa = pytest.mark.skipif(sopa_python(skip=False) is None, reason=NO_SOPA)


def harness_path(*parts: str, skip: bool = True) -> Path | None:
    """A file or folder in the measuring harness's folder ($SOPA_COMPANION_HARNESS_DIR/parts...), only ever read. Not
    given, or not in it: the test is skipped, saying which; with skip=False, None."""
    path = Path(HARNESS, *parts) if HARNESS else None
    if path is not None and path.exists():
        return path
    if skip:
        pytest.skip(f"not in the harness folder: {'/'.join(parts)}" if HARNESS else
                    f"no harness folder here: set {HARNESS_VAR}")
    return None


def harness_crop(skip: bool = True) -> Path | None:
    """The harness's crop of a real sample, by the name its fixtures/fixtures.json records (the harness makes it; tests
    only read it). Not made yet, or no harness folder: skipped (skip=False: None)."""
    registry = harness_path("fixtures", "fixtures.json", skip=skip)
    try:
        name = Path(json.loads(registry.read_text())["crop"]["path"]).name if registry else ""
    except (OSError, ValueError, KeyError, TypeError):    # a registry without a crop
        name = ""
    if name:
        return harness_path("fixtures", name, skip=skip)
    if skip:
        pytest.skip("the harness has made no crop (fixtures/fixtures.json)")
    return None


def write_xenium_bundle(root: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tifffile

    folder = root / "output-XETG00000__0000000__TESTREGION-1__20260101__000000"
    (folder / "morphology_focus").mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 400, n).astype(np.float32)
    y = rng.uniform(0, 300, n).astype(np.float32)
    cells = [f"c{int(a // 40)}{int(b // 40)}-1" if rng.random() < .7 else "UNASSIGNED" for a, b in zip(x, y)]
    table = pa.table({
        "transcript_id": pa.array(np.arange(n, dtype=np.uint64)),
        "cell_id": pa.array(cells, pa.large_string()),
        "overlaps_nucleus": pa.array(rng.integers(0, 2, n).astype(np.uint8)),
        "feature_name": pa.array([f"G{i % 7}" for i in range(n)], pa.large_string()),
        "x_location": pa.array(x), "y_location": pa.array(y),
        "z_location": pa.array(rng.uniform(5, 15, n).astype(np.float32)),
        "qv": pa.array(np.full(n, 40, np.float32)),
        "fov_name": pa.array(["A1"] * n, pa.large_string()),
        "nucleus_distance": pa.array(np.zeros(n, np.float32)),
        "codeword_index": pa.array(np.zeros(n, np.int32)),
        "codeword_category": pa.array(["predesigned_gene"] * n, pa.large_string()),
        "is_gene": pa.array(np.ones(n, bool)),
    })
    pq.write_table(table, folder / "transcripts.parquet")
    (folder / "experiment.xenium").write_text(json.dumps({"region_name": "TESTREGION-1", "num_cells": 70}))
    # A 2-channel pyramidal OME-TIFF, 0.2125 um/px: 400 x 300 um.
    width, height = int(400 / 0.2125), int(300 / 0.2125)
    data = (rng.random((2, height, width)) * 1000).astype(np.uint16)
    with tifffile.TiffWriter(folder / "morphology_focus" / "ch0000_dapi.ome.tif", ome=True) as tif:
        meta = {"axes": "CYX", "PhysicalSizeX": 0.2125, "PhysicalSizeY": 0.2125, "Channel": {"Name": ["DAPI", "Boundary"]}}
        tif.write(data, subifds=2, tile=(256, 256), metadata=meta)
        for level in (1, 2):
            tif.write(data[:, :: 2 ** level, :: 2 ** level], subfiletype=1, tile=(256, 256))
    return folder


@pytest.fixture(scope="session")
def xenium_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("xenium")
    write_xenium_bundle(root)
    return root


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "fake-proseg"
    path.write_text(f"#!{sys.executable}\nimport runpy, sys\nrunpy.run_path({str(HERE / 'fake_proseg.py')!r}, run_name='__main__')\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def old_fake_binary(tmp_path):
    """A proseg-live 3.2.0 from before proseg's own counts (the fake with FAKE_PROSEG_OLD=1): its --help lacks
    --live-counts, it refuses those flags, and it ignores the counts and where control lines."""
    path = tmp_path / "fake-proseg-3.2.0"
    path.write_text(f"#!{sys.executable}\nimport os, runpy\nos.environ['FAKE_PROSEG_OLD'] = '1'\n"
                    f"runpy.run_path({str(HERE / 'fake_proseg.py')!r}, run_name='__main__')\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def no_desktop(monkeypatch):
    # Never let a test pop a dialog on the user's desktop.
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


@pytest.fixture(autouse=True)
def no_retired_variables(monkeypatch):
    # The previous version's variables (PROSEG_LIVE_*) are not read, and its private lock ones are refused when set
    # alone (names.retired_isolation): a shell that still sets one changes no test.
    for name in [name for name in os.environ if name.startswith("PROSEG_LIVE_")]:
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def private_zarr_locks(monkeypatch, tmp_path_factory, no_retired_variables):
    # Saves into throwaway sample copies take their locks here, not in the folders shared with the user's engines
    # (~/.cache/proseg_live/zarr-locks and ~/.cache/sopa-companion/zarr-locks); the engines tests start prune this
    # folder, never the user's.
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(tmp_path_factory.mktemp("zarr-locks")))


def outside(prefix: str):
    """A folder of a test's own in the system's temporary folder (a generator: the folder, then its removal, whatever
    modes the test left in it). Training patches go into folders like it: the export dialog refuses hidden folders, and
    pytest's own temporary folders may be inside one (the gate keeps them under ~/.local)."""
    folder = Path(tempfile.mkdtemp(prefix=prefix))
    yield folder
    for path in [folder, *folder.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(autouse=True)
def private_patches_home(monkeypatch):
    # The export dialog's default place for training folders (engine/patch_folders.py default_home), which the engine
    # also reads for earlier sets: a test's own, never the user's ~/sopa-companion.
    for folder in outside("patches-home-"):
        monkeypatch.setenv("SOPA_COMPANION_PATCHES_HOME", str(folder))
        yield folder


@pytest.fixture
def chosen():
    """A folder a test exports training patches into (outside)."""
    yield from outside("patch-folders-")


# An installed copy's runs and caches in the user's home folder (dirs.data_dir), as the tests find it when they start.
USER_DATA = Path.home() / ".local" / "share" / "sopa-companion"


@pytest.fixture(autouse=True)
def private_user_folders(monkeypatch, tmp_path_factory):
    # A copy without pyproject.toml counts as installed (dirs.source_tree), and an installed copy reads your own settings
    # file and keeps its runs in your data folder: a test's fake app folder must never read the user's
    # ~/.config/sopa-companion/config.toml nor write ~/.local/share/sopa-companion. The data folder is fixed in the home
    # folder (whatever $XDG_DATA_HOME says): a test that reaches the user's own fails instead, and tests of it give it a
    # home folder of their own.
    folder = tmp_path_factory.mktemp("user")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(folder / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(folder / "data"))
    from sopa_companion import dirs
    real = dirs.data_dir

    def guarded(root=None):
        found = real(root)
        if Path(found) == USER_DATA:
            raise AssertionError(f"a test reached the user's own data folder, {USER_DATA}: give it a home of its own")
        return found
    monkeypatch.setattr(dirs, "data_dir", guarded)


@pytest.fixture(autouse=True)
def private_counts_spool(monkeypatch, tmp_path_factory):
    # Proseg's own counts the engines tests start ask for wait in a spool (engine/counts_spool.py): one of the tests'
    # own, never the RAM disk the user's engines share (and sweep).
    from sopa_companion.engine.counts_spool import ROOT_ENV
    monkeypatch.setenv(ROOT_ENV, str(tmp_path_factory.mktemp("counts-spool")))


@pytest.fixture(autouse=True)
def private_fit_lock(monkeypatch, tmp_path_factory):
    # The mixing check's computer-wide fit locks (engine/qc.py fit_lock_paths; this file alone when it is named): tests
    # learn with fake trainers, and must neither wait for a real fit of the user's engine nor hold up one.
    monkeypatch.setenv("SOPA_COMPANION_QC_FIT_LOCK", str(tmp_path_factory.mktemp("qc-fit") / "qc-fit.lock"))


@pytest.fixture(autouse=True)
def no_real_state_move(monkeypatch):
    # The previous version's state folder is moved into this version's at the default desktop instance's first start
    # (desktop/migrate.py), in the home folder. A test that reaches it with the default instance's settings must never
    # move the user's real folders: it fails instead. Tests of the move give it a home folder of their own. The notebook
    # side's tests (tests/api) run where the engine's code cannot be imported, nor the move reached.
    # Nor on Windows: the desktop route is not for it (its move takes file locks with fcntl), so nothing is moved there.
    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("fcntl") is None:
        return
    from sopa_companion.desktop import migrate
    real = migrate.migrate

    def guarded(config, *args, home=None, **kwargs):
        if home is None and migrate.default_instance(config):
            raise AssertionError("a test reached the state folder's move with the user's own home folder")
        return real(config, *args, home=home, **kwargs)
    monkeypatch.setattr(migrate, "migrate", guarded)


@pytest.fixture(autouse=True)
def desktop_helpers():
    # The helpers' mode is the process's (engine/children.py configure: the engine made last sets it): an API-mode engine
    # made in one test never leaves it set for the tests after it.
    yield
    from sopa_companion.engine import children
    children.configure({})


@pytest.fixture
def config(tmp_path, xenium_root, fake_binary):
    from sopa_companion.engine.config import load_config
    return load_config(overrides={
        "server": {"port": 8799},
        # Closing a test websocket must not SIGTERM pytest. The app's own window (Chrome), as this workstation's
        # config.toml has it ("auto"): without that file (a fresh checkout, the CI) the shipped default would open the
        # desktop's web browser instead; a test of that names it itself.
        "desktop": {"exit_when_closed": False, "browser": "chrome"},
        "paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state"), "sample_roots": [str(xenium_root)]},
        "proseg": {"binary": str(fake_binary), "threads": 2},
        # No analysis worker: Clusters & UMAP are always on, so with config.toml's RAPIDS python every run a test
        # starts would start the real worker on GPU 1. Tests that analyse set the fake worker themselves.
        "analysis": {"python": "off"},
        # The mixing check's probe (engine/qc.py QcModels) runs at every engine's start in [qc] python, this computer's
        # sopa environment here: its fake answers instead, so no test imports torch there or starts the GPU driver.
        "qc": {"probe_module": "sopa_companion.tests.fake_qc_probe"},
    })


def wait_for(predicate, timeout=20.0, interval=0.02):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    raise AssertionError("condition not reached in time")
