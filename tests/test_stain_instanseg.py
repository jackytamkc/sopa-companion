"""Method selection, cache safety and the inference boundary without GPU/network access."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile
from contextlib import nullcontext

import numpy as np
import pytest

from sopa_companion.engine.runs import RunError
from sopa_companion.zarr_bridge import stain_instanseg as I
from test_stain import runs, ended  # shared isolated engine fixture
from conftest import needs_sopa, sopa_python, wait_for

SETTINGS = dict(method="instanseg", image="image", channels=["DAPI", "Boundary"])


@pytest.fixture
def manager(runs):
    runs.stain.gpu = ""
    runs.stain.instanseg_enabled = True          # InstanSeg is dormant unless its internal switch is on
    runs.stain.instanseg_python = sys.executable
    runs.stain.instanseg_module = "sopa_companion.tests.fake_instanseg"
    return runs


def test_instanseg_settings_scale_and_environment(manager):
    manager.zarr.image["levels"][0]["pixel"] = [.25, .25]
    manager.zarr.image["to_frame"][0][0] = manager.zarr.image["to_frame"][1][1] = 2
    result = manager.create("image-sample", kind="stain", region={"kind": "full"},
        stain={**SETTINGS, "pixel_size": 99, "model_path": "/untrusted", "model_sha256": "untrusted"})
    settings = result["stain"]
    # scale0 (the image as stored) unless chosen: the model's pixel size is that level's, 0.25 px times 2 µm per px.
    assert settings == dict(**SETTINGS, model="fluorescence_nuclei_and_cells", model_kind="builtin",
                           output="cells", resolve_cells=False, pixel_size=.5, tissue_only=True, level=0,
                           resolution=dict(level=0, factor=1.0, um_per_px=.5, um_per_px_y=.5, label="scale0"))
    run = manager.get(result["id"])
    assert run.process is None and run.meta["stain_started"] is False
    job = json.loads((run.dir / "stain/job.json").read_text())
    assert job["instanseg_module"] == "sopa_companion.tests.fake_instanseg"
    env = manager.stain.helper_env(run.dir / "cache")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and "DISPLAY" not in env
    for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
        assert env[key] == str(manager.root.parent / "instanseg-models")
    assert len(manager.stain.models("cellpose")) == 4 and len(manager.stain.models("instanseg")) == 3


@pytest.mark.parametrize("change", [
    {"channels": []}, {"channels": ["missing"]}, {"channels": ["DAPI", "DAPI"]}, {"channels": "DAPI"},
    {"model": "../weights"}, {"output": "both"}, {"resolve_cells": 1},
    {"model": "single_channel_nuclei"}, {"model": "brightfield_nuclei"},
    {"model": "single_channel_nuclei", "channels": ["DAPI"], "output": "cells"},
    {"model": "single_channel_nuclei", "channels": ["DAPI"], "resolve_cells": True},
])
def test_invalid_instanseg_choices_create_no_run(manager, change):
    with pytest.raises(RunError):
        manager.create("image-sample", kind="stain", region={"kind": "full"}, stain={**SETTINGS, **change})
    assert not manager.runs and not list(manager.root.iterdir())


def test_instanseg_nuclei_defaults_and_unavailable_environment(manager):
    value = manager.stain.settings(manager.zarr.info,
        {**SETTINGS, "model": "single_channel_nuclei", "channels": ["DAPI"]})
    assert value["output"] == "nuclei"
    manager.zarr.image["levels"][0]["pixel"] = [.5, 1]
    with pytest.raises(RunError, match="equal pixel sizes"):
        manager.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
    manager.stain.instanseg_python = "/missing/python"
    assert not manager.stain.method_available("instanseg")
    assert manager.stain.method_available("cellpose")
    with pytest.raises(RunError, match="unavailable"):
        manager.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
    assert not manager.runs


def test_instanseg_engine_live_outlines_and_save_provenance(manager, monkeypatch):
    from sopa_companion.engine import vectors, runlog
    run = manager.get(manager.create("image-sample", kind="stain", region={"kind": "full"},
                                     stain=SETTINGS)["id"])
    manager.control(run.meta["id"], "resume")
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    outlines = vectors.decode(manager.stain.outline_bytes(run))
    assert outlines["source"]["cells"] == "InstanSeg"
    wait_for(lambda: ended(run))
    assert run.meta["save_defaults"]["shapes_key"] == "instanseg_boundaries_1"
    log = runlog.Text()
    runlog.summary_part(log, SimpleNamespace(runs=manager), run.meta["id"])
    assert "Stain segmentation · InstanSeg" in "\n".join(log.lines)
    saved = []
    monkeypatch.setattr(manager.zarr, "existing_elements", lambda sid: [], raising=False)
    monkeypatch.setattr(manager.zarr, "save",
                        lambda *args, **kw: saved.append(args[6]) or {"cells": 6}, raising=False)
    manager.save_to_zarr(run.meta["id"], "instanseg_shapes", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving")
    assert run.meta["zarr_save"]["state"] == "saved"
    assert saved[0]["method"] == "instanseg"
    assert saved[0]["parameters"]["channels"] == SETTINGS["channels"]
    assert saved[0]["model"] == saved[0]["runtime"]["model"] == I.MODEL_NAMES[0]


def test_save_defaults_use_method_and_skip_names_without_case_collisions(config, monkeypatch):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    registry = ZarrSamples(config)
    monkeypatch.setattr(registry, "existing_elements", lambda sid: ["INSTANSEG_boundaries_1", "cellpose_boundaries_1"])
    assert registry.save_defaults("sample", kind="stain", method="instanseg")["shapes_key"] == "instanseg_boundaries_2"
    assert registry.save_defaults("sample", kind="stain")["shapes_key"] == "cellpose_boundaries_2"
    assert registry.save_defaults("sample")["shapes_key"] == "proseg_boundaries"


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_instanseg_patches_gui_correction(manager, channels, chosen):
    from sopa_companion.engine.stain_worker import StainWorker
    from test_stain_patches import correct

    sopa = sopa_python()                         # skipped before the engine's worker is touched
    names = ["DAPI","Boundary","Third","Fourth","Fifth"][:channels]
    manager.zarr.image["channels"] = names
    manager.zarr.image["levels"][0]["shape"][0] = channels
    manager.stain.worker.close()
    manager.stain.worker = StainWorker(sopa, "sopa_companion.tests.fake_stain_patches")
    run = manager.get(manager.create("image-sample", kind="stain", region={"kind":"full"},
                                    stain={**SETTINGS,"channels":names})["id"])
    manager.control(run.meta["id"], "resume")
    wait_for(lambda: ended(run))
    patches = manager.stain.patches
    value = patches.change(run, {"version":0,"boxes":[[0,0,128,128]]})
    value = patches.export(run, {"version":value["version"], "folder": str(chosen / "patches")})
    folder = Path(value["exported"]["folder"])
    assert value["exported"]["method"] == "instanseg" and value["exported"]["channels"] == names
    patches.launcher = "sopa_companion.tests.fake_stain_gui"
    assert patches.open_gui(run, {"set":value["set"]})["running"]
    patches.processes[str(folder)].wait(timeout=5)
    assert json.loads((folder/"gui-launch.json").read_text())["display"] is None
    correct(folder, value["exported"]["patches"][0]["mask"])
    assert patches.status(run)["exported"]["corrected"] == 1


def test_instanseg_model_route_and_health(config):
    from sopa_companion.server.app import create_app
    config["stain"]["python"] = sopa_python()       # InstanSeg counts only beside the sopa environment (health)
    config["stain"]["instanseg_enabled"] = True     # and only while its internal switch is on (dormant otherwise)
    app = create_app(config)
    stain = app.state.runs.stain
    stain.gpu = ""
    stain.instanseg_python, stain.instanseg_module = sys.executable, "sopa_companion.tests.fake_instanseg"
    try:
        routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
        result = routes["/api/v1/stain/models"](method="instanseg")
        assert [m["name"] for m in result["models"]] == list(I.MODEL_NAMES)
        assert all("path" not in m for m in result["models"])
        with pytest.raises(RunError, match="^Choose Cellpose or InstanSeg$"):
            routes["/api/v1/stain/models"](method="unknown")
        assert routes["/api/v1/health"]()["stain"]["methods"] == {"cellpose": True, "instanseg": True}
    finally:
        app.state.qc.close()
        app.state.deeper.close()
        app.state.analysis.close()
        app.state.zarr.close()
        app.state.runs.close(save=False)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root, out = tmp_path / "models", tmp_path / "run"
    (out / "stain").mkdir(parents=True)
    for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
        monkeypatch.setenv(key, str(root))
    monkeypatch.setattr(I, "model_index", lambda: [dict(name=I.MODEL_NAMES[0], version="0.1.1",
        url="https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/model.zip")])
    events = []
    monkeypatch.setattr(I, "emit", lambda ev, **kw: events.append((ev, kw)))
    return root, out, events


def archive():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as file:
        file.writestr("instanseg.pt", b"fake weights")
        file.writestr("../outside", b"must not extract")
    return stream.getvalue()


def test_instanseg_download_staged_cached_and_hash_pinned(cache, monkeypatch):
    root, out, events = cache
    request = Mock(side_effect=lambda *a, **kw: io.BytesIO(archive()))
    monkeypatch.setattr(I.urllib.request, "urlopen", request)
    record = I.model_file(dict(model=I.MODEL_NAMES[0]), out, lambda p: p.read_bytes())
    assert record["model_sha256"] == hashlib.sha256(b"fake weights").hexdigest()
    assert Path(record["model_path"]).is_relative_to(root)
    assert events == [("progress", {"message": "Downloading model…"})]
    assert not (root / "outside").exists()
    assert not list(root.glob(".fluorescence_nuclei_and_cells-*"))
    assert request.call_args.kwargs["timeout"] == 30
    (out / "stain/model.json").write_text(json.dumps(record))
    assert I.model_file(dict(model=I.MODEL_NAMES[0]), out, Mock()) == record
    assert request.call_count == 1
    Path(record["model_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file has changed"):
        I.model_file(dict(model=I.MODEL_NAMES[0]), out, Mock())


@pytest.mark.parametrize("failure", ["offline", "zip", "weights"])
def test_instanseg_download_failure_is_retryable_and_leaves_no_weights(cache, monkeypatch, failure):
    root, out, events = cache
    request = Mock(side_effect=OSError("offline")) if failure == "offline" else Mock(
        return_value=io.BytesIO(b"broken zip" if failure == "zip" else archive()))
    monkeypatch.setattr(I.urllib.request, "urlopen", request)
    load = Mock(side_effect=RuntimeError("bad weights"))
    with pytest.raises(RuntimeError, match="Check the connection"):
        I.model_file(dict(model=I.MODEL_NAMES[0]), out, load)
    assert not list(root.rglob("*.pt")) and not list(root.glob(".fluorescence_nuclei_and_cells-*"))
    monkeypatch.setattr(I.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(archive()))
    assert Path(I.model_file(dict(model=I.MODEL_NAMES[0]), out, Mock())["model_path"]).is_file()


@pytest.mark.parametrize("gpu", [True, False])
def test_instanseg_inference_falls_back_once_and_propagates_cpu_errors(monkeypatch, gpu):
    model = object.__new__(I.InstanSeg)
    model.gpu = gpu
    model.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
    model.load = Mock(return_value="cpu model")
    model.evaluate = Mock(side_effect=[RuntimeError("CUDA out of memory"), np.ones((2, 2), np.uint32)])
    events = []
    monkeypatch.setattr(I, "emit", lambda ev, **kw: events.append((ev, kw)))
    if gpu:
        assert model(np.zeros((2, 2, 4))).shape == (2, 2)
        assert model.evaluate.call_count == 2 and model.gpu is False
        assert events[0][0] == "fallback" and model.model == "cpu model"
    else:
        with pytest.raises(RuntimeError, match="out of memory"):
            model(np.zeros((2, 2, 4)))
        model.load.assert_not_called()


@pytest.mark.parametrize("output,resolve", [("cells", False), ("nuclei", False), ("cells", True), ("nuclei", True)])
def test_instanseg_channel_layout_output_and_resolving(monkeypatch, output, resolve):
    class Tensor:
        def __init__(self, array): self.array = np.asarray(array)
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.array, axis))
        def __getitem__(self, key): return Tensor(self.array[key])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.array

    model = object.__new__(I.InstanSeg)
    model.settings = dict(model=I.MODEL_NAMES[0], output=output, resolve_cells=resolve, pixel_size=.2125)
    model.torch = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
    labels = np.full((1, 2 if resolve else 1, 6, 17), 3)
    if resolve:
        labels[:, 1] = 8
    model.model = SimpleNamespace(eval_medium_image=Mock(return_value=Tensor(labels)))
    masks = model.evaluate(np.ones((1, 17, 5), np.float32))
    assert masks.shape == (1, 17) and masks.dtype == np.uint32
    assert (masks == (8 if resolve and output == "cells" else 3)).all()
    args, kwargs = model.model.eval_medium_image.call_args
    assert args[0].array.shape == (1, 5, 6, 17)
    assert kwargs == dict(pixel_size=.2125, target="all_outputs" if resolve else output, return_image_tensor=False,
                          resolve_cell_and_nucleus=resolve)


def test_instanseg_load_failure_uses_cpu_and_records_model(cache, monkeypatch):
    _, out, _ = cache
    record = dict(model=I.MODEL_NAMES[0], model_path="/fake/weights", model_sha256="abc")
    monkeypatch.setattr(I, "model_file", lambda *a: record)
    monkeypatch.setattr(I.importlib.metadata, "version", lambda name: "test")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=Mock()),
                            jit=SimpleNamespace(load=Mock(return_value="weights")))
    factory = Mock(side_effect=[RuntimeError("CUDA unavailable"), "cpu model"])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "instanseg", SimpleNamespace(InstanSeg=factory))
    model = I.InstanSeg({}, out)
    assert not model.gpu and model.model == "cpu model"
    assert [call.kwargs["device"] for call in factory.call_args_list] == ["cuda:0", "cpu"]
    assert json.loads((out / "stain/model.json").read_text()) == record


@needs_sopa
def test_instanseg_real_tiles_with_fake_inference_process(tmp_path, monkeypatch):
    for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
        monkeypatch.setenv(key, str(tmp_path / "models"))
    # Keep the existing protocol/seam check; replace its GPU-1-only double with our CPU double.
    code = ("import runpy, sys\n"
            "from sopa_companion.zarr_bridge.stain_instanseg import InstanSegProcess\n"
            "original = InstanSegProcess.__init__\n"
            "def init(self, job):\n"
            "    job['instanseg_module'] = 'sopa_companion.tests.fake_stain_patches'\n"
            "    from sopa_companion.zarr_bridge.stain import atomic_json\n"
            "    from pathlib import Path\n"
            "    atomic_json(Path(job['out'])/'stain/job.json', job)\n"
            "    original(self, job)\n"
            "InstanSegProcess.__init__ = init\n"
            "sys.argv = sys.argv[1:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    result = subprocess.run([sopa_python(), "-c", code,
                            str(Path(__file__).with_name("instanseg_tiles_check.py")), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
             "NUMBA_CACHE_DIR": str(tmp_path / "numba"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "XDG_CACHE_HOME": str(tmp_path / "cache"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pause/stop/resume, seams and pixel scale passed" in result.stdout


@pytest.mark.parametrize("model_pixel_size", [.5, .25])
def test_instanseg_level_pixel_size_and_tile_cap(model_pixel_size):
    from sopa_companion.zarr_bridge.stain import check_tile, resolution_info, tile_layout
    # Levels as inspect lists them (dims and shape: engine/stain_levels.py offers only levels it can read).
    image = dict(to_frame=np.eye(3).tolist(), levels=[
        dict(path=f"images/i/{k}", shape=[1, 4096 // f, 4096 // f], dims=["c", "y", "x"],
             pixel=[.2125*f, .2125*f*(1.0002 if f > 1 else 1)], origin=[0, 0]) for k, f in enumerate((1, 2, 4, 8))])
    for k in range(4):
        job = dict(settings=dict(method="instanseg", level=k, pixel_size=99), image=image,
                   tile_pixels=1024, overlap_pixels=128)
        size, overlap = tile_layout(job, model_pixel_size)
        info = resolution_info(job)
        scale = info["um_per_px"] / model_pixel_size
        assert size <= 1024 and (np.ceil(size * scale))**2 <= 1024**2
        assert overlap <= size // 2
        check_tile((size, size), dict(method="instanseg", pixel_size=info["um_per_px"]),
                   model_pixel_size=model_pixel_size)
        if k == 3:
            assert size < 310
            with pytest.raises(ValueError, match="tile is too large"):
                check_tile((1024, 1024), dict(method="instanseg", pixel_size=info["um_per_px"]),
                           model_pixel_size=model_pixel_size)
