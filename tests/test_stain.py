"""Stain engine/API lifecycle with a real subprocess double; no Cellpose required."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from sopa_companion.engine.runs import RunError, Runs
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples

IMAGE = {"key": "image", "channels": ["DAPI", "Boundary"], "to_frame": np.eye(3).tolist(),
         "levels": [{"path": "images/image/scale0/image", "shape": [2, 300, 400], "dims": ["c", "y", "x"],
                     "origin": [0, 0], "pixel": [1, 1]}]}
SETTINGS = dict(image="image", nucleus="DAPI", cytoplasm="Boundary", diameter=None,
                flow_threshold=.4, cellprob_threshold=0)


@pytest.fixture(autouse=True)
def stain_cpu(monkeypatch):
    """Even the subprocess doubles run without a visible GPU (a shared computer's GPUs are others' too)."""
    from sopa_companion.engine.stain import StainRuns
    original = StainRuns.helper_env
    monkeypatch.setattr(StainRuns, "helper_env",
                        lambda self, cache, *pythons: {**original(self, cache, *pythons), "CUDA_VISIBLE_DEVICES": ""})


class Images:
    _geometry = staticmethod(ZarrSamples._geometry)
    _window = ZarrSamples._window

    def __init__(self, path):
        self.image = copy.deepcopy(IMAGE)
        self.info = dict(id="image-sample", name="Test", kind="zarr", path=str(path), extent=[0, 0, 400, 300],
                         images=[self.image], transcripts=50)

    def known(self, sid): return sid == "image-sample"
    def detail(self, sid): return self.info
    def _image_info(self, sid, key): return self.image
    def _stats(self, *args): return [dict(name=c) for c in self.image["channels"]]
    def tissue(self, sid, key=None):     # tissue over the whole test sample: Tissue only changes nothing here
        ring = [[-1, -1], [401, -1], [401, 301], [-1, 301], [-1, -1]]
        return dict(source="region_of_interest", polygons=[ring], shapes=[[ring]], area_um2=402. * 302.)
    def save_defaults(self, sid, kind="proseg", method="cellpose"):
        return dict(shapes_key=f"{method}_boundaries_1", table_key=None, existing=[], set_default=True)


@pytest.fixture
def runs(config, tmp_path):
    # A private helper and tile configuration, independent of this computer's settings.
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", tile_pixels=128,
                           overlap_pixels=16, gpu=1)
    manager = Runs(config, Samples(config), Images(tmp_path / "unwritten.zarr"))
    yield manager
    manager.close(save=False)


def start(runs, resume=True, **args):
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"},
                              stain=SETTINGS, **args)["id"])
    if resume:
        runs.control(run.meta["id"], "resume")
    return run


def ended(run):
    return run.meta["state"] in ("done", "stopped", "failed") and not (run.worker and run.worker.is_alive())


def test_stain_tiles_pause_stop_resume_and_delete(runs):
    from sopa_companion.engine import vectors
    run = start(runs)
    rid = run.meta["id"]
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    runs.control(rid, "pause")
    wait_for(lambda: run.ack and run.ack["state"] == "paused")
    n = run.meta["progress"]["done"]
    before = (run.dir / "stain/tile-000000.masks.npz").read_bytes()
    time.sleep(.25)
    assert run.meta["progress"]["done"] == n < 6
    assert run.public()["phase"] == "paused"
    live = vectors.decode(runs.stain.outline_bytes(run))
    assert len(live["cells"]["offsets"])-1 == n
    with pytest.raises(RunError):
        runs.delete(rid)
    runs.control(rid, "stop")
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "stopped" and run.meta["progress"]["resolved"]
    assert run.meta["summary"]["cells"] == n
    revision = run.meta["progress"]["revision"]
    runs.control(rid, "resume")
    assert run.meta["progress"]["revision"] > revision and not run.meta["progress"]["resolved"]
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done" and run.meta["summary"]["cells"] == 6
    assert (run.dir / "stain/tile-000000.masks.npz").read_bytes() == before
    packed = vectors.decode(runs.stain.outline_bytes(run, [9, 9, 13, 13]))
    assert len(packed["cells"]["offsets"])-1 == 1
    empty = vectors.decode(runs.stain.outline_bytes(run, [200, 200, 300, 300]))
    assert len(empty["cells"]["offsets"]) == 1
    assert run.meta["stain_runtime"] == {**run.meta["stain_runtime"], "cuda_visible_devices": "", "nice": 19, "display": None}
    assert run.meta["last_tile"]["tile"] == 5
    assert next(r for r in runs.list() if r["id"] == rid)["kind"] == "stain"
    with pytest.raises(RunError):
        runs.control(rid, "resume")
    runs.delete(rid)
    assert not run.dir.exists()


def test_closing_the_app_stops_a_stain_run_where_it_is(runs, config, monkeypatch):
    """A stain run with finished tiles is a result not yet in the sample, listed with the others
    (Runs.pending, the close popup's one list). Closing with no answer discards it: the helper is stopped where it is,
    never finished as if it were complete, and the run is removed. A test engine (keep_unsaved) stops it the same way
    whatever the answer and keeps its finished tiles, to resume."""
    monkeypatch.setenv("FAKE_STAIN_DELAY", ".4")
    run = start(runs)
    wait_for(lambda: run.meta["state"] == "running" and run.process is not None and run.meta["progress"]["done"] >= 1)
    (row,) = runs.pending()
    assert row["id"] == run.meta["id"] and row["kind"] == "stain" and row["can_add"]
    runs.close(timeout=30)
    assert run.meta["state"] == "stopped" and not run.meta.get("finish_requested")
    assert run.meta["progress"]["done"] < 6 and not run.dir.exists() and not runs.runs
    config["proseg"]["keep_unsaved"] = True
    kept = Runs(config, runs.samples, runs.zarr)
    run = start(kept)
    wait_for(lambda: run.meta["state"] == "running" and run.process is not None and run.meta["progress"]["done"] >= 1)
    kept.close(save=True, timeout=30)
    assert run.meta["state"] == "stopped" and not run.meta.get("finish_requested")
    assert run.meta["progress"]["done"] < 6 and run.dir.exists()


def test_stain_failure_and_restart_keeps_committed_tiles(runs, config, monkeypatch):
    monkeypatch.setenv("FAKE_STAIN_FAIL", "2")
    run = start(runs)
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "failed" and run.meta["progress"]["done"] == 2
    assert "Fake tile failure" in run.meta["message"]
    before = (run.dir / "stain/tile-000000.npz").read_bytes()
    # A test engine keeps runs across a restart; a one-run engine removes this version's leftovers as it starts.
    config["proseg"]["keep_unsaved"] = True
    restarted = Runs(config, runs.samples, runs.zarr)
    try:
        restored = restarted.get(run.meta["id"])
        monkeypatch.delenv("FAKE_STAIN_FAIL")
        restarted.control(run.meta["id"], "resume")
        wait_for(lambda: ended(restored))
        assert restored.meta["state"] == "done"
        assert (restored.dir / "stain/tile-000000.npz").read_bytes() == before
    finally:
        restarted.close(save=False)


@pytest.mark.parametrize("change", [
    {"method": "instanseg"}, {"image": "missing"}, {"nucleus": "unknown"}, {"cytoplasm": "DAPI"},
    {"diameter": True}, {"diameter": 0}, {"flow_threshold": float("nan")}, {"cellprob_threshold": 21},
    {"model": "unknown"}, {"model": "../cpsam"}, {"model": "/tmp/cpsam"}, {"model": "a\\cpsam"},
    {"model": None}, {"model": []}, {"model": {}}, {"model": ""},
])
def test_stain_settings_reject_invalid_choices_before_creating_files(runs, change):
    with pytest.raises(RunError):
        runs.create("image-sample", kind="stain", region={"kind": "full"}, stain={**SETTINGS, **change})
    assert not runs.runs and not list(runs.root.iterdir())


def test_stain_blocks_unavailable_or_volume_images(runs):
    runs.zarr.image["levels"][0]["shape"].insert(1, 3)
    runs.zarr.image["levels"][0]["dims"].insert(1, "z")
    with pytest.raises(RunError, match="two-dimensional"):
        start(runs)
    runs.stain.python = "/missing/python"
    with pytest.raises(RunError, match="unavailable"):
        start(runs)


def test_stain_respects_active_limit_and_proseg_controls_are_rejected(runs):
    runs.max_active = 1
    run = start(runs)
    with pytest.raises(RunError, match="Another run is open"):
        start(runs)
    with pytest.raises(RunError, match="Pause"):
        runs.control(run.meta["id"], "step")
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "stopped"
    with pytest.raises(RunError, match="join its finished tiles"):
        runs.save_to_zarr(run.meta["id"], "cellpose_boundaries_1", None)


def test_stain_reopening_interrupted_run_and_resuming_waits_for_save(runs, config):
    run = start(runs)
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    runs.control(run.meta["id"], "stop")
    wait_for(lambda: ended(run))
    run.meta.update(state="running", zarr_save={"state": "saving"})
    run.save()
    run.meta.update(state="stopped", zarr_save=None)    # only its file says so: this engine closes as usual
    config["proseg"]["keep_unsaved"] = True             # a test engine keeps the run across the restart
    restored = Runs(config, runs.samples, runs.zarr)
    try:
        r = restored.get(run.meta["id"])
        assert r.meta["state"] == "stopped" and r.meta["zarr_save"]["state"] == "failed"
        r.meta["zarr_save"]["state"] = "saving"
        with pytest.raises(RunError, match="finish saving"):
            restored.control(r.meta["id"], "resume")
        r.meta.update(zarr_save=None, progress={**r.meta["progress"], "resolved": False})
        assert (r.dir / "output/boundaries.geojson").exists()
        with pytest.raises(RunError, match="join its finished tiles"):
            restored.save_to_zarr(r.meta["id"], "cellpose_boundaries_1", None)
    finally:
        restored.close(save=False)


def test_stain_api_lists_progress_vectors_and_log_without_qc(config, tmp_path, monkeypatch):
    from sopa_companion.server import create_app
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    app = create_app(config, token="t"*40)
    app.state.runs.zarr = Images(tmp_path / "unwritten.zarr")
    base = "http://127.0.0.1:8799"
    headers = {"Authorization": "Bearer "+"t"*40, "Origin": base}
    with TestClient(app, base_url=base, headers=headers) as client:
        assert client.get("/api/v1/stain/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        models = client.get("/api/v1/stain/models")
        assert models.status_code == 200
        assert [m["name"] for m in models.json()["models"]] == ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
        assert all("path" not in m for m in models.json()["models"])
        bad = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="nope", region={"kind": "full"}))
        assert bad.status_code == 422
        response = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain",
                               region={"kind": "full"}, stain=SETTINGS))
        assert response.status_code == 200, response.text
        rid = response.json()["id"]
        run = app.state.runs.get(rid)
        assert response.json()["state"] == "paused"
        assert client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}).status_code == 200
        wait_for(lambda: ended(run))
        assert client.get(f"/api/v1/runs/{rid}").json()["summary"]["cells"] == 6
        view = client.get(f"/api/v1/runs/{rid}/stain/vectors", params={"box": "9,9,13,13"})
        assert view.status_code == 200 and view.content[:4] == b"PLPV"
        assert client.get(f"/api/v1/runs/{rid}/stain/vectors?box=bad").status_code == 422
        log = client.get(f"/api/v1/runs/{rid}/log").text
        assert "Cellpose" in log and "Stain segmentation" in log and "cpsam" in log
        assert "FAKE" not in run.meta["stain"]["model"]
        assert client.get("/api/v1/runs").json()[0]["kind"] == "stain"
        assert client.delete(f"/api/v1/runs/{rid}").status_code == 200


def test_stain_models_route_sends_the_level_a_custom_model_learned_at(config, monkeypatch):
    """The picker's "learned at scale1" (stain.mjs modelLabel) is made from the patch set's level the route sends."""
    from sopa_companion.server import create_app
    from test_cellpose_settings import gui_file
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    monkeypatch.setenv("FAKE_STAIN_MODELS", json.dumps([dict(name="cpsam", kind="builtin", path=None, date=None),
        dict(name="learned_model", kind="custom", path="/missing/learned_model", date="2026-10-21")]))
    app = create_app(config, token="t"*40)
    folder = app.state.runs.stain.patches.root / "run" / "set"
    folder.mkdir(parents=True)
    (folder / "patches.json").write_text(json.dumps(dict(level=1, factor=2, um_per_px=.425)))
    gui_file(folder, model_path="/missing/learned_model")      # the Cellpose GUI trained it on these patches
    base = "http://127.0.0.1:8799"
    with TestClient(app, base_url=base, headers={"Authorization": "Bearer "+"t"*40, "Origin": base}) as client:
        models = client.get("/api/v1/stain/models").json()["models"]
    assert [(m["name"], m["resolution"]) for m in models] == [("cpsam", None), ("learned_model", dict(level=1, factor=2))]
    assert all("path" not in m for m in models)


def test_a_set_in_a_folder_exported_into_labels_the_model_it_trained(runs, monkeypatch, chosen):
    """A set exported into a chosen folder (found through the folders exported into) names the model the GUI trained
    on it: the models list says the level its patches were cut at, as for a set of the earlier layout."""
    from test_cellpose_settings import gui_file
    from test_patch_folders import plant
    monkeypatch.setenv("FAKE_STAIN_MODELS", json.dumps([dict(name="cpsam", kind="builtin", path=None, date=None),
        dict(name="learned_model", kind="custom", path="/missing/learned_model", date="2026-10-21")]))
    folder = chosen / "training"
    manifest = plant(folder, level=1)
    manifest["factor"] = 2
    (folder / ".sopa-companion" / "sets" / f"{manifest['set']}.json").write_text(json.dumps(manifest))
    gui_file(folder, name=manifest["patches"][0]["mask"].removesuffix("_seg.npy"), model_path="/missing/learned_model")
    runs.stain.patches.folders.used(folder, ["image-sample", 1, ["DAPI", "Boundary", None]])
    models = runs.stain.models()
    assert [(m["name"], m.get("resolution")) for m in models] == [("cpsam", None), ("learned_model", dict(level=1, factor=2))]
    assert models[1]["label"] == "learned_model · learned at scale1"


def test_start_never_reads_the_exported_sets(runs, monkeypatch):
    """Start checks the model against the list without its labels (and Use settings from Cellpose reads the list so
    too): reading every exported set's folder, some on a slow or stalled disk, is never on its way."""
    def unreadable(sample=None):
        raise AssertionError("the exported sets were read")
    monkeypatch.setattr(runs.stain.patches, "sources", unreadable)
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)["id"])
    assert run.meta["stain"]["model"] == "cpsam" and run.meta["stain"]["model_kind"] == "builtin"


def test_labels_are_worked_out_once_per_refresh_of_the_list(runs, monkeypatch):
    calls = []
    real = runs.stain.patches.sources
    monkeypatch.setattr(runs.stain.patches, "sources", lambda sample=None: calls.append(sample) or real(sample))
    runs.stain.models()
    runs.stain.models()
    runs.stain.models(labels=False)
    assert calls == [None]
    runs.stain._models_until = 0                       # the 60 s refresh
    runs.stain.models()
    assert calls == [None, None]


def test_stain_model_discovery_is_cached_and_refreshed(runs, monkeypatch):
    from sopa_companion.engine import stain
    calls = []
    execute = stain.subprocess.run

    def discover(*args, **kwargs):
        calls.append((args, kwargs))
        return execute(*args, **kwargs)

    monkeypatch.setattr(stain.subprocess, "run", discover)
    first = runs.stain.models()
    first.clear()
    assert len(runs.stain.models()) == 4 and len(calls) == 1
    assert calls[0][0][0][-1] == "--models"
    assert "DISPLAY" not in calls[0][1]["env"]
    runs.stain._models_until = 0
    runs.stain.models()
    assert len(calls) == 2


def test_stain_model_discovery_failure_is_plain_and_retryable(runs, monkeypatch):
    from sopa_companion.engine import stain
    execute = stain.subprocess.run

    def fail(*args, **kwargs):
        raise stain.subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(stain.subprocess, "run", fail)
    with pytest.raises(RunError, match="model list could not be loaded") as error:
        runs.stain.models()
    assert error.value.status == 503
    monkeypatch.setattr(stain.subprocess, "run", execute)
    assert len(runs.stain.models()) == 4


def test_stain_selected_model_and_download_progress_reach_run_and_log(runs, monkeypatch):
    monkeypatch.setenv("FAKE_STAIN_DOWNLOAD", "1")
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"},
        stain={**SETTINGS, "model": "cpdino", "model_path": "/untrusted/page/path", "model_sha256": "fake"})["id"])
    assert run.meta["stain"]["model"] == "cpdino" and "model_path" not in run.meta["stain"]
    assert "model_sha256" not in run.meta["stain"]
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: run.meta["message"] == "Downloading the model…")
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done" and run.meta["message"] == ""
    assert run.meta["stain_runtime"]["model"] == "cpdino"
    assert json.loads((run.dir / "stain/job.json").read_text())["settings"]["model"] == "cpdino"
    log = (run.dir / "stain.log").read_text()
    assert "cpdino" in log and "Downloading the model" in log


@pytest.mark.parametrize("model", ["cpdino", "cpsam_20250102"])
def test_stain_save_passes_selected_model_and_custom_identity(runs, tmp_path, monkeypatch, model):
    path, digest = None, None
    if model != "cpdino":
        path = str(tmp_path / model)
        Path(path).write_bytes(b"trained weights")
        digest = hashlib.sha256(b"trained weights").hexdigest()
        monkeypatch.setenv("FAKE_STAIN_MODELS", json.dumps([
            dict(name=model, kind="custom", path=path, date="2025-01-02")]))
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"},
        stain={**SETTINGS, "model": model, "model_path": "/untrusted", "model_sha256": "untrusted"})["id"])
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: ended(run))
    written = []

    def save(*args, **kwargs):
        written.append(copy.deepcopy(args[6]))
        return {"cells": 6}

    monkeypatch.setattr(runs.zarr, "existing_elements", lambda sid: [], raising=False)
    monkeypatch.setattr(runs.zarr, "save", save, raising=False)
    runs.save_to_zarr(run.meta["id"], "cellpose_shapes", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving")
    assert run.meta["zarr_save"]["state"] == "saved"
    assert len(written) == 1
    assert {k: written[0][k] for k in ("model", "model_path", "model_sha256")} == {
        "model": model, "model_path": path, "model_sha256": digest}
    assert written[0]["parameters"]["model"] == written[0]["runtime"]["model"] == model
    if digest:
        assert written[0]["parameters"]["model_sha256"] == written[0]["runtime"]["model_sha256"] == digest
        assert digest in (run.dir / "stain.log").read_text()


def test_stain_vectors_endpoint_serves_committed_tiles_then_resolved_shapes(config, tmp_path, monkeypatch):
    from sopa_companion.engine import vectors
    from sopa_companion.server import create_app
    from sopa_companion.tests.fake_stain import arrays

    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    app = create_app(config, token="t"*40)
    runs = app.state.runs
    runs.zarr = Images(tmp_path / "unwritten.zarr")
    monkeypatch.setattr(runs.stain, "launch", lambda run: None)
    base = "http://127.0.0.1:8799"
    with TestClient(app, base_url=base, headers={"Authorization": "Bearer "+"t"*40, "Origin": base}) as client:
        response = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain",
                               region={"kind": "full"}, stain=SETTINGS))
        assert response.status_code == 200, response.text
        rid = response.json()["id"]
        run = runs.get(rid)
        root = run.dir / "stain"
        rings = [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]],
                 [[15, 10], [25, 10], [25, 20], [15, 20], [15, 10]]]
        for i, ring in enumerate(rings):
            np.savez(root / f"tile-{i:06d}.npz", **arrays([ring]))
        (root / "tiles.json").write_text(json.dumps([{"box": [10, 10, 25, 20]}]*2))

        def view(box, revision):
            result = client.get(f"/api/v1/runs/{rid}/stain/vectors",
                                params={"box": box, "revision": revision}, headers={"Accept-Encoding": "identity"})
            assert result.status_code == 200, result.text
            assert result.headers["cache-control"] == "private, no-cache"
            assert result.headers["content-type"] == "application/octet-stream"
            value = vectors.decode(result.content)
            assert value["revision"] == revision and value["source"] == {"cells": "Cellpose"}
            assert value["nuclei"] is None
            return value

        for done in (1, 2):
            runs._set(run, state="running", progress=dict(done=done, total=2, revision=done, resolved=False))
            live = view("0,0,30,30", done)
            assert not live["resolved"] and live["total"] == done
            assert live["cells"]["offsets"].tolist() == list(range(0, done*5+1, 5))
            np.testing.assert_array_equal(live["cells"]["xy"], np.array(rings[:done]).reshape(-1, 2))
        final = root / "resolved-000002"
        final.mkdir()
        joined = [[10, 10], [25, 10], [25, 20], [10, 20], [10, 10]]
        for name, value in arrays([joined]).items():
            np.save(final / f"{name}.npy", value)
        runs._set(run, state="done", progress=dict(done=2, total=2, revision=3, resolved=True, complete=True))
        for box in ("0,0,30,30", "21,11,30,22"):   # Full view and a pan into the resolved polygon's right edge.
            resolved = view(box, 3)
            assert resolved["resolved"] and resolved["total"] == 1
            assert resolved["cells"]["offsets"].tolist() == [0, 5]
            np.testing.assert_array_equal(resolved["cells"]["xy"], joined)
        empty = view("100,100,110,110", 3)
        assert empty["total"] == 0 and empty["cells"]["offsets"].tolist() == [0]
        assert client.post(f"/api/v1/runs/{rid}/control", json={"action": "pause"}).status_code == 409


@pytest.mark.parametrize("diameter", [1, 5, 9.9])
def test_stain_tiny_diameter_is_rejected_before_creating_files(runs, diameter):
    with pytest.raises(RunError, match="between 10 and 1000"):
        runs.create("image-sample", kind="stain", region={"kind": "full"},
                    stain={**SETTINGS, "diameter": diameter})
    assert not runs.runs and not list(runs.root.iterdir())


def test_stain_forgetting_sample_mid_run_reopens_recorded_path(runs, config, monkeypatch):
    registry = ZarrSamples(config)
    registry._remember(runs.zarr.info)
    opened = []

    def reopen(path, recent=True):
        opened.append((path, recent))
        registry._remember(runs.zarr.info, shown=recent)
        return runs.zarr.info

    monkeypatch.setattr(runs.zarr, "known", registry.known)
    monkeypatch.setattr(runs.zarr, "open", reopen, raising=False)
    monkeypatch.setattr(runs.zarr, "save_defaults", registry.save_defaults)
    run = start(runs)
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    registry.forget(run.meta["sample"])
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done", run.meta["message"]
    assert run.meta["progress"]["resolved"] and run.meta["progress"]["complete"]
    # Opened again where the run found it, for the run: off the open dialog's list.
    assert opened == [(run.meta["sample_path"], False)]
    assert run.meta["save_defaults"]["shapes_key"] == "cellpose_boundaries_1"
    assert (run.dir / "output/boundaries.geojson").is_file()


def test_stain_defaults_failure_does_not_fail_completed_output(runs, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("Sample is offline")

    monkeypatch.setattr(runs.zarr, "save_defaults", unavailable)
    run = start(runs)
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done", run.meta["message"]
    assert run.meta["progress"]["resolved"] and run.meta["progress"]["complete"]
    assert run.meta["outputs"]
    assert "Sample is offline" in run.meta["message"]


def test_stain_completed_resolved_failed_run_can_be_saved(runs, monkeypatch):
    run = start(runs)
    wait_for(lambda: ended(run))
    runs._set(run, state="failed", message="Unknown sample: open it again")
    monkeypatch.setattr(runs.zarr, "existing_elements", lambda sid: [], raising=False)
    written = []

    def save(*args, **kwargs):
        written.append(json.loads((run.dir / "output/boundaries.geojson").read_text()))
        return {"cells": len(written[-1]["features"])}

    monkeypatch.setattr(runs.zarr, "save", save, raising=False)
    runs.save_to_zarr(run.meta["id"], "cellpose_full", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving")
    assert run.meta["zarr_save"]["state"] == "saved"
    assert len(written) == 1 and len(written[0]["features"]) == 6


def test_stain_resume_marks_saved_revision_outdated_until_saved_again(runs, config, monkeypatch):
    run = start(runs)
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    runs.control(run.meta["id"], "stop")
    wait_for(lambda: ended(run))
    old_revision, old_cells = run.meta["progress"]["revision"], run.meta["summary"]["cells"]
    assert 0 < old_cells < 6
    monkeypatch.setattr(runs.zarr, "existing_elements", lambda sid: [], raising=False)
    written = []

    def save(*args, **kwargs):
        written.append(copy.deepcopy(args[6]))
        return {"cells": run.meta["summary"]["cells"]}

    monkeypatch.setattr(runs.zarr, "save", save, raising=False)
    runs.save_to_zarr(run.meta["id"], "cellpose_partial", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    runs.control(run.meta["id"], "resume")
    assert run.meta["zarr_save"]["state"] == "outdated"
    assert run.meta["zarr_save"]["result_revision"] == old_revision
    wait_for(lambda: ended(run))
    assert run.meta["summary"]["cells"] == 6
    assert run.meta["zarr_save"]["state"] == "outdated"
    assert run.meta["zarr_save"]["cells"] == old_cells and len(written) == 1
    config["proseg"]["keep_unsaved"] = True             # a test engine keeps the run across the restart
    restored = Runs(config, runs.samples, runs.zarr)
    try:
        assert restored.get(run.meta["id"]).meta["zarr_save"]["state"] == "outdated"
    finally:
        restored.close(save=False)
    runs.save_to_zarr(run.meta["id"], "cellpose_full", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    assert run.meta["zarr_save"]["result_revision"] == run.meta["progress"]["revision"] > old_revision
    assert run.meta["zarr_save"]["cells"] == 6
    assert [p["result_revision"] for p in written] == [old_revision, run.meta["progress"]["revision"]]


@pytest.mark.parametrize("level", [-1, True, 1.5, "1", 4])
def test_resolution_is_validated_before_creating_a_run(runs, level):
    before = list(runs.root.iterdir())
    with pytest.raises(RunError, match="Choose a resolution"):
        runs.create("image-sample", kind="stain", region={"kind": "full"}, stain={**SETTINGS, "level": level})
    assert list(runs.root.iterdir()) == before


def test_level_settings_meta_patch_frame_and_whole_pyramid_are_kept(runs):
    runs.zarr.image["levels"].append({**copy.deepcopy(IMAGE["levels"][0]), "pixel": [2, 2], "shape": [2, 150, 200]})
    created = runs.create("image-sample", kind="stain", region={"kind": "full"}, stain={**SETTINGS, "level": 1, "diameter": 12})
    run = runs.get(created["id"])
    assert created["session_model"] == 2 and created["sample_tools"] == ["proseg", "stain"]
    assert created["covered"] is None and created["stain"]["diameter"] == 12
    assert created["stain"]["resolution"] == dict(level=1, factor=2, um_per_px=2, um_per_px_y=2, label="scale1")
    job = json.loads((run.dir / "stain/job.json").read_text())
    assert len(job["image"]["levels"]) == 2 and job["settings"]["level"] == 1
    draft = runs.stain.patches.draft(run)
    assert draft["level"] == 1 and draft["factor"] == 2 and draft["um_per_px"] == 2
    assert draft["run_pixel_to_sample"] == [0, 0, 2, 2]


def test_a_run_on_a_sample_in_pixels_says_so_for_its_page(runs):
    """The run's page told a patch's size in µm on a sample whose transcripts seem to be in pixels (its open notes say
    pixel_frame), where the picker said none: the run now carries it (pixel_frame), and only then."""
    runs.one_run = False                  # two runs of one sample side by side, as a test engine keeps them
    plain = runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
    assert "pixel_frame" not in plain
    runs.zarr.info["notes"] = [{"code": "pixel_frame", "text": "These transcripts seem to be in pixels.", "detail": ""}]
    framed = runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
    assert framed["pixel_frame"] is True and runs.get(framed["id"]).public()["pixel_frame"] is True


def test_region_under_32_level_pixels_is_refused(runs):
    runs.zarr.image["levels"].append({**copy.deepcopy(IMAGE["levels"][0]), "pixel": [4, 4], "shape": [2, 75, 100]})
    with pytest.raises(RunError, match="only 25 pixels across at scale1$"):
        runs.create("image-sample", kind="stain", region={"kind": "box", "box": [0, 0, 100, 100]},
                    stain={**SETTINGS, "level": 1})
    assert not runs.runs and not list(runs.root.iterdir())


def test_instanseg_uses_geometric_pixel_size_of_rounded_level(runs, monkeypatch):
    runs.stain.instanseg_enabled = True          # InstanSeg is dormant unless its internal switch is on
    runs.stain.instanseg_python = sys.executable
    monkeypatch.setattr(runs.stain, "models", lambda method=None: [dict(name="single_channel_nuclei", kind="builtin")])
    runs.zarr.image["levels"].append({**copy.deepcopy(IMAGE["levels"][0]), "pixel": [2, 2.0004], "shape": [2, 150, 200]})
    ready = runs.stain.prepare("image-sample", {"kind": "full"}, dict(method="instanseg", model="single_channel_nuclei",
                            image="image", channels=["DAPI"], level=1))
    assert ready["settings"]["pixel_size"] == pytest.approx(np.sqrt(2 * 2.0004))


def test_one_run_also_blocks_ended_runs_and_patch_items(runs):
    run = start(runs, resume=False)
    runs._set(run, state="done")
    with pytest.raises(RunError, match="Another run is open") as error:
        runs.create("image-sample", kind="patches", region={"kind": "full"}, stain={**SETTINGS, "start_from": "empty"})
    assert error.value.status == 409 and len(runs.runs) == 1
    runs.one_run = False                                # a test engine: several runs at once
    second = runs.create("image-sample", kind="patches", region={"kind": "full"}, stain={**SETTINGS, "start_from": "empty"})
    assert second["id"] != run.meta["id"]


def test_a_stain_run_is_checked_again_where_it_is_registered(runs, monkeypatch):
    """Runs.create refuses while another run is open (check_open) before StainRuns.create prepares the run; what
    happens meanwhile is seen where the run is registered, under the registry's lock: a run
    registered meanwhile (a test square's never counts), a close begun meanwhile. Nothing is left behind."""
    from types import SimpleNamespace
    prepare = runs.stain.prepare
    meanwhile = []

    def preparing(*args, **kwargs):
        ready = prepare(*args, **kwargs)
        for happen in meanwhile:
            happen()
        return ready
    monkeypatch.setattr(runs.stain, "prepare", preparing)
    square = SimpleNamespace(meta={"id": "square", "state": "running", "qc_test": {"of": "user"}})
    other = SimpleNamespace(meta={"id": "other", "state": "paused"})
    try:
        meanwhile[:] = [lambda: runs.runs.update(other=other)]
        with pytest.raises(RunError, match="Another run is open") as refused:
            runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
        assert refused.value.status == 409 and list(runs.runs) == ["other"] and not list(runs.root.iterdir())
        runs.runs.clear()
        meanwhile[:] = [lambda: setattr(runs, "closing", True)]
        with pytest.raises(RunError, match="shutting down") as refused:
            runs.create("image-sample", kind="patches", region={"kind": "full"},
                        stain={**SETTINGS, "start_from": "empty"})
        assert refused.value.status == 503 and not runs.runs and not list(runs.root.iterdir())
        runs.closing = False
        meanwhile[:] = [lambda: runs.runs.update(square=square)]
        made = runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
        assert set(runs.runs) == {"square", made["id"]}
    finally:
        runs.closing = False
        runs.runs.pop("square", None)
        runs.runs.pop("other", None)


def test_the_model_lists_start_from_the_apps_own_entry(runs, monkeypatch):
    """Like every helper (engine/children.py app_path), never from the engine's working folder (the desktop launcher's
    is the home folder), which python -m would put first on their path; on the desktop the app alone on PYTHONPATH."""
    from types import SimpleNamespace
    from sopa_companion.engine import children, stain
    from sopa_companion.engine.config import helper_path
    seen = []

    def run(command, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(stdout=json.dumps({"models": [{"name": "cpsam", "kind": "builtin"}]}))
    monkeypatch.setattr(stain.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    runs.stain.instanseg_enabled = True          # InstanSeg is dormant unless its internal switch is on
    runs.stain.instanseg_python = sys.executable
    assert runs.stain.models()[0]["name"] == runs.stain.models("instanseg")[0]["name"] == "cpsam"
    assert [kwargs["cwd"] for kwargs in seen] == [children.app_path()] * 2
    assert [kwargs["env"]["PYTHONPATH"] for kwargs in seen] == [helper_path()] * 2


def test_join_ended_tiles_never_segments_another_tile(runs, monkeypatch):
    monkeypatch.setenv("FAKE_STAIN_FAIL", "2")
    run = start(runs)
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "failed" and run.meta["progress"]["done"] == 2
    before = [(run.dir / f"stain/tile-{k:06d}.npz").read_bytes() for k in range(2)]
    result = runs.stain.join(run)
    assert result["progress"]["done"] == 2 and result["progress"]["resolved"]
    assert result["summary"]["cells"] == 2 and not (run.dir / "stain/tile-000002.npz").exists()
    assert before == [(run.dir / f"stain/tile-{k:06d}.npz").read_bytes() for k in range(2)]
    assert json.loads((run.dir / "stain/join.json").read_text())["join_only"] is True
    assert runs.stain.join(run)["progress"] == result["progress"]


def test_a_moved_run_is_joined_and_resumed_in_its_folder_as_it_is_now(config, tmp_path, monkeypatch):
    """A stain run's job names its folder by its full path, as it was when the run was made. Its data folder moved since
    (an app folder moves with its var/): the join and the resume work in the run's folder as it is now, and nothing is
    read from or made at the old place."""
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", tile_pixels=128,
                           overlap_pixels=16, gpu=1)
    config["proseg"]["keep_unsaved"] = True             # a test engine keeps its runs across the restart
    images = Images(tmp_path / "unwritten.zarr")
    monkeypatch.setenv("FAKE_STAIN_FAIL", "2")
    first = Runs(config, Samples(config), images)
    try:
        run = start(first)
        wait_for(lambda: ended(run))
    finally:
        first.close(save=False)
    assert run.meta["state"] == "failed" and run.meta["progress"]["done"] == 2
    old, new = Path(config["paths"]["data_dir"]), tmp_path / "moved" / "var"
    new.parent.mkdir()
    old.rename(new)
    moved = {**config, "paths": {**config["paths"], "data_dir": str(new)}}
    monkeypatch.delenv("FAKE_STAIN_FAIL")
    second = Runs(moved, Samples(moved), images)
    try:
        again = second.get(run.meta["id"])
        assert again.dir == new / "runs" / run.meta["id"]
        assert json.loads((again.dir / "stain/job.json").read_text())["out"] == str(run.dir)      # as it was made
        result = second.stain.join(again)
        assert result["progress"]["resolved"] and result["summary"]["cells"] == 2
        assert json.loads((again.dir / "stain/join.json").read_text())["out"] == str(again.dir)
        assert (again.dir / "output/boundaries.geojson").exists()
        second.control(again.meta["id"], "resume")
        wait_for(lambda: ended(again))
        assert again.meta["state"] == "done" and again.meta["summary"]["cells"] == 6
        assert json.loads((again.dir / "stain/job.json").read_text())["out"] == str(again.dir)
    finally:
        second.close(save=False)
    assert not old.exists()


def test_each_launch_gives_the_helper_this_engines_instanseg(runs, monkeypatch):
    """An InstanSeg run's job names the InstanSeg environment and module of the engine that made it (an earlier
    version's own package's): each launch, and a join, hand the helper this engine's, and the run's folder now."""
    import io
    from types import SimpleNamespace
    from sopa_companion.engine import stain
    runs.stain.instanseg_python = sys.executable
    run = start(runs, resume=False)
    path = run.dir / "stain/job.json"
    job = json.loads(path.read_text())
    job["settings"]["method"] = "instanseg"
    job.update(out=f"/gone/var/runs/{run.meta['id']}", instanseg_python="/gone/python",
               instanseg_module="earlier_package.zarr_bridge.stain_instanseg")
    path.write_text(json.dumps(job))
    given = []

    def popen(command, **kwargs):
        given.append(json.loads(Path(command[-1]).read_text()))
        return SimpleNamespace(pid=123, stdout=io.StringIO(""), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(stain.subprocess, "Popen", popen)
    runs.stain.execute(run)
    runs.stain.execute(run, join=True)
    assert [(j["out"], j["instanseg_python"], j["instanseg_module"], j.get("join_only")) for j in given] == [
        (str(run.dir), sys.executable, "sopa_companion.zarr_bridge.stain_instanseg", None),
        (str(run.dir), sys.executable, "sopa_companion.zarr_bridge.stain_instanseg", True)]


def test_done_stores_covered_and_resume_clears_it_and_marks_queued_save_old(runs, monkeypatch):
    import io
    from types import SimpleNamespace
    from sopa_companion.engine import stain
    run = start(runs, resume=False)
    covered = dict(region={"kind": "box", "box": [0, 0, 100, 100]}, done=1, of=6, why="tiles", pieces=1)
    progress = dict(done=0, total=6, revision=1, resolved=False, complete=False)
    events = [dict(ev="plan", progress=progress), dict(ev="done", progress=progress, covered=covered)]
    process = SimpleNamespace(pid=123, stdout=io.StringIO("\n".join(json.dumps(e) for e in events)),
                              poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(stain.subprocess, "Popen", lambda *a, **kw: process)
    runs.stain.execute(run)
    assert run.meta["covered"] == covered and run.meta["state"] == "stopped"
    runs._set(run, zarr_save={"state": "queued"})
    monkeypatch.setattr(runs.stain, "launch", lambda *a: None)
    runs.control(run.meta["id"], "resume")
    assert run.meta["covered"] is None and run.meta["zarr_save"]["state"] == "outdated"
