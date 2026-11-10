"""InstanSeg is dormant: the engine offers Cellpose alone unless the internal switch [stain] instanseg_enabled is on. Off
(the default), health lists Cellpose only, InstanSeg's model list, runs, Training patches and their boxes are refused in
plain words (422) before anything is made or started, a configured [stain] instanseg_python is never run, and a run or
a Training patches item InstanSeg made while it was on segments or exports nothing more. What InstanSeg saved stays an
ordinary layer. What it does while switched on is test_stain_instanseg.py's (its tests switch it on)."""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from test_stain import Images, SETTINGS, runs, stain_cpu  # noqa: F401  (the stain engine's fixtures)
from sopa_companion.engine.runs import RunError
from sopa_companion.engine.stain import StainRuns
from sopa_companion.server import create_app

BASE, TOKEN = "http://127.0.0.1:8799", "t" * 40
INSTANSEG = dict(method="instanseg", model="fluorescence_nuclei_and_cells", image="image", channels=["DAPI", "Boundary"])


def started_instanseg(started: list) -> list:
    """The processes started for InstanSeg (its worker module, the fake one here, or the real one)."""
    return [argv for argv in started if any("instanseg" in str(part) for part in argv)]


@pytest.fixture
def engine(config, tmp_path, monkeypatch):
    """An engine whose settings name InstanSeg's environment, as a computer that had InstanSeg keeps it in config.toml,
    and leave its switch unset. Every process the stain engine starts is recorded (and started as before)."""
    from sopa_companion.engine import stain
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", gpu=1,
                           instanseg_python=sys.executable, instanseg_module="sopa_companion.tests.fake_instanseg")
    config["stain"].pop("instanseg_enabled", None)
    started, run, popen = [], stain.subprocess.run, stain.subprocess.Popen
    monkeypatch.setattr(stain.subprocess, "run", lambda argv, **kw: started.append(list(argv)) or run(argv, **kw))
    monkeypatch.setattr(stain.subprocess, "Popen", lambda argv, **kw: started.append(list(argv)) or popen(argv, **kw))
    app = create_app(config, token=TOKEN)
    app.state.runs.zarr = Images(tmp_path / "unwritten.zarr")
    with TestClient(app, base_url=BASE, headers={"Authorization": "Bearer " + TOKEN, "Origin": BASE}) as client:
        yield client, app.state.runs, started


def test_health_and_the_models_route_offer_cellpose_alone(engine):
    client, runs, started = engine
    assert runs.stain.instanseg_enabled is False and runs.stain.methods == ("cellpose",)
    assert client.get("/api/v1/health").json()["stain"] == {"available": True, "methods": {"cellpose": True}}
    assert not runs.stain.method_available("instanseg"), "its environment is there, but it is switched off"
    refused = client.get("/api/v1/stain/models", params={"method": "instanseg"})
    assert (refused.status_code, refused.json()["detail"]) == (422, "Choose Cellpose")
    listed = client.get("/api/v1/stain/models").json()["models"]
    assert [m["name"] for m in listed] == ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
    assert not started_instanseg(started)


@pytest.mark.parametrize("kind,extra", [("stain", {}), ("patches", {"start_from": "model", "count": 2, "size": 64})])
def test_an_instanseg_run_or_patch_item_is_refused_and_nothing_is_made_or_started(engine, kind, extra):
    client, runs, started = engine
    answer = client.post("/api/v1/runs", json=dict(sample="image-sample", kind=kind, region={"kind": "full"},
                                                   stain={**INSTANSEG, **extra}))
    assert (answer.status_code, answer.json()["detail"]) == (422, "Choose Cellpose")
    preview = client.post("/api/v1/samples/image-sample/patches/preview",
                          json={"region": {"kind": "full"}, "stain": {**INSTANSEG, "start_from": "model", "count": 2,
                                                                      "size": 64}})
    assert (preview.status_code, preview.json()["detail"]) == (422, "Choose Cellpose")
    assert not runs.runs and not [p for p in runs.root.iterdir() if not p.name.startswith(".")]
    assert not started_instanseg(started)
    # Cellpose's run is made as before (paused until Start: nothing segments yet).
    made = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain", region={"kind": "full"},
                                                 stain=SETTINGS))
    assert made.status_code == 200, made.text
    assert made.json()["stain"]["method"] == "cellpose" and not started_instanseg(started)


def test_settings_that_are_not_a_method_this_engine_offers_name_the_ones_it_does(runs):
    stain = runs.stain
    for method in ("instanseg", "other", None, ["cellpose"], {"cellpose": True}):
        with pytest.raises(RunError, match="^Choose Cellpose$") as refused:
            stain.offered(method)
        assert refused.value.status == 422
    assert stain.offered("cellpose") == "cellpose"
    stain.instanseg_enabled = True
    assert stain.methods == ("cellpose", "instanseg") and stain.offered("instanseg") == "instanseg"
    with pytest.raises(RunError, match="^Choose Cellpose or InstanSeg$"):
        stain.offered("other")


def test_an_instanseg_run_made_while_it_was_on_segments_nothing_once_it_is_off(runs, monkeypatch):
    """A test engine keeps its runs between starts: an InstanSeg run made while it was switched on is never resumed
    once the switch is off (its helper would start InstanSeg's environment), whether it never started or it stopped."""
    stain = runs.stain
    stain.gpu = ""
    stain.instanseg_enabled, stain.instanseg_python = True, sys.executable
    stain.instanseg_module = "sopa_companion.tests.fake_instanseg"
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=INSTANSEG)["id"])
    stain.instanseg_enabled = False
    monkeypatch.setattr(StainRuns, "launch", lambda *a, **kw: pytest.fail("a switched-off method's helper started"))
    with pytest.raises(RunError, match="^Choose Cellpose$"):
        runs.control(run.meta["id"], "resume")
    assert run.meta["state"] == "paused" and run.meta["stain_started"] is False and run.worker is None
    runs._set(run, state="stopped", stain_started=True)
    with pytest.raises(RunError, match="^Choose Cellpose$"):
        runs.control(run.meta["id"], "resume")
    assert run.meta["state"] == "stopped" and run.worker is None


def test_a_patch_item_instanseg_made_while_it_was_on_exports_nothing_once_it_is_off(runs, monkeypatch, chosen):
    """A test engine keeps its Training patches items between starts too: one whose boxes InstanSeg's model outlines
    (start_from "model") is not exported once the switch is off (its export runs InstanSeg in its environment), and no
    patch set is begun. Cellpose's own item is not held up by it."""
    stain = runs.stain
    stain.gpu = ""
    stain.instanseg_enabled, stain.instanseg_python = True, sys.executable
    stain.instanseg_module = "sopa_companion.tests.fake_instanseg"
    item = runs.get(runs.create("image-sample", kind="patches", region={"kind": "full"},
                                stain={**INSTANSEG, "start_from": "model", "count": 2, "size": 64})["id"])
    stain.instanseg_enabled = False
    reached = []

    def model_export(run, *args):                   # where the model's process would start: stop there
        reached.append(run.meta["id"])
        raise RunError("stopped before the model", 503)
    monkeypatch.setattr(stain.patches, "model_export", model_export)
    monkeypatch.setattr(stain.worker, "call", lambda *a, **kw: pytest.fail("the patch helper was asked"))
    folder = chosen / "patches"
    with pytest.raises(RunError, match="^Choose Cellpose$") as refused:
        stain.patches.export(item, {"version": 0, "folder": str(folder)})
    assert refused.value.status == 422 and reached == [] and not stain.patches.exporting
    assert not folder.exists() and not stain.patches.staging.exists()
    runs.delete(item.meta["id"])
    cellpose = runs.get(runs.create("image-sample", kind="patches", region={"kind": "full"},
                                    stain={**SETTINGS, "start_from": "model", "count": 2, "size": 64})["id"])
    with pytest.raises(RunError, match="stopped before the model"):
        stain.patches.export(cellpose, {"version": 0, "folder": str(folder)})
    assert reached == [cellpose.meta["id"]]


def test_the_switch_is_true_or_false(config, tmp_path):
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    for value in ("yes", 1, "true"):
        config["stain"]["instanseg_enabled"] = value
        with pytest.raises(ValueError, match="stain.instanseg_enabled must be true or false"):
            Runs(config, Samples(config), Images(tmp_path / "unwritten.zarr"))


@pytest.mark.parametrize("resolution,name", [({}, "saved stain segmentation"),
                                             (dict(level=1, factor=2), "saved stain segmentation · scale1")])
def test_what_instanseg_saved_is_an_ordinary_layer_named_neutrally(resolution, name):
    from sopa_companion.engine.zarr_samples import describe_segmentation
    made_by = dict(shapes=[dict(key="instanseg_boundaries_1", app=dict(kind="stain", method="instanseg",
                                                                       model="fluorescence_nuclei_and_cells",
                                                                       resolution=resolution))])
    described = describe_segmentation("instanseg_boundaries_1", made_by)
    assert described["name"] == name and described["model"] == "fluorescence_nuclei_and_cells"
    assert "instanseg" not in described["name"].lower()
