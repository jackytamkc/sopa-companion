"""Patch API and folder round trip; only private files and a fake desktop launcher. Exports go into folders of the
tests' own (conftest.py chosen), as the user chooses one at every export."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import sopa_python, wait_for
from test_stain import Images, SETTINGS, stain_cpu
from sopa_companion.engine import patch_folders as F
from sopa_companion.server import create_app

BASE, TOKEN = "http://127.0.0.1:8799", "t"*40


def test_range_change_during_patch_proposal_refuses_stale_boxes(patches, monkeypatch):
    client, runs, run, url = patches
    manager = runs.stain.patches
    real = runs.stain.worker.call
    def changed(*args, **kw):
        result = real(*args, **kw)
        path = run.dir / "stain/exclusions.json"
        path.write_text(json.dumps(dict(version=1, areas=[], size=dict(min=200, max=None))))
        return result
    monkeypatch.setattr(runs.stain.worker, "call", changed)
    response = client.post(url + "/patches/propose", json=dict(version=0, count=1, size=128))
    assert response.status_code == 409 and "boxes changed" in response.json()["detail"]
    assert response.json()["reason"] == "changed"           # the page tells these apart (stain-patches.mjs)
    assert manager.draft(run)["version"] == 0


def test_legacy_patch_draft_reads_as_full_and_export_carries_filter_digest(patches, chosen):
    _, runs, run, _ = patches
    manager = runs.stain.patches
    draft = manager.draft(run)
    for key in ("level", "factor", "um_per_px"):
        draft.pop(key)
    manager.save(run, draft)
    restored = manager.draft(run)
    assert restored["level"] == 0 and restored["factor"] == restored["um_per_px"] == 1
    result = exported(patches, chosen / "patches")
    assert result["exported"]["level"] == 0 and result["exported"]["factor"] == 1
    assert result["exported"]["exclusions_digest"] == runs.stain.exclusions.digest({})


@pytest.fixture
def patches(config, tmp_path, monkeypatch, request):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    # The sopa environment and GPU 1 said here, not taken from this computer's settings (a new clone has neither).
    config["proseg"]["keep_unsaved"] = True       # this fixture intentionally keeps several independent items
    config["stain"].update(python=sopa_python(), patch_python=sopa_python(), gpu=1,
                           patch_module="sopa_companion.tests.fake_stain_patches")
    app = create_app(config, token=TOKEN)
    runs = app.state.runs
    runs.zarr = Images(tmp_path / "unwritten")
    monkeypatch.setattr(runs.zarr, "tissue", lambda *a: {
        "source": "region_of_interest", "polygons": [[[0,0],[400,0],[400,300],[0,300]]], "area_um2": 120000},
        raising=False)
    method = getattr(request, "param", "cellpose")
    runs.stain.instanseg_enabled = True          # switched on for the cases that choose the dormant InstanSeg
    runs.stain.instanseg_python = sys.executable
    monkeypatch.setattr(runs.stain, "models", lambda method="cellpose", labels=True: [
        dict(name="fluorescence_nuclei_and_cells" if method == "instanseg" else "cpsam", kind="builtin")])
    settings = SETTINGS if method == "cellpose" else dict(method=method, image="image", channels=["DAPI","Boundary"])
    with TestClient(app, base_url=BASE, headers={"Authorization": "Bearer "+TOKEN, "Origin": BASE}) as client:
        response = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain",
                               region={"kind": "full"}, stain=settings))
        assert response.status_code == 200, response.text
        run = runs.get(response.json()["id"])
        features = [dict(type="Feature", id=i+1, properties={}, geometry=dict(type="Polygon", coordinates=[[
            [x,10],[x+10,10],[x+10,20],[x,20],[x,10]]])) for i, x in enumerate([10,20,160])]
        (run.dir / "output/boundaries.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)))
        runs._set(run, state="done", paused=False, stain_started=True,
                  progress=dict(done=1, total=1, revision=2, resolved=True, complete=True))
        yield client, runs, run, f"/api/v1/runs/{run.meta['id']}/stain"


def export(client, url, version, folder):
    """POST <url>/export (url: .../stain/patches) into `folder`, the folder the user chose in the export dialog."""
    return client.post(url+"/export", json={"version": version, "folder": str(folder)})


def exported(patches, folder):
    client, _, _, url = patches
    first = client.get(url+"/patches").json()
    assert first["run_pixel_to_sample"] == [0,0,1,1]  # Add a box also works before the first proposal.
    result = client.post(url+"/patches", json={"version": first["version"], "boxes": [[0,0,128,128],[128,0,256,128]]})
    assert result.status_code == 200, result.text
    result = export(client, url+"/patches", result.json()["version"], folder)
    assert result.status_code == 200, result.text
    return result.json()


@pytest.fixture
def kept(config, chosen):
    """An API session's patches_dir ([session] patches_dir, launch(patches_dir=...)), set before the app is made; the
    session's data and state folders beside it, outside pytest's folders (which may be in a hidden one)."""
    folder = chosen / "kept patches"
    config["session"] = {"patches_dir": str(folder)}
    config["paths"].update(data_dir=str(chosen / "var"), state_dir=str(chosen / "state"),
                           cache_dir=str(chosen / "var" / "cache"))
    return folder


def test_in_a_notebook_the_suggestion_is_in_its_patches_dir_outside_the_sessions_folders(kept, patches, config):
    """launch(patches_dir=...) gives the export dialog its base: the suggestion is a folder named after the sample and
    its resolution there, an export into it lands outside both session folders, and the data folder is refused."""
    client, runs, run, url = patches
    assert runs.stain.patches.root == kept and kept.is_dir()
    suggested = client.get(url+"/patches/folders").json()
    real = Path(os.path.realpath(kept))
    assert suggested["path"] == str(real / "Test-scale0") and suggested["refused"] is None
    folder = Path(exported(patches, suggested["path"])["exported"]["folder"])
    assert folder == real / "Test-scale0" and len(list(folder.glob("Test-*.tif"))) == 2
    assert not folder.is_relative_to(Path(config["paths"]["data_dir"]))
    assert not folder.is_relative_to(Path(config["paths"]["state_dir"]))
    data = Path(config["paths"]["data_dir"])
    assert client.get(url+"/patches/folders", params={"path": str(data)}).json()["refused"] == F.words("app", name="var")
    refused = export(client, url+"/patches", client.get(url+"/patches").json()["version"], data / "patches")
    assert (refused.status_code, refused.json()["field"]) == (422, "folder")
    assert refused.json()["detail"] == F.words("app", name="patches") and not (data / "patches").exists()


def test_a_patches_dir_that_is_a_link_or_not_a_folder_is_refused(config, tmp_path):
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    (tmp_path / "file").write_text("")
    for value in (tmp_path / "link", tmp_path / "file"):
        config["session"] = {"patches_dir": str(value)}
        with pytest.raises(ValueError, match="patches_dir"):
            Runs(config, Samples(config), None)
    config["session"] = {"patches_dir": ""}                      # none given: the data folder's, as on the desktop
    runs = Runs(config, Samples(config), None)
    assert runs.stain.patches.root == Path(config["paths"]["data_dir"]) / "stain-patches"
    runs.close("discard")


def test_an_export_from_the_model_says_each_patch_and_is_timed_per_patch(patches, monkeypatch, chosen):
    """An export that runs the model (start_from "model": Cellpose on each patch, a minute a 256-pixel patch on a CPU)
    says how far it is while it runs (GET .../stain/patches "exporting": {done, total}, the dialog's "patch 2 of 3"),
    and its process is stopped only when one patch takes too long (EXPORT_PATCH_S), never for the whole export's time:
    three patches of 0.6 s each pass a 1 s limit; one patch over it fails plainly, with nothing written. A model
    missing from Cellpose's cache is downloaded first: the process says so with no count ("exporting" downloading:
    true until the first patch), which is not a failed export (0.8.0 answered 500, KeyError 'done', and stopped the
    download: a computer without the model could never export)."""
    from sopa_companion.engine import stain_patches as SP
    client, runs, _, _ = patches
    runs.stain.python = sopa_python()
    runs.stain.patch_export_module = "sopa_companion.tests.fake_stain_patches"
    monkeypatch.setattr(runs.stain, "launch", lambda *a: pytest.fail("segmentation started"))
    monkeypatch.setattr(SP, "EXPORT_FIRST_S", 120.0)
    monkeypatch.setattr(SP, "EXPORT_PATCH_S", 1.0)
    settings = {**SETTINGS, "start_from": "model", "count": 3, "size": 128}
    made = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches", region={"kind": "full"},
                                                 stain=settings))
    assert made.status_code == 200, made.text
    run = runs.get(made.json()["id"])
    url = f"/api/v1/runs/{run.meta['id']}/stain/patches"
    proposed = client.post(url+"/propose", json={"version": 0}).json()
    assert len(proposed["patches"]) == 3 and client.get(url).json()["exporting"] is None
    (run.dir/"stain/slow-export").write_text("0.6")
    (run.dir/"stain/download-export").write_text("0.5")
    answers, seen = [], []
    worker = threading.Thread(target=lambda: answers.append(export(client, url, proposed["version"], chosen/"slow")))
    worker.start()
    while worker.is_alive():
        exporting = client.get(url).json()["exporting"]
        if exporting is not None and exporting not in seen:
            seen.append(exporting)
    worker.join()
    assert answers[0].status_code == 200, answers[0].text
    assert {"done": 0, "total": 3} in seen and any(s["done"] in (1, 2) and s["total"] == 3 for s in seen), seen
    assert {"done": 0, "total": 3, "downloading": True} in seen and not any(s.get("downloading") and s["done"] for s in seen), seen
    (run.dir/"stain/download-export").unlink()
    assert client.get(url).json()["exporting"] is None
    # One patch past the limit: stopped, said plainly, nothing written into the folder.
    (run.dir/"stain/slow-export").write_text("2.5")
    version = client.get(url).json()["version"]
    failed = export(client, url, version, chosen/"too-slow")
    assert failed.status_code == 503 and "could not be exported" in failed.json()["detail"], failed.text
    assert not list((chosen/"too-slow").glob("*.tif")) and client.get(url).json()["exporting"] is None


def correct(folder, mask):
    """A correction in the Cellpose GUI of patch file `mask` (its name, from the export's manifest)."""
    path = Path(folder) / mask
    data = np.load(path, allow_pickle=True).item()
    data["masks"][40:50,40:50] = int(data["masks"].max())+1
    np.save(path, data)


@pytest.mark.parametrize("start,method", [("empty", "cellpose"), ("model", "cellpose"), ("model", "instanseg")])
def test_direct_patch_item_needs_no_cells_and_survives_restart(patches, config, monkeypatch, chosen, start, method):
    from sopa_companion.engine.runs import Runs
    client, runs, _, _ = patches
    runs.stain.python = sopa_python()
    runs.stain.patch_export_module = "sopa_companion.tests.fake_stain_patches"
    monkeypatch.setattr(runs.stain, "launch", lambda *a: pytest.fail("segmentation started"))
    settings = {**SETTINGS, "start_from": start, "count": 3, "size": 128}
    if method == "instanseg":
        settings.update(method=method, model="fluorescence_nuclei_and_cells", channels=["DAPI", "Boundary"])
    if start == "empty":
        monkeypatch.setattr(runs.stain, "models", lambda *a, **k: pytest.fail("empty patches loaded a model list"))
    response = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches",
                           region={"kind":"full"}, stain=settings))
    assert response.status_code == 200, response.text
    meta = response.json()
    run = runs.get(meta["id"])
    url = f"/api/v1/runs/{meta['id']}/stain/patches"
    assert meta["kind"] == "patches" and meta["state"] == "done" and meta["progress"]["done"] == 0
    assert not list((run.dir/"output").iterdir()) and run.worker is None
    first = client.get(url).json()
    assert first["count"] == 3 and first["size"] == 128 and first["start_from"] == start
    proposed = client.post(url+"/propose", json={"version":0})
    assert proposed.status_code == 200, proposed.text
    data = proposed.json()
    boxes = [p["box"] for p in data["patches"]]
    assert len(boxes) == 3
    assert all(not (a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3])
               for i,a in enumerate(boxes) for b in boxes[i+1:])
    response = export(client, url, data["version"], chosen / "patches")
    assert response.status_code == 200, response.text
    exported = response.json()
    folder = Path(exported["exported"]["folder"])
    assert all(p["state"] == "untouched" and p["cells"] == (0 if start=="empty" else 1)
               for p in exported["exported"]["states"])
    evaluations = run.dir/"crop-evaluations.jsonl"
    if start == "model":
        assert [json.loads(line) for line in evaluations.read_text().splitlines()] == [
            [128,128,2 if method=="instanseg" else 3]]*3
        assert run.meta["stain_runtime"]["cuda_visible_devices"] == ""
        assert run.meta["stain_runtime"]["nice"] == 19
        (run.dir/"stain/fail-export").touch()
        assert export(client, url, exported["version"], folder).status_code == 503
        assert client.get(url).json()["set"] == exported["set"]
        # Nothing of the failed export is left: its staging went, no part file is in the folder, no set was added.
        assert not list(runs.stain.patches.staging.iterdir()) and not list(folder.rglob("*.part"))
        assert len(list((folder/F.META/"sets").iterdir())) == 1
    else:
        assert not evaluations.exists()
    correct(folder, exported["exported"]["patches"][0]["mask"])
    assert client.get(url).json()["exported"]["corrected"] == 1
    assert client.post(f"/api/v1/runs/{meta['id']}/zarr",json={"shapes_key":"no", "table_key":"no"}).status_code == 409
    runs.stain.worker.close()
    restarted = Runs(config, runs.samples, runs.zarr)
    try:
        kept = restarted.get(meta["id"])
        assert kept.public()["kind"] == "patches"
        assert restarted.stain.patches.status(kept)["exported"]["corrected"] == 1
        restarted.delete(meta["id"])
        assert folder.exists() and not run.dir.exists()  # Exported corrections are kept, as for stain runs.
    finally:
        restarted.close(save=False)


@pytest.mark.parametrize("value", [{"start_from":"run"}, {"count":True}, {"count":0}, {"size":32}])
def test_direct_patch_settings_rejected_before_item_creation(patches, value):
    client, runs, _, _ = patches
    before = set(runs.runs)
    response = client.post("/api/v1/runs",json=dict(sample="image-sample",kind="patches",
        region={"kind":"full"},stain={**SETTINGS,**value}))
    assert response.status_code == 422 and set(runs.runs) == before


PREVIEW = "/api/v1/samples/image-sample/patches/preview"


@pytest.mark.parametrize("start,method", [("empty", "cellpose"), ("model", "cellpose"), ("model", "instanseg")])
def test_preview_boxes_become_the_items_first_draft_in_its_frame(patches, monkeypatch, start, method):
    client, runs, _, _ = patches
    monkeypatch.setattr(runs.stain, "launch", lambda *a: pytest.fail("segmentation started"))
    settings = {**SETTINGS, "start_from": start, "count": 3, "size": 64}
    if method == "instanseg":
        settings.update(method=method, model="fluorescence_nuclei_and_cells", channels=["DAPI", "Boundary"])
    if start == "empty":
        monkeypatch.setattr(runs.stain, "models", lambda *a, **k: pytest.fail("empty patches loaded a model list"))
    region = {"kind": "box", "box": [50.5, 40.2, 350, 290]}   # the image window starts at pixel (50, 40)
    preview = client.post(PREVIEW, json={"region": region, "stain": settings})
    assert preview.status_code == 200, preview.text
    found = preview.json()
    boxes = [p["box"] for p in found["patches"]]
    assert found["run_pixel_to_sample"] == [50, 40, 1, 1] and not found["message"]
    assert len(boxes) == 3 and all(0 <= b[0] < b[2] <= 300 and 0 <= b[1] < b[3] <= 250 for b in boxes)
    created = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches", region=region,
                                                    stain={**settings, "boxes": boxes}))
    assert created.status_code == 200, created.text
    url = f"/api/v1/runs/{created.json()['id']}/stain/patches"
    draft = client.get(url).json()
    assert draft["version"] == 1 and draft["patches"] == found["patches"] and draft["count"] == 3
    assert draft["run_pixel_to_sample"] == found["run_pixel_to_sample"] and draft["start_from"] == start
    # The item's own suggestion uses the same frame: the same region, settings and seed give the same boxes.
    again = client.post(url+"/propose", json={"version": 1})
    assert again.status_code == 200, again.text
    assert again.json()["patches"] == found["patches"]


def test_preview_writes_nothing_and_says_when_fewer_fit(patches):
    client, runs, _, _ = patches
    data = runs.root.parent
    def tree(): return {p.relative_to(data) for p in data.rglob("*")}
    before, items = tree(), set(runs.runs)
    found = client.post(PREVIEW, json={"region": {"kind": "box", "box": [0, 0, 150, 150]},
                                       "stain": {**SETTINGS, "count": 3, "size": 128}})
    assert found.status_code == 200, found.text
    assert len(found.json()["patches"]) == 1
    assert found.json()["message"] == ("Found 1 of 3 requested patches. "
                                       "Try a smaller size or another region for more.")
    # The shared helper started by this first call keeps its log in the helpers' cache folder; nothing else.
    assert tree() - before <= {Path("stain-cache"), Path("stain-cache/stain-patches.log")}
    after = tree()
    assert client.post(PREVIEW, json={"region": {"kind": "full"}, "stain": SETTINGS}).status_code == 200
    assert tree() == after and set(runs.runs) == items


def test_preview_gives_the_boxes_frame_and_does_not_wait_for_the_model_list(patches, monkeypatch):
    """The page keeps moved boxes inside the region with `extent` (and a lasso's `outline` in pixels); the boxes do
    not depend on the model, so a slow or failing model list never delays or refuses the preview."""
    client, runs, _, _ = patches
    monkeypatch.setattr(runs.stain, "models", lambda *a, **k: pytest.fail("the preview consulted the model list"))
    box = client.post(PREVIEW, json={"region": {"kind": "box", "box": [50.5, 40.2, 350, 290]},
                                     "stain": {**SETTINGS, "count": 2, "size": 64}})
    assert box.status_code == 200, box.text
    assert box.json()["extent"] == [300, 250] and "outline" not in box.json()
    lasso = client.post(PREVIEW, json={"region": {"kind": "polygon", "points": [[60, 50], [340, 50], [60, 280]]},
                                       "stain": {**SETTINGS, "count": 1, "size": 64}})
    assert lasso.status_code == 200, lasso.text
    ox, oy = lasso.json()["run_pixel_to_sample"][:2]
    assert lasso.json()["outline"] == [[60 - ox, 50 - oy], [340 - ox, 50 - oy], [60 - ox, 280 - oy]]
    w, h = lasso.json()["extent"]
    assert all(0 <= x <= w and 0 <= y <= h for x, y in lasso.json()["outline"])


@pytest.mark.parametrize("case", ["no tissue", "region without tissue", "missing channel", "count", "size"])
def test_preview_refuses_with_the_same_plain_reason_as_start(patches, monkeypatch, case):
    client, runs, _, _ = patches
    region, settings = {"kind": "full"}, dict(SETTINGS)
    if case == "no tissue":
        monkeypatch.setattr(runs.zarr, "tissue", lambda *a: {"source": "none", "polygons": [], "area_um2": 0.})
        message = "No tissue was found in this sample. Untick Tissue only to use it anyway."
    elif case == "region without tissue":
        monkeypatch.setattr(runs.zarr, "tissue", lambda *a: {
            "source": "region_of_interest", "polygons": [[[0,0],[50,0],[50,50],[0,50]]], "area_um2": 2500.})
        region = {"kind": "box", "box": [200, 200, 300, 290]}
        message = "This region has no tissue. Untick Tissue only to use it anyway."
    elif case == "missing channel":
        del settings["nucleus"]
        message = "Choose channels from this image"
    elif case == "count":
        settings["count"] = 0
        message = "Choose Model outlines or Empty, 1–64 patches and a size of 64 to 4096 pixels"
    else:
        region, settings["size"] = {"kind": "box", "box": [0, 0, 100, 100]}, 128
        message = "Choose a smaller patch size for this region"
    before, folders = set(runs.runs), set(runs.root.iterdir())
    refused = client.post(PREVIEW, json={"region": region, "stain": settings})
    assert refused.status_code == 422 and refused.json()["detail"] == message
    if case != "size":   # creating an item checks the settings; whether patches of that size fit is the boxes' check
        start = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches", region=region,
                                                      stain=settings))
        assert start.status_code == 422 and start.json()["detail"] == message
    assert set(runs.runs) == before and set(runs.root.iterdir()) == folders


@pytest.mark.parametrize("boxes,message", [
    ([[0,0,10,10]], "Box 1 must be 16 to 4096 pixels on each side"),
    ([[0,0,600,128]], "Keep box 1 inside the region"),
    ([[False,0,128,128]], "Draw box 1 inside the region"),
    ("bad", "Choose between 1 and 64 patches"),
    ([[0,0,32,32]]*65, "Choose between 1 and 64 patches"),
    ([[300,200,380,280]], "Keep box 1 inside the region"),     # outside the lasso
])
def test_start_with_invalid_boxes_creates_nothing(patches, boxes, message):
    client, runs, _, _ = patches
    before, folders = set(runs.runs), set(runs.root.iterdir())
    response = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches",
        region={"kind": "polygon", "points": [[0,0],[400,0],[0,300]]}, stain={**SETTINGS, "boxes": boxes}))
    assert response.status_code == 422 and response.json()["detail"] == message
    assert set(runs.runs) == before and set(runs.root.iterdir()) == folders


def test_start_with_every_box_dropped_keeps_an_empty_draft(patches):
    client, _, _, _ = patches
    created = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches", region={"kind": "full"},
                                                    stain={**SETTINGS, "boxes": []}))
    assert created.status_code == 200, created.text
    draft = client.get(f"/api/v1/runs/{created.json()['id']}/stain/patches").json()
    assert draft["version"] == 1 and draft["patches"] == [] and draft["run_pixel_to_sample"] == [0, 0, 1, 1]


def test_preview_route_bounds_bytes_auth_and_origin(patches):
    from sopa_companion.server.app import MAX_STAIN_BYTES
    client, _, _, _ = patches
    assert client.post(PREVIEW, json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post(PREVIEW, json={}, headers={"Origin": "https://elsewhere.test"}).status_code == 403
    raw = b"x"*(MAX_STAIN_BYTES+1)
    assert client.post(PREVIEW, content=raw, headers={"Content-Type": "application/json"}).status_code == 413
    assert client.post(PREVIEW, content='{"stain":{"size":1e999}}',
                       headers={"Content-Type": "application/json"}).status_code == 422


def test_boxes_propose_move_drop_add_persist_and_export(patches, chosen):
    client, runs, run, url = patches
    proposed = client.post(url+"/patches/propose", json={"version": 0, "count": 8, "size": 128})
    assert proposed.status_code == 200, proposed.text
    value = proposed.json()
    assert 1 <= len(value["patches"]) <= 8 and value["run_pixel_to_sample"] == [0,0,1,1]
    boxes = [p["box"] for p in value["patches"]][:-1] + [[0,128,128,256]]
    boxes[0] = [1,1,129,129]
    moved = client.post(url+"/patches", json={"version": value["version"], "boxes": boxes})
    assert moved.status_code == 200, moved.text
    assert client.post(url+"/patches", json={"version": value["version"], "boxes": boxes}).status_code == 409
    assert json.loads((run.dir/"patches.json").read_text())["patches"][0]["box"] == boxes[0]
    runs.stain.worker.close()
    assert client.get(url+"/patches").json()["patches"] == moved.json()["patches"]
    result = exported(patches, chosen / "patches")
    folder = Path(result["exported"]["folder"])
    assert len(list(folder.glob("*.tif"))) == len(list(folder.glob("*_seg.npy"))) == 2
    assert result["exported"]["channels"] == ["DAPI","Boundary",None]
    assert [s["state"] for s in result["exported"]["states"]] == ["untouched"]*2
    assert (folder/F.META/"sets"/f"{result['set']}.json").is_file() and not (folder/"patches.json").exists()
    # The export's manifest is the set's own: its stem and sample's name, the chosen folder's layout, not the draft's map.
    manifest = json.loads((folder/F.META/"sets"/f"{result['set']}.json").read_text())
    assert "exports" not in manifest and manifest["layout"] == "flat" and manifest["sample_name"] == "Test"
    assert manifest["stem"].startswith("Test-") and F.valid_manifest(manifest, set=result["set"]) is not None


def test_watch_corrections_missing_and_unchanged_save(patches, monkeypatch, chosen):
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    folder = Path(value["exported"]["folder"])
    first, second = value["exported"]["patches"]
    path = folder/first["mask"]
    np.save(path, {**np.load(path, allow_pickle=True).item(), "manual_changes": []})
    assert client.get(url+"/patches").json()["exported"]["corrected"] == 0
    correct(folder, first["mask"])
    value = client.get(url+"/patches").json()
    assert value["exported"]["corrected"] == 1 and value["exported"]["cells"] == 3
    # The steady watch uses stat only; no helper, image read or mask decode.
    with monkeypatch.context() as patch:
        patch.setattr(runs.stain.worker, "call", lambda *a, **k: pytest.fail("unchanged poll decoded masks"))
        assert client.get(url+"/patches").json()["exported"]["states"] == value["exported"]["states"]
    (folder/second["mask"]).write_bytes(b"half saved")
    assert client.get(url+"/patches").json()["exported"]["states"][1]["state"] == "missing"
    (folder/first["image"]).unlink()
    assert client.get(url+"/patches").json()["exported"]["states"][0]["state"] == "missing"


def test_a_mask_that_became_a_pipe_or_a_link_reads_as_missing_at_once(patches, chosen):
    """The folder may be one others can write to: a FIFO put in a mask's place is never waited on (the patch helper
    would wait for a writer, holding every patch call of every run until its own call timed out a minute later and
    the poll failed) and a link is never read through; each such patch reads as missing, the others as they are."""
    _, runs, run, _ = patches
    manager = runs.stain.patches
    value = exported(patches, chosen / "patches")
    folder = Path(value["exported"]["folder"])
    first, second = (folder/p["mask"] for p in value["exported"]["patches"])
    second.unlink()
    os.mkfifo(second)
    answers = []
    poll = threading.Thread(target=lambda: answers.append(manager.status(run)), daemon=True)
    poll.start()
    poll.join(30)
    waited = poll.is_alive()
    if waited:                              # the helper waits for a writer: one comes and goes, so the poll ends
        os.close(os.open(second, os.O_WRONLY | os.O_NONBLOCK))
        poll.join(90)
    assert not waited, "the poll waited on the planted pipe"
    assert [p["state"] for p in answers[0]["exported"]["states"]] == ["untouched", "missing"]
    elsewhere = chosen / "elsewhere_seg.npy"
    shutil.copy(first, elsewhere)
    first.unlink()
    first.symlink_to(elsewhere)
    assert [p["state"] for p in manager.status(run)["exported"]["states"]] == ["missing", "missing"]


@pytest.mark.parametrize("body", [{"count":0},{"count":True},{"size":63},{"size":512},{"size":4097},{"count":65}])
def test_invalid_proposals(patches, body):
    client, _, run, url = patches
    result = client.post(url+"/patches/propose", json={"version":0, **body})
    assert result.status_code == 422, result.text
    assert not (run.dir/"patches.json").exists()


@pytest.mark.parametrize("boxes", [[[-1,0,128,128]],[[0,0,600,128]],[[0,0,10,10]],[[False,0,128,128]],"bad"])
def test_invalid_boxes(patches, boxes):
    client, _, _, url = patches
    assert client.post(url+"/patches", json={"version":0, "boxes":boxes}).status_code == 422


def test_free_patch_sizes_shortfall_message_and_non_square_export(patches, chosen):
    from sopa_companion.engine.cellpose_settings import gui_settings
    client, _, _, url = patches
    version = 0
    for size in (64, 97, 256):
        result = client.post(url+"/patches/propose", json={"version":version, "count":3, "size":size})
        assert result.status_code == 200, result.text
        data = result.json()
        version = data["version"]
        assert all(p["box"][2]-p["box"][0] == size for p in data["patches"])
    assert len(data["patches"]) == 1 and "1 of 3" in data["message"]
    assert client.get(url+"/patches").json()["message"] == data["message"]
    changed = client.post(url+"/patches", json={"version":version, "boxes":[[0,0,300,16]]})
    assert changed.status_code == 200 and not changed.json()["message"]
    saved = export(client, url+"/patches", changed.json()["version"], chosen / "patches")
    assert saved.status_code == 200, saved.text
    manifest = saved.json()["exported"]
    folder = Path(manifest["folder"])
    data = np.load(folder/manifest["patches"][0]["mask"], allow_pickle=True).item()
    assert data["masks"].shape == (16,300) and data["diameter"] == 0
    assert data["filename"] == str(folder/manifest["patches"][0]["image"])
    source = {"folder": folder, "manifest": manifest, "set": manifest["set"], "run": manifest["run"]}
    assert gui_settings(source, []) is None and "manual_changes" not in data


def test_removed_areas_reach_helper_and_invalidate_cached_result(patches, chosen):
    client, runs, run, url = patches
    before = exported(patches, chosen / "patches")
    path = run.dir/"stain/exclusions.json"
    revision = run.meta["progress"]["revision"]
    area = {"id":1, "points":[[14,14],[16,14],[16,16],[14,16]]}
    path.write_text(json.dumps({"version":1, "areas":[area]}))
    runs._set(run, exclusions={"version":1, "count":1, "removed":1})
    after = export(client, url+"/patches", before["version"], chosen / "patches")
    assert after.status_code == 200, after.text
    assert after.json()["exported"]["states"][0]["cells"] == 1  # Original first patch had two whole cells.
    path.write_text(json.dumps({"version":2, "areas":[]}))
    runs._set(run, exclusions={"version":2, "count":0, "removed":0})
    restored = export(client, url+"/patches", after.json()["version"], chosen / "patches")
    assert restored.status_code == 200, restored.text
    assert restored.json()["exported"]["states"][0]["cells"] == 2
    assert run.meta["progress"]["revision"] == revision
    # Also key the rings themselves: stale metadata must not reuse another removal's geometry.
    path.write_text(json.dumps({"version":2, "areas":[{"id":2, "points":[[0,0],[130,0],[130,300],[0,300]]}]}))
    result = client.post(url+"/patches/propose", json={"version":restored.json()["version"], "size":64, "count":3})
    assert result.status_code == 200, result.text
    assert all((p["box"][0]+p["box"][2])/2 > 130 for p in result.json()["patches"])


@pytest.mark.parametrize("endpoint", ["patches","patches/propose","patches/export","patches/open"])
def test_remaining_routes_bound_bytes_auth_and_origin(patches, endpoint):
    from sopa_companion.server.app import MAX_STAIN_BYTES
    client, _, _, url = patches
    target = url+"/"+endpoint
    assert client.post(target,json={},headers={"Authorization":"Bearer wrong"}).status_code == 401
    assert client.post(target,json={},headers={"Origin":"https://elsewhere.test"}).status_code == 403
    raw = b"x"*(MAX_STAIN_BYTES+1)
    for content in (raw, iter([raw[:100],raw[100:]])):
        assert client.post(target,content=content,headers={"Content-Type":"application/json"}).status_code == 413
    assert client.post(target,content='{"version":1e999}',headers={"Content-Type":"application/json"}).status_code == 422


def test_no_display_fallback_and_fake_detached_launcher(patches, chosen):
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    runs.stain.patches.window = None
    result = client.post(url+"/patches/open", json={"set":value["set"]}).json()
    assert not result["running"] and "stain_gui" in result["command"] and result["folder"]
    runs.stain.patches.launcher = "sopa_companion.tests.fake_stain_gui"
    result = client.post(url+"/patches/open", json={"set":value["set"]})
    assert result.status_code == 200 and result.json()["running"]
    folder = Path(result.json()["folder"])
    wait_for(lambda: (folder/"gui-launch.json").is_file())
    recorded = json.loads((folder/"gui-launch.json").read_text())
    assert recorded["display"] is None and recorded["gpu"] == "" and recorded["nice"] == os.getpriority(os.PRIO_PROCESS, 0)
    assert recorded["session"] == recorded["pid"]
    # Started in its own cache folder, never on the patches' folder, with that folder kept off its import path.
    assert os.path.realpath(recorded["cwd"]) == os.path.realpath(runs.stain.patches.gui_cache())
    assert recorded["safe_path"] == "1" and recorded["image"] == str(folder/value["exported"]["patches"][0]["image"])
    assert not (folder/".cache").exists() and (folder/F.META/"gui.json").is_file()
    assert recorded["log_blocking"]           # its log, opened without waiting, is handed over as any log
    runs.stain.patches.processes[str(folder)].wait(timeout=5)
    assert not client.get(url+"/patches").json()["exported"]["gui_running"]


def test_a_gui_that_closes_with_an_error_says_why(patches, chosen):
    """A GUI this engine started that closes with an error: the panel's status names the line of its own launch's log
    that reads as an error (not a line it printed after it); one a signal ended before it wrote anything names the
    signal, not an earlier launch's line; one that closed as usual, or long ago, names nothing. A Qt that could open no
    window (a new WSL2 or server install without the system's display libraries; no display to reach) is said in the
    app's words, with the line that installs the libraries, never as Qt's own "Reinstalling the application may fix
    this problem"."""
    from sopa_companion.engine.patch_folders import GUI_LIBRARIES
    from sopa_companion.engine.stain_patches import GUI_ERROR_S
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    manager = runs.stain.patches
    manager.window, manager.launcher = None, "sopa_companion.tests.fake_stain_gui"
    folder = Path(value["exported"]["folder"])
    def launch(flag=None):
        for name in ("fail-gui", "kill-gui", "qt-libraries-gui", "qt-display-gui"):
            (folder/name).unlink(missing_ok=True)
        if flag:
            (folder/flag).touch()
        assert client.post(url+"/patches/open", json={"set": value["set"]}).json()["running"]
        process = manager.processes[str(folder)]
        return process, process.wait(timeout=10)
    process, code = launch("fail-gui")
    failed = client.get(url+"/patches").json()["exported"]["gui_error"]
    assert code == 1 and failed == {"pid": process.pid, "code": 1, "reason": "ModuleNotFoundError: No module named 'qtpy'"}
    process, code = launch("kill-gui")
    failed = client.get(url+"/patches").json()["exported"]["gui_error"]
    assert code == -9 and failed == {"pid": process.pid, "code": -9, "reason": "SIGKILL"}
    process, code = launch("qt-libraries-gui")
    reason = client.get(url+"/patches").json()["exported"]["gui_error"]["reason"]
    assert code == 134 and reason.startswith("its window could not open, because this Linux system is missing the libraries")
    assert reason.endswith("then open it again: " + GUI_LIBRARIES) and "Reinstalling" not in reason
    assert GUI_LIBRARIES.startswith("sudo apt install -y ") and "libxcb-cursor0" in GUI_LIBRARIES.split()
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert GUI_LIBRARIES in readme.splitlines(), "the README gives the same line under Optional extras"
    process, code = launch("qt-display-gui")
    reason = client.get(url+"/patches").json()["exported"]["gui_error"]["reason"]
    assert code == 134 and reason == "its window could not open, because it could not reach this computer's display."
    manager.exits[str(folder)] = (process.pid, manager.exits[str(folder)][1] - GUI_ERROR_S - 1)
    assert client.get(url+"/patches").json()["exported"]["gui_error"] is None
    process, code = launch()
    assert code == 0 and client.get(url+"/patches").json()["exported"]["gui_error"] is None


def test_a_gui_another_engine_opened_after_this_ones_closed_counts_as_open(patches, chosen):
    """One GUI per folder across engines: once this engine's own GUI on the folder has closed, the record names the one
    another engine opened there since (here a stand-in process), and Open answers running without starting a second."""
    import psutil
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    manager = runs.stain.patches
    manager.window, manager.launcher = None, "sopa_companion.tests.fake_stain_gui"
    folder = Path(value["exported"]["folder"])
    assert client.post(url+"/patches/open", json={"set": value["set"]}).json()["running"]
    own = manager.processes[str(folder)]
    assert own.wait(timeout=10) == 0
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = {"pid": other.pid, "started": psutil.Process(other.pid).create_time()}
        (folder/F.META/"gui.json").write_text(json.dumps(record))
        assert client.get(url+"/patches").json()["exported"]["gui_running"]
        assert client.post(url+"/patches/open", json={"set": value["set"]}).json()["running"]
        assert manager.processes[str(folder)] is own                # no second GUI was started
    finally:
        other.kill()
        other.wait()
    assert not client.get(url+"/patches").json()["exported"]["gui_running"]


def test_a_gui_without_its_packages_is_refused_with_the_command_that_installs_them(patches, chosen, monkeypatch):
    """With a desktop, the GUI's interpreter is asked for its packages before it starts: without them nothing starts,
    and the refusal names the interpreter and the command that installs them (also on its own, for the panel)."""
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    manager = runs.stain.patches
    monkeypatch.setattr(manager, "display", lambda: {"DISPLAY": ":5"})
    monkeypatch.setattr(manager, "gui_lacking", lambda: ["qtpy", "PyQt6"])
    monkeypatch.setattr(manager, "open_log", lambda meta: pytest.fail("the real GUI was started"))
    refused = client.post(url+"/patches/open", json={"set": value["set"]})
    install = shlex.join([manager.stain.python, "-m", "pip", "install", "cellpose[gui]"])
    assert refused.status_code == 503 and refused.json()["command"] == install
    assert refused.json()["detail"] == F.words("gui_missing", python=manager.stain.python, command=install)
    folder = Path(value["exported"]["folder"])
    assert not manager.processes and not (folder/F.META/"gui.json").exists()


def test_the_gui_probe_asks_its_interpreter_and_trusts_one_found_complete(patches, tmp_path, monkeypatch):
    """gui_lacking runs the GUI's interpreter on GUI_PROBE: what it lacks, as it lists them; [] when it cannot answer
    (the GUI then starts as before); an interpreter found complete is not asked again."""
    from importlib.util import find_spec
    _, runs, _, _ = patches
    manager = runs.stain.patches
    def interpreter(name, body):
        path = tmp_path / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
        return str(path)
    monkeypatch.setattr(manager.stain, "python", interpreter("lacks", "echo '[\"qtpy\", \"PyQt6\"]'\n"))
    assert manager.gui_lacking() == ["qtpy", "PyQt6"]
    broken = interpreter("broken", "exit 3\n")
    monkeypatch.setattr(manager.stain, "python", broken)
    assert manager.gui_lacking() == [] and broken not in manager.gui_ready
    complete = interpreter("complete", "echo '[]'\n")
    monkeypatch.setattr(manager.stain, "python", complete)
    assert manager.gui_lacking() == [] and complete in manager.gui_ready
    Path(complete).write_text("#!/bin/sh\necho '[\"qtpy\"]'\n")
    assert manager.gui_lacking() == []                      # not asked again: asked, it would now lack qtpy
    # The probe itself, in this interpreter: what it finds missing here.
    monkeypatch.setattr(manager.stain, "python", sys.executable)
    lacking = [m for m in ("qtpy", "pyqtgraph", "superqt") if find_spec(m) is None]
    if not any(find_spec(m) for m in ("PyQt6", "PyQt5", "PySide6", "PySide2")):
        lacking.append("PyQt6")
    assert manager.gui_lacking() == lacking


def test_the_command_to_open_the_gui_by_hand_keeps_its_working_folder_off_the_import_path(patches, chosen):
    client, runs, _, url = patches
    value = exported(patches, chosen / "patches")
    runs.stain.patches.window = None
    command = shlex.split(client.post(url+"/patches/open", json={"set": value["set"]}).json()["command"])
    first = Path(value["exported"]["folder"]) / value["exported"]["patches"][0]["image"]
    assert command[0] == "env" and command[1].startswith("PYTHONPATH=") and command[2] == "PYTHONSAFEPATH=1"
    assert command[-3:] == ["-m", "sopa_companion.zarr_bridge.stain_gui", str(first)]


def test_one_gui_per_folder_whichever_set_opened_it_and_across_a_manager_restart(patches, chosen):
    from sopa_companion.engine.stain_patches import StainPatches
    client, runs, run, url = patches
    value = exported(patches, chosen / "patches")
    folder = Path(value["exported"]["folder"])
    second = export(client, url+"/patches", value["version"], folder)
    assert second.status_code == 200, second.text
    second = second.json()
    flag = folder/"keep-gui"; flag.touch()
    manager = runs.stain.patches
    manager.launcher = "sopa_companion.tests.fake_stain_gui"
    manager.window = None
    try:
        assert manager.open_gui(run,{"set":value["set"]})["running"]
        process = manager.processes[str(folder)]
        wait_for(lambda:(folder/"gui-launch.json").exists())
        record = json.loads((folder/F.META/"gui.json").read_text())
        assert record["pid"] == process.pid and not (folder/"gui.json").exists()
        assert manager.open_gui(run,{"set":value["set"]})["running"]
        assert manager.open_gui(run,{"set":second["set"]})["running"]          # the other set in the same folder
        assert manager.processes[str(folder)] is process and len(manager.processes) == 1
        assert client.get(url+"/patches").json()["exported"]["gui_running"]
        restarted = StainPatches(runs.stain,{"gui_module":"sopa_companion.tests.fake_stain_gui"})
        assert restarted.open_gui(run,{"set":second["set"]})["running"]
        assert not restarted.processes
    finally:
        flag.unlink()
        if str(folder) in manager.processes:
            manager.processes[str(folder)].wait(timeout=5)
    assert not restarted.gui_running(restarted.selected(run, second["set"]))


def test_display_comes_from_app_process_and_refuses_harness(patches, monkeypatch):
    from types import SimpleNamespace
    _, runs, _, _ = patches
    monkeypatch.delenv("PLH_TAG", raising=False)
    monkeypatch.delenv("PLH_DISPLAY", raising=False)
    env = {"DISPLAY":":42", "XAUTHORITY":"/unused/auth", "XDG_RUNTIME_DIR":"/unused/runtime"}
    class Window:
        found = SimpleNamespace(environ=lambda:env)
        def __call__(self): return True
    manager = runs.stain.patches
    manager.window = Window()
    assert manager.display() == env
    env["DISPLAY"] = ":240"
    assert manager.display() == {}
    env["DISPLAY"] = ":42"
    env["PLH_TAG"] = "fake"
    assert manager.display() == {}
    del env["PLH_TAG"]
    monkeypatch.setenv("PLH_TAG","fake")
    assert manager.display() == {}


def test_the_gui_opens_on_the_app_windows_desktop_else_where_a_page_is_open_in_any_browser(patches, monkeypatch):
    """One rule for every browser (the engine never reads which one shows the page): the app's own Chrome window's
    desktop when it is found (browser = "chrome"); else this engine's own desktop while a page is open, in any browser,
    on a desktop session (server/app.py page_here); else none (the page shows the command instead)."""
    from types import SimpleNamespace
    _, runs, _, _ = patches
    for name in ("PLH_TAG", "PLH_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setenv("XAUTHORITY", "/engine/auth")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/engine/runtime")
    chrome_env, here = {"DISPLAY": ":42", "XAUTHORITY": "/chrome/auth"}, [False]
    class Window:
        alive = False
        found = SimpleNamespace(environ=lambda: chrome_env)
        def __call__(self): return self.alive
    manager = runs.stain.patches
    manager.window, manager.page_here = Window(), lambda: here[0]
    assert manager.display() == {}                                        # neither the window nor a page
    here[0] = True                                                        # a page open in, say, Firefox
    assert manager.display() == {"DISPLAY": ":7", "XAUTHORITY": "/engine/auth", "XDG_RUNTIME_DIR": "/engine/runtime"}
    Window.alive, here[0] = True, False                                   # the app's Chrome, whose page went away
    assert manager.display() == chrome_env
    here[0] = True
    assert manager.display() == chrome_env                                # the window first
    def broken():
        raise RuntimeError("no page count")
    Window.alive, manager.page_here = False, broken
    assert manager.display() == {}


def test_the_gui_opens_on_a_mac_or_windows_desktop_without_x11(patches, monkeypatch):
    """macOS and Windows need no X11 display to show a window: there the GUI opens while a page of this engine is open on
    this computer's own desktop session (page_here), with no display variables; without a page, or in the harness, the
    page shows the command instead. Linux keeps its X11 rule (display())."""
    import sopa_companion.engine.stain_patches as SP
    _, runs, _, _ = patches
    for name in ("PLH_TAG", "PLH_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    manager, here = runs.stain.patches, [False]
    manager.window, manager.page_here = None, lambda: here[0]
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(SP.sys, "platform", platform)
        here[0] = False
        assert manager.gui_desktop() is None
        here[0] = True
        assert manager.gui_desktop() == {}
        monkeypatch.setenv("PLH_TAG", "fake")
        assert manager.gui_desktop() is None
        monkeypatch.delenv("PLH_TAG")
    monkeypatch.setattr(SP.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert manager.gui_desktop() is None                                  # a page, but no X11 display: Linux's rule
    monkeypatch.setenv("DISPLAY", ":7")
    assert manager.gui_desktop() == {"DISPLAY": ":7", **{k: v for k, v in os.environ.items()
                                                         if k in ("XAUTHORITY", "XDG_RUNTIME_DIR")}}


def test_an_export_waits_for_its_patches_longer_than_a_read():
    """An export writes every patch before the helper answers (up to 64 of up to 4096 x 4096 pixels): it waits as long
    as a model's export may run, a read 60 s, so a long export is never cut off and called unreadable."""
    from sopa_companion.engine import stain_worker as W
    assert W.WAIT_S["export"] == 900 and W.READ_WAIT_S == 60
    assert all(W.WAIT_S.get(action, W.READ_WAIT_S) == 60 for action in ("propose", "boxes", "preview", "inspect"))


def test_the_engine_tells_the_gui_whether_a_page_is_open_on_its_desktop(config, monkeypatch):
    """server/app.py: page_here is a page open now (Presence) on a desktop session (desktop/browser.py desktop_session);
    the app window is still looked for by its Chrome profile."""
    from sopa_companion.desktop.window import AppWindow, chrome_profile
    application = create_app(config, token=TOKEN)
    patches = application.state.runs.stain.patches
    assert isinstance(patches.window, AppWindow) and patches.window.flag == "--user-data-dir=" + str(chrome_profile(config))
    monkeypatch.setenv("DISPLAY", ":7")
    assert patches.page_here() is False
    application.state.presence.connected("page-1")
    assert patches.page_here() is True
    monkeypatch.delenv("DISPLAY")
    assert patches.page_here() is False


def test_reexport_into_the_same_folder_keeps_earlier_sets_and_their_corrections(patches, chosen):
    client, runs, run, url = patches
    first = exported(patches, chosen / "patches")
    folder = Path(first["exported"]["folder"])
    mask = first["exported"]["patches"][0]["mask"]
    correct(folder, mask)
    corrected = (folder/mask).read_bytes()
    newer = export(client, url+"/patches", first["version"], folder)
    assert newer.status_code == 200, newer.text
    assert newer.json()["set"] != first["set"] and newer.json()["exported"]["corrected"] == 0
    assert newer.json()["exported"]["stem"] != first["exported"]["stem"]
    assert (folder/mask).read_bytes() == corrected and len(list(folder.glob("*_seg.npy"))) == 4
    assert {p.stem for p in (folder/F.META/"sets").iterdir()} == {first["set"], newer.json()["set"]}
    assert runs.stain.patches.selected(run,first["set"]).folder == folder


def test_an_export_without_a_folder_is_refused_before_anything_starts(patches, monkeypatch):
    """An older page posting {version} only, or an empty field: the folder's words (422, field "folder") before the
    run is read, the helper asked or anything staged; the boxes' version stays."""
    client, runs, run, url = patches
    version = client.get(url+"/patches").json()["version"]
    monkeypatch.setattr(runs.stain.worker, "call", lambda *a, **k: pytest.fail("the helper was asked"))
    for body in ({"version": version}, {"version": version, "folder": ""}, {"version": version, "folder": 7}):
        refused = client.post(url+"/patches/export", json=body)
        assert refused.status_code == 422 and refused.json() == {"detail": F.words("empty"), "status": 422,
                                                                 "field": "folder"}
    staging = runs.stain.patches.staging
    assert not staging.exists() or not list(staging.iterdir())
    assert client.get(url+"/patches").json()["version"] == version


def test_an_export_into_a_zarr_store_is_refused(patches, chosen):
    client, _, _, url = patches
    store = chosen / "sample"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    version = client.get(url+"/patches").json()["version"]
    for folder in (store / "images" / "patches", chosen / "other.zarr" / "patches"):
        refused = export(client, url+"/patches", version, folder)
        assert refused.status_code == 422 and refused.json()["field"] == "folder"
        assert "is a zarr store" in refused.json()["detail"] and not folder.parent.exists()
    assert sorted(p.name for p in store.iterdir()) == ["zarr.json"]


def test_two_runs_of_two_samples_export_into_one_folder(patches, chosen):
    client, runs, run, url = patches
    folder = chosen / "shared"
    first = exported(patches, folder)
    created = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="patches", region={"kind": "full"},
                                                    stain={**SETTINGS, "start_from": "empty", "count": 1, "size": 64}))
    assert created.status_code == 200, created.text
    other = runs.get(created.json()["id"])
    runs._set(other, sample="other-sample", sample_name="Other sample")
    items = f"/api/v1/runs/{other.meta['id']}/stain/patches"
    boxes = client.post(items, json={"version": 0, "boxes": [[0, 0, 64, 64]]}).json()
    second = export(client, items, boxes["version"], folder)
    assert second.status_code == 200, second.text
    assert second.json()["exported"]["stem"].startswith("Other-sample-") and first["exported"]["stem"].startswith("Test-")
    listed = client.get(url+"/patches/folders", params={"path": str(folder)}).json()
    assert listed["holds"] == {"patches": 3, "sets": 2, "samples": ["Other sample", "Test"], "more": 0}
    assert len(list(Path(listed["path"]).glob("*.tif"))) == 3


def test_a_folder_deleted_reads_as_missing_open_says_so_and_export_works_again(patches, chosen):
    client, runs, run, url = patches
    value = exported(patches, chosen / "patches")
    folder = Path(value["exported"]["folder"])
    shutil.rmtree(folder)
    status = client.get(url+"/patches")
    assert status.status_code == 200
    gone = status.json()["exported"]
    assert gone == {"set": value["set"], "folder": str(folder), "missing": True, "patches": [], "states": [],
                    "gui_running": False, "corrected": 0, "cells": 0}
    opened = client.post(url+"/patches/open", json={"set": value["set"]})
    assert opened.status_code == 404 and opened.json()["detail"] == F.words("gone", folder=folder)
    again = export(client, url+"/patches", status.json()["version"], chosen / "again")
    assert again.status_code == 200, again.text
    assert not again.json()["exported"].get("missing") and len(again.json()["exported"]["states"]) == 2


def test_running_run_and_removed_routes(patches):
    client, runs, run, url = patches
    runs._set(run,state="running",paused=False)
    assert client.post(url+"/patches/propose",json={"version":0}).status_code == 409
    runs._set(run,state="done")
    for endpoint in ("edits","undo","redo","carry"):
        assert client.post(url+"/"+endpoint,json={}).status_code in (404,405)


@pytest.mark.parametrize("case", ["compatibility","geometry","proposal","unsafe","empty","direct-proposal",
                                "pixel-edges", "tissue", "exclusions", "size-rule", "brightness-rule", "settings", "sizes", "model-pixels", "model-centroids",
                                "instanseg-1","instanseg-2","instanseg-3","instanseg-5","preview","stem"])
def test_patch_helper_in_installed_sopa(tmp_path, case):
    env = {**os.environ, "HOME":str(tmp_path), "CUDA_VISIBLE_DEVICES":"", "QT_QPA_PLATFORM":"offscreen",
           "NUMBA_CACHE_DIR":str(tmp_path/"numba"), "MPLCONFIGDIR":str(tmp_path/"mpl"),
           "XDG_CACHE_HOME":str(tmp_path/"cache"), "CELLPOSE_LOCAL_MODELS_PATH":str(tmp_path/"cellpose"),
           "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "PYTHONDONTWRITEBYTECODE":"1"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    check = Path(__file__).with_name("stain_patches_check.py")
    result = subprocess.run([sopa_python(),"-B",str(check),str(tmp_path),case],
                            env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr


def test_display_falls_back_to_the_engine_when_chrome_hides_its_environment(patches, monkeypatch):
    """Chrome overwrites its environment block with its process title: an open window then reports no DISPLAY."""
    from types import SimpleNamespace
    _, runs, _, _ = patches
    for key in ("PLH_TAG", "PLH_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISPLAY", ":77")
    monkeypatch.setenv("XAUTHORITY", "/unused/auth")
    class Window:
        found = SimpleNamespace(environ=lambda: {})
        def __call__(self): return True
    manager = runs.stain.patches
    manager.window = Window()
    assert manager.display()["DISPLAY"] == ":77" and manager.display()["XAUTHORITY"] == "/unused/auth"
    monkeypatch.setenv("DISPLAY", ":241")
    assert manager.display() == {}
    manager.window = None
    monkeypatch.setenv("DISPLAY", ":77")
    assert manager.display() == {}
