"""Tissue only, removed areas and the Cellpose GUI's settings on the engine side (no Cellpose, no zarr)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sopa_companion.engine import stain_exclusions as X
from sopa_companion.engine.runs import RunError, Runs
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples
from sopa_companion.tests.fake_stain import arrays
from test_stain import SETTINGS, Images, stain_cpu

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
TOKEN = "t" * 40
BASE = "http://127.0.0.1:8799"


def test_centres_and_areas_of_open_closed_and_flat_rings():
    rings = [SQUARE, [[20, 0], [30, 0], [20, 10]], [[0, 50], [10, 50], [20, 50]]]
    coords = np.array([p for r in rings for p in r], float)
    offsets = np.cumsum([0] + [len(r) for r in rings])
    centres, areas = X.centres_and_areas(offsets, coords)
    np.testing.assert_allclose(centres[0], [5, 5])
    np.testing.assert_allclose(centres[1], [70 / 3, 10 / 3])
    np.testing.assert_allclose(centres[2], [10, 50])            # flat ring: the mean of its points
    np.testing.assert_allclose(areas, [100, 50, 0])


def test_inside_follows_a_lasso_including_concave_ones():
    lasso = [[0, 0], [10, 0], [10, 10], [5, 3], [0, 10]]           # a notch from the top
    points = [[2, 2], [5, 8], [9.6, 9], [11, 5], [-1, -1]]       # (9.6, 9): the right prong; (5, 8): the notch
    assert X.inside(points, lasso).tolist() == [True, False, True, False, False]
    assert X.inside([], lasso).tolist() == [] and not X.inside([[1, 1]], [[0, 0], [1, 1]]).any()
    # On the edge or a corner counts as inside (the save's shapely intersects_xy agrees); just outside does not.
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert X.inside([[0, 5], [5, 0], [10, 10], [10, 5], [10.001, 5]], square).tolist() == [True, True, True, True, False]


class TissueImages(Images):
    def __init__(self, path, polygons):
        super().__init__(path)
        self.polygons = polygons

    def tissue(self, sid, key=None):
        return dict(source="region_of_interest", polygons=self.polygons, shapes=[[r] for r in self.polygons], area_um2=1.)


@pytest.fixture
def runs(config, tmp_path):
    config["proseg"]["keep_unsaved"] = True       # tissue choices are compared as separate test runs
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    manager = Runs(config, Samples(config), TissueImages(tmp_path / "unwritten.zarr",
                                                          [[[0, 0], [100, 0], [100, 300], [0, 300], [0, 0]],
                                                           [[900, 900], [950, 900], [950, 950], [900, 900]]]))
    yield manager
    manager.close(save=False)


def job(run):
    return json.loads((run.dir / "stain/job.json").read_text())


def test_tissue_only_passes_the_tissue_in_the_region_and_refuses_a_region_without_tissue(runs):
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "box", "box": [50, 50, 200, 200]},
                               stain=SETTINGS)["id"])
    assert job(run)["tissue"] == [[[[0, 0], [100, 0], [100, 300], [0, 300], [0, 0]]]]     # the far ring is left out
    assert job(run)["stitch"] == "own" and run.meta["stain"]["tissue_only"] is True
    assert run.meta["tissue"] == dict(source="region_of_interest", area_um2=1., tiles_skipped=None)
    with pytest.raises(RunError, match="no tissue"):
        runs.create("image-sample", kind="stain", region={"kind": "box", "box": [200, 50, 300, 200]}, stain=SETTINGS)
    off = runs.get(runs.create("image-sample", kind="stain", region={"kind": "box", "box": [200, 50, 300, 200]},
                               stain={**SETTINGS, "tissue_only": False})["id"])
    assert job(off)["tissue"] is None and off.meta["tissue"]["source"] == "off"
    with pytest.raises(RunError, match="inside the tissue"):
        runs.create("image-sample", kind="stain", region={"kind": "full"}, stain={**SETTINGS, "tissue_only": 1})


def test_more_cellpose_settings_are_checked_and_kept(runs):
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"}, stain={
        **SETTINGS, "niter": 300, "min_size": 30, "augment": True,
        "normalize": {"percentile": [2, 98], "tile_norm_blocksize": 128, "invert": True}})["id"])
    stain = job(run)["settings"]
    assert stain["niter"] == 300 and stain["min_size"] == 30 and stain["augment"] is True
    assert stain["normalize"]["percentile"] == [2., 98.] and stain["normalize"]["tile_norm_blocksize"] == 128
    with pytest.raises(RunError, match="between 0 and 100 for percentile"):
        runs.create("image-sample", kind="stain", region={"kind": "full"},
                    stain={**SETTINGS, "normalize": {"percentile": [2, 198]}})


def test_zarr_tissue_keeps_holes_reads_region_of_interest_or_detects_and_caches(config, tmp_path, monkeypatch):
    samples = ZarrSamples(config)
    monkeypatch.setattr(samples, "_image_info", lambda sid, key=None: {"key": "im"})
    monkeypatch.setattr(samples, "path", lambda sid: str(tmp_path / "s.zarr"))
    detail = {"shapes": [{"key": "region_of_interest"}], "points_key": "transcripts"}
    monkeypatch.setattr(samples, "detail", lambda sid: detail)
    hole = [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]
    calls = []
    answer = {"polygons": [[SQUARE, hole]], "area_um2": 84.}
    monkeypatch.setattr(samples, "helper", lambda command, args, **kw: calls.append((command, args.get("points_key"))) or
                        answer)
    stamp = [1.0]
    monkeypatch.setattr(samples, "_mtime", lambda path: stamp[0])
    found = samples.tissue("s")
    assert found["source"] == "region_of_interest" and found["shapes"] == [[SQUARE, hole]]
    assert found["polygons"] == [SQUARE, hole] and found["area_um2"] == 84.
    assert samples.tissue("s") == found and calls == [("tissue", "transcripts")]       # cached beside the app
    detail["shapes"] = []
    assert samples.tissue("s")["source"] == "detected" and calls[-1] == ("tissue", None)
    stamp[0] = 2.0
    samples.tissue("s")
    assert len(calls) == 3                                       # the sample changed: read again
    answer.update(polygons=[], area_um2=0.)
    stamp[0] = 3.0
    assert samples.tissue("s") == {"source": "none", "polygons": [], "shapes": [], "area_um2": 0.}


def test_no_tissue_in_the_sample_is_refused_while_tissue_only(config, tmp_path):
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    images = TissueImages(tmp_path / "unwritten.zarr", [])
    images.tissue = lambda sid, key=None: dict(source="none", polygons=[], shapes=[], area_um2=0.)
    manager = Runs(config, Samples(config), images)
    try:
        with pytest.raises(RunError, match="No tissue was found"):
            manager.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)
        assert manager.create("image-sample", kind="stain", region={"kind": "full"},
                              stain={**SETTINGS, "tissue_only": False})["tissue"]["source"] == "off"
    finally:
        manager.close(save=False)


@pytest.fixture
def client(config, tmp_path, monkeypatch):
    from sopa_companion.server import create_app
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    app = create_app(config, token=TOKEN)
    runs = app.state.runs
    runs.zarr = Images(tmp_path / "unwritten.zarr")
    monkeypatch.setattr(runs.stain, "launch", lambda run: None)
    with TestClient(app, base_url=BASE, headers={"Authorization": "Bearer " + TOKEN, "Origin": BASE}) as c:
        yield c, runs


def finished_run(client, runs):
    rid = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain", region={"kind": "full"},
                                                stain=SETTINGS)).json()["id"]
    run = runs.get(rid)
    final = run.dir / "stain/resolved-000001"
    final.mkdir()
    rings = [[[x, 10], [x + 10, 10], [x + 10, 20], [x, 20], [x, 10]] for x in (10, 40, 70)]
    for name, value in arrays(rings).items():
        np.save(final / f"{name}.npy", value)
    (run.dir / "output/boundaries.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": i + 1, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [r]}}
        for i, r in enumerate(rings)]}))
    runs._set(run, state="done", progress=dict(done=1, total=1, revision=2, resolved=True, complete=True),
              summary=dict(cells=3, area_um2=dict(p10=100, median=100, p90=100)))
    return rid, run


def cells(client, rid):
    from sopa_companion.engine import vectors
    value = vectors.decode(client.get(f"/api/v1/runs/{rid}/stain/vectors", params={"box": "0,0,100,100"},
                                      headers={"Accept-Encoding": "identity"}).content)
    return value["total"], len(value["cells"]["offsets"]) - 1


def saved_mask(runs, run) -> list[bool]:
    """The cells a Save would keep now: Runs._stain_keep over the committed rules, as the Save runs it."""
    return np.load(runs._stain_keep(run, runs.stain.exclusions.load(run)), allow_pickle=False).tolist()


def test_removed_areas_leave_cells_out_until_put_back(client):
    client, runs = client
    rid, run = finished_run(client, runs)
    url = f"/api/v1/runs/{rid}/stain/exclusions"
    initial = client.get(url).json()
    assert initial["version"] == initial["areas_version"] == initial["removed"] == 0
    assert initial["areas"] == [] and initial["size"] is None and initial["brightness"] == {}
    assert initial["cells"] == initial["kept"] == 3
    lasso = [[35, 5], [60, 5], [60, 30], [35, 30]]                # around the middle cell's centre (45, 15)
    assert client.post(url, json={"version": 3, "points": lasso}).status_code == 409
    for bad in ([[1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]], "x", [[0, 0], [1, "a"], [2, 0]]):
        assert client.post(url, json={"version": 0, "points": bad}).status_code == 422
    added = client.post(url, json={"version": 0, "points": lasso}).json()
    assert added["version"] == 1 and added["removed"] == 1 and added["areas"][0]["cells"] == 1
    detail = client.get(f"/api/v1/runs/{rid}").json()
    assert detail["exclusions"] == {k: v for k, v in added.items() if k != "areas"}
    assert detail["summary"]["cells"] == 2 and detail["summary_model"]["cells"] == 3
    assert cells(client, rid) == (2, 2)
    area = added["areas"][0]["id"]
    assert client.delete(f"{url}/{area}", params={"version": 0}).status_code == 409
    back = client.delete(f"{url}/{area}", params={"version": 1}).json()
    assert back["version"] == back["areas_version"] == 2 and back["removed"] == 0 and back["areas"] == []
    assert back["digest"] == initial["digest"]
    assert client.get(f"/api/v1/runs/{rid}").json()["summary"]["cells"] == 3 and cells(client, rid) == (3, 3)
    assert client.delete(f"{url}/{area}", params={"version": 2}).status_code == 404
    runs._set(run, state="running")
    assert client.post(url, json={"version": 2, "points": lasso}).status_code == 409


def test_gui_settings_come_from_this_samples_exported_sets_only(client, tmp_path):
    client, runs = client
    assert client.get("/api/v1/samples/image-sample/cellpose-settings").json() == {"found": False}
    root = runs.stain.patches.root
    for sample, run in (("image-sample", "r1"), ("other", "r2")):
        folder = root / run / "set1"
        folder.mkdir(parents=True)
        (folder / "patches.json").write_text(json.dumps({"sample": sample}))
        np.save(folder / "patch-01_seg.npy", {"masks": np.zeros((4, 4), np.uint16), "manual_changes": [],
                                              "model_path": "cpsam", "flow_threshold": .6 if sample == "other" else .3,
                                              "cellprob_threshold": -1., "diameter": 0.,
                                              "normalize_params": {"percentile": [1., 99.]}})
    runs.stain.models = lambda method="cellpose", labels=True: [{"name": "cpsam", "kind": "builtin"}]
    found = client.get("/api/v1/samples/image-sample/cellpose-settings").json()
    assert found["found"] and found["run"] == "r1" and found["settings"]["flow_threshold"] == .3
    assert found["settings"]["model"] == "cpsam" and found["settings"]["diameter"] is None
    assert client.get("/api/v1/samples/nope/cellpose-settings").status_code == 404


def test_outlines_meet_by_area_or_crossing_edges_not_by_bounding_boxes():
    triangle = [[0, 0], [300, 0], [0, 300]]
    assert not X.meet(triangle, [[250, 250], [300, 250], [300, 300], [250, 300]])   # boxes overlap, shapes do not
    assert X.meet(triangle, [[10, 10], [20, 10], [20, 20], [10, 20]])                # inside
    assert X.meet([[10, 10], [20, 10], [20, 20], [10, 20]], triangle)                # around
    wide, tall = [[0, 4], [10, 4], [10, 6], [0, 6]], [[4, 0], [6, 0], [6, 10], [4, 10]]
    assert X.meet(wide, tall)                                                        # a cross: no vertex inside
    assert not X.meet(wide, [[20, 0], [30, 0], [30, 10]]) and not X.meet(wide, [[0, 0], [1, 1]])


def test_a_region_inside_the_tissues_bounding_box_but_outside_the_tissue_is_refused(config, tmp_path):
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    manager = Runs(config, Samples(config), TissueImages(tmp_path / "unwritten.zarr", [[[0, 0], [300, 0], [0, 300], [0, 0]]]))
    try:
        with pytest.raises(RunError, match="no tissue"):
            manager.create("image-sample", kind="stain", region={"kind": "box", "box": [250, 250, 300, 300]},
                           stain=SETTINGS)
        lasso = manager.create("image-sample", kind="stain", stain=SETTINGS,
                               region={"kind": "polygon", "points": [[100, 100], [140, 100], [100, 140]]})
        assert json.loads((manager.get(lasso["id"]).dir / "stain/job.json").read_text())["tissue"]
        assert not manager.runs.keys() - {lasso["id"]}
    finally:
        manager.close(save=False)


def test_concurrent_changes_never_publish_an_older_version(client, monkeypatch):
    import threading
    client, runs = client
    rid, run = finished_run(client, runs)
    exclusions = runs.stain.exclusions
    lasso = [[35, 5], [60, 5], [60, 30], [35, 30]]
    first_counting, release = threading.Event(), threading.Event()
    real = exclusions.counts

    def slow(run_, state=None):            # the first change stalls while counting, inside its refresh
        if not release.is_set():
            first_counting.set()
            release.wait(5)
        return real(run_, state)
    monkeypatch.setattr(exclusions, "counts", slow)
    added = {}
    worker = threading.Thread(target=lambda: added.update(exclusions.change(run, {"version": 0, "points": lasso})))
    worker.start()
    assert first_counting.wait(5)
    second = threading.Thread(target=lambda: exclusions.change(run, {"version": 1}, 1))
    second.start()
    second.join(.3)
    assert second.is_alive()                # waits for the first change to publish, instead of racing it
    release.set()
    worker.join(5)
    second.join(5)
    on_disk = exclusions.load(run)
    assert on_disk["version"] == 2 and not on_disk["areas"]
    assert run.meta["exclusions"]["version"] == run.meta["exclusions"]["areas_version"] == 2
    assert run.meta["exclusions"]["removed"] == 0 and run.meta["exclusions"]["kept"] == 3
    assert run.meta["summary"]["cells"] == 3


def test_a_save_records_the_removed_areas_it_used(client):
    client, runs = client
    rid, run = finished_run(client, runs)
    runs.stain.exclusions.change(run, {"version": 0, "points": [[35, 5], [60, 5], [60, 30], [35, 30]]})
    started = __import__("threading").Event()
    runs.zarr.existing_elements = lambda sid: []
    runs.zarr.save = lambda *a, **kw: started.set() or {"cells": 2}
    runs.save_to_zarr(rid, "kept", None)
    assert started.wait(5)
    assert run.meta["zarr_save"]["exclusions_digest"] == run.meta["exclusions"]["digest"]
    runs.stain.exclusions.change(run, {"version": 1}, 1)
    assert run.meta["exclusions"]["digest"] != run.meta["zarr_save"]["exclusions_digest"]


def sized_run(client, runs):
    """A finished run whose three cells enclose 16, 100 and 200 µm²."""
    rid = client.post("/api/v1/runs", json=dict(sample="image-sample", kind="stain", region={"kind": "full"},
                                                stain=SETTINGS)).json()["id"]
    run = runs.get(rid)
    final = run.dir / "stain/resolved-000001"
    final.mkdir()
    rings = [[[10, 10], [14, 10], [14, 14], [10, 14], [10, 10]],
             [[40, 10], [50, 10], [50, 20], [40, 20], [40, 10]],
             [[70, 10], [90, 10], [90, 20], [70, 20], [70, 10]]]
    for name, value in arrays(rings).items():
        np.save(final / f"{name}.npy", value)
    (run.dir / "output/boundaries.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": i + 1, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [r]}}
        for i, r in enumerate(rings)]}))
    runs._set(run, state="done", progress=dict(done=1, total=1, revision=2, resolved=True, complete=True),
              summary=dict(cells=3, area_um2=dict(p10=16, median=100, p90=200)))
    return rid, run


def test_a_size_range_combines_with_areas_and_vectors_keep_values_for_live_hiding(client):
    client, runs = client
    rid, run = sized_run(client, runs)
    url = f"/api/v1/runs/{rid}/stain/exclusions"
    assert client.get(url).json()["size"] is None
    ruled = client.post(url + "/size", json={"version": 0, "min": 150, "max": None}).json()
    assert ruled["version"] == 1 and ruled["size"] == {"min": 150, "max": None}
    assert ruled["by_size"] == ruled["removed"] == 2 and ruled["areas_version"] == 0
    detail = client.get(f"/api/v1/runs/{rid}").json()
    assert detail["exclusions"] == {k: v for k, v in ruled.items() if k != "areas"}
    assert detail["summary"]["cells"] == 1 and detail["summary_model"]["cells"] == 3
    assert cells(client, rid) == (3, 3)           # ranges are applied page-side, so no vectors request is needed
    assert saved_mask(runs, run) == [False, False, True]
    both = client.post(url, json={"version": 1, "points": [[60, 0], [100, 0], [100, 30], [60, 30]]}).json()
    assert both["removed"] == 3 and both["by_size"] == 2 and both["areas"][0]["cells"] == 1
    lower = client.post(url + "/size", json={"version": 2, "min": 50, "max": 150}).json()
    assert lower["by_size"] == 2 and lower["removed"] == 2 and lower["kept"] == 1
    cleared = client.post(url + "/size", json={"version": 3, "min": None, "max": None}).json()
    assert cleared["size"] is None and cleared["by_size"] == 0 and cleared["removed"] == 1
    for bad in (-1, True, "10", 2e6, float("inf")):
        response = client.post(url + "/size", content=json.dumps({"version": 4, "min": bad, "max": None}).replace(
            "Infinity", "1e999"), headers={"Content-Type": "application/json"})
        assert response.status_code == 422, (bad, response.text)
    assert client.post(url + "/size", json={"version": 1, "min": 10, "max": None}).status_code == 409
    assert client.post(url + "/size", json={"version": 4, "min_area_um2": 10}).status_code == 422


def test_a_save_leaves_out_cells_below_the_size_rule_and_records_it(client):
    client, runs = client
    rid, run = sized_run(client, runs)
    runs.stain.exclusions.change(run, {"version": 0, "min": 150, "max": None}, size=True)
    saved = {}
    runs.zarr.existing_elements = lambda sid: []
    runs.zarr.save = lambda *a, **kw: saved.update(args=a, kw=kw) or {"cells": 1}
    runs.save_to_zarr(rid, "kept", None)
    for _ in range(100):
        if saved:
            break
        __import__("time").sleep(.05)
    assert np.load(saved["kw"]["keep"], allow_pickle=False).tolist() == [False, False, True]
    assert saved["args"][6]["filters"]["size"] == {"min": 150, "max": None}


# ---- the brightness rule: cells outside a brightness range in one channel (a nucleus stain, say) are removed ---------
def bright_run(client, runs, tmp_path):
    """sized_run's three cells (16, 100 and 200 µm², 1 µm pixels) over a real two-channel zarr image: in DAPI the first
    cell is 500 throughout, the second 20 with a bright nucleus of 15 pixels at 900, the third 60 (no nucleus);
    Boundary is 7 everywhere. Their brightness (the mean of their brightest 5 µm²): 500, 900 and 60."""
    import zarr
    rid, run = sized_run(client, runs)
    job = json.loads((run.dir / "stain/job.json").read_text())
    level = job["image"]["levels"][0]
    planes = np.zeros(level["shape"], np.uint16)
    planes[1] = 7
    planes[0, 10:14, 10:14] = 500
    planes[0, 10:20, 40:50] = 20
    planes[0, 12:15, 42:47] = 900
    planes[0, 10:20, 70:90] = 60
    array = zarr.open_array(store=str(Path(job["path"]) / level["path"]), mode="w", shape=planes.shape,
                            dtype=planes.dtype, chunks=(1, 64, 64))
    array[:] = planes
    runs.zarr._array = lambda path, level: zarr.open_array(store=str(Path(path) / level["path"]), mode="r")
    return rid, run


def test_a_brightness_rule_removes_cells_whose_brightest_part_is_dim(client, tmp_path):
    client, runs = client
    rid, run = bright_run(client, runs, tmp_path)
    url = f"/api/v1/runs/{rid}/stain/exclusions"
    spread = client.get(f"/api/v1/runs/{rid}/stain/brightness").json()
    assert spread == {"channel": "DAPI", "channels": ["DAPI", "Boundary"], "default": "DAPI", "cells": 3,
                      "measured": 3, **dict(zip(("p10", "median", "p90"), np.percentile([500, 900, 60], [10, 50, 90])))}
    assert client.get(f"/api/v1/runs/{rid}/stain/brightness", params={"channel": "Boundary"}).json()["median"] == 7
    assert client.get(f"/api/v1/runs/{rid}/stain/brightness", params={"channel": "Nope"}).status_code == 422
    stored = sorted(p.name for p in (run.dir / "stain/brightness").iterdir())
    assert len(stored) == 2 and stored[0].startswith("resolved-000001-c0-")       # one file per outline file and channel
    ruled = client.post(url + "/brightness", json={"version": 0, "channel": "DAPI", "min": 100, "max": 800}).json()
    assert ruled["brightness"] == {"DAPI": {"min": 100, "max": 800}} and ruled["by_brightness"] == 2
    detail = client.get(f"/api/v1/runs/{rid}").json()
    assert detail["exclusions"] == {k: v for k, v in ruled.items() if k != "areas"}
    assert detail["summary"]["cells"] == 1 and detail["summary_model"]["cells"] == 3
    assert cells(client, rid) == (3, 3)
    assert saved_mask(runs, run) == [True, False, False]
    lower = client.post(url + "/brightness", json={"version": 1, "channel": "DAPI", "min": 600, "max": None}).json()
    assert lower["by_brightness"] == 2
    both = client.post(url + "/size", json={"version": 2, "min": 150, "max": None}).json()
    assert both["by_size"] == 2 and both["by_brightness"] == 2 and both["removed"] == 3
    cleared = client.post(url + "/brightness", json={"version": 3, "channel": "DAPI", "min": None, "max": None}).json()
    assert cleared["brightness"] == {} and cleared["by_brightness"] == 0 and cleared["removed"] == 2
    for bad in (-1, True, "10", 2e9, float("inf")):
        response = client.post(url + "/brightness", content=json.dumps({"version": 4, "channel": "DAPI", "min": bad, "max": None})
                               .replace("Infinity", "1e999"), headers={"Content-Type": "application/json"})
        assert response.status_code == 422, (bad, response.text)
    assert client.post(url + "/brightness", json={"version": 4, "channel": "Nope", "min": 10, "max": None}).status_code == 422
    assert client.post(url + "/brightness", json={"version": 1, "channel": "DAPI", "min": 10, "max": None}).status_code == 409
    assert client.post(url + "/brightness", json={"version": 4, "channel": "DAPI", "value": 10}).status_code == 422
    runs._set(run, state="running")
    assert client.post(url + "/brightness", json={"version": 4, "channel": "DAPI", "min": 10, "max": None}).status_code == 409


def test_brightness_is_measured_once_per_outline_file_and_channel_and_again_for_new_outlines(client, tmp_path,
                                                                                         monkeypatch):
    client, runs = client
    rid, run = bright_run(client, runs, tmp_path)
    measured = []
    real = X.stain_intensity.brightness
    monkeypatch.setattr(X.stain_intensity, "brightness", lambda *a, **k: measured.append(1) or real(*a, **k))
    exclusions = runs.stain.exclusions
    for _ in range(3):
        exclusions.brightness(run, "DAPI")
    assert len(measured) == 1
    exclusions.bright.clear()                                       # a restarted engine reads what it stored
    exclusions.brightness(run, "DAPI")
    assert len(measured) == 1
    offsets = run.dir / "stain/resolved-000001/offsets.npy"
    __import__("os").utime(offsets, ns=(offsets.stat().st_mtime_ns + 10**9,) * 2)   # the outlines written again
    exclusions.brightness(run, "DAPI")
    assert len(measured) == 2 and len(list((run.dir / "stain/brightness").glob("resolved-000001-c0-*.npy"))) == 1


def test_a_save_leaves_out_dim_cells_by_the_run_image_and_records_the_rule(client, tmp_path):
    client, runs = client
    rid, run = bright_run(client, runs, tmp_path)
    runs.stain.exclusions.change(run, {"version": 0, "channel": "DAPI", "min": 100, "max": None}, brightness=True)
    saved = {}
    runs.zarr.existing_elements = lambda sid: []
    runs.zarr.save = lambda *a, **kw: saved.update(args=a, kw=kw) or {"cells": 2}
    runs.save_to_zarr(rid, "kept", None)
    for _ in range(100):
        if saved:
            break
        __import__("time").sleep(.05)
    rule = {"DAPI": {"min": 100, "max": None}}
    assert np.load(saved["kw"]["keep"], allow_pickle=False).tolist() == [True, True, False]
    assert saved["args"][6]["filters"]["brightness"] == rule


def test_measuring_a_deleted_run_never_brings_its_folder_back(client, tmp_path, monkeypatch):
    """A run deleted while its cells are measured stays deleted: storing the values never creates folders."""
    client, runs = client
    rid, run = bright_run(client, runs, tmp_path)
    exclusions, real = runs.stain.exclusions, runs.stain.exclusions._measure_file
    def delete_meanwhile(*args):
        values = real(*args)
        __import__("shutil").rmtree(run.dir)                          # the delete lands while the image is read
        return values
    monkeypatch.setattr(exclusions, "_measure_file", delete_meanwhile)
    file = exclusions.files(run)[0]
    assert len(exclusions.file_brightness(run, file, "DAPI")) == 3
    assert not run.dir.exists()
    from sopa_companion.engine import leftovers
    assert leftovers.DERIVED.fullmatch("resolved-000001-c0-top5um2-1761058123000000000.npy")
    assert leftovers.DERIVED.fullmatch("tile-000004.npz-c1-top5um2-1761058123000000000.npy.tmp")
    assert not leftovers.DERIVED.fullmatch("tile-000004.npz") and not leftovers.DERIVED.fullmatch("job.json")

