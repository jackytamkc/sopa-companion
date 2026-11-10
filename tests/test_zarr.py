"""SpatialData samples (open / browse / prior / regions), saving results into the zarr with sopa, and
whole-sample (tiled) runs. The SpatialData parts need the sopa environment ([zarr] python); they are skipped
without it. Tile and transcript-index readers are tested on their own."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import HERE, needs_sopa, sopa_python, wait_for

BASE = "http://127.0.0.1:8799"


def test_proseg_default_names_ignore_case(config, monkeypatch):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    monkeypatch.setattr(samples, "existing_elements", lambda sid: ["Proseg_Boundaries", "PROSEG_TABLE_2"])
    try:
        result = samples.save_defaults("sample")
        assert result["shapes_key"] == "proseg_boundaries_3" and result["table_key"] == "proseg_table_3"
    finally:
        samples.close()


def test_sample_detail_build_is_single_flight(config, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    monkeypatch.setattr(samples, "path", lambda sid: "unread")
    started, release = threading.Event(), threading.Event()
    calls = []
    def build(sid, path, refresh):
        calls.append(sid)
        started.set()
        assert release.wait(5)
        with samples.lock:
            samples.details[sid] = dict(id=sid)
        return samples.details[sid]
    monkeypatch.setattr(samples, "_build", build)
    try:
        with ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(samples.detail, "s") for _ in range(8)]
            assert started.wait(5)
            release.set()
            results = [f.result(5) for f in futures]
        assert calls == ["s"] and all(result is results[0] for result in results)
    finally:
        samples.close()


def test_sample_detail_exposes_scales_and_median_area_and_refreshes_old_inspection_once(config, tmp_path, monkeypatch):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    from test_stain_levels import pyramid
    samples = ZarrSamples(config)
    image = pyramid()
    info = dict(name="pyramid", mtime=0, units=dict(frame="um"), images=[image], points=[], points_key=None,
                shapes=[dict(key="cells", count=3, geometry="Polygon", to_frame=np.eye(3).tolist(), median_area_um2=100)])
    folder = samples.cache / "s"
    folder.mkdir(parents=True)
    (folder / "inspect.json").write_text(json.dumps(info))
    calls = []
    monkeypatch.setattr(samples, "helper", lambda *a, **kw: calls.append(a) or {**info, "areas": 1})
    monkeypatch.setattr(samples, "_mtime", lambda path: 0)
    monkeypatch.setattr(samples, "_overview", lambda *a: None)
    monkeypatch.setattr(samples, "_stats", lambda *a: [])
    try:
        detail = samples._build("s", str(tmp_path), refresh=True)
        assert len(calls) == 1
        assert [s["label"] for s in detail["images"][0]["scales"]] == ["scale0", "scale1", "scale2", "scale3"]
        assert detail["shapes"][0]["median_area_um2"] == 100
        assert detail["tools"] == ["stain"]
        samples._build("s", str(tmp_path), refresh=True)
        assert len(calls) == 1
    finally:
        samples.close()


def test_save_passes_the_committed_keep_file_to_helper(config, tmp_path, monkeypatch):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    monkeypatch.setattr(samples, "detail", lambda sid: dict(points_key=None))
    monkeypatch.setattr(samples, "path", lambda sid: str(tmp_path / "sample"))
    calls = []
    monkeypatch.setattr(samples, "helper", lambda command, args, **kw: calls.append((command, args)) or dict(cells=2))
    keep = tmp_path / "keep.npy"
    np.save(keep, np.array([True, False, True]), allow_pickle=False)
    try:
        assert samples.save("s", tmp_path, "cells", None, False, False, dict(kind="stain"), keep=keep) == dict(cells=2)
        assert calls[0][0] == "save" and calls[0][1]["keep"] == str(keep)
        assert "pair_with" not in calls[0][1]
    finally:
        samples.close()


@pytest.mark.parametrize("resolution,name", [
    (dict(level=2, factor=4.004), "Cellpose · scale2"),
    (dict(level=1, factor=2, label="1/2"), "Cellpose · scale1"),          # saved by an earlier version: by its level
    (dict(factor=4), "Cellpose · scale2"),                                 # only a factor: its power of two
    (dict(level=0, factor=1, label="Full"), "Cellpose"), ({}, "Cellpose")])   # scale0, the image as stored: not named
def test_saved_stain_layer_label_includes_resolution(resolution, name):
    from sopa_companion.engine.zarr_samples import describe_segmentation
    made_by = dict(shapes=[dict(key="cells", app=dict(kind="stain", method="cellpose", model="cpsam",
                                                     resolution=resolution))])
    assert describe_segmentation("cells", made_by)["name"] == name


@needs_sopa
def test_pyramid_sample_fixture_offers_all_four_real_levels(config, tmp_path):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    path = tmp_path / "pyramid.zarr"
    subprocess.run([sopa_python(), str(HERE / "make_spatialdata.py"), "pyramid", str(path)],
                   check=True, capture_output=True, timeout=90)
    config["zarr"]["python"] = sopa_python()
    samples = ZarrSamples(config)
    try:
        detail = samples.open(str(path))
        scales = detail["images"][0]["scales"]
        assert [s["label"] for s in scales] == ["scale0", "scale1", "scale2", "scale3"]
        assert scales[0]["um_per_px"] == .5 and scales[3]["factor"] == pytest.approx(8, rel=.02)
    finally:
        samples.close()


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    sopa = sopa_python()
    root = tmp_path_factory.mktemp("spatialdata")
    for kind, path in (("sample", root / "synth.zarr"), ("output", root / "template" / "proseg-output.zarr")):
        subprocess.run([sopa, str(HERE / "make_spatialdata.py"), kind, str(path)], check=True, capture_output=True)
    return root


@pytest.fixture
def sample_path(synthetic, tmp_path):
    """A private copy: saving writes into it."""
    path = tmp_path / "samples" / "synth.zarr"
    shutil.copytree(synthetic / "synth.zarr", path)
    return path


def make_app(config, monkeypatch, synthetic, **proseg):
    from sopa_companion.engine.config import merge
    from sopa_companion.server import create_app
    monkeypatch.setenv("FAKE_PROSEG_OUTPUT", str(synthetic / "template" / "proseg-output.zarr"))
    overrides = merge(config, {"zarr": {"python": sopa_python()}, "proseg": proseg})
    return create_app(overrides, token="t" * 40)


@pytest.fixture
def client(config, monkeypatch, synthetic):
    with TestClient(make_app(config, monkeypatch, synthetic), base_url=BASE) as client:
        yield client


def read_spatialdata(path: Path, code: str) -> str:
    script = f"import warnings; warnings.filterwarnings('ignore')\nimport spatialdata as sd, sopa\ns = sd.read_zarr({str(path)!r})\n{code}"
    return subprocess.run([sopa_python(), "-c", script], check=True, capture_output=True, text=True).stdout.strip()


def dead_pid() -> int:
    """The pid of a process that has ended (and been reaped): a save killed long ago."""
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


# ---- samples ---------------------------------------------------------------------------------------------------
@needs_sopa
def test_open_browse_forget(client, sample_path, tmp_path):
    browse = client.get("/api/v1/browse", params={"path": str(sample_path.parent)}, headers=auth()).json()
    assert [(e["name"], e["kind"]) for e in browse["entries"]] == [("synth.zarr", "zarr")]
    assert client.get("/api/v1/browse", params={"path": str(tmp_path / "missing")}, headers=auth()).status_code == 404
    (tmp_path / "plain").mkdir()
    bad = client.post("/api/v1/samples/open", json={"path": str(tmp_path / "plain")}, headers=auth())
    assert bad.status_code == 422 and "SpatialData" in bad.json()["detail"]

    detail = client.post("/api/v1/samples/open", json={"path": str(sample_path)}, headers=auth()).json()
    assert detail["kind"] == "zarr" and detail["transcripts"] == 3000 and detail["points_key"] == "transcripts"
    assert detail["extent"][2] == pytest.approx(300, abs=1) and detail["extent"][3] == pytest.approx(200, abs=1)
    assert detail["overview"]["um_per_px"] == pytest.approx(0.5) and detail["overview"]["width"] == 600
    assert [c["name"] for c in detail["channels"]] == ["DAPI", "Boundary"]
    assert detail["suggested_prior"] == {"cells": {"kind": "shapes", "key": "cell_boundaries"},
                                         "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}}
    cells = detail["prior_options"]["cells"]
    assert cells[0]["key"] == "cell_boundaries" and cells[-1] == {**cells[-1], "kind": "column", "column": "cell_id",
                                                                   "unassigned": "UNASSIGNED"}
    assert detail["prior_options"]["nuclei"][0]["key"] == "nucleus_boundaries"
    sid = detail["id"]
    assert [s["id"] for s in client.get("/api/v1/samples", headers=auth()).json()] == [sid]
    png = client.get(f"/api/v1/samples/{sid}/overview.png", headers=auth())
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    shapes = client.get(f"/api/v1/samples/{sid}/shapes/nucleus_boundaries", params={"box": "0,0,50,50"}, headers=auth()).json()
    assert shapes["total"] == 1 and shapes["polygons"][0]["id"] == "nucleus-0"
    assert np.allclose(np.array(shapes["polygons"][0]["poly"]).min(0), [14, 14])
    assert client.get(f"/api/v1/samples/{sid}/shapes/nope", headers=auth()).status_code == 404
    image = client.get(f"/api/v1/samples/{sid}/image/morphology_focus/c1.u16", params={"box": "0,0,100,50", "px": 64},
                       headers=auth({"Accept-Encoding": "identity"}))
    meta = json.loads(image.headers["x-image"])
    assert meta["level"] == 1 and len(image.content) == meta["width"] * meta["height"] * 2
    assert client.delete(f"/api/v1/samples/{sid}", headers=auth()).status_code == 200
    assert client.get("/api/v1/samples", headers=auth()).json() == []
    assert client.get(f"/api/v1/samples/{sid}", headers=auth()).status_code == 404


def test_region_spec():
    from sopa_companion.engine.runs import RunError, region_spec
    extent = [0, 0, 300, 200]
    assert region_spec({"kind": "full"}, extent)["bbox"] == [0, 0, 300, 200]
    assert region_spec({"kind": "box", "box": [-10, 20, 120, 400]}, extent)["bbox"] == [0, 20, 120, 200]
    polygon = region_spec({"kind": "polygon", "points": [[10, 10], [110, 10], [60, 90]]}, extent)
    assert polygon["bbox"] == [10, 10, 110, 90] and len(polygon["points"]) == 3
    for bad in ({"kind": "box", "box": [0, 0, 5, 5]}, {"kind": "polygon", "points": [[0, 0], [1, 1]]},
                {"kind": "polygon", "points": [[0, 0], [100, 0], [100, 1]]}, {"kind": "circle"}, None):
        with pytest.raises(RunError):
            region_spec(bad, extent)


# ---- a full-detail lasso run, saved into the sample --------------------------------------------------------------
@needs_sopa
def test_lasso_run_with_shape_prior_saved_into_zarr(client, sample_path):
    import pyarrow.parquet as pq
    sid = client.post("/api/v1/samples/open", json={"path": str(sample_path)}, headers=auth()).json()["id"]
    detail = client.get(f"/api/v1/samples/{sid}", headers=auth()).json()
    missing = client.post("/api/v1/runs", json={"sample": sid, "region": {"kind": "full"}, "prior": {}}, headers=auth())
    assert missing.status_code == 422 and "prior" in missing.json()["detail"]
    lasso = [[5, 5], [150, 5], [150, 120], [5, 60]]
    run = client.post("/api/v1/runs", json={"sample": sid, "region": {"kind": "polygon", "points": lasso},
                                            "prior": detail["suggested_prior"]}, headers=auth()).json()
    assert run["mode"] == "detail" and run["sample_kind"] == "zarr" and run["box"] == [5, 5, 150, 120]
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())   # paused on its prior
    app = client.app
    runs = app.state.runs
    wait_for(lambda: len(runs.get(run["id"]).frames) >= 3, timeout=60)
    meta = runs.get(run["id"]).meta
    assert meta["prepared"]["prior_cells"] > 0 and meta["image"]["channels"][0]["name"] == "DAPI"
    table = pq.read_table(runs.get(run["id"]).dir / "input" / "transcripts.parquet")
    x, y = table["x_location"].to_numpy(), table["y_location"].to_numpy()
    cell = np.asarray(table["cell_id"].to_pylist(), dtype=object)
    nucleus = table["overlaps_nucleus"].to_numpy()
    assert table.num_rows == meta["transcripts"] and (y <= 5 + (x - 5) * 55 / 145 + 60.01).all()   # under the lasso's top edge
    assigned = cell != "UNASSIGNED"
    assert all(c.startswith("cell-") for c in cell[assigned])
    k = np.array([int(c[5:]) for c in cell[assigned]])
    assert (np.abs(x[assigned] - (20 + 40 * (k // 5))) <= 18.01).all() and (np.abs(y[assigned] - (20 + 40 * (k % 5))) <= 18.01).all()
    near = (np.abs(x - 20 - 40 * (x // 40)) <= 6.01) & (np.abs(y - 20 - 40 * (y // 40)) <= 6.01)
    assert nucleus.sum() and (near[nucleus == 1]).all() and not nucleus[~near].any()
    prior_shapes = json.loads(client.app.state.runs.xenium_boundaries(run["id"]).read_text())    # the reference form
    assert prior_shapes["source"] == {"cells": "cell_boundaries", "nuclei": "nucleus_boundaries"} and prior_shapes["cells"]
    from sopa_companion.engine import vectors       # the binary the viewer loads carries exactly what the JSON carries
    packed = client.get(f"/api/v1/runs/{run['id']}/xenium", params={"format": "bin", "v": 1}, headers=auth({"Accept-Encoding": "zstd"}))
    assert packed.headers["content-encoding"] == "zstd" and vectors.to_json(vectors.decode(packed.content)) == prior_shapes
    assert client.get(f"/api/v1/runs/{run['id']}/xenium/masks", headers=auth()).status_code == 404
    early = client.post(f"/api/v1/runs/{run['id']}/zarr", json={"shapes_key": "a", "table_key": "b"}, headers=auth())
    assert early.status_code == 409

    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "finish"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "done", timeout=60)
    done = client.get(f"/api/v1/runs/{run['id']}", headers=auth()).json()
    defaults = done["save_defaults"]
    assert defaults["shapes_key"] == "proseg_boundaries" and "cell_boundaries" in defaults["existing"]
    clash = client.post(f"/api/v1/runs/{run['id']}/zarr", json={"shapes_key": "cell_boundaries", "table_key": "t"}, headers=auth())
    assert clash.status_code == 409 and "already exist" in clash.json()["detail"]
    for bad in ({"shapes_key": "a b", "table_key": "t"}, {"shapes_key": "same", "table_key": "same"}):
        assert client.post(f"/api/v1/runs/{run['id']}/zarr", json=bad, headers=auth()).status_code == 422
    saving = client.post(f"/api/v1/runs/{run['id']}/zarr", json={**{k: defaults[k] for k in ("shapes_key", "table_key")},
                                                                 "set_default": True}, headers=auth()).json()
    assert saving["zarr_save"]["state"] == "saving"
    wait_for(lambda: runs.get(run["id"]).meta["zarr_save"]["state"] != "saving", timeout=120)
    result = runs.get(run["id"]).meta["zarr_save"]
    assert result["state"] == "saved", result
    assert result["cells"] == 6 and runs.get(run["id"]).meta["save_defaults"]["shapes_key"] == "proseg_boundaries_2"
    check = read_spatialdata(sample_path, "t = s.tables['proseg_table']\n"
                                          "print(len(s.shapes['proseg_boundaries']), t.uns['spatialdata_attrs']['region'], "
                                          "sopa.utils.get_boundaries(s, return_key=True)[0], t.uns['sopa_companion']['run'], "
                                          "'proseg_live' in t.uns)")
    # The record under this version's name only (the previous version's readers do not know it: names.py).
    assert check.split() == ["6", "proseg_boundaries", "proseg_boundaries", run["id"], "False"]
    from sopa_companion.engine.zarr_samples import table_run, unfinished_writes     # the engine's own reading
    assert table_run(str(sample_path), "proseg_table") == run["id"] and unfinished_writes(str(sample_path)) == set()
    again = client.post(f"/api/v1/runs/{run['id']}/zarr", json={"shapes_key": "proseg_boundaries", "table_key": "proseg_table"},
                        headers=auth())
    assert again.status_code == 409
    client.post(f"/api/v1/runs/{run['id']}/zarr", json={"shapes_key": "proseg_boundaries", "table_key": "proseg_table",
                                                        "overwrite": True, "set_default": False}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["zarr_save"]["state"] != "saving", timeout=120)
    assert runs.get(run["id"]).meta["zarr_save"]["state"] == "saved"


@needs_sopa
def test_a_prior_without_nuclei_seeds_proseg_with_every_transcript_in_a_cell(synthetic, tmp_path):
    """proseg seeds its starting cells from nuclear transcripts only. A prior with cell shapes and no nuclei (the
    Cellpose shapes, 2026-10-19) reached it with none flagged: 0 cells, and proseg panicked (clustering: remainder with a
    divisor of zero). Without nuclei every transcript inside a prior cell now seeds that cell (proseg's own rule without
    compartment information); with nuclei the flag stays the nuclei's."""
    import pyarrow.parquet as pq

    def prepare(prior, name):
        out = tmp_path / name
        args = {"path": str(synthetic / "synth.zarr"), "points_key": "transcripts", "region": {"kind": "full"},
                "prior": prior, "out": str(out)}
        done = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", "prepare", json.dumps(args)],
                              capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONPATH": str(HERE.parent.parent)})
        stats = json.loads(done.stdout.strip().splitlines()[-1])
        table = pq.read_table(out)
        assigned = np.asarray(table["cell_id"].to_pylist(), dtype=object) != "UNASSIGNED"
        return stats, assigned, table["overlaps_nucleus"].to_numpy().astype(bool)

    cells = {"kind": "shapes", "key": "cell_boundaries"}
    stats, assigned, seeds = prepare({"cells": cells, "nuclei": None}, "cells-only.parquet")
    assert assigned.any() and (seeds == assigned).all()
    assert stats["seeded_from"] == "cells" and stats["nuclear"] == stats["assigned"] == int(assigned.sum())
    stats, assigned, seeds = prepare({"cells": cells, "nuclei": {"kind": "shapes", "key": "nucleus_boundaries"}},
                                     "with-nuclei.parquet")
    assert stats["seeded_from"] == "nuclei" and 0 < seeds.sum() < assigned.sum() and not (seeds & ~assigned).any()
    # a pairing whose nucleus flag marks nothing inside the cells (an all-0 column, as some samples' overlaps_nucleus)
    # would give proseg no seed at all: the cells seed it instead
    stats, assigned, seeds = prepare({"cells": cells, "nuclei": {"kind": "column", "column": "overlaps_nucleus", "nuclear": "7"}},
                                     "empty-flag.parquet")
    assert stats["seeded_from"] == "cells" and (seeds == assigned).all() and assigned.any()


# ---- whole-sample (tiled) runs -------------------------------------------------------------------------------------
@needs_sopa
def test_tiled_run_streams_what_is_on_screen(config, monkeypatch, synthetic, sample_path):
    with TestClient(make_app(config, monkeypatch, synthetic, max_region_um=100), base_url=BASE) as client:
        runs = client.app.state.runs
        sid = client.post("/api/v1/samples/open", json={"path": str(sample_path)}, headers=auth()).json()["id"]
        prior = {"cells": {"kind": "column", "column": "cell_id", "unassigned": "UNASSIGNED"}, "nuclei": None}
        run = client.post("/api/v1/runs", json={"sample": sid, "region": {"kind": "full"}, "prior": prior}, headers=auth()).json()
        assert run["mode"] == "tiled"
        rid = run["id"]
        client.post(f"/api/v1/runs/{rid}/control", json={"action": "resume"}, headers=auth())
        wait_for(lambda: len(runs.get(rid).frames) >= 3, timeout=60)
        meta = runs.get(rid).meta
        assert "--live-tiled" in (runs.get(rid).dir / "proseg.log").read_text()
        # a cells-only prior seeds proseg with every assigned transcript (it used to reach proseg with 0 nuclear: 0 cells)
        assert meta["image"]["source"] == "sample" and meta["prepared"]["seeded_from"] == "cells"
        assert meta["prepared"]["nuclear"] == meta["prepared"]["assigned"] > 0

        seq = len(runs.get(rid).frames) - 1
        frame = runs.get(rid).frames[seq]
        labels = np.array([(frame["iter"] + p) % 3 for p in range(35)], np.uint32).reshape(5, 7)
        view = client.get(f"/api/v1/runs/{rid}/frames/{seq}/view", params={"box": "0,0,7,5"}, headers=auth())
        geometry = json.loads(view.headers["x-view"])
        planes = np.frombuffer(view.content, "<u4").reshape(2, 5, 7)
        assert geometry["stride"] == 1 and (planes[0] == labels).all() and (planes[1] == frame["iter"]).all()
        part = client.get(f"/api/v1/runs/{rid}/frames/{seq}/view", params={"box": "1.5,1.2,6.9,4.9", "stride": 2,
                                                                         "planes": "consensus"}, headers=auth())
        g = json.loads(part.headers["x-view"])
        assert (g["x0"], g["y0"], g["voxel"], g["planes"]) == (0.0, 0.0, 2.0, ["consensus"])
        assert (np.frombuffer(part.content, "<u4").reshape(g["ny"], g["nx"]) == labels[0::2, 0::2]).all()
        assert client.get(f"/api/v1/runs/{rid}/frames/{seq}/view", params={"box": "50,50,60,60"}, headers=auth()).status_code == 404
        assert client.get(f"/api/v1/runs/{rid}/frames/{seq}/view", params={"box": "0,0,7,5", "planes": "layers"},
                          headers=auth()).status_code == 422

        assert client.post(f"/api/v1/runs/{rid}/view", json={"box": [2, 1, 4, 3]}, headers=auth()).status_code == 200
        wait_for(lambda: (runs.get(rid).frames[-1].get("window") or {}).get("i0") == 2, timeout=30)
        seq = len(runs.get(rid).frames) - 1
        window = client.get(f"/api/v1/runs/{rid}/frames/{seq}/window", headers=auth())
        w = json.loads(window.headers["x-window"])
        assert (w["x0"], w["y0"], w["nx"], w["ny"], w["nz"], w["source"]) == (2.0, 1.0, 3, 3, 2, "frame")
        assert len(window.content) == 3 * 3 * 2 * 4

        client.post(f"/api/v1/runs/{rid}/control", json={"action": "pause"}, headers=auth())
        wait_for(lambda: (runs.get(rid).ack or {}).get("state") == "paused", timeout=30)
        client.post(f"/api/v1/runs/{rid}/view", json={"box": [4, 2, 6, 4]}, headers=auth())
        wait_for(lambda: (runs.get(rid).window or {}).get("i0") == 4, timeout=30)
        seq = runs.get(rid).window["seq"]
        paused = json.loads(client.get(f"/api/v1/runs/{rid}/frames/{seq}/window", headers=auth()).headers["x-window"])
        assert paused["source"] == "paused" and paused["x0"] == 4.0

        prior_view = client.get(f"/api/v1/runs/{rid}/prior/view", params={"box": "0,0,7,5"}, headers=auth())
        pv = np.frombuffer(prior_view.content, "<u4").reshape(2, 5, 7)
        assert (pv[0] == np.arange(35).reshape(5, 7) % 3).all() and set(np.unique(pv[1])) == {0, 800}
        found = client.get(f"/api/v1/runs/{rid}/transcripts/view", params={"box": "0,0,2,2"}, headers=auth())
        assert found.headers["x-count"] == "1" and struct.unpack("<fffI", found.content) == (1.0, 1.0, pytest.approx(0.2), 0)
        assert client.get(f"/api/v1/runs/{rid}/transcripts/view", params={"box": "0,0,5,5", "limit": 1},
                          headers=auth()).status_code == 413
        image = client.get(f"/api/v1/runs/{rid}/image/view/c0.u16", params={"box": "0,0,300,200", "px": 300}, headers=auth())
        assert json.loads(image.headers["x-image"])["level"] == 1

        client.post(f"/api/v1/runs/{rid}/control", json={"action": "finish"}, headers=auth())
        wait_for(lambda: runs.get(rid).meta["state"] == "done", timeout=60)
        assert not any(o["name"] == "transcript-metadata.csv.gz" for o in runs.get(rid).meta["outputs"])


# ---- readers -------------------------------------------------------------------------------------------------------
def write_container(path, nx, ny, tile, planes, window=None):
    import zstandard
    zstd = zstandard.ZstdCompressor(level=1)
    tiles_x, tiles_y = -(-nx // tile), -(-ny // tile)
    blobs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            for plane in planes:
                block = np.ascontiguousarray(plane[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile], "<u4")
                blobs.append(zstd.compress(block.tobytes()) if block.any() else b"")
    blobs.append(zstd.compress(np.ascontiguousarray(window[1], "<u4").tobytes()) if window else b"")
    header = json.dumps({"nx": nx, "ny": ny, "nz": 2, "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
                         "planes": ["consensus", "last_change"][:len(planes)], "window": window[0] if window else None}).encode()
    offset = 8 + len(header) + 12 * len(blobs)
    head = b"PLT1" + struct.pack("<I", len(header)) + header
    for blob in blobs:
        head += struct.pack("<QI", offset, len(blob))
        offset += len(blob)
    Path(path).write_bytes(head + b"".join(blobs))


def test_tile_container_compose(tmp_path):
    from sopa_companion.engine.tiles import Container
    rng = np.random.default_rng(1)
    consensus = rng.integers(0, 50, (37, 53)).astype(np.uint32)
    consensus[:16, :16] = 0                                     # an all-zero tile
    age = rng.integers(0, 9, (37, 53)).astype(np.uint32)
    layers = rng.integers(0, 50, (2, 4, 6)).astype(np.uint32)
    write_container(tmp_path / "f.tiles", 53, 37, 16, [consensus, age], ({"i0": 3, "j0": 2, "nx": 6, "ny": 4, "nz": 2}, layers))
    c = Container(tmp_path / "f.tiles")
    assert (c.full("consensus") == consensus).all() and (c.full("last_change") == age).all()
    for c0, r0, c1, r1, s in [(0, 0, 53, 37, 1), (5, 3, 40, 30, 3), (15, 15, 17, 17, 1), (-4, -3, 60, 50, 4), (52, 36, 53, 37, 2)]:
        expected = np.zeros(((r1 - r0 + s - 1) // s, (c1 - c0 + s - 1) // s), np.uint32)
        for oy in range(expected.shape[0]):
            for ox in range(expected.shape[1]):
                y, x = r0 + oy * s, c0 + ox * s
                if 0 <= y < 37 and 0 <= x < 53:
                    expected[oy, ox] = consensus[y, x]
        assert (c.compose("consensus", c0, r0, c1, r1, s) == expected).all(), (c0, r0, c1, r1, s)
    info, got = c.window()
    assert info["nx"] == 6 and (got == layers).all()
    from sopa_companion.engine import tiles
    cached = sum(1 for key in tiles.CACHE.items if key[0] == str(tmp_path / "f.tiles"))
    assert cached == 2 * c.header["tiles_x"] * c.header["tiles_y"] - 1      # the all-zero tile is never cached
    small = tiles.TileCache(limit_bytes=4096)
    for k in range(100):
        small.put(k, np.zeros(0, np.uint32))
    assert len(small.items) <= 8                                               # empty entries still count
    (tmp_path / "bad.tiles").write_bytes(b"PLT0" + bytes(20))
    with pytest.raises(ValueError):
        Container(tmp_path / "bad.tiles")


def test_transcript_index(tmp_path):
    from sopa_companion.engine.transcript_index import TranscriptIndex
    rng = np.random.default_rng(2)
    n = 20000
    records = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
    records["x"], records["y"] = rng.uniform(-50, 950, n), rng.uniform(10, 520, n)
    records["gene"] = np.arange(n)
    with gzip.open(tmp_path / "transcripts.bin.gz", "wb") as stream:
        stream.write(records.tobytes())
    index = TranscriptIndex(tmp_path)
    for box in [(0, 0, 100, 100), (123.4, 250.5, 777.7, 300.1), (-100, -100, 2000, 2000), (5000, 5000, 6000, 6000)]:
        found, count = index.query(box, 10 ** 6)
        inside = (records["x"] >= box[0]) & (records["x"] < box[2]) & (records["y"] >= box[1]) & (records["y"] < box[3])
        assert count == inside.sum() and sorted(found["gene"].tolist()) == sorted(records["gene"][inside].tolist())
    found, count = index.query((-100, -100, 2000, 2000), 100)
    assert found is None and count > 100
    assert TranscriptIndex(tmp_path).grid["count"] == n        # reuses the index on disk


def test_transcript_index_counts_only_the_box(tmp_path):
    """A dense grid cell around a tiny empty box is not "too many transcripts in view"."""
    from sopa_companion.engine.transcript_index import TranscriptIndex
    records = np.zeros(50, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
    records["x"] = records["y"] = 99.0
    with gzip.open(tmp_path / "transcripts.bin.gz", "wb") as stream:
        stream.write(records.tobytes())
    found, count = TranscriptIndex(tmp_path).query((99.1, 99.1, 99.2, 99.2), 10)
    assert count == 0 and found is not None and found.size == 0


# ---- save safety ------------------------------------------------------------------------------------------
def done_run(client, sample_path):
    sid = client.post("/api/v1/samples/open", json={"path": str(sample_path)}, headers=auth()).json()["id"]
    prior = client.get(f"/api/v1/samples/{sid}", headers=auth()).json()["suggested_prior"]
    run = client.post("/api/v1/runs", json={"sample": sid, "region": {"kind": "box", "box": [0, 0, 120, 120]},
                                            "prior": prior}, headers=auth()).json()
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
    runs = client.app.state.runs
    wait_for(lambda: len(runs.get(run["id"]).frames) >= 2, timeout=60)
    client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "finish"}, headers=auth())
    wait_for(lambda: runs.get(run["id"]).meta["state"] == "done", timeout=60)
    return run["id"]


def save(client, rid, **body):
    runs = client.app.state.runs
    response = client.post(f"/api/v1/runs/{rid}/zarr", json=body, headers=auth())
    if response.status_code != 200:
        return response
    wait_for(lambda: runs.get(rid).meta["zarr_save"]["state"] != "saving", timeout=120)
    return runs.get(rid).meta["zarr_save"]


@needs_sopa
def test_failed_save_rolls_back_and_interrupted_save_is_recovered(client, sample_path, monkeypatch):
    rid = done_run(client, sample_path)
    first = save(client, rid, shapes_key="proseg_boundaries", table_key="proseg_table")
    assert first["state"] == "saved"
    snapshot = "print(len(s.shapes['proseg_boundaries']), s.tables['proseg_table'].uns['sopa_companion']['saved'], " \
               "sorted(s.shapes), sorted(s.tables), s.attrs.get('cell_segmentation_boundaries', s.attrs.get('boundaries')))"
    before = read_spatialdata(sample_path, snapshot)

    monkeypatch.setenv("SOPA_COMPANION_TEST_FAIL_AFTER", "shapes")      # the table write never happens
    failed = save(client, rid, shapes_key="proseg_boundaries", table_key="proseg_table", overwrite=True)
    assert failed["state"] == "failed" and "injected failure" in failed["message"]
    assert read_spatialdata(sample_path, snapshot) == before            # the previous pair is back, untouched
    assert not (sample_path.parent / ".proseg-live-backups").exists()
    monkeypatch.delenv("SOPA_COMPANION_TEST_FAIL_AFTER")

    # A save killed half-way: the old shapes moved aside, a new element half-written, no commit in the journal (which
    # names its zarr and its process, long gone; the backup folder is named <zarr>-<date>-<time>-<pid>).
    dead = dead_pid()
    backup = sample_path.parent / ".proseg-live-backups" / f"{sample_path.name}-20260101-000000-{dead}"
    (backup / "shapes").mkdir(parents=True)
    shutil.move(str(sample_path / "shapes" / "proseg_boundaries"), str(backup / "shapes" / "proseg_boundaries"))
    shutil.copytree(sample_path / "shapes" / "cell_boundaries", sample_path / "shapes" / "half_written")
    (backup / "journal.json").write_text(json.dumps({"state": "writing", "path": str(sample_path.resolve()),
                                                     "writer": {"pid": dead}, "moved": [["shapes", "proseg_boundaries"]],
                                                     "written": [["shapes", "half_written"]]}))
    from sopa_companion.engine.zarr_samples import unfinished_writes      # the engine reads it as the helper names it
    assert unfinished_writes(sample_path) == {("shapes", "half_written")}
    recovered = save(client, rid, shapes_key="second_boundaries", table_key="second_table")
    assert recovered["state"] == "saved", recovered
    assert unfinished_writes(sample_path) == set()
    after = read_spatialdata(sample_path, "print(sorted(s.shapes), len(s.shapes['proseg_boundaries']))")
    assert "half_written" not in after and "'proseg_boundaries'" in after and "'second_boundaries'" in after
    assert not (sample_path.parent / ".proseg-live-backups").exists()


@needs_sopa
def test_save_refuses_names_that_would_break_the_sample(client, sample_path):
    rid = done_run(client, sample_path)
    assert save(client, rid, shapes_key="proseg_boundaries", table_key="proseg_table")["state"] == "saved"
    case = save(client, rid, shapes_key="Proseg_Boundaries", table_key="x_table")
    assert case.status_code == 422 and "only by case" in case.json()["detail"]
    assert save(client, rid, shapes_key="__hidden", table_key="t").status_code == 422
    orphan = save(client, rid, shapes_key="proseg_boundaries", table_key="other_table", overwrite=True)
    assert orphan["state"] == "failed" and "annotates" in orphan["message"]
    assert read_spatialdata(sample_path, "print(len(s.shapes['proseg_boundaries']), sorted(s.tables))") == "6 ['proseg_table']"


@needs_sopa
def test_prior_columns_with_booleans_nulls_and_no_statistics(tmp_path):
    script = f'''
import json, sys
sys.path.insert(0, {str(HERE.parents[1])!r})
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from sopa_companion.zarr_bridge import helper
assert helper._flags(pa.array([True, False, None])).tolist() == ["1", "0", "0"]
assert helper._flags(pa.array([1, 0, None], pa.uint8())).tolist() == ["1", "0", "0"]
assert helper._as_strings(pa.array(["a", None])).tolist() == ["a", None]
path = {str(tmp_path / "p.parquet")!r}
ids = np.arange(2000) // 2                    # 1000 cells, two transcripts each; cell 0 is an ordinary cell
x = np.linspace(1.0, 5.0, 2000); y = np.linspace(2.0, 9.0, 2000)
pq.write_table(pa.table({{"x": x, "y": y, "cell_id": pa.array(ids, pa.int32())}}), path, write_statistics=False)
stats = helper._column_stats([path], ["x", "y"])
assert stats["min"] == {{"x": 1.0, "y": 2.0}} and stats["max"] == {{"x": 5.0, "y": 9.0}} and stats["complete"]["x"]
cells = helper._prior_columns([path], {{"x": "float", "y": "float", "cell_id": "int32"}}, {{"instance_key": "cell_id"}}, {{}}, stats)["cells"]
assert cells[0]["unassigned"] == "", cells          # ids 0 and 1 are both cells: nothing is background
print("ok")
'''
    result = subprocess.run([sopa_python(), "-c", script], capture_output=True, text=True)
    assert result.stdout.strip().endswith("ok"), result.stderr[-2000:]


# ---- segmentation layers and Compare's counts ----------------------------------------------------------------------
# A sample whose elements say how they were made (tables with Xenium's or an import's methods, an app's proseg table
# and stain saves) and whose transcripts test the counting rule: cells A, B, C and D (D overlaps A's right edge).
SEGMENTED = r'''
import sys, warnings
warnings.filterwarnings("ignore")
import anndata, dask.dataframe as dd, geopandas as gpd, numpy as np, pandas as pd, shapely, spatialdata as sd, zarr
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

path = sys.argv[1]
rows = [*[(15 + i, 15, "G1", 40, True) for i in range(3)],          # A: 3 x G1
        (16, 25, "G1", 10, True),                                   # A: quality under 20
        (17, 25, "NegControlProbe_1", 40, False),                   # A: not a gene
        (18, 18, "G2", 30, True), (19, 19, "G2", 20, True),         # A: 2 x G2 (quality 20 counts)
        *[(55 + i, 20, "G3", 25, True) for i in range(4)],          # B: 4 x G3
        (27, 20, "G1", 40, True),                                   # in A and D: counted once
        *[(150 + i, 150, "G1", 40, True) for i in range(5)]]        # in no cell
frame = pd.DataFrame(rows, columns=["x", "y", "feature_name", "qv", "is_gene"]).astype({"x": np.float32, "y": np.float32,
                                                                                         "qv": np.float32})
frame["z"] = np.float32(1)
frame["feature_name"] = pd.Categorical(frame["feature_name"])
points = PointsModel.parse(dd.from_pandas(frame, npartitions=2), coordinates={"x": "x", "y": "y", "z": "z"},
                           feature_key="feature_name", transformations={"global": Identity()})


def layer(boxes, prefix):
    return ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(*b) for b in boxes],
                                              index=[f"{prefix}-{k}" for k in range(len(boxes))]),
                             transformations={"global": Identity()})


cells = [(10, 10, 30, 30), (50, 10, 70, 30), (90, 10, 110, 30), (25, 10, 45, 30)]
spots = gpd.GeoDataFrame({"radius": [5.0, 5.0]}, geometry=[shapely.Point(20, 20), shapely.Point(60, 20)])
shapes = {"cells": layer(cells, "cell"), "xenium_cell_boundaries": layer(cells[:3], "x"),
          "proseg_boundaries": layer(cells[:2], "p"), "proseg_boundaries_2": layer(cells[2:], "q"),
          "cellpose_boundaries_1": layer(cells[:2], "c"),
          "instanseg_boundaries_1": layer(cells[1:3], "i"), "nuclei": layer([(15, 15, 20, 20), (55, 15, 60, 20)], "n"),
          "region_of_interest": layer([(0, 0, 200, 200)], "roi"),
          "spots": ShapesModel.parse(spots, transformations={"global": Identity()})}
anndata.settings.allow_write_nullable_strings = True


def table(region, methods=None, uns=None):
    n = len(shapes[region])
    obs = pd.DataFrame({"cell_id": list(shapes[region].index), "region": pd.Categorical([region] * n)},
                       index=[str(k) for k in range(n)])
    if methods is not None:
        obs["segmentation_method"] = methods
    adata = anndata.AnnData(np.ones((n, 2), np.float32), obs=obs, var=pd.DataFrame(index=["G1", "G2"]), uns=uns or {})
    return TableModel.parse(adata, region=region, region_key="region", instance_key="cell_id")


tables = {"cells_table": table("cells", pd.array(["Imported Cell Segmentation"] * 4, dtype="string")),
          "copy_table": table("xenium_cell_boundaries"),
          "xenium_table": table("xenium_cell_boundaries", pd.Categorical(["Segmented by interior stain (18S)",
                                                                          "Segmented by nucleus expansion of 5.0µm", None])),
          "proseg_table": table("proseg_boundaries", uns={"sopa_companion": {"app": "SoPA companion", "run": "r-1"}}),
          # a proseg save of the previous version of the app: its record under its own name
          "proseg_table_2": table("proseg_boundaries_2", uns={"proseg_live": {"app": "Proseg Live", "run": "r-0"}})}
sd.SpatialData(points={"transcripts": points}, shapes=shapes, tables=tables).write(path)
# Stain saves: this version's record (sopa_companion), and the previous version's (proseg_live) on the InstanSeg one.
for key, name, method, model, done in (
        ("cellpose_boundaries_1", "sopa_companion", "cellpose", "cp4_x", {"done": 3, "total": 10, "complete": False}),
        ("instanseg_boundaries_1", "proseg_live", "instanseg", "fluorescence_nuclei_and_cells",
         {"done": 10, "total": 10, "complete": True})):
    zarr.open_group(f"{path}/shapes/{key}", mode="r+").attrs[name] = {"kind": "stain", "method": method, "model": model,
                                                                      "progress": done}
'''
# The nuclei written again (as a notebook would): three outlines where there were two.
REWRITE_NUCLEI = r'''
import shutil, sys, warnings
warnings.filterwarnings("ignore")
import geopandas as gpd, shapely, spatialdata as sd
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity
s = sd.read_zarr(sys.argv[1])
shutil.rmtree(sys.argv[1] + "/shapes/nuclei")
s["nuclei"] = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(1, 1, 3, 3), shapely.box(5, 5, 6, 6),
                                                           shapely.box(40, 40, 42, 42)], index=["a", "b", "c"]),
                                transformations={"global": Identity()})
s.write_element("nuclei")
s.write_consolidated_metadata()
'''


@pytest.fixture(scope="session")
def segmented(tmp_path_factory):
    sopa = sopa_python()
    path = tmp_path_factory.mktemp("segmented") / "segmented.zarr"
    made = subprocess.run([sopa, "-c", SEGMENTED, str(path)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr[-3000:]
    return path


@pytest.fixture
def segmented_copy(segmented, tmp_path):
    """A private copy: the tests write elements into it again."""
    path = tmp_path / "samples" / "segmented.zarr"
    shutil.copytree(segmented, path)
    return path


def test_segmentations_are_named_by_what_made_them():
    from sopa_companion.engine.zarr_samples import describe_segmentation
    made_by = {"shapes": [{"key": "cp", "app": {"kind": "stain", "method": "cellpose", "model": "cp4", "complete": False,
                                                "done": 3, "total": 10}},
                          {"key": "is", "app": {"kind": "stain", "method": "instanseg", "model": None, "complete": True,
                                                "done": 4, "total": 4}}],
               "tables": [{"key": "plain", "region": ["xen"], "proseg": False, "methods": None},
                          {"key": "xt", "region": ["xen", "other"], "proseg": False,
                           "methods": {"xen": {"Segmented by interior stain (18S)": 9, "Segmented by boundary stain (x)": 2},
                                       "other": {"Imported Cell Segmentation": 5}}},
                          {"key": "it", "region": ["imp"], "proseg": False, "methods": {"imp": {"Imported Cell Segmentation": 4}}},
                          {"key": "pt", "region": ["pro"], "proseg": True, "methods": None},
                          {"key": "odd", "region": ["mixed"], "proseg": False,
                           "methods": {"mixed": {"Segmented by interior stain (18S)": 1, "by hand": 1}}}]}
    named = {key: describe_segmentation(key, made_by) for key in ("cp", "is", "xen", "other", "imp", "pro", "mixed", "none")}
    assert named["cp"] == {"name": "Cellpose", "source": "cellpose", "model": "cp4", "table": None,
                           "coverage": {"done": 3, "of": 10}}
    assert named["is"] == {"name": "saved stain segmentation", "source": "instanseg", "model": None, "table": None,
                           "coverage": None}                 # InstanSeg is dormant: its saves are named neutrally
    assert named["xen"] == {"name": "Xenium cells", "source": "xenium", "model": None, "table": "xt", "coverage": None}
    assert (named["other"]["source"], named["imp"]["name"], named["imp"]["table"]) == ("imported", "earlier segmentation (imported)", "it")
    assert (named["pro"]["name"], named["pro"]["source"]) == ("proseg (this app)", "proseg")
    for key in ("mixed", "none"):                            # nothing that says what made it: its key
        assert named[key] == {"name": key, "source": "other", "model": None, "table": None if key == "none" else "odd",
                              "coverage": None}
    assert describe_segmentation("cp", None)["name"] == "cp"   # the helper could not say
    # Whose count Compare reads for each: Xenium's table for Xenium's cells, an app's proseg save's table (its record),
    # else the outlines (Cellpose, an import, a table that says nothing, none at all).
    from sopa_companion.engine.zarr_samples import count_source
    counted = {key: tuple(count_source(key, made_by).values()) for key in ("cp", "is", "xen", "other", "imp", "pro",
                                                                          "mixed", "none")}
    assert counted == {"cp": ("outlines", None), "is": ("outlines", None), "xen": ("xenium", "xt"),
                       "other": ("outlines", None), "imp": ("outlines", None), "pro": ("proseg", "pt"),
                       "mixed": ("outlines", None), "none": ("outlines", None)}
    assert count_source("cp", None) == {"count_by": "outlines", "count_table": None}


@needs_sopa
def test_the_catalogue_lists_cell_like_layers_and_serves_their_outlines(client, segmented_copy):
    from sopa_companion.engine import vectors
    sid = client.post("/api/v1/samples/open", json={"path": str(segmented_copy)}, headers=auth()).json()["id"]
    listed = client.get(f"/api/v1/samples/{sid}/segmentations", headers=auth()).json()["segmentations"]
    assert [(e["key"], e["name"], e["source"], e["model"], e["table"], e["count"], e["coverage"]) for e in listed] == [
        ("xenium_cell_boundaries", "Xenium cells", "xenium", None, "xenium_table", 3, None),
        ("cells", "earlier segmentation (imported)", "imported", None, "cells_table", 4, None),
        ("proseg_boundaries", "proseg (this app)", "proseg", None, "proseg_table", 2, None),
        ("proseg_boundaries_2", "proseg (this app)", "proseg", None, "proseg_table_2", 2, None),   # the previous version's
        ("cellpose_boundaries_1", "Cellpose", "cellpose", "cp4_x", None, 2, {"done": 3, "of": 10}),
        # The dormant InstanSeg's earlier save: an ordinary layer, named as any saved stain segmentation.
        ("instanseg_boundaries_1", "saved stain segmentation", "instanseg", "fluorescence_nuclei_and_cells", None, 2, None),
        ("nuclei", "nuclei", "other", None, None, 2, None)]     # never region_of_interest (one shape) nor circles
    # Whose count Compare reads: Xenium's table, the proseg saves' tables (either version's record), else the outlines.
    assert [(e["count_by"], e["count_table"]) for e in listed] == [
        ("xenium", "xenium_table"), ("outlines", None), ("proseg", "proseg_table"), ("proseg", "proseg_table_2"),
        ("outlines", None), ("outlines", None), ("outlines", None)]
    cells = listed[1]
    assert cells["extent"] == pytest.approx([10, 10, 110, 30])
    assert cells["outlines"] == f"/api/v1/samples/{sid}/segmentations/cells/outlines?v={cells['version']}"
    # the outlines: every polygon, as the picker's selection of the whole element, the cells layer only
    packed = client.get(cells["outlines"], headers=auth({"Accept-Encoding": "zstd"}))
    assert packed.headers["content-encoding"] == "zstd" and "immutable" in packed.headers["cache-control"]
    value = vectors.decode(packed.content)
    assert value["nuclei"] is None and value["source"] == {"cells": "cells"}
    assert vectors.to_json(value)["cells"] == client.app.state.zarr.shapes(sid, "cells", None, 10 ** 6)["polygons"]
    assert len(value["cells"]["ids"]) == 4
    plain = client.get(f"/api/v1/samples/{sid}/segmentations/cells/outlines", headers=auth({"Accept-Encoding": "identity"}))
    assert plain.headers["cache-control"] == "private, no-cache" and len(vectors.decode(plain.content)["cells"]["ids"]) == 4
    for missing in ("region_of_interest", "spots", "nope", "..x"):
        assert client.get(f"/api/v1/samples/{sid}/segmentations/{missing}/outlines", headers=auth()).status_code == 404
    # an element written again (a notebook): a new version, exported again, never the old outlines
    nuclei = listed[-1]
    assert len(vectors.decode(client.get(nuclei["outlines"], headers=auth()).content)["cells"]["ids"]) == 2
    subprocess.run([sopa_python(), "-c", REWRITE_NUCLEI, str(segmented_copy)], check=True, capture_output=True)
    client.post("/api/v1/samples/open", json={"path": str(segmented_copy)}, headers=auth())     # inspected again
    fresh = next(e for e in client.get(f"/api/v1/samples/{sid}/segmentations", headers=auth()).json()["segmentations"]
                 if e["key"] == "nuclei")
    assert fresh["version"] != nuclei["version"] and fresh["count"] == 3
    assert len(vectors.decode(client.get(fresh["outlines"], headers=auth()).content)["cells"]["ids"]) == 3
    assert len(client.app.state.zarr._outlines(sid, "nuclei")["ids"]) == 3
    # a forgotten sample is not listed (runs open theirs again)
    client.delete(f"/api/v1/samples/{sid}", headers=auth())
    assert client.get(f"/api/v1/samples/{sid}/segmentations", headers=auth()).status_code == 404


def dense(counts) -> np.ndarray:
    matrix = np.zeros(tuple(counts["shape"]), np.int64)
    for row in range(matrix.shape[0]):
        a, b = counts["indptr"][row], counts["indptr"][row + 1]
        matrix[row, counts["indices"][a:b]] = counts["data"][a:b]
    return matrix


def run_count(sample, out, key="cells", box=None, **rule):
    """The sopa helper's `count` of `key` in `sample` -> (its answer, the counts file), or (the error, None)."""
    args = {"path": str(sample), "key": key, "points_key": "transcripts", "out": str(out), "box": box, **rule}
    done = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", "count", json.dumps(args)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(HERE.parent.parent)})
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    if done.returncode:
        return answer, None
    return answer, np.load(out, allow_pickle=False)


@needs_sopa
def test_counting_keeps_quality_20_genes_in_the_outline_holding_them(segmented, tmp_path):
    def count(key, out, box=None):
        stats, counts = run_count(segmented, tmp_path / out, key, box)
        assert counts is not None, stats
        return stats, counts

    stats, counts = count("cells", "cells.npz")
    assert str(counts["rule"]) == "qv20-v4" and counts["genes"].tolist() == ["G1", "G2", "G3"]
    assert str(counts["count"]) == stats["count"] == "outlines" and stats["table"] is None
    assert counts["ids"].tolist() == ["cell-0", "cell-1", "cell-2", "cell-3"] and counts["index"].tolist() == [0, 1, 2, 3]
    assert np.allclose(np.c_[counts["x"], counts["y"]], [[20, 20], [60, 20], [100, 20], [35, 20]])
    assert counts["data"].dtype == np.uint32 and counts["indices"].dtype == np.int32 and counts["indptr"].dtype == np.int64
    a, b, c, d = dense(counts)
    assert a[1] == 2 and b.tolist() == [0, 0, 4] and c.sum() == 0      # qv 30 and 20 count; B's 4; C holds none
    assert a[0] + d[0] == 4 and a[0] >= 3 and d[1:].sum() == 0          # A's 3 x G1, the one in A and D counted once
    assert (stats["cells"], stats["transcripts"], stats["assigned"]) == (4, 10, 10)   # qv 10 and the control never read
    # only the outlines whose centre lies in the box, their transcripts read whole
    stats, counts = count("cells", "box.npz", [0, 0, 40, 40])
    assert counts["ids"].tolist() == ["cell-0", "cell-3"] and counts["index"].tolist() == [0, 3]
    assert dense(counts).sum() == 6 and stats["cells"] == 2
    stats, counts = count("cells", "none.npz", [500, 500, 600, 600])
    assert counts["shape"].tolist() == [0, 0] and stats["transcripts"] == 0


@needs_sopa
def test_counts_are_kept_per_element_and_box(client, segmented_copy, monkeypatch):
    from sopa_companion.engine.samples import SampleError
    zarr = client.app.state.zarr
    sid = client.post("/api/v1/samples/open", json={"path": str(segmented_copy)}, headers=auth()).json()["id"]
    calls, helper = [], zarr.helper
    monkeypatch.setattr(zarr, "helper", lambda command, args, **kwargs: calls.append((command, args.get("box")))
                        or helper(command, args, **kwargs))
    key, stamp = "xenium_cell_boundaries", zarr.element_stamp(sid, "xenium_cell_boundaries")
    box = zarr.counts(sid, key, [0, 0, 40, 40])
    assert box.name == f"{key}-qv20-v4-{stamp}-0_0_40_40.npz" and calls == [("count", [0.0, 0.0, 40.0, 40.0])]
    assert zarr.counts(sid, key, [10, 10, 30, 30]) == box and len(calls) == 1        # a cached box holding it
    assert zarr.counts(sid, key, [0, 0, 80, 40], cached_only=True) is None
    whole = zarr.counts(sid, key)
    assert whole.name == f"{key}-qv20-v4-{stamp}.npz" and len(calls) == 2
    assert zarr.counts(sid, key, [0, 0, 80, 40]) == whole and len(calls) == 2        # the whole serves any box
    assert np.load(whole)["index"].tolist() == [0, 1, 2]
    # Another run's input rule: its own file, counted by its rule, never served another rule's counts.
    strict = zarr.counts(sid, key, rule={"min_qv": 30})
    assert strict.name == f"{key}-qv30-v4-{stamp}.npz" and len(calls) == 3 and whole.exists()
    assert str(np.load(strict)["rule"]) == "qv30-v4" and dense(np.load(strict)).sum() < dense(np.load(whole)).sum()
    assert zarr.counts(sid, key, rule={"min_qv": 30.0}) == strict and zarr.counts(sid, key) == whole and len(calls) == 3
    for bad in ("region_of_interest", "spots", "nope"):
        with pytest.raises(SampleError):
            zarr.counts(sid, bad)


@needs_sopa
def test_counting_follows_the_runs_own_input_rule(segmented, tmp_path):
    """Every column is counted by the run's input rule (its min-qv and excluded-genes
    options, proseg's own filters), not always quality 20."""
    from sopa_companion.zarr_bridge.helper import count_rule
    default, base = run_count(segmented, tmp_path / "default.npz")
    stats, counts = run_count(segmented, tmp_path / "q30.npz", min_qv=30)
    assert str(counts["rule"]) == stats["rule"] == "qv30-v4" and stats["filters"]["min_qv"] == 30.0
    a, b, c, d = dense(counts)                          # A's G2 at quality 20 and B's 4 x G3 at 25 left out
    assert counts["genes"].tolist() == ["G1", "G2"] and a[1] == 1 and b.sum() == 0
    assert stats["transcripts"] == 5 < default["transcripts"] == 10
    stats, counts = run_count(segmented, tmp_path / "x.npz", exclude="G3")      # matched anywhere in the name
    assert str(counts["rule"]) == count_rule(20, "G3") != "qv20-v4" and stats["transcripts"] == 6
    assert counts["genes"].tolist() == ["G1", "G2"] and dense(counts)[1].sum() == 0
    assert np.array_equal(dense(counts)[[0, 3]], dense(base)[[0, 3], :2])         # the other genes counted as before
    error, counts = run_count(segmented, tmp_path / "bad.npz", exclude="(G3")
    assert counts is None and "not a valid pattern" in error["error"]
    error, counts = run_count(segmented, tmp_path / "bad.npz", min_qv=-1)
    assert counts is None and "0 or more" in error["error"]


# Two outlines across a box's bottom edge (y = 100): "out" (centre y 92, outside) and "in" (centre y 105, inside), "out"
# first in the element, left of "in" and below it, so a whole count gives their overlap to "out" whichever way sopa's
# join breaks the tie (its spatial index's order); a third outline far away.
STRADDLE = r'''
import sys, warnings
warnings.filterwarnings("ignore")
import dask.dataframe as dd, geopandas as gpd, numpy as np, pandas as pd, shapely, spatialdata as sd
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Identity

rows = [(50, 108, "G1"), (50, 110, "G1"), (52, 112, "G1"),          # in "in" only
        (50, 98, "G2"), (50, 100, "G2"), (51, 102, "G2"),           # in both
        (45, 85, "G3"), (45, 90, "G3")]                             # in "out" only
frame = pd.DataFrame(rows, columns=["x", "y", "feature_name"]).astype({"x": np.float32, "y": np.float32})
frame["z"], frame["qv"], frame["is_gene"] = np.float32(1), np.float32(40), True
frame["feature_name"] = pd.Categorical(frame["feature_name"])
points = PointsModel.parse(dd.from_pandas(frame, npartitions=1), coordinates={"x": "x", "y": "y", "z": "z"},
                           feature_key="feature_name", transformations={"global": Identity()})
cells = gpd.GeoDataFrame(geometry=[shapely.box(38, 80, 58, 104), shapely.box(42, 95, 62, 115),
                                   shapely.box(300, 300, 320, 320)], index=["out", "in", "far"])
sd.SpatialData(points={"transcripts": points},
               shapes={"cells": ShapesModel.parse(cells, transformations={"global": Identity()})}).write(sys.argv[1])
'''


@needs_sopa
def test_a_box_count_puts_each_transcript_where_the_whole_count_does(tmp_path):
    """A box count joined only the outlines whose centre is in the box, so a transcript
    in the overlap of one inside and one outside went to the inside one in a box count only. Now every outline of the
    element is joined: an outline counted in a box holds exactly what it holds in the whole count."""
    sample = tmp_path / "straddle.zarr"
    made = subprocess.run([sopa_python(), "-c", STRADDLE, str(sample)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr[-3000:]
    _, whole = run_count(sample, tmp_path / "whole.npz")
    stats, box = run_count(sample, tmp_path / "box.npz", box=[0, 100, 200, 200])

    def by_gene(counts, row):
        values = dense(counts)[row]
        return {gene: int(values[g]) for g, gene in enumerate(counts["genes"].tolist()) if values[g]}
    assert whole["ids"].tolist() == ["out", "in", "far"]
    assert by_gene(whole, 0) == {"G2": 3, "G3": 2} and by_gene(whole, 1) == {"G1": 3}   # the overlap: "out"'s
    assert box["ids"].tolist() == ["in"] and box["index"].tolist() == [1]
    assert by_gene(box, 0) == by_gene(whole, 1) == {"G1": 3}
    assert np.allclose(np.c_[box["x"], box["y"]], [[52, 105]]) and stats["cells"] == 1
    assert stats["transcripts"] == 6 and stats["assigned"] == 3     # the overlap read, and counted in no row here


# ---- a segmentation's own count: its table's (Compare) ---------------------------------------------------------------
# Outlines whose tables hold their own counts: Xenium's cells (its table's rows in another order, one of no outline,
# and an outline it holds no row of), proseg's cells in a table of two elements that share an id (sopa's reading of
# proseg's output: its uns proseg_run alone; its X dense) and a table whose counts are not whole numbers. One
# transcript in each Xenium cell, for the outlines' own count.
TABLED = r"""
import sys, warnings
warnings.filterwarnings("ignore")
import anndata, dask.dataframe as dd, geopandas as gpd, numpy as np, pandas as pd, shapely, spatialdata as sd
from scipy import sparse
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

frame = pd.DataFrame({"x": np.float32([15, 55, 95, 135]), "y": np.float32([20] * 4), "z": np.float32([1] * 4),
                      "feature_name": pd.Categorical(["G1", "G2", "G3", "G1"]), "qv": np.float32([40] * 4),
                      "is_gene": [True] * 4})
points = PointsModel.parse(dd.from_pandas(frame, npartitions=1), coordinates={"x": "x", "y": "y", "z": "z"},
                           feature_key="feature_name", transformations={"global": Identity()})


def layer(boxes, ids):
    return ShapesModel.parse(gpd.GeoDataFrame(geometry=[shapely.box(*b) for b in boxes], index=ids),
                             transformations={"global": Identity()})


boxes = [(10, 10, 30, 30), (50, 10, 70, 30), (90, 10, 110, 30), (130, 10, 150, 30)]
shapes = {"xenium_cells": layer(boxes, ["xa-1", "xb-1", "xc-1", "xd-1"]),
          "proseg_cells": layer(boxes[:3], ["p-0", "p-1", "p-2"]),
          "proseg_more": layer(boxes[2:], ["p-0", "p-9"]), "plain_cells": layer(boxes[:2], ["n-0", "n-1"])}


def table(regions, ids, counts, methods=None, uns=None, dense=False):
    obs = pd.DataFrame({"cell_id": ids, "region": pd.Categorical(regions)}, index=[str(k) for k in range(len(ids))])
    if methods is not None:
        obs["segmentation_method"] = pd.Categorical([methods] * len(ids))
    x = np.asarray(counts, np.float32)
    adata = anndata.AnnData(x if dense else sparse.csr_matrix(x), obs=obs,
                            var=pd.DataFrame(index=["G1", "G2", "G3", "NegControlProbe_1"]), uns=uns or {})
    return TableModel.parse(adata, region=sorted(set(regions)), region_key="region", instance_key="cell_id")


tables = {"xenium_table": table(["xenium_cells"] * 4, ["xc-1", "xa-1", "xz-1", "xb-1"],
                                [[0, 0, 7, 0], [3, 0, 0, 1], [9, 9, 9, 0], [0, 5, 0, 0]],
                                methods="Segmented by interior stain (18S)"),
          "proseg_table": table(["proseg_cells", "proseg_cells", "proseg_more", "proseg_cells", "proseg_more"],
                                ["p-0", "p-2", "p-0", "p-1", "p-9"],
                                [[1, 2, 0, 0], [0, 0, 4, 0], [8, 8, 8, 0], [0, 0, 0, 0], [1, 1, 1, 0]],
                                uns={"proseg_run": {"iterations": 3}}, dense=True),
          "normalised_table": table(["plain_cells"] * 2, ["n-0", "n-1"], [[0.5, 1.2, 0, 0], [1, 0, 0, 0]])}
sd.SpatialData(points={"transcripts": points}, shapes=shapes, tables=tables).write(sys.argv[1])
"""


@pytest.fixture(scope="session")
def tabled(tmp_path_factory):
    path = tmp_path_factory.mktemp("tabled") / "tabled.zarr"
    made = subprocess.run([sopa_python(), "-c", TABLED, str(path)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr[-3000:]
    return path


def by_cell(counts) -> dict:
    """{cell id: {gene: count}} of a counts file."""
    genes = counts["genes"].tolist()
    return {cell: {gene: int(v) for gene, v in zip(genes, row) if v}
            for cell, row in zip(counts["ids"].tolist(), dense(counts))}


@needs_sopa
def test_a_tables_own_counts_are_read_for_the_outlines_it_holds(tabled, tmp_path):
    """Compare reads Xenium's cells by Xenium's table and proseg's by its table's X, each by its own count: each row is
    the outline its instance id names, whatever the table's order; an outline the table holds no row of has none, a row
    of no outline is left out; a table of several elements gives this one's rows only (an id the other element shares
    is not this one's); dense or sparse X, written as the same counts file as an outline count, which says whose count
    it holds. Counts that are not whole numbers, a table that holds no cells of the element, one not there, or a table
    asked for the outlines' count are refused in plain words. Without a table: the transcripts inside its outlines."""
    from sopa_companion.zarr_bridge.helper import COUNT_VERSION
    stats, counts = run_count(tabled, tmp_path / "x.npz", "xenium_cells", table="xenium_table", by="xenium")
    assert counts is not None, stats
    assert str(counts["count"]) == stats["count"] == "xenium" and str(counts["rule"]) == "table:xenium_table"
    assert counts["ids"].tolist() == ["xa-1", "xb-1", "xc-1"] and counts["index"].tolist() == [0, 1, 2]
    assert by_cell(counts) == {"xa-1": {"G1": 3, "NegControlProbe_1": 1}, "xb-1": {"G2": 5}, "xc-1": {"G3": 7}}
    assert counts["genes"].tolist() == ["G1", "G2", "G3", "NegControlProbe_1"]
    assert np.allclose(np.c_[counts["x"], counts["y"]], [[20, 20], [60, 20], [100, 20]])
    assert counts["data"].dtype == np.uint32 and counts["indices"].dtype == np.int32 and counts["indptr"].dtype == np.int64
    assert (stats["cells"], stats["transcripts"], stats["table"]) == (3, 16, "xenium_table")
    from sopa_companion.analysis import worker as W                  # the GPU worker reads it as any counts file
    matrix, ids, x, y, genes = W.read_counts_file(tmp_path / "x.npz")
    assert ids == ["xa-1", "xb-1", "xc-1"] and matrix.shape == (3, 4) and matrix.data.sum() == 16
    stats, box = run_count(tabled, tmp_path / "box.npz", "xenium_cells", [40, 0, 200, 40], table="xenium_table",
                           by="xenium")
    assert box["ids"].tolist() == ["xb-1", "xc-1"] and box["index"].tolist() == [1, 2]
    stats, counts = run_count(tabled, tmp_path / "p.npz", "proseg_cells", table="proseg_table", by="proseg")
    assert str(counts["count"]) == "proseg" and counts["ids"].tolist() == ["p-0", "p-1", "p-2"]
    assert by_cell(counts) == {"p-0": {"G1": 1, "G2": 2}, "p-1": {}, "p-2": {"G3": 4}}      # an empty row stays
    stats, counts = run_count(tabled, tmp_path / "m.npz", "proseg_more", table="proseg_table", by="proseg")
    assert by_cell(counts) == {"p-0": {"G1": 8, "G2": 8, "G3": 8}, "p-9": {"G1": 1, "G2": 1, "G3": 1}}
    for table, by, words in (("normalised_table", "xenium", "doesn't hold whole transcript counts"),
                             ("xenium_table", "xenium", "holds no cells of 'plain_cells'"),
                             ("nope", "xenium", "has no table 'nope'"), ("normalised_table", "outlines", "own count")):
        error, counts = run_count(tabled, tmp_path / "bad.npz", "plain_cells", table=table, by=by)
        assert counts is None and words in error["error"], (table, error)
    stats, counts = run_count(tabled, tmp_path / "o.npz", "xenium_cells")
    assert str(counts["count"]) == stats["count"] == "outlines" and str(counts["rule"]) == f"qv20-v{COUNT_VERSION}"
    assert by_cell(counts) == {"xa-1": {"G1": 1}, "xb-1": {"G2": 1}, "xc-1": {"G3": 1}, "xd-1": {"G1": 1}}


@needs_sopa
def test_the_catalogue_says_whose_count_compare_reads_and_counts_by_it(client, tabled, tmp_path):
    """Through the engine and the real helper: the sample's listing says whose count Compare reads for each
    segmentation (a table carrying proseg's own run record, sopa's reading of proseg's output, is proseg's count;
    Xenium's cells, Xenium's; a table that says nothing, its outlines'), and ZarrSamples.counts reads a table's own
    count by its table, kept by its name and the table's version."""
    sample = tmp_path / "samples" / "tabled.zarr"
    shutil.copytree(tabled, sample)
    zarr = client.app.state.zarr
    sid = client.post("/api/v1/samples/open", json={"path": str(sample)}, headers=auth()).json()["id"]
    listed = {e["key"]: (e["source"], e["count_by"], e["count_table"]) for e in
              client.get(f"/api/v1/samples/{sid}/segmentations", headers=auth()).json()["segmentations"]}
    assert listed == {"xenium_cells": ("xenium", "xenium", "xenium_table"),
                      "proseg_cells": ("other", "proseg", "proseg_table"),
                      "proseg_more": ("other", "proseg", "proseg_table"),
                      "plain_cells": ("other", "outlines", None)}
    own = zarr.counts(sid, "xenium_cells", count_by="xenium", table="xenium_table")
    version = zarr.table_stamp(sid, "xenium_table")
    assert own.name.startswith("xenium_cells-t") and version is not None
    with np.load(own) as counts:
        assert str(counts["count"]) == "xenium" and by_cell(counts)["xb-1"] == {"G2": 5}
    assert zarr.counts(sid, "xenium_cells", count_by="xenium", table="xenium_table") == own
    outlines = zarr.counts(sid, "xenium_cells")
    assert outlines != own and own.exists()
    with np.load(outlines) as counts:
        assert str(counts["count"]) == "outlines" and by_cell(counts)["xd-1"] == {"G1": 1}


# ---- API mode's bind, the helper's process, Compare's outline reads, a save's API arguments ------------------------------
def stood_in(config, tmp_path, inspected: dict):
    """ZarrSamples whose helper answers `inspect` with `inspected` (the arguments it was asked with are kept)."""
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    asked = []
    def helper(command, args, **kwargs):
        asked.append((command, dict(args)))
        if command == "inspect":
            return {**inspected, "points_key": args.get("points_key") or inspected.get("points_key")}
        raise AssertionError(f"helper {command} was not expected")
    samples.helper = helper
    return samples, asked


def plain_store(path: Path) -> Path:
    import zarr
    zarr.open_group(str(path), mode="w")
    (path / "points").mkdir()
    return path


def inspected_of(path: Path) -> dict:
    from sopa_companion.engine.zarr_samples import ZarrSamples
    return {"schema": 2, "path": str(path), "name": "view", "points_key": "transcripts",
            "points": [{"key": key, "count": 10, "extent": [0.0, 0.0, 50.0, 50.0], "columns": {}}
                       for key in ("transcripts", "transcripts_2")],
            "images": [], "shapes": [], "labels": [], "tables": [], "attrs": {}, "areas": 1,
            "mtime": ZarrSamples._mtime(str(path)), "units": {"frame": "um", "why": ""}}


def test_bind_names_the_sessions_sample_and_knows_only_it(config, tmp_path):
    """API mode (bind, server/api_serve.py): the view is opened as Open does, named as the notebook gives it (a view
    folder is called view.zarr), with the transcripts the notebook names, and it is the only sample the engine knows."""
    other = plain_store(tmp_path / "other.zarr")
    view = plain_store(tmp_path / "view.zarr")
    samples, asked = stood_in(config, tmp_path, inspected_of(view))
    try:
        samples.open(str(other))
        detail = samples.bind(str(view), name="sample-7 (in memory)", points_key="transcripts_2")
        assert detail["name"] == "sample-7 (in memory)" and detail["points_key"] == "transcripts_2"
        assert asked[-1] == ("inspect", {"path": str(view.resolve()), "points_key": "transcripts_2"})
        assert [s["id"] for s in samples.list()] == [detail["id"]] and not samples.known(samples.list()[0]["id"] + "x")
        assert samples.entry(detail["id"])["name"] == "sample-7 (in memory)"
        again = samples.detail(detail["id"])
        assert again["name"] == "sample-7 (in memory)" and len(asked) == 2      # kept: no second inspection
        with pytest.raises(Exception):
            samples.bind(str(tmp_path / "missing.zarr"), name="x", points_key=None)
    finally:
        samples.close()


FAKE_HELPER = r'''#!{python}
import ctypes, json, os, sys, time
command = sys.argv[sys.argv.index("-m") + 2]
if command == "slow":
    time.sleep(30)
signal = ctypes.c_int(0)
ctypes.CDLL(None).prctl(2, ctypes.byref(signal))           # PR_GET_PDEATHSIG
print(json.dumps({{"sid": os.getsid(0), "pid": os.getpid(), "pdeathsig": signal.value}}))
'''


def fake_sopa(tmp_path) -> str:
    import sys
    path = tmp_path / "fake-sopa-python"
    path.write_text(FAKE_HELPER.format(python=sys.executable))
    path.chmod(0o755)
    return str(path)


def test_a_helper_runs_in_a_session_of_its_own_and_a_read_only_one_dies_with_the_engine(config, tmp_path):
    """The helper shared the engine's process group, so a terminal's hang-up or Ctrl+C killed a save
    mid-write. Every helper leads its own session; a read-only command also gets the parent-death signal (it goes with
    the engine), never a save (its journalled write finishes or rolls back by itself)."""
    import shutil as sh
    from sopa_companion.engine.zarr_samples import ZarrSamples
    config["zarr"]["python"] = fake_sopa(tmp_path)
    samples = ZarrSamples(config)
    try:
        read = samples.helper("inspect", {})
        save = samples.helper("save", {})
        assert read["sid"] == read["pid"] and save["sid"] == save["pid"]
        assert save["pdeathsig"] == 0
        assert read["pdeathsig"] == (15 if sh.which("setpriv") else 0)
    finally:
        samples.close()


def outlines_engine(config, tmp_path, monkeypatch):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    config["zarr"]["python"] = fake_sopa(tmp_path)
    samples = ZarrSamples(config)
    element = tmp_path / "sample.zarr" / "shapes" / "cells"
    element.mkdir(parents=True)
    (element / "zarr.json").write_text("{}")
    monkeypatch.setattr(samples, "path", lambda sid: str(tmp_path / "sample.zarr"))
    monkeypatch.setattr(samples, "detail", lambda sid: {"shapes": [{"key": "cells"}], "points_key": None})
    return samples, element


def test_an_outline_export_stops_with_its_comparison_and_never_waits_behind_another(config, tmp_path, monkeypatch):
    """Compare's outline export had no stop: a comparison replaced or the engine closing left it
    running (after the engine, an orphan with no time limit), and a new comparison waited behind another caller's
    export. The export and the wait for another's both stop with `should_stop`."""
    import threading
    import time
    from sopa_companion.engine.zarr_samples import HelperFailed
    samples, element = outlines_engine(config, tmp_path, monkeypatch)
    original = samples.helper
    monkeypatch.setattr(samples, "helper", lambda command, args, **kw: original("slow", args, **kw))
    stop = threading.Event()
    failed = []
    def read():
        try:
            samples._outlines("s", "cells", should_stop=stop.is_set)
        except HelperFailed as exc:
            failed.append((str(exc), exc.status, time.monotonic()))
    worker = threading.Thread(target=read)
    worker.start()
    time.sleep(0.5)
    stop.set()
    asked = time.monotonic()
    worker.join(5)
    assert failed and failed[0][:2] == ("Stopped", 409) and failed[0][2] - asked < 2
    guard = samples.exporting[("s", "cells")]
    with guard:                                                  # another caller's export of the same element
        started = time.monotonic()
        with pytest.raises(HelperFailed, match="Stopped"):
            samples._outlines("s", "cells", should_stop=lambda: time.monotonic() - started > 0.3)
        assert time.monotonic() - started < 2
    samples.close()


def test_an_export_or_a_count_of_an_element_written_meanwhile_is_never_kept_under_its_old_stamp(config, tmp_path, monkeypatch):
    """The helper reads the element in its own process; one written again meanwhile (a notebook,
    another engine) left an export or counts of either version under the stamp read before. The stamp is read again
    after: an export is made once more (then refused), counts are refused; nothing stays under the old stamp."""
    from sopa_companion.engine.samples import SampleError
    samples, element = outlines_engine(config, tmp_path, monkeypatch)
    writes = []
    def helper(command, args, **kwargs):
        out = Path(args["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if command == "shapes":
            for suffix in (".bin", ".json"):
                Path(str(out) + suffix).write_text("x")
        else:
            out.write_text("x")
        writes.append(command)
        (element / f"part-{len(writes)}").write_text("written again")   # the element changes while it is read
        return {}
    monkeypatch.setattr(samples, "helper", helper)
    base = samples.cache / "s" / "shapes" / "cells"
    with pytest.raises(SampleError) as exported:
        samples._shapes_base("s", "cells")
    assert exported.value.status == 409 and writes == ["shapes", "shapes"]
    assert not any(Path(str(base) + suffix).exists() for suffix in (".bin", ".json", ".stamp"))
    monkeypatch.setattr(samples, "_segmentation", lambda sid, key: {"key": key})
    from sopa_companion.zarr_bridge.helper import COUNT_RULE
    monkeypatch.setattr(samples, "count_rule", lambda rule: (20.0, None, COUNT_RULE))   # counted the current way
    with pytest.raises(SampleError) as counted:
        samples.counts("s", "cells")
    assert counted.value.status == 409 and not list((samples.cache / "s" / "counts").glob("*.npz"))
    calls = len(writes)
    monkeypatch.setattr(samples, "helper", lambda command, args, **kw: (writes.append(command), Path(args["out"]).write_text("x"))[0] or {})
    assert samples.counts("s", "cells").exists() and len(writes) == calls + 1       # a quiet element: counted, kept
    samples.close()


def test_a_save_in_an_api_session_hands_the_helper_its_four_destinations(config, monkeypatch):
    """`path` (the object's zarr or None), `view` (the engine's own when that is not it), `frame_from`, `stage_to` and
    `consolidate` go to the helper in place of the sample alone."""
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    asked = []
    monkeypatch.setattr(samples, "detail", lambda sid: {"points_key": "transcripts"})
    monkeypatch.setattr(samples, "path", lambda sid: "/data/view.zarr")
    monkeypatch.setattr(samples, "helper", lambda command, args, **kw: asked.append(args) or {"cells": 1})
    try:
        samples.save("s", Path("/runs/r/output"), "cells_1", "table_1", False, False, {"kind": "proseg"})
        assert asked[-1]["path"] == "/data/view.zarr" and "stage_to" not in asked[-1]
        where = {"path": None, "view": "/data/view.zarr", "frame_from": "/data/view.zarr", "stage_to": "/d/1/staged.zarr",
                 "consolidate": True}
        samples.save("s", Path("/runs/r/output"), "cells_1", "table_1", False, False, {"kind": "proseg"}, deliver=where)
        assert {k: asked[-1][k] for k in where} == where
    finally:
        samples.close()


def test_the_layers_a_run_saved_are_found_from_the_samples_own_record(config, tmp_path, monkeypatch):
    """An older run's "Saved already": a stain save's shapes name their run (their attrs), a proseg save's table names it
    in its uns (the helper's `run`, else read from the zarr here: uns/sopa_companion, or the previous version's
    uns/proseg_live)."""
    import zarr
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    root = tmp_path / "sample.zarr"
    run = zarr.create_array(store=str(root / "tables" / "proseg_table_2" / "uns" / "sopa_companion" / "run"), shape=(),
                            dtype=str)
    run[()] = "older-run"
    oldest = zarr.create_array(store=str(root / "tables" / "proseg_table_3" / "uns" / "proseg_live" / "run"), shape=(),
                               dtype=str)
    oldest[()] = "previous-version-run"
    made = {"shapes": [{"key": "cellpose_boundaries_1", "app": {"run": "stain-run"}}, {"key": "cells", "app": None}],
            "tables": [{"key": "proseg_table", "region": ["proseg_boundaries"], "proseg": True, "run": "proseg-run"},
                       {"key": "proseg_table_2", "region": ["proseg_boundaries_2"], "proseg": True},
                       {"key": "proseg_table_3", "region": ["proseg_boundaries_3"], "proseg": True}]}
    monkeypatch.setattr(samples, "_made_by", lambda sid: made)
    monkeypatch.setattr(samples, "path", lambda sid: str(root))
    try:
        assert samples.saved_from("s", "stain-run") == ["cellpose_boundaries_1"]
        assert samples.saved_from("s", "proseg-run") == ["proseg_boundaries", "proseg_table"]
        assert samples.saved_from("s", "older-run") == ["proseg_boundaries_2", "proseg_table_2"]
        assert samples.saved_from("s", "previous-version-run") == ["proseg_boundaries_3", "proseg_table_3"]
        assert samples.saved_from("s", "never-saved") == []
    finally:
        samples.close()


def test_a_save_cut_off_is_never_saved_from_and_a_stain_saves_record_is_read_from_its_shapes(config, tmp_path,
                                                                                               monkeypatch):
    """An older run: a save killed before it committed leaves its elements until the next save into the
    sample rolls them back (its journal still "writing"): never "Saved already". Another zarr's journal, and a committed
    one, are not this. A stain save's record (result_revision, filters) is read from its shapes' own metadata, v3 or v2."""
    import json
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(config)
    root = tmp_path / "sample.zarr"
    records = {"cellpose_boundaries_1": {"run": "r", "result_revision": 7, "filters": {"areas": []}},
               "cellpose_boundaries_2": {"run": "r", "result_revision": 9}}
    for key, record in records.items():
        (root / "shapes" / key).mkdir(parents=True)
        (root / "shapes" / key / "zarr.json").write_text(json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"sopa_companion": record}}))
    (root / "shapes" / "cellpose_boundaries_3").mkdir()                   # a v2 element, saved by the previous version
    (root / "shapes" / "cellpose_boundaries_3" / ".zattrs").write_text(json.dumps({"proseg_live": {"run": "r"}}))
    backups = tmp_path / ".proseg-live-backups"
    for name, state, written in (("sample.zarr-20261023-101010-4242", "writing", "cellpose_boundaries_2"),
                                 ("sample.zarr-20261023-090909-4141", "committed", "cellpose_boundaries_1"),
                                 ("sample.zarr-copy-20261023-101010-4343", "writing", "cellpose_boundaries_3")):
        (backups / name).mkdir(parents=True)
        (backups / name / "journal.json").write_text(json.dumps({"state": state, "written": [["shapes", written]]}))
    made = {"shapes": [{"key": key, "app": {"run": "r"}} for key in (*records, "cellpose_boundaries_3")], "tables": []}
    monkeypatch.setattr(samples, "_made_by", lambda sid: made)
    monkeypatch.setattr(samples, "path", lambda sid: str(root))
    try:
        assert samples.saved_from("s", "r") == ["cellpose_boundaries_1", "cellpose_boundaries_3"]
        assert samples.saved_provenance("s", "cellpose_boundaries_1") == records["cellpose_boundaries_1"]
        assert samples.saved_provenance("s", "cellpose_boundaries_3") == {"run": "r"}
        assert samples.saved_provenance("s", "missing") is None and samples.saved_provenance("s", "../shapes") is None
    finally:
        samples.close()
