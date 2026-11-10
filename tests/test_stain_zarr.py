"""Real SpatialData reads and transactional shapes-only saves, with deterministic segmentation."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import HERE, needs_sopa, sopa_python, wait_for
from test_zarr import synthetic, sample_path, read_spatialdata  # shared private-copy fixtures
from sopa_companion.engine.runs import Runs, RunError
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples


def snapshot(path):
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def stain_runs(config):
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain")
    config["zarr"]["python"] = sopa_python()
    config["proseg"]["max_region_um"] = 100
    samples = ZarrSamples(config)
    manager = Runs(config, Samples(config), samples)
    manager.stain.gpu = ""                 # the subprocess doubles and every real read stay CPU-only
    yield manager
    manager.close(save=False)


@needs_sopa
@pytest.mark.parametrize("model", ["cpsam", "cpsam_20250102", "fluorescence_nuclei_and_cells"])
def test_stain_save_is_shapes_only_has_provenance_and_can_seed_proseg(stain_runs, sample_path, tmp_path, monkeypatch, model):
    runs = stain_runs
    path, digest = None, None
    method = "instanseg" if model == "fluorescence_nuclei_and_cells" else "cellpose"
    if method == "instanseg":
        runs.stain.instanseg_enabled = True      # InstanSeg is dormant unless its internal switch is on
        runs.stain.instanseg_python = sys.executable
        runs.stain.instanseg_module = "sopa_companion.tests.fake_instanseg"
    if model not in ("cpsam", "fluorescence_nuclei_and_cells"):
        path = str(tmp_path / model)
        Path(path).write_bytes(b"custom weights")
        digest = hashlib.sha256(b"custom weights").hexdigest()
        monkeypatch.setenv("FAKE_STAIN_MODELS", json.dumps([
            dict(name=model, kind="custom", path=path, date="2025-01-02")]))
    detail = runs.zarr.open(str(sample_path))
    before = snapshot(sample_path)
    run = runs.get(runs.create(detail["id"], kind="stain", region={"kind": "full"},
                    stain={"method": method, "channels": ["DAPI", "Boundary"], "tissue_only": False,   # no tissue to find
                           "image": "morphology_focus", "nucleus": "DAPI", "model": model,
                           "model_path": "/page/cannot/choose/a/path", "model_sha256": "untrusted"})["id"])
    assert run.meta["state"] == "paused"
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: run.meta["state"] in ("done", "failed") and not run.worker.is_alive(), timeout=60)
    assert run.meta["state"] == "done", run.meta
    assert run.meta["mode"] == "tiled" and snapshot(sample_path) == before
    assert run.meta["save_defaults"]["shapes_key"] == f"{method}_boundaries_1"
    key = f"{method}_boundaries_1"
    runs.save_to_zarr(run.meta["id"], key, None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving", timeout=120)
    assert run.meta["zarr_save"]["state"] == "saved", run.meta["zarr_save"]
    assert run.meta["zarr_save"]["table_key"] is None
    assert run.meta["save_defaults"]["shapes_key"] == f"{method}_boundaries_2"
    text = read_spatialdata(sample_path,
        "import json, zarr\n"
        f"a = zarr.open_group(str(s.path)+'/shapes/{key}',mode='r').attrs\n"
        "p = a['sopa_companion']\n"
        f"print(json.dumps(['proseg_live' in a, len(s.shapes['{key}']), list(s.tables), p['method'], p['model'], "
        "p['parameters'].get('nucleus', p['parameters'].get('channels', [None])[0]), "
        "p['region']['kind'], sopa.utils.get_boundaries(s, return_key=True)[0], "
        f"p['model_path'], p['model_sha256'], all(i.startswith('{method}-') for i in s.shapes['{key}'].index)]))")
    assert json.loads(text) == [False, 6, [], method, model, "DAPI", "full", key, path, digest, True]
    detail = runs.zarr.detail(detail["id"])
    assert any(p.get("key") == key for p in detail["prior_options"]["cells"])
    with pytest.raises(RunError, match="already exist"):
        runs.save_to_zarr(run.meta["id"], key, None)
    with pytest.raises(RunError, match="only by case"):
        runs.save_to_zarr(run.meta["id"], key.upper(), None, overwrite=True)
    # A failed replacement restores the old shapes, their provenance and the sample's default boundary.
    shape_before = snapshot(sample_path / "shapes" / key)
    with monkeypatch.context() as patch:
        patch.setenv("SOPA_COMPANION_TEST_FAIL_AFTER", "shapes")
        runs.save_to_zarr(run.meta["id"], key, None, overwrite=True)
        wait_for(lambda: run.meta["zarr_save"]["state"] != "saving", timeout=120)
        assert run.meta["zarr_save"]["state"] == "failed"
        assert snapshot(sample_path / "shapes" / key) == shape_before
    runs.save_to_zarr(run.meta["id"], key, None, overwrite=True)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving", timeout=120)
    assert run.meta["zarr_save"]["state"] == "saved"
    # Removed areas stay out of the saved shapes (whole cells by centre): the engine passes the kept cells (keep) and
    # the committed rules, recorded with them as `filters` (no per-rule counts in the provenance).
    x0, y0 = run.meta["region"]["bbox"][:2]
    runs.stain.exclusions.change(run, {"version": 0, "points": [[x0+9, y0+9], [x0+17, y0+9], [x0+17, y0+13], [x0+9, y0+13]]})
    state = run.meta["exclusions"]
    assert {k: state[k] for k in ("version", "count", "cells", "kept", "removed", "in_areas", "size", "by_size",
                                  "brightness", "by_brightness", "error")} == \
        {"version": 1, "count": 1, "cells": 6, "kept": 4, "removed": 2, "in_areas": 2, "size": None, "by_size": 0,
         "brightness": {}, "by_brightness": 0, "error": None}
    assert run.meta["summary"]["cells"] == 4
    runs.save_to_zarr(run.meta["id"], f"{key}_kept", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving", timeout=120)
    assert run.meta["zarr_save"]["state"] == "saved", run.meta["zarr_save"]
    kept = read_spatialdata(sample_path,
        "import json, zarr\n"
        f"p = zarr.open_group(str(s.path)+'/shapes/{key}_kept',mode='r').attrs['sopa_companion']\n"
        f"print(json.dumps([len(s.shapes['{key}_kept']), len(p['filters']['areas']), "
        "sorted({'removed_cells', 'removed_areas', 'min_area_um2', 'min_brightness'} & p.keys())]))")
    assert json.loads(kept) == [4, 1, []]
    # A brightness range (engine/stain_intensity.py): the save leaves out exactly the cells the engine counts as
    # outside it, from the sample's own image.
    runs.stain.exclusions.change(run, {"version": 1}, 1)
    spread = runs.stain.exclusions.brightness(run)
    limits = {"min": spread["median"], "max": None}
    runs.stain.exclusions.change(run, {"version": 2, "channel": spread["channel"], **limits}, brightness=True)
    dim = run.meta["exclusions"]["by_brightness"]
    assert spread["cells"] == spread["measured"] == 6 and 0 < dim < 6
    assert run.meta["summary"]["cells"] == 6 - dim
    runs.save_to_zarr(run.meta["id"], f"{key}_bright", None)
    wait_for(lambda: run.meta["zarr_save"]["state"] != "saving", timeout=120)
    assert run.meta["zarr_save"]["state"] == "saved", run.meta["zarr_save"]
    bright = read_spatialdata(sample_path,
        "import json, zarr\n"
        f"p = zarr.open_group(str(s.path)+'/shapes/{key}_bright',mode='r').attrs['sopa_companion']\n"
        f"print(json.dumps([len(s.shapes['{key}_bright']), p['filters']['brightness'], p['filters']['areas']]))")
    assert json.loads(bright) == [6 - dim, {spread["channel"]: limits}, []]
    # The ordinary preparation path consumes it as any other shapes prior, with no nuclei layer required.
    out = tmp_path / "seed.parquet"
    result = runs.zarr.prepare(detail["id"], {"kind": "full", "bbox": detail["extent"]},
                              {"cells": {"kind": "shapes", "key": key}, "nuclei": None}, out)
    assert result["assigned"] > 0 and result["seeded_from"] == "cells"


@needs_sopa
def test_stain_helper_reads_tiles_resolves_seams_and_clips_lasso(stain_runs, sample_path, tmp_path):
    detail = stain_runs.zarr.open(str(sample_path))
    before = snapshot(sample_path)
    out = tmp_path / "geometry"
    (out / "stain").mkdir(parents=True)
    (out / "output").mkdir()
    (out / "stain/control.json").write_text("{}")
    job = dict(path=str(sample_path), image=stain_runs.zarr._image_info(detail["id"], "morphology_focus"),
               settings=dict(nucleus="DAPI", cytoplasm="Boundary"), out=str(out), tile_pixels=128, overlap_pixels=32,
               region=dict(kind="polygon", bbox=[0,0,100,100], points=[[0,0],[100,0],[0,100]]))
    path = out / "job.json"
    path.write_text(json.dumps(job))
    result = subprocess.run([sopa_python(), str(HERE / "stain_geometry_check.py"), str(path)],
        capture_output=True, text=True, timeout=180, env={**os.environ, "PYTHONPATH": str(HERE.parent.parent)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["cells"] == 2
    assert snapshot(sample_path) == before


@needs_sopa
def test_a_stain_save_without_cells_says_so_and_leaves_the_sample_alone(sample_path, tmp_path):
    """A stain run that found no cell (or whose every cell a rule removed) could be saved: spatialdata then failed
    on the empty shapes with a raw IndexError. The save says so in plain words, before anything is written."""
    out = tmp_path / "run"
    out.mkdir()
    before = snapshot(sample_path)

    def save(features, **rules):
        (out / "boundaries.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        args = {"path": str(sample_path), "points_key": "transcripts", "output": str(out), "kind": "stain",
                "shapes_key": "cellpose_boundaries_1", "table_key": None, "set_default": True, "overwrite": False,
                "provenance": {"kind": "stain", "method": "cellpose"}, **rules}
        done = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", "save", json.dumps(args)],
                              capture_output=True, text=True, timeout=300,
                              env={**os.environ, "PYTHONPATH": str(HERE.parent.parent)})
        return done.returncode, json.loads(done.stdout.strip().splitlines()[-1])

    square = {"type": "Feature", "id": 1, "properties": {},
              "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [14, 10], [14, 14], [10, 14], [10, 10]]]}}
    for features, rules, words in (([], {}, "this run found none, or all were removed"),
                                   ([square], {"min_area_um2": 100}, "this run found none, or all were removed")):
        status, answer = save(features, **rules)
        assert status == 1 and answer["error"] == f"There are no cells to save: {words}", answer
    assert snapshot(sample_path) == before and not (sample_path.parent / ".proseg-live-backups").exists()


def helper_save(sample_path, tmp_path, mask, **extra):
    output = tmp_path / "kept-output"
    output.mkdir(exist_ok=True)
    features = [dict(type="Feature", id=i, properties={}, geometry=dict(type="Polygon", coordinates=[
        [[x, 10], [x+20, 10], [x+20, 30], [x, 30], [x, 10]]])) for i, x in ((7, 10), (10, 40), (42, 70))]
    (output / "boundaries.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)))
    np.save(tmp_path / "keep.npy", mask, allow_pickle=False)
    filters = dict(size=dict(min=100, max=500), brightness={"DAPI": dict(min=20, max=70),
                                                        "Boundary": dict(min=None, max=90)})
    args = dict(path=str(sample_path), points_key="transcripts", output=str(output), kind="stain",
                shapes_key="kept_stain", table_key=None, set_default=False, overwrite=False,
                keep=str(tmp_path / "keep.npy"), provenance=dict(kind="stain", method="cellpose", filters=filters,
                    resolution=dict(level=1, factor=2, um_per_px=.425)), **extra)
    result = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", "save", json.dumps(args)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks")})
    return result, json.loads(result.stdout.strip().splitlines()[-1]), filters


@needs_sopa
def test_keep_mask_is_authoritative_and_records_ranges(sample_path, tmp_path):
    # Legacy arguments would remove every cell; the committed mask alone decides this save.
    result, answer, filters = helper_save(sample_path, tmp_path, np.array([True, False, True]),
                                         min_area_um2=10000, min_brightness={"channel": "missing", "value": 999})
    assert result.returncode == 0, result.stdout + result.stderr
    assert answer["cells"] == 2 and answer["table_key"] is None and "pairing" not in answer
    saved = read_spatialdata(sample_path,
        "import json, zarr\n"
        "p = zarr.open_group(str(s.path)+'/shapes/kept_stain', mode='r').attrs['sopa_companion']\n"
        "print(json.dumps([list(s.shapes['kept_stain'].index), p, list(s.tables)]))")
    ids, provenance, tables = json.loads(saved)
    assert ids == ["cellpose-7", "cellpose-42"] and not tables
    assert provenance["filters"] == filters
    assert not {"removed_areas", "min_area_um2", "min_brightness", "removed_cells", "removed_version"} & provenance.keys()
    assert provenance["resolution"] == dict(level=1, factor=2, um_per_px=.425)
    assert "pairing" not in provenance
    inspected = read_spatialdata(sample_path,
        "import json\nfrom sopa_companion.zarr_bridge.helper import inspect, segmentations\n"
        "data = inspect({'path': str(s.path)})\n"
        "print(json.dumps([data['areas'], {g['key']: g['median_area_um2'] for g in data['shapes']},\n"
        "                  segmentations({'path': str(s.path)})]))")
    marker, areas, catalogue = json.loads(inspected)
    assert marker == 1 and areas["kept_stain"] == 400 and areas["nucleus_boundaries"] == 144
    app = next(s["app"] for s in catalogue["shapes"] if s["key"] == "kept_stain")
    assert app["resolution"] == provenance["resolution"]


@needs_sopa
@pytest.mark.parametrize("mask,words", [
    (np.array([True, False]), "The cells changed since the save was asked for. Save again."),
    (np.array([[True, False, True]]), "The cells changed since the save was asked for. Save again."),
    (np.array([1, 0, 1]), "The cells changed since the save was asked for. Save again."),
    (np.array([False, False, False]), "There are no cells to save: this run found none, or all were removed"),
])
def test_keep_mask_refusals_do_not_write(sample_path, tmp_path, mask, words):
    before = snapshot(sample_path)
    result, answer, _ = helper_save(sample_path, tmp_path, mask)
    assert result.returncode == 1 and answer["error"] == words, result.stdout + result.stderr
    assert snapshot(sample_path) == before
    assert not (sample_path.parent / ".proseg-live-backups").exists()
