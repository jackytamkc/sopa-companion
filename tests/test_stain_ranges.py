"""Committed ranges and per-cell values share exactly the page's float32 inputs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from sopa_companion.engine import stain_rules as rules
from sopa_companion.engine import vectors
from sopa_companion.engine.runs import RunError
from test_stain import runs, start, ended, stain_cpu
from conftest import wait_for

CASES = json.loads((Path(__file__).parent / "fixtures/stain_ranges.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_shared_range_cases(case):
    actual = rules.keep(np.asarray(case["area"], np.float32),
                        {c: np.asarray(v, np.float32) for c, v in case["brightness"].items()},
                        case["in_area"], case["rules"])
    assert actual.dtype == bool and actual.tolist() == case["keep"]


def test_legacy_state_is_normalised_without_mutation_and_digest_ignores_edit_history():
    old = dict(version=4, next=2, areas=[dict(id=1, points=[[0, 0], [1, 0], [0, 1]])],
               min_area_um2=10, min_brightness=dict(channel="DAPI", value=20))
    copy = deepcopy(old)
    state = rules.normalise(old)
    assert old == copy and state["size"] == dict(min=10, max=None)
    assert state["brightness"] == {"DAPI": dict(min=20, max=None)} and state["areas_version"] == 4
    assert rules.normalise(state) == state
    changed = deepcopy(state)
    changed.update(version=9, areas_version=8, next=5)
    changed["areas"][0]["id"] = 4
    assert rules.digest(changed) == rules.digest(state)
    changed["size"]["max"] = 100
    assert rules.digest(changed) != rules.digest(state)


@pytest.mark.parametrize("body", [{}, {"min_area_um2": 2}, {"min": 10, "max": 9}, {"min": True, "max": None},
                                  {"min": -1, "max": None}, {"min": None, "max": float("nan")},
                                  {"min": None, "max": 1e6 + 1}])
def test_invalid_and_old_bodies_are_refused(body):
    with pytest.raises(ValueError):
        rules.check(body, "size", ["DAPI"])


def test_empty_range_clears_only_named_channel_and_equal_limits_are_valid():
    assert rules.check(dict(min=None, max=None, channel="DAPI"), "brightness", ["DAPI"]) is None
    assert rules.check(dict(min=0, max=0), "size") == dict(min=0, max=0)
    with pytest.raises(ValueError, match="no channel"):
        rules.check(dict(min=None, max=None, channel="missing"), "brightness", ["DAPI"])


def test_one_copy_of_the_saves_mask_and_of_the_ring_geometry():
    """The cells a Save keeps are Runs._stain_keep's alone (StainExclusions had a second copy nothing called, which its
    tests pinned instead of what Save runs), and a ring's centre and area come from centres_and_areas over the outline
    arrays (areas_of, over shapely polygons, had no caller)."""
    from sopa_companion.engine.stain_exclusions import StainExclusions
    assert not hasattr(StainExclusions, "keep_mask") and not hasattr(rules, "areas_of")


def test_ranges_commit_once_undo_restores_digest_and_vector_ids_stay_fixed(runs, monkeypatch):
    run = start(runs)
    wait_for(lambda: ended(run))
    exclusions = runs.stain.exclusions
    initial = exclusions.public(run)
    packed = vectors.decode(runs.stain.outline_bytes(run, [18, 10, 20, 12]))
    assert packed["cells"]["ids"].tolist() == [3]
    assert packed["areas_version"] == 0
    monkeypatch.setattr(exclusions, "file_brightness", lambda *args: np.array([10, 20, 30, 40, 50, np.nan], np.float32))
    body = dict(version=0, min=20, max=40, channel="DAPI")
    changed = exclusions.change(run, body, brightness=True)
    assert changed["version"] == 1 and changed["kept"] == 4 and changed["by_brightness"] == 2
    assert changed["areas_version"] == 0 and changed["digest"] != initial["digest"]
    assert vectors.decode(runs.stain.outline_bytes(run))["cells"]["ids"].tolist() == [1, 2, 3, 4, 5, 6]
    values = vectors.decode_cells(runs.stain.cell_bytes(run, packed["set"], "DAPI"))
    assert values["ids"].tolist() == [1, 2, 3, 4, 5, 6]
    assert values["area"].tolist() == [4] * 6 and np.isnan(values["brightness"]["DAPI"][-1])
    assert np.load(runs._stain_keep(run, exclusions.load(run))).tolist() == [False, True, True, True, False, True]
    back = exclusions.change(run, dict(version=1, min=None, max=None, channel="DAPI"), brightness=True)
    assert back["version"] == 2 and back["digest"] == initial["digest"] and back["kept"] == 6
    with pytest.raises(RunError, match="another window"):
        exclusions.change(run, body, brightness=True)
    removed = exclusions.change(run, dict(version=2, points=[[18, 10], [20, 10], [20, 12], [18, 12]]))
    assert removed["areas_version"] == 1 and removed["in_areas"] == 1
    assert vectors.decode(runs.stain.outline_bytes(run))["cells"]["ids"].tolist() == [1, 2, 4, 5, 6]
    flags = vectors.decode_cells(runs.stain.cell_bytes(run, packed["set"]))["flags"]
    assert flags.tolist() == [0, 0, 1, 0, 0, 0]


def test_tile_ids_use_offsets_when_counts_are_missing_and_refuse_mismatched_counts(runs, monkeypatch):
    run = start(runs)
    wait_for(lambda: ended(run))
    run.meta["progress"]["resolved"] = False
    path = run.dir / "stain/tiles.json"
    records = json.loads(path.read_text())
    for record in records:
        record.pop("cells")
    path.write_text(json.dumps(records))
    view = vectors.decode(runs.stain.outline_bytes(run, [26, 10, 28, 12]))
    assert view["cells"]["ids"].tolist() == [5]
    values = vectors.decode_cells(runs.stain.cell_bytes(run, view["set"]))
    assert values["ids"].tolist() == list(range(1, 7))
    load = np.load
    def visible_only(path, *args, **kwargs):
        assert Path(path).name == "tile-000004.npz", "a pan reopened an offscreen tile"
        return load(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np, "load", visible_only)
        view = vectors.decode(runs.stain.outline_bytes(run, [26, 10, 28, 12]))
        assert view["cells"]["ids"].tolist() == [5]
    records[0]["cells"] = 2
    path.write_text(json.dumps(records))
    with pytest.raises(RunError, match="match their tile"):
        runs.stain.outline_bytes(run, [26, 10, 28, 12])
    with pytest.raises(RunError, match="match their tile"):
        runs.stain.cell_bytes(run)


def test_brightness_count_is_the_union_across_channels(runs, monkeypatch):
    run = start(runs)
    wait_for(lambda: ended(run))
    ex = runs.stain.exclusions
    values = {"DAPI": [0, 1, 2, 3, 4, 5], "Boundary": [0, 0, 0, 5, 5, 5]}
    monkeypatch.setattr(ex, "file_brightness", lambda run, file, channel: np.asarray(values[channel], np.float32))
    ex.change(run, dict(version=0, channel="DAPI", min=2, max=None), brightness=True)
    state = ex.change(run, dict(version=1, channel="Boundary", min=1, max=None), brightness=True)
    assert state["by_brightness"] == state["removed"] == 3 and state["kept"] == 3
    state = ex.change(run, dict(version=2, channel="DAPI", min=None, max=None), brightness=True)
    assert state["brightness"] == {"Boundary": dict(min=1, max=None)} and state["kept"] == 3


def test_cells_refuse_old_set_and_measurement_that_crosses_an_area_edit(runs, monkeypatch):
    run = start(runs)
    wait_for(lambda: ended(run))
    ex = runs.stain.exclusions
    with pytest.raises(RunError, match="cells changed"):
        runs.stain.cell_bytes(run, "old")
    with pytest.raises(RunError, match="no channel"):
        runs.stain.cell_bytes(run, channel="missing")
    def changed(*args):
        ex.change(run, dict(version=0, points=[[9, 9], [13, 9], [13, 13], [9, 13]]))
        return np.ones(6, np.float32)
    monkeypatch.setattr(ex, "file_brightness", changed)
    with pytest.raises(RunError, match="cells changed"):
        runs.stain.cell_bytes(run, channel="DAPI")


def test_failed_brightness_read_during_tiles_keeps_run_running_and_reports_chooser_error(runs, monkeypatch):
    run = start(runs, resume=False)
    ex = runs.stain.exclusions
    ex.path(run).write_text(json.dumps(dict(version=1, areas=[], brightness={"DAPI": dict(min=20, max=None)})))
    def failed(*args):
        raise OSError("test image unavailable")
    monkeypatch.setattr(ex, "file_brightness", failed)
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: run.meta["progress"]["done"] >= 1)
    assert run.meta["state"] in ("running", "done")
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done" and run.meta["summary"]["cells"] == 6
    assert "test image unavailable" in run.meta["exclusions"]["error"]
    values = vectors.decode_cells(runs.stain.cell_bytes(run, channel="DAPI"))
    assert np.isnan(values["brightness"]["DAPI"]).all() and values["error"]


def test_unexpected_refresh_failure_never_fails_a_completed_segmentation(runs, monkeypatch):
    def failed(*args):
        raise ValueError("test refresh failure")
    monkeypatch.setattr(runs.stain.exclusions, "refresh", failed)
    run = start(runs)
    wait_for(lambda: ended(run))
    assert run.meta["state"] == "done" and run.meta["summary"]["cells"] == 6
    assert run.meta["exclusions"]["error"]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7])
def test_plcv_roundtrip_alignment_and_strict_length(n):
    blob = vectors.encode_cells(np.arange(n), np.arange(n) % 2,
                               {"DAPI": np.full(n, np.nan), "Boundary": np.arange(n)},
                               {"set": "r:resolved:1:2", "areas_version": 7})
    result = vectors.decode_cells(blob)
    assert result["ids"].tolist() == list(range(1, n + 1)) and result["area"].tolist() == list(range(n))
    assert result["set"] == "r:resolved:1:2" and result["areas_version"] == 7
    assert np.isnan(result["brightness"]["DAPI"]).all()
    for data in (blob[:-1], blob + b"\0", b"wrong" + blob[5:]):
        with pytest.raises(ValueError):
            vectors.decode_cells(data)


def test_unknown_image_pixels_do_not_poison_another_cells_brightness():
    from sopa_companion.engine.stain_intensity import brightness
    plane = np.array([[np.nan, 4]], np.float32)
    coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]
    values = brightness(lambda x0, y0, x1, y1: plane[y0:y1, x0:x1], (0, 0, 1, 1), [0, 5, 10], coords, (1, 2))
    assert np.isnan(values[0]) and values[1] == 4
