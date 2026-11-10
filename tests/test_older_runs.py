"""Runs from before the one-run model at the first start: the app lists those that
hold a result never saved, apart from the registry (they never block a new run), and the user chooses for each, Save to
the sample or Discard, or for all at once, or Later. Saving one never makes it the sample's segmentation. Those with
nothing to save go at start and are counted. A folder a save of the last session may still use is left alone.
An older run is a run.json without the marker ("session_model": 2); the sample is a double (tests/test_runs_save.py)."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from conftest import wait_for
from test_runs_save import Sample, ended_run
from sopa_companion.engine import leftovers, runs as R
from sopa_companion.engine.runs import RunError, Runs
from sopa_companion.engine.samples import Samples


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "sample.zarr"
    folder.mkdir()
    (folder / "zarr.json").write_text('{"zarr_format":3,"node_type":"group","attributes":{}}')
    return Sample(folder)


def older(config, sample, run_id, kind="proseg", **meta):
    """An older run on the sample double's folder (it is there: the run can be saved)."""
    meta.setdefault("sample_path", sample.path)
    return ended_run(config, run_id, kind, **meta)


def unjoined(config, sample, run_id, **meta):
    """An older stain run with finished tiles not joined yet (its job and tile records are there)."""
    meta.setdefault("state", "stopped")
    older(config, sample, run_id, kind="stain",
          progress={"done": 3, "total": 6, "revision": 3, "resolved": False, "complete": False}, **meta)
    folder = Path(config["paths"]["data_dir"]) / "runs" / run_id
    (folder / "stain" / "job.json").write_text("{}")
    (folder / "stain" / "tiles.json").write_text("[]")
    (folder / "output" / "boundaries.geojson").unlink()
    return run_id


def rows(runs) -> dict:
    return {row["id"]: row for row in runs.older()["runs"]}


def folder(config, run_id) -> Path:
    return Path(config["paths"]["data_dir"]) / "runs" / run_id


def test_older_runs_are_listed_or_removed_and_counted_at_start(config, sample):
    listed = {
        "done": older(config, sample, "done"),
        "save-failed": older(config, sample, "save-failed", zarr_save={"state": "failed", "message": "disk full"}),
        "stain-joined": older(config, sample, "stain-joined", kind="stain", state="stopped"),
        "stain-unjoined": unjoined(config, sample, "stain-unjoined"),
        "stain-failed": unjoined(config, sample, "stain-failed", state="failed"),
        "stain-changed": older(config, sample, "stain-changed", kind="stain", zarr_save={"state": "saved", "result_revision": 5}),
        "patch-draft": older(config, sample, "patch-draft", kind="patches"),
    }
    (folder(config, "patch-draft") / "patches.json").write_text(json.dumps({"patches": [{"box": [0, 0, 1, 1]}], "set": None}))
    removed = [older(config, sample, "stopped", state="stopped"),
               older(config, sample, "no-tiles", kind="stain", state="stopped", progress={"done": 0, "total": 6}),
               older(config, sample, "saved", zarr_save={"state": "saved"}),
               older(config, sample, "patch-exported", kind="patches"),
               older(config, sample, "square", qc_test={"of": "done", "role": "test", "test": "t"}),
               older(config, sample, "interrupted", state="running")]
    (folder(config, "patch-exported") / "patches.json").write_text(json.dumps({"patches": [{"box": [0, 0, 1, 1]}],
                                                                               "set": "0" * 32}))
    marked = ended_run(config, "this-version", session_model=2, sample_path=sample.path)
    runs = Runs(config, Samples(config), sample)
    try:
        found = rows(runs)
        assert set(found) == set(listed) and not runs.runs
        assert runs.older()["count"] == len(listed) and runs.leftovers["older_removed"] == len(removed)
        assert all(found[i]["can_add"] for i in listed if i != "patch-draft")
        assert found["patch-draft"]["can_add"] is False and found["patch-draft"]["why"] == "patches"
        assert "copy them before you discard it" in found["patch-draft"]["detail"]
        assert found["stain-changed"]["why"] == "changed" and found["save-failed"]["zarr_save"]["state"] == "failed"
        assert all(not folder(config, i).exists() for i in removed) and not folder(config, marked).exists()
        assert [r["id"] for r in runs.leftovers["runs"]] == [marked]
    finally:
        runs.close()
    assert all(folder(config, i).exists() for i in listed)         # closing leaves the unanswered ones on disk


def test_a_test_engine_loads_every_run_and_lists_none(config, sample):
    config["proseg"]["keep_unsaved"] = True
    older(config, sample, "kept")
    runs = Runs(config, Samples(config), sample)
    assert set(runs.runs) == {"kept"} and runs.older() == {"runs": [], "done": [], "count": 0, "version": 0}
    runs.close()


def test_a_row_says_why_its_run_cannot_be_saved(config, sample):
    older(config, sample, "moved", sample_path="/no/such/sample-7.zarr", sample_name="sample-7")
    older(config, sample, "bundle", sample_kind="xenium")
    older(config, sample, "emptied")
    (folder(config, "emptied") / "output" / "proseg-output.zarr").rmdir()
    older(config, sample, "filtered", kind="stain", summary={"cells": 0}, summary_model={"cells": 4})
    older(config, sample, "ancient", sample_path=None)
    runs = Runs(config, Samples(config), sample)
    found = rows(runs)
    assert {i: (r["can_add"], r["why"]) for i, r in found.items()} == {
        "moved": (False, "sample_gone"), "bundle": (False, "xenium"), "emptied": (False, "missing"),
        "filtered": (False, "filtered"), "ancient": (False, "too_old")}
    assert found["moved"]["reason"] == "sample-7 is no longer where it was."
    assert found["moved"]["detail"] == ("It was at /no/such/sample-7.zarr. Put it back and start the app again to save this "
                                        "run.")
    assert found["bundle"]["detail"].endswith("/bundle/output: copy them before you discard it.")
    assert found["filtered"]["reason"] == "Its filters leave no cells, so there is nothing to save."
    for run_id in found:
        with pytest.raises(RunError) as refused:
            runs.answer_older(run_id, "add")
        assert refused.value.status == 409
    runs.close()


def test_two_runs_of_one_sample_and_region_are_told_apart(config, sample):
    older(config, sample, "first", created="2026-10-14T15:42:00", options={"samples": 2000})
    older(config, sample, "second", created="2026-10-14T16:05:00", options={})
    runs = Runs(config, Samples(config), sample)
    listed = runs.older()["runs"]
    assert [r["id"] for r in listed] == ["second", "first"]                       # newest first
    assert listed[0]["created"] != listed[1]["created"] and listed[1]["options"] == {"samples": 2000}
    assert listed[0]["region"] == listed[1]["region"] and listed[0]["method"] == "proseg"
    runs.close()


def test_save_writes_new_names_never_the_segmentation_and_removes_the_run(config, sample):
    """Add saves an inert run, one run at a time: next free names, never the default (a newer run or sopa may have set
    the sample's segmentation since), then the folder goes; Save all over two whole-sample runs ticks none."""
    for run_id in ("one", "two", "three"):
        older(config, sample, run_id)
    runs = Runs(config, Samples(config), sample)
    row = runs.answer_older("one", "add")
    assert row["answer"] == "add" and row["stage"] in ("waiting", "checking", "writing")
    wait_for(lambda: "one" not in rows(runs))
    assert sample.saved[-1]["set_default"] is False and sample.saved[-1]["shapes_key"] == "proseg_boundaries"
    assert not folder(config, "one").exists()
    assert runs.older()["done"][0]["message"] == "Saved as proseg_boundaries + proseg_table"
    runs.answer_all_older("add")
    wait_for(lambda: not rows(runs))
    assert [s["set_default"] for s in sample.saved] == [False, False, False]
    assert [s["shapes_key"] for s in sample.saved] == ["proseg_boundaries", "proseg_boundaries_2", "proseg_boundaries_3"]
    assert runs.older()["count"] == 0 and not runs.runs
    runs.close()


def test_an_unjoined_stain_run_is_joined_then_saved(config, sample, monkeypatch):
    unjoined(config, sample, "tiles", state="failed")
    runs = Runs(config, Samples(config), sample)
    steps = []
    def join(run):
        steps.append("join")
        (run.dir / "output" / "boundaries.geojson").write_text('{"type": "FeatureCollection", "features": []}')
        run.meta.update(state="done", progress={**run.meta["progress"], "resolved": True, "revision": 4})
    monkeypatch.setattr(runs.stain, "join", join)
    monkeypatch.setattr(runs, "_stain_keep", lambda run, state: (steps.append("keep"), run.dir / "stain" / "keep.npy")[1])
    runs.answer_older("tiles", "add")
    wait_for(lambda: "tiles" not in rows(runs))
    assert steps == ["join", "keep"] and sample.saved[-1]["shapes_key"] == "cellpose_boundaries_1"
    assert sample.saved[-1]["set_default"] is False and sample.saved[-1]["provenance"]["run"] == "tiles"
    runs.close()


def test_a_layer_made_from_the_run_means_saved_already(config, sample):
    """A save of the last session that finished before its run's folder went: the sample's own record names the run
    (a layer whose provenance `run` is its id), so the row says so and no second copy is written."""
    older(config, sample, "finished-saving", zarr_save={"state": "failed", "message": "Interrupted"})
    sample.saved_from = lambda sid, run_id: ["proseg_boundaries_4", "proseg_table_4"] if run_id == "finished-saving" else []
    runs = Runs(config, Samples(config), sample)
    runs.answer_older("finished-saving", "add")
    wait_for(lambda: not rows(runs))
    assert sample.saved == [] and not folder(config, "finished-saving").exists()
    done = runs.older()["done"][0]
    assert done["stage"] == "saved_already" and done["message"] == "Saved already as proseg_boundaries_4 + proseg_table_4"
    runs.close()


NO_FILTERS = {"version": 0, "areas_version": 0, "next": 1, "areas": [], "size": None, "brightness": {}}


@pytest.mark.parametrize("layer,again", [
    ({"result_revision": 5, "filters": NO_FILTERS}, True),                  # an earlier result: resumed since
    ({"result_revision": 7, "filters": {**NO_FILTERS, "size": {"min": 9.0, "max": None}}}, True),   # other filters
    ({"result_revision": 7, "filters": NO_FILTERS}, False),                 # this result: saved already
    ({"result_revision": 7}, False),                                        # by an earlier version, no filter then
    ({"result_revision": 7, "min_area_um2": 9.0, "removed_version": 2}, True),   # ... with a size rule taken away since
    ({"filters": NO_FILTERS}, True),                                        # no revision recorded: never taken as it
])
def test_a_stain_layer_counts_as_saved_only_when_it_holds_the_runs_result_now(config, sample, monkeypatch, layer, again):
    """An older stain run "changed since you saved it" (resumed for more tiles, filtered again)
    found its earlier layer in the sample and was removed as "Saved already": its newer cells were never written. A
    layer counts only when made at the run's revision and filters now (the layer's own record; a save by an earlier
    version has its areas and minimums under their older keys), else the run is saved under new names."""
    older(config, sample, "changed", kind="stain",
          zarr_save={"state": "saved", "shapes_key": "cellpose_boundaries_1", "result_revision": 5})
    sample.existing.append("cellpose_boundaries_1")
    sample.saved_from = lambda sid, run_id: ["cellpose_boundaries_1"] if run_id == "changed" else []
    sample.saved_provenance = lambda sid, key: {"run": "changed", **layer} if key == "cellpose_boundaries_1" else None
    runs = Runs(config, Samples(config), sample)
    monkeypatch.setattr(runs, "_stain_keep", lambda run, state: run.dir / "stain" / "keep.npy")
    try:
        assert rows(runs)["changed"]["why"] == "changed"
        runs.answer_older("changed", "add")
        wait_for(lambda: "changed" not in rows(runs))
        done = runs.older()["done"][0]
        if again:
            assert [s["shapes_key"] for s in sample.saved] == ["cellpose_boundaries_2"]
            assert sample.saved[0]["set_default"] is False and done["message"] == "Saved as cellpose_boundaries_2"
        else:
            assert sample.saved == [] and done["stage"] == "saved_already"
            assert done["message"] == "Saved already as cellpose_boundaries_1"
        assert not folder(config, "changed").exists()
    finally:
        runs.close()


def test_later_keeps_a_row_aside_once_the_last_sessions_save_frees_it(config, sample, monkeypatch):
    """A row still being saved by the last session when Later was pressed came back, not set aside,
    once that save ended (health.older counted it again within the session). It stays aside until the next start."""
    older(config, sample, "busy", zarr_save={"state": "saving", "message": "Writing…"})
    older(config, sample, "plain")
    busy = {"on": True}
    monkeypatch.setattr(leftovers, "named", lambda folders: {str(f) for f in folders if str(f).endswith("busy")}
                        if busy["on"] else set())
    runs = Runs(config, Samples(config), sample)
    try:
        assert rows(runs)["busy"]["still_saving"] is True
        assert runs.answer_all_older("later")["count"] == 0
        busy["on"] = False
        assert runs.recheck_still_saving() == 0
        row = rows(runs)["busy"]
        assert row["still_saving"] is False and row["can_add"] is True and row["later"] is True
        assert runs.older()["count"] == 0
    finally:
        runs.close()
    again = Runs(config, Samples(config), sample)
    assert set(rows(again)) == {"busy", "plain"} and again.older()["count"] == 2
    again.close()


@pytest.mark.parametrize("sign", ["held", "named", "locked"])
def test_a_folder_the_last_session_still_saves_is_left_alone_until_it_is_free(config, sample, monkeypatch, sign):
    """The check for a save of the last session still at work, each sign alone: a process holding a file in the
    folder, a process naming it on its command line (a save helper's arguments), the sample's save lock taken. The row
    has no buttons and the "interrupted" fix waits; once free it is sorted as at start. A folder this version made is
    left too, then removed."""
    run_id = older(config, sample, "busy", zarr_save={"state": "saving", "message": "Writing…"})
    marked = ended_run(config, "busy-marked", session_model=2, sample_path=sample.path,
                       zarr_save={"state": "saving"})
    busy = {"on": True}
    monkeypatch.setattr(leftovers, "holders", lambda folders: {str(f) + "/output/x" for f in folders}
                        if busy["on"] and sign == "held" else set())
    monkeypatch.setattr(leftovers, "named", lambda folders: {str(f) for f in folders}
                        if busy["on"] and sign == "named" else set())
    monkeypatch.setattr(R, "sample_locked", lambda path: busy["on"] and sign == "locked")
    runs = Runs(config, Samples(config), sample)
    try:
        row = rows(runs)["busy"]
        assert row["still_saving"] is True and row["can_add"] is False
        assert row["reason"] == "Still being saved by the last session"
        assert json.loads((folder(config, run_id) / "run.json").read_text())["zarr_save"]["state"] == "saving"
        assert folder(config, marked).exists() and runs.leftovers["runs"][0]["still_saving"] is True
        with pytest.raises(RunError) as refused:
            runs.answer_older("busy", "discard")
        assert refused.value.status == 409
        assert runs.recheck_still_saving() == 2                  # still in use: nothing moves
        busy["on"] = False
        assert runs.recheck_still_saving() == 0
        row = rows(runs)["busy"]
        assert row["still_saving"] is False and row["can_add"] is True
        assert row["zarr_save"]["state"] == "failed"              # the interrupted save, said now
        assert not folder(config, marked).exists()
    finally:
        runs.close()


def test_the_sample_save_lock_is_tried_without_blocking(tmp_path, monkeypatch):
    from sopa_companion.zarr_bridge.helper import _lock_paths
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(tmp_path / "locks"))
    store = tmp_path / "sample.zarr"
    store.mkdir()
    assert R.sample_locked(str(store)) is False and R.sample_locked(None) is False
    [path] = _lock_paths(store.resolve())
    with open(path, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        stamp = os.stat(path).st_mtime_ns
        assert R.sample_locked(str(store)) is True
        assert os.stat(path).st_mtime_ns == stamp                # tried read-only: its time stays
    assert R.sample_locked(str(store)) is False


def test_discard_removes_the_run_and_leaves_the_sample_byte_identical(config, sample):
    sample_folder = Path(sample.path)
    (sample_folder / "cells").write_bytes(bytes(range(256)))
    digest = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sample_folder.rglob("*") if p.is_file()}
    before = digest()
    older(config, sample, "unwanted")
    runs = Runs(config, Samples(config), sample)
    runs.answer_older("unwanted", "discard")
    wait_for(lambda: not rows(runs))
    assert digest() == before and sample.saved == [] and runs.older()["done"][0]["stage"] == "discarded"
    wait_for(lambda: not folder(config, "unwanted").exists() and not list((folder(config, "unwanted").parent / ".trash").iterdir()))
    runs.close()


def test_an_answer_for_no_run_in_particular_answers_every_row(config, sample):
    """answer_older(None, answer): every row, as answer_all_older (Discard all here), which answers older()."""
    for run_id in ("first", "second"):
        older(config, sample, run_id)
    runs = Runs(config, Samples(config), sample)
    listed = runs.answer_older(None, "discard")
    assert {r["id"] for r in listed["runs"]} <= {"first", "second"} and "count" in listed
    wait_for(lambda: not rows(runs))
    assert sorted(d["id"] for d in runs.older()["done"]) == ["first", "second"] and sample.saved == []
    runs.close()


def test_a_failing_save_keeps_the_row_and_the_next_start_lists_it_again(config, sample, monkeypatch):
    older(config, sample, "stubborn")
    runs = Runs(config, Samples(config), sample)
    save = sample.save
    def fail(*args, **kwargs):
        raise OSError("Sample is read-only")
    monkeypatch.setattr(sample, "save", fail)
    runs.answer_older("stubborn", "add")
    wait_for(lambda: rows(runs)["stubborn"]["stage"] == "failed")
    row = rows(runs)["stubborn"]
    assert row["message"].startswith("Couldn't save:") and "read-only" in row["message"]
    assert row["zarr_save"]["state"] == "failed" and row["can_add"] is True and folder(config, "stubborn").exists()
    runs.close()
    again = Runs(config, Samples(config), sample)
    assert rows(again)["stubborn"]["zarr_save"]["state"] == "failed"
    monkeypatch.setattr(sample, "save", save)
    again.answer_older("stubborn", "add")                         # Try again
    wait_for(lambda: not rows(again))
    assert len(sample.saved) == 1
    again.close()


def blocking(sample, monkeypatch):
    """The sample's save waits for the test's go (each save waits on its own go; `entered` says one is under way)."""
    entered, go = threading.Semaphore(0), threading.Semaphore(0)
    original = sample.save
    def save(*args, **kwargs):
        entered.release()
        assert go.acquire(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(sample, "save", save)
    return entered, go


def test_later_sets_the_rest_aside_while_the_save_under_way_finishes(config, sample, monkeypatch):
    older(config, sample, "a-older", created="2026-10-01T10:00:00")
    older(config, sample, "b-newer", created="2026-10-02T10:00:00")
    older(config, sample, "c-untouched", created="2026-09-01T10:00:00")
    runs = Runs(config, Samples(config), sample)
    entered, go = blocking(sample, monkeypatch)
    runs.answer_all_older("add")
    assert entered.acquire(timeout=5)                            # the newest first
    assert runs.older_current == "b-newer"
    listed = runs.answer_all_older("later")
    assert listed["count"] == 0
    queued = {r["id"]: r for r in listed["runs"]}
    assert queued["a-older"]["later"] is True and queued["a-older"]["answer"] is None
    assert queued["b-newer"]["answer"] == "add" and queued["b-newer"]["later"] is True
    go.release()
    wait_for(lambda: "b-newer" not in rows(runs) and runs.older_worker is None)
    assert [s["provenance"]["run"] for s in sample.saved] == ["b-newer"]
    assert folder(config, "a-older").exists() and folder(config, "c-untouched").exists()
    runs.close()
    again = Runs(config, Samples(config), sample)
    assert set(rows(again)) == {"a-older", "c-untouched"} and again.older()["count"] == 2
    again.close()


def test_new_runs_start_while_older_rows_are_left_or_being_saved(config, sample, monkeypatch):
    """Older runs are outside the registry: a new run is never refused for them (no 409), even while one is saved."""
    older(config, sample, "left-for-later")
    older(config, sample, "being-saved")
    runs = Runs(config, Samples(config), sample)
    monkeypatch.setattr(runs, "_execute", lambda run: None)
    bundle = runs.samples.list()[0]["id"]
    first = runs.create(bundle, [50, 50, 150, 150])
    assert first["session_model"] == 2
    runs.leave(first["id"], "discard")
    wait_for(lambda: not runs.runs)
    entered, go = blocking(sample, monkeypatch)
    runs.answer_older("being-saved", "add")
    assert entered.acquire(timeout=5)
    second = runs.create(bundle, [60, 60, 160, 160])
    assert second["id"] in runs.runs
    go.release()
    wait_for(lambda: "being-saved" not in rows(runs))
    runs.close("discard")


def test_closing_during_save_all_finishes_the_row_under_way_and_leaves_the_queued(config, sample, monkeypatch):
    older(config, sample, "first", created="2026-10-02T10:00:00")
    older(config, sample, "second", created="2026-10-01T10:00:00")
    runs = Runs(config, Samples(config), sample)
    entered, go = blocking(sample, monkeypatch)
    runs.answer_all_older("add")
    assert entered.acquire(timeout=5)
    closer = threading.Thread(target=runs.close)
    closer.start()
    wait_for(lambda: runs.closing)
    assert closer.is_alive()                                      # the row under way is waited for
    go.release()
    closer.join(10)
    assert not closer.is_alive()
    assert [s["provenance"]["run"] for s in sample.saved] == ["first"]
    assert not folder(config, "first").exists() and folder(config, "second").exists()
    assert "session_model" not in json.loads((folder(config, "second") / "run.json").read_text())
    again = Runs(config, Samples(config), sample)
    assert set(rows(again)) == {"second"}
    again.close()
