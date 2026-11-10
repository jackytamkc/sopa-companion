"""Saving a run into its sample: the names a save is offered are the sample's as they are now, a name
clash says which names are taken (409 {error, clash, existing}), a stain run of part of the sample is never made the
sample's segmentation by default, and a stain run with no cell left is refused in plain words.
Leaving and closing: a leave saves under new names and removes the run, a Finish & save is
decided at close, a closing answer binds the runs it was given for, a failed close save is kept for the next start.
The sample is a double: its element names change as saves write them; nothing is written."""
import json
from pathlib import Path
import re
import sys
import threading
import hashlib

import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from test_stain import SETTINGS, Images, ended
from sopa_companion.engine import runs as R
from sopa_companion.engine.runs import NO_CELLS, Run, RunError, Runs, leave_kind
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples

BASE = "http://127.0.0.1:8799"
HEADERS = {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}


class Sample:
    """A SpatialData sample as Runs asks it (sid "zs"): its element names, which a save adds to, and its saves (kept
    here, never written). The names a save is offered are worked out by the engine's own rule (ZarrSamples)."""
    save_defaults = ZarrSamples.save_defaults

    def __init__(self, path):
        self.path, self.existing, self.saved = str(path), ["cell_boundaries", "nucleus_boundaries", "transcripts"], []

    def known(self, sid):
        return sid == "zs"

    def open(self, path, recent=True):
        return {"id": "zs"}

    def existing_elements(self, sid):
        return sorted(self.existing)

    def save(self, sid, output, shapes_key, table_key, set_default, overwrite, provenance, **kwargs):
        self.saved.append({"shapes_key": shapes_key, "table_key": table_key, "set_default": set_default,
                           "overwrite": overwrite, "provenance": provenance, **kwargs})
        self.existing += [key for key in (shapes_key, table_key) if key and key not in self.existing]
        return {"cells": 6}


def ended_run(config, run_id, kind="proseg", **meta):
    """A run's folder as an engine leaves it: a proseg run done, or a stain run with its tiles joined, on the sample
    "zs", holding the result a save reads. Made before an engine starts, it is from the last session (sorted at start,
    with the marker a leftover, without it an older run); adopt() makes it one of this session's."""
    folder = Path(config["paths"]["data_dir"]) / "runs" / run_id
    (folder / "output").mkdir(parents=True)
    value = {"id": run_id, "kind": kind, "sample": "zs", "sample_name": "Sample", "sample_kind": "zarr",
             "sample_path": "/data/sample.zarr", "state": "done", "created": f"2026-01-01T00:00:0{len(run_id) % 10}",
             "region": {"kind": "full", "bbox": [0, 0, 400, 300]}, "zarr_save": None, "outputs": []}
    if kind == "stain":
        (folder / "output" / "boundaries.geojson").write_text('{"type": "FeatureCollection", "features": []}')
        (folder / "stain").mkdir()
        value.update(stain={**SETTINGS, "method": "cellpose", "model": "cpsam"}, summary={"cells": 6},
                     progress={"done": 6, "total": 6, "revision": 7, "resolved": True, "complete": True})
    else:
        (folder / "output" / "proseg-output.zarr").mkdir()
    value.update(meta)
    (folder / "run.json").write_text(json.dumps(value))
    return run_id


def adopt(runs, run_id) -> Run:
    """A run of this engine session, from its folder (a user engine keeps no run of the last session in its
    registry: Runs sorts those at its start)."""
    folder = runs.root / run_id
    meta = json.loads((folder / "run.json").read_text())
    meta.setdefault("session_model", 2)
    run = Run(folder, meta)
    run.save()
    run.read_new()
    runs.runs[run_id] = run
    return run


def session_run(runs, config, run_id, kind="proseg", **meta) -> Run:
    return adopt(runs, ended_run(config, run_id, kind, **meta))


@pytest.fixture
def sample(tmp_path):
    return Sample(tmp_path / "sample.zarr")


def test_a_save_into_the_sample_renews_every_other_runs_names(config, sample):
    """A run that ended before another run's save still offered the names that save took, and its
    dialog had no Replace. Once a save commits, every run of the sample that can be saved offers free names again, and
    lists the names the sample has now."""
    runs = Runs(config, Samples(config), sample)
    for run_id in ("first", "second"):
        session_run(runs, config, run_id, save_defaults={"shapes_key": "proseg_boundaries", "table_key": "proseg_table",
                                                         "existing": ["cell_boundaries"]})
    try:
        first, second = runs.get("first"), runs.get("second")
        runs.save_to_zarr("first", "proseg_boundaries", "proseg_table")
        wait_for(lambda: (first.meta["zarr_save"] or {}).get("state") == "saved")
        offered = second.meta["save_defaults"]
        assert (offered["shapes_key"], offered["table_key"]) == ("proseg_boundaries_2", "proseg_table_2")
        assert {"proseg_boundaries", "proseg_table"} <= set(offered["existing"]) and offered["set_default"] is True
        assert json.loads((second.dir / "run.json").read_text())["save_defaults"] == offered   # kept for a restart
        assert first.meta["save_defaults"]["shapes_key"] == "proseg_boundaries_2"
    finally:
        runs.close(save=False)


def test_a_name_clash_says_which_names_are_taken_and_every_name_in_the_sample(config, sample):
    """409 {detail, error, clash, existing} on a clash (the dialog offers Replace from it); the run's names are worked out
    again with it. The case-only twin refusal (422) lists the names too. Replacing, asked for, saves."""
    from sopa_companion.server import create_app
    sample.existing += ["proseg_boundaries", "proseg_table"]                 # another run's save, since this one ended
    application = create_app(config, token="t" * 40)
    application.state.runs.zarr = sample
    session_run(application.state.runs, config, "late",
                save_defaults={"shapes_key": "proseg_boundaries", "table_key": "proseg_table", "existing": []})
    with TestClient(application, base_url=BASE, headers=HEADERS) as client:
        clash = client.post("/api/v1/runs/late/zarr", json={"shapes_key": "proseg_boundaries", "table_key": "proseg_table"})
        assert clash.status_code == 409
        body = clash.json()
        assert body["detail"] == body["error"] == ("proseg_boundaries and proseg_table already exist in the sample: "
                                                   "choose other names or allow replacing")
        assert body["clash"] == ["proseg_boundaries", "proseg_table"] and body["existing"] == sorted(sample.existing)
        renewed = application.state.runs.get("late").meta["save_defaults"]
        assert renewed["shapes_key"] == "proseg_boundaries_2" and "proseg_table" in renewed["existing"]
        twin = client.post("/api/v1/runs/late/zarr", json={"shapes_key": "Proseg_Boundaries", "table_key": "t2"})
        assert twin.status_code == 422 and "only by case" in twin.json()["detail"]
        assert twin.json()["existing"] == sorted(sample.existing)
        replaced = client.post("/api/v1/runs/late/zarr", json={"shapes_key": "proseg_boundaries",
                                                               "table_key": "proseg_table", "overwrite": True})
        assert replaced.status_code == 200
        wait_for(lambda: (application.state.runs.get("late").meta["zarr_save"] or {}).get("state") == "saved")
        assert sample.saved[-1]["overwrite"] is True


def test_case_twins_are_never_offered_and_the_offered_names_save(config, sample):
    """A sample with Proseg_Boundaries and PROSEG_TABLE was offered proseg_boundaries and
    proseg_table, which the save then refused (422, case twins), Replace or not. The next free names ignoring case are
    offered, and a save with them goes through."""
    from sopa_companion.server import create_app
    sample.existing = ["Proseg_Boundaries", "PROSEG_TABLE", "transcripts"]
    application = create_app(config, token="t" * 40)
    application.state.runs.zarr = sample
    session_run(application.state.runs, config, "twins")
    with TestClient(application, base_url=BASE, headers=HEADERS) as client:
        offered = client.get("/api/v1/runs/twins").json()["save_defaults"]
        assert (offered["shapes_key"], offered["table_key"]) == ("proseg_boundaries_2", "proseg_table_2")
        saved = client.post("/api/v1/runs/twins/zarr", json={"shapes_key": offered["shapes_key"],
                                                              "table_key": offered["table_key"]})
        assert saved.status_code == 200
        wait_for(lambda: (application.state.runs.get("twins").meta["zarr_save"] or {}).get("state") == "saved")
    assert ZarrSamples.save_defaults(sample, "zs", "stain", also=["CELLPOSE_boundaries_1"])["shapes_key"] == \
        "cellpose_boundaries_2"


def test_the_dialog_asks_for_the_names_as_the_sample_has_them_now(config, sample):
    """GET /runs/{id} (the Save dialog asks it each time it opens): a run that can be saved has its save defaults
    worked out against the sample's names now, not as they were when it ended; kept in the run (and pushed) only when
    they changed. A run that cannot be saved (a stain run whose tiles are not joined yet) keeps what it has."""
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    application.state.runs.zarr = sample
    runs = application.state.runs
    session_run(runs, config, "done-run", save_defaults={"shapes_key": "proseg_boundaries", "table_key": "proseg_table",
                                                         "existing": []})
    session_run(runs, config, "going", kind="stain", state="stopped", save_defaults=None,
                progress={"done": 3, "total": 6, "revision": 3, "resolved": False, "complete": False})
    with TestClient(application, base_url=BASE, headers=HEADERS) as client:
        sample.existing += ["proseg_boundaries", "proseg_table"]             # saved by another app meanwhile
        offered = client.get("/api/v1/runs/done-run").json()["save_defaults"]
        assert offered["shapes_key"] == "proseg_boundaries_2" and "proseg_boundaries" in offered["existing"]
        version = runs.get("done-run").version
        assert client.get("/api/v1/runs/done-run").json()["save_defaults"] == offered
        assert runs.get("done-run").version == version    # unchanged: nothing written or pushed
        assert client.get("/api/v1/runs/going").json()["save_defaults"] is None


@pytest.mark.parametrize("region,progress,whole", [
    ({"kind": "full"}, {"complete": True}, True),
    ({"kind": "box", "box": [0, 0, 100, 100]}, {"complete": True}, False),
    ({"kind": "polygon", "points": [[0, 0], [100, 0], [0, 100]]}, {"complete": True}, False),
    ({"kind": "full"}, {"complete": False}, False),                       # stopped part-way: some tiles undone
])
def test_a_stain_save_of_part_of_the_sample_is_never_its_segmentation_by_default(config, sample, region, progress, whole, monkeypatch):
    """A 100 µm Cellpose box was saved as "the sample's segmentation" (the dialog ticked it for every save),
    and sopa's later steps would take it as the sample's cells. Only a stain run over the whole sample, every tile done,
    starts ticked; a save that does not say uses the run's default."""
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "stain", kind="stain", state="done" if progress["complete"] else "stopped", region=region,
                progress={"done": 6, "total": 6, "revision": 7, "resolved": True, **progress})
    session_run(runs, config, "proseg")
    try:
        monkeypatch.setattr(runs, "_stain_keep", lambda run, state: run.dir / "stain" / "keep.npy")
        assert runs.current_save_defaults("stain")["set_default"] is whole
        assert runs.current_save_defaults("proseg")["set_default"] is True    # a proseg run: as before
        runs.save_to_zarr("stain", "cellpose_boundaries_1", None)            # set_default not given
        wait_for(lambda: (runs.get("stain").meta["zarr_save"] or {}).get("state") == "saved")
        assert sample.saved[-1]["set_default"] is whole and runs.get("stain").meta["zarr_save"]["set_default"] is whole
        runs.save_to_zarr("stain", "cellpose_boundaries_2", None, set_default=True)   # the user's tick wins
        wait_for(lambda: runs.get("stain").meta["zarr_save"]["shapes_key"] == "cellpose_boundaries_2" and
                 runs.get("stain").meta["zarr_save"]["state"] == "saved")
        assert sample.saved[-1]["set_default"] is True
    finally:
        runs.close(save=False)


def test_a_stain_save_takes_its_filters_digest_from_the_frozen_state(config, sample, monkeypatch):
    """A hang: a stain save worked out its digest by counting every cell of the run, which
    raised for a run whose outline files could not be read, after the run was marked "saving": closing then waited an
    hour for a save that never started. The digest is the frozen state's own (stain_rules.digest), and nothing that
    fails before the save starts leaves the run saving."""
    from sopa_companion.engine import stain_rules
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "digest", kind="stain")
    state = {"version": 3, "areas_version": 1, "next": 2, "areas": [],
             "size": {"min": 5, "max": 40}, "brightness": {}}
    monkeypatch.setattr(runs.stain.exclusions, "load", lambda run: state)
    monkeypatch.setattr(runs.stain.exclusions, "public", lambda run: pytest.fail("every cell counted to save"))
    monkeypatch.setattr(runs, "_stain_keep", lambda run, state: run.dir / "stain" / "keep.npy")
    runs.save_to_zarr("digest", "cellpose_boundaries_1", None)
    wait_for(lambda: (run.meta["zarr_save"] or {}).get("state") == "saved")
    assert run.meta["zarr_save"]["exclusions_digest"] == stain_rules.digest(state)
    assert run.meta["zarr_save"]["exclusions_version"] == 3
    failing = session_run(runs, config, "unwritable", kind="stain")
    original = Run.save
    def save(self):
        if self is failing:
            raise OSError("No space left on device")
        original(self)
    monkeypatch.setattr(Run, "save", save)
    with pytest.raises(OSError):
        runs.save_to_zarr("unwritable", "cellpose_boundaries_2", None)
    assert failing.meta["zarr_save"] is None and runs.pending()[0]["decided"] is False
    monkeypatch.setattr(Run, "save", original)
    closer = threading.Thread(target=runs.close, args=("discard",))
    closer.start()
    closer.join(20)
    assert not closer.is_alive() and not runs.runs


def test_a_stain_run_ending_on_a_box_offers_its_save_unticked(config, tmp_path):
    """The names a stain run offers as it ends (engine/stain.py) carry the same default: unticked for a box."""
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", tile_pixels=128,
                           overlap_pixels=16)
    config["proseg"]["keep_unsaved"] = True
    runs = Runs(config, Samples(config), Images(tmp_path / "unwritten.zarr"))
    try:
        found = {}
        for name, region in (("box", {"kind": "box", "box": [0, 0, 200, 150]}), ("full", {"kind": "full"})):
            run = runs.get(runs.create("image-sample", kind="stain", region=region, stain=SETTINGS)["id"])
            runs.control(run.meta["id"], "resume")
            wait_for(lambda: ended(run))
            found[name] = run.meta["save_defaults"]["set_default"]
        assert found == {"box": False, "full": True}
    finally:
        runs.close(save=False)


def test_a_stain_run_with_no_cell_left_is_not_saved(config, sample):
    """A run that found no cell (or whose cells were all removed) was accepted, then the helper failed with a
    raw IndexError. It is refused before anything starts, in plain words."""
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "empty", kind="stain", summary={"cells": 0})
    try:
        with pytest.raises(RunError) as refused:
            runs.save_to_zarr("empty", "cellpose_boundaries_1", None)
        assert refused.value.status == 409 and str(refused.value) == NO_CELLS
        assert NO_CELLS == "There are no cells to save: this run found none, or all were removed"
        assert sample.saved == [] and runs.get("empty").meta["zarr_save"] is None
    finally:
        runs.close(save=False)


@pytest.mark.parametrize("region,whole", [({"kind": "full"}, True), ({"kind": "box"}, False),
                                           ({"kind": "polygon"}, False)])
def test_proseg_uses_the_same_full_region_default(config, sample, region, whole):
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "finished", region=region)
    assert runs.current_save_defaults("finished")["set_default"] is whole
    runs.leave("finished", "add")
    wait_for(lambda: "finished" not in runs.runs)
    assert sample.saved[-1]["set_default"] is whole
    assert sample.saved[-1]["overwrite"] is False
    assert runs.gone == {"finished": "left"}


@pytest.mark.parametrize("state", ["running", "paused", "finishing", "done"])
def test_leave_finishes_then_saves_and_removes(config, sample, monkeypatch, state):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "leaving")
    run.meta["state"] = state
    run.frames = [{"iter": 8, "seq": 1}]
    actions = []
    def control(rid, action):
        actions.append(action)
        runs._set(run, state="done")
    monkeypatch.setattr(runs, "control", control)
    if state == "finishing":
        worker = threading.Thread(target=lambda: runs._set(run, state="done"))
        run.worker = worker
        worker.start()
    assert runs.leave("leaving", "add")["leaving"]["answer"] == "add"
    # The run leaves the list first, then its folder goes (Runs._delete_run): a slow computer shows the moment between
    wait_for(lambda: "leaving" not in runs.runs and not run.dir.exists())
    assert actions == (["finish"] if state in ("running", "paused") else [])
    assert len(sample.saved) == 1


def test_leave_retries_a_name_clash_once_with_new_names(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "clash")
    save = sample.save
    calls = []
    def clash(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 1:
            sample.existing += [args[2], args[3]]
            raise RunError("Names taken", 409, clash=[args[2]])
        return save(*args, **kwargs)
    monkeypatch.setattr(sample, "save", clash)
    runs.leave("clash", "add")
    wait_for(lambda: "clash" not in runs.runs)
    assert calls == ["proseg_boundaries", "proseg_boundaries_2"]
    assert sample.saved[0]["overwrite"] is False


def test_stay_during_save_keeps_the_run_and_another_answer_is_refused(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "stay")
    entered, release = threading.Event(), threading.Event()
    save = sample.save
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return save(*args, **kwargs)
    monkeypatch.setattr(sample, "save", slow)
    runs.leave("stay", "add")
    assert entered.wait(5)
    assert runs.pending()[0]["decided"] is True
    with pytest.raises(RunError, match="being saved already") as refused:
        runs.leave("stay", "discard")
    assert refused.value.status == 409
    assert runs.leave("stay", "stay")["leaving"] is None
    release.set()
    run.leave_worker.join(5)
    assert run.meta["zarr_save"]["state"] == "saved" and runs.get("stay") is run
    runs.leave("stay", "discard")
    wait_for(lambda: not runs.runs)
    assert len(sample.saved) == 1


def test_an_answer_older_than_one_taken_from_its_window_is_ignored(config, sample):
    """A Discard still on its way when Stay was pressed landed after it and deleted the run the window
    stays on. Each window numbers its answers (revision): an older one than the last taken from it changes nothing;
    another window's answers are its own."""
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "ordered")
    assert runs.leave("ordered", "stay", revision=5, page="a")["leaving"] is None
    assert runs.leave("ordered", "discard", revision=4, page="a")["leaving"] is None     # late: nothing done
    assert runs.get("ordered") is run and run.leave_worker is None
    with pytest.raises(RunError):
        runs.leave("ordered", "discard", revision=True, page="a")
    runs.leave("ordered", "discard", revision=1, page="b")                             # another window's
    wait_for(lambda: "ordered" not in runs.runs)
    assert sample.saved == []


def test_discard_waits_for_a_dialog_save_and_patch_export(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "wait")
    entered, release = threading.Event(), threading.Event()
    save = sample.save
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return save(*args, **kwargs)
    monkeypatch.setattr(sample, "save", slow)
    runs.save_to_zarr("wait", "shapes", "table")
    assert entered.wait(5)
    runs.stain.patches.exporting.add("wait")
    runs.leave("wait", "discard")
    release.set()
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    assert runs.get("wait") is run
    runs.stain.patches.exporting.remove("wait")
    wait_for(lambda: "wait" not in runs.runs)
    assert len(sample.saved) == 1


def test_a_failed_leave_keeps_the_run_and_a_failed_close_save_is_removed_at_the_next_start(config, sample, monkeypatch):
    """A save that fails while the app closes: the run is removed with the other leftovers at the next
    start, whose notice says the save failed and why (never reopened; no close_add_failed)."""
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "failed")
    def fail(*args, **kwargs):
        raise OSError("Sample is read-only")
    monkeypatch.setattr(sample, "save", fail)
    runs.leave("failed", "add")
    run.leave_worker.join(5)
    assert runs.get("failed") is run and run.meta["leaving"] is None
    assert run.meta["zarr_save"]["state"] == "failed" and "read-only" in run.meta["zarr_save"]["message"]
    assert run.meta["leave_error"]["answer"] == "add" and "read-only" in run.meta["leave_error"]["message"]
    runs.close("add", finish_limit=1)
    assert not runs.runs and "close_add_failed" not in run.meta
    kept = json.loads((run.dir / "run.json").read_text())
    assert kept["closed"] == "save failed" and "read-only" in kept["zarr_save"]["message"]
    again = Runs(config, Samples(config), sample)
    assert not run.dir.exists() and not again.runs and again.older()["runs"] == []
    notice = again.leftovers["runs"]
    assert len(notice) == 1 and notice[0]["sample_name"] == "Sample" and "read-only" in notice[0]["failed"]
    again.close()


def test_a_failed_discard_or_a_failure_after_the_save_keeps_the_save_record(config, sample, monkeypatch):
    """Any failed leave wrote zarr_save "failed", a saved run's included (a discard whose run would
    not stop, an add whose save went through before removing the run failed): the run then asked to be saved again, a
    second copy. Only an add that never reached the sample is a failed save; every failure is said (leave_error), and
    goes once another answer starts."""
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "saved-run", zarr_save={"state": "saved", "shapes_key": "proseg_boundaries",
                                                            "table_key": "proseg_table"})
    stop, stuck = runs._stop, {"on": True}
    def stopping(run, timeout=30):
        if stuck["on"]:
            raise RunError("The run has not stopped yet.", 409)
        return stop(run, timeout)
    monkeypatch.setattr(runs, "_stop", stopping)
    runs.leave("saved-run", "discard")
    run.leave_worker.join(5)
    assert run.meta["zarr_save"]["state"] == "saved" and leave_kind(run.public())["kind"] == "saved"
    assert run.meta["leaving"] is None and run.meta["leave_error"]["answer"] == "discard"
    assert run.meta["leave_error"]["message"] == "The run has not stopped yet." and runs.get("saved-run") is run
    # Each failure is told from the one before by its own id, however close together they came (two within the same
    # millisecond share `at`): a page reading the earlier one late never takes it for the leave it just asked for
    # (main.js updateLeaving).
    first = run.meta["leave_error"]["id"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", run.meta["leave_error"]["at"])
    runs.leave("saved-run", "discard")
    run.leave_worker.join(5)
    assert run.meta["leave_error"]["message"] == "The run has not stopped yet." and run.meta["leave_error"]["id"] != first
    added = session_run(runs, config, "added")
    runs.leave("added", "add")
    added.leave_worker.join(5)
    assert [s["provenance"]["run"] for s in sample.saved] == ["added"] and runs.get("added") is added
    assert added.meta["zarr_save"]["state"] == "saved" and added.meta["leave_error"]["answer"] == "add"
    assert runs.leave("added", "stay")["leave_error"] is None
    run.meta["leave_error"] = {"answer": "discard", "message": "earlier"}
    runs.save_to_zarr("saved-run", "proseg_boundaries_9", "proseg_table_9")     # a save starts: that failure is past
    assert run.meta["leave_error"] is None
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    stuck["on"] = False
    assert runs.leave("saved-run", "discard")["leave_error"] is None
    runs.leave("added", "discard")
    wait_for(lambda: not runs.runs)
    assert [s["shapes_key"] for s in sample.saved] == ["proseg_boundaries", "proseg_boundaries_9"]


def test_another_answer_while_a_discard_is_under_way_says_it_is_being_discarded(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "going")
    entered, release = threading.Event(), threading.Event()
    def slow(run, timeout=30):
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(runs, "_stop", slow)
    runs.leave("going", "discard")
    assert entered.wait(5)
    with pytest.raises(RunError, match="^It is being discarded already.$") as refused:
        runs.leave("going", "add")
    assert refused.value.status == 409
    release.set()
    wait_for(lambda: not runs.runs)
    assert sample.saved == [] and not run.dir.exists()


def test_closing_with_add_discards_a_run_with_nothing_to_save(config, sample):
    """Closing with "add", a stain run whose filters leave no cells (nothing to save) was kept on
    disk as a failed close save, and the next start said its save had failed. It is discarded, as its leave offers
    (Stay or Discard): nothing is kept and nothing said; the other runs are saved."""
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "filtered", kind="stain", summary={"cells": 0}, summary_model={"cells": 4})
    session_run(runs, config, "with-cells")
    runs.close("add", finish_limit=1)
    assert not runs.runs and not (runs.root / "filtered").exists()
    assert [s["provenance"]["run"] for s in sample.saved] == ["with-cells"]
    again = Runs(config, Samples(config), sample)
    assert again.leftovers is None and again.older()["runs"] == []
    again.close()


@pytest.mark.parametrize("policy", [None, "discard", "add"])
def test_test_engine_close_never_changes_sample_bytes(config, sample, tmp_path, policy, monkeypatch):
    config["proseg"]["keep_unsaved"] = True
    sample_path = tmp_path / "sample.zarr"
    sample_path.mkdir()
    (sample_path / "zarr.json").write_text('{"zarr_format":3,"node_type":"group","attributes":{}}')
    (sample_path / "cells").write_bytes(bytes(range(256)))
    digest = lambda: {str(p.relative_to(sample_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sample_path.rglob("*") if p.is_file()}
    before = digest()
    ended_run(config, "test-engine", sample_path=str(sample_path))
    runs = Runs(config, Samples(config), sample)
    monkeypatch.setattr(sample, "save", lambda *a, **k: pytest.fail("A test engine wrote into a sample at close"))
    runs.close(policy)
    assert digest() == before and runs.get("test-engine").dir.exists()


def test_close_without_an_answer_discards_finished_unsaved_results(config, sample):
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "pending")
    assert runs.pending()[0]["can_add"] is True and runs.pending()[0]["decided"] is False
    runs.close()
    assert runs.runs == {} and sample.saved == []


def test_a_closing_answer_binds_the_runs_it_was_given_for(config, sample):
    """One window's answer was carried out on every run, a run started afterwards in another window
    (never shown to anyone) included. Close takes {run id: answer}: the runs it was given for get it, any other run
    nobody answered for is discarded."""
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "shown")
    session_run(runs, config, "started-after")
    runs.close({"shown": "add"})
    assert [s["provenance"]["run"] for s in sample.saved] == ["shown"]
    assert not runs.runs
    with pytest.raises(ValueError):
        Runs._answers({"shown": "keep"})
    assert Runs._answers({"a": "save"})("a") == "add" and Runs._answers(None)("x") == "discard"


def test_a_finish_and_save_is_decided_at_close_until_its_dialog_is_cancelled(config, sample):
    """A run whose Finish & save was pressed counts as answered: the app closing with no answer (or Discard) saves it
    under new names. Cancel in its Save dialog (leave stay) takes that back: then it is discarded."""
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "finished", finish_requested=True)
    assert runs.pending()[0]["decided"] is True
    runs.close("discard")
    assert [s["provenance"]["run"] for s in sample.saved] == ["finished"] and not runs.runs
    sample.saved.clear()
    again = Runs(config, Samples(config), sample)
    session_run(again, config, "cancelled", finish_requested=True)
    assert again.leave("cancelled", "stay")["save_cancelled"] is True
    assert again.pending()[0]["decided"] is False
    again.close()
    assert sample.saved == [] and not again.runs


def test_a_save_still_running_after_the_hour_cap_keeps_its_folder(config, sample, monkeypatch):
    """A save into the sample is waited for (an hour); one still running then is never deleted under it: its folder
    stays with `closed: "save running"` for the next start (which leaves it while a save still uses it)."""
    monkeypatch.setattr(R, "SAVE_WAIT_S", 0.3)
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "slow")
    entered, release = threading.Event(), threading.Event()
    save = sample.save
    def slow(*args, **kwargs):
        entered.set()
        release.wait(10)
        return save(*args, **kwargs)
    monkeypatch.setattr(sample, "save", slow)
    runs.save_to_zarr("slow", "proseg_boundaries", "proseg_table")
    assert entered.wait(5)
    runs.close("discard", timeout=1, finish_limit=1)
    assert not runs.runs and run.dir.exists()
    assert json.loads((run.dir / "run.json").read_text())["closed"] == "save running"
    release.set()
    run.saver.join(5)
    assert json.loads((run.dir / "run.json").read_text())["zarr_save"]["state"] == "saved"


def test_stain_leave_stops_joins_and_saves_a_frozen_mask(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "stain-leave", kind="stain",
                      covered={"region": {"kind": "box"}, "done": 3, "of": 6, "why": "tiles", "pieces": 1})
    run.meta.update(state="running", summary_model={"cells": 6})
    run.meta["progress"].update(resolved=False, complete=False)
    run.meta["stain"]["resolution"] = {"level": 2, "factor": 4.004, "um_per_px": 0.851}
    actions = []
    def control(rid, action):
        actions.append(action)
        runs._set(run, state="stopped")
    def join(run):
        actions.append("join")
        run.meta["progress"]["resolved"] = True
    monkeypatch.setattr(runs, "control", control)
    monkeypatch.setattr(runs.stain, "join", join, raising=False)
    filters = {"version": 2, "areas_version": 0, "next": 1, "areas": [],
               "size": {"min": 5, "max": 40}, "brightness": {"0": {"min": None, "max": 100}}}
    monkeypatch.setattr(runs.stain.exclusions, "load", lambda run: filters)
    monkeypatch.setattr(runs.stain.exclusions, "digest", lambda state: "frozen")
    masks = []
    monkeypatch.setattr(runs, "_stain_keep", lambda run, state: (masks.append(state), run.dir / "stain" / "keep.npy")[1])
    runs.leave("stain-leave", "add")
    wait_for(lambda: "stain-leave" not in runs.runs)
    saved = sample.saved[0]
    assert actions == ["stop", "join"] and masks == [filters]
    assert saved["keep"].endswith("/stain/keep.npy") and saved["set_default"] is False
    assert saved["provenance"]["filters"] == filters
    assert saved["provenance"]["resolution"] == run.meta["stain"]["resolution"]
    assert saved["provenance"]["covered"] == run.meta["covered"] == run.meta["zarr_save"]["covered"]
    assert run.meta["zarr_save"]["exclusions_digest"] == "frozen"
    assert not {"pairing", "fits", "pair_with"} & saved.keys()


def test_keep_mask_uses_the_shared_stain_rules(config, sample, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "mask", kind="stain")
    exclusions = runs.stain.exclusions
    state = {"areas": [], "size": {"min": 10, "max": 20}, "brightness": {"0": {"min": 3, "max": None}}}
    monkeypatch.setattr(exclusions, "files", lambda run: [Path("resolved")])
    monkeypatch.setattr(exclusions, "file_cells", lambda file: (None, (np.zeros((3, 2)), np.array([9, 15, 21]))))
    monkeypatch.setattr(exclusions, "file_brightness", lambda run, file, channel: np.array([4, 5, np.nan], np.float32))
    calls = []
    def keep(areas, brightness, inside, rules):
        calls.append((areas, brightness, inside, rules))
        return np.array([False, True, False])
    rules = SimpleNamespace(normalise=lambda state: state, in_area=lambda centres, areas: np.zeros(3, bool), keep=keep)
    import sopa_companion.engine as engine
    monkeypatch.setitem(sys.modules, "sopa_companion.engine.stain_rules", rules)
    monkeypatch.setattr(engine, "stain_rules", rules)              # `from . import stain_rules` reads it from here
    path = runs._stain_keep(run, state)
    assert np.load(path).tolist() == [False, True, False]
    assert calls[0][0].dtype == np.float32 and calls[0][3] == state
    assert set(calls[0][1]) == {"0"}
    runs.close("discard")


def test_committed_ranges_keep_exactly_the_cells_saved_with_o2_rules(config, sample, monkeypatch):
    import numpy as np
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "shared-rules", kind="stain")
    rules = {"version": 1, "areas_version": 0, "next": 1, "areas": [],
             "size": {"min": 10, "max": 20}, "brightness": {"0": {"min": 3, "max": 8}}}
    exclusions = runs.stain.exclusions
    monkeypatch.setattr(exclusions, "files", lambda run: [Path("resolved")])
    monkeypatch.setattr(exclusions, "file_cells", lambda file: (None, (np.zeros((5, 2)), np.array([9, 10, 15, 20, 21]))))
    monkeypatch.setattr(exclusions, "file_brightness", lambda run, file, channel: np.array([5, 2, np.nan, 8, 5]))
    assert np.load(runs._stain_keep(run, rules)).tolist() == [False, False, True, True, False]
    runs.close("discard")


def test_a_brightness_range_is_measured_by_its_channels_name_when_saved(config, sample, monkeypatch):
    """A brightness rule names its channel (stain_rules.check takes the image's channel names). The save measured it by
    int(name), which failed for every real channel ("DAPI"): the rule was dropped as unreadable and dim cells were saved."""
    import numpy as np
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "by-name", kind="stain")
    rules = {"version": 1, "areas_version": 0, "next": 1, "areas": [], "size": None,
             "brightness": {"DAPI": {"min": 3, "max": None}}}
    exclusions = runs.stain.exclusions
    asked, noted = [], []
    monkeypatch.setattr(exclusions, "files", lambda run: [Path("resolved")])
    monkeypatch.setattr(exclusions, "file_cells", lambda file: (None, (np.zeros((4, 2)), np.array([9, 10, 15, 20]))))
    def brightness(run, file, channel):                 # as StainExclusions does: the job's channel list, by name
        asked.append(channel)
        ["DAPI", "ATP1A1"].index(channel)
        return np.array([5, 2, 1, 8], np.float32)
    monkeypatch.setattr(exclusions, "file_brightness", brightness)
    monkeypatch.setattr(runs, "note", lambda text, run_id=None: noted.append(text))
    assert np.load(runs._stain_keep(run, rules)).tolist() == [True, False, False, True]
    assert asked == ["DAPI"] and noted == []
    runs.close("discard")


def test_close_waits_for_a_decided_leave_even_after_a_discard_answer(config, sample, monkeypatch):
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "decided")
    entered, release = threading.Event(), threading.Event()
    original = sample.save
    def save(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(sample, "save", save)
    runs.leave("decided", "add")
    assert entered.wait(5)
    closer = threading.Thread(target=runs.close, args=("discard",))
    closer.start()
    assert closer.is_alive()
    release.set()
    closer.join(5)
    assert not closer.is_alive() and len(sample.saved) == 1 and not runs.runs


def test_saved_sample_detail_is_warmed_after_a_success(config, sample, monkeypatch):
    refreshed = threading.Event()
    monkeypatch.setattr(sample, "detail", lambda sid: refreshed.set(), raising=False)
    runs = Runs(config, Samples(config), sample)
    session_run(runs, config, "warm")
    runs.leave("warm", "add")
    assert refreshed.wait(5)
    wait_for(lambda: not runs.runs)


def test_stay_cancels_a_leave_waiting_to_delete(config, sample):
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "late-stay")
    with runs.creation_lock:
        runs.leave("late-stay", "add")
        wait_for(lambda: (run.meta.get("zarr_save") or {}).get("state") == "saved")
        assert runs.leave("late-stay", "stay")["leaving"] is None
    run.leave_worker.join(5)
    assert runs.get("late-stay") is run and run.dir.exists()
    runs.close("discard")


def test_a_saving_record_with_no_save_writing_it_is_never_waited_for(config, sample):
    """A run whose record says "saving" while no save of this engine writes it (a run.json written by hand, as
    test_stain's interrupted-run test does) held closing for SAVE_WAIT_S (an hour), and a delete refused it for good. A
    save only counts while its thread runs, or from the moment it is the run's until it starts."""
    import time
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "stale", zarr_save={"state": "saving", "message": "Writing…"})
    assert runs._save_open(run) is False
    run.saver = threading.Thread(target=lambda: None)              # made by _start_save, not started yet
    assert runs._save_open(run) is True
    run.saver = None
    started = time.monotonic()
    runs.close("discard", timeout=5, finish_limit=5)
    assert time.monotonic() - started < 10 and not runs.runs and sample.saved == []
    kept = json.loads((run.dir / "run.json").read_text())            # as an interrupted save: answer 3's leftover
    assert kept["closed"] == "save failed" and kept["zarr_save"]["message"].startswith("Interrupted")


def test_one_runs_failure_while_closing_never_keeps_the_others_from_closing(config, sample):
    """A run whose leave fails so badly that its failure cannot even be written (its folder gone under the engine)
    raised out of close(): the runs after it were never closed, and the lifespan's close raised."""
    import shutil
    runs = Runs(config, Samples(config), sample)
    gone = session_run(runs, config, "gone", state="paused", frames=3)
    other = session_run(runs, config, "other")
    shutil.rmtree(gone.dir)
    runs.close("discard", timeout=5, finish_limit=5)
    assert not runs.runs and not other.dir.exists() and sample.saved == []


def test_a_save_that_does_not_end_in_time_while_closing_is_left_for_the_next_start(config, sample, monkeypatch):
    """A run whose save is under way is decided: closing waits for it whatever the answer. One that does not end in
    time leaves the registry; its folder stays for the next start (`closed: "save failed"`), which removes it with
    the other leftovers and says why in its notice."""
    runs = Runs(config, Samples(config), sample)
    run = session_run(runs, config, "timeout")
    run.meta["zarr_save"] = {"state": "saving"}
    def timed_out(*args):
        raise RunError("The save has not finished yet.", 409)
    monkeypatch.setattr(runs, "_wait_save", timed_out)
    runs.close("discard", timeout=0.1, finish_limit=0.1)
    assert not runs.runs and runs.older()["runs"] == []
    assert json.loads((run.dir / "run.json").read_text())["closed"] == "save failed"
    again = Runs(config, Samples(config), sample)
    assert not run.dir.exists() and again.leftovers["runs"][0]["failed"] == "The save has not finished yet."
    again.close()
