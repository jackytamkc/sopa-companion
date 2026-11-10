"""What deleted runs leave behind (engine/leftovers.py), here Compare's results: a
comparison still being scored when its run was deleted leaves compare/<job>/ files in a folder without run.json; they
are derived from the run, so the start-up sweep removes such a folder, as it does the builders' files. The sweep runs
here without an engine (`held`: nothing in use)."""
from __future__ import annotations

import json

from sopa_companion.engine import leftovers


def nothing_held(folders):
    return set()


def test_comparisons_left_by_a_deleted_run_go_with_its_folder(tmp_path):
    root = tmp_path / "runs"
    job = root / "deleted" / "compare" / "0123456789abcdef"
    job.mkdir(parents=True)
    for name in ("result.json", "cells-0.npy", "cells-2.npy", "points-1-0.bin", "points-2-3.bin",
                 ".result.json.4242", ".points-2-1.bin.4242", "points-2-1.bin.tmp"):
        (job / name).write_bytes(b"x")
    (root / "deleted" / "live").mkdir()
    (root / "deleted" / "live" / "transcripts.sorted.bin").write_bytes(b"x")   # a builder's, as before
    kept = root / "kept"
    (kept / "compare" / "fedcba9876543210").mkdir(parents=True)
    (kept / "run.json").write_text(json.dumps({"id": "kept", "state": "done"}))
    (kept / "compare" / "fedcba9876543210" / "result.json").write_text("{}")
    swept = leftovers.sweep(root, held=nothing_held)
    assert swept["folders"] == [str(root / "deleted")] and not swept["left"] and not (root / "deleted").exists()
    assert (kept / "compare" / "fedcba9876543210" / "result.json").exists()      # a run's own comparisons stay


def test_only_comparisons_where_compare_keeps_them_count_as_derived(tmp_path):
    root = tmp_path / "runs"
    for name, place, file in (("loose", "compare", "result.json"),                  # not in a comparison's folder
                              ("odd_job", "compare/not-a-job", "result.json"),
                              ("elsewhere", "live/0123456789abcdef", "result.json"),
                              ("other_file", "compare/0123456789abcdef", "notes.txt"),
                              ("other_points", "compare/0123456789abcdef", "points-x.bin")):
        (root / name / place).mkdir(parents=True)
        (root / name / place / file).write_bytes(b"x")
    swept = leftovers.sweep(root, held=nothing_held)
    assert not swept["folders"] and sorted(swept["left"]) == sorted(str(p) for p in root.iterdir())
    assert all((root / name).exists() for name in ("loose", "odd_job", "elsewhere", "other_file", "other_points"))



def test_start_quarantines_marked_runs_before_background_removal(config, monkeypatch):
    """Every folder this version made (the marker) is a leftover at the next start: moved into .trash/ and emptied in
    the background, noted once (health.leftovers) when it held something to lose; the notice goes once a run starts.
    A folder without the marker is an older run's, listed (Runs.older), never a leftover."""
    from pathlib import Path
    from test_runs_save import ended_run
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    pending = ended_run(config, "crashed", session_model=2)
    ended_run(config, "saved", session_model=2, zarr_save={"state": "saved"})
    ended_run(config, "older")
    root = Path(config["paths"]["data_dir"]) / "runs"
    (root / "crashed" / "live").mkdir()
    (root / "crashed" / "live" / "frames.jsonl").write_text('{"seq":81,"iter":80}\n')
    queued = []
    monkeypatch.setattr(leftovers, "remove_later", lambda folder, **kw: queued.append(folder))
    runs = Runs(config, Samples(config))
    assert not runs.runs and len(queued) == 2
    assert all(folder.parent == root / ".trash" and folder.exists() for folder in queued)
    assert not (root / pending).exists()
    notice = runs.leftovers
    assert len(notice["runs"]) == 1 and notice["runs"][0]["iter"] == 80 and notice["stamp"]
    assert notice["runs"][0]["sample_name"] == "Sample" and notice["runs"][0]["failed"] is None
    assert notice["older_removed"] == 0
    assert runs.older()["runs"][0]["id"] == "older"
    runs.answer_older("older", "discard")
    from conftest import wait_for
    wait_for(lambda: not runs.older()["runs"])
    assert runs.older()["done"][0]["stage"] == "discarded" and len(queued) == 3
    monkeypatch.setattr(runs, "_execute", lambda run: None)
    runs.create(runs.samples.list()[0]["id"], [50, 50, 150, 150])
    assert runs.leftovers is None
    runs.close("discard")


def test_a_leftover_whose_save_ended_after_the_engine_is_noted_as_its_sample_says(config, tmp_path, monkeypatch):
    """A run the engine left "saving" (closed "save running", or a crash mid-save) whose helper
    finished after the engine went was said "not saved to the sample" at the next start, its layer in the sample all
    the same. The sample's own files say: committed there (the element names the run; a stain save's at the revision
    saved) is nothing lost, no line; else its line says the save did not finish (a save cut off before it committed is
    rolled back by the next save into the sample)."""
    import zarr
    from pathlib import Path
    from test_runs_save import ended_run
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    root = tmp_path / "sample.zarr"
    zarr.open_group(str(root), mode="w")
    for key, record in (("cellpose_boundaries_1", {"run": "stain-landed", "result_revision": 7}),
                        ("cellpose_boundaries_2", {"run": "stain-cut-off", "result_revision": 7}),
                        ("cellpose_boundaries_3", {"run": "stain-replaced", "result_revision": 5})):
        (root / "shapes" / key).mkdir(parents=True)
        (root / "shapes" / key / "zarr.json").write_text(json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"sopa_companion": record}}))
    named = zarr.create_array(store=str(root / "tables" / "proseg_table" / "uns" / "sopa_companion" / "run"), shape=(),
                              dtype=str)
    named[()] = "proseg-landed"
    journal = tmp_path / ".proseg-live-backups" / "sample.zarr-20261023-101010-4242"
    journal.mkdir(parents=True)
    (journal / "journal.json").write_text(json.dumps({"state": "writing", "path": str(root.resolve()),
                                                      "written": [["shapes", "cellpose_boundaries_2"]]}))
    def left(run_id, kind="proseg", **save):
        saved = {"state": "saving", **save}
        return ended_run(config, run_id, kind, session_model=2, closed="save running", sample_path=str(root),
                         zarr_save=saved)
    left("stain-landed", "stain", shapes_key="cellpose_boundaries_1", result_revision=7)
    left("proseg-landed", shapes_key="proseg_boundaries", table_key="proseg_table")
    left("stain-cut-off", "stain", shapes_key="cellpose_boundaries_2", result_revision=7)
    left("stain-replaced", "stain", shapes_key="cellpose_boundaries_3", result_revision=7)
    left("proseg-missing", shapes_key="proseg_boundaries_2", table_key="proseg_table_2")
    monkeypatch.setattr(leftovers, "remove_later", lambda folder, **kw: None)
    runs = Runs(config, Samples(config))
    try:
        notice = {row["id"]: row["failed"] for row in runs.leftovers["runs"]}
        assert notice == {"stain-cut-off": "it did not finish", "stain-replaced": "it did not finish",
                          "proseg-missing": "it did not finish"}
        assert not any((Path(config["paths"]["data_dir"]) / "runs" / run_id).exists() for run_id in notice)
    finally:
        runs.close()


def test_unmarked_runs_stay_listed_until_answered_and_do_not_become_crash_leftovers(config):
    from test_runs_save import ended_run
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    from conftest import wait_for
    ended_run(config, "older-done")
    ended_run(config, "older-active", kind="stain", state="running")
    runs = Runs(config, Samples(config))
    rows = {r["id"]: r for r in runs.older()["runs"]}
    assert set(rows) == {"older-done", "older-active"} and not runs.runs
    assert rows["older-active"]["state"] == "stopped" and rows["older-done"]["can_add"] is False
    assert rows["older-done"]["why"] == "sample_gone"                 # /data/sample.zarr is not on this computer
    assert {"id", "sample", "sample_name", "kind", "method", "state", "iter", "tiles", "created", "region", "prior",
            "options", "stain", "can_add", "why", "still_saving", "answer", "stage", "message", "zarr_save"} <= set(rows["older-done"])
    assert all(r.dir.exists() and "session_model" not in r.meta for r in runs.older_runs.values())
    again = Runs(config, Samples(config))
    assert {r["id"] for r in again.older()["runs"]} == set(rows)
    again.answer_older("older-done", "discard")
    wait_for(lambda: len(again.older()["runs"]) == 1)
    again.answer_all_older("discard")
    wait_for(lambda: not again.older()["runs"])


def test_a_process_naming_a_folder_is_found_up_to_its_path_boundary(tmp_path):
    """The start's check for a save still at work: a save helper names the run's output folder in its JSON arguments,
    and holds nothing in it before it reads it: its command line says so (leftovers.named), never a folder whose name
    only starts the same."""
    import json
    import subprocess
    import sys
    import time
    folder, other = tmp_path / "runs" / "abc", tmp_path / "runs" / "xyz"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                              json.dumps({"output": str(folder / "output"),
                                          "copy": str(tmp_path / "backup") + str(other / "output")})])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not leftovers.named([folder]):
            time.sleep(0.05)
        assert leftovers.named([folder, tmp_path / "runs" / "ab", tmp_path / "runs" / "abc-2", other]) == {str(folder)}
    finally:
        child.kill()
        child.wait()
    assert leftovers.named([folder]) == set()


def test_background_removal_retries_held_folders(tmp_path):
    folder = tmp_path / "held"
    folder.mkdir()
    (folder / "file").write_text("kept until released")
    calls = []
    def held(folders):
        calls.append(folders)
        return {str(folder / "file")} if len(calls) < 3 else set()
    worker = leftovers.remove_later(folder, held=held, delays=(0, 0, 0))
    worker.join(2)
    assert len(calls) == 3 and not folder.exists()


def test_sweep_leaves_trash_to_the_background_remover(tmp_path):
    folder = tmp_path / ".trash" / "old"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text("{}")
    assert leftovers.sweep(tmp_path, held=nothing_held) == {"hidden": [], "folders": [], "left": []}
    assert folder.exists()
