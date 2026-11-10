"""Go deeper's engine side (engine/deeper.py DeeperManager) with stand-ins for the check (QcManager, QcModels) and the
analysis scheduler: what Go deeper can do for a run (its states, in the
window's order of 9), refused requests (404/422/503: nothing asked), kept answers at once and the rest to the
scheduler, a request made while learning sent by itself once ready, the memory LRU and the disk cache (bounded, read
back by another session, gone with the run, in the folder of results on proseg's own count: an answer worked out on
the outlines' count is never read), the history (each frame read once per cell, of that folder), the WebSocket events
(every answer asked for, the newest followed one, the details job at most once a second), the failures' reasons (a run
without proseg's counts) and the cleanup of runs nobody watches."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.engine import deeper as D
from sopa_companion.engine.deeper import DeeperError, DeeperManager, error_key, situation_of
from sopa_companion.engine.qc import QcError
from sopa_companion.engine.runs import RunError

MODEL = "0123456789abcdef0123456789abcdef"
NCELLS = 50


class Run:
    """A run on a SpatialData sample whose cells started without nuclei (engine/qc_rules.py start_of "cells": the
    only start that has a start line)."""

    def __init__(self, folder: Path, run_id="r1", frames=8, sample_kind="zarr"):
        self.lock = threading.RLock()
        self.meta = {"id": run_id, "sample": "s1", "sample_kind": sample_kind, "state": "done",
                     "prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": None}}
        self.info = {"cells": NCELLS}
        self.frames = [{"seq": s, "iter": 10 * s} for s in range(frames)]
        self.dir = Path(folder)
        (self.dir / "live").mkdir(parents=True, exist_ok=True)
        (self.dir / "live" / "transcripts.bin.gz").write_bytes(b"")
        (self.dir / "qc").mkdir(exist_ok=True)

    @property
    def live(self):
        return self.dir / "live"


class Runs:
    def __init__(self, *runs):
        self.runs = {r.meta["id"]: r for r in runs}

    def get(self, run_id):
        if run_id not in self.runs:
            raise RunError("Unknown run", 404)
        return self.runs[run_id]


class Qc:
    """QcManager's side: the run's view (GET /runs/{id}/qc), its model, and its series."""

    def __init__(self, model_dir: Path):
        self.version = 0
        self.model_dir = model_dir
        self.view = {"enabled": True, "state": "ready", "reason": None, "sample_state": "ready", "progress": None,
                     "eta_s": None, "model": {"id": MODEL, "deep": True}, "start_source": "xenium"}
        self.series_points, self.series_start = [], {"per100": 10.0}
        self.described = 0

    def describe(self, run_id):
        self.described += 1
        return copy.deepcopy(self.view)

    def qc_model(self, run):
        ready = self.view["enabled"] and self.view["state"] in ("starting", "ready")
        return {"id": MODEL, "path": str(self.model_dir)} if ready else None

    def series(self, run_id, points):
        if not self.series_points:
            raise QcError("No mixing check for this run yet", 404)
        return {"model": MODEL, "start": self.series_start, "points": list(self.series_points)}


class Models:
    """QcModels' side: the sample's view and every sample's (WebSocket snapshot)."""

    def __init__(self):
        self.version = 0
        self.sample = {"state": "ready", "reason": None, "estimate_s": 120, "model": {"id": MODEL},
                       "detail": {"state": "none"}}
        self.views = {}

    def describe(self, sample_id):
        return copy.deepcopy(self.sample)

    def snapshot(self):
        return copy.deepcopy(self.views)


class Scheduler:
    available = True

    def __init__(self):
        self.calls, self.provider = [], None

    def attach_deeper(self, provider):
        self.provider = provider

    def request_deeper(self, run_id, seq, cell, follow=None):
        self.calls.append(("request", run_id, seq, cell, follow))

    def follow_deeper(self, run_id, cell):
        self.calls.append(("follow", run_id, cell))

    def stop_deeper(self, run_id):
        self.calls.append(("stop", run_id))


def world(tmp_path, clock=None, **runs_kwargs):
    run = Run(tmp_path / "runs" / "r1", **runs_kwargs)
    qc, models, scheduler = Qc(tmp_path / "model"), Models(), Scheduler()
    manager = DeeperManager({}, Runs(run), qc, models, scheduler, **({"clock": clock} if clock else {}))
    assert scheduler.provider is manager                      # it attaches itself as the deeper provider
    return manager, run, qc, models, scheduler


def answer(cell, state="ready", **extra):
    return {"v": 1, "state": state, "cell": f"cell-{cell}", "code": 116, "parts": [0.54, 0.459, 0.001], "ms": 1.3,
            **extra}


def deliver(manager, seq, cell, explicit=True, iter_=None, model=MODEL, run="r1", **extra):
    manager.deeper_result({"run": run, "seq": seq, "iter": 10 * seq if iter_ is None and seq != "start" else iter_,
                           "cell": cell, "model": model, "explicit": explicit, "result": answer(cell, **extra)})


def stored(run, seq, cell):
    name = f"start-{cell}.json" if seq == "start" else f"{seq:06d}-{cell}.json"
    return run.dir / "qc" / f"{MODEL[:8]}.pc1" / "deeper" / name


# ---- the states ------------------------------------------------------------------------------------------------------
def test_what_go_deeper_can_do_follows_the_check_and_the_models_details():
    ready = {"enabled": True, "state": "ready", "sample_state": "ready", "model": {"id": MODEL, "deep": True}}
    assert situation_of(ready, None) == {"model": MODEL, "deep": True, "state": "go"}
    assert situation_of({**ready, "state": "starting"}, None)["state"] == "go", "the start line need not exist yet"
    # Unavailable first (the window's priority 1), whatever the switch says.
    assert situation_of({"enabled": True, "state": "unavailable", "reason": "no_transcripts"}, None) == \
        {"model": None, "deep": False, "state": "unavailable", "reason": "no_transcripts"}
    off_unusable = {"enabled": False, "state": "off", "reason": None, "sample_state": "unavailable"}
    assert situation_of(off_unusable, {"reason": "not_counts"})["reason"] == "not_counts"
    assert situation_of(off_unusable, None)["reason"] == "no_table", "a run on a Xenium bundle: no table"
    assert situation_of({**ready, "enabled": False, "state": "off"}, None)["state"] == "off"
    for state, extra in (("waiting", {"reason": "gpu_busy"}), ("learning", {"progress": {"step": 20, "of": 50},
                                                                             "eta_s": 61}),
                         ("failed", {"reason": "oom"})):
        found = situation_of({**ready, "state": state, "model": None, **extra}, None)
        assert found["state"] == state and all(found[k] == v for k, v in extra.items())
    # The model is ready: its details decide.
    plain = {**ready, "model": {"id": MODEL}}
    assert situation_of(plain, {"estimate_s": 117, "detail": {"state": "none"}}) == \
        {"model": MODEL, "deep": False, "state": "needs_detail", "estimate_s": 117}
    assert situation_of({**plain, "detail": {"state": "learning", "progress": {"step": 3, "of": 50}, "eta_s": 90}},
                        None) == {"model": MODEL, "deep": False, "state": "detail", "progress": {"step": 3, "of": 50},
                                  "eta_s": 90}
    assert situation_of(plain, {"detail": {"state": "queued"}})["state"] == "detail", "the sample's own view"
    assert situation_of({**plain, "detail": {"state": "failed", "reason": "detail_mismatch"}}, None) == \
        {"model": MODEL, "deep": False, "state": "detail_failed", "reason": "detail_mismatch"}
    assert situation_of(plain, None, deep_on_disk=True)["state"] == "go", "deep.npz in the model folder"
    assert situation_of({**ready, "model": None}, None)["state"] == "waiting", "ready a moment ago, looked at again"
    assert [error_key(r) for r in ({"reason": "needs_detail"}, {"reason": None, "error": "the GPU worker stopped"},
                                   {"error": "ValueError: boom"}, {"reason": "strange", "error": ""})] == \
        ["needs_detail", "worker", "failed", "failed"]
    # A run without proseg's own counts (made before they were kept, or an older proseg-live): said as such.
    assert [error_key({"reason": r}) for r in ("no_counts", "old_proseg")] == ["no_counts", "old_proseg"]


def test_refused_requests_ask_nothing(tmp_path):
    manager, run, qc, models, scheduler = world(tmp_path)
    with pytest.raises(RunError) as refused:
        manager.request("nope", {"seq": 1, "cell": 1})
    assert refused.value.status == 404
    for body in ([], {"cell": 1}, {"seq": 1.5, "cell": 1}, {"seq": True, "cell": 1}, {"seq": "last", "cell": 1},
                 {"seq": 1, "cell": "3"}, {"seq": 1, "cell": True}, {"seq": 1, "cell": 2, "follow": "yes"}):
        with pytest.raises(DeeperError) as refused:
            manager.request("r1", body)
        assert refused.value.status == 422, body
    for body in ({"seq": 99, "cell": 1}, {"seq": 1, "cell": NCELLS}, {"seq": 1, "cell": -1}):
        with pytest.raises(DeeperError) as refused:
            manager.request("r1", body)
        assert refused.value.status == 404, body
    # Only a start from cells alone has a start line (proseg's count at iteration 0): a start from nuclei only, or from
    # cells with their nuclei (a Xenium bundle's own, a sample's cells with nuclei: proseg's count at iteration 0 holds
    # only the nuclei), is refused, saying so, reason "no_start" and how it started; nothing is asked or kept.
    cells_only = dict(run.meta)
    for prior, kind, words in (
            ({"cells": None, "nuclei": {"kind": "shapes", "key": "nuclei"}}, "nuclei",
             "This run started from nuclei only: it has no start line"),
            ({"cells": {"kind": "shapes", "key": "xenium_cell_boundaries"},
              "nuclei": {"kind": "shapes", "key": "xenium_nucleus_boundaries"}}, "paired",
             "This run started from cells with their nuclei: it has no start line"),
            (None, "paired", "This run started from cells with their nuclei: it has no start line")):
        run.meta.update(prior=prior, sample_kind="zarr" if prior else "xenium")
        with pytest.raises(DeeperError) as refused:
            manager.request("r1", {"seq": "start", "cell": 1, "follow": True})
        assert (refused.value.status, str(refused.value), refused.value.extra) == \
            (404, words, {"reason": "no_start", "start_kind": kind}), prior
        assert manager.request("r1", {"seq": 3, "cell": 1})["state"] == "working", "its frames are worked out"
        scheduler.calls.clear()
    run.meta = cells_only
    scheduler.available = False                               # no GPU worker environment
    with pytest.raises(DeeperError) as refused:
        manager.request("r1", {"seq": 1, "cell": 1})
    assert refused.value.status == 503
    assert scheduler.calls == [] and not manager.pending
    scheduler.available = True
    assert manager.request("r1", {"seq": "2", "cell": 4})["seq"] == 2, "a frame number as text is a frame number"


def test_the_states_reach_the_page_and_only_a_ready_model_with_details_is_asked(tmp_path):
    manager, run, qc, models, scheduler = world(tmp_path)
    qc.view.update(enabled=False, state="off")
    assert manager.request("r1", {"seq": 1, "cell": 3}) == {"seq": 1, "cell": 3, "model": MODEL, "deep": True,
                                                            "state": "off"}
    qc.view.update(enabled=True, state="learning", progress={"step": 20, "of": 50}, eta_s=60, model=None)
    found = manager.request("r1", {"seq": 1, "cell": 3})
    assert (found["state"], found["progress"], found["eta_s"], found["model"]) == ("learning", {"step": 20, "of": 50},
                                                                                   60, MODEL)
    qc.view.update(state="ready", progress=None, eta_s=None, model={"id": MODEL})
    models.sample["detail"] = {"state": "none"}
    assert manager.request("r1", {"seq": 1, "cell": 3})["state"] == "needs_detail"
    assert manager.request("r1", {"seq": 1, "cell": 3})["estimate_s"] == 120
    models.sample["detail"] = {"state": "learning", "progress": {"step": 2, "of": 50}, "eta_s": 100}
    assert manager.request("r1", {"seq": 1, "cell": 3})["state"] == "detail"
    models.sample["detail"] = {"state": "failed", "reason": "detail_mismatch"}
    found = manager.request("r1", {"seq": 1, "cell": 3})
    assert (found["state"], found["reason"], found["deep"]) == ("detail_failed", "detail_mismatch", False)
    assert scheduler.calls == [], "nothing is asked of the scheduler (nor learned) in these states"
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "deep.npz").write_bytes(b"")       # the details arrived (deep.npz installed)
    found = manager.request("r1", {"seq": 1, "cell": 3, "follow": True})
    assert found == {"seq": 1, "cell": 3, "model": MODEL, "deep": True, "state": "working"}
    assert scheduler.calls == [("request", "r1", 1, 3, True)] and not manager.pending
    (run.live / "transcripts.bin.gz").unlink()                # a run made before live analysis
    assert manager.request("r1", {"seq": 1, "cell": 3})["state"] == "unavailable"


def test_kept_answers_come_at_once_and_asked_ones_are_kept_on_disk(tmp_path):
    manager, run, qc, models, scheduler = world(tmp_path)
    assert manager.request("r1", {"seq": 3, "cell": 7})["state"] == "working"
    deliver(manager, 3, 7)                                    # the worker's answer, through the scheduler
    found = manager.request("r1", {"seq": 3, "cell": 7, "follow": True})
    assert found["state"] == "ready" and found["iter"] == 30 and found["result"] == answer(7)
    assert scheduler.calls == [("request", "r1", 3, 7, None), ("follow", "r1", 7)], "no second worker call"
    wait_for(lambda: stored(run, 3, 7).exists())
    kept = json.loads(stored(run, 3, 7).read_text())
    assert kept == {"v": 1, "model": MODEL, "seq": 3, "cell": 7, "iter": 30, "result": answer(7)}
    assert manager.cached("r1", "3", 7) == answer(7) and manager.cached("r1", 3, 7, model=MODEL) == answer(7)
    for args, kwargs in ((("r1", 3, 7), {"model": "another"}), (("r1", 4, 7), {}), (("r1", "x", 7), {}),
                         (("r1", 3, NCELLS), {})):
        with pytest.raises(DeeperError) as refused:
            manager.cached(*args, **kwargs)
        assert refused.value.status == 404
    # The start line's answers too; a followed cell's stay in memory until someone asks for them.
    deliver(manager, "start", 7, iter_=None)
    deliver(manager, 5, 2, explicit=False)
    manager.close()                                           # the writer has written everything queued
    assert stored(run, "start", 7).exists() and not stored(run, 5, 2).exists()
    assert manager.request("r1", {"seq": 5, "cell": 2})["state"] == "ready"
    manager.close()
    assert stored(run, 5, 2).exists(), "asked for now: kept with the run"
    # Another session finds them on disk; a damaged or foreign file is not an answer.
    again, *_ = world(tmp_path)
    assert again.request("r1", {"seq": 3, "cell": 7})["result"] == answer(7)
    stored(run, 5, 2).write_text("{not json")
    stored(run, "start", 7).write_text(json.dumps({**kept, "seq": "start", "model": "someone else's"}))
    assert again.request("r1", {"seq": 5, "cell": 2})["state"] == "working"
    assert again.request("r1", {"seq": "start", "cell": 7})["state"] == "working"
    # An answer worked out on the outlines' count (the folder before proseg's own count) is never read.
    outline = run.dir / "qc" / MODEL[:8] / "deeper" / "000006-4.json"
    outline.parent.mkdir(parents=True)
    outline.write_text(json.dumps({**kept, "seq": 6, "cell": 4}))
    assert again.request("r1", {"seq": 6, "cell": 4})["state"] == "working"
    assert D.qc_folder(run.dir, MODEL) == run.dir / "qc" / f"{MODEL[:8]}.pc1"


def test_a_frame_without_its_own_count_is_answered_kept_and_found_as_the_counted_frame_it_snaps_to(tmp_path):
    """The scheduler works a frame without proseg's count of its own (between counted frames, or its file pruned from the
    spool) out at the counted frame it snaps to (engine/analysis.py snap_deeper): the reply names that frame, the answer
    is kept and found there by the reply, the GET and every later ask, and a frame whose file went is answered by a
    kept answer of an earlier counted frame before any frame whose answer must still be worked out."""
    manager, run, qc, models, scheduler = world(tmp_path)
    counted, files = [0, 2, 5], {0, 2, 5}                     # proseg counted frames 0, 2 and 5 (request mode)

    def snap(run_id, seq, kept=None):                         # AnalysisManager._snap's rule
        if seq == "start":
            return seq
        order = [s for s in counted if s <= seq][::-1] + [s for s in counted if s > seq]
        return next((s for s in order if s in files or (kept is not None and kept(s))), seq)
    scheduler.snap_deeper = snap
    assert manager.request("r1", {"seq": 3, "cell": 7}) == {"seq": 2, "cell": 7, "model": MODEL, "deep": True,
                                                            "state": "working"}
    assert scheduler.calls == [("request", "r1", 2, 7, None)]
    deliver(manager, 2, 7)                                    # the worker's answer, at the frame it worked out
    assert manager.cached("r1", 3, 7) == answer(7), "the GET of the frame asked finds it"
    found = manager.request("r1", {"seq": 3, "cell": 7})
    assert (found["state"], found["seq"], found["result"]) == ("ready", 2, answer(7)), "asked again: ready at once"
    assert manager.request("r1", {"seq": 4, "cell": 7})["seq"] == 2 and len(scheduler.calls) == 1, "nothing worked out"
    # Frame 2's file pruned: its kept answer still answers frames 2 to 4 for this cell; another cell's goes to frame 0.
    files.discard(2)
    assert manager.request("r1", {"seq": 4, "cell": 7})["state"] == "ready"
    assert manager.request("r1", {"seq": 4, "cell": 8}) == {"seq": 0, "cell": 8, "model": MODEL, "deep": True,
                                                            "state": "working"}
    assert scheduler.calls[1:] == [("request", "r1", 0, 8, None)]
    assert manager.request("r1", {"seq": 6, "cell": 7})["seq"] == 5, "a counted frame with its file is its own"
    assert manager.request("r1", {"seq": "start", "cell": 7})["seq"] == "start"
    with pytest.raises(DeeperError) as refused:              # an unknown frame is refused before any snapping
        manager.request("r1", {"seq": 99, "cell": 7})
    assert refused.value.status == 404


def test_an_answer_is_served_only_for_the_count_its_frame_is_read_on_now(tmp_path):
    """Answers are kept by the count they read (the worker's result `count`; none: "averaged"): a frame counted for a
    token, then again for the hold (with f: the count proseg saves), is answered again on f; the answer on e is never
    served for it again, the GET with ?count= says 404 for a count the frame is not read on now, the sampled answer is
    kept on disk as <seq>-<cell>.sampled.json beside the averaged one (another session finds each for its count), and a
    followed answer of the frame counted again is pushed again."""
    manager, run, qc, models, scheduler = world(tmp_path)
    run.frames[3]["counts"] = {"seq": 3, "reason": "request", "has_f": False}
    assert manager.request("r1", {"seq": 3, "cell": 7})["state"] == "working"
    deliver(manager, 3, 7)                                    # an answer of the averaged count (it names none)
    assert manager.request("r1", {"seq": 3, "cell": 7})["result"] == answer(7)
    assert manager.kept_answer("r1", 3, 7, MODEL, "averaged") == (answer(7), True)
    run.frames[3]["counts"] = {"seq": 3, "reason": "hold", "has_f": True}       # the hold counts it again, with f
    assert manager.request("r1", {"seq": 3, "cell": 7})["state"] == "working" and len(scheduler.calls) == 2
    for count in ("averaged", None):
        with pytest.raises(DeeperError) as refused:
            manager.kept_answer("r1", 3, 7, MODEL, count)
        assert refused.value.status == 404
    deliver(manager, 3, 7, count="sampled")
    found = manager.request("r1", {"seq": 3, "cell": 7})
    assert found["state"] == "ready" and found["result"] == answer(7, count="sampled")
    assert manager.kept_answer("r1", 3, 7, MODEL, "sampled") == (answer(7, count="sampled"), True)
    with pytest.raises(DeeperError) as refused:
        manager.kept_answer("r1", 3, 7, MODEL, "averaged")
    assert "count changed" in str(refused.value)
    manager.close()
    folder = stored(run, 3, 7).parent
    assert sorted(path.name for path in folder.iterdir()) == ["000003-7.json", "000003-7.sampled.json"]
    assert json.loads((folder / "000003-7.sampled.json").read_text())["result"] == answer(7, count="sampled")
    again, *_ = world(tmp_path)
    again.runs.runs["r1"].frames[3]["counts"] = {"seq": 3, "reason": "hold", "has_f": True}
    assert again.request("r1", {"seq": 3, "cell": 7})["result"] == answer(7, count="sampled")
    again.runs.runs["r1"].frames[3]["counts"] = {"seq": 3, "reason": "request", "has_f": False}
    assert again.request("r1", {"seq": 3, "cell": 7})["result"] == answer(7)
    # Followed: the same answer again is not news; the frame's answer on its new count is.
    run.frames[5]["counts"] = {"seq": 5, "reason": "request", "has_f": False}
    deliver(manager, 5, 2, explicit=False)
    assert manager.deeper_cached("r1", 5, 2, MODEL, False)
    sent = manager.outbox["r1"]["follow"][0]
    assert manager.deeper_cached("r1", 5, 2, MODEL, False) and manager.outbox["r1"]["follow"][0] == sent
    run.frames[5]["counts"] = {"seq": 5, "reason": "finish", "has_f": True}
    manager._remember(("r1", 5, 2, MODEL, D.RESULT_V, "sampled"),
                      {"iter": 50, "result": answer(2, count="sampled"), "disk": False, "count": "sampled"})
    assert manager.deeper_cached("r1", 5, 2, MODEL, False)
    follow = manager.outbox["r1"]["follow"]
    assert follow[0] > sent and follow[1]["result"] == answer(2, count="sampled") and follow[1]["cached"]


def test_a_request_made_while_learning_fills_in_by_itself(tmp_path):
    now = [100.0]
    manager, run, qc, models, scheduler = world(tmp_path, clock=lambda: now[0])
    qc.view.update(state="learning", model=None)
    assert manager.request("r1", {"seq": 2, "cell": 9, "follow": True})["state"] == "learning"
    manager.events({}, "r1", 0.0)
    assert scheduler.calls == [] and manager.pending["r1"]["cell"] == 9
    qc.view.update(state="ready", model={"id": MODEL, "deep": True})
    qc.version += 1                                           # the sample finished learning
    now[0] += D.PENDING_CHECK_S
    manager.events({}, "r1", 1.0)
    assert scheduler.calls == [("request", "r1", 2, 9, True)] and not manager.pending
    # Kept already by then: pushed to the pages (nobody waits on an HTTP answer).
    qc.view.update(state="learning", model=None)
    manager.request("r1", {"seq": 3, "cell": 9})
    deliver(manager, 3, 9)
    cursor = {}
    manager.events(cursor, "r1", 2.0)
    qc.view.update(state="ready", model={"id": MODEL, "deep": True})
    qc.version += 1
    now[0] += D.PENDING_CHECK_S
    events = manager.events(cursor, "r1", 3.0)
    assert [(e["seq"], e["cell"], e["cached"], e["explicit"]) for e in events if e.get("run")] == [(3, 9, True, True)]
    # Too old, or the window closed: dropped.
    qc.view.update(state="learning", model=None)
    manager.request("r1", {"seq": 4, "cell": 1})
    now[0] += D.PENDING_TTL_S
    qc.version += 1
    manager.events({}, "r1", 4.0)
    assert not manager.pending
    manager.request("r1", {"seq": 4, "cell": 1})
    assert manager.log_info("r1")["run"]["pending"] == {"seq": 4, "cell": 1, "follow": None}
    manager.stop("r1")
    assert not manager.pending and scheduler.calls[-1] == ("stop", "r1")
    manager.close()                                           # the answer asked for is on disk by now
    assert manager.log_info("r1")["run"] == {"kept": 1, "pending": None, "on_disk": 1, "watchers": 0}


def test_answers_are_bounded_in_memory_and_on_disk_and_go_with_their_run(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE_KEPT", 3)
    monkeypatch.setattr(D, "DISK_KEPT", 2)
    manager, run, qc, models, scheduler = world(tmp_path)
    for cell in range(4):
        deliver(manager, 1, cell)
    manager.close()
    assert len(manager.cache) == 3 and ("r1", 1, 0, MODEL, 1, "averaged") not in manager.cache, "the oldest left memory"
    assert sorted(p.name for p in stored(run, 1, 0).parent.iterdir()) == ["000001-2.json", "000001-3.json"]
    other = Run(tmp_path / "runs" / "r2", run_id="r2")
    manager.runs.runs["r2"] = other
    deliver(manager, 1, 5, run="r2")
    manager.close()
    assert (other.dir / "qc" / f"{MODEL[:8]}.pc1" / "deeper" / "000001-5.json").exists()
    manager.forget("r2")                                      # the run was deleted (its folder with it)
    assert all(key[0] != "r2" for key in manager.cache) and scheduler.calls[-1] == ("stop", "r2")
    deliver(manager, 2, 5, run="r2")                          # a pass that was under way: nothing kept
    assert all(key[0] != "r2" for key in manager.cache) and "r2" not in manager.outbox


def test_the_history_reads_each_frame_once_per_cell(tmp_path, monkeypatch):
    import zstandard
    manager, run, qc, models, scheduler = world(tmp_path)
    folder = run.dir / "qc" / f"{MODEL[:8]}.pc1"       # the results on proseg's own count (analysis.py qc_folder)
    folder.mkdir(parents=True)
    for seq in range(6):
        values = np.full(NCELLS, 255, np.uint8)
        values[7] = 10 * seq
        (folder / f"{seq:06d}.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    start = np.full(NCELLS, 3, np.uint8).tobytes() + b"the rest of start.bin"
    (folder / "start.bin").write_bytes(start)                  # raw bytes are read too
    reads = []
    real = D._read_codes
    monkeypatch.setattr(D, "_read_codes", lambda path: reads.append(Path(path).name) or real(path))
    with pytest.raises(QcError):                              # before the check has a model
        manager.history("r1", 7)
    qc.series_points = [{"seq": s, "iter": 10 * s} for s in (0, 2, 4, 9)]   # frame 9's file is not there
    found = manager.history("r1", 7, points=4)
    assert found == {"model": MODEL, "start": 3, "points": [[0, 0], [20, 20], [40, 40]]}
    assert sorted(reads) == ["000000.qc.bin", "000002.qc.bin", "000004.qc.bin", "000009.qc.bin", "start.bin"]
    reads.clear()
    qc.series_points.append({"seq": 5, "iter": 50})
    # What is new is read, and a file that could not be read before is looked at again: a failed read (a file being
    # replaced, a disk hiccup) is never kept as a gap in the line.
    assert manager.history("r1", 7)["points"][-1] == [50, 50] and sorted(reads) == ["000005.qc.bin", "000009.qc.bin"]
    values = np.full(NCELLS, 255, np.uint8)
    values[7] = 90
    (folder / "000009.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    reads.clear()
    assert [90, 90] in manager.history("r1", 7)["points"] and reads == ["000009.qc.bin"], "it lands once readable, read once"
    qc.series_points = [p for p in qc.series_points if p["seq"] != 9]
    assert manager.history("r1", 8)["points"] == [[0, 255], [20, 255], [40, 255], [50, 255]], "not judged: 255"
    for cell in (-1, NCELLS):
        with pytest.raises(DeeperError):
            manager.history("r1", cell)
    qc.series_start = None                                    # no start line (yet)
    assert manager.history("r1", 9)["start"] is None
    # The outlines' count's files (the folder before proseg's own count) are never read.
    old = run.dir / "qc" / MODEL[:8]
    old.mkdir()
    values = np.full(NCELLS, 77, np.uint8)
    (old / "000007.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    qc.series_points = [{"seq": 7, "iter": 70}]
    assert manager.history("r1", 11)["points"] == []
    # The upscale frame repeats the last burn-in frame's iteration: one point per iteration, the newer frame's byte, so
    # the window's small line always rises (the first real qc_deeper run saw iteration 200 twice).
    qc.series_points = [{"seq": 0, "iter": 0}, {"seq": 2, "iter": 20}, {"seq": 4, "iter": 20}, {"seq": 5, "iter": 50}]
    assert manager.history("r1", 7)["points"] == [[0, 0], [20, 40], [50, 50]]


def test_the_history_reads_a_frame_again_once_it_is_scored_on_another_count(tmp_path, monkeypatch):
    """A frame counted again (a token's count, then the hold's: the count proseg saves) is scored again and its .qc.bin
    rewritten; its series point says the new count, and the history reads that frame's byte again, the others as they
    were read."""
    import zstandard
    manager, run, qc, models, scheduler = world(tmp_path)
    folder = run.dir / "qc" / f"{MODEL[:8]}.pc1"
    folder.mkdir(parents=True)

    def write(seq, byte):
        values = np.full(NCELLS, 255, np.uint8)
        values[7] = byte
        (folder / f"{seq:06d}.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    for seq in range(3):
        write(seq, 10 * seq)
    reads = []
    real = D._read_codes
    monkeypatch.setattr(D, "_read_codes", lambda path: reads.append(Path(path).name) or real(path))
    qc.series_start = None
    qc.series_points = [{"seq": s, "iter": 10 * s, "count": "averaged"} for s in range(3)]
    assert manager.history("r1", 7)["points"] == [[0, 0], [10, 10], [20, 20]] and len(reads) == 3
    write(2, 99)                                              # the hold's count, on f: frame 2 scored again
    reads.clear()
    assert manager.history("r1", 7)["points"] == [[0, 0], [10, 10], [20, 20]] and reads == [], "read once a count"
    qc.series_points[2] = {"seq": 2, "iter": 20, "count": "sampled"}
    assert manager.history("r1", 7)["points"] == [[0, 0], [10, 10], [20, 99]] and reads == ["000002.qc.bin"]
    reads.clear()
    assert manager.history("r1", 7)["points"][-1] == [20, 99] and reads == []


def test_the_history_waits_for_a_frame_being_scored_again(tmp_path, monkeypatch):
    """A frame counted again (its counts line now the hold's, with f, its count's file there) is being scored again:
    the worker rewrites its .qc.bin on f before the new result is in, so the history does not read it until its series
    point says the new count (no byte of either count stands for it meanwhile), and reads it then."""
    import zstandard
    manager, run, qc, models, scheduler = world(tmp_path)
    folder = run.dir / "qc" / f"{MODEL[:8]}.pc1"
    folder.mkdir(parents=True)
    for seq, byte in ((0, 0), (1, 10), (2, 99)):              # frame 2's bytes rewritten on f already
        values = np.full(NCELLS, 255, np.uint8)
        values[7] = byte
        (folder / f"{seq:06d}.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    (run.live / "frames").mkdir(parents=True, exist_ok=True)
    (run.live / "frames" / "000002.counts.zst").write_bytes(b"PLC1")
    run.frames[2].update(counts_file="frames/000002.counts.zst", counts={"seq": 2, "reason": "hold", "has_f": True})
    reads = []
    real = D._read_codes
    monkeypatch.setattr(D, "_read_codes", lambda path: reads.append(Path(path).name) or real(path))
    qc.series_start = None
    qc.series_points = [{"seq": s, "iter": 10 * s, "count": "averaged"} for s in range(3)]
    assert manager.history("r1", 7)["points"] == [[0, 0], [10, 10]] and "000002.qc.bin" not in reads
    qc.series_points[2] = {"seq": 2, "iter": 20, "count": "sampled"}          # its new result is in
    assert manager.history("r1", 7)["points"] == [[0, 0], [10, 10], [20, 99]] and reads[-1] == "000002.qc.bin"


def test_pages_get_every_answer_they_asked_for_and_the_newest_followed_one(tmp_path):
    manager, run, qc, models, scheduler = world(tmp_path)
    deliver(manager, 1, 2, explicit=False)
    deliver(manager, 2, 2, explicit=False)
    deliver(manager, 0, 5)
    cursor = {}
    first = manager.events(cursor, "r1", 0.0)
    assert [(e["seq"], e["explicit"]) for e in first] == [(2, False)], "a new watch: the newest followed answer"
    event = first[0]
    assert set(event) == {"ev", "run", "seq", "iter", "cell", "model", "result", "explicit", "cached", "latency_ms",
                          "worker_ms"}
    assert (event["ev"], event["iter"], event["worker_ms"], event["cached"]) == ("deeper", 20, 1.3, False)
    deliver(manager, 2, 2, explicit=False, worker_ms=4.5)       # the worker's whole share of its pass, when it says
    assert manager.events(cursor, "r1", 0.05)[0]["worker_ms"] == 4.5
    assert manager.events(cursor, "r1", 0.1) == [], "nothing new"
    for cell in range(6):
        deliver(manager, 3, cell)
    deliver(manager, 3, 2, explicit=False)
    deliver(manager, 4, 2, explicit=False)
    manager.deeper_result({"run": "r1", "seq": 6, "iter": 60, "cell": 1, "model": MODEL, "explicit": True,
                           "error": "the GPU worker stopped", "reason": "worker"})
    events = manager.events(cursor, "r1", 0.2)
    asked = [e for e in events if e["explicit"]]
    assert [(e.get("seq"), e.get("cell")) for e in asked] == [(3, 3), (3, 4), (3, 5), (6, 1)], "the newest 4"
    assert asked[-1] == {"ev": "deeper", "run": "r1", "seq": 6, "cell": 1, "state": "error", "error": "worker",
                         "explicit": True}
    assert [(e["seq"], e["cell"]) for e in events if not e["explicit"]] == [(4, 2)], "the newest followed one only"
    assert manager.deeper_cached("r1", 4, 2, MODEL, False) is True
    assert manager.events(cursor, "r1", 0.3) == [], "the same followed answer again is not news"
    assert manager.events({}, "r2", 0.0) == [] and manager.events({}, None, 0.0) == []
    # Latency of an answer asked for, from the request to the answer (the harness's deeper.latency_ms).
    now = [10.0]
    manager.clock = lambda: now[0]
    manager.request("r1", {"seq": 7, "cell": 4})
    now[0] += 0.25
    deliver(manager, 7, 4)
    (event,) = manager.events(cursor, "r1", 0.4)
    assert event["latency_ms"] == 250.0 and event["cached"] is False


def test_the_details_job_is_heard_at_most_once_a_second(tmp_path):
    manager, run, qc, models, scheduler = world(tmp_path)
    detail = lambda state, step=None, reason=None: {"s1": {"state": "ready", "detail": {
        "state": state, "progress": None if step is None else {"step": step, "of": 50}, "eta_s": 60, "reason": reason}}}
    models.views = detail("none")
    cursor = {}
    assert manager.events(cursor, None, 0.0) == []
    models.views = detail("learning", 1)
    assert manager.events(cursor, None, 0.0) == [{"ev": "deeper", "sample": "s1", "state": "detail",
                                                  "progress": {"step": 1, "of": 50}, "eta_s": 60, "reason": None}]
    models.views = detail("learning", 2)
    assert manager.events(cursor, None, 0.5) == [] and cursor["pending"] is True
    assert [e["progress"]["step"] for e in manager.events(cursor, None, 1.1)] == [2]
    models.views = detail("ready")
    assert [e["state"] for e in manager.events(cursor, None, 1.2)] == ["ready"]
    assert manager.events({}, None, 1.3) == [], "a page that never saw it being learned has nothing to fill in"
    models.views = detail("failed", reason="detail_mismatch")
    assert [(e["state"], e["reason"]) for e in manager.events(cursor, None, 1.4)] == [("detail_failed",
                                                                                         "detail_mismatch")]


def test_a_run_nobody_watches_stops_being_followed(tmp_path):
    now = [0.0]
    manager, run, qc, models, scheduler = world(tmp_path, clock=lambda: now[0])
    manager.watching(None, "r1")                              # two pages watch the run
    manager.watching(None, "r1")
    manager.watching("r1", None)
    now[0] += 2 * D.UNWATCHED_GRACE_S
    qc.version += 1
    manager.events({}, None, 0.0)
    assert ("follow", "r1", None) not in scheduler.calls, "one page still watches it"
    manager.watching("r1", None)                              # the last one went (without closing its window)
    now[0] += D.UNWATCHED_GRACE_S / 2
    manager.watching(None, "r1")                              # ... and came back (a reload): still followed
    manager.watching("r1", "r2")
    deliver(manager, 1, 1, explicit=False)
    now[0] += D.UNWATCHED_GRACE_S
    manager.events({}, None, 1.0)
    assert scheduler.calls == [("follow", "r1", None)] and "r1" not in manager.outbox


def test_an_answer_json_cannot_carry_is_made_safe(tmp_path):
    """A non-finite number in the worker's answer (JSON has none) becomes null: the page's JSON.parse and the HTTP
    answer (allow_nan=False) never fail on it."""
    manager, run, qc, models, scheduler = world(tmp_path)
    manager.deeper_result({"run": "r1", "seq": 1, "iter": 10, "cell": 2, "model": MODEL, "explicit": True,
                           "result": answer(2, same_kind_share=float("nan"),
                                            neighbours=[{"similar": float("inf"), "tx": 3.5}])})
    kept = manager.request("r1", {"seq": 1, "cell": 2})["result"]
    assert kept["same_kind_share"] is None and kept["neighbours"] == [{"similar": None, "tx": 3.5}]
    json.dumps(kept, allow_nan=False)
    manager.close()
    assert json.loads(stored(run, 1, 2).read_text())["result"] == kept


def test_the_cells_word_since_the_start_last_shown_goes_back_to_the_worker(tmp_path):
    """The worker keeps a cell's word since the start near its band as last shown (start_word); the manager hands it the
    word the newest worked-out answer for the cell said (per run, model and cell), and forgets it with the run."""
    manager, run, qc, models, scheduler = world(tmp_path)
    assert manager.deeper_start_prev("r1", 7, MODEL) is None
    deliver(manager, 3, 7, start_change={"points": 6.0, "band": 5.0, "direction": "more"})
    assert manager.deeper_start_prev("r1", 7, MODEL) == "more"
    assert manager.deeper_start_prev("r1", 7, "another model") is None and manager.deeper_start_prev("r1", 8, MODEL) is None
    deliver(manager, 4, 7, explicit=False, start_change={"points": 1.0, "band": 5.0, "direction": "same"})
    assert manager.deeper_start_prev("r1", 7, MODEL) == "same", "the newest answer's"
    deliver(manager, 5, 7, explicit=False, start_change={"points": 1.0, "band": 5.0, "direction": "up"})
    deliver(manager, 6, 7, explicit=False)
    assert manager.deeper_start_prev("r1", 7, MODEL) == "same", "an answer without a known word changes nothing"
    for cell in range(100, 170):                              # bounded: the oldest cells go first
        deliver(manager, 4, cell, explicit=False, start_change={"points": -9.0, "band": 3.0, "direction": "fewer"})
    assert len(manager.start_words) == D.SAME_KEPT and manager.deeper_start_prev("r1", 169, MODEL) == "fewer"
    manager.forget("r1")
    assert manager.deeper_start_prev("r1", 169, MODEL) is None and not manager.start_words


def test_a_followed_cells_word_since_the_start_is_held_steady(tmp_path):
    """While a cell is followed its word since the start changes only once the answers' own word has been the new one
    for 5 s over at least 2 answers (qc_rules.Steady, as the tissue's verdicts): a one-frame dip of its like-for-like
    estimate under the band no longer flips the window's word. A held answer says so (held, direction_now); answers
    asked for are judged alone; the word handed back to the worker is the one shown."""
    now = [100.0]
    manager, run, qc, models, scheduler = world(tmp_path, clock=lambda: now[0])

    def follow(seq, word, explicit=False):
        deliver(manager, seq, 7, explicit=explicit, start_change={"points": 1.0, "band": 3.0, "direction": word})
        return manager.cache[("r1", seq, 7, MODEL, D.RESULT_V, "averaged")]["result"]["start_change"]
    assert follow(1, "more") == {"points": 1.0, "band": 3.0, "direction": "more"}
    now[0] += 0.2
    dip = follow(2, "same")
    assert dip["direction"] == "more" and dip["held"] is True and dip["direction_now"] == "same"
    assert manager.deeper_start_prev("r1", 7, MODEL) == "more", "the worker hears the word shown"
    now[0] += 0.2
    assert follow(3, "more") == {"points": 1.0, "band": 3.0, "direction": "more"}, "back: nothing held"
    for k in range(4, 9):                                     # the same word for 5 s: then it shows
        now[0] += 1.25
        shown = follow(k, "same")
    assert shown == {"points": 1.0, "band": 3.0, "direction": "same"}
    now[0] += 0.2
    asked = follow(9, "fewer", explicit=True)
    assert asked["direction"] == "fewer" and "held" not in asked, "asked for: judged alone"
    manager.forget("r1")
    assert not manager.start_steady.verdicts


def test_a_held_word_keeps_the_two_numbers_its_change_compares(tmp_path):
    """A followed cell's held word is an earlier answer's, and so are the two numbers the step line puts beside it (now
    and start: the cell's share from neighbours now and at the start, like for like): those of the newest answer whose
    own word was the one shown, never the newest answer's, which may have the other sign ("less" beside 44 % now, 43 %
    at the start). The rest of start_change (points, band, the answer's own word as direction_now) is the answer's
    own; once its own word shows again, its own numbers do."""
    now = [100.0]
    manager, run, qc, models, scheduler = world(tmp_path, clock=lambda: now[0])

    def follow(seq, points, at_now, at_start, word):
        now[0] += 0.2
        deliver(manager, seq, 7, explicit=False, start_change={"points": points, "band": 5.0, "now": at_now,
                                                               "start": at_start, "direction": word})
        return manager.cache[("r1", seq, 7, MODEL, D.RESULT_V, "averaged")]["result"]["start_change"]
    assert follow(1, -6.0, 37.0, 43.0, "fewer")["now"] == 37.0
    assert follow(2, -3.5, 39.5, 43.0, "same") == {"points": -3.5, "band": 5.0, "now": 37.0, "start": 43.0,
                                                   "direction": "fewer", "held": True, "direction_now": "same"}
    assert follow(3, 1.0, 44.0, 43.0, "same") == {"points": 1.0, "band": 5.0, "now": 37.0, "start": 43.0,
                                                  "direction": "fewer", "held": True, "direction_now": "same"}
    assert follow(4, -5.5, 38.0, 43.5, "fewer") == {"points": -5.5, "band": 5.0, "now": 38.0, "start": 43.5,
                                                    "direction": "fewer"}
    assert follow(5, 1.0, 44.0, 43.0, "same")["now"] == 38.0             # held again: the newer word's numbers
    for seq in range(6, 11):                                                # the same word for 5 s: then it shows
        now[0] += 1.0
        shown = follow(seq, 1.0, 44.0, 43.0, "same")
    assert shown == {"points": 1.0, "band": 5.0, "now": 44.0, "start": 43.0, "direction": "same"}
    manager.forget("r1")


def test_at_rest_a_followed_cells_word_since_the_start_is_its_own_with_its_two_numbers(tmp_path):
    """An answer read on the count proseg saves (a frame proseg counted for a pause, the hold, a finish or the end: the
    run rests there) says the frame's own word since the start with its own two numbers, as the panel's verdict does at
    rest, never a word held from earlier answers (the audit's pause: "less" held from 26.5 % now, 32 % at the start,
    beside the frame's own 29.5 % and 31.6 %, about the same); the word held before it is let go, and the followed
    answers after it are held from it."""
    now = [100.0]
    manager, run, qc, models, scheduler = world(tmp_path, clock=lambda: now[0])

    def follow(seq, points, at_now, at_start, word, count="averaged"):
        now[0] += 0.2
        deliver(manager, seq, 7, explicit=False, count=count, start_change={"points": points, "band": 5.0,
                                                                            "now": at_now, "start": at_start,
                                                                            "direction": word})
        return manager.cache[("r1", seq, 7, MODEL, D.RESULT_V, count)]["result"]["start_change"]
    assert follow(1, -5.5, 26.5, 32.0, "fewer")["direction"] == "fewer"
    assert follow(2, -2.1, 29.5, 31.6, "same")["held"] is True                   # running: held, its tooltip's
    rest = follow(3, -2.1, 29.5, 31.6, "same", count="sampled")
    assert rest == {"points": -2.1, "band": 5.0, "now": 29.5, "start": 31.6, "direction": "same"}
    assert manager.deeper_start_prev("r1", 7, MODEL) == "same"
    assert follow(4, -5.6, 26.4, 32.0, "fewer") == {"points": -5.6, "band": 5.0, "now": 29.5, "start": 31.6,
                                                    "direction": "same", "held": True, "direction_now": "fewer"}
    manager.forget("r1")


def test_the_neighbours_classes_last_shown_go_back_to_the_worker(tmp_path):
    """The worker keeps a neighbour's same-kind class while its likeness sits near the cut-off; the manager hands it
    the classes the newest worked-out answer for the cell used (per run, model and cell)."""
    manager, run, qc, models, scheduler = world(tmp_path)
    assert manager.deeper_prev("r1", 7, MODEL) == {}
    deliver(manager, 3, 7, neighbours=[{"cell": 2, "same": True}, {"cell": 5, "same": False}, {"cell": 9}])
    assert manager.deeper_prev("r1", 7, MODEL) == {"2": True, "5": False}, "a row without its class is not guessed"
    assert manager.deeper_prev("r1", 7, "another model") == {} and manager.deeper_prev("r1", 8, MODEL) == {}
    deliver(manager, 4, 7, explicit=False, neighbours=[{"cell": 2, "same": False}])
    assert manager.deeper_prev("r1", 7, MODEL) == {"2": False}, "the newest answer's"
    for cell in range(100, 170):                              # bounded: the oldest cells go first
        deliver(manager, 4, cell, explicit=False, neighbours=[{"cell": 1, "same": True}])
    assert len(manager.same_kind) == D.SAME_KEPT and manager.deeper_prev("r1", 169, MODEL) == {"1": True}
    deliver(manager, 4, 7, explicit=False, neighbours=[{"cell": 2, "same": True}])
    manager.forget("r1")
    assert manager.deeper_prev("r1", 7, MODEL) == {}
