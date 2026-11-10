"""Go deeper in the analysis scheduler (engine/analysis.py): explicit
requests before UMAP requests, the newest per run winning; ["qc", "deeper"] for a frame not scored yet, never "umap";
a followed cell riding along every live job of its run (not background passes); `_release` keeping followed runs and
runs with a request waiting; requests waiting for the model and the start line, dropped when the check is off; and
`_process`: the deeper provider's cache asked first (a cached answer never reaches the worker), the request block, the
hand-off of answers and failures (a Go deeper failure never fails the pass; a followed cell backs off), the start line
on proseg's count at iteration 0, the like-for-like start of a cells-only start, and once through a worker process.
The runs are counted (proseg's own count with every frame: tests/test_analysis_counts.py's CountedRun and store)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.engine import analysis as A
from sopa_companion.engine.analysis import (DEEPER_K, DEEPER_RETRY_S, DEEPER_V, POLL_S, QC_MODEL_POLL_S, AnalysisError,
                                         AnalysisManager, Job)
from test_analysis_counts import (GEOMETRY, Clock, CountedRun, CountsRuns, Provider, ask, finish, manager, start_run,
                                 stub_worker)

JUDGE, NCELLS = 100, 12


def counted(tmp_path, name="r1", frames=1, state="running", cells=3000, start=False) -> CountedRun:
    """A run in its folder with `frames` frames, every one counted (frame 0 by itself, the others for a token);
    `start`: a cells-only start with its prior exported (a start line)."""
    run = start_run(tmp_path, name, cells=cells, state=state) if start else \
        CountedRun(tmp_path / name, run_id=name, cells=cells, state=state)
    for _ in range(frames - 1):
        run.add_frame("request")
    return run


class Deeper:
    """A deeper provider (engine/deeper.py's role): answers it holds already ("cached"), and every record it gets."""

    def __init__(self, cached=()):
        self.cached, self.asked, self.records = set(cached), [], []

    def deeper_cached(self, run_id, seq, cell, model, explicit):
        self.asked.append((run_id, seq, cell, model, explicit))
        return (run_id, seq, cell) in self.cached

    def deeper_result(self, record):
        self.records.append(record)


def deep_model(folder: Path) -> str:
    """A model folder holding the details Go deeper needs (only their presence matters to the scheduler)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "deep.npz").write_bytes(b"")
    return str(folder)


def checked(tmp_path, *runs, model=("0123456789abcdef", "/models/m1"), **settings):
    """A scheduler with a QC provider and every run's check on (test_analysis_counts.checked, with this file's
    model)."""
    m = manager(tmp_path, *runs, **settings)
    m.attach_qc(Provider(model=model))
    for run in runs:
        m.configure_qc(run.meta["id"], enabled=True)
    return m


def deeper_stub(m, calls, fail=False, missing=False):
    """stub_worker plus op "deeper": its answer echoes the block (`fail`: that op fails; `missing`: an older worker
    that drops the op silently). `calls` gets every request as sent."""
    stub_worker(m, calls)
    base = m._call

    def call(request):
        ops = request.get("ops") or []
        inner = {**request, "ops": [op for op in ops if op != "deeper"]}
        if request["op"] == "qc_start" or inner["ops"]:
            reply = base(inner)
            calls[-1] = request
        else:
            calls.append(request)
            reply = {"run": request["run"], "seq": request.get("seq"), "ok": True}
        if "deeper" in ops and not missing:
            if fail:
                reply.setdefault("errors", {})["deeper"] = "ValueError: deep.npz failed its self-test here"
                reply["error"], reply["ok"] = reply.get("error") or reply["errors"]["deeper"], False
            else:
                block = request["deeper"]
                reply["deeper"] = {"v": block["v"], "state": "ready", "cell": block["cell"], "code": 116,
                                   "parts": [0.54, 0.459, 0.001], "ms": 1.2,
                                   **({"count": request["count"]} if request.get("count", "averaged") != "averaged"
                                      else {})}
        return reply
    m._call = call


# ---- which job comes next ----------------------------------------------------------------------------------------------
def test_explicit_requests_go_first_newest_per_run_and_never_with_umap(tmp_path):
    run, other = counted(tmp_path, frames=8, state="done"), counted(tmp_path, "r2", frames=8, state="done")
    m = checked(tmp_path, run, other)
    m.configure("r1", seq=5)                                  # the viewer stepped there: UMAP and the check
    m.request_deeper("r1", 3, 7)
    m.request_deeper("r1", 4, 9)                              # the newest request of the run wins
    job, wait = m._next_job()
    assert wait is None and job == ("r1", 4) and job.kind == "request"
    assert job.ops == ("qc", "deeper"), "a frame not scored yet: the card and the window change together"
    assert job.deeper == {"cell": 9, "start": False, "explicit": True} and job.qc.model == "0123456789abcdef"
    finish(m, job)
    job, _ = m._next_job()
    assert job == ("r1", 5) and job.ops == ("umap", "qc") and job.deeper is None   # then the UMAP request, as before
    finish(m, job)
    m.request_deeper("r1", 4, 2)                              # scored now: the window alone, never UMAP
    m.request_deeper("r2", 6, 1)
    m.configure("r2", seq=6)
    first, _ = m._next_job()
    assert first == ("r1", 4) and first.ops == ("deeper",)
    finish(m, first)
    # One explicit Clusters & UMAP request between explicit Go deeper jobs (while a window
    # was scrubbed, a new Go deeper request came during every job and the UMAP pane waited until the scrubbing stopped).
    umap, _ = m._next_job()
    assert umap == ("r2", 6) and umap.ops == ("umap", "qc") and umap.deeper is None
    finish(m, umap)
    second, _ = m._next_job()
    assert second == ("r2", 6) and second.ops == ("deeper",) and "umap" not in second.ops, "scored by the UMAP job meanwhile"
    finish(m, second)
    # The check's own request for the same frame is answered by the window's job (counted once).
    m.configure_qc("r1", seq=7)
    m.request_deeper("r1", 7, 0)
    job, _ = m._next_job()
    assert job == ("r1", 7) and job.ops == ("qc", "deeper") and "r1" not in m.qc_requested
    for bad, status in (("x", 422), (99, 404)):
        with pytest.raises(AnalysisError) as refused:
            m.request_deeper("r1", bad, 0)
        assert refused.value.status == status and "r1" not in m.deeper_requested
    with pytest.raises(AnalysisError):
        m.request_deeper("r1", 2, -1)
    m.python = ""                                             # no GPU worker environment
    with pytest.raises(AnalysisError) as refused:
        m.request_deeper("r1", 2, 0)
    assert refused.value.status == 503


def test_a_followed_cell_rides_along_every_live_job_and_no_background_pass(tmp_path):
    run = counted(tmp_path, start=True)
    m = checked(tmp_path, run)
    job, _ = m._next_job()
    assert job.ops == ("umap", "qc") and job.deeper is None   # nobody follows a cell: as before
    finish(m, job)
    m.follow_deeper("r1", 11)
    background, _ = m._next_job()                             # the start line (a background pass) carries no cell
    assert background.kind == "start" and background.ops == ("qc_start",) and background.deeper is None
    finish(m, background)
    run.add_frame()
    job, _ = m._next_job()
    assert job.kind == "live" and job.ops == ("umap", "qc", "deeper")
    assert job.deeper == {"cell": 11, "start": False, "explicit": False}
    finish(m, job)
    run.add_frame()
    m._store("r1", 2, {"seq": 2})                             # its UMAP exists: the check alone, with the cell
    job, _ = m._next_job()
    assert job.ops == ("qc", "deeper")
    finish(m, job)
    run.add_frame()
    job.qc.done[3] = "averaged"                               # scored already: UMAP alone, with the cell
    job, _ = m._next_job()
    assert job.ops == ("umap", "deeper") and job.deeper["cell"] == 11
    finish(m, job)
    m.request_deeper("r1", 1, 4, follow=True)                 # another cell: followed from now on
    assert m._next_job()[0].deeper == {"cell": 4, "start": False, "explicit": True}
    run.add_frame()
    assert m._next_job()[0].deeper["cell"] == 4
    m.request_deeper("r1", 1, 6)                              # follow absent: following stays as it is
    assert m.deeper_follow == {"r1": 4}
    m.stop_deeper("r1")                                       # the window closed
    assert not m.deeper_follow and not m.deeper_requested
    run.add_frame()
    job, _ = m._next_job()
    assert job.kind == "live" and "deeper" not in job.ops
    m.follow_deeper("r1", 4)
    m.configure_qc("r1", enabled=False)                       # the check off: nothing to explain it with
    run.add_frame()
    job, _ = m._next_job()
    assert job.ops == ("umap",) and job.deeper is None


def test_release_keeps_a_followed_run_and_one_waiting_for_an_answer(tmp_path):
    m = checked(tmp_path, counted(tmp_path, "old", frames=3, state="done"))
    m.follow_deeper("old", 2)
    while (job := m._next_job()[0]) is not None:
        finish(m, job)
    assert not m.to_forget, "a window follows a cell of it: each click would load it again"
    m.follow_deeper("old", None)
    assert m._next_job() == (None, None) and m.to_forget == {"old"}
    m.to_forget.clear()
    m.request_deeper("old", 1, 0)                             # asked again: loaded again, released again afterwards
    job, _ = m._next_job()
    assert job.ops == ("deeper",) and job.qc.released is False and not m.to_forget
    finish(m, job)
    assert m._next_job() == (None, None) and m.to_forget == {"old"}
    m.to_forget.clear()
    m.qc_runs["old"].released = False
    m.deeper_requested["old"] = (1, 0)                         # a request waiting (not served yet)
    m._release("old", m.qc_runs["old"])
    assert not m.to_forget and m.qc_runs["old"].released is False


def test_requests_wait_for_the_model_and_the_start_line_and_follow_the_switch(tmp_path):
    run = counted(tmp_path, frames=4, state="paused", start=True)
    m = manager(tmp_path, run)
    m.configure("r1", enabled=False)                          # Clusters & UMAP off by a tool: the check alone
    provider = Provider(model=None)                           # its sample is being learned again
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    m.request_deeper("r1", 2, 5)
    assert m._next_job() == (None, QC_MODEL_POLL_S) and m.deeper_requested == {"r1": (2, 5)}
    provider.model = ("feedc0de99", "/models/m2")
    m.qc_changed()
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.ops == ("qc", "deeper") and job.kind == "request"
    finish(m, job)
    m.request_deeper("r1", "start", 5)                        # the start line: scored first, then explained
    job, wait = m._next_job()
    assert job.kind != "request" and m.deeper_requested == {"r1": ("start", 5)}
    order = []
    while job is not None:
        order.append((job.kind, job.seq, job.ops))
        finish(m, job)
        job, wait = m._next_job()
    assert order[0] == ("live", 3, ("qc",)) and ("start", None, ("qc_start",)) in order
    assert ("request", None, ("deeper",)) in order and order.index(("request", None, ("deeper",))) == \
        order.index(("start", None, ("qc_start",))) + 1, "explained as soon as the start line is done"
    # No start line to come (a run started from nuclei): dropped, its failure handed over after the lock.
    nuclei = start_run(tmp_path, "nuclei", meta={"sample_kind": "zarr", "prior": {"cells": None,
                                                                                  "nuclei": {"kind": "shapes"}}},
                       state="done")
    nuclei.add_frame("end")
    m.runs.runs["nuclei"] = nuclei
    m.configure_qc("nuclei", enabled=True)
    while (job := m._next_job()[0]) is not None:              # its newest frame (the start line: "none")
        finish(m, job)
    m.request_deeper("nuclei", "start", 0)
    assert m._next_job()[0] is None and "nuclei" not in m.deeper_requested
    assert m.deeper_dropped == [{"run": "nuclei", "seq": "start", "cell": 0, "model": "feedc0de99",
                                 "error": "this run has no start line", "reason": "no_start"}]
    m.deeper_dropped.clear()
    # The check switched off before the request was served: Go deeper needs it.
    m.request_deeper("r1", 1, 3)
    m.configure_qc("r1", enabled=False)
    assert m._next_job()[0] is None and not m.deeper_requested
    assert [(d["run"], d["reason"]) for d in m.deeper_dropped] == [("r1", "off")]
    # A deleted run's request and following go with it.
    m.configure_qc("r1", enabled=True)
    m.request_deeper("r1", 1, 3, follow=True)
    del m.runs.runs["r1"]
    m._next_job()
    assert "r1" not in m.deeper_requested and "r1" not in m.deeper_follow and "r1" in m.to_forget


# ---- what one job does --------------------------------------------------------------------------------------------------
def test_the_provider_is_asked_first_then_the_block_goes_with_the_frame_and_the_answer_comes_back(tmp_path):
    run = counted(tmp_path, frames=3)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, model=("0123456789abcdef", model))
    deeper = Deeper()
    m.attach_deeper(deeper)
    calls = []
    deeper_stub(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    order = []
    m.qc_provider.qc_result = lambda record: order.append(("qc", record["seq"]))
    deeper.deeper_result = lambda record: (order.append(("deeper", record["seq"])), deeper.records.append(record))
    want = {"cell": 7, "start": False, "explicit": True}
    assert m._process(Job("r1", 1, ("qc", "deeper"), "request", qc, deeper=dict(want))) is None
    request = calls[-1]
    assert request["ops"] == ["qc", "deeper"] and request["cells_file"] == str(run.live / "frames/000001.cells.zst")
    assert request["counts_file"] == str(run.count_path(1)) and "init_counts_file" not in request   # a paired start
    assert request["deeper"] == {"model": model, "model_id": "0123456789abcdef", "cell": 7, "start": False,
                                 "k": DEEPER_K, "fit_min": 20, "judge_min": JUDGE,
                                 "codes": str(qc.dir / "000001.qc.bin"), "cell_ids": str(run.live / "cells.txt"),
                                 "v": DEEPER_V}
    assert deeper.asked == [("r1", 1, 7, "0123456789abcdef", True)]
    assert order == [("qc", 1), ("deeper", 1)], "the card's numbers first, then the window's"
    record = deeper.records[-1]
    assert {k: record[k] for k in ("run", "seq", "iter", "cell", "model", "explicit")} == \
        {"run": "r1", "seq": 1, "iter": 1, "cell": 7, "model": "0123456789abcdef", "explicit": True}
    assert record["result"]["cell"] == 7 and "error" not in record
    # Scored already, and the window's answer kept by the provider: no worker call at all.
    deeper.cached.add(("r1", 1, 7))
    count = len(calls)
    job = Job("r1", 1, ("deeper",), "request", qc, deeper=dict(want))
    assert m._process(job) is None and len(calls) == count and job.deeper_done
    # A live pass with a followed cell: one request, the frame counted once for the three.
    job = Job("r1", 2, ("umap", "qc", "deeper"), "live", qc, deeper={"cell": 3, "start": False, "explicit": False})
    assert m._process(job) is None
    assert calls[-1]["ops"] == ["umap", "qc", "deeper"] and calls[-1]["deeper"]["cell"] == 3
    assert deeper.records[-1]["explicit"] is False and m.results["r1"][2] == {"seq": 2}
    # The frame's bytes: named when they exist (or the same request writes them), else not.
    job = Job("r1", 0, ("deeper",), "request", qc, deeper=dict(want))
    m._process(job)
    assert calls[-1]["deeper"]["codes"] is None
    # A model without its details (deep.npz): no worker call; the window hears why.
    (Path(model) / "deep.npz").unlink()
    count = len(calls)
    job = Job("r1", 0, ("deeper",), "request", qc, deeper=dict(want))
    assert m._process(job) is None and len(calls) == count
    assert deeper.records[-1]["reason"] == "needs_detail" and "result" not in deeper.records[-1]


def test_a_failure_never_fails_the_pass_and_a_followed_cell_backs_off(tmp_path):
    run = counted(tmp_path, frames=3)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, model=("0123456789abcdef", model))
    deeper = Deeper()
    m.attach_deeper(deeper)
    now = [100.0]
    m.clock = lambda: now[0]
    m.follow_deeper("r1", 3)                                  # a window follows cell 3
    calls = []
    deeper_stub(m, calls, fail=True)
    qc = m._qc_state("r1", m._qc_models())
    qc.start, qc.backfill = "none", []                        # live passes only here
    follow = {"cell": 3, "start": False, "explicit": False}
    assert m._process(Job("r1", 1, ("umap", "qc", "deeper"), "live", qc, deeper=dict(follow))) is None
    assert m.results["r1"][1] == {"seq": 1} and 1 in qc.done, "Clusters & UMAP and the check keep their results"
    assert deeper.records[-1]["error"] == "ValueError: deep.npz failed its self-test here"
    assert m.deeper_retry["r1"] == (100.0 + DEEPER_RETRY_S[0], 1)
    assert any("Go deeper failed for frame 1, cell 3" in text for _, _, text in m.events)
    m.follow_deeper("r1", 3)                                  # the same cell again: its back-off stays
    run.add_frame("request")
    job, _ = m._next_job()
    assert job.kind == "live" and "deeper" not in job.ops, "backing off: the live pass goes without the cell"
    m._process(job)
    now[0] += DEEPER_RETRY_S[0]
    run.add_frame("request")
    job, _ = m._next_job()
    assert "deeper" in job.ops
    m._process(job)
    assert m.deeper_retry["r1"] == (now[0] + 2 * DEEPER_RETRY_S[0], 2), "doubling"
    m.request_deeper("r1", 1, 3)                              # an explicit request still goes
    job, _ = m._next_job()
    assert job.kind == "request" and job.ops == ("deeper",) and m.deeper_follow == {"r1": 3}
    deeper_stub(m, calls)
    m._process(Job("r1", 2, ("deeper",), "live", qc, deeper=dict(follow)))
    assert "r1" not in m.deeper_retry and "result" in deeper.records[-1], "an answer ends the back-off"
    # An older worker that drops the op: the window hears it, nothing waits 5 minutes.
    deeper_stub(m, calls, missing=True)
    m._process(Job("r1", 2, ("deeper",), "request", qc, deeper={"cell": 1, "start": False, "explicit": True}))
    assert deeper.records[-1]["error"] == "the GPU worker sent no result"
    # The worker dies mid-pass: the failure goes over once (the loop's _failed does not repeat it).
    def gone(request):
        raise AnalysisError("the GPU worker stopped")
    m._call = gone
    job = Job("r1", 2, ("deeper",), "request", qc, deeper={"cell": 1, "start": False, "explicit": True})
    count = len(deeper.records)
    with pytest.raises(AnalysisError):
        m._process(job)
    m._failed(job, job.ops, "the GPU worker stopped")
    assert len(deeper.records) == count + 1 and deeper.records[-1]["reason"] == "worker"
    # A run without exported transcripts: said once, as the check says it.
    (run.live / "transcripts.bin.gz").rename(run.live / "moved.bin.gz")
    job = Job("r1", 0, ("deeper",), "request", qc, deeper={"cell": 1, "start": False, "explicit": True})
    assert m._process(job) is not None and deeper.records[-1]["reason"] == "no_transcripts"


def test_a_go_deeper_answer_on_another_count_than_asked_is_refused(tmp_path):
    """Go deeper on a frame read on f (counted for a pause: the count proseg saves) by a worker that read another count
    (an older one that knows no `count` reads the averaged one): the answer is refused, the window told why (reason
    count_mismatch), never handed over as the frame's answer, as the check's own result is refused."""
    run = counted(tmp_path, frames=2)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, model=("0123456789abcdef", model))
    deeper = Deeper()
    m.attach_deeper(deeper)
    calls, sent = [], []
    deeper_stub(m, calls)
    current = m._call

    def older(request):                                       # reads e whatever it is asked
        sent.append(request.get("count"))
        return current({key: value for key, value in request.items() if key != "count"})
    m._call = older
    run.meta["state"] = "paused"
    paused = run.add_frame("pause")                           # 2: read on f
    qc = m._qc_state("r1", m._qc_models())
    want = {"cell": 4, "start": False, "explicit": True}
    m._process(Job("r1", paused, ("qc", "deeper"), "request", qc, deeper=dict(want)))
    assert sent == ["sampled"]
    record = deeper.records[-1]
    assert "result" not in record and record.get("reason") == "count_mismatch"
    assert (record["seq"], record["cell"]) == (2, 4)
    assert record["error"] == "the worker read the averaged count, not the sampled one asked for"
    assert paused not in qc.done and m.qc_status["r1"] == {"state": "error", "seq": 2, "error": record["error"],
                                                           "count": "sampled"}


def test_the_start_line_is_explained_on_proseg_s_count_at_iteration_0(tmp_path):
    run = counted(tmp_path, frames=3, start=True)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, model=("0123456789abcdef", model))
    deeper = Deeper()
    m.attach_deeper(deeper)
    calls = []
    deeper_stub(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    assert m._process(Job("r1", None, ("qc_start",), "start", qc)) is None    # the start line (stub: start.bin)
    job = Job("r1", None, ("deeper",), "request", qc, deeper={"cell": 2, "start": True, "explicit": True})
    assert m._process(job) is None
    request = calls[-1]
    assert request["op"] == "analyze" and request["ops"] == ["deeper"] and request["seq"] is None
    assert "frame" not in request and request["ncells"] == 3000 and request["z_spacing"] == 0.5
    assert request["counts_file"] == str(run.live / "frames" / "000000.counts.zst")   # proseg's count at iteration 0
    assert request["prior"] == {"file": str(run.live / "prior.bin.zst"), "format": "dense", **GEOMETRY}
    assert request["deeper"]["start"] is True and request["deeper"]["codes"] == str(qc.dir / "start.bin")
    assert request["deeper"]["start_file"] == str(qc.dir / "start.npz")
    assert deeper.asked[-1] == ("r1", "start", 2, "0123456789abcdef", True)
    assert deeper.records[-1]["seq"] == "start" and deeper.records[-1]["iter"] is None
    # Later frames pair with the start like for like: frame 0's count and the start block of their own.
    m._process(Job("r1", 1, ("qc",), "live", qc))
    assert calls[-1]["init_counts_file"] == request["counts_file"] and "prior" not in calls[-1]["qc"]
    assert calls[-1]["start_counts_file"] == str(run.count_path(1))
    # Frame 0's count gone (a run folder emptied by hand): no start line to explain, said, never sent.
    count = len(calls)
    run.count_path(0).unlink()
    job = Job("r1", None, ("deeper",), "request", qc, deeper={"cell": 2, "start": True, "explicit": True})
    assert m._process(job) is None and len(calls) == count and deeper.records[-1]["reason"] == "no_start"


def test_go_deeper_through_a_worker_process(tmp_path, monkeypatch):
    """The protocol end to end with the scheduler's own thread: a worker process (a stand-in that writes and answers
    what the worker does for the check and the start line, and answers op "deeper" from the request block) gets the
    requests, each naming proseg's count of its frame, the answers reach the provider, and a followed cell rides along
    the live pass."""
    worker = tmp_path / "workers" / "deeper_worker.py"
    worker.parent.mkdir()
    worker.write_text(textwrap.dedent('''
        import json, sys
        from pathlib import Path
        print(json.dumps({"ok": True, "ready": True}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("op") == "forget":
                continue
            ops = ["qc_start"] if request.get("op") == "qc_start" else request.get("ops") or []
            reply = {"ok": True, "run": request["run"], "seq": request.get("seq")}
            if "qc" in ops or "qc_start" in ops:
                Path(request["qc"]["out_cells"]).write_bytes(b"\\0")
                reply["qc"] = {"kind": "start" if "qc_start" in ops else "frame",
                               "counts_file": request.get("counts_file")}
            if "deeper" in ops:
                block = request["deeper"]
                reply["deeper"] = {"v": block["v"], "run": request["run"], "seq": request.get("seq"),
                                   "state": "ready", "cell": block["cell"], "start": block["start"],
                                   "cells_file": request.get("cells_file"), "counts_file": request.get("counts_file"),
                                   "ms": 0.5}
            print(json.dumps(reply), flush=True)
    '''))
    monkeypatch.setenv("PYTHONPATH", str(worker.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = counted(tmp_path, frames=4, state="done", start=True)
    deep_model(tmp_path / "model")
    provider = Provider(model=("c0ffee1234", str(tmp_path / "model")))
    config = {"analysis": {"python": sys.executable, "worker_module": "deeper_worker"},
              "paths": {"state_dir": str(tmp_path / "state")},
              "qc": {"judge_min_counts": JUDGE, "fit_min_counts": 5, "bin_um_detail": 8}}
    m = AnalysisManager(config, CountsRuns(run), start=False)
    m.attach_qc(provider)
    deeper = Deeper()
    m.attach_deeper(deeper)
    m.thread.start()
    try:
        m.configure_qc("r1", enabled=True)
        wait_for(lambda: m.describe_qc("r1")["start"] == "done" and len(m.describe_qc("r1")["seqs"]) == 4, timeout=60)
        m.request_deeper("r1", "start", 6, follow=True)
        wait_for(lambda: any(r.get("seq") == "start" for r in deeper.records), timeout=30)
        m.request_deeper("r1", 2, 5)
        wait_for(lambda: len(deeper.records) == 2, timeout=30)
    finally:
        m.close()
        m.thread.join(5)
    start, frame = deeper.records
    assert start["seq"] == "start" and start["cell"] == 6 and start["model"] == "c0ffee1234" and start["explicit"]
    assert start["result"]["start"] is True and start["result"]["seq"] is None and start["iter"] is None
    assert start["result"]["counts_file"] == str(run.live / "frames" / "000000.counts.zst")
    assert frame["seq"] == 2 and frame["cell"] == 5 and frame["iter"] == 2
    assert frame["result"]["cells_file"] == str(run.live / "frames/000002.cells.zst")
    assert frame["result"]["counts_file"] == str(run.count_path(2))
    assert {r["seq"]: r["summary"]["counts_file"] for r in provider.records if r["kind"] == "frame"} == \
        {seq: str(run.count_path(seq)) for seq in range(4)}
    assert m.deeper_follow == {"r1": 6}, "followed from the first request (the second left it)"


def test_explicit_go_deeper_jobs_let_one_umap_request_through_between_them(tmp_path):
    """Scrubbing with the window open asks for a Go deeper answer and a UMAP view per frame; the window's jobs go
    first, but after each one a waiting UMAP request goes, so the always-on UMAP pane keeps up (at least every other job)."""
    run = counted(tmp_path, frames=8, state="done")
    m = checked(tmp_path, run)
    order = []
    for seq in range(2, 7):                                   # each step: a UMAP request and a Go deeper request
        m.configure("r1", seq=seq)
        m.request_deeper("r1", seq, 5)
        job, _ = m._next_job()
        order.append(("deeper" if job.deeper else "umap", job.seq))
        finish(m, job)
    assert [kind for kind, _ in order] == ["deeper", "umap", "deeper", "umap", "deeper"], order
    assert ("umap", 3) in order, "the UMAP pane gets the frames scrubbed, not only the last"
    job, _ = m._next_job()                                    # the last step's UMAP request
    assert job.deeper is None and job.seq == 6
    finish(m, job)
    # Without a UMAP request waiting, explicit Go deeper jobs follow one another.
    m.request_deeper("r1", 6, 1)
    job, _ = m._next_job()
    assert job.deeper is not None
    finish(m, job)
    m.request_deeper("r1", 5, 2)
    job, _ = m._next_job()
    assert job.deeper is not None and job.seq == 5


def test_a_follow_nobody_renews_ends_by_itself(tmp_path):
    """A followed cell is a lease (the window renews it every 30 s while open on a live run): a window closed before
    its answer came, or a page that reloaded or closed, no longer leaves every pass explaining a cell nobody looks at."""
    run = counted(tmp_path, frames=2)
    m = checked(tmp_path, run)
    t = [100.0]
    m.clock = lambda: t[0]
    m.follow_deeper("r1", 4)
    assert m.deeper_follow == {"r1": 4}
    t[0] += A.DEEPER_FOLLOW_TTL_S - 1
    m.follow_deeper("r1", 4)                                  # renewed (the same cell): the lease starts again
    t[0] += A.DEEPER_FOLLOW_TTL_S - 1
    m._next_job()
    assert m.deeper_follow == {"r1": 4}
    t[0] += 2
    m._next_job()
    assert m.deeper_follow == {} and "r1" not in m.deeper_follow_at, "ended: nobody renewed it"
    m.request_deeper("r1", 1, 3, True)                        # an explicit request with follow renews it too
    assert m.deeper_follow == {"r1": 3} and m.deeper_follow_at["r1"] == t[0]
    m.stop_deeper("r1")
    assert m.deeper_follow == {} and "r1" not in m.deeper_follow_at


def test_a_follow_and_its_box_wait_out_a_rest_for_the_first_token_after_resume(tmp_path):
    """The page renews neither a followed cell nor its where box while the run rests (paused, or held at the end of its
    schedule: its ack "paused"), so both leases start again at every look then: a rest longer than both keeps them and
    no token goes, and the first token after Resume asks for the box the resting frame's reply armed (its where export:
    the first reply after Resume is exact). Once the run plays, both leases run out as before."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    run.add_frame("request")                                  # 1
    m = checked(tmp_path, run, clock=clock)
    box = [110.0, 60.0, 130.5, 80.25]
    m.follow_deeper("r1", 5)
    rest = m.runs.pause("r1")                                 # 2: the paused state, counted ("pause")
    run.ack = {"state": "paused", "iter": rest, "seq": rest}
    m.follow_where("r1", box)                                 # the resting frame's reply arms it (deeper_where _follow)
    for _ in range(3):                                        # resting far longer than both leases
        clock.now += max(A.DEEPER_FOLLOW_TTL_S, A.WHERE_BOX_TTL_S) + 1.0
        job, _ = m._next_job()
        if job is not None:
            finish(m, job)
        assert ask(m) == {} and m.runs.wheres == []
    assert m.deeper_follow == {"r1": 5} and m.where_box == {"r1": box}
    run.meta["state"] = "running"                             # Resume
    run.ack = {"state": "running", "iter": rest, "seq": rest}
    m._next_job()
    ask(m)
    assert m.runs.wheres[0] == ("r1", box, 0), "the first token after Resume asks for the box"
    clock.now += max(A.DEEPER_FOLLOW_TTL_S, A.WHERE_BOX_TTL_S)
    m._next_job()
    assert m.deeper_follow == {} and m.where_box == {}, "playing: nobody renewed them"


def test_a_run_that_ended_while_resting_keeps_no_lease(tmp_path):
    """A run stopped or done while paused keeps proseg's last ack ("paused"), but it no longer rests: a followed cell
    and its where box run out as on a playing run (no lease is kept for a run that ended)."""
    for ended in ("stopped", "done"):
        clock = Clock()
        run = CountedRun(tmp_path / ended, clock=clock)
        m = checked(tmp_path / ended, run, clock=clock)
        m.follow_deeper("r1", 5)
        m.follow_where("r1", [110.0, 60.0, 130.5, 80.25])
        run.meta["state"] = ended
        run.ack = {"state": "paused", "iter": 0, "seq": 0}
        clock.now += max(A.DEEPER_FOLLOW_TTL_S, A.WHERE_BOX_TTL_S) + 1.0
        m._next_job()
        assert m.deeper_follow == {} and m.where_box == {}, ended


def test_the_block_carries_the_classes_last_shown_and_the_start_line_for_the_like_for_like_change(tmp_path):
    """A frame's block names the neighbours' classes the provider kept (same_prev) and, for a run that started from
    cells alone (its frame 0 counted), asks for the cell's like-for-like change since the start (start_line: the
    request names frame 0's count and the start block the frame pairs with; no prior, nothing recounted)."""
    paired = counted(tmp_path, "paired", frames=3)            # cells with their nuclei: no start line
    run = counted(tmp_path, frames=3, start=True)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, paired, model=("0123456789abcdef", model))

    class Kept(Deeper):
        def deeper_prev(self, run_id, cell, model):
            return {"4": True, "9": False}

        def deeper_start_prev(self, run_id, cell, model):
            return "more"
    deeper = Kept()
    m.attach_deeper(deeper)
    calls = []
    deeper_stub(m, calls)
    want = {"cell": 7, "start": False, "explicit": True}
    pqc = m._qc_state("paired", m._qc_models())
    m._process(Job("paired", 1, ("qc", "deeper"), "request", pqc, deeper=dict(want)))
    block = calls[-1]["deeper"]
    assert block["same_prev"] == {"4": True, "9": False} and "start_line" not in block, "no start line for this start"
    assert "start_prev" not in block and "init_counts_file" not in calls[-1], "no word since the start without one"
    qc = m._qc_state("r1", m._qc_models())
    assert qc.start == "pending"
    m._process(Job("r1", 1, ("deeper",), "request", qc, deeper=dict(want)))
    request = calls[-1]
    block = request["deeper"]
    assert block["start_line"] is True and "prior" not in block, "its start block, before start.npz is there"
    assert request["init_counts_file"] == str(run.count_path(0))
    assert request["start_counts_file"] == str(run.count_path(1))
    assert block["start_prev"] == "more", "the cell's word since the start as last shown"
    Kept.deeper_start_prev = lambda self, run_id, cell, model: None
    m._process(Job("r1", 2, ("qc", "deeper"), "request", qc, deeper=dict(want)))
    assert calls[-1]["deeper"]["start_line"] is True and "start_prev" not in calls[-1]["deeper"], "none shown: none sent"
    qc.start = "done"
    m._process(Job("r1", None, ("deeper",), "request", qc, deeper={**want, "start": True}))
    assert "start_line" not in calls[-1]["deeper"], "the start line is not compared with itself"


def test_a_frame_the_run_rests_on_asks_for_the_cells_own_word_since_the_start(tmp_path):
    """A frame read on the count proseg saves (counted for a pause, the hold, a finish or the end: the run rests there)
    sends no word last shown (start_prev), followed or asked for: the worker's word there is its band rule alone, never
    one kept from earlier answers, so its answer is never held (a change of -3 points against a band of 5 after "fewer"
    was shown reads "same" there, where the word last shown would have kept "fewer", held). A frame read on the averaged
    count still sends it."""
    from sopa_companion.analysis import worker as W
    run = counted(tmp_path, frames=2, start=True)
    model = deep_model(tmp_path / "model")
    m = checked(tmp_path, run, model=("0123456789abcdef", model))

    class Kept(Deeper):
        def deeper_start_prev(self, run_id, cell, model):
            return "fewer"
    m.attach_deeper(Kept())
    calls = []
    deeper_stub(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    for explicit in (False, True):
        want = {"cell": 7, "start": False, "explicit": explicit}
        m._process(Job("r1", 1, ("deeper",), "request" if explicit else "live", qc, deeper=dict(want)))
        assert calls[-1]["count"] == "averaged" and calls[-1]["deeper"]["start_prev"] == "fewer", explicit
    run.meta["state"] = "paused"
    paused = run.add_frame("pause")
    for explicit in (False, True):
        want = {"cell": 7, "start": False, "explicit": explicit}
        m._process(Job("r1", paused, ("deeper",), "request" if explicit else "live", qc, deeper=dict(want)))
        request = calls[-1]
        assert request["count"] == "sampled" and request["deeper"]["start_line"] is True, explicit
        assert "start_prev" not in request["deeper"], explicit
    assert W.start_word(-3.0, 5.0, request["deeper"].get("start_prev")) == ("same", False)
    assert W.start_word(-3.0, 5.0, "fewer") == ("fewer", True)
