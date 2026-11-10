"""Live analysis without a GPU: the scheduler's cadence (back to back by default, or automatic analyses spaced by
min_interval_s; explicit requests at once; no polling while idle) and the worker's pure helpers (view bounds,
centroids, tiled decode, transcript lookups, warm starts, stable cluster ids), its GPU memory rules (cupy's pool for
the check whatever the allocator, freed blocks given back) and its idle-time UMAP warm-up; the mixing check: the
calculator against torch's reference vectors, model exports, gene matching, codes and per-bin sums against cell-by-cell
references, the worker's QC ops on the CPU on proseg's own count of each frame (its counts file: never the outlines;
no_counts without one), and the op-aware scheduler (ops per request, cadence, start line,
backfill, cached results op by op, errors, the provider's records), also through the worker process
(tests/fake_analysis_worker.py)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
import time
import types

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.engine.runs import ACTIVE
from sopa_companion.analysis import worker as W
from sopa_companion.engine.analysis import (BACKFILL_MAX, BINS_PREFIX, IDLE_LOOK_S, POLL_S, QC_LARGE_INTERVAL_S,
                                         QC_MODEL_POLL_S, SETTINGS, START_ATTEMPTS, AnalysisError, AnalysisManager, Job,
                                         auto_interval, backfill_plan, coarse_to_fine, min_interval_setting, model_key,
                                         qc_settings, transient_dir)


COUNTED = str(Path(__file__).resolve())      # a file that is there: every fake frame names it as its proseg count


def counted_frame(seq: int) -> dict:
    """A frame record as engine/runs.py merges a counts.jsonl line into it (proseg counted it: frame 0 by itself)."""
    return {"seq": seq, "iter": seq, "counts_file": COUNTED,
            "counts": {"seq": seq, "reason": "init" if seq == 0 else "request", "has_start": False, "bytes": 0,
                       "ms": {"extract": 0.0, "write": 0.0}}}


# A run that started from cells alone (engine/qc_rules.py start_of "cells"): the only start with a start line.
CELLS_START = {"sample_kind": "zarr", "sample": "z0", "prior": {"cells": {"kind": "shapes", "key": "cells"}}}


class FakeRun:
    def __init__(self, run_id="r1", cells=40070, frames=1, state="running", directory=None):
        self.lock = threading.RLock()
        self.frames = [counted_frame(s) for s in range(frames)]
        self.info = {"cells": cells}
        self.meta = {"id": run_id, "state": state}
        self.dir = Path(directory) if directory else Path("/nonexistent/sopa-companion-tests") / run_id

    @property
    def live(self):
        return self.dir / "live"

    @property
    def counted(self):
        """The counted frames' seqs in order, as Run keeps them (`counted`)."""
        with self.lock:
            return [s for s, frame in enumerate(self.frames) if frame.get("counts_file")]

    def add_frame(self):
        with self.lock:
            self.frames.append(counted_frame(len(self.frames)))


class FakeRuns:
    def __init__(self, *runs):
        self.runs = {r.meta["id"]: r for r in runs}

    def get(self, run_id):
        return self.runs[run_id]

    def active(self):
        return [r for r in self.runs.values() if r.meta.get("state") in ACTIVE]


def manager(tmp_path, *runs, qc=None, **settings):
    """A manager whose scheduler a test starts itself, often with a stub for its worker (`_call`): no worker is started
    while a run prepares (prestart), so nothing but the stub ever answers."""
    config = {"analysis": {"python": sys.executable, **settings}, "paths": {"state_dir": str(tmp_path)}}
    if qc is not None:
        config["qc"] = qc
    return AnalysisManager(config, FakeRuns(*runs), start=False, prestart=False)


# ---- scheduling ------------------------------------------------------------------------------------------------
def test_min_interval_setting_and_the_automatic_value():
    assert min_interval_setting("auto") is None and min_interval_setting(None) is None and min_interval_setting(" Auto ") is None
    assert min_interval_setting(0) == 0 and min_interval_setting(1.5) == 1.5 and min_interval_setting("2") == 2.0
    for bad in ("fast", -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="min_interval_s"):
            min_interval_setting(bad)
    assert auto_interval(40070) == pytest.approx(2.671, abs=1e-3)        # a whole sample: one update every ~2.7 s
    assert auto_interval(2000) == 0.5 and auto_interval(None) == 0.5 and auto_interval(10 ** 6) == 6.0


def test_by_default_analyses_run_back_to_back(tmp_path):
    run = FakeRun(cells=40070, frames=3)
    m = manager(tmp_path, run)                              # no min_interval_s: as fast as the GPU worker goes
    now = [100.0]
    m.clock = lambda: now[0]
    assert m.min_interval == 0 and m.configure("r1", enabled=True)["interval_s"] == 0
    assert m._next_job() == (("r1", 2), None)
    m._store("r1", 2, {"seq": 2})
    run.add_frame()
    run.add_frame()
    assert m._next_job() == (("r1", 4), None)               # the next starts at once, on the newest frame
    m._store("r1", 4, {"seq": 4})
    assert m._next_job() == (None, POLL_S) and POLL_S <= 0.1   # caught up: a newer frame is picked up within 0.1 s


def test_automatic_analyses_wait_for_the_interval_but_requests_do_not(tmp_path):
    run = FakeRun(cells=40070, frames=3)
    m = manager(tmp_path, run, min_interval_s="auto")
    now = [100.0]
    m.clock = lambda: now[0]
    assert m.describe("r1")["enabled"] and m.describe("r1")["interval_s"] == pytest.approx(2.67)   # always on
    assert m._next_job() == (("r1", 2), None)               # the first automatic analysis starts at once (newest frame)
    m._store("r1", 2, {"seq": 2})
    run.add_frame()
    now[0] += 1.0
    job, wait = m._next_job()
    assert job is None and wait == pytest.approx(auto_interval(40070) - 1.0)
    m.configure("r1", seq=0)                                # scrubbing: never delayed by the cadence
    assert m._next_job() == (("r1", 0), None)
    m._store("r1", 0, {"seq": 0})
    assert m._next_job()[0] is None                         # ... and it does not use up the automatic slot either
    run.add_frame()
    now[0] += 1.7
    assert m._next_job() == (("r1", 4), None)               # due: the newest frame; frame 3 was skipped, not queued
    m._store("r1", 4, {"seq": 4})
    assert m._next_job() == (None, POLL_S)                  # caught up while sampling: look again shortly
    run.meta["state"] = "done"
    assert m._next_job() == (None, None)                    # an ended run gets no new frames: no polling


def test_fixed_interval_from_the_config(tmp_path):
    small, big = FakeRun("small", cells=100), FakeRun("big", cells=10 ** 6)
    m = manager(tmp_path, small, big, min_interval_s=1.5)
    assert m.interval(small) == m.interval(big) == 1.5 and m.describe("big")["interval_s"] == 1.5
    m = manager(tmp_path, small, min_interval_s=0)          # 0: back to back, as before
    now = [10.0]
    m.clock = lambda: now[0]
    m.configure("small", enabled=True)
    assert m._next_job() == (("small", 0), None)
    m._store("small", 0, {"seq": 0})
    small.add_frame()
    assert m._next_job() == (("small", 1), None)
    with pytest.raises(ValueError):
        manager(tmp_path, small, min_interval_s="soon")


def test_scheduler_sleeps_until_there_is_work(tmp_path):
    run = FakeRun(cells=2, frames=5)
    m = manager(tmp_path, run, min_interval_s=0.4)
    m.configure("r1", enabled=False)                        # switched off by a tool: nothing to do
    calls, started = [], []
    next_job = m._next_job
    m._next_job = lambda *models: (calls.append(time.monotonic()), next_job(*models))[1]

    def analyse(run_id, seq):
        started.append((time.monotonic(), seq))
        m._store(run_id, seq, {"seq": seq})
    m._analyse = analyse
    m.thread.start()
    time.sleep(0.5)
    assert len(calls) <= 2                                  # idle: no polling (it used to wake every 50 ms)
    m.configure("r1", enabled=True)
    wait_for(lambda: started, timeout=2)
    stop = threading.Event()

    def sample():                                           # proseg: a new frame every 10 ms
        while not stop.is_set():
            run.add_frame()
            time.sleep(0.01)
    threading.Thread(target=sample, daemon=True).start()
    time.sleep(0.6)
    asked = time.monotonic()
    m.configure("r1", seq=1)
    wait_for(lambda: any(seq == 1 for _, seq in started), timeout=2)
    assert next(t for t, seq in started if seq == 1) - asked < 0.3   # a scrub request goes at once
    time.sleep(1.2)
    stop.set()
    automatic = [t for t, seq in started if seq != 1]
    assert len(automatic) >= 4 and min(np.diff(automatic)) >= 0.4 - 0.03
    m.close()
    m.thread.join(2)
    assert not m.thread.is_alive()                          # close() wakes the sleeping scheduler


def simulate(m, runs, seconds, took, fails=()):
    """_loop on a fake clock: every run gets a frame every 0.1 s; an analysis of run r takes took[r] seconds, and one
    of a run in `fails` fails that fast and is followed by the loop's 0.5 s back-off. Analyses started per run."""
    now, next_frame, counts = [0.0], 0.0, {r.meta["id"]: 0 for r in runs}
    m.clock = lambda: now[0]
    for r in runs:
        m.configure(r.meta["id"], enabled=True)
    while now[0] < seconds:
        while next_frame <= now[0]:
            for r in runs:
                r.add_frame()
            next_frame += 0.1
        job, wait = m._next_job()
        if job is None:
            now[0] = min(now[0] + wait, next_frame) if wait is not None else next_frame
            continue
        run_id, seq = job
        counts[run_id] += 1
        now[0] += took[run_id]
        if run_id in fails:
            m._set_status(run_id, {"state": "error", "seq": seq, "error": "only 12 cells have >= 10 transcripts"})
            now[0] += 0.5
        else:
            m._store(run_id, seq, {"seq": seq})
    return counts


def test_two_enabled_runs_share_the_worker(tmp_path):
    # A small region whose analyses take (or fail within) its whole 0.5 s interval is due again whenever the worker is
    # free; a whole sample enabled after it must still be analysed about once per its own interval (2.7 s).
    small, whole = FakeRun("small", cells=5000), FakeRun("whole", cells=40070)
    counts = simulate(manager(tmp_path, small, whole, min_interval_s="auto"), [small, whole], 60,
                      {"small": 0.55, "whole": 0.6})
    assert counts["whole"] >= 18 and counts["small"] >= 70, counts
    small, whole = FakeRun("small", cells=15), FakeRun("whole", cells=40070)
    counts = simulate(manager(tmp_path, small, whole, min_interval_s="auto"), [small, whole], 60,
                      {"small": 0.02, "whole": 0.6}, fails={"small"})
    assert counts["whole"] >= 18, counts
    small, whole = FakeRun("small", cells=15), FakeRun("whole", cells=40070)   # the default, 0: still shared
    counts = simulate(manager(tmp_path, small, whole), [small, whole], 60, {"small": 0.02, "whole": 0.6}, fails={"small"})
    assert counts["whole"] >= 40, counts
    a, b = FakeRun("a", cells=40070), FakeRun("b", cells=40070)  # 0 = back to back: they take turns
    counts = simulate(manager(tmp_path, a, b, min_interval_s=0), [a, b], 30, {"a": 0.6, "b": 0.6})
    assert abs(counts["a"] - counts["b"]) <= 1 and counts["a"] >= 20, counts


def test_a_refused_configure_changes_nothing(tmp_path):
    run = FakeRun(cells=2000, frames=3, state="done")      # ended: no automatic pass unless switched on
    m = manager(tmp_path, run)
    version, events = m.version, list(m.events)
    names = "resolution, n_neighbors, min_dist, min_counts and n_top_genes"
    for request, status, words in (
            ({"enabled": True, "seq": 99}, 404, "Unknown frame"), ({"enabled": True, "seq": "first"}, 422, None),
            ({"enabled": True, "params": {"resolution": "high"}}, 422, "resolution must be a number"),
            ({"enabled": True, "params": ["resolution"]}, 422,
             "params must be an object of settings (resolution, n_neighbors, min_dist, min_counts, n_top_genes)"),
            ({"params": {"n_neighbours": 20}}, 422, f"unknown setting 'n_neighbours' (the settings are {names})"),
            ({"params": {"n_neighbors": 15.5}}, 422, "n_neighbors must be a whole number"),
            ({"params": {"n_neighbors": True}}, 422, "n_neighbors must be a number"),
            ({"params": {"resolution": 0}}, 422, "resolution must be from 0.1 to 3"),
            ({"params": {"resolution": "0.5"}}, 422, "resolution must be a number"),    # strings are refused now
            ({"params": {"resolution": float("nan")}}, 422, "resolution must be a number"),
            ({"params": {"min_counts": 1e9}}, 422, "min_counts must be from 1 to 500"),
            ({"params": {"n_top_genes": 10050}}, 422, "n_top_genes must be from 50 to 10000"),
            ({"params": {"resolution": 2, "x" * 60: 1}}, 422, f"unknown setting '{'x' * 40}' (the settings are {names})"),
            ({"params": {"resolution": 2, "min_dist": 5}}, 422, "min_dist must be from 0 to 1")):
        with pytest.raises(AnalysisError) as refused:
            m.configure("r1", **request)
        assert refused.value.status == status
        if words:
            assert str(refused.value) == words
    assert "r1" not in m.enabled and not m.requested and not m.run_params and m.version == version
    assert m.gen == {} and m.prefs == {} and list(m.events) == events
    assert m._next_job() == (None, None)
    m.configure("r1", enabled=True, seq="1", params={"resolution": 0.5})   # the same, valid: applied and woken
    assert m.enabled["r1"] and m.requested == {"r1": 1} and m.run_params == {"r1": {"resolution": 0.5}}
    assert m.gen == {"r1": 1} and m.version == version + 1


def test_the_scheduler_survives_a_huge_interval_and_its_own_errors(tmp_path):
    run = FakeRun(cells=2000, frames=3)
    m = manager(tmp_path, run, min_interval_s=1e10)        # "never again automatically": beyond threading.TIMEOUT_MAX
    started, next_job, broken = [], m._next_job, [True]

    def analyse(run_id, seq):
        started.append(seq)
        m._store(run_id, seq, {"seq": seq})

    def flaky(*models):
        if broken[0]:
            broken[0] = False
            raise RuntimeError("a run in an unexpected state")
        return next_job(*models)
    m._analyse, m._next_job = analyse, flaky
    m.thread.start()
    m.configure("r1", enabled=True)
    wait_for(lambda: started == [2], timeout=3)            # the first automatic analysis, after the failed check
    run.add_frame()
    time.sleep(0.4)                                         # due in 1e10 s: the scheduler sleeps (at most an hour)
    assert m.thread.is_alive() and started == [2]
    m.configure("r1", seq=0)                                # a scrub is still served at once
    wait_for(lambda: started == [2, 0], timeout=3)
    m.close()
    m.thread.join(2)
    assert not m.thread.is_alive()


def test_clusters_and_umap_are_on_for_every_active_run_without_a_switch(tmp_path):
    """The page has no Clusters & UMAP switch: every active run has its newest frame analysed from its first frame on,
    an ended run gets the frames asked for (and automatic passes only when switched on explicitly), a tool can still
    switch a run off, and without the GPU worker's environment nothing is started at all."""
    live, old, new = FakeRun("live", cells=3000, frames=3), FakeRun("old", cells=3000, frames=5, state="done"), \
        FakeRun("new", cells=3000, frames=0, state="preparing")
    new.info = None
    m = manager(tmp_path, live, old, new)
    assert all(m.describe(r)["enabled"] for r in ("live", "old", "new")) and not m.enabled   # on, never switched
    job, wait = m._next_job()
    assert job == ("live", 2) and job.ops == ("umap",) and job.kind == "live" and wait is None
    m._store("live", 2, {"seq": 2})
    assert m._next_job() == (None, POLL_S)                  # caught up; "new" still prepares: looked at again shortly
    new.info, new.frames = {"cells": 3000}, [counted_frame(0)]
    job, _ = m._next_job()
    assert job == ("new", 0) and job.kind == "live"          # its first frame (the prior) at once
    m._store("new", 0, {"seq": 0})
    m.configure("old", seq=1)                                # a frame of an ended run, as the page asks while scrubbing
    job, _ = m._next_job()
    assert job == ("old", 1) and job.kind == "request"
    m._store("old", 1, {"seq": 1})
    for name in ("live", "new"):
        m.configure(name, enabled=False)                     # a tool (the harness) switches them off
    assert m._next_job() == (None, None) and not m.describe("live")["enabled"]
    assert m.describe("live")["status"] == {"state": "idle"}   # a status the WebSocket pushes: the page follows the switch
    assert [text for _, run_id, text in m.events if run_id == "live"] == ["Clusters & UMAP switched off through the API"]
    m.configure("old", enabled=True)                         # an ended run switched on: its newest frame, once
    job, _ = m._next_job()
    assert job == ("old", 4) and job.kind == "live"
    m._store("old", 4, {"seq": 4})
    assert m._next_job() == (None, None)                     # ended: no new frames, no polling
    bare = manager(tmp_path, FakeRun("bare", cells=3000, frames=3))
    bare.python = str(tmp_path / "no-rapids" / "python")     # no GPU worker environment: nothing is started
    assert not bare.describe("bare")["available"] and bare._next_job() == (None, None)


def test_a_new_run_is_found_without_anyone_waking_the_scheduler(tmp_path):
    """Nobody tells the scheduler that the engine created a run: with nothing to do it looks for one every IDLE_LOOK_S
    (a run prepares for seconds before its first frame), and without the GPU worker's environment it does not look."""
    m = manager(tmp_path)
    started, calls, next_job = [], [], m._next_job
    m._next_job = lambda *models: (calls.append(time.monotonic()), next_job(*models))[1]
    m._analyse = lambda run_id, seq: (started.append((run_id, seq)), m._store(run_id, seq, {"seq": seq}))
    m.thread.start()
    time.sleep(0.2)
    m.runs.runs["late"] = FakeRun("late", cells=2, frames=2)
    wait_for(lambda: started, timeout=IDLE_LOOK_S + 2)
    assert started[0] == ("late", 1)
    m.close()
    m.thread.join(2)
    bare = manager(tmp_path)
    bare.python = str(tmp_path / "no-rapids" / "python")
    looks, bare_next = [], bare._next_job
    bare._next_job = lambda *models: (looks.append(1), bare_next(*models))[1]
    bare.thread.start()
    time.sleep(IDLE_LOOK_S + 0.5)
    assert len(looks) == 1                                   # one look at start, then asleep until woken
    bare.close()
    bare.thread.join(2)


def test_the_log_gets_each_failure_once_and_every_result_with_its_timings(tmp_path):
    run = FakeRun(cells=3000, frames=6)
    m = manager(tmp_path, run)
    for _ in range(3):                                        # every pass fails the same way: noted once
        m._set_status("r1", {"state": "error", "seq": 2, "error": "only 12 cells have >= 10 transcripts"})
    m._store("r1", 3, {"seq": 3, "iter": 30, "ms": {"load": 6.0, "umap": 106.0}})
    m._set_status("r1", {"state": "error", "seq": 4, "error": "only 12 cells have >= 10 transcripts"})   # news again
    m._store("r1", 1, {"seq": 1, "iter": 10, "ms": {"load": 5.0}})
    info = m.log_info("r1")
    assert [text for _, run_id, text in info["events"]] == ["Clusters & UMAP failed for frame 2: only 12 cells have >= 10 "
                                                             "transcripts"] + ["Clusters & UMAP failed for frame 4: only 12 "
                                                                               "cells have >= 10 transcripts"]
    assert [seq for seq, _ in info["run"]["results"]] == [1, 3] and info["run"]["results"][1][1]["ms"]["umap"] == 106.0
    assert info["run"]["enabled"] and not info["run"]["explicit"] and info["run"]["interval_s"] == 0
    assert info["worker"]["pid"] is None and info["available"] and info["params"] == {}
    # The settings runs start from (the worker's own, [analysis] setting none here) and those new runs start with.
    assert info["defaults"] == {k: W.DEFAULTS[k] for k in ("resolution", "n_neighbors", "min_dist", "min_counts",
                                                          "n_top_genes")}
    assert info["new_runs"] == info["defaults"] and info["fixed"] == {}
    assert info["run"]["settings"] == info["defaults"] and not info["run"]["changed"] and info["run"]["generation"] == 0
    assert m.log_info("other")["events"] == [] and "run" not in m.log_info()


def test_the_log_reads_the_events_while_the_scheduler_notes_more(tmp_path):
    """_note appends from the scheduler's thread without the lock (the worker starting, failing, timing out): reading
    the events for the log meanwhile must never raise 'deque mutated during iteration', which made the log's GPU
    worker part say it could not be read exactly when something had just failed."""
    m = manager(tmp_path, FakeRun())
    for i in range(300):
        m._note(f"event {i}")
    stop, problems = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            m._note(f"more {i}", "r1" if i % 2 else None)
            i += 1
    thread = threading.Thread(target=writer, daemon=True)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                                 # switch threads as often as possible
    try:
        thread.start()
        for _ in range(2000):
            try:
                m.log_info("r1")
            except RuntimeError as exc:
                problems.append(str(exc))
                break
    finally:
        stop.set()
        thread.join(5)
        sys.setswitchinterval(interval)
    assert problems == []


def test_too_few_cells_holds_automatic_passes_back_until_the_run_holds_more(tmp_path):
    """A small region (a 60 x 60 µm rectangle: ~20 cells) can never be clustered: every new frame failed again, a GPU pass
    each (~2 a second) and a failure in the UMAP pane for the whole run. The worker names the reason (too_few_cells):
    the run then gets no automatic pass until its newest frame holds materially more cells, the page hears why
    (describe's `unavailable`, the status's `reason`), explicit requests still go, and a result clears it. A run without
    exported transcripts says so the same way."""
    run = disk_run(tmp_path / "run")
    for f in run.frames:
        f["cells"] = 20
    m = manager(tmp_path, run)
    calls = []

    def call(request):
        calls.append(request)
        message = "TooFewCells: only 20 cells have >= 10 transcripts in this frame"
        return {"run": request["run"], "seq": request["seq"], "ok": False, "errors": {"umap": message}, "error": message,
                "reasons": {"umap": "too_few_cells"}}
    m._call = call
    assert m.describe("r1")["unavailable"] is None
    job, _ = m._next_job()
    assert job == ("r1", 2) and m._process(job).startswith("TooFewCells:")
    assert m.status["r1"]["reason"] == "too_few_cells" and m.describe("r1")["unavailable"] == "too_few_cells"
    def more(seq, cells):                                       # counted as the frame before (its line merged)
        run.frames.append({**run.frames[-1], "seq": seq, "iter": seq, "cells": cells})
        run.counted.append(seq)
    more(3, 22)
    more(4, 26)                                                 # more frames of the same small region
    assert m._next_job() == (None, POLL_S) and len(calls) == 1  # held back: no pass for them
    m.configure("r1", seq=1)                                    # a frame asked for (scrubbing) still goes
    assert m._next_job()[0] == ("r1", 1)
    more(5, 40)                                                 # cells formed: materially more
    job, _ = m._next_job()
    assert job == ("r1", 5) and "r1" not in m.too_few and m.describe("r1")["unavailable"] is None
    m._process(job)                                             # still too few: held back again
    more(6, 41)
    assert m._next_job() == (None, POLL_S)
    m._store("r1", 6, {"seq": 6})                               # a result after all (another request): cleared
    assert "r1" not in m.too_few and m.describe("r1")["unavailable"] is None
    legacy = FakeRun("legacy", cells=3000, frames=2, state="done", directory=tmp_path / "legacy")
    old = manager(tmp_path, legacy)
    assert old.describe("legacy")["unavailable"] == "no_transcripts"
    assert old._process(Job("legacy", 1, ("umap",), "request")).startswith("This run has no exported transcripts")
    assert old.status["legacy"]["reason"] == "no_transcripts"


# ---- worker helpers --------------------------------------------------------------------------------------------
def test_layout_bounds_and_cluster_centroids():
    x = np.arange(101, dtype=np.float32)
    layout = np.stack([x, -2 * x], axis=1)
    assert W.layout_bounds(layout) == {"x": [2.0, 98.0], "y": [-196.0, -4.0]}   # 2nd and 98th percentiles per axis
    with_unanalysed = np.vstack([layout, [[np.nan, np.nan], [1e6, np.nan]]])
    assert W.layout_bounds(with_unanalysed) == W.layout_bounds(layout)
    wire = np.zeros((101, 3), np.float32)                   # the wire format's 3 components: z is ignored
    wire[:, :2] = layout
    assert W.layout_bounds(wire) == W.layout_bounds(layout) and W.layout_bounds(np.full((3, 2), np.nan)) is None
    clusters = np.array([2, 0, 2, 0, 2, 5])
    points = np.array([[0, 0], [1, 1], [2, 4], [3, 3], [10, 8], [np.nan, 1]], np.float32)
    centroids = W.cluster_centroids(clusters, points)
    assert sorted(centroids) == [0, 2, 5] and centroids[5] is None
    assert centroids[2].tolist() == [2.0, 4.0] and centroids[0].tolist() == [2.0, 2.0]


def write_container(path, planes: dict, tile: int):
    """proseg-live's tile container (native/proseg/src/live.rs): tiles row-major, planes fastest, then the window."""
    import zstandard
    names = list(planes)
    ny, nx = planes[names[0]].shape
    tiles_x, tiles_y = -(-nx // tile), -(-ny // tile)
    blobs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            for name in names:
                block = np.ascontiguousarray(planes[name][ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile], "<u4")
                blobs.append(zstandard.ZstdCompressor(level=1).compress(block.tobytes()) if block.any() else b"")
    blobs.append(b"")                                       # no window
    header = json.dumps({"nx": nx, "ny": ny, "nz": 4, "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
                         "planes": names, "window": None}).encode()
    offset, head = 8 + len(header) + 12 * len(blobs), b"PLT1" + struct.pack("<I", len(header)) + header
    for blob in blobs:
        head += struct.pack("<QI", offset, len(blob))
        offset += len(blob)
    path.write_bytes(head + b"".join(blobs))


@pytest.mark.parametrize("threads", [0, 3])
def test_tiled_frames_decode_only_their_tissue_and_find_every_transcript(tmp_path, threads):
    rng = np.random.default_rng(1)
    tile, nx, ny = 8, 30, 21                                # partial edge tiles in both directions
    def frame(name, empty):
        consensus = rng.integers(1, 50, (ny, nx)).astype(np.uint32)
        for rows, cols in empty:
            consensus[rows, cols] = 0
        write_container(tmp_path / name, {"consensus": consensus, "last_change": rng.integers(0, 9, (ny, nx))}, tile)
        return consensus
    empty = [(slice(0, 8), slice(0, 16)), (slice(16, None), slice(24, None))]   # two full tiles and the corner tile
    first = frame("a.tiles", empty)
    pool = ThreadPoolExecutor(threads) if threads else None
    header, present, buffer = W.plane_tiles(str(tmp_path / "a.tiles"), "consensus", None, pool)
    ntiles = header["tiles_x"] * header["tiles_y"]
    assert (header["tiles_x"], header["tiles_y"]) == (4, 3) and len(present) == ntiles - 3
    assert np.array_equal(W.read_tiled_consensus(str(tmp_path / "a.tiles")), first)

    # Transcripts anywhere, some outside the grid: each gets the label of its column, 0 outside.
    x0, y0, voxel = 10.0, -5.0, 2.0
    x = rng.uniform(x0 - 10, x0 + nx * voxel + 10, 5000).astype(np.float32)
    y = rng.uniform(y0 - 10, y0 + ny * voxel + 10, 5000).astype(np.float32)
    ids, offsets = W.tile_lookup(np, x, y, x0, y0, voxel, nx, ny, tile, header["tiles_x"], header["tiles_y"])
    i, j = np.floor((x - x0) / voxel).astype(int), np.floor((y - y0) / voxel).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    assert inside.any() and not inside.all()

    def labels(present, buffer):
        tiles = np.zeros((1 + len(present), tile, tile), np.uint32)
        tiles[1:] = buffer[:len(present)]
        return W.tile_labels(np, tiles, present, ntiles, ids, offsets)

    def expected(consensus):
        return np.where(inside, consensus[np.clip(j, 0, ny - 1), np.clip(i, 0, nx - 1)], 0)
    assert np.array_equal(labels(present, buffer), expected(first))
    second = frame("b.tiles", empty)                        # the next frame reuses the buffer ...
    _, present2, buffer2 = W.plane_tiles(str(tmp_path / "b.tiles"), "consensus", buffer, pool)
    assert buffer2 is buffer and np.array_equal(labels(present2, buffer2), expected(second))
    third = frame("c.tiles", [])                            # ... until the tissue outgrows it
    _, present3, buffer3 = W.plane_tiles(str(tmp_path / "c.tiles"), "consensus", buffer, pool)
    assert len(present3) == ntiles and np.array_equal(labels(present3, buffer3), expected(third))
    if pool:
        pool.shutdown()


def test_dense_frames_find_each_transcripts_voxel():
    rng = np.random.default_rng(2)
    nx, ny, nz = 9, 7, 3
    layers = rng.integers(1, 20, (nz, ny, nx)).astype(np.uint32)
    x0, y0, voxel, z_origin, z_spacing = 1.0, 2.0, 1.5, 0.1, 0.3
    x = rng.uniform(x0 - 3, x0 + nx * voxel + 3, 4000).astype(np.float32)
    y = rng.uniform(y0 - 3, y0 + ny * voxel + 3, 4000).astype(np.float32)
    z = rng.uniform(-0.2, 1.2, 4000).astype(np.float32)
    index = W.voxel_lookup(np, x, y, z, x0, y0, voxel, nx, ny, nz, z_origin, z_spacing)
    i, j = np.floor((x - x0) / voxel).astype(int), np.floor((y - y0) / voxel).astype(int)
    k = np.clip(np.floor((z - z_origin) / z_spacing), 0, nz - 1).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    want = np.where(inside, layers[k, np.clip(j, 0, ny - 1), np.clip(i, 0, nx - 1)], 0)
    assert np.array_equal(np.append(layers.ravel(), 0)[index], want) and inside.any() and not inside.all()


def test_lookups_go_in_steps():
    """A grid's lookup (positions only: the where engine's and the settings test's places; the worker counts nothing
    from it) is computed LOOKUP_CHUNK transcripts at a time (its temporaries never hold a whole sample's 34 M at once),
    with the same result."""
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-5, 70, 1000).astype(np.float32), rng.uniform(-5, 50, 1000).astype(np.float32)
    z = rng.uniform(0, 1, 1000).astype(np.float32)
    whole = W.tile_lookup(np, x, y, 0.0, 0.0, 2.0, 30, 21, 8, 4, 3)
    steps = W.tile_lookup(np, x, y, 0.0, 0.0, 2.0, 30, 21, 8, 4, 3, chunk=7)
    assert all(np.array_equal(a, b) and b.dtype == np.int32 for a, b in zip(whole, steps))
    assert np.array_equal(W.voxel_lookup(np, x, y, z, 0.0, 0.0, 2.0, 30, 21, 3, 0.0, 0.4),
                          W.voxel_lookup(np, x, y, z, 0.0, 0.0, 2.0, 30, 21, 3, 0.0, 0.4, chunk=13))
    assert W.tile_lookup(np, x[:0], y[:0], 0.0, 0.0, 2.0, 30, 21, 8, 4, 3)[0].shape == (0,)
    assert not hasattr(W.Worker, "lookup") and not hasattr(W.Worker, "labels")    # no op looks a transcript up


def test_cached_blocks_go_back_once_cuml_takes_over_the_allocator():
    """Importing cuML (the first UMAP pass) switches cupy to RMM's allocator, which here does no caching: what cupy's
    own pool cached before (every earlier pass's transient high-water, runs dropped since) was never used again and
    held GPU 1 for the worker's life (3.6 GB on a whole sample). It goes back after each request from then on; while
    the pool is still the allocator its blocks are kept for reuse. A fake cupy: the worker's tests have no GPU."""
    class Pool:
        def __init__(self):
            self.used, self.free = 500 * 2 ** 20, 3000 * 2 ** 20

        def malloc(self, size):
            raise AssertionError("not called")

        def used_bytes(self):
            return self.used

        def free_bytes(self):
            return self.free

        def total_bytes(self):
            return self.used + self.free

        def free_all_blocks(self):
            self.free = 0
    pool, allocator = Pool(), []
    allocator.append(pool.malloc)
    worker = W.Worker.__new__(W.Worker)
    worker.xp = types.SimpleNamespace(get_default_memory_pool=lambda: pool,
                                      cuda=types.SimpleNamespace(get_allocator=lambda: allocator[0]))
    assert worker.pool_mb() == {"pool_used_mb": 500.0, "pool_total_mb": 3500.0, "allocator": "pool"}
    assert worker.release_stale_blocks() == 0 and pool.free == 3000 * 2 ** 20       # still cupy's: kept for reuse
    allocator[0] = lambda size: None                                                 # cuML loaded: RMM's allocator
    assert worker.pool_mb()["allocator"] == "rmm"
    assert worker.release_stale_blocks() == 3000 * 2 ** 20 and worker.pool_mb()["pool_total_mb"] == 500.0
    assert worker.release_stale_blocks() == 0
    numpy_worker = W.Worker(xp=np)
    assert numpy_worker.pool_mb() is None and numpy_worker.release_stale_blocks() == 0


def test_passes_of_the_check_alone_give_cached_blocks_back():
    """A run whose Clusters & UMAP stay off is scored by passes of the mixing check alone, while cupy's pool is still the
    allocator (cuML never loaded): each pass's transient high-water went back only at the worker's end and held GPU 1
    between passes. After a request without UMAP (and a dropped run) the cached blocks go back now; the reply says how
    much, and what the pool keeps between passes. A UMAP pass keeps them for reuse."""
    class Pool:
        def __init__(self):
            self.used, self.free = 600 * 2 ** 20, 900 * 2 ** 20

        def malloc(self, size):
            raise AssertionError("not called")

        def used_bytes(self):
            return self.used

        def free_bytes(self):
            return self.free

        def total_bytes(self):
            return self.used + self.free

        def free_all_blocks(self):
            self.free = 0
    pool = Pool()
    worker = W.Worker.__new__(W.Worker)
    worker.xp = types.SimpleNamespace(get_default_memory_pool=lambda: pool,
                                      cuda=types.SimpleNamespace(get_allocator=lambda: pool.malloc))
    worker.process = lambda request: {"qc": {"kind": "frame", "gpu": worker.pool_mb()}, "errors": {}}
    reply = worker.handle({"op": "analyze", "ops": ["umap", "qc"], "run": "r", "seq": 1})
    assert pool.free == 900 * 2 ** 20 and reply["qc"]["gpu"]["released_mb"] == 0.0     # UMAP: kept for reuse
    assert reply["qc"]["gpu"]["pool_kept_mb"] == 1500.0
    reply = worker.handle({"op": "analyze", "ops": ["qc"], "run": "r", "seq": 2})
    assert pool.free == 0 and reply["ok"] and reply["qc"]["gpu"] == {
        "pool_used_mb": 600.0, "pool_total_mb": 1500.0, "allocator": "pool", "released_mb": 900.0, "pool_kept_mb": 600.0}
    pool.free = 200 * 2 ** 20
    assert worker.release_stale_blocks(qc_only=True) == 200 * 2 ** 20 and pool.free == 0   # a run dropped ("forget")
    worker.process = lambda request: (_ for _ in ()).throw(ValueError("boom"))
    pool.free = 100 * 2 ** 20
    reply = worker.handle({"op": "qc", "run": "r", "seq": 3})                           # also when the pass failed
    assert pool.free == 0 and reply["ok"] is False and reply["errors"] == {"qc": "ValueError: boom"}


class FakeCupy:
    """Just enough of cupy's allocator API for the worker's memory rules: a pool, a global allocator (cupy's pool until
    cuML switches it to RMM) and a thread-local one set by using_allocator, which get_allocator returns first."""

    def __init__(self, rmm: bool):
        import contextlib

        class Pool:
            def __init__(self):
                self.used, self.free, self.freed = 300 * 2 ** 20, 0, []

            def malloc(self, size):
                raise AssertionError("not called")

            def used_bytes(self):
                return self.used

            def free_bytes(self):
                return self.free

            def total_bytes(self):
                return self.used + self.free

            def free_all_blocks(self):
                self.freed.append(self.free)
                self.free = 0
        self.pool, self.local = Pool(), None
        self.rmm = lambda size: None                                  # rmm_cupy_allocator, once cuML is imported
        self.global_allocator = self.rmm if rmm else self.pool.malloc
        fake = self

        @contextlib.contextmanager
        def using_allocator(allocator):
            fake.local = allocator
            try:
                yield
            finally:
                fake.local = None
        self.cuda = types.SimpleNamespace(get_allocator=lambda: fake.local or fake.global_allocator,
                                          using_allocator=using_allocator,
                                          Device=lambda: types.SimpleNamespace(synchronize=lambda: None))

    def get_default_memory_pool(self):
        return self.pool


def test_the_checks_arrays_come_from_cupys_pool_whatever_the_allocator():
    """Once a UMAP pass has run (or the warm-up), cuML has switched cupy to RMM's allocator, whose resource here
    caches nothing: every array of a pass of the mixing check alone was a cudaMalloc and every free a cudaFree waiting
    for the GPU (crop: counts 4.6 -> 7.5 ms, score 1.1 -> 3.7, bins 2.3 -> 4.2). The frame's counts and the check now
    take cupy's pool whatever the allocator; clusters and UMAP do not, and what the pooled steps freed goes back before
    the UMAP part, so it never sits beside cuML's arrays at the peak."""
    for rmm in (True, False):
        xp = FakeCupy(rmm)
        worker = W.Worker.__new__(W.Worker)
        worker.xp, worker.warm_steps, seen, refs = xp, ["step"], {}, {}

        def note(name):
            seen[name] = (xp.cuda.get_allocator() == xp.pool.malloc, worker.pool_mb()["allocator"])

        def count(run, request, watch=None):
            note("count")
            refs["read"] = types.SimpleNamespace(e=np.ones(3, np.float32))    # the frame's counts file, as read
            return "counts", refs["read"]

        def score_frame(run, request, counts, read, ms, kept=None):
            note("score")
            kept["totals"] = "totals"                               # handed on to Go deeper in the same pass
            xp.pool.free = 700 * 2 ** 20                           # what the pooled steps freed
            return {"kind": "frame", "gpu": worker.pool_mb()}

        def deeper(run, request, counts, kept):                    # Go deeper: after the check, before clusters
            note("deeper")
            seen["deeper_kept"] = kept.get("totals") == "totals" and kept["read"] is refs["read"] and "umap" not in seen
            return {"v": 1, "state": "ready"}

        def umap(run, request, counts, read, ms):
            note("umap")
            seen["dropped"] = counts == "counts" and read is refs["read"]
            seen["free_at_umap"] = xp.pool.free
            xp.pool.free = 50 * 2 ** 20
            return {"seq": request["seq"]}
        worker.state, worker.count = (lambda request: note("state") or "run"), count
        worker.score_frame, worker.umap, worker.deeper = score_frame, umap, deeper
        reply = worker.handle({"op": "analyze", "ops": ["umap", "deeper", "qc"], "run": "r", "seq": 7, "ncells": 3,
                               "deeper": {"cell": 0}})
        assert reply["ok"] and reply["summary"] == {"seq": 7} and xp.local is None
        assert reply["deeper"] == {"v": 1, "state": "ready"} and seen["deeper_kept"]
        outer = "rmm" if rmm else "pool"
        assert all(seen[k] == (True, outer) for k in ("state", "count", "score", "deeper")), seen   # pooled:
        assert seen["umap"] == (not rmm, outer) and seen["dropped"]                                # the outer allocator
        assert reply["qc"]["gpu"]["allocator"] == outer and worker.warm_steps == []    # a real pass: warm already
        if rmm:     # RMM: freed before the UMAP part and after the request; the reply counts both
            assert seen["free_at_umap"] == 0 and xp.pool.freed == [700 * 2 ** 20, 50 * 2 ** 20]
            assert reply["qc"]["gpu"]["released_mb"] == 750.0
        else:       # cuML not loaded yet: the UMAP part allocates from the pool too and reuses those blocks
            assert seen["free_at_umap"] == 700 * 2 ** 20 and xp.pool.freed == []
            assert reply["qc"]["gpu"]["released_mb"] == 0.0
    # The start line is pooled too; a failed UMAP part leaves the warm-up to do.
    worker.warm_steps, calls = ["step"], []
    worker._qc_start = lambda request: calls.append(xp.cuda.get_allocator() == xp.pool.malloc) or {"kind": "start"}
    assert worker.handle({"op": "qc_start", "run": "r", "qc": {}})["ok"] and calls == [True]
    worker.umap = lambda *args: (_ for _ in ()).throw(ValueError("only 3 cells"))
    reply = worker.handle({"op": "analyze", "ops": ["umap"], "run": "r", "seq": 8, "ncells": 3})
    assert reply["errors"] == {"umap": "ValueError: only 3 cells"} and worker.warm_steps == ["step"]
    numpy_worker = W.Worker(xp=np)
    with numpy_worker.pooled():                                      # a no-op on the CPU
        assert numpy_worker.outer_allocator is None


def test_lines_come_whole_and_the_stream_ends_cleanly():
    """The worker reads its requests from the descriptor itself, so select() sees every line that has not been taken
    (a buffered reader could hold a second request where select() cannot see it, and the warm-up would make it wait)."""
    r, w = os.pipe()
    with os.fdopen(r, "rb", buffering=0) as stream:
        lines = W.Lines(stream)
        os.write(w, b'{"a": 1}\n{"b"')
        assert lines.get(0.5) == '{"a": 1}'
        assert lines.get(0.05) is None                               # half a line is not a line
        os.write(w, b': 2}\n\xc2\xb5m\n')
        assert lines.get(0.5) == '{"b": 2}' and lines.get(0.5) == "µm"
        os.write(w, b"tail")
        os.close(w)
        assert lines.get(0.5) == "tail"
        with pytest.raises(EOFError):
            lines.get(0.5)


def test_the_worker_warms_umap_up_only_while_idle():
    """The first UMAP pass in a worker paid cuML's and cuGraph's start-up (~1.4 s: a whole sample's first pass
    1.9 s, and the mixing check's result riding on it waited as long). The worker now takes its warm-up steps while
    idle, one at a time and only after idle_s without a request, so requests that come back to back never wait for
    one, and a request waits for at most one step. A failing step ends the warm-up."""
    worker = W.Worker(xp=np)
    assert worker.warm_umap_steps() == [] and worker.warm_umap() == {}            # nothing to warm on the CPU
    fake = W.Worker.__new__(W.Worker)
    fake.xp = types.SimpleNamespace()
    assert [s.__name__ for s in fake.warm_umap_steps()] == ["import_cuml", "import_cugraph", "pca_neighbours",
                                                             "leiden", "layouts"]
    events, idle_s = [], 0.2

    def step(name, fail=False):
        def run():
            events.append((name, time.monotonic()))
            if fail:
                raise ImportError("no cuML here")
        run.__name__ = name
        return run
    worker.warm_steps = [step("first"), step("second", fail=True), step("never")]
    def handle(request):
        events.append((f"seq{request['seq']}", time.monotonic()))
        return {"ok": True, "seq": request["seq"]}
    worker.handle = handle
    r, w = os.pipe()
    replies_r, replies_w = os.pipe()
    out = os.fdopen(replies_w, "w")
    stream = os.fdopen(r, "rb", buffering=0)
    thread = threading.Thread(target=W.serve, args=(worker, stream, out, idle_s))
    thread.start()
    try:
        for seq in range(3):                                         # back to back: no warm-up step between them
            os.write(w, json.dumps({"op": "analyze", "seq": seq}).encode() + b"\n")
            time.sleep(idle_s / 4)
        wait_for(lambda: any(name == "second" for name, _ in events), timeout=10)
        os.write(w, b"not json\n" + json.dumps({"op": "forget", "run": "x"}).encode() + b"\n" +
                 json.dumps({"op": "unknown"}).encode() + b"\n" + json.dumps({"op": "qc", "seq": 3})[:9].encode())
        time.sleep(idle_s * 2)                                       # half a request: neither answered nor idle work
        os.write(w, json.dumps({"op": "qc", "seq": 3})[9:].encode() + b"\n")
        wait_for(lambda: any(name == "seq3" for name, _ in events), timeout=10)
    finally:
        os.close(w)
        thread.join(10)
    assert not thread.is_alive()                                     # the stream ended: serve returned
    names = [name for name, _ in events]
    assert names == ["seq0", "seq1", "seq2", "first", "second", "seq3"] and worker.warm_steps == []
    at = dict(events)
    assert at["first"] - at["seq2"] >= idle_s * 0.9 and at["second"] - at["first"] >= idle_s * 0.9
    out.close()
    with os.fdopen(replies_r) as replies:
        replies = [json.loads(line) for line in replies]
    # One reply per request but "forget": a line that is not JSON and an unknown op are answered with an error at
    # once (the engine reads one reply per request: a dropped one made it wait 300 s), never handled.
    assert [r["seq"] for r in replies] == [0, 1, 2, None, None, 3]
    assert replies[3]["ok"] is False and replies[3]["reasons"] == {"request": "unreadable"}
    assert replies[4] == {"ok": False, "run": None, "seq": None, "error": "ValueError: unknown op 'unknown'",
                          "errors": {"unknown": "ValueError: unknown op 'unknown'"}, "reasons": {"unknown": "unknown_op"}}
    stream.close()


def test_the_worker_keeps_the_runs_it_uses(tmp_path):
    """At most RUNS_KEPT runs' transcripts stay on the GPU, least recently used out first: the run followed live (used
    by every pass) is never pushed out by the start lines and backfill of runs that are only looked at."""
    worker = W.Worker(xp=np)
    paths = {}
    for name in ("live", "a", "b", "c", "d"):
        folder = write_run(tmp_path / name, frames=1)
        paths[name] = {"run": name, "transcripts": str(folder / "live" / "transcripts.bin.gz"),
                       "genes": str(folder / "live" / "genes.txt")}
    live = worker.state(paths["live"])
    for name in ("a", "b", "c", "d"):
        worker.state(paths[name])                    # a run looked at ...
        assert worker.state(paths["live"]) is live   # ... and the live run's next pass
    assert list(worker.runs) == ["b", "c", "d", "live"] and len(worker.runs) == W.RUNS_KEPT   # "a" went first


def test_new_cells_start_among_their_known_neighbours():
    previous = np.full((6, 2), np.nan, np.float32)
    previous[[0, 1, 2, 4]] = [[0, 0], [2, 0], [0, 2], [4, 4]]
    neighbours = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 0, 1], [4, 2, 1], [5, 3, 5]])
    init, warm = W.Worker.initial_embedding(types.SimpleNamespace(embedding=previous), np.arange(6), neighbours, 6)
    assert warm
    np.testing.assert_allclose(init[[0, 1, 2, 4]], previous[[0, 1, 2, 4]], atol=0.01)
    np.testing.assert_allclose(init[3], [1, 0], atol=0.01)            # the mean of its known neighbours 0 and 1
    np.testing.assert_allclose(init[5], [1.5, 1.5], atol=0.01)        # none known: the mean of every known cell


def test_clusters_keep_their_ids_across_updates(monkeypatch):
    def linear_sum_assignment(cost):                        # brute force stand-in: scipy is not in the test venv
        cost = np.asarray(cost)
        if cost.shape[0] > cost.shape[1]:
            cols, rows = linear_sum_assignment(cost.T)
            order = np.argsort(rows)
            return rows[order], cols[order]
        best = min(itertools.permutations(range(cost.shape[1]), cost.shape[0]),
                   key=lambda cols: cost[np.arange(len(cols)), list(cols)].sum())
        return np.arange(cost.shape[0]), np.array(best)
    scipy, optimize = types.ModuleType("scipy"), types.ModuleType("scipy.optimize")
    optimize.linear_sum_assignment, scipy.optimize = linear_sum_assignment, optimize
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)

    run = types.SimpleNamespace(clusters=None, next_cluster=0)
    keep = np.array([0, 1, 2, 3, 5])
    assert W.Worker.match_clusters(run, keep, np.array([0, 0, 1, 1, 2]), 6).tolist() == [0, 0, 1, 1, -1, 2]
    # Leiden numbers its clusters afresh each time: the same groups keep their ids (and colours) ...
    assert W.Worker.match_clusters(run, keep, np.array([5, 5, 3, 3, 9]), 6).tolist() == [0, 0, 1, 1, -1, 2]
    # ... and a new group gets a fresh id.
    clusters = W.Worker.match_clusters(run, np.arange(6), np.array([1, 1, 0, 0, 7, 4]), 6)
    assert clusters.tolist() == [0, 0, 1, 1, 3, 2] and run.next_cluster == 4


def test_cluster_colours_are_never_shared_on_screen_and_stay_put():
    """Cluster ids only grow as clusters come and go, so colour = id mod 40 repeated colours on screen (two clusters of
    the crop in one olive, #999933). Each cluster now takes a palette colour no other cluster on screen has: it keeps it
    while it stays, a new one gets the free colour most distinct from those shown (never a vanished one's within the
    grace period), and only beyond 40 clusters are colours shared."""
    fresh = lambda: types.SimpleNamespace(colours={}, freed={})
    run = fresh()
    first = W.assign_colours(run, list(range(12)), 0.0)
    assert len(set(first.values())) == 12 and first == W.assign_colours(fresh(), list(range(12)), 0.0)   # deterministic
    assert [first[c] for c in range(3)] == [0, 1, 2]                 # the palette's own order while colours are distinct
    shown = list(first.values())
    near = min(W.colour_distance(a, b) for a, b in itertools.combinations(shown, 2))
    assert near >= 15, near                                          # no near-twins: #44aa99 / #44bb99 is ~5
    worst = min(W.colour_distance(a, b) for a, b in itertools.combinations(range(40), 2))
    assert worst < 10                                                # the palette has them: picking around them matters
    second = W.assign_colours(run, [*range(11), 12], 1.0)            # cluster 11 went, 12 came
    assert all(second[c] == first[c] for c in range(11))             # those that stayed keep their colours
    assert second[12] not in [first[c] for c in range(11)] and second[12] != first[11]   # not the one just freed
    assert run.freed[first[11]] == 1.0 and 11 not in run.colours
    later = W.assign_colours(run, [*range(11), 12, 13], 1.0 + W.COLOUR_GRACE_S)
    assert later[13] not in [later[c] for c in [*range(11), 12]]    # still none shared
    crowded = fresh()
    W.assign_colours(crowded, list(range(39)), 0.0)
    W.assign_colours(crowded, [*range(38), 40], 0.5)                 # 38 went: its colour is the only one in grace ...
    taken = W.assign_colours(crowded, [*range(38), 40, 41], 0.6)    # ... and the only free one left for 41
    assert len(set(taken.values())) == 40
    many = W.assign_colours(fresh(), list(range(45)), 0.0)
    counts = sorted(list(many.values()).count(i) for i in range(40))
    assert len(set(many.values())) == 40 and counts[-1] == 2 and counts.count(2) == 5   # only the overflow shares
    # A long run whose clusters keep coming and going (9-18 on screen, a new one most results): the colours on screen
    # never come near-twins (the palette's closest pairs, #88ccee / #99ddff, are 6 apart), however many ids it has used.
    import random
    rng, churn, on, nxt, worst = random.Random(0), fresh(), list(range(12)), 12, math.inf
    for step in range(3000):
        if rng.random() < 0.6:
            on.remove(rng.choice(on)); on.append(nxt); nxt += 1
        if rng.random() < 0.2 and len(on) < 18:
            on.append(nxt); nxt += 1
        if rng.random() < 0.2 and len(on) > 9:
            on.remove(rng.choice(on))
        shown = list(W.assign_colours(churn, sorted(on), 0.2 * step).values())
        assert len(set(shown)) == len(shown)
        worst = min(worst, min(W.colour_distance(a, b) for a, b in itertools.combinations(shown, 2)))
    assert worst >= 20, worst


def test_the_workers_palette_is_the_viewers():
    """A summary's colour is an index into the page's palette (frontend/src/clusters.js): both lists must agree."""
    import re
    js = (Path(W.__file__).resolve().parents[1] / "frontend" / "src" / "clusters.js").read_text()
    block = js[js.index("CLUSTER_PALETTE"):js.index("]);")]
    assert tuple(re.findall(r"#[0-9a-f]{6}", block)) == W.PALETTE and len(W.PALETTE) == 40


def test_too_few_cells_and_a_deleted_run_are_named_not_traced(tmp_path, monkeypatch, capsys):
    """A small region's frame has too few cells to cluster: the reply names the reason (reasons: too_few_cells) and the
    worker's log gets no traceback for it. A run deleted while its frame was analysed is one plain line in the log, not
    a FileNotFoundError traceback that every later log would show."""
    import shutil
    folder = write_run(tmp_path / "run")
    live = folder / "live"
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[1])
    request = {"op": "analyze", "ops": ["umap"], "run": "r1", "seq": 1, "iter": 1, "frame": str(live / frame["file"]),
               "format": "dense", **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
               "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"), "ncells": NCELLS,
               "counts_file": str(live / "frames" / "000001.counts.zst"),
               "z_origin": 0.0, "z_spacing": 0.5, "out_json": str(folder / "a.json"), "out_cells": str(folder / "a.bin")}

    def umap(self, run, request, counts, read, ms):
        raise W.TooFewCells("only 12 cells have >= 10 transcripts in this frame")
    monkeypatch.setattr(W.Worker, "umap", umap)
    worker = W.Worker(xp=np)
    capsys.readouterr()
    reply = worker.handle(request)
    assert not reply["ok"] and reply["reasons"] == {"umap": "too_few_cells"}
    assert reply["errors"]["umap"] == "TooFewCells: only 12 cells have >= 10 transcripts in this frame"
    assert "Traceback" not in capsys.readouterr().err
    shutil.rmtree(folder)                                     # the run is deleted (its counts file with it)
    reply = worker.handle({**request, "seq": 2})
    assert not reply["ok"] and reply["reasons"] == {"umap": "run_deleted"}
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.strip() == "analysis: run r1 was deleted while frame 2 was being analysed: " \
                                                     "nothing kept"


# ---- the mixing check: the calculator, model exports, codes, per-bin sums ----------------------------------------------
SELFTEST = Path(__file__).with_name("data") / "qc_selftest.npz"      # torch's outputs: tests/data/make_qc_selftest.py
RUN_GENES = [f"G{i}" for i in range(10)] + ["NegControlProbe_1"]      # a run's genes.txt
MODEL_GENES = [f"G{i}" for i in range(1, 10)] + ["ONLY_IN_MODEL"]     # its model's: G0 and the probe are not in it


def make_model(path, genes, hidden=16, seed=0, **extra):
    """A small random calculator in the export's format; its reference outputs come from our own numpy calculator (the
    plumbing: torch parity is tests/data/qc_selftest.npz)."""
    rng = np.random.default_rng(seed)
    weights = {"W0": rng.normal(0, 0.6, (hidden, len(genes))), "b0": rng.normal(0, 0.1, hidden),
               "W1": rng.normal(0, 0.3, (hidden, hidden)), "b1": rng.normal(0, 0.1, hidden),
               "Wv": rng.normal(0, 1.5, (3, hidden)), "bv": np.array([1.0, -1.0, -3.0])}
    weights = {k: v.astype(np.float32) for k, v in weights.items()}
    counts = rng.poisson(rng.gamma(0.5, 4.0, (64, len(genes)))).astype(np.int32)
    calculator = {**weights, "var_eps": 1e-3, "diffusion_eps": 1e-2, "ln_eps": 1e-5}
    expected = W.mixing_weights(np, np.log1p(counts.astype(np.float32)) @ weights["W0"].T, calculator)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights, genes=np.array(genes), selftest_counts=counts, selftest_expected=expected, **extra)
    return Path(path)


def mixing_model(path, rows, genes=MODEL_GENES, seed=0, at=0.30):
    """make_model's calculator with the bias of its 3-way head set so that about half of `rows` (count rows over the
    model's genes) read `at` or more of their RNA as from neighbours: a check under which some cells look mixed and some
    do not, so its numbers move with the counts it reads. Its reference outputs are made again for that head."""
    make_model(path, genes, seed=seed)
    stored = dict(np.load(path))
    calculator = {**{k: stored[k] for k in W.ENCODER_LAYERS}, "var_eps": 1e-3, "diffusion_eps": 1e-2, "ln_eps": 1e-5}
    h = W.layer_norm_relu(np, np.log1p(np.asarray(rows, np.float32)) @ stored["W0"].T + stored["b0"], 1e-5)
    head = W.layer_norm_relu(np, h @ stored["W1"].T + stored["b1"], 1e-5) @ stored["Wv"].T
    lean = float(np.median(head[:, 1] - head[:, 0]))
    calculator["bv"] = np.array([0.0, math.log(at / (1 - at)) - lean, -8.0], np.float32)
    counts = stored["selftest_counts"].astype(np.float32)
    expected = W.mixing_weights(np, np.log1p(counts) @ stored["W0"].T, calculator)
    np.savez(path, **{**stored, "bv": calculator["bv"], "selftest_expected": expected})
    return Path(path)


def test_the_calculator_matches_torch_on_the_exported_reference_cells():
    # 256 cells through scviva's own diffusion encoder (torch 2.13, scvi-tools 1.5.1, CPU), and their raw weights.
    encoder = W.load_encoder(SELFTEST)
    counts, expected = encoder["selftest"]
    assert counts.shape == (256, 60) and expected.shape == (256, 3) and encoder["hidden"] == 128
    assert W.selftest_error(np, encoder) <= 1e-5
    weights = W.mixing_weights(np, np.log1p(counts) @ encoder["W0"].T, encoder)
    np.testing.assert_allclose(weights.sum(axis=1), 1.013, atol=1e-5)            # 1 + 3 x 0.001 + 0.01
    share = W.neighbour_share(weights)
    np.testing.assert_allclose(share, expected[:, 1] / expected.sum(axis=1), atol=1e-5)
    assert (share >= 0.3).any() and (share < 0.3).any()                           # the cells span the threshold
    # The same cells as a run counts them: its genes in another order, two the model lacks (dropped), and ten model
    # genes the run lacks (count 0). The first layer's weights are gathered into the run's order instead.
    rng = np.random.default_rng(3)
    run_genes = [f"G{i}" for i in rng.permutation(60)[:50]] + ["NegControlProbe_1", "Unknown"]
    index, key, matched = W.gene_index(encoder, run_genes)
    assert key == "genes" and matched == 50 and (index[-2:] == -1).all()
    run_counts = np.zeros((256, len(run_genes)), np.float32)
    run_counts[:, :50] = counts[:, index[:50]]
    run_counts[:, 50:] = rng.poisson(5, (256, 2))
    model_counts = np.zeros_like(counts)
    model_counts[:, index[:50]] = counts[:, index[:50]]
    via_run = W.mixing_weights(np, np.log1p(run_counts) @ W.first_layer(encoder, index), encoder)
    via_model = W.mixing_weights(np, np.log1p(model_counts) @ encoder["W0"].T, encoder)
    np.testing.assert_allclose(via_run, via_model, atol=1e-6)


def test_model_exports_are_read_strictly(tmp_path):
    stored = dict(np.load(SELFTEST))
    good = {k: stored[k] for k in W.ENCODER_LAYERS}
    counts, weights = stored["selftest_counts"], stored["selftest_expected"]

    def export(name, **arrays):
        np.savez(tmp_path / f"{name}.npz", **arrays)
        return tmp_path / f"{name}.npz"
    # The torch state_dict's own names (with the unused mean head) in a model folder, proportions as the outputs.
    (tmp_path / "model").mkdir()
    np.savez(tmp_path / "model" / "encoder.npz", genes=stored["genes"], selftest_counts=counts,
             selftest_expected=weights / weights.sum(axis=1, keepdims=True), layer_norm_eps=np.float32(1e-5),
             **{"_guide.diffusion_encoder." + W.ENCODER_LAYERS[k]: v for k, v in good.items()},
             **{"_guide.diffusion_encoder.mean_encoder.weight": np.zeros((3, 128), np.float32)})
    encoder = W.load_encoder(tmp_path / "model")
    assert all(np.array_equal(encoder[k], good[k]) for k in good) and W.selftest_error(np, encoder) <= 1e-5
    # Reference cells as CSR and outputs as neighbour shares; Ensembl ids beside the symbols (the better match wins).
    rows, cols = np.nonzero(counts)
    encoder = W.load_encoder(export("csr", **good, gene_symbols=stored["genes"],
                                    gene_ids=np.array([f"ENSG{i:011d}" for i in range(60)]),
                                    selftest_indptr=np.searchsorted(rows, np.arange(len(counts) + 1)),
                                    selftest_indices=cols, selftest_data=counts[rows, cols],
                                    selftest_expected=weights[:, 1] / weights.sum(axis=1)))
    assert sorted(encoder["genes"]) == ["gene_ids", "gene_symbols"] and W.selftest_error(np, encoder) <= 1e-5
    assert W.gene_index(encoder, ["ENSG00000000003", "G1", "G2"])[1:] == ("gene_symbols", 2)
    assert W.gene_index(encoder, ["ENSG00000000003", "ENSG00000000004", "G2"])[1:] == ("gene_ids", 2)
    # Anything that is not ResolVI's two-layer calculator is refused, never guessed at.
    for bad, problem in (({**good, "W2": good["W1"]}, "unexpected"),
                         ({**good, "encoder.fc_layers.Layer 2.0.weight": good["W1"]}, "unexpected"),
                         ({**good, "encoder.fc_layers.Layer 0.2.weight": np.ones(128, np.float32)}, "unexpected"),
                         ({**good, "encoder.fc_layers.Layer 0.1.running_mean": np.ones(128, np.float32)}, "unexpected"),
                         ({k: v for k, v in good.items() if k != "b1"}, "no b1"),
                         ({**good, "Wv": good["Wv"][:2]}, "Wv has shape"),
                         ({**good, "W1": np.full((128, 128), np.nan, np.float32)}, "non-finite"),
                         ({**good, "W0": good["W0"][None]}, "2-D")):
        with pytest.raises(ValueError, match=problem):
            W.load_encoder(export("bad", **bad, genes=stored["genes"]))
    with pytest.raises(ValueError, match="gene names"):
        W.load_encoder(export("nameless", **good))
    with pytest.raises(ValueError, match="self-test shapes"):
        W.load_encoder(export("misfit", **good, genes=stored["genes"], selftest_counts=counts[:, :10],
                              selftest_expected=weights))


def trainer_export(path, **changes):
    """The torch reference model written the way analysis/qc_trainer.py exports a fit: its FORMAT and exactly its
    ENCODER_KEYS (reference cells as CSR, selftest_indptr / selftest_indices / selftest_data, and the weights torch gave
    them, selftest_expected)."""
    from sopa_companion.analysis import qc_trainer as T
    stored = dict(np.load(SELFTEST))
    counts = stored["selftest_counts"].astype(np.float32)
    rows, cols = np.nonzero(counts)
    arrays = {"format": np.array(T.FORMAT), "genes": stored["genes"],
              "gene_ids": np.array([f"ENSG{i:011d}" for i in range(60)]), **{k: stored[k] for k in W.ENCODER_LAYERS},
              "ln_eps": np.float32(1e-5), "var_eps": np.float32(1e-3), "diffusion_eps": np.float32(0.01),
              "selftest_indptr": np.searchsorted(rows, np.arange(len(counts) + 1)).astype(np.int64),
              "selftest_indices": cols.astype(np.int32), "selftest_data": counts[rows, cols],
              "selftest_expected": stored["selftest_expected"].astype(np.float32)}
    assert set(arrays) == T.ENCODER_KEYS, "the trainer's export format changed: update the worker's loader and this test"
    np.savez(path, **{**arrays, **changes})
    return path, counts


def test_the_trainers_exports_are_read_as_it_writes_them(tmp_path):
    from sopa_companion.analysis import qc_trainer as T
    path, counts = trainer_export(tmp_path / "encoder.npz")
    assert T.FORMAT in W.FORMATS
    theirs = T.load_encoder(path)                            # the trainer's own strict reader takes this export ...
    assert T.self_test(theirs) <= 1e-5
    ours = W.load_encoder(path)                              # ... and so does the worker, reference cells included
    assert W.selftest_error(np, ours) <= 1e-5 and ours["selftest"][0].shape == (256, 60)
    np.testing.assert_allclose(W.mixing_weights(np, np.log1p(counts) @ ours["W0"].T, ours),
                               T.mixing_weights(theirs, counts), atol=1e-6)     # one calculator, two implementations
    assert W.gene_index(ours, ["G3", "ENSG00000000003"])[1:] == ("genes", 1)
    trainer_export(tmp_path / "future.npz", format=np.array("proseg-live-mixing-encoder/2"))
    with pytest.raises(ValueError, match="unknown calculator format"):
        W.load_encoder(tmp_path / "future.npz")
    trainer_export(tmp_path / "broken.npz", selftest_indices=np.full(len(counts.nonzero()[0]), 99, np.int32))
    with pytest.raises(ValueError, match="malformed reference cells"):
        W.load_encoder(tmp_path / "broken.npz")


def test_the_worker_checks_each_model_against_its_reference_cells(tmp_path):
    stored = dict(np.load(SELFTEST))
    np.savez(tmp_path / "ok.npz", **stored)
    worker = W.Worker(xp=np)
    encoder, _, stamp = worker.model(tmp_path / "ok.npz")
    assert encoder["selftest_error"] <= 1e-5 and worker.model(tmp_path / "ok.npz")[0] is encoder   # once per version
    np.savez(tmp_path / "bad.npz", **{**stored, "W1": stored["W1"] * 1.01})   # not the calculator its outputs came from
    with pytest.raises(ValueError, match="self-test"):
        worker.model(tmp_path / "bad.npz")
    np.savez(tmp_path / "untested.npz", **{k: v for k, v in stored.items() if not k.startswith("selftest")})
    assert worker.model(tmp_path / "untested.npz")[0]["selftest_error"] is None      # nothing to check it against


def test_one_byte_per_cell_and_the_threshold_agree():
    share = np.array([0.0, 0.0999, 0.1, 0.2999, 0.3, 0.75, 1.0, np.nan, 0.5], np.float32)
    judged = np.array([True] * 7 + [False, False])
    codes = W.share_codes(np, share, judged)
    assert codes.dtype == np.uint8 and codes.tolist() == [0, 24, 25, 74, 75, 187, 250, 255, 255]
    assert W.code_at(0.30) == 75 and W.code_at(0.20) == 50 and W.code_at(0.10) == 25
    # A cell "looks mixed" exactly when its byte says so: the colours and the headline count the same cells.
    rng = np.random.default_rng(0)
    share = rng.random(100000).astype(np.float32)
    codes = W.share_codes(np, share, np.ones(len(share), bool))
    assert np.array_equal(codes >= W.code_at(0.30), share >= np.float32(0.30))


def reference_tables(share, totals, qc, grid, stats, start):
    """qc_tables cell by cell, in plain Python: {bin: {field: value}} and the cells' growth ratios."""
    table, ratios = {}, []
    mixed_at = math.ceil(qc["mixed_at"] * 250 - 1e-9)
    for c in range(len(totals)):
        scored = totals[c] >= qc["fit_min"] and np.isfinite(share[c])
        judged = scored and totals[c] >= qc["judge_min"]
        code = min(250, int(np.floor(share[c] * 250))) if judged else 255
        mixed = judged and code >= mixed_at
        for x, y in ((stats["cx"][c], stats["cy"][c]), (start["cx"][c], start["cy"][c])):   # never a count's centroid
            if np.isfinite(x) and np.isfinite(y):
                break
        else:
            continue
        bx = min(max(int(np.floor((x - grid["x0"]) / grid["bin"])), 0), grid["nx"] - 1)
        by = min(max(int(np.floor((y - grid["y0"]) / grid["bin"])), 0), grid["ny"] - 1)
        row = table.setdefault(by * grid["nx"] + bx, {})

        def add(field, value=1):
            row[field] = row.get(field, 0) + value
        start_judged = start["code"][c] != 255
        start_mixed = start_judged and start["code"][c] >= mixed_at
        area, begun = float(stats["area_um2"][c]), float(start["area_um2"][c])
        for field, true in (("cells", True), ("scored", scored), ("judged", judged), ("mixed", mixed),
                            ("some", judged and not mixed and code >= 25), ("paired", judged and start_judged),
                            ("paired_mixed_now", judged and start_judged and mixed),
                            ("paired_mixed_start", judged and start_judged and start_mixed),
                            ("n10", judged and start_judged and mixed and not start_mixed),
                            ("n01", judged and start_judged and start_mixed and not mixed),
                            ("judged_start", start_judged), ("mixed_start", start_mixed),
                            ("tiny", area < 10), ("tiny_start", begun < 10), ("growth_n", begun > 0),
                            ("grown", begun > 0 and area / begun >= 1.1), ("shrunk", begun > 0 and area / begun < 0.9),
                            ("lost_half", begun > 0 and area < 0.5 * begun)):
            add(field, int(bool(true)))
        add("tx", float(totals[c]))
        add("tx_start", float(start["totals"][c]))
        add("changed", float(stats["changed"][c]))
        add("share_sum", float(share[c]) if judged else 0.0)
        if begun > 0:
            ratios.append(area / begun)
    return table, ratios


def test_per_bin_sums_match_a_cell_by_cell_reference():
    rng = np.random.default_rng(11)
    n, grid = 600, {"x0": 0.0, "y0": 0.0, "bin": 50.0, "nx": 6, "ny": 4}
    share = rng.random(n).astype(np.float32)
    share[rng.random(n) < 0.05] = np.nan                                  # not scored
    totals = (rng.integers(0, 400 * 32, n) / 32).astype(np.float32)      # proseg's averaged count: steps of 1/32

    def positions(missing):
        x, y = rng.uniform(-20, 320, n).astype(np.float32), rng.uniform(-20, 220, n).astype(np.float32)
        x[rng.random(n) < missing] = np.nan
        return x, y
    (cx, cy), (sx, sy) = positions(0.2), positions(0.5)
    stats = {"area_um2": rng.choice([0.0, 5.0, 40.0, 80.0, 120.0], n).astype(np.float32), "cx": cx, "cy": cy,
             "changed": rng.integers(0, 9, n).astype(np.uint32), "assigned": rng.integers(0, 500, n).astype(np.uint32),
             "foreground": rng.integers(0, 400, n).astype(np.uint32)}
    start = {"code": np.where(rng.random(n) < 0.2, 255, rng.integers(0, 251, n)).astype(np.uint8),   # the start block
             "totals": (rng.integers(0, 400 * 32, n) / 32).astype(np.float32), "cx": sx, "cy": sy,   # and footprints
             "area_um2": rng.choice([0.0, 8.0, 60.0, 80.0], n).astype(np.float32)}
    qc = {"mixed_at": 0.3, "judge_min": 100, "fit_min": 20}
    codes, table, summary = W.qc_tables(np, share, totals, qc, grid, stats, start)
    expected, ratios = reference_tables(share, totals, qc, grid, stats, start)
    assert sorted(table["bin"].tolist()) == sorted(expected)             # only bins holding a cell
    for k, b in enumerate(table["bin"].tolist()):
        for field in W.BIN_COUNTS:
            assert table[field][k] == expected[b][field], (b, field)
        for field in ("tx", "tx_start", "changed", "share_sum"):
            assert table[field][k] == pytest.approx(expected[b][field], rel=1e-5), (b, field)
    judged = (totals >= 100) & np.isfinite(share)
    assert summary["judged"] == judged.sum() and summary["mixed"] == (judged & (share >= np.float32(0.3))).sum()
    assert summary["per100"] == round(100 * summary["mixed"] / summary["judged"], 2)
    assert summary["located"] == sum(row["cells"] for row in expected.values()) < n   # some cells have no position
    # The whole run's numbers count every cell, placed or not.
    codes_now, start_judged = W.share_codes(np, share, judged), start["code"] != 255
    mixed, start_mixed = judged & (codes_now >= 75), start_judged & (start["code"] >= 75)
    assert summary["paired"] == {"n": (judged & start_judged).sum(), "mixed_now": (judged & start_judged & mixed).sum(),
                                 "mixed_start": (judged & start_judged & start_mixed).sum(),
                                 "n10": (judged & start_judged & mixed & ~start_mixed).sum(),
                                 "n01": (judged & start_judged & ~mixed & start_mixed).sum()}
    assert summary["growth"]["n"] == (start["area_um2"] > 0).sum()
    assert summary["growth"]["median"] == pytest.approx(np.median(stats["area_um2"][start["area_um2"] > 0] /
                                                                  start["area_um2"][start["area_um2"] > 0]), abs=1e-3)
    bands = np.searchsorted([250, 500, 1000], totals[judged], side="right")
    assert summary["bands"]["judged"] == np.bincount(bands, minlength=4).tolist()
    assert np.array_equal(table["judged_band"].sum(axis=1), table["judged"])
    assert np.array_equal(table["mixed_band"].sum(axis=1), table["mixed"])
    assert table["growth_hist"].sum() == table["growth_n"].sum() and table["growth_hist"].shape[1] == 13
    assert summary["codes"] == {"scale": 250, "not_judged": 255, "mixed": 75, "some": 25}
    assert summary["tx_sum"] == summary["transcripts"] == round(float(totals.astype(np.float64).sum()), 1)
    # Without the start line (a frame scored before it): no pairs, no growth, and cells are placed without it.
    codes2, table2, summary2 = W.qc_tables(np, share, totals, qc, grid, stats, None)
    assert np.array_equal(codes, codes2) and "paired" not in summary2 and "growth" not in summary2
    assert "n10" not in table2 and "growth_hist" not in table2 and summary2["judged"] == summary["judged"]
    assert summary2["located"] == int((np.isfinite(cx) & np.isfinite(cy)).sum())
    # The start's two parts apart: the start block alone pairs the cells; the footprints alone place them and grow.
    _, table3, summary3 = W.qc_tables(np, share, totals, qc, grid, stats, {k: start[k] for k in ("code", "totals")})
    assert summary3["paired"] == summary["paired"] and "growth" not in summary3 and "tiny_start" not in summary3
    assert table3["judged_start"].sum() == (start["code"][np.isfinite(cx) & np.isfinite(cy)] != 255).sum()
    _, table4, summary4 = W.qc_tables(np, share, totals, qc, grid, stats, {k: start[k] for k in ("area_um2", "cx", "cy")})
    assert "paired" not in summary4 and summary4["growth"] == summary["growth"] and "tx_start" not in table4
    assert summary4["located"] == summary["located"] and summary4["tiny_start"] == summary["tiny_start"]


class CupyLike:
    """numpy with cupy's stricter edges, to meet them without a GPU: bincount refuses an empty input (cupy reduces it
    with a max that has no identity)."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def bincount(values, weights=None, minlength=0):
        if not len(values):
            raise ValueError("zero-size array to reduction operation CUPY_CUB_MAX which has no identity")
        return np.bincount(values, weights, minlength)


class GpuLike(CupyLike):
    """numpy dressed as cupy, to run the GPU worker's own code paths without a GPU (Worker(xp=GpuLike(),
    backend="gpu")): cupy's stricter bincount (CupyLike), its pool, allocator and device (FakeCupy's), and no RawKernel,
    so Go deeper's row kernel cannot compile and its rows come through cupyx's row indexing, as on a GPU without NVRTC.
    Its matrices come from fake_cupyx's csr_matrix."""

    def __init__(self):
        fake = FakeCupy(rmm=False)
        self.cuda, self.pool = fake.cuda, fake.pool

    def get_default_memory_pool(self):
        return self.pool


def fake_cupyx(monkeypatch, csr_matrix):
    """cupyx.scipy.sparse for the GPU worker's matrices (Worker.wrap's cupyx branch), with `csr_matrix` its CSR class
    (called as cupyx's: csr_matrix((data, indices, indptr), shape=...)), for this test only."""
    modules = [types.ModuleType(name) for name in ("cupyx", "cupyx.scipy", "cupyx.scipy.sparse")]
    modules[2].csr_matrix = csr_matrix
    modules[0].scipy, modules[1].sparse = modules[1], modules[2]
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)


def test_empty_categories_count_as_zero_as_on_the_gpu():
    assert W.tally(np, np.array([], np.int64), 3).tolist() == [0, 0, 0]
    assert W.tally(np, np.array([2, 2]), 3, np.array([0.5, 1.0])).tolist() == [0.0, 0.0, 1.5]
    n, grid = 50, {"x0": 0.0, "y0": 0.0, "bin": 10.0, "nx": 3, "ny": 3}
    share = np.zeros(n, np.float32)                          # nobody looks mixed: no mixed cell, no discordant pair
    totals = np.full(n, 150, np.float32)
    position = np.linspace(0, 29, n).astype(np.float32)
    stats = {"area_um2": np.full(n, 50, np.float32), "cx": position, "cy": position}
    start = {"code": np.zeros(n, np.uint8), "totals": np.full(n, 150, np.int32), "cx": position, "cy": position,
             "area_um2": np.full(n, 50, np.float32)}
    for xp in (np, CupyLike()):
        codes, table, summary = W.qc_tables(xp, share, totals, {"mixed_at": 0.3, "judge_min": 100, "fit_min": 20},
                                            grid, stats, start)
        assert summary["mixed"] == 0 and summary["paired"]["n10"] == 0 and table["mixed"].sum() == 0
        assert summary["bands"]["mixed"] == [0, 0, 0, 0] and table["lost_half"].sum() == 0
    empty = np.zeros((2, 4, 4), np.uint32)                   # a raster without a single cell
    header = {"nx": 4, "ny": 4, "tile": 4, "tiles_x": 1, "tiles_y": 1}
    area, cx, _ = W.footprint(CupyLike(), ("tiles", header, np.array([0]), empty), 3, 0.0, 0.0, 1.0)
    assert area.tolist() == [0, 0, 0] and np.isnan(cx).all()


def test_starting_footprints_on_tiled_and_dense_priors():
    rng = np.random.default_rng(5)
    ny, nx, tile, ncells = 11, 13, 4, 5                     # partial edge tiles both ways
    plane = rng.integers(0, 5, (ny, nx)).astype(np.uint32)  # label 5 never occurs: cell 4 has no footprint
    plane[0, 0] = 99                                         # beyond the run's cells: ignored
    tiles_x, tiles_y = -(-nx // tile), -(-ny // tile)
    present, blocks = [], []
    for t in range(tiles_x * tiles_y):
        r0, c0 = t // tiles_x * tile, t % tiles_x * tile
        part = plane[r0:r0 + tile, c0:c0 + tile]
        block = np.full((tile, tile), 3, np.uint32)          # stale values past a partial tile's edge (a reused buffer)
        block[:part.shape[0], :part.shape[1]] = part
        if part.any():
            present.append(t)
            blocks.append(block)
    tiles = np.zeros((1 + len(blocks), tile, tile), np.uint32)
    tiles[1:] = blocks
    header = {"nx": nx, "ny": ny, "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y}
    area, cx, cy = W.footprint(np, ("tiles", header, np.array(present), tiles), ncells, 10.0, 20.0, 2.0)
    layers = np.stack([plane, np.roll(plane, 1, axis=1)])   # dense: a column counts once per cell in any layer
    darea, dcx, dcy = W.footprint(np, ("dense", np.append(layers.ravel(), 0), nx, ny, 2), ncells, 10.0, 20.0, 2.0)
    for c in range(ncells):
        for got, mask in (((area, cx, cy), plane == c + 1), ((darea, dcx, dcy), (layers == c + 1).any(axis=0))):
            rows, cols = np.nonzero(mask)
            assert got[0][c] == pytest.approx(4.0 * len(rows))
            if len(rows):
                assert got[1][c] == pytest.approx(10 + 2 * (cols.mean() + 0.5), rel=1e-6)
                assert got[2][c] == pytest.approx(20 + 2 * (rows.mean() + 0.5), rel=1e-6)
            else:
                assert np.isnan(got[1][c]) and np.isnan(got[2][c]) and c == 4


# ---- the mixing check in the worker, without a GPU -----------------------------------------------------------------
X0, Y0, VOXEL, NX, NY, NZ, NCELLS, JUDGE = 100.0, 50.0, 2.0, 16, 12, 2, 12, 100
RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
KEEP = 0.75             # proseg's averaged count in write_run's frames: e = 0.75 n of every gene (multiples of 1/32)
COUNTS_INFO = {"mode": "request", "format": "PLC1", "version": 1, "e_scale": 32, "start": True, "start_every": None,
               "spool": None, "where": 1}       # info.json "counts" (native: --live-counts request --live-counts-start)


def prior_plane():
    """The prior: 12 cells, 4 x 4-voxel blocks of a 16 x 12 grid of 2 µm voxels (label = cell + 1)."""
    j, i = np.mgrid[0:NY, 0:NX]
    return ((j // 4) * 4 + i // 4 + 1).astype(np.uint32)


def prior_upper_layer():
    """A dense prior's second layer: there, cell 1 also holds a column of cell 2."""
    plane = prior_plane()
    plane[0:4, 4] = 1
    return plane


def frame_plane(f):
    """Frame f: cell 1 has grown f columns into cell 2; cell 12 has vanished from frame 2 on."""
    plane = prior_plane()
    plane[0:4, 4:4 + f] = 1
    if f >= 2:
        plane[plane == 12] = 0
    return plane


def cell_stats(plane, f):
    """A frame's per-cell statistics as proseg-live writes them: 7 x cells u32 (area, voxels, assigned, foreground,
    changed, then centroid x and y as f32 bits, NaN for a cell without a column)."""
    stats = np.zeros((7, NCELLS), np.uint32)
    cx, cy = np.full(NCELLS, np.nan, np.float32), np.full(NCELLS, np.nan, np.float32)
    for c in range(NCELLS):
        rows, cols = np.nonzero(plane == c + 1)
        stats[:5, c] = len(rows), 2 * len(rows), 3 * len(rows), 2 * len(rows), f
        if len(rows):
            cx[c], cy[c] = X0 + VOXEL * (cols.mean() + 0.5), Y0 + VOXEL * (rows.mean() + 0.5)
    stats[5], stats[6] = cx.view(np.uint32), cy.view(np.uint32)
    return stats


def write_run(folder, tiled=False, frames=3, seed=0, keep=KEEP):
    """A run folder as proseg-live leaves it: live/info.json, genes.txt, transcripts.bin.gz, the prior (prior.tiles, or
    the dense prior.bin.zst with two layers), frames (tile containers or gzip) with their cell statistics, frames.jsonl,
    and proseg's own count of every frame (tests/counts_fixtures.py: frames/NNNNNN.counts.zst, its sidecar and
    counts.jsonl, as --live-counts writes them): n = the transcripts in each cell's voxels of the frame (both layers
    hold the frame's plane), e = keep x n; the start block (proseg's iteration-0 assignment at the frame's keep) =
    keep x frame 0's n; f (the sampled count) with the init frame and the last one. Transcripts are dense over the top
    two rows of cells and sparse over the bottom one."""
    from numcodecs import Zstd
    rng = np.random.default_rng(seed)
    live = folder / "live"
    (live / "frames").mkdir(parents=True)
    n = 6000
    top = rng.random(n) < 0.85
    x = rng.uniform(X0 - 3, X0 + NX * VOXEL + 3, n)
    y = np.where(top, rng.uniform(Y0, Y0 + 16, n), rng.uniform(Y0 - 3, Y0 + NY * VOXEL + 3, n))
    i, j = np.floor((x - X0) / VOXEL).astype(int), np.floor((y - Y0) / VOXEL).astype(int)
    inside = (i >= 0) & (i < NX) & (j >= 0) & (j < NY)
    cell = np.where(inside, prior_plane()[np.clip(j, 0, NY - 1), np.clip(i, 0, NX - 1)], 0)
    profiles = rng.gamma(0.3, 1.0, (NCELLS + 1, len(RUN_GENES)))
    profiles /= profiles.sum(axis=1, keepdims=True)
    gene = np.minimum((rng.random(n)[:, None] > np.cumsum(profiles[cell], axis=1)).sum(axis=1), len(RUN_GENES) - 1)
    records = np.zeros(n, RECORD)
    records["x"], records["y"], records["z"], records["gene"] = x, y, rng.uniform(0, 1, n), gene
    (live / "transcripts.bin.gz").write_bytes(gzip.compress(records.tobytes()))
    (live / "genes.txt").write_text("".join(g + "\n" for g in RUN_GENES))
    plane = prior_plane()
    geometry = {"nx": NX, "ny": NY, "x0": X0, "y0": Y0, "voxel": VOXEL}
    if tiled:
        write_container(live / "prior.tiles", {"label": plane, "permille": np.where(plane > 0, 800, 0)}, 5)
        prior = {"file": "prior.tiles", "format": "tiles", "tile": 5, "nz": 1, **geometry}
    else:
        layers = np.stack([plane, prior_upper_layer()])
        raw = np.concatenate([layers.ravel(), np.where(layers > 0, 800, 0).ravel()]).astype("<u4")
        (live / "prior.bin.zst").write_bytes(Zstd(level=1).encode(raw.tobytes()))
        prior = {"file": "prior.bin.zst", "nz": NZ, **geometry}
    from counts_fixtures import counts_from_labels, write_frame_counts
    grid = {**geometry, "nz": NZ}
    first = None
    index = []
    for f in range(frames):
        labels = frame_plane(f)
        made = counts_from_labels(records, np.stack([labels, labels]).ravel(), grid, 0.0, 0.5, NCELLS, len(RUN_GENES),
                                  keep)
        first = first or made
        write_frame_counts(live, f, f, ncells=NCELLS, ngenes=len(RUN_GENES), indptr=made["indptr"],
                           genes=made["genes"], n=made["n"], e=made["e"],
                           f=made["f"] if f in (0, frames - 1) else None, start_e=first["n"] * float(keep),
                           reason="init" if f == 0 else "end" if f == frames - 1 else "request")
        if tiled:
            name, extra = f"frames/{f:06d}.tiles", {"format": "tiles", "tile": 5, "window": None}
            write_container(live / name, {"consensus": labels, "last_change": np.zeros_like(labels)}, 5)
        else:
            name, extra = f"frames/{f:06d}.bin.gz", {}
            raw = np.concatenate([labels.ravel(), np.zeros(NX * NY), labels.ravel(), labels.ravel()]).astype("<u4")
            (live / name).write_bytes(gzip.compress(raw.tobytes()))
        cells_file = f"frames/{f:06d}.cells.zst"
        (live / cells_file).write_bytes(Zstd(level=1).encode(cell_stats(labels, f).astype("<u4").tobytes()))
        index.append({"seq": f, "iter": f, "phase": "burnin", "file": name, "nz": NZ, **geometry, "t": 0.1 * f,
                      "live": {"cell-compactness": 0.04}, "cells": NCELLS, "cells_file": cells_file, **extra})
    (live / "info.json").write_text(json.dumps({"cells": NCELLS, "genes": len(RUN_GENES), "layers": NZ, "prior": prior,
                                                "transcripts": n, "z": {"grid_origin": 0.0, "grid_spacing": 0.5},
                                                "counts": COUNTS_INFO}))
    (live / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    return folder


def counted(folder, raster):
    """Each cell's transcripts per run gene on a label raster [layers, NY, NX], counted from the records directly."""
    records = np.frombuffer(gzip.decompress((folder / "live" / "transcripts.bin.gz").read_bytes()), RECORD)
    i = np.floor((records["x"] - X0) / VOXEL).astype(int)
    j = np.floor((records["y"] - Y0) / VOXEL).astype(int)
    k = np.clip(np.floor(records["z"] / 0.5), 0, raster.shape[0] - 1).astype(int)
    inside = (i >= 0) & (i < NX) & (j >= 0) & (j < NY)
    label = np.where(inside, raster[k, np.clip(j, 0, NY - 1), np.clip(i, 0, NX - 1)], 0)
    counts = np.zeros((NCELLS, len(RUN_GENES)), np.float32)
    np.add.at(counts, (label[label > 0].astype(int) - 1, records["gene"][label > 0].astype(int)), 1)
    return counts


def expected_scores(encoder, counts, fit_min=5):
    """(transcripts of the model's genes, neighbour share) per cell, through the model's own gene order."""
    model = model_rows(encoder, counts)
    totals = model.sum(axis=1)
    share = np.full(len(counts), np.nan, np.float32)
    rows = totals >= fit_min
    share[rows] = W.neighbour_share(W.mixing_weights(np, np.log1p(model[rows]) @ encoder["W0"].T, encoder))
    return totals, share


def model_rows(encoder, counts):
    """A run's cell x gene counts (RUN_GENES) in the model's gene order (genes it lacks left out)."""
    index = W.gene_index(encoder, RUN_GENES)[0]
    model = np.zeros((len(counts), encoder["ngenes"]), np.float32)
    model[:, index[index >= 0]] = counts[:, index >= 0]
    return model


def proseg_count(folder, f, keep=KEEP):
    """proseg's averaged count of write_run's frame f, as its counts file holds it: keep x the transcripts in each cell's
    voxels of the frame (both layers hold its plane)."""
    labels = frame_plane(f)
    return (counted(folder, np.stack([labels, labels])) * np.float32(keep)).astype(np.float32)


def counts_files(live, seq, start=True) -> dict:
    """A frame request's fields for proseg's count, as the engine names them (engine/analysis.py): its counts file, and
    with a start line frame 0's and the one holding the frame's start block (its own: every frame of write_run holds
    one)."""
    out = {"counts_file": str(live / "frames" / f"{seq:06d}.counts.zst")}
    if start:
        out.update(init_counts_file=str(live / "frames" / "000000.counts.zst"), start_counts_file=out["counts_file"])
    return out


def like_for_like_flags(encoder, now, start, qc):
    """The like-for-like change written out cell by cell (DESIGN 1.7): each cell judged now and at the start (the start
    block: proseg's iteration-0 assignment at the frame's keep) has both counts scaled to the shallower total of the
    model's genes, both scored in one batch; per cell (compared, looks mixed now, looked mixed at the start), False
    where it is not compared: what the bins' MATCHED_SUMS add up per bin."""
    tn, sn = expected_scores(encoder, now)
    ts, ss = expected_scores(encoder, start)
    compared = (tn >= qc["judge_min"]) & np.isfinite(sn) & (ts >= qc["judge_min"]) & np.isfinite(ss)
    now_mixed, start_mixed = np.zeros(len(now), bool), np.zeros(len(now), bool)
    cells = np.nonzero(compared)[0]
    if cells.size:
        depth = np.minimum(tn[cells], ts[cells])
        rows = np.concatenate([model_rows(encoder, now[cells]) * (depth / tn[cells])[:, None],
                               model_rows(encoder, start[cells]) * (depth / ts[cells])[:, None]]).astype(np.float32)
        share = W.neighbour_share(W.mixing_weights(np, np.log1p(rows) @ encoder["W0"].T, encoder))
        mixed = np.floor(share * 250) >= W.code_at(qc["mixed_at"])
        now_mixed[cells], start_mixed[cells] = mixed[:cells.size], mixed[cells.size:]
    return compared, now_mixed, start_mixed


def like_for_like_reference(encoder, now, start, qc):
    """like_for_like_flags over the run: (paired, mixed now, mixed at the start)."""
    return tuple(int(flags.sum()) for flags in like_for_like_flags(encoder, now, start, qc))


@pytest.mark.parametrize("tiled", [False, True])
def test_the_worker_scores_the_start_line_and_frames_without_a_gpu(tmp_path, tiled):
    """The check reads proseg's own count of each frame (its counts file: the same for a dense and a tiled run, the
    frame's labels never counted); the start line is proseg's count of the init frame, its footprints the prior's; each
    frame is read against its start block (the same cells at the start) and like for like."""
    from numcodecs import Zstd
    folder = write_run(tmp_path / "run", tiled=tiled)
    live, out = folder / "live", tmp_path / "qc"
    out.mkdir()
    encoder = W.load_encoder(make_model(tmp_path / "encoder.npz", MODEL_GENES))
    info = json.loads((live / "info.json").read_text())
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    qc = {"model": str(tmp_path / "encoder.npz"), "model_id": "0123abcd", "mixed_at": 0.3, "judge_min": JUDGE,
          "fit_min": 5, "grid": {"x0": X0, "y0": Y0, "bin": 10.0, "nx": 4, "ny": 3}}
    common = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    worker = W.Worker(xp=np)
    reply = worker.handle({"op": "qc_start", **common, "seq": None, **counts_files(live, 0, start=False),
                           "prior": {**info["prior"], "file": str(live / info["prior"]["file"])},
                           "qc": {**qc, "out_cells": str(out / "start.bin"), "out_start": str(out / "start.npz"),
                                  "out_bins": str(out / "start.bins.npz")}})
    assert reply["ok"], reply
    start = reply["qc"]
    # The start line: proseg's count of the init frame (frame 0's counts file); the prior gives the footprints only.
    e0 = proseg_count(folder, 0)
    totals0, share0 = expected_scores(encoder, e0)
    codes0 = W.share_codes(np, share0, (totals0 >= JUDGE) & np.isfinite(share0))
    stored = np.load(out / "start.npz")
    assert stored["totals"].dtype == np.float32 and np.array_equal(stored["totals"], totals0)
    assert np.array_equal(stored["code"], codes0) and Zstd().decode((out / "start.bin").read_bytes()) == codes0.tobytes()
    area0 = np.full(NCELLS, 64.0)
    area0[0] = 64.0 if tiled else 80.0                       # the dense prior's cell 1 has one more column upstairs
    np.testing.assert_allclose(stored["area_um2"], area0)
    assert start["kind"] == "start" and start["footprints"] == NCELLS and start["judged"] == (codes0 != 255).sum()
    assert 0 < start["judged"] < NCELLS                      # the sparse bottom row is scored but not judged
    assert start["genes"] == {"run": 11, "model": 10, "matched": 9, "names": "genes"}
    assert (start["basis"], start["count"]) == ("proseg", "averaged") and start["tx_sum"] == round(float(totals0.sum()), 1)
    assert np.load(out / "start.bins.npz")["cells"].sum() == NCELLS
    # Frame 2, QC alone (no UMAP, and 12 cells are plenty), with the start line: the same cells in its start block.
    request = {"op": "analyze", "ops": ["qc"], **common, "seq": 2, "iter": 2, **counts_files(live, 2),
               "frame": str(live / frame["file"]), "format": frame.get("format", "dense"),
               **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
               "cells_file": str(live / frame["cells_file"]),
               "qc": {**qc, "start": str(out / "start.npz"), "out_cells": str(out / "000002.qc.bin"),
                      "out_bins": str(out / "000002.bins.npz")}}
    reply = worker.handle(request)
    assert reply["ok"] and "summary" not in reply, reply
    summary = reply["qc"]
    e2 = proseg_count(folder, 2)
    totals, share = expected_scores(encoder, e2)
    judged = (totals >= JUDGE) & np.isfinite(share)
    codes = W.share_codes(np, share, judged)
    assert Zstd().decode((out / "000002.qc.bin").read_bytes()) == codes.tobytes()
    mixed, start_mixed, paired = judged & (codes >= 75), (codes0 != 255) & (codes0 >= 75), judged & (codes0 != 255)
    assert (summary["judged"], summary["mixed"]) == (judged.sum(), mixed.sum())
    assert summary["tx_sum"] == summary["transcripts"] == round(float(totals.sum()), 1)
    assert summary["paired"] == {"n": paired.sum(), "mixed_now": (paired & mixed).sum(),
                                 "mixed_start": (paired & start_mixed).sum(), "n10": (paired & mixed & ~start_mixed).sum(),
                                 "n01": (paired & ~mixed & start_mixed).sum()}
    # The start block (write_run: keep x frame 0's assignment) read at its own depth: the start line's own numbers.
    j0, m0 = int((codes0 != 255).sum()), int(((codes0 != 255) & (codes0 >= 75)).sum())
    assert summary["start_block"] == {"per100": round(100 * m0 / j0, 2), "judged": j0, "mixed": m0,
                                      "tx_sum": round(float(totals0.sum()), 1),
                                      "median_tx": round(float(np.median(totals0[(totals0 >= 5) & np.isfinite(share0)])), 1),
                                      "iteration": 2}
    n, now_m, start_m = like_for_like_reference(encoder, e2, e0, qc)
    matched = summary["matched"]
    assert n > 0 and {k: v for k, v in matched.items() if k != "ms"} == {
        "paired": n, "used": n, "per100_now": round(100 * now_m / n, 4), "per100_start": round(100 * start_m / n, 4),
        "change": round(100 * now_m / n - 100 * start_m / n, 4), "sd": None, "seeds": 0}
    # Cell 1 grew by 8 columns, cell 2 lost them, cell 12 vanished (no centroid: placed where it started).
    assert summary["growth"]["n"] == NCELLS and summary["growth"]["lost_half"] == 1 and summary["tiny"] == 1
    assert summary["growth"]["grown"] == 1 and summary["growth"]["shrunk"] == 2
    bins = np.load(out / "000002.bins.npz")
    assert bins["cells"].sum() == NCELLS and bins["judged"].sum() == judged.sum() and bins["lost_half"].sum() == 1
    assert bins["judged_start"].sum() == j0 and bins["mixed_start"].sum() == m0
    assert bins["tx_start"].sum() == pytest.approx(totals0.sum()) and bins["matched_paired"].sum() == n
    assert bins["grid"].tolist() == [X0, Y0, 10.0, 4, 3] and bins["depth_edges"].tolist() == [100, 250, 500, 1000]
    assert summary["has_start"] and summary["cellstats"] and summary["selftest"] <= 1e-5 and summary["model"] == "0123abcd"
    assert (summary["basis"], summary["count"]) == ("proseg", "averaged")
    assert {"load", "counts", "score", "bins", "write"} <= set(summary["ms"])
    # A frame without its cell statistics still works (no growth); cells sit at their starting footprints, or nowhere
    # without the start line (never at an outline's transcripts).
    reply = worker.handle({**request, "seq": 1, **counts_files(live, 1), "cells_file": None,
                           "qc": {**qc, "start": str(out / "start.npz"), "out_cells": str(out / "x.qc.bin")}})
    assert reply["ok"] and not reply["qc"]["cellstats"] and "growth" not in reply["qc"]
    assert reply["qc"]["located"] == NCELLS
    alone = {k: v for k, v in request.items() if not k.endswith("_counts_file")}      # a run without a start line
    reply = worker.handle({**alone, "seq": 1, **counts_files(live, 1, start=False), "cells_file": None,
                           "qc": {**qc, "start": None, "out_cells": str(out / "x.qc.bin")}})
    assert reply["ok"] and reply["qc"]["located"] == 0 and reply["qc"]["start_block"] is None
    assert reply["qc"]["matched"] is None and "paired" not in reply["qc"] and not reply["qc"]["has_start"]
    # A model none of whose genes is in the run fails the check alone, with a reason.
    make_model(tmp_path / "other.npz", ["X1", "X2"])
    reply = worker.handle({**request, "qc": {**qc, "model": str(tmp_path / "other.npz"),
                                             "out_cells": str(out / "y.qc.bin"), "out_bins": str(out / "y.npz")}})
    assert not reply["ok"] and "none of this run's genes" in reply["errors"]["qc"] and "qc" not in reply


def test_a_frame_without_proseg_count_is_never_counted_from_its_outlines(tmp_path, capsys):
    """Every op of a frame reads proseg's own count (its counts file) or fails with reason "no_counts" (one plain line
    in the worker's log): no counts file named (a run made before proseg's counts were kept), its file gone (pruned
    from the spool), or one that does not fit the run. The frame's labels are never counted instead."""
    folder = write_run(tmp_path / "run")
    live = folder / "live"
    make_model(tmp_path / "encoder.npz", MODEL_GENES)
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    request = {"op": "analyze", "ops": ["qc", "umap"], "run": "r1", "seq": 2, "iter": 2,
               "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"), "ncells": NCELLS,
               "frame": str(live / frame["file"]), "format": "dense", "cells_file": str(live / frame["cells_file"]),
               **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")}, "z_origin": 0.0, "z_spacing": 0.5,
               "out_json": str(tmp_path / "2.json"), "out_cells": str(tmp_path / "2.cells.bin"),
               "qc": {"model": str(tmp_path / "encoder.npz"), "model_id": "m", "mixed_at": 0.3, "judge_min": JUDGE,
                      "fit_min": 5, "grid": {"x0": X0, "y0": Y0, "bin": 10.0, "nx": 4, "ny": 3}, "start": None,
                      "out_cells": str(tmp_path / "2.qc.bin")}}
    other = write_run(tmp_path / "other")
    (other / "live" / "genes.txt").write_text("".join(g + "\n" for g in RUN_GENES[:5]))   # another run's genes
    worker = W.Worker(xp=np)
    capsys.readouterr()
    for counts, words in ((None, "names no counts file"), (str(live / "frames" / "000009.counts.zst"), "is gone"),
                          (str(tmp_path / "bad.counts.zst"), "not a whole zstd frame")):
        (tmp_path / "bad.counts.zst").write_bytes(b"not zstd")
        reply = worker.handle({**request, "counts_file": counts})
        assert not reply["ok"] and reply["reasons"] == {"qc": "no_counts", "umap": "no_counts"}, reply
        assert words in reply["errors"]["qc"] and "qc" not in reply and "summary" not in reply
        assert not (tmp_path / "2.qc.bin").exists() and not (tmp_path / "2.json").exists()
    err = capsys.readouterr().err
    assert "Traceback" not in err and err.count("has no proseg count to read") == 3, err
    reply = worker.handle({**request, "run": "r2", "genes": str(other / "live" / "genes.txt"),
                           "counts_file": str(live / "frames" / "000002.counts.zst")})
    assert reply["reasons"] == {"qc": "no_counts", "umap": "no_counts"}
    assert "CountsError: the counts file holds 11 genes, the run 5" in reply["errors"]["qc"]
    reply = worker.handle({**request, "ncells": NCELLS + 1, "counts_file": str(live / "frames" / "000002.counts.zst")})
    assert "the counts file holds 12 cells, the run 13" in reply["errors"]["qc"]
    reply = worker.handle({**request, "ops": ["qc"], "counts_file": str(live / "frames" / "000002.counts.zst")})
    assert reply["ok"], reply
    info = json.loads((live / "info.json").read_text())       # the start line: the init frame's count, or the reason
    start = {"op": "qc_start", "run": "r1", "seq": None, "genes": str(live / "genes.txt"), "ncells": NCELLS,
             "transcripts": str(live / "transcripts.bin.gz"), "z_origin": 0.0, "z_spacing": 0.5,
             "prior": {**info["prior"], "file": str(live / info["prior"]["file"])},
             "qc": {**request["qc"], "out_cells": str(tmp_path / "start.bin")}}
    reply = worker.handle(start)
    assert not reply["ok"] and reply["reasons"] == {"qc_start": "no_counts"} and not (tmp_path / "start.bin").exists()
    assert worker.handle({**start, "counts_file": str(live / "frames" / "000000.counts.zst")})["ok"]


def test_count_is_proseg_own_count_of_the_frame(tmp_path, monkeypatch):
    """Worker.count: the frame's cell x gene matrix is proseg's count as its counts file holds it (the averaged count
    e: LIVE_COUNT's), the file beside it (n, e, f as stored), nothing counted from the frame's labels; dense with numpy
    alone, a scipy CSR on the CPU worker and a cupyx CSR on the GPU (int32 indices and row pointers), both without the
    pairs whose count is 0: each the same hand-built CSR."""
    from sopa_companion.analysis import counts_file
    from counts_fixtures import write_frame_counts
    live = tmp_path / "live"
    (live / "frames").mkdir(parents=True)
    genes = ["A", "B", "C", "D", "E"]
    (live / "genes.txt").write_text("".join(g + "\n" for g in genes))
    (live / "info.json").write_text(json.dumps({"cells": 3, "transcripts": 77}))
    rows = {"indptr": [0, 3, 3, 5], "genes": [0, 2, 4, 1, 3], "n": [4, 1, 9, 2, 3],
            "e": [2.5, 0.0, 6.25, 1.5, 3.0], "f": [2, 0, 6, 1, 3]}       # (0, gene 2): assigned, but averages to 0
    write_frame_counts(live, 7, 70, ncells=3, ngenes=5, **rows, reason="pause")
    request = {"run": "r", "genes": str(live / "genes.txt"), "ncells": 3,
               "counts_file": str(live / "frames" / "000007.counts.zst")}
    worker = W.Worker(xp=np)
    run = worker.state(request)
    assert run.genes == genes and run.transcripts == 77 and not hasattr(run, "x")
    matrix, read = worker.count(run, request)
    hand = np.zeros((3, 5), np.float32)
    hand[[0, 0, 0, 2, 2], [0, 2, 4, 1, 3]] = rows["e"]
    assert matrix.dtype == np.float32 and np.array_equal(matrix, hand)
    assert read.iteration == 70 and read.n.tolist() == rows["n"] and read.f.tolist() == rows["f"]
    assert counts_file.LIVE_COUNT == "averaged" and counts_file.LIVE_ARRAY == "e"
    assert np.array_equal(worker.matrix(read, "f"), np.array([[2, 0, 0, 0, 6], [0] * 5, [0, 1, 0, 3, 0]], np.float32))

    class Sparse:                                       # what scipy's or cupyx's csr_matrix is given (this venv has
        def __init__(self, arrays, shape):              # neither)
            self.data, self.indices, self.indptr = arrays
            self.shape = shape
    scipy, sparse = types.ModuleType("scipy"), types.ModuleType("scipy.sparse")
    sparse.csr_matrix, scipy.sparse = Sparse, sparse
    monkeypatch.setitem(sys.modules, "scipy", scipy)    # put back as they were after the test
    monkeypatch.setitem(sys.modules, "scipy.sparse", sparse)
    cpu = W.Worker(xp=np, backend="cpu")
    fake_cupyx(monkeypatch, Sparse)
    gpu = W.Worker(xp=GpuLike(), backend="gpu")
    for worker in (cpu, gpu):                           # the stored 0 left out; float32 counts
        csr = worker.count(worker.state(request), request)[0]
        assert isinstance(csr, Sparse) and csr.shape == (3, 5)
        assert np.asarray(csr.indptr).tolist() == [0, 2, 2, 4] and np.asarray(csr.indices).tolist() == [0, 4, 1, 3]
        assert np.asarray(csr.data).tolist() == [2.5, 6.25, 1.5, 3.0] and np.asarray(csr.data).dtype == np.float32
    assert csr.indices.dtype == csr.indptr.dtype == np.int32           # cupyx's CSR: int32 indices and row pointers
    assert np.asarray(gpu.matrix(read, "f").data).tolist() == [2, 6, 1, 3]


def test_the_worker_takes_the_qc_managers_own_request_fields(tmp_path):
    """engine/qc.py names the grid by the run box, a bin size and an origin, the start line by its byte file, and may
    ask for the summary as a file; per-bin sums only when asked for."""
    folder = write_run(tmp_path / "run")
    live, out = folder / "live", tmp_path / "qc"
    out.mkdir()
    model = make_model(tmp_path / "model" / "encoder.npz", MODEL_GENES).parent       # a model folder
    info = json.loads((live / "info.json").read_text())
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    block = {"model": str(model), "model_id": "feed1234", "mixed_at": 0.3, "judge_min": JUDGE, "fit_min": 5,
             "bin_um": 10, "origin": [X0, Y0], "box": [X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL]}
    common = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    worker = W.Worker(xp=np)
    reply = worker.handle({"op": "qc_start", **common, "seq": None, **counts_files(live, 0, start=False),
                           "prior": {**info["prior"], "file": str(live / info["prior"]["file"])},
                           "qc": {**block, "out_cells": str(out / "start.bin"), "out_json": str(out / "start.json")}})
    assert reply["ok"], reply
    written = json.loads((out / "start.json").read_text())
    assert written["model_id"] == "feed1234" and written["judged"] == reply["qc"]["judged"] and written["cells"] == NCELLS
    assert (out / "start.npz").exists() and not list(out.glob("*bins*"))   # start values beside start.bin; no bins
    assert reply["qc"]["grid"] == {"x0": X0, "y0": Y0, "bin": 10.0, "nx": 4, "ny": 3}
    reply = worker.handle({"op": "qc", **common, "seq": 2, **counts_files(live, 2), "frame": str(live / frame["file"]),
                           "format": "dense", **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                           "cells_file": str(live / frame["cells_file"]),
                           "qc": {**block, "start": str(out / "start.bin"), "out_cells": str(out / "000002.qc.bin"),
                                  "out_json": str(out / "000002.qc.json")}})
    assert reply["ok"] and reply["qc"]["has_start"] and reply["qc"]["paired"]["n"] > 0
    stored = json.loads((out / "000002.qc.json").read_text())
    assert stored["paired"] == reply["qc"]["paired"] and stored["start_block"] == reply["qc"]["start_block"]
    assert reply["qc"]["files"]["bins"] is None and not list(out.glob("*bins*"))


# ---- the mixing check in the scheduler -----------------------------------------------------------------------------
class Provider:
    """A QC provider (engine/qc.py's role): one model for every run, or None while the sample learns; it keeps every
    record it is handed."""

    def __init__(self, model=("0123456789abcdef", "/models/m1")):
        self.model, self.records = model, []

    def qc_model(self, run):
        return None if self.model is None else {"id": self.model[0], "path": self.model[1]}

    def qc_result(self, record):
        self.records.append(record)


def finish(m, job):
    """What a successful pass leaves behind, without a worker (its results on the count its frame is read on)."""
    count = m._frame_count(m.runs.get(job.run), job.seq) if job.seq is not None else None
    if "umap" in job.ops:
        m._store(job.run, job.seq, {"seq": job.seq, "count": count})
    if "qc" in job.ops:
        job.qc.done[job.seq] = count
        m.qc_status[job.run] = {"state": "ready", "seq": job.seq}
    if "qc_start" in job.ops:
        job.qc.start = "done"


def test_qc_settings_and_model_folders():
    assert qc_settings(None)["mixed_at"] == 0.30 and qc_settings({"mixed_at": 0.2, "python": "x"})["mixed_at"] == 0.2
    assert qc_settings({"judge_min_counts": 50})["judge_min_counts"] == 50.0
    # One validator for [qc] (engine/qc.py QcSettings): what the engine refuses at start-up is refused here too (a
    # judge_min_counts of 100.5 was truncated to 100 here and refused there).
    for bad in ({"mixed_at": 1.5}, {"judge_min_counts": "many"}, {"bin_um_tiled": 0}, {"fit_min_counts": True},
                {"judge_min_counts": 100.5}, {"fit_min_counts": 150}):
        with pytest.raises(ValueError, match=r"\[qc\]"):
            qc_settings(bad)
    assert model_key("0123456789abcdef") == "01234567" and model_key("../../x") == "x" and model_key("") == "model"


def test_the_bin_grid_is_anchored_at_the_runs_raster(tmp_path):
    """The areas were calibrated on a grid anchored at the run's raster origin (the prior's and every frame's x0,
    y0), which can lie far from the run box's corner (a whole sample: 26 µm and 306 µm away)."""
    m = manager(tmp_path, qc={})
    whole = FakeRun(cells=40070)
    whole.meta.update(mode="tiled", box=[11.140625, 19.203125, 10812.296875, 10883.40625])
    whole.info["prior"] = {"format": "tiles", "x0": 37.390625, "y0": 325.0, "nx": 5400, "ny": 5325, "voxel": 2.0}
    assert m.qc_grid(whole) == {"x0": 37.390625, "y0": 325.0, "bin": 100.0, "nx": 108, "ny": 107}
    detail = FakeRun(cells=3000)                                   # no exported prior: the first frame's raster
    detail.meta.update(mode="detail", box=[0.0, 0.0, 300.0, 200.0])
    detail.frames[0].update(x0=4.0, y0=6.0, nx=150, ny=99, voxel=2.0)
    assert m.qc_grid(detail) == {"x0": 4.0, "y0": 6.0, "bin": 50.0, "nx": 6, "ny": 4}
    early = FakeRun(cells=3000, frames=0)                          # nothing exported yet: the run box
    early.meta.update(mode="detail", box=[10.0, 20.0, 110.0, 70.0])
    assert m.qc_grid(early) == {"x0": 10.0, "y0": 20.0, "bin": 50.0, "nx": 2, "ny": 1}


def test_without_a_provider_or_a_model_nothing_is_scored(tmp_path):
    run = FakeRun(cells=3000, frames=3)
    m = manager(tmp_path, run)
    m.configure("r1", enabled=False)                         # Clusters & UMAP switched off by a tool: the check alone
    assert m.configure_qc("r1", enabled=True)["enabled"] and not m.describe_qc("r1")["available"]
    assert m._next_job() == (None, None)                     # no provider: nothing to score with, nothing to wait for
    m.configure("r1", enabled=True)
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.ops == ("umap",) and job.kind == "live"   # clusters and UMAP exactly as before
    finish(m, job)
    provider = Provider(model=None)                          # the sample is still learning
    m.attach_qc(provider)
    assert m._next_job() == (None, POLL_S) and m.describe_qc("r1")["model"] is None
    m.enabled["r1"] = False
    assert m._next_job() == (None, QC_MODEL_POLL_S)          # only the check left: it asks again in a while
    provider.model = ("abcdef0123", "/models/m1")
    m.qc_changed()                                           # learned: at once
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.ops == ("qc",) and job.qc.dir == run.dir / "qc" / "abcdef01.pc1"
    assert m.describe_qc("r1")["backfill"] == 2              # frames 0 and 1 came before the model
    m.configure_qc("r1", enabled=False)
    finish(m, job)
    assert m._next_job() == (None, None)                     # switched off: nothing more, no backfill either
    for bad, status in (("x", 422), (99, 404)):
        with pytest.raises(AnalysisError) as refused:
            m.configure_qc("r1", seq=bad)
        assert refused.value.status == status and not m.qc_requested


def test_the_check_rides_with_every_umap_pass_and_keeps_its_own_cadence_alone(tmp_path):
    whole, crop = FakeRun("whole", cells=40070), FakeRun("crop", cells=3439)
    m = manager(tmp_path, whole, crop)
    for name in ("whole", "crop"):
        m.configure(name, enabled=False)                     # Clusters & UMAP switched off by a tool: the check alone
    m.attach_qc(Provider())
    now = [100.0]
    m.clock = lambda: now[0]
    m.configure_qc("whole", enabled=True)
    job, _ = m._next_job()
    assert job == ("whole", 0) and job.ops == ("qc",) and job.kind == "live"
    finish(m, job)
    whole.add_frame()
    assert m._next_job() == (None, pytest.approx(QC_LARGE_INTERVAL_S))   # >= 20,000 cells: at most once a second alone
    now[0] += QC_LARGE_INTERVAL_S
    job, _ = m._next_job()
    assert job == ("whole", 1) and job.ops == ("qc",)
    finish(m, job)
    m.configure("whole", enabled=True)                       # Clusters & UMAP on: the check rides along with each pass
    for _ in range(3):
        whole.add_frame()
        job, _ = m._next_job()
        assert job == ("whole", len(whole.frames) - 1) and job.ops == ("umap", "qc")
        finish(m, job)
    whole.add_frame()
    m._store("whole", len(whole.frames) - 1, {"seq": len(whole.frames) - 1})   # that frame's UMAP exists already
    now[0] += QC_LARGE_INTERVAL_S
    job, _ = m._next_job()
    assert job == ("whole", len(whole.frames) - 1) and job.ops == ("qc",)
    finish(m, job)
    m.configure_qc("crop", enabled=True)                     # a region: the check alone goes back to back
    for _ in range(3):
        job, _ = m._next_job()
        assert job == ("crop", len(crop.frames) - 1) and job.ops == ("qc",)
        finish(m, job)
        crop.add_frame()
    m.min_interval = 5.0                                     # a spaced UMAP does not hold the check back: QC alone
    m.configure("crop", enabled=True)
    job, _ = m._next_job()
    assert job.run == "crop" and job.ops == ("umap", "qc")
    finish(m, job)
    crop.add_frame()
    job, _ = m._next_job()
    assert job.run == "crop" and job.ops == ("qc",)


def test_requests_go_first_and_a_frame_on_screen_is_counted_once(tmp_path):
    run = FakeRun(cells=3000, frames=8, state="done")
    m = manager(tmp_path, run)
    m.attach_qc(Provider())
    m.configure("r1", seq=5)
    m.configure_qc("r1", seq=5)
    job, _ = m._next_job()
    assert job == ("r1", 5) and job.ops == ("umap", "qc") and job.kind == "request" and not m.qc_requested
    finish(m, job)
    m.configure_qc("r1", seq=6)                              # whatever the switch says
    job, _ = m._next_job()
    assert job == ("r1", 6) and job.ops == ("qc",) and job.kind == "request"
    finish(m, job)
    m.configure_qc("r1", seq=6)                              # scored already
    m.configure("r1", seq=6)
    job, _ = m._next_job()
    assert job == ("r1", 6) and job.ops == ("umap",)         # only what the frame lacks
    finish(m, job)
    assert m._next_job() == (None, None)                     # a finished run with both switches off: nothing else
    assert m.describe_qc("r1")["seqs"] == [5, 6] and m.describe_qc("r1")["latest_seq"] == 6


def test_the_start_line_and_backfill_go_between_live_frames(tmp_path):
    run = FakeRun(cells=3000, frames=121, directory=tmp_path / "run")
    run.meta.update(CELLS_START)
    run.info.update(prior={"file": "prior.bin.zst"})
    run.live.mkdir(parents=True)
    for name in ("prior.bin.zst", "transcripts.bin.gz"):
        (run.live / name).write_bytes(b"")
    provider = Provider(model=None)
    m = manager(tmp_path, run)
    m.configure("r1", enabled=False)                         # Clusters & UMAP switched off by a tool: the check alone
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    assert m._next_job() == (None, QC_MODEL_POLL_S)
    provider.model = ("feedc0de99", "/models/m2")            # the sample finished learning: frame 120 is the newest
    jobs = []
    for _ in range(2 * BACKFILL_MAX + 4):
        job, _ = m._next_job()
        jobs.append(job)
        finish(m, job)
        run.add_frame()                                      # proseg keeps sampling: a live frame is always due
    assert (jobs[0].kind, jobs[0].seq, jobs[0].ops) == ("live", 120, ("qc",))
    assert (jobs[1].kind, jobs[1].seq, jobs[1].ops) == ("start", None, ("qc_start",))
    kinds = [job.kind for job in jobs[2:2 + 2 * BACKFILL_MAX]]
    assert kinds == ["live", "backfill"] * BACKFILL_MAX and [job.kind for job in jobs[-2:]] == ["live", "live"]
    backfill = [job.seq for job in jobs if job.kind == "backfill"]
    assert sorted(backfill) == sorted({int(round(v)) for v in np.linspace(0, 119, BACKFILL_MAX)}) and backfill[0] == 0
    assert max(backfill[:4]) - min(backfill[:4]) > 60       # coarse to fine: the whole run's shape first
    assert jobs[0].qc.dir == run.dir / "qc" / "feedc0de.pc1" and m.describe_qc("r1")["start"] == "done"
    # A finished run opened later: no new frames, so the background passes go back to back.
    old = FakeRun("old", cells=3000, frames=30, state="done", directory=tmp_path / "old")
    old.live.mkdir(parents=True)
    (old.live / "transcripts.bin.gz").write_bytes(b"")
    m = manager(tmp_path, old)
    m.attach_qc(provider)
    m.configure_qc("old", enabled=True)
    kinds = []
    while True:
        job, _ = m._next_job()
        if job is None:
            break
        kinds.append(job.kind)
        finish(m, job)
    assert kinds == ["live"] + ["backfill"] * 29             # (its info names no prior: no start line)
    assert m.describe_qc("old")["start"] == "none"


def test_no_start_line_without_starting_cells(tmp_path):
    for name, prior, info in (("nuclei", {"cells": None, "nuclei": {"kind": "shapes", "key": "nuclei"}},
                               {"prior": {"file": "prior.bin.zst"}}),
                              ("unexported", {"cells": {"kind": "shapes", "key": "cells"}}, {})):
        run = FakeRun(name, cells=3000, frames=1, state="done", directory=tmp_path / name)
        run.meta["prior"] = prior
        run.info.update(info)
        run.live.mkdir(parents=True)
        for file in ("prior.bin.zst", "transcripts.bin.gz"):
            (run.live / file).write_bytes(b"")
        m = manager(tmp_path, run)
        m.attach_qc(Provider())
        m.configure_qc(name, enabled=True)
        job, _ = m._next_job()
        assert job.kind == "live"
        finish(m, job)
        assert m._next_job() == (None, None) and m.describe_qc(name)["start"] == "none", name


def test_backfill_spreads_over_the_run_and_skips_what_is_there():
    assert coarse_to_fine(list(range(8))) == [0, 4, 2, 6, 1, 3, 5, 7] and coarse_to_fine([]) == []
    assert backfill_plan(0, set()) == [] and backfill_plan(1, set()) == [0]
    assert backfill_plan(10, set()) == coarse_to_fine(list(range(10)))      # short runs: every earlier frame
    plan = backfill_plan(1000, set())
    targets = sorted({int(round(v)) for v in np.linspace(0, 999, BACKFILL_MAX)})
    assert len(plan) == BACKFILL_MAX and sorted(plan) == targets and plan[0] == 0
    assert backfill_plan(1000, set(range(0, 1000, 2))) == []                 # scored live often enough already
    later = set(range(500, 1000))
    assert sorted(backfill_plan(1000, later)) == [t for t in targets if t < 500 - 1000 / BACKFILL_MAX / 2]


def test_a_new_model_starts_the_check_again(tmp_path):
    run = FakeRun(cells=3000, frames=1)
    provider = Provider()
    m = manager(tmp_path, run)
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    job, _ = m._next_job()
    finish(m, job)
    first = job.qc
    provider.model = ("ffff0000aaaa", "/models/relearned")   # learned again (new software): a new yardstick
    job, _ = m._next_job()
    assert job == ("r1", 0) and job.qc is not first and job.qc.done == {} and job.qc.dir.name == "ffff0000.pc1"
    assert m.describe_qc("r1")["model"] == "ffff0000aaaa"


def test_a_frame_asked_while_its_sample_learns_is_scored_once_the_model_is_ready(tmp_path):
    """POST /runs/{id}/qc {seq} while the sample still learns (2-6 min after it was opened): the request waits for the
    model instead of being dropped (the frame on screen stayed "Not checked yet"), then goes first. Neither is one
    dropped that came in after this pass asked the provider for its models, and a deleted run's goes with the run."""
    run = FakeRun(cells=3000, frames=1000, state="paused")
    m = manager(tmp_path, run)
    m.configure("r1", enabled=False)                         # Clusters & UMAP switched off by a tool: the check alone
    provider = Provider(model=None)
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    m.configure_qc("r1", seq=517)                            # scrubbed to iteration 517 while the sample learns
    assert m._next_job() == (None, QC_MODEL_POLL_S) and m.describe_qc("r1")["requested"] == 517
    assert m._next_job() == (None, QC_MODEL_POLL_S) and m.qc_requested == {"r1": 517}   # kept, asked again
    provider.model = ("feedc0de99", "/models/m2")            # learned
    m.qc_changed()
    job, _ = m._next_job()
    assert job == ("r1", 517) and job.kind == "request" and job.ops == ("qc",) and not m.qc_requested
    finish(m, job)
    m.configure_qc("r1", enabled=False)
    m.runs.runs["off"] = FakeRun("off", cells=3000, frames=10, state="done")
    m.configure_qc("off", seq=4)                             # its check is off: the provider was not asked about it
    assert m._next_job(models={"r1": ("feedc0de99", "/models/m2")}) == (None, 0.0) and m.qc_requested == {"off": 4}
    job, _ = m._next_job()                                   # asked again at once
    assert job == ("off", 4) and job.kind == "request"
    finish(m, job)
    m.configure_qc("off", seq=5)
    del m.runs.runs["off"]                                   # deleted before its frame was scored
    assert m._next_job() == (None, None) and not m.qc_requested and "off" in m.to_forget


def test_ended_runs_leave_the_gpu_worker_once_scored(tmp_path):
    """Looking at a finished run scores it (the check is on by default): its newest frame, start line and backfill.
    Then the GPU worker may drop its transcripts, once: runs that are only looked at must not hold GPU 1's memory nor
    push the live run out of the worker. Not while it has passes to come or its clusters are on; a deleted run at once."""
    def ended(name, frames=6):
        run = FakeRun(name, cells=3000, frames=frames, state="done", directory=tmp_path / name)
        run.meta.update(CELLS_START)
        run.info.update(prior={"file": "prior.bin.zst"})
        run.live.mkdir(parents=True)
        for file in ("prior.bin.zst", "transcripts.bin.gz"):
            (run.live / file).write_bytes(b"")
        return run
    m = manager(tmp_path, ended("old"), ended("clusters"))
    m.attach_qc(Provider())
    m.configure_qc("old", enabled=True)
    kinds = []
    while (job := m._next_job()[0]) is not None:
        assert not m.to_forget                               # never while passes are still to come
        kinds.append(job.kind)
        finish(m, job)
    assert kinds == ["live", "start"] + ["backfill"] * 5 and m.to_forget == {"old"}
    m.to_forget.clear()
    assert m._next_job() == (None, None) and not m.to_forget     # once: a later request loads it again
    m.configure("clusters", enabled=True)                    # Clusters & UMAP on: the UMAP warm-starts from its layout
    m.configure_qc("clusters", enabled=True)
    while (job := m._next_job()[0]) is not None:
        finish(m, job)
    assert not m.to_forget
    del m.runs.runs["clusters"]                              # deleted
    assert m._next_job() == (None, None) and m.to_forget == {"clusters"}
    sent = []                                                # the scheduler's thread tells a running worker ...
    m.process = types.SimpleNamespace(poll=lambda: None, stdin=types.SimpleNamespace(write=sent.append, flush=lambda: None))
    m._send({"op": "forget", "run": "old"})
    assert [json.loads(line) for line in sent] == [{"op": "forget", "run": "old"}]
    m.process = None
    m._send({"op": "forget", "run": "old"})                  # ... and starts none to tell it
    assert len(sent) == 1


def test_a_run_without_exported_transcripts_is_not_scored(tmp_path):
    """A run made before live analysis existed has no live/transcripts.bin.gz: with the check on (the default) it got no
    start line ever ("Waiting for proseg's starting cells..." for good) and about 41 failing passes, each an error
    event, over half an hour of back-off. Now nothing is tried and it says why."""
    run = FakeRun("legacy", cells=3000, frames=30, state="done", directory=tmp_path / "legacy")
    run.info.update(prior={"file": "prior.bin.zst"})
    provider = Provider()
    m = manager(tmp_path, run)
    m.attach_qc(provider)
    m.configure_qc("legacy", enabled=True)
    assert m._next_job() == (None, None) and not provider.records
    view = m.describe_qc("legacy")
    assert view["unavailable"] == "no_transcripts" and view["start"] == "none" and view["backfill"] == 0


def test_the_worker_runs_on_its_own_gpu_in_pci_order(tmp_path, monkeypatch):
    """The trainer's GPU 0 and the worker's GPU 1 are both nvidia-smi's (PCI bus) numbers: with CUDA's default order
    the worker's "1" could be the trainer's card. Its hello names its bus, and another card is refused."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)   # every GPU seen (a pinned list maps "1" within it)
    m = manager(tmp_path, gpu=1)
    env = m._worker_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    m.probe = lambda: {"bus": "00000000:41:00.0"}           # nvidia-smi --id=1
    assert m._wrong_gpu("0000:41:00.0") is None and m._wrong_gpu(None) is None
    assert "PCI bus 11" in m._wrong_gpu("0000:11:00.0")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    config = {"analysis": {"python": sys.executable, "worker_module": "fake_analysis_worker", "gpu": 1},
              "paths": {"state_dir": str(tmp_path / "state")}}
    m = AnalysisManager(config, FakeRuns(), start=False, probe=lambda: {"bus": "00000000:41:00.0"})
    monkeypatch.setenv("FAKE_ANALYSIS_BUS", "0000:11:00.0")
    with pytest.raises(AnalysisError, match="landed on PCI bus 11, not on GPU 1"):
        m._ensure_worker()
    assert m.process is None
    monkeypatch.setenv("FAKE_ANALYSIS_BUS", "0000:41:00.0")
    m.failed_at = 0.0
    m._ensure_worker()
    assert m.process is not None and m.process.poll() is None
    # The log names both: the refused worker and the one that started (engine/runlog.py).
    events = [text for _, _, text in m.events]
    assert events[0] == "The GPU worker could not start: the GPU worker landed on PCI bus 11, not on GPU 1 (bus 41)"
    assert events[1].startswith(f"GPU worker started on GPU 1, PCI bus 0000:41:00.0 (pid {m.process.pid}) in ")
    assert m.log_info()["worker"]["pid"] == m.process.pid and m.log_info()["worker"]["since"] is not None
    m.close()


# ---- the worker's backend, its start and where Clusters & UMAP cannot run -------------------------------------------
LACKING = ["scipy", "scikit-learn", "igraph", "umap-learn", "numba"]      # what the app's venv lacks for the CPU path


def fake_worker(tmp_path, monkeypatch, *runs, start=False, **analysis):
    """A manager on the fake worker (tests/fake_analysis_worker.py: FAKE_ANALYSIS_BACKEND, FAKE_ANALYSIS_LASTING)."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    config = {"analysis": {"python": sys.executable, "worker_module": "fake_analysis_worker", **analysis},
              "paths": {"state_dir": str(tmp_path / "state")}}
    return AnalysisManager(config, FakeRuns(*runs), start=start)


def worker_starts(monkeypatch, module="fake_analysis_worker") -> list:
    """Every start of the worker `module` (subprocess.Popen, counted)."""
    import subprocess
    started, real = [], subprocess.Popen

    def counting(command, *args, **kwargs):
        if isinstance(command, list) and module in command:
            started.append(command)
        return real(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", counting)
    return started


def test_a_python_without_either_stack_turns_clusters_off_once(tmp_path, monkeypatch):
    """The real worker in a Python that cannot run Clusters & UMAP at all (the app's own venv: neither RAPIDS nor the
    CPU libraries) says so in its hello, naming what it lacks (lasting: the same Python answers the same until the
    app's settings or packages change): they are off for the session, said once in the log; the worker is never started
    again; every request is refused saying what is missing; describe says it (available false, `missing`) and the run's
    status is "unavailable" (the page reads describe again). Run in a Python that has the CPU libraries, the fake
    worker's lasting hello stands in for the real one."""
    from sopa_companion.analysis import backend
    lacking = backend.distributions(backend.missing(backend.CPU_MODULES))
    module = "sopa_companion.analysis.worker"
    if not lacking:
        module, lacking = "fake_analysis_worker", LACKING
        monkeypatch.setenv("FAKE_ANALYSIS_LASTING", ",".join(lacking))
    started = worker_starts(monkeypatch, module)
    run = FakeRun(cells=3000, frames=0, state="preparing")
    m = fake_worker(tmp_path, monkeypatch, run, start=True, worker_module=module)   # started with the preparing run
    try:
        wait_for(lambda: not m.available, timeout=30)
        described = m.describe("r1")
        assert described["available"] is False and described["missing"] == lacking
        assert described["status"] == {"state": "unavailable"} and described["backend"] is None
        why = f"this Python environment lacks {', '.join(lacking[:-1])} and {lacking[-1]}" if len(lacking) > 1 else \
            f"this Python environment lacks {lacking[0]}"
        if lacking == LACKING:                                       # the app's venv: numcodecs alone of them
            assert why == "this Python environment lacks scipy, scikit-learn, igraph, umap-learn and numba"
        assert m.unavailable_why() == why and m.log_info()["why_not"] == why
        run.add_frame()                                              # frames come: nothing is started for them
        run.meta["state"] = "running"
        with pytest.raises(AnalysisError) as refused:
            m.configure("r1", seq=0)
        assert refused.value.status == 503 and str(refused.value) == f"Clusters and the UMAP can't run here: {why}"
        for ask, what in ((lambda: m.request_deeper("r1", 0, 1), "Go deeper"),
                          (lambda: m.submit_compare({"run": "r1"}), "Compare"),
                          (lambda: m.configure_qc("r1", enabled=True), "The mixing check")):
            with pytest.raises(AnalysisError, match=rf"^{what} needs Clusters and the UMAP, which can't run here: "):
                ask()
        time.sleep(6)                                                # longer than a failed start's retry (5 s)
        assert len(started) == 1 and m.process is None and m.thread.is_alive()
        events = [text for _, _, text in m.events]
        assert events.count(f"Clusters & UMAP are off for this session: {why}") == 1, events
        assert m.describe("r1")["status"] == {"state": "unavailable"}
    finally:
        m.close()
        m.thread.join(5)
    # The worker's own sentence keeps its reason alone where no package is named (its libraries do not load).
    m = fake_worker(tmp_path, monkeypatch)
    m._turn_off({"ok": False, "ready": False, "lasting": True, "missing": [],
                 "error": "Clusters & UMAP cannot run in this Python environment: its CPU libraries could not be loaded "
                          "(ImportError: Numba needs NumPy 2.5 or less. Got NumPy 2.6.)"})
    assert not m.available and m.describe("r1")["missing"] == []
    assert m.unavailable_why() == ("its CPU libraries could not be loaded (ImportError: Numba needs NumPy 2.5 or less. "
                                   "Got NumPy 2.6.)")
    with pytest.raises(AnalysisError, match="^Clusters and the UMAP can't run here: its CPU libraries"):
        m._ensure_worker()


def test_a_worker_that_cannot_be_started_never_stops_the_scheduler(tmp_path, monkeypatch):
    """[analysis] python is a file that cannot run (or the state folder cannot be written): the start fails with the
    one error every caller handles, said once in the log; the scheduler thread goes on, an early start is tried again
    after PRESTART_RETRY_S, and a frame asked for later is still answered (its status says the failure)."""
    from sopa_companion.engine import analysis as A
    python = tmp_path / "python"
    python.write_text("not a program")
    python.chmod(0o644)
    monkeypatch.setattr(A, "PRESTART_RETRY_S", 0.1)
    run = FakeRun(cells=3000, frames=0, state="preparing", directory=tmp_path / "run")
    run.live.mkdir(parents=True)
    (run.live / "transcripts.bin.gz").write_bytes(b"")
    run.info = {"cells": 3000, "z": {"grid_origin": 0.0, "grid_spacing": 1.0}}
    config = {"analysis": {"python": str(python)}, "paths": {"state_dir": str(tmp_path / "state")}}
    m = AnalysisManager(config, FakeRuns(run), start=False)
    attempts, ensure = [], m._ensure_worker
    m._ensure_worker = lambda: (attempts.append(time.monotonic()), ensure())[1]
    m.thread.start()
    try:
        wait_for(lambda: len(attempts) >= 3, timeout=10)
        assert m.thread.is_alive() and m.process is None
        events = [text for _, _, text in m.events]
        noted = [text for text in events if text.startswith("The analysis worker could not start: PermissionError")]
        assert len(noted) == 1, events
        with run.lock:                                               # the run's first frame, asked for
            run.frames.append({**counted_frame(0), "file": "000000.bin", "nx": 1, "ny": 1, "nz": 1, "x0": 0.0,
                               "y0": 0.0, "voxel": 1.0})
        run.meta["state"] = "running"
        m.configure("r1", seq=0)
        wait_for(lambda: m.describe("r1")["status"].get("state") == "error", timeout=10)
        error = m.describe("r1")["status"]["error"]
        assert "could not start" in error or "failed recently" in error, error
        assert m.thread.is_alive()
    finally:
        m.close()
        m.thread.join(5)
    with pytest.raises(AnalysisError, match="^The analysis worker could not start: PermissionError"):
        AnalysisManager(config, FakeRuns(), start=False)._ensure_worker()


def test_a_manager_needs_no_cache_dir(tmp_path):
    """The CPU worker's compiled code goes to the engine's cache folder, or the state folder's where a hand-made
    config has none."""
    from sopa_companion.analysis.backend import CPU_CACHE_ENV
    m = AnalysisManager({"analysis": {"python": sys.executable}, "paths": {"state_dir": str(tmp_path)}}, FakeRuns(),
                        start=False)
    assert m._worker_env()[CPU_CACHE_ENV] == str(tmp_path / "cache" / "numba")
    m = AnalysisManager({"analysis": {"python": sys.executable},
                         "paths": {"state_dir": str(tmp_path), "cache_dir": str(tmp_path / "var" / "cache")}},
                        FakeRuns(), start=False)
    assert m._worker_env()[CPU_CACHE_ENV] == str(tmp_path / "var" / "cache" / "numba")
    assert not (tmp_path / "cache").exists() and not (tmp_path / "var").exists()       # nothing made here


def test_the_cpu_caps_genes_used(tmp_path, monkeypatch):
    """On the CPU, Genes used stops at CPU_MAX_GENES (its PCA grows with the genes squared): the engine sends that as the
    field's limit and refuses more, changing nothing; on the GPU it goes up to 10,000."""
    from sopa_companion.engine import analysis as A
    assert A.CPU_MAX_GENES == 5000 and A.LIMITS["n_top_genes"]["max"] == 10000
    for backend, most in (("cpu", 5000), ("gpu", 10000)):
        monkeypatch.setenv("FAKE_ANALYSIS_BACKEND", backend)
        m = fake_worker(tmp_path / backend, monkeypatch, FakeRun(cells=3000, frames=2, state="done"))
        try:
            assert m.describe("r1")["limits"]["n_top_genes"]["max"] == 10000      # before the worker's hello
            m._ensure_worker()
            assert m.backend == backend and m.describe("r1")["limits"]["n_top_genes"]["max"] == most
            version = m.version
            if backend == "cpu":
                with pytest.raises(AnalysisError) as refused:
                    m.configure("r1", params={"n_top_genes": 6000})
                assert refused.value.status == 422 and str(refused.value) == "n_top_genes must be from 50 to 5000"
                assert m.version == version and m.settings_of("r1")["n_top_genes"] == 2000
                assert m.configure("r1", params={"n_top_genes": 5000})["settings"]["n_top_genes"] == 5000
            else:
                assert m.configure("r1", params={"n_top_genes": 6000})["settings"]["n_top_genes"] == 6000
        finally:
            m.close()


def test_a_default_above_the_cpus_limit_is_taken_so_reset_goes_ahead(tmp_path, monkeypatch):
    """[analysis] n_top_genes above the CPU's limit (8,000, valid on the GPU): on a CPU worker the page's Reset, which
    sends the defaults, and an Apply that carries that default unchanged both go ahead (the worker holds to its own
    limit whatever it is sent); any other value above the limit is still refused."""
    monkeypatch.setenv("FAKE_ANALYSIS_BACKEND", "cpu")
    m = fake_worker(tmp_path, monkeypatch, FakeRun(cells=3000, frames=2, state="done"), n_top_genes=8000)
    try:
        m._ensure_worker()
        view = m.describe("r1")
        assert view["limits"]["n_top_genes"]["max"] == 5000 and view["defaults"]["n_top_genes"] == 8000
        applied = m.configure("r1", params={**m.defaults, "resolution": 1.5})      # Apply, genes used left as it is
        assert applied["settings"] == {**m.defaults, "resolution": 1.5} and applied["changed"]
        reset = m.configure("r1", params=dict(m.defaults))                        # Reset: the defaults
        assert reset["settings"] == m.defaults and not reset["changed"] and m.prefs == {}
        version = m.version
        with pytest.raises(AnalysisError) as refused:
            m.configure("r1", params={"n_top_genes": 7000})
        assert refused.value.status == 422 and str(refused.value) == "n_top_genes must be from 50 to 5000"
        assert m.version == version
    finally:
        m.close()


def test_the_cpu_worker_spaces_its_passes(tmp_path, monkeypatch):
    """min_interval_s = 0 is back to back on a GPU; the CPU worker spaces its passes by "auto" (the run's cells /
    15,000 s, 0.5-6 s) instead, which would keep its cores busy for the whole run otherwise. A number of seconds, or
    "auto", holds on both."""
    run = FakeRun(cells=40070)
    monkeypatch.setenv("FAKE_ANALYSIS_BACKEND", "cpu")
    m = fake_worker(tmp_path, monkeypatch, run)
    try:
        assert m.interval(run) == 0                                  # before a hello: the GPU's
        m._ensure_worker()
        assert m.backend == "cpu" and m.interval(run) == auto_interval(40070) == pytest.approx(40070 / 15000)
    finally:
        m.close()
    for setting, gpu, cpu in ((0, 0, auto_interval(40070)), (3, 3, 3), ("auto", auto_interval(40070),
                                                                          auto_interval(40070))):
        m = manager(tmp_path, run, min_interval_s=setting)
        for backend, expected in (("gpu", gpu), ("cpu", cpu)):
            m.backend = backend                                     # what the worker's hello said
            assert m.interval(run) == expected, (setting, backend)
    assert manager(tmp_path, FakeRun(cells=3000)).interval(FakeRun(cells=3000)) == 0
    small = manager(tmp_path, FakeRun(cells=3000))
    small.backend = "cpu"
    assert small.interval(FakeRun(cells=3000)) == 0.5


def test_the_worker_starts_with_the_first_active_run(tmp_path, monkeypatch):
    """The worker starts while the first active run prepares, before its first frame (its imports and warm-up then run
    meanwhile): the log says where it runs. A stain run starts nothing; a start that failed is not tried early again
    for PRESTART_RETRY_S (a job still starts it as before)."""
    from sopa_companion.engine import analysis as A
    monkeypatch.setenv("FAKE_ANALYSIS_BACKEND", "cpu")
    started = worker_starts(monkeypatch)
    run = FakeRun(cells=3000, frames=0, state="preparing")
    m = fake_worker(tmp_path, monkeypatch, run, start=True)
    try:
        wait_for(lambda: m.process is not None and m.worker_since is not None, timeout=20)
        assert run.frames == [] and len(started) == 1 and m.backend == "cpu"
        events = [text for _, _, text in m.events]
        assert any(text.startswith(f"Analysis worker started on the CPU with {A.worker_threads()} threads (pid "
                                   f"{m.process.pid}) in ") and text.endswith(": no CUDA device is visible "
                                                                               "(CUDA_VISIBLE_DEVICES='')")
                   for text in events), events
        assert m.log_info()["backend_why"] == "no CUDA device is visible (CUDA_VISIBLE_DEVICES='')"
        # A worker that ends by itself while nothing is asked of it (a crash in its warm-up) is not started again at
        # once: one that keeps dying would be started again and again.
        m.process.kill()
        wait_for(lambda: m.process is None, timeout=10)
        time.sleep(1.5)
        assert len(started) == 1 and m.prestart_failed > -math.inf
    finally:
        m.close()
        m.thread.join(5)
    stain = FakeRun("s1", cells=3000, frames=0, state="preparing")
    stain.meta["kind"] = "stain"
    m = fake_worker(tmp_path / "stain", monkeypatch, stain, start=True)
    try:
        time.sleep(1.5)
        assert m.process is None and len(started) == 1
    finally:
        m.close()
        m.thread.join(5)
    # A worker that does not start (its module is not there): not started early again for PRESTART_RETRY_S.
    failing = worker_starts(monkeypatch, "no_such_worker")
    m = fake_worker(tmp_path / "failing", monkeypatch, FakeRun(cells=3000, frames=0, state="preparing"), start=True,
                    worker_module="no_such_worker")
    try:
        wait_for(lambda: m.prestart_failed > -math.inf, timeout=20)
        # A hello that says nothing lasting (a worker of the previous version, a GPU that cannot be reached): retried as
        # before, never turned off for the session.
        assert m.available and m.lasting is None
        assert "The analysis worker could not start: the GPU worker exited (code 1)" in [t for _, _, t in m.events]
        m.failed_at = 0.0                                            # a job could start it again at once
        time.sleep(1.0)
        assert len(failing) == 1
        monkeypatch.setattr(A, "PRESTART_RETRY_S", 0.1)
        with m.lock:
            m.wake.notify_all()
        wait_for(lambda: len(failing) >= 2, timeout=20)
    finally:
        m.close()
        m.thread.join(5)


def test_the_workers_environment(tmp_path, monkeypatch):
    """The worker's thread pools (at most 4 of the engine's cores), [analysis] backend and the CPU path's code folder
    (one the engine's own environment names is kept); nothing else is added for the CPU path (the GPU's environment
    changes by those two variables only)."""
    from sopa_companion.analysis.backend import BACKEND_ENV, CPU_CACHE_ENV
    from sopa_companion.engine import children
    from sopa_companion.engine.config import cores
    monkeypatch.delenv(CPU_CACHE_ENV, raising=False)
    m = manager(tmp_path, backend="cpu")
    env = m._worker_env()
    threads = str(min(4, cores()))
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == threads
    assert env[BACKEND_ENV] == "cpu" and env[CPU_CACHE_ENV] == str(tmp_path / "cache" / "numba")
    base = children.env(children.kind(sys.executable), gpu=0)
    added = set(env) - set(base)
    assert added <= {"PYTHONUNBUFFERED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", BACKEND_ENV,
                     CPU_CACHE_ENV}, added
    assert "NUMBA_CACHE_DIR" not in added and "OMP_WAIT_POLICY" not in added
    assert manager(tmp_path)._worker_env()[BACKEND_ENV] == "auto"
    monkeypatch.setenv(CPU_CACHE_ENV, str(tmp_path / "mine"))
    assert manager(tmp_path)._worker_env()[CPU_CACHE_ENV] == str(tmp_path / "mine")


def test_a_bad_backend_setting_stops_the_engine(tmp_path):
    for bad in ("tpu", "GPU", "", 1, None):
        with pytest.raises(ValueError, match=r'^\[analysis\] backend must be "auto", "gpu" or "cpu"'):
            manager(tmp_path, backend=bad)
    assert manager(tmp_path, backend="gpu").backend_setting == "gpu" and manager(tmp_path).backend_setting == "auto"


def test_describe_and_the_log_name_the_backend(tmp_path, monkeypatch):
    """describe says the running worker's backend (None before its hello) and, where Clusters & UMAP cannot run, the
    packages missing; the Log's info says the backend, why, its threads, the setting and why they cannot run."""
    monkeypatch.setenv("FAKE_ANALYSIS_BACKEND", "cpu")
    m = fake_worker(tmp_path, monkeypatch, FakeRun(cells=3000, frames=1), backend="auto")
    try:
        described, info = m.describe("r1"), m.log_info()
        assert described["backend"] is None and described["missing"] == [] and info["why_not"] is None
        m._ensure_worker()
        described, info = m.describe("r1"), m.log_info("r1")
        assert described["backend"] == "cpu" and info["backend"] == "cpu" and info["backend_setting"] == "auto"
        assert info["backend_why"] == "no CUDA device is visible (CUDA_VISIBLE_DEVICES='')" and info["threads"] is None
    finally:
        m.close()
    # [analysis] python "auto" that found this Python lacking (the engine's settings say so): known before any start.
    located = {"path": "", "how": "auto", "missing": ["scipy", "sklearn", "umap"], "problem": "", "fix": ""}
    m = AnalysisManager({"analysis": {"python": ""}, "paths": {"state_dir": str(tmp_path)},
                         "locate": {"analysis": located}}, FakeRuns(FakeRun()), start=False)
    assert not m.available and m.describe("r1")["missing"] == ["scipy", "scikit-learn", "umap-learn"]
    assert m.log_info()["why_not"] == "this Python environment lacks scipy, scikit-learn and umap-learn"
    for python, why in (("", "python under [analysis] is off"), ("off", "python under [analysis] is off"),
                        (str(tmp_path / "gone"), f"the python set under [analysis] ({tmp_path / 'gone'}) is not there")):
        m = AnalysisManager({"analysis": {"python": python}, "paths": {"state_dir": str(tmp_path)}}, FakeRuns(),
                            start=False)
        assert m.unavailable_why() == why and m.log_info()["why_not"] == why and m.describe("r1")["missing"] == []


def disk_run(folder, **kwargs):
    """A real engine Run over a folder written by write_run: a sample's run started from its cells alone (a start
    line: engine/qc_rules.py start_of "cells")."""
    from sopa_companion.engine.runs import Run
    write_run(folder, **kwargs)
    run = Run(folder, {"id": "r1", "state": "running", "box": [X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL],
                       "mode": "detail", "sample_kind": "zarr",
                       "prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": None}})
    run.read_new()
    return run


def stub_worker(m, calls, fail=()):
    """m._call without a worker process: writes what a worker writes and replies as it does; ops in `fail` fail. The
    count it read is said when it is not the averaged one (a summary without one is on the averaged count)."""
    def call(request):
        calls.append(request)
        reply, errors = {"run": request["run"], "seq": request.get("seq")}, {}
        said = {"count": request["count"]} if request.get("count", "averaged") != "averaged" else {}
        ops = ["qc_start"] if request["op"] == "qc_start" else request["ops"]
        for op in ops:
            if op in fail:
                errors[op] = "ValueError: boom"
            elif op == "umap":
                Path(request["out_json"]).write_text(json.dumps({"seq": request["seq"], **said}))
                Path(request["out_cells"]).write_bytes(b"")
                reply["summary"] = {"seq": request["seq"], **said}
            else:
                qc = request["qc"]
                if qc.get("out_bins"):
                    W.write_bins(qc["out_bins"], {"bin": np.array([0], np.int32), "cells": np.array([3], np.int32)},
                                 qc["grid"])
                Path(qc["out_cells"]).write_bytes(b"\0")
                if qc.get("out_cells_e"):
                    Path(qc["out_cells_e"]).write_bytes(b"\0")
                reply["qc"] = {"kind": "start" if op == "qc_start" else "frame",
                               "files": {"bins": qc.get("out_bins"), "cells": qc["out_cells"]}, **said}
        reply["ok"] = not errors
        if errors:
            reply["errors"], reply["error"] = errors, next(iter(errors.values()))
        return reply
    m._call = call


def test_cached_results_are_reused_op_by_op_and_reach_the_provider(tmp_path):
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run)
    provider = Provider()
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    assert m._process(Job("r1", 1, ("umap", "qc"), "request", qc)) is None
    request = calls[-1]
    assert request["ops"] == ["umap", "qc"] and request["cells_file"] == str(run.live / "frames/000001.cells.zst")
    bins_file = Path(request["qc"].pop("out_bins"))
    assert request["qc"] == {"mixed_at": 0.30, "judge_min": 100, "fit_min": 20, "model": "/models/m1",
                             "model_id": "0123456789abcdef", "start": None,
                             "grid": {"x0": X0, "y0": Y0, "bin": 50.0, "nx": 1, "ny": 1},
                             "out_cells": str(qc.dir / "000001.qc.bin")}
    # The frame's per-bin sums (the areas' input) go through a transient file on the RAM disk, never the run folder.
    assert bins_file.parent == m.bins_dir and bins_file.parent.name.startswith(f"sopa-companion-bins-{os.getpid()}-")
    if os.path.isdir("/dev/shm"):
        assert str(bins_file).startswith("/dev/shm/")
    record = provider.records[-1]
    assert (record["kind"], record["seq"], record["source"], record["single"]) == ("frame", 1, "request", True)
    assert record["frame"]["live"] == {"cell-compactness": 0.04} and record["dir"] == str(qc.dir)
    assert record["bins"]["cells"].tolist() == [3] and record["bins"]["grid"].tolist() == [X0, Y0, 50.0, 1, 1]
    assert not bins_file.exists() and not list(qc.dir.glob("*.bins.npz"))     # read, then deleted
    assert 1 in qc.done and m.results["r1"][1] == {"seq": 1} and m.qc_status["r1"] == {"state": "ready", "seq": 1}
    # Frame 2, the run's end (proseg counted it with f), is read on the count proseg saves: its result on disk is.
    m._paths(run, 2)[0].write_text(json.dumps({"seq": 2, "from": "disk", "count": "sampled"}))
    m._paths(run, 2)[1].write_bytes(b"")
    m._process(Job("r1", 2, ("umap", "qc"), "live", qc))
    assert calls[-1]["ops"] == ["qc"] and m.results["r1"][2]["from"] == "disk"   # a cached UMAP never skips the check
    assert calls[-1]["count"] == "sampled" and provider.records[-1]["count"] == "sampled"
    assert provider.records[-1]["single"] is False
    (qc.dir / "000000.qc.json").write_text("{}")
    (qc.dir / "000000.qc.bin").write_bytes(b"\0")
    m._process(Job("r1", 0, ("umap", "qc"), "live", qc))
    assert calls[-1]["ops"] == ["umap"] and 0 in qc.done      # ... nor the reverse
    count = len(calls)
    m._process(Job("r1", 0, ("umap", "qc"), "request", qc))  # both on disk now: no worker call at all
    assert len(calls) == count
    # A later session finds what this one left complete.
    m2 = manager(tmp_path, run)
    m2.attach_qc(provider)
    m2.configure_qc("r1", enabled=True)
    assert m2._qc_state("r1", m2._qc_models()).done == {0: None}, "found on disk: its count read once asked"


def test_the_frames_per_bin_sums_never_linger(tmp_path):
    """The transient folder of the frames' per-bin sums: one per engine (a crashed engine's is removed by the next),
    nothing left in it when the worker dies mid-pass, gone with the engine."""
    stale, alive = tmp_path / f"{BINS_PREFIX}999999999-x", tmp_path / f"{BINS_PREFIX}{os.getpid()}-y"
    # The previous version's engines named theirs proseg-live-bins-<pid>-*: a dead one's goes too, a live one's stays.
    old_stale, old_alive = tmp_path / "proseg-live-bins-999999999-x", tmp_path / f"proseg-live-bins-{os.getpid()}-z"
    for folder in (stale, alive, old_stale, old_alive):
        folder.mkdir()
    made = transient_dir(tmp_path)
    assert BINS_PREFIX == "sopa-companion-bins-"
    assert not stale.exists() and alive.exists() and made.parent == tmp_path
    assert not old_stale.exists() and old_alive.exists()
    assert made.name.startswith(f"{BINS_PREFIX}{os.getpid()}-") and made != alive
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run)
    m.attach_qc(Provider())
    m.configure_qc("r1", enabled=True)

    def call(request):                                        # writes the file, then the worker is gone
        W.write_bins(request["qc"]["out_bins"], {"bin": np.array([0], np.int32), "cells": np.array([3], np.int32)},
                     request["qc"]["grid"])
        raise AnalysisError("the GPU worker stopped")
    m._call = call
    qc = m._qc_state("r1", m._qc_models())
    with pytest.raises(AnalysisError):
        m._process(Job("r1", 1, ("qc",), "live", qc))
    assert m.bins_dir.is_dir() and not list(m.bins_dir.iterdir())
    m.close()
    assert not m.bins_dir.exists()


def test_the_start_line_pairs_later_frames_and_failures_back_off(tmp_path):
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run, qc={"judge_min_counts": 50})
    m.configure("r1", enabled=False)                         # Clusters & UMAP switched off by a tool: the check alone
    provider = Provider()
    m.attach_qc(provider)
    m.configure_qc("r1", enabled=True)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    assert m._process(Job("r1", None, ("qc_start",), "start", qc)) is None
    request = calls[-1]
    assert request["op"] == "qc_start" and request["qc"]["judge_min"] == 50 and request["seq"] is None
    assert request["prior"] == {"file": str(run.live / "prior.bin.zst"), "format": "dense", "nx": NX, "ny": NY,
                                "nz": NZ, "x0": X0, "y0": Y0, "voxel": VOXEL}
    assert request["qc"]["out_start"] == str(qc.dir / "start.npz") and qc.start == "done"
    assert provider.records[-1]["kind"] == "start" and provider.records[-1]["bins"]["cells"].tolist() == [3]
    assert (qc.dir / "start.bins.npz").exists()                # the start's bins are kept
    m._process(Job("r1", 1, ("qc",), "live", qc))
    assert calls[-1]["qc"]["start"] == str(qc.dir / "start.npz")   # later frames are paired with it
    # The check fails; the UMAP of the same pass is kept; automatic passes of the run wait, doubling.
    now = [500.0]
    m.clock = lambda: now[0]
    stub_worker(m, calls, fail={"qc"})
    assert m._process(Job("r1", 2, ("umap", "qc"), "live", qc)) == "ValueError: boom"
    assert m.status["r1"] == {"state": "ready", "seq": 2} and m.qc_status["r1"]["error"] == "ValueError: boom"
    assert qc.retry_at == 501.0 and provider.records[-1]["kind"] == "error" and 2 not in qc.done
    m._process(Job("r1", 2, ("qc",), "request", qc))
    assert qc.retry_at == 502.0
    run.frames.append({**run.frames[-1], "seq": 3})            # counted as the frame before (its line merged)
    run.counted.append(3)
    assert m._next_job() == (None, pytest.approx(2.0))       # no automatic pass of the check until then
    m.configure_qc("r1", seq=2)
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.kind == "request"        # ... but a request still goes at once
    # A start line that keeps failing is given up after START_ATTEMPTS.
    stub_worker(m, calls, fail={"qc_start"})
    fresh = m._qc_state("r1", {"r1": ("9999aaaa", "/models/m3")})
    for attempt in range(START_ATTEMPTS):
        assert fresh.start == "pending"
        assert m._process(Job("r1", None, ("qc_start",), "start", fresh)) == "ValueError: boom"
    assert fresh.start == "error" and m.qc_status["r1"]["start"]


def test_the_check_runs_through_the_worker_process(tmp_path, monkeypatch):
    """The scheduler and the worker process (the fake worker: the worker's own code for the check) on a run whose
    frames carry proseg's own count: every frame is read from its counts file, the start line from the init frame's."""
    from numcodecs import Zstd
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = disk_run(tmp_path / "run", frames=4)
    run.meta["state"] = "done"
    make_model(tmp_path / "model" / "encoder.npz", MODEL_GENES)
    provider = Provider(model=("c0ffee1234", str(tmp_path / "model")))
    config = {"analysis": {"python": sys.executable, "worker_module": "fake_analysis_worker"},
              "paths": {"state_dir": str(tmp_path / "state")},
              "qc": {"judge_min_counts": JUDGE, "fit_min_counts": 5, "bin_um_detail": 8}}
    m = AnalysisManager(config, FakeRuns(run), start=False)
    m.attach_qc(provider)
    m.thread.start()
    try:
        m.configure_qc("r1", enabled=True)
        wait_for(lambda: m.describe_qc("r1")["start"] == "done" and len(m.describe_qc("r1")["seqs"]) == 4, timeout=60)
        m.configure("r1", seq=3)                             # clusters for a frame already scored: UMAP alone
        wait_for(lambda: 3 in m.describe("r1")["seqs"], timeout=30)
    finally:
        m.close()
        m.thread.join(5)
    order = [(r["kind"], r["seq"]) for r in provider.records]
    assert order == [("frame", 3), ("start", None), ("frame", 0), ("frame", 2), ("frame", 1)], order
    records = {r["seq"]: r for r in provider.records}
    newest, backfilled = records[3], records[0]
    assert newest["source"] == "live" and not newest["single"] and not newest["summary"]["has_start"]
    assert backfilled["source"] == "backfill" and backfilled["single"] and backfilled["summary"]["has_start"]
    # Frame 3 went before the start line: the cell that vanished has neither a centroid nor a start to stand in.
    assert newest["summary"]["ncells"] == NCELLS and newest["summary"]["located"] == NCELLS - 1
    assert records[2]["summary"]["located"] == NCELLS and records[2]["summary"]["growth"]["lost_half"] == 1
    # Every scored frame's per-bin sums reach the provider (the areas' input), the start line's too (and are kept).
    assert newest["bins"]["cells"].sum() == NCELLS - 1 and records[None]["bins"]["cells"].sum() == NCELLS
    assert records[2]["bins"]["judged"].sum() == records[2]["summary"]["judged"]
    assert newest["bins"]["grid"].tolist() == [X0, Y0, 8.0, 4, 3]
    assert newest["summary"]["grid"] == {"x0": X0, "y0": Y0, "bin": 8.0, "nx": 4, "ny": 3}
    qc = Path(newest["dir"])
    assert qc == run.dir / "qc" / "c0ffee12.pc1" and not list(qc.glob("*.bins.npz.tmp"))   # proseg's count's folder
    assert sorted(p.name for p in qc.glob("*.bins.npz")) == ["start.bins.npz"]   # frames' went through the RAM disk
    assert m.bins_dir is not None and not m.bins_dir.exists()                      # ... and are gone with the engine
    assert len(Zstd().decode((qc / "000003.qc.bin").read_bytes())) == NCELLS
    assert records[None]["summary"]["footprints"] == NCELLS and m.results["r1"][3]["nclusters"] == 2
    # proseg's own count, every one of them: the start line's from the init frame, each frame against its start block.
    # Frame 3, the run's end (counted with f), on the count proseg saves; every other frame and the start averaged.
    assert all((r["summary"]["basis"], r["summary"]["count"]) == ("proseg", "sampled" if r["seq"] == 3 else "averaged")
               for r in provider.records)
    assert records[2]["summary"]["start_block"]["iteration"] == 2 and records[2]["summary"]["matched"]["paired"] > 0


# ---- Clusters & UMAP settings ----------------------------------------------------------------------------------------
KEYS = ("resolution", "n_neighbors", "min_dist", "min_counts", "n_top_genes")


def settings_manager(tmp_path, *runs, **settings):
    """A manager with a data folder of its own: the settings new runs start with are kept there."""
    config = {"analysis": {"python": sys.executable, **settings},
              "paths": {"state_dir": str(tmp_path / "state"), "data_dir": str(tmp_path / "var")}}
    return AnalysisManager(config, FakeRuns(*runs), start=False, prestart=False)


def folder_run(tmp_path, run_id, **kwargs):
    run = FakeRun(run_id, directory=tmp_path / "runs" / run_id, **kwargs)
    run.dir.mkdir(parents=True)
    return run


def test_cluster_settings_come_from_the_worker_and_the_config_by_the_same_rules(tmp_path):
    assert tuple(SETTINGS) == KEYS
    m = manager(tmp_path, FakeRun())
    assert m.defaults == {k: W.DEFAULTS[k] for k in KEYS} and m.fixed == {}
    m = manager(tmp_path, FakeRun(), resolution=1.5, n_neighbors=30.0, n_top_genes=500, n_pcs=20)
    assert m.defaults == {**{k: W.DEFAULTS[k] for k in KEYS}, "resolution": 1.5, "n_neighbors": 30, "n_top_genes": 500}
    assert type(m.defaults["n_neighbors"]) is int and type(m.defaults["resolution"]) is float
    assert m.fixed == {"n_pcs": 20}                          # passed to the worker as it is
    described = m.describe("r1")
    assert described["params"] == {"n_pcs": 20, **m.defaults} and described["settings"] == m.defaults
    assert described["defaults"] == m.defaults and described["changed"] is False and described["generation"] == 0
    assert described["limits"] == {"resolution": {"min": 0.1, "max": 3.0, "step": 0.05, "whole": False},
                                   "n_neighbors": {"min": 5, "max": 50, "step": 1, "whole": True},
                                   "min_dist": {"min": 0.0, "max": 1.0, "step": 0.05, "whole": False},
                                   "min_counts": {"min": 1, "max": 500, "step": 1, "whole": True},
                                   "n_top_genes": {"min": 50, "max": 10000, "step": 50, "whole": True}}
    for key, bad in (("resolution", 0), ("resolution", "high"), ("n_neighbors", 15.5), ("n_neighbors", True),
                     ("min_dist", float("inf")), ("min_counts", 0), ("n_top_genes", 49), ("n_top_genes", 10050)):
        with pytest.raises(ValueError, match=rf"^\[analysis\] {key} must "):
            manager(tmp_path, FakeRun(), **{key: bad})


def test_a_change_starts_a_new_generation_and_asks_for_the_frame_on_screen(tmp_path):
    run = FakeRun(cells=3000, frames=4)                      # under /nonexistent: nothing is written for it
    m = manager(tmp_path, run)
    m._store("r1", 3, {"seq": 3})
    m.too_few["r1"], m.started["r1"] = 20, 5.0
    version = m.version
    described = m.configure("r1", seq=1, params={"resolution": 1.5, "n_neighbors": 30.0})
    assert m.gen == {"r1": 1} and not m.results.get("r1") and m.status["r1"] == {"state": "idle"}
    assert "r1" not in m.too_few and "r1" not in m.started and m.requested == {"r1": 1}
    assert m.version == version + 1 and "r1" in m.sweeps
    settings = {**m.defaults, "resolution": 1.5, "n_neighbors": 30}
    assert type(m.settings_of("r1")["n_neighbors"]) is int                  # 30.0 counts as 30
    assert described["settings"] == settings and described["defaults"] == {k: W.DEFAULTS[k] for k in KEYS}
    assert described["changed"] is True and described["generation"] == 1 and described["params"] == settings
    assert described["limits"]["n_neighbors"] == {"min": 5, "max": 50, "step": 1, "whole": True}
    assert m.prefs == {"resolution": 1.5, "n_neighbors": 30}               # new runs start with them (in memory here)
    said = lambda: [text for _, run_id, text in m.events if run_id == "r1"]
    assert said() == ["Clusters & UMAP settings changed: resolution 1 -> 1.5, n_neighbors 15 -> 30"]
    m.configure("r1", params={"resolution": 1.5})            # the same values: nothing changes
    assert m.gen == {"r1": 1} and len(said()) == 1
    m.configure("r1", params={"min_dist": 0.5})              # a part: merged over the run's settings
    assert m.settings_of("r1") == {**settings, "min_dist": 0.5} and m.gen == {"r1": 2}
    assert said()[-1] == "Clusters & UMAP settings changed: min_dist 0.3 -> 0.5"
    back = m.configure("r1", params=dict(m.defaults))        # Reset: the defaults again, and for new runs too
    assert "r1" not in m.run_params and m.gen == {"r1": 3} and back["changed"] is False and m.prefs == {}
    assert said()[-1] == ("Clusters & UMAP settings changed: resolution 1.5 -> 1, n_neighbors 30 -> 15, min_dist 0.5 -> "
                          "0.3; back to the defaults")
    # The run's folder does not exist (/nonexistent): nothing was written, and nothing failed to be.
    assert len(said()) == 3 and not [text for _, _, text in m.events if "could not" in text]


def test_results_of_older_settings_are_dropped_and_their_files_removed(tmp_path):
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run)
    folder, calls = run.dir / "analysis" / "pc1", []

    def changing(request):                                   # the settings change while the worker runs
        calls.append(request)
        Path(request["out_json"]).write_text(json.dumps({"seq": request["seq"]}))
        Path(request["out_cells"]).write_bytes(b"")
        Path(request["out_json"] + ".tmp").write_text("{}")
        m.configure("r1", params={"resolution": 2.0})
        return {"run": request["run"], "seq": request["seq"], "ok": True, "summary": {"seq": request["seq"]}}
    m._call = changing
    assert m._process(Job("r1", 1, ("umap",), "request")) is None
    outdated = Path(calls[0]["out_json"])
    assert outdated.name == "000001.json" and calls[0]["params"]["resolution"] == 1.0
    assert not outdated.exists() and not Path(calls[0]["out_cells"]).exists() and not Path(f"{outdated}.tmp").exists()
    assert not m.results.get("r1") and m.status["r1"] == {"state": "idle"} and m.gen == {"r1": 1}

    def too_few(request):                                    # ... its failure is no news either
        calls.append(request)
        m.configure("r1", params={"resolution": 2.2})
        message = "TooFewCells: only 20 cells have >= 10 transcripts in this frame"
        return {"run": request["run"], "seq": request["seq"], "ok": False, "errors": {"umap": message}, "error": message,
                "reasons": {"umap": "too_few_cells"}}
    m._call = too_few
    assert m._process(Job("r1", 1, ("umap",), "request")) is None
    assert "r1" not in m.too_few and m.status["r1"] == {"state": "idle"} and m.gen == {"r1": 2}

    def raising(request):                                    # the worker timed out meanwhile
        calls.append(request)
        m.configure("r1", params={"resolution": 2.5})
        raise AnalysisError("the GPU worker timed out")
    m._call = raising
    job = Job("r1", 2, ("umap",), "request")
    with pytest.raises(AnalysisError) as failed:
        m._process(job)
    assert failed.value.generation == 2 and m.gen == {"r1": 3}
    m._failed(job, job.ops, str(failed.value), generation=failed.value.generation)
    assert m.status["r1"] == {"state": "idle"} and not [t for _, _, t in m.events if "failed for frame" in t]
    stub_worker(m, calls)                                    # the next pass: the run's own generation, its own files
    assert m._process(Job("r1", 2, ("umap",), "request")) is None
    assert Path(calls[-1]["out_json"]).name == "000002-s3.json" and calls[-1]["params"]["resolution"] == 2.5
    assert m.results["r1"][2] == {"seq": 2, "count": "sampled"} and m.status["r1"] == {"state": "ready", "seq": 2}
    # The sweep: files of every other generation, and only those; settings.json and the current ones stay.
    for name in ("000001.json", "000001.cells.bin", "000003-s1.json", "000003-s1.cells.bin.tmp", "000004-s9.json",
                 "notes.txt", "000002.json.bak"):
        (folder / name).write_text("x")
    target = tmp_path / "elsewhere.json"
    target.write_text("kept")
    (folder / "000005-s1.json").symlink_to(target)          # a link named like a result: the link goes, never its target
    (folder / "000006-s1.json").mkdir()                       # a folder: left alone
    m._sweep_generations("r1")
    assert sorted(p.name for p in folder.iterdir()) == ["000002-s3.cells.bin", "000002-s3.json", "000002.json.bak",
                                                        "000006-s1.json", "notes.txt"]
    assert target.read_text() == "kept"
    # An idle scheduler (no job at all) still sweeps.
    m.configure("r1", enabled=False)
    (folder / "000007.json").write_text("x")
    m.thread.start()
    try:
        with m.lock:
            m.sweeps.add("r1")
            m.wake.notify_all()
        wait_for(lambda: not (folder / "000007.json").exists(), timeout=5)
    finally:
        m.close()
        m.thread.join(2)


def test_a_failure_the_pass_recorded_keeps_its_reason(tmp_path):
    """A clusters-only pass that found too few cells recorded its status with the reason; the scheduler's own failure
    handling set the status again without it (the page then took the run's small region for a failure)."""
    run = disk_run(tmp_path / "run")
    for f in run.frames:
        f["cells"] = 20
    m = manager(tmp_path, run)

    def call(request):
        message = "TooFewCells: only 20 cells have >= 10 transcripts in this frame"
        return {"run": request["run"], "seq": request["seq"], "ok": False, "errors": {"umap": message}, "error": message,
                "reasons": {"umap": "too_few_cells"}}
    failed, handled = [], m._failed
    m._call = call
    m._failed = lambda *args, **kw: (failed.append(kw), handled(*args, **kw))
    m.thread.start()
    try:
        wait_for(lambda: failed, timeout=10)
    finally:
        m.close()
        m.thread.join(2)
    assert failed[0]["recorded"] is True
    assert m.status["r1"]["reason"] == "too_few_cells" and m.describe("r1")["unavailable"] == "too_few_cells"


def test_new_runs_start_with_the_last_settings_applied(tmp_path):
    a = folder_run(tmp_path, "a")
    m = settings_manager(tmp_path, a)
    prefs = tmp_path / "var" / "analysis-settings.json"
    m.configure("a", params={"resolution": 1.5, "min_counts": 20})
    assert json.loads(prefs.read_text()) == {"v": 1, "settings": {"resolution": 1.5, "min_counts": 20}}
    assert json.loads((a.dir / "analysis" / "settings.json").read_text()) == {
        "v": 1, "settings": {**m.defaults, "resolution": 1.5, "min_counts": 20}, "generation": 1}
    b = folder_run(tmp_path, "b")
    again = settings_manager(tmp_path, a, b)                 # an engine started again
    assert again.prefs == {"resolution": 1.5, "min_counts": 20}
    assert again.describe("a")["generation"] == 1 and again.settings_of("a")["min_counts"] == 20
    again.run_created({"id": "b"})
    described = again.describe("b")
    assert described["generation"] == 1 and described["changed"] and described["settings"]["resolution"] == 1.5
    assert json.loads((b.dir / "analysis" / "settings.json").read_text())["generation"] == 1
    assert [t for _, r, t in again.events if r == "b"] == [
        "Clusters & UMAP start with the settings last applied: resolution 1.5, min_counts 20"]
    again.configure("b", params=dict(again.defaults))         # Reset: new runs follow the defaults again
    assert json.loads(prefs.read_text()) == {"v": 1, "settings": {}}
    c = folder_run(tmp_path, "c")
    again.runs.runs["c"] = c
    again.run_created({"id": "c"})
    assert again.describe("c")["generation"] == 0 and not (c.dir / "analysis").exists()
    prefs.write_text("{not json")                             # a broken file: the defaults, said once
    broken = settings_manager(tmp_path, a)
    assert broken.prefs == {}
    [(_, run_id, text)] = list(broken.events)
    assert run_id is None and text.startswith("The Clusters & UMAP settings for new runs could not be read (")
    assert text.endswith("): new runs start with the defaults")
    # A run without its settings.json reads as the defaults at generation 0: files of other settings are swept.
    d = folder_run(tmp_path, "d")
    (d.dir / "analysis" / "pc1").mkdir(parents=True)
    for name in ("000001-s1.json", "000001-s1.cells.bin", "000001.json", "000001.cells.bin"):
        (d.dir / "analysis" / "pc1" / name).write_text("x")
    broken.runs.runs["d"] = d
    assert broken.describe("d")["generation"] == 0 and "d" in broken.sweeps
    broken._sweep_generations("d")
    assert sorted(p.name for p in (d.dir / "analysis" / "pc1").iterdir()) == ["000001.cells.bin", "000001.json"]
    (d.dir / "analysis" / "settings.json").write_text(json.dumps({"v": 1, "settings": {"resolution": 9}, "generation": 2}))
    e = FakeRun("e", directory=d.dir)                         # a bad settings.json: the defaults, said once
    broken.runs.runs["e"] = e
    assert broken.describe("e")["generation"] == 0
    assert [t for _, r, t in broken.events if r == "e"] == [
        "The Clusters & UMAP settings of this run could not be read (resolution must be from 0.1 to 3): the defaults are "
        "used"]
    shutil.rmtree(d.dir)                                      # a deleted run's folder is never made again
    broken.configure("d", params={"resolution": 2.0})
    assert not d.dir.exists() and broken.gen["d"] == 2       # above the generation 1 files its folder held when read


def test_a_restarted_engine_keeps_a_runs_settings_and_its_files(tmp_path):
    run = disk_run(tmp_path / "run")
    m = manager(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    m.configure("r1", params={"resolution": 1.5})
    m.configure("r1", params={"resolution": 2.0})
    assert m._process(Job("r1", 1, ("umap",), "request")) is None
    (run.dir / "analysis" / "pc1" / "000001-s1.json").write_text(json.dumps({"seq": 1, "from": "generation 1"}))
    again = manager(tmp_path, run)                            # an engine started again
    assert again.generation("r1") == 2 and again.settings_of("r1")["resolution"] == 2.0
    assert again.summary("r1", 1) == again.summary("r1", 1, generation=2) == {"seq": 1}   # read from 000001-s2.json
    assert again.cells_path("r1", 1).name == "000001-s2.cells.bin"
    for read in (again.summary, again.cells_path):            # generation 1's file is there, never served
        with pytest.raises(AnalysisError) as missing:
            read("r1", 1, generation=1)
        assert missing.value.status == 404
    assert again.summary("r1", 1, 2) == {"seq": 1} and m.summary("r1", 1, 2) == {"seq": 1}


def test_a_change_before_the_sweep_never_takes_the_name_of_files_of_other_settings(tmp_path):
    """A run whose settings.json is lost (or older than its result files: a write that failed) reads back at a lower
    generation, and its files of other generations wait for the scheduler's sweep. A change made before that sweep is
    numbered above every generation found on disk, so its pass never reuses a result computed with other settings."""
    run = disk_run(tmp_path / "run")
    folder, calls = run.dir / "analysis" / "pc1", []
    folder.mkdir(parents=True)
    # An earlier session's result of generation 1 (resolution 2.5), and a write of generation 3 it left unfinished.
    (folder / "000001-s1.json").write_text(json.dumps({"seq": 1, "params": {"resolution": 2.5}}))
    (folder / "000001-s1.cells.bin").write_bytes(b"")
    (folder / "000002-s3.cells.bin.tmp").write_bytes(b"")
    m = manager(tmp_path, run)
    stub_worker(m, calls)
    assert m.describe("r1")["generation"] == 0 and "r1" in m.sweeps
    assert m.configure("r1", seq=1, params={"resolution": 1.5})["generation"] == 4     # Apply, before the sweep
    with m.lock:                                              # the scheduler's next turn: the sweep, then the pass
        sweeps, m.sweeps = m.sweeps, set()
    for run_id in sweeps:
        m._sweep_generations(run_id)
    assert m._process(Job("r1", 1, ("umap",), "request")) is None
    assert len(calls) == 1 and calls[0]["params"]["resolution"] == 1.5
    assert Path(calls[0]["out_json"]).name == "000001-s4.json"
    assert m.summary("r1", 1, generation=4) == {"seq": 1}
    assert sorted(p.name for p in folder.iterdir()) == ["000001-s4.cells.bin", "000001-s4.json"]
    # An older settings.json (generation 1; generation 2's was never written): its own files are served, a change goes
    # above generation 2's.
    (folder.parent / "settings.json").write_text(json.dumps({"v": 1, "settings": {**m.defaults, "resolution": 1.5},
                                                             "generation": 1}))
    for name in ("000001-s1.json", "000001-s2.json"):
        (folder / name).write_text(json.dumps({"seq": 1, "from": name}))
        (folder / name.replace(".json", ".cells.bin")).write_bytes(b"")
    again = manager(tmp_path, run)
    stub_worker(again, calls)
    assert again.generation("r1") == 1 and again.summary("r1", 1, generation=1) == {"seq": 1, "from": "000001-s1.json"}
    assert again.configure("r1", params={"resolution": 2.0})["generation"] == 5   # above 000001-s4 too
    assert again._process(Job("r1", 1, ("umap",), "request")) is None and len(calls) == 2
    assert Path(calls[-1]["out_json"]).name == "000001-s5.json" and again.summary("r1", 1) == {"seq": 1}
    assert json.loads((folder.parent / "settings.json").read_text())["generation"] == 5
    # Swept, nothing on disk is above the run's own generation: the next engine's change goes up by one, as always.
    again._sweep_generations("r1")
    assert sorted(p.name for p in folder.iterdir()) == ["000001-s5.cells.bin", "000001-s5.json"]
    third = manager(tmp_path, run)
    assert third.generation("r1") == 5 and third.configure("r1", params={"resolution": 2.2})["generation"] == 6


def test_the_last_of_two_changes_is_the_one_on_disk(tmp_path, monkeypatch):
    from sopa_companion.engine import analysis as engine_analysis
    run = folder_run(tmp_path, "r1")
    m = settings_manager(tmp_path, run)
    write, writing, slowed = engine_analysis._atomic, threading.Event(), []

    def slow(path, data):                                     # the first change's first write is the slow one: with
        if not slowed:                                        # two writers at once it would end last, on 1.5
            slowed.append(path)
            writing.set()
            time.sleep(0.6)
        write(path, data)
    monkeypatch.setattr(engine_analysis, "_atomic", slow)
    first = threading.Thread(target=m.configure, args=("r1",), kwargs={"params": {"resolution": 1.5}})
    first.start()
    assert writing.wait(5)
    m.configure("r1", params={"resolution": 2.5})             # while the first change is still being written
    first.join(5)
    stored = json.loads((run.dir / "analysis" / "settings.json").read_text())
    assert stored["settings"]["resolution"] == 2.5 and stored["generation"] == 2
    assert json.loads((tmp_path / "var" / "analysis-settings.json").read_text())["settings"] == {"resolution": 2.5}


def test_a_resolution_change_keeps_the_layout_warm_and_a_map_change_lays_it_out_afresh():
    run = types.SimpleNamespace(layout_key=None)
    assert not W.Worker.layout_settings_changed(run, dict(W.DEFAULTS))            # a first layout is no change
    assert not W.Worker.layout_settings_changed(run, {**W.DEFAULTS, "resolution": 2.0})   # regrouped: still warm
    for key, value in (("n_neighbors", 30), ("min_dist", 0.5), ("min_counts", 20), ("n_top_genes", 500), ("n_pcs", 20)):
        W.Worker.layout_settings_changed(run, dict(W.DEFAULTS))
        assert W.Worker.layout_settings_changed(run, {**W.DEFAULTS, key: value}), key   # another map: afresh
        assert not W.Worker.layout_settings_changed(run, {**W.DEFAULTS, key: value}), key   # then warm again
    # Never more principal components than the genes used (a panel smaller than n_top_genes: all of it) or the cells.
    assert W.principal_components(30, 1000, 2000) == 30 and W.principal_components(30, 1000, 20) == 20
    assert W.principal_components(30, 12, 2000) == 11 and W.principal_components(30.0, 2, 1) == 1
