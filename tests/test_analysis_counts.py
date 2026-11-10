"""The analysis scheduler on proseg's own count (engine/analysis.py): only counted frames are analysed (never an outline
count in their place), a run without proseg's count gets no job and says why ("old_proseg", "no_counts"); counts tokens
one at a time, asked as a live pass starts (or ahead of a spaced pass), within the extraction budget (a token held back
asked for while the pass runs), none while paused, none while the spool is full (said once, not with every swing);
frames asked for snapped to the nearest counted frame, the reply naming it; backfill from the forced files and the
spool's reserve; pruning that never removes a file a job needs (a frame counted while it runs included), keeps the
newest only while they fit the run's share of the spool, has an ended run give way and reads only the frames whose file
it could remove (as the looks for the newest counted frame, a request's frame and a pass's start block read only those
near where they look, and backfill lists the spool once), never every frame of a long run; Compare's run column
held while its comparison goes on; Watch priority over backfill; the start line on frame 0's count for cells-only
starts; the requests' count files; the where box of a followed cell; a test's own runs left to the test; and the result
folders of proseg's count (analysis/pc1/, qc/<model8>.pc1/).

The run store here (CountsRuns, FakeSpool) speaks the side of engine/runs.py and engine/counts_spool.py the scheduler
uses (counts_status, request_counts, request_where, the spool's files, usage, refusals and prune), and CountedRun holds
frame records as engine/runs.py's Run does once counts.jsonl is merged; tests/test_deeper_scheduler.py uses them
too."""
from __future__ import annotations

import bisect
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.analysis.counts_file import BASIS
from sopa_companion.engine import analysis as A
from sopa_companion.engine.analysis import (BACKFILL_MAX, COUNTS_LEAD_S, COUNTS_RETRY_S, KEEP_NEWEST, NEWEST_LOOK,
                                         NO_COUNT, POLL_S, AnalysisError, AnalysisManager, Job, backfill_plan,
                                         coarse_to_fine, counts_settings, nearest_each, qc_folder)
from sopa_companion.engine.runs import ACTIVE

X0, Y0, VOXEL, NX, NY, NZ = 100.0, 50.0, 2.0, 16, 12, 2
GEOMETRY = {"nx": NX, "ny": NY, "nz": NZ, "x0": X0, "y0": Y0, "voxel": VOXEL}
DURABLE = ("init", "upscale", "hold", "finish", "end")      # proseg-live writes these in the run folder, not the spool
CELLS_START = {"sample_kind": "zarr", "prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": None}}
MODEL = ("0123456789abcdef", "/models/m1")


class Clock:
    """The engine's monotonic clock, set by hand (the scheduler's and the run store's alike)."""

    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class CountedRun:
    """A run as engine/runs.py's Run holds one once proseg's counts.jsonl is merged into its frame records: each counted
    frame's `counts_file` (an absolute path in the run's spool, outside its folder as /dev/shm is;
    frames/NNNNNN.counts.zst in its live folder for the forced ones), `counted_file` and its counts line (`counts`);
    info.json's counts block; its where exports (`wheres`); `version` bumped by every change; `counted`, the counted
    frames in order. Its count files hold a few bytes: the scheduler never reads them. Frame 0 is counted ("init") unless
    init=False."""

    def __init__(self, folder, run_id="r1", cells=3000, state="running", mode="request", spool=True, start_every=None,
                 meta=None, clock=None, init=True):
        self.dir = Path(folder)
        self.lock = threading.RLock()
        self.version = 0
        self.meta = {"id": run_id, "state": state, **(meta or {})}
        self.spool_dir = self.dir.parent / f"spool-{run_id}" if spool else None
        self.start_every = start_every
        self.clock = clock or time.monotonic
        (self.live / "frames").mkdir(parents=True, exist_ok=True)
        (self.live / "transcripts.bin.gz").write_bytes(b"")
        self.info = {"cells": cells, "z": {"grid_origin": 0.0, "grid_spacing": 0.5},
                     "counts": {"mode": mode, "format": "PLC1", "version": 1, "e_scale": 32,
                                "start": start_every is not None, "start_every": start_every,
                                "spool": str(self.spool_dir) if spool else None, "where": 1}}
        self.frames: list[dict] = []
        self.wheres: list[dict] = []
        self.counted_at: dict[int, float] = {}      # seq -> when its counts line merged (the store's newest_at)
        if init:
            self.add_frame("init")

    @property
    def live(self) -> Path:
        return self.dir / "live"

    @property
    def counted(self) -> list[int]:
        """The counted frames' seqs in order, as Run keeps them (`counted`)."""
        with self.lock:
            return [seq for seq, frame in enumerate(self.frames) if frame.get("counts_file")]

    def add_frame(self, counted: str | None = "request", cells=None, **count) -> int:
        """Proseg's next frame, counted for `counted` (a counts line's reason), or not counted (None). Its seq."""
        with self.lock:
            seq = len(self.frames)
            self.frames.append({"seq": seq, "iter": seq, "phase": "burnin", "t": 0.1 * seq, "live": {},
                                "file": f"frames/{seq:06d}.bin.gz", "cells_file": f"frames/{seq:06d}.cells.zst",
                                "cells": self.info["cells"] if cells is None else cells, **GEOMETRY})
            self.version += 1
        if counted:
            self.count(seq, counted, **count)
        return seq

    def count(self, seq: int, reason: str = "request", *, has_start=None, size: int = 1000, extract_ms: float = 0.0,
              token=None) -> Path:
        """Frame `seq`'s counts line merged into its record (its PLC1 file written first: the commit point)."""
        name = f"{seq:06d}.counts.zst"
        durable = reason in DURABLE or self.spool_dir is None
        path = self.live / "frames" / name if durable else self.spool_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PLC1")
        line = {"seq": seq, "iter": seq, "file": f"frames/{name}" if durable else str(path),
                "counted_file": f"frames/{seq:06d}.counted.zst", "reason": reason, "token": token, "where": None,
                "nnz": 1, "sum_n": 1, "sum_e": 1.0, "sum_f": None if reason in ("request", "every") else 1,
                "has_f": reason not in ("request", "every"),
                "has_start": (self.start_every is not None) if has_start is None else bool(has_start),
                "saturated": 0, "bytes": int(size), "ms": {"extract": float(extract_ms), "write": 0.0}}
        with self.lock:
            self.frames[seq].update(counts_file=line["file"], counted_file=line["counted_file"], counts=line)
            self.counted_at[seq] = self.clock()
            self.version += 1
        return path

    def count_path(self, seq: int) -> Path:
        return self.live / self.frames[seq]["counts_file"]


class FakeSpool:
    """engine/counts_spool.py as the scheduler uses it: each run's spooled count files, prune(run, keep) (every record
    of a prune kept), usage_mb (by the counts lines' bytes of the files there: the files themselves hold a few bytes),
    and why a run's tokens stop: `why` (run -> "run_cap" | "engine_cap" | "low_space"), else with `cap_mb` set as
    CountsSpool.can_issue says it: the run's files and one more the size of its newest above the cap ("run_cap"), the
    engine's above twice the cap ("engine_cap")."""

    def __init__(self, store):
        self.store, self.why, self.pruned, self.cap_mb = store, {}, [], None

    def files(self, run_id: str) -> list[str]:
        folder = self.store.runs[run_id].spool_dir
        return sorted(str(path) for path in folder.glob("*.counts.zst")) if folder and folder.is_dir() else []

    def prune(self, run_id: str, keep: set) -> set:
        removed = {path for path in self.files(run_id) if path not in keep}
        for path in removed:
            os.unlink(path)
        self.pruned.append((run_id, set(keep), removed))
        return removed

    def _bytes(self, run_id: str) -> tuple[int, int]:
        """The run's spooled files' bytes, and its newest file's."""
        run = self.store.runs.get(run_id)
        if run is None or run.spool_dir is None:
            return 0, 0
        with run.lock:
            sizes = {os.path.normpath(frame["counts_file"]): frame["counts"]["bytes"] for frame in run.frames
                     if frame.get("counts_file") and os.path.isabs(frame["counts_file"])}
        files = [os.path.normpath(path) for path in self.files(run_id)]
        return sum(sizes.get(path, 0) for path in files), sizes.get(files[-1], 0) if files else 0

    def usage_mb(self, run_id: str | None = None) -> float:
        return sum(self._bytes(r)[0] for r in ([run_id] if run_id is not None else list(self.store.runs))) / 1e6

    def refusal(self, run_id: str) -> str | None:
        if run_id in self.why or self.cap_mb is None:
            return self.why.get(run_id)
        used, frame = self._bytes(run_id)
        if used and used + frame > self.cap_mb * 1e6:
            return "run_cap"
        engine = self.usage_mb() * 1e6
        return "engine_cap" if engine and engine + frame > 2 * self.cap_mb * 1e6 else None


class CountsRuns:
    """The run store's side the scheduler uses (engine/runs.py): get, active, counts_status (its `why` as Runs says
    it: "ended" for a run that is not active, else its counts reason, else the spool's refusal unless it is paused),
    request_counts, request_where (every call kept in `wheres`) and the spool. A token is served by the next frame
    counted for it (`serve`), or at once while the run is paused (by its paused state, with no line): `tokens` holds
    each run's [issued, served]."""

    def __init__(self, *runs):
        self.runs = {run.meta["id"]: run for run in runs}
        self.spool = FakeSpool(self)
        self.tokens: dict[str, list[int]] = {}
        self.wheres: list[tuple] = []

    def get(self, run_id: str):
        return self.runs[run_id]

    def active(self) -> list:
        return [run for run in self.runs.values() if run.meta.get("state") in ACTIVE]

    def counts_status(self, run_id: str) -> dict:
        run = self.runs[run_id]
        issued, served = self.tokens.get(run_id, [0, 0])
        block = (run.info or {}).get("counts")
        reason = "old_proseg" if run.meta.get("live_counts") == "old_proseg" else \
            "no_counts" if run.info and not block else None
        with run.lock:
            counted = [frame for frame in run.frames if frame.get("counts_file")]
        newest = counted[-1] if counted else None
        state = run.meta.get("state")
        why = "ended" if state not in ACTIVE else reason or (None if state == "paused" else self.spool.refusal(run_id))
        return {"mode": (block or {}).get("mode"), "reason": reason, "issued": issued, "served": served,
                "outstanding": issued > served, "newest": newest["seq"] if newest else None,
                "newest_at": run.counted_at.get(newest["seq"]) if newest else None,
                "extract_ms": newest["counts"]["ms"]["extract"] if newest else None,
                "bytes": newest["counts"]["bytes"] if newest else None, "spool_mb": 0.0,
                "can_issue": why is None, "why": why}

    def request_counts(self, run_id: str) -> int | None:
        run, status = self.runs[run_id], self.counts_status(run_id)
        if status["outstanding"]:
            return status["issued"]
        if not status["can_issue"]:
            return None
        tokens = self.tokens.setdefault(run_id, [0, 0])
        tokens[0] += 1
        if run.meta.get("state") == "paused":                # read while paused: served by the paused state
            tokens[1] = tokens[0]
        return tokens[0]

    def request_where(self, run_id: str, box, wait_s: float = 2.0):
        self.wheres.append((run_id, list(box), wait_s))
        if not self.counts_status(run_id)["outstanding"]:
            self.request_counts(run_id)
        return None

    def serve(self, run_id: str, reason: str = "request", **count) -> int:
        """Proseg's next frame, counted for the outstanding token: its seq."""
        tokens = self.tokens.setdefault(run_id, [0, 0])
        seq = self.runs[run_id].add_frame(reason, token=tokens[0] if tokens[0] > tokens[1] else None, **count)
        tokens[1] = tokens[0]
        return seq

    def pause(self, run_id: str):
        """The run pauses: its state's frame gets its counts line ("pause"), and a token read then is served by it."""
        run = self.runs[run_id]
        run.meta["state"] = "paused"
        seq = run.add_frame("pause")
        tokens = self.tokens.setdefault(run_id, [0, 0])
        tokens[1] = tokens[0]
        return seq


class Provider:
    """A QC provider (engine/qc.py's role): one model for every run (None while the sample learns), every record it is
    handed, and the runs a Watch runs on (`watched`)."""

    def __init__(self, model=MODEL, watched=()):
        self.model, self.records, self.watched = model, [], set(watched)

    def qc_model(self, run):
        return None if self.model is None else {"id": self.model[0], "path": self.model[1]}

    def qc_result(self, record):
        self.records.append(record)

    def watching(self, run_id):
        return run_id in self.watched


def manager(tmp_path, *runs, clock=None, qc=None, **settings) -> AnalysisManager:
    """A scheduler over a CountsRuns store, started by the test itself (no worker is started while a run prepares)."""
    config = {"analysis": {"python": sys.executable, **settings}, "paths": {"state_dir": str(tmp_path / "state")}}
    if qc is not None:
        config["qc"] = qc
    m = AnalysisManager(config, CountsRuns(*runs), start=False, prestart=False)
    if clock is not None:
        m.clock = clock
    return m


def checked(tmp_path, *runs, model=MODEL, watched=(), **settings) -> AnalysisManager:
    """A scheduler with a QC provider and every run's check on."""
    m = manager(tmp_path, *runs, **settings)
    m.attach_qc(Provider(model=model, watched=watched))
    for run in runs:
        m.configure_qc(run.meta["id"], enabled=True)
    return m


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


def ask(m) -> dict:
    """What the scheduler's thread does with the tokens a pass left (_loop: after the lock)."""
    asks, m.to_ask = m.to_ask, {}
    m._ask_counts(asks)
    return asks


def drive(m, ids, steps: int, clock, size: int) -> list[dict]:
    """The scheduler's thread a step a second, as _loop goes: the next job, the tokens it left asked for after the
    lock, the pass, then the prunes of its run and of those a full spool asked for (`_prune_runs`); proseg counts its
    next frame (`size` bytes) for each token out. The tokens issued to each of `ids` after each step."""
    store, issued = m.runs, []
    for _ in range(steps):
        clock.now += 1.0
        job, _ = m._next_job()
        ask(m)
        prunes, m.to_prune = m.to_prune, set()
        if job is not None:
            finish(m, job)
        m._prune_runs(prunes | ({job.run} if job is not None else set()))
        for run_id in ids:
            if store.counts_status(run_id)["outstanding"]:
                store.serve(run_id, size=size)
        issued.append({run_id: store.tokens.get(run_id, [0, 0])[0] for run_id in ids})
    return issued


def spooled(run) -> list[int]:
    """The frames whose count files are in the run's spool now."""
    return sorted(int(path.name[:6]) for path in run.spool_dir.glob("*.counts.zst"))


class WorkerProcess:
    """A worker process as _call talks to it: what it is sent is kept; its replies are put on the manager's queue by
    the test."""
    pid = 4242

    def __init__(self):
        self.sent, self.stdin = [], self

    def poll(self):
        return None

    def write(self, text: str):
        self.sent.append(json.loads(text))

    def flush(self):
        pass


def stub_worker(m, calls: list, fail=()):
    """m._call without a worker process: writes what the worker writes and replies as it does (ops in `fail` fail).
    The count it read is said when it is not the averaged one (a summary or answer without one is on the averaged
    count)."""
    def call(request):
        calls.append(request)
        reply, errors = {"run": request["run"], "seq": request.get("seq")}, {}
        said = {"count": request["count"]} if request.get("count", "averaged") != "averaged" else {}
        for op in ["qc_start"] if request["op"] == "qc_start" else request["ops"]:
            if op in fail:
                errors[op] = "ValueError: boom"
            elif op == "umap":
                Path(request["out_json"]).write_text(json.dumps({"seq": request["seq"], **said}))
                Path(request["out_cells"]).write_bytes(b"")
                reply["summary"] = {"seq": request["seq"], **said}
            elif op == "deeper":
                reply["deeper"] = {"v": request["deeper"]["v"], "state": "ready", "cell": request["deeper"]["cell"],
                                   **said}
            else:
                Path(request["qc"]["out_cells"]).write_bytes(b"\0")
                if request["qc"].get("out_cells_e"):
                    Path(request["qc"]["out_cells_e"]).write_bytes(b"\0")
                reply["qc"] = {"kind": "start" if op == "qc_start" else "frame", **said}
        reply["ok"] = not errors
        if errors:
            reply["errors"], reply["error"] = errors, next(iter(errors.values()))
        return reply
    m._call = call


# ---- settings and folders ----------------------------------------------------------------------------------------------
def test_counts_settings_and_the_folders_of_proseg_s_count(tmp_path):
    assert counts_settings(None) == {"counts_extract_share": 0.10, "counts_mb_s": 0.0, "counts_spool_mb": 1024.0}
    assert counts_settings({"counts_extract_share": 1, "counts_mb_s": 25, "counts_spool_mb": 64}) == \
        {"counts_extract_share": 1.0, "counts_mb_s": 25.0, "counts_spool_mb": 64.0}
    for key, bad in (("counts_extract_share", 0), ("counts_extract_share", 1.5), ("counts_extract_share", "x"),
                     ("counts_extract_share", True), ("counts_mb_s", -1), ("counts_mb_s", float("inf")),
                     ("counts_spool_mb", 63), ("counts_spool_mb", None)):
        with pytest.raises(ValueError, match=rf"^\[analysis\] {key} must be "):
            counts_settings({key: bad})
        with pytest.raises(ValueError, match=rf"^\[analysis\] {key} must be "):
            manager(tmp_path, **{key: bad})
    # Results on proseg's count live apart from the outline-era ones, which are never read.
    assert BASIS == "pc1"
    assert qc_folder(tmp_path / "run", "0123456789abcdef") == tmp_path / "run" / "qc" / "01234567.pc1"
    run = CountedRun(tmp_path / "run")
    m = manager(tmp_path, run)
    assert m.qc_dir(run, "feedc0de99") == run.dir / "qc" / "feedc0de.pc1"
    results = run.dir / "analysis" / "pc1"
    assert m._paths(run, 7) == (results / "000007.json", results / "000007.cells.bin")
    assert m._paths(run, 7, 3)[0] == run.dir / "analysis" / "pc1" / "000007-s3.json"
    # An outline-era result of the same frame (analysis/000000.json) is never served, nor its check's (qc/<model8>/).
    (run.dir / "analysis").mkdir()
    (run.dir / "analysis" / "000000.json").write_text(json.dumps({"seq": 0, "from": "outlines"}))
    (run.dir / "analysis" / "000000.cells.bin").write_bytes(b"")
    with pytest.raises(AnalysisError) as missing:
        m.summary("r1", 0)
    assert missing.value.status == 404
    old = run.dir / "qc" / "01234567"
    old.mkdir(parents=True)
    for name in ("000000.qc.json", "000000.qc.bin", "start.npz", "start.bin", "start.json"):
        (old / name).write_bytes(b"x")
    m.attach_qc(Provider())
    m.configure_qc("r1", enabled=True)
    qc = m._qc_state("r1", m._qc_models())
    assert qc.dir == run.dir / "qc" / "01234567.pc1" and qc.done == {} and qc.start == "pending"


def test_nearest_picks_and_backfill_over_the_frames_there_are():
    assert nearest_each([0, 10, 20, 30], [1, 2, 3]) == [0, 10, 20], "each value its nearest free frame"
    assert nearest_each([0, 10, 20], [5, 5, 5, 5]) == [0, 10, 20] and nearest_each([], [1]) == []
    assert backfill_plan(100, set(), available=[]) == [] and backfill_plan(0, set(), available=[0]) == []
    # Only frames whose count's file is there: the forced ones and the spool's reserve, coarse to fine.
    pool = [0, 7, 30, 31, 55, 80, 99, 120]                  # 120: the newest's own (the live pass's), never backfilled
    plan = backfill_plan(120, set(), available=pool)
    assert sorted(plan) == [0, 7, 30, 31, 55, 80, 99] and plan[0] == 0 and plan == coarse_to_fine(sorted(plan))
    assert sorted(backfill_plan(120, set(), limit=3, available=pool)) == [0, 55, 99]   # evenly spaced among them
    # Scored already near a target (80: within half the spacing of 40 from 99): that one is not backfilled.
    assert sorted(backfill_plan(120, {30, 80}, limit=3, available=pool)) == [0, 55]
    near = backfill_plan(1000, set(range(0, 1000, 2)), available=range(0, 1000, 3))
    assert near == [], "scored live often enough already"
    assert backfill_plan(10, set()) == coarse_to_fine(list(range(10)))   # without `available`: every frame, as before


# ---- what is analysed --------------------------------------------------------------------------------------------------
def test_a_live_pass_takes_the_newest_counted_frame_whose_file_is_there(tmp_path):
    run = CountedRun(tmp_path / "r1")
    run.add_frame(None)                                       # 1: not counted (proseg counts only when asked)
    run.add_frame("request")                                  # 2
    run.add_frame(None)                                       # 3
    m = manager(tmp_path, run)
    job, wait = m._next_job()
    assert job == ("r1", 2) and job.kind == "live" and job.ops == ("umap",) and wait is None
    finish(m, job)
    run.add_frame(None)                                       # 4: newer, not counted: nothing to do
    assert m._next_job() == (None, POLL_S)
    run.add_frame("request")                                  # 5, its file gone before its pass: the newest counted
    run.count_path(5).unlink()                                # with a file is 2, analysed already
    assert m._next_job() == (None, POLL_S)
    # No counted frame at all yet (frame 0's count comes by itself): looked at again shortly.
    bare = CountedRun(tmp_path / "bare", run_id="bare", init=False)
    bare.add_frame(None)
    m = manager(tmp_path, bare)
    assert m._next_job() == (None, POLL_S) and not m.to_ask
    # _process names the frame's count file; a frame without one fails, never counted from its outlines.
    calls = []
    stub_worker(m, calls)
    bare.add_frame("request")
    assert m._process(Job("bare", 1, ("umap",), "request")) is None
    assert calls[-1]["counts_file"] == str(bare.count_path(1)) and os.path.isabs(calls[-1]["counts_file"])
    assert Path(calls[-1]["out_json"]) == bare.dir / "analysis" / BASIS / "000001.json"
    count = len(calls)
    assert m._process(Job("bare", 0, ("umap",), "request")) == NO_COUNT and len(calls) == count
    assert m.status["bare"] == {"state": "error", "seq": 0, "error": NO_COUNT, "reason": "no_counts"}


def test_a_pause_count_of_a_frame_read_before_is_analysed(tmp_path):
    """A pause that finds no frame pending counts the last frame proseg wrote, whose record the engine read already:
    its line merges into that record (the run's version changes, not its frames' number), and that frame is the
    newest counted one now. The newest counted frame is looked at again once its file goes."""
    run = CountedRun(tmp_path / "r1")                         # 0: init
    run.add_frame("request")                                  # 1
    run.add_frame(None)                                       # 2: not counted
    m = manager(tmp_path, run)
    job, _ = m._next_job()
    assert job == ("r1", 1)
    finish(m, job)
    assert m._next_job() == (None, POLL_S)
    run.meta["state"] = "paused"
    run.count(2, "pause")
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.kind == "live"
    assert m._newest_counted(run) == 2
    run.count_path(2).unlink()                                # gone (the spool pruned, say): no change of its frames
    assert m._newest_counted(run) == 1


def test_a_run_without_proseg_s_count_gets_no_job_and_says_why(tmp_path):
    made_before = CountedRun(tmp_path / "old", run_id="old", init=False)
    made_before.info.pop("counts")                            # info.json without a counts block: made before
    made_before.add_frame(None)
    made_before.add_frame(None)
    older = CountedRun(tmp_path / "older", run_id="older", meta={"live_counts": "old_proseg"}, init=False)
    older.info.pop("counts")
    older.add_frame(None)
    older.count(0, "init")                                    # even a counted record (none there) is not analysed
    m = checked(tmp_path, made_before, older)
    for run_id, reason in (("old", "no_counts"), ("older", "old_proseg")):
        described = m.describe(run_id)
        assert described["unavailable"] == reason and described["status"] == {"state": "unavailable", "reason": reason}
        assert m.describe_qc(run_id)["unavailable"] == reason, "active runs say it too"
        m.configure(run_id, seq=0)                            # frames asked for: nothing is recorded
        m.configure_qc(run_id, seq=0)
        assert run_id not in m.requested and run_id not in m.qc_requested
    m.request_deeper("old", 1, 4)
    m.request_deeper("older", 0, 2)
    version = m.version
    assert m._next_job() == (None, None) and not m.to_ask, "no job at all, no token"
    assert sorted((d["run"], d["reason"]) for d in m.deeper_dropped) == [("old", "no_counts"), ("older", "old_proseg")]
    assert m.status["old"] == {"state": "unavailable", "reason": "no_counts"} and m.version > version   # pushed
    # old_proseg first, then no_counts, then no_transcripts, then too_few_cells.
    plain = CountedRun(tmp_path / "plain", run_id="plain")
    m = manager(tmp_path, plain)
    m.too_few["plain"] = 20
    assert m.describe("plain")["unavailable"] == "too_few_cells"
    (plain.live / "transcripts.bin.gz").unlink()
    m.exported.clear()                                        # looked at once per run: here, again
    assert m.describe("plain")["unavailable"] == "no_transcripts"
    plain.info.pop("counts")
    assert m.describe("plain")["unavailable"] == "no_counts"
    plain.meta["live_counts"] = "old_proseg"
    assert m.describe("plain")["unavailable"] == "old_proseg"


def test_frames_asked_for_before_a_run_s_reason_was_known_are_dropped(tmp_path):
    """A request recorded while the run's proseg count stood (proseg started again since, with a proseg-live that
    cannot keep it, say) is dropped when its turn comes, and the run's status says why: nothing is analysed."""
    run = CountedRun(tmp_path / "r1", state="done")
    run.add_frame("request")
    m = checked(tmp_path, run)
    assert m.configure("r1", seq=1)["requested"] == 1 and m.configure_qc("r1", seq=1)["requested"] == 1
    run.meta["live_counts"] = "old_proseg"
    assert m._next_job() == (None, None)
    assert "r1" not in m.requested and "r1" not in m.qc_requested
    assert m.status["r1"] == {"state": "unavailable", "reason": "old_proseg"}


# ---- counts tokens -----------------------------------------------------------------------------------------------------
def test_one_token_at_a_time_asked_as_a_live_pass_starts(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock)
    store = m.runs
    job, _ = m._next_job()
    assert job == ("r1", 0) and m.to_ask == {"r1": None}, "asked as the pass starts: its frame comes meanwhile"
    ask(m)
    assert store.tokens["r1"] == [1, 0] and m.asked["r1"] == 100.0
    finish(m, job)
    clock.now += 1.0
    assert m._next_job() == (None, POLL_S) and not m.to_ask, "its token outstanding: none more"
    run.add_frame(None)                                       # proseg's frame that came before it read the token
    store.serve("r1")                                         # 2: counted for token 1
    clock.now += 1.0
    job, _ = m._next_job()
    assert job == ("r1", 2) and m.to_ask == {"r1": None}
    ask(m)
    assert store.tokens["r1"] == [2, 1]
    # A pass of the same frame asked again (explicit) asks for nothing; neither does a run caught up with one coming.
    m.configure("r1", seq=2)
    clock.now += 1.0
    job, _ = m._next_job()
    assert job.kind == "request" and not m.to_ask
    # A token the store refuses is not asked for again before COUNTS_RETRY_S.
    finish(m, job)
    store.serve("r1")                                         # 3, for token 2
    store.request_counts = lambda run_id: None
    clock.now += 1.0
    job, _ = m._next_job()
    asks = ask(m)
    assert job == ("r1", 3) and asks == {"r1": None} and m.ask_after["r1"] == clock.now + COUNTS_RETRY_S
    finish(m, job)
    clock.now += COUNTS_RETRY_S - 0.01
    m._next_job()
    assert not m.to_ask
    clock.now += 0.02
    m._next_job()
    assert m.to_ask == {"r1": None}
    # "every": proseg counts each frame by itself, nothing is asked for; a store that tells nothing, likewise.
    every = CountedRun(tmp_path / "every", run_id="every", mode="every", clock=clock)
    m = manager(tmp_path, every, clock=clock)
    job, _ = m._next_job()
    assert job == ("every", 0) and not m.to_ask
    m.runs.counts_status = lambda run_id: (_ for _ in ()).throw(KeyError(run_id))
    every.add_frame("every")
    job, _ = m._next_job()
    assert job == ("every", 1) and not m.to_ask and m.describe("every")["unavailable"] is None


def test_a_token_held_back_by_the_budget_is_asked_for_while_the_pass_runs(tmp_path):
    """The pass starts before the extraction budget allows the next token: the token is asked for at its time while
    the pass waits for the worker (`_call`), so the next counted frame is on its way when the pass ends."""
    run = CountedRun(tmp_path / "r1", init=False)             # the engine's own clock
    run.add_frame("init", extract_ms=30.0)                    # the budget: 0.3 s after its line merged
    m = manager(tmp_path, run)
    store = m.runs
    job, _ = m._next_job()
    assert job == ("r1", 0) and not m.to_ask
    assert m.ask_plan["r1"][0] == pytest.approx(run.counted_at[0] + 0.3)
    asked_at, request_counts = [], store.request_counts
    store.request_counts = lambda run_id: (asked_at.append(time.monotonic()), request_counts(run_id))[1]
    m.process = WorkerProcess()

    def worker():                                             # the pass takes 1.5 s
        time.sleep(1.5)
        m.replies.put({"ok": True, "run": "r1", "seq": 0, "summary": {"seq": 0}})
    threading.Thread(target=worker, daemon=True).start()
    reply = m._call({"op": "analyze", "ops": ["umap"], "run": "r1", "seq": 0})
    answered = time.monotonic()
    assert reply["seq"] == 0 and m.process.sent[0]["seq"] == 0
    assert store.tokens["r1"] == [1, 0] and len(asked_at) == 1, "asked once"
    assert run.counted_at[0] + 0.3 <= asked_at[0] < answered - 0.5, "at its time, during the pass"
    assert "r1" not in m.ask_plan and m.asked["r1"] == pytest.approx(asked_at[0], abs=0.05)
    # Its time comes while the token is out already (or the run paused): nothing more is asked.
    m.ask_plan["r1"] = (time.monotonic(), run, True, None)
    m._ask_due()
    assert len(asked_at) == 1 and "r1" not in m.ask_plan
    # A reply that does not come: the wait ends at its time, whatever is held back.
    m.ask_plan["r1"] = (time.monotonic() + 60.0, run, True, None)
    started = time.monotonic()
    with pytest.raises(queue.Empty):
        m._await_reply(0.2)
    assert 0.15 <= time.monotonic() - started < 2.0


def test_the_extraction_budget_spaces_the_tokens(tmp_path):
    """No token sooner than ms.extract / counts_extract_share after the newest counted frame (proseg's counting stays
    within that share of its time); with counts_mb_s, nor than its bytes take at that pace."""
    clock = Clock(100.0)
    run = CountedRun(tmp_path / "r1", clock=clock, init=False)
    run.add_frame("init", extract_ms=230.0, size=55_000_000)  # a whole sample: 230 ms to count, 55 MB
    m = manager(tmp_path, run, clock=clock)
    job, _ = m._next_job()
    assert job == ("r1", 0) and not m.to_ask, "at 100.0: due at 102.3"
    finish(m, job)
    clock.now = 102.29
    m._next_job()
    assert not m.to_ask and m.ask_plan["r1"][0] == pytest.approx(102.3), "held back: asked at its time (_ask_due)"
    clock.now = 102.3
    m._next_job()
    assert m.to_ask == {"r1": None} and "r1" not in m.ask_plan
    m.to_ask.clear()
    clock.now = 102.31
    m.ask_after["r1"] = 103.0                                 # held back again, then no pass of it needs one
    m._next_job()
    assert m.ask_plan["r1"][0] == 103.0
    m.configure("r1", enabled=False)
    m._next_job()
    assert "r1" not in m.ask_plan and not m.to_ask
    m = manager(tmp_path, run, clock=clock, counts_mb_s=10, counts_extract_share=0.5)
    m._store("r1", 0, {"seq": 0})
    clock.now = 105.49                                        # 55 MB at 10 MB/s: 5.5 s (0.46 s by the share)
    m._next_job()
    assert not m.to_ask
    clock.now = 105.5
    m._next_job()
    assert m.to_ask == {"r1": None}


def test_a_paused_run_asks_for_no_token_and_a_resumed_one_asks_again(tmp_path):
    """A token read while paused is served by the paused state, with no new frame: nothing is asked while paused or
    held (its state is counted by itself), and the run asks afresh once it samples again."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock)
    store = m.runs
    job, _ = m._next_job()
    ask(m)
    finish(m, job)
    seq = store.pause("r1")                                   # paused before token 1 counted a frame: served by it
    assert store.tokens["r1"] == [1, 1] and run.frames[seq]["counts"]["reason"] == "pause"
    clock.now += 1.0
    job, _ = m._next_job()
    assert job == ("r1", seq) and not m.to_ask, "the paused state analysed, no token"
    finish(m, job)
    for _ in range(3):
        clock.now += 1.0
        assert m._next_job() == (None, POLL_S) and not m.to_ask
    assert "r1" not in m.asked, "its wait is not learnt from a token the paused state served"
    run.meta["state"] = "running"                             # resumed: a stale token never forces a frame
    clock.now += 1.0
    m._next_job()
    assert m.to_ask == {"r1": None}
    ask(m)
    assert store.tokens["r1"] == [2, 1]
    for state in ("finishing", "preparing", "done"):
        run.meta["state"] = state
        clock.now += 1.0
        store.serve("r1")
        job, _ = m._next_job()
        assert not m.to_ask, state
        if job is not None:
            finish(m, job)


def test_spaced_passes_ask_for_their_frame_ahead_of_time_and_learn_how_long_it_takes(tmp_path):
    """Passes spaced by min_interval_s ask for their counted frame the expected wait before they are due (not at the
    previous pass's start: that frame would be the interval old), learning that wait from the run's own tokens."""
    clock = Clock(100.0)
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock, min_interval_s=3.0)
    store = m.runs
    job, _ = m._next_job()
    assert job == ("r1", 0) and not m.to_ask, "the next pass is due at 103"
    finish(m, job)
    clock.now = 102.49
    m._next_job()
    assert not m.to_ask
    clock.now = 103.0 - COUNTS_LEAD_S[0]
    m._next_job()
    assert m.to_ask == {"r1": None}
    ask(m)
    clock.now = 103.3                                         # its frame came 0.8 s after the token
    store.serve("r1")
    job, _ = m._next_job()
    assert job == ("r1", 1) and m.leads["r1"] == pytest.approx(0.8)
    clock.now = 106.3 - 0.81                                  # due at 106.3: asked 0.8 s ahead
    finish(m, job)
    m._next_job()
    assert not m.to_ask
    clock.now = 106.3 - 0.79
    m._next_job()
    assert m.to_ask == {"r1": None}
    # Back to back (the default on a GPU): asked as each pass starts.
    m = manager(tmp_path, run, clock=clock)
    job, _ = m._next_job()
    assert job == ("r1", 1) and m.to_ask == {"r1": None}


def test_a_full_spool_stops_the_tokens_and_has_the_run_pruned(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock)
    store = m.runs
    m._store("r1", 0, {"seq": 0})
    store.spool.why["r1"] = "run_cap"
    m._next_job()
    assert not m.to_ask and m.to_prune == {"r1"}
    m.to_prune.clear()
    clock.now += A.PRUNE_RETRY_S / 2
    m._next_job()
    assert not m.to_prune, "pruned at most every PRUNE_RETRY_S"
    clock.now += A.PRUNE_RETRY_S
    m._next_job()
    assert m.to_prune == {"r1"}
    said = [text for _, run_id, text in m.events if run_id == "r1"]
    assert said == ["Proseg's count is not asked for while this run's spool is full"], said
    store.spool.why["r1"] = "low_space"                       # the RAM disk: no prune would help
    m.to_prune.clear()
    clock.now += 5.0
    m._next_job()
    assert not m.to_prune and not m.to_ask
    # Said by what holds, wherever the spool is (the RAM disk, or the temporary folder it was passed over for), and a
    # run whose spool folder could not be made by that, whatever the disk's room.
    store.spool.why["r1"] = "spool_lost"; m.noted.pop("counts:r1", None); m.to_prune.clear(); m._next_job()
    said = [text for _, run_id, text in m.events if run_id == "r1"]
    assert said[-2:] == ["Proseg's count is not asked for while the disk the spool is on has less than twice a run's spool "
                         "cap free", "Proseg's count is not asked for while this run's spool folder could not be made"], said
    assert not any("RAM disk" in text or "nearly full" in text for text in said)
    assert not m.to_prune, "no prune helps either"
    del store.spool.why["r1"]
    clock.now += 1.0
    m._next_job()
    assert m.to_ask == {"r1": None}


def test_past_iteration_2000_a_token_goes_once_the_run_is_a_thinning_step_past_its_newest_count(tmp_path):
    """A counts token makes proseg-live write the frame it serves whatever its history thinning keeps (live.rs: every
    frame up to iteration 2,000, then every 5th, every 25th past 10,000, every 100th past 50,000), and those frames'
    files are never pruned: past 2,000 a token goes only once the run's newest frame is a thinning step past its newest
    counted frame, so an open-ended run writes at most twice the frames thinning keeps, not one per iteration."""
    assert [A.thinning_step(i) for i in (0, 2000, 2001, 10_000, 10_001, 50_000, 50_001, 10 ** 6)] == \
        [1, 1, 5, 5, 25, 25, 100, 100]
    cases = ((1990, 1999, True), (1999, 2001, False), (1996, 2001, True), (2000, 2004, False), (2000, 2005, True),
             (2003, 2007, False), (9990, 10_010, False), (9990, 10_015, True), (49_950, 50_049, False),
             (49_950, 50_050, True))
    for n, (counted, newest, asked) in enumerate(cases):
        clock = Clock()
        run = CountedRun(tmp_path / f"r{n}", clock=clock)       # frame 0 counted (init) at iteration `counted` ...
        run.add_frame(None)                                     # ... frame 1, the newest, at `newest`: not counted
        run.frames[0]["iter"], run.frames[1]["iter"] = counted, newest
        m = manager(tmp_path / f"m{n}", run, clock=clock)
        m._store("r1", 0, {"seq": 0})
        m._next_job()
        assert (m.to_ask == {"r1": None}) is asked, (counted, newest, m.to_ask)


# ---- frames asked for --------------------------------------------------------------------------------------------------
def test_frames_asked_for_snap_to_the_nearest_counted_frame(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)            # 0: init (run folder)
    for counted in (None, "request", None, "request", None, "request"):   # 1 .. 6: counted 2, 4, 6
        run.add_frame(counted)
    run.count_path(4).unlink()                                # pruned since
    m = checked(tmp_path, run, clock=clock)
    described = m.configure("r1", seq=5)
    assert described["requested"] == 2 and m.requested == {"r1": 2}, "the newest at or before it whose file is there"
    assert m.describe("r1")["requested"] == 2, "waiting for it"
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.kind == "request" and job.ops == ("umap", "qc")
    finish(m, job)
    assert 2 in m.describe("r1")["seqs"] and m.describe("r1")["requested"] is None   # stored under the counted frame
    m._store("r1", 4, {"seq": 4})                             # frame 4's results are kept: they need no file
    assert m.configure("r1", seq=5)["requested"] == 4
    job, _ = m._next_job()
    assert job == ("r1", 4) and job.ops == ("umap",), "its count's file is gone: the check cannot ride along"
    assert m.configure("r1", seq=1)["requested"] == 0
    m.requested.clear()
    assert m.configure_qc("r1", seq=3)["requested"] == 2 and m.describe_qc("r1")["requested"] == 2
    qc = m.qc_runs["r1"]
    qc.done[4] = "averaged"
    assert m.configure_qc("r1", seq=5)["requested"] == 4
    assert m.request_deeper("r1", 5, 3) == 2 and m.deeper_requested["r1"] == (2, 3)
    assert m.request_deeper("r1", "start", 3) == "start"
    # None counted at or before it: the oldest after it.
    late = CountedRun(tmp_path / "late", run_id="late", init=False)
    late.add_frame(None)
    late.add_frame(None)
    late.add_frame("upscale")
    m2 = manager(tmp_path, late)
    assert m2.configure("late", seq=0)["requested"] == 2
    assert m2.compare_frame("late", 1)["seq"] == 2
    # Compare: the run's own column names the counted frame's file, kept in the spool meanwhile.
    frame = m.compare_frame("r1", 5)
    assert frame["seq"] == 2 and frame["iter"] == 2 and frame["counts_file"] == str(run.count_path(2))
    assert m.compare_frame("r1")["seq"] == 6 and set(m.compare_files["r1"]) == {6}, "the newer comparison's alone"
    bare = CountedRun(tmp_path / "bare", run_id="bare", init=False)
    bare.add_frame(None)
    m.runs.runs["bare"] = bare
    with pytest.raises(AnalysisError) as refused:
        m.compare_frame("bare")
    assert refused.value.status == 409 and str(refused.value) == "This run has no frame with proseg's count"
    for bad, status in (("x", 422), (99, 404)):
        with pytest.raises(AnalysisError) as refused:
            m.compare_frame("r1", bad)
        assert refused.value.status == status
    # A frame asked for whose file went while it waited is snapped again when its turn comes.
    m.configure("r1", seq=6)
    run.count_path(6).unlink()
    job, _ = m._next_job()
    assert job == ("r1", 4) and job.kind == "request", "6 gone, 4's results kept"


def test_the_reply_names_the_frame_asked_for_even_once_the_scheduler_took_it(tmp_path):
    """configure and configure_qc wake the scheduler's thread, which may take the request before the reply is made:
    the reply still names the counted frame the request snapped to (the page waits for that frame's results)."""
    run = CountedRun(tmp_path / "r1", state="done")           # 0: init
    for counted in (None, "request", None):                  # 1, 2 (counted), 3
        run.add_frame(counted)
    m = checked(tmp_path, run)
    store = m.runs
    status = store.counts_status

    def taken(run_id):                                        # the scheduler's thread took the request meanwhile
        m.requested.pop(run_id, None)
        m.qc_requested.pop(run_id, None)
        return status(run_id)
    store.counts_status = taken
    assert m.configure("r1", seq=3)["requested"] == 2 and m.describe("r1")["requested"] is None
    assert m.configure_qc("r1", seq=1)["requested"] == 0 and m.describe_qc("r1")["requested"] is None
    assert m.configure("r1", enabled=True)["requested"] is None, "nothing asked: nothing named"


def test_a_check_request_snaps_again_when_its_model_comes(tmp_path):
    """A frame asked of the check while its sample learns waits for the model; when its turn comes it snaps again,
    since the spool may have let its file go meanwhile."""
    run = CountedRun(tmp_path / "r1", state="done")           # 0: init (the run folder)
    for counted in ("request", None, "request", None):       # 1, 3 counted
        run.add_frame(counted)
    m = manager(tmp_path, run)
    provider = Provider(model=None)                           # the sample is being learned
    m.attach_qc(provider)
    assert m.configure_qc("r1", seq=4)["requested"] == 3
    assert m._next_job() == (None, A.QC_MODEL_POLL_S) and m.qc_requested == {"r1": 3}
    run.count_path(3).unlink()                                # its file went while the model was learned
    provider.model = MODEL
    m.qc_changed()
    job, _ = m._next_job()
    assert job == ("r1", 1) and job.ops == ("qc",) and job.kind == "request"


def test_compare_s_run_column_keeps_its_file_while_its_comparison_goes_on(tmp_path):
    """The frame compare_frame gave stays in the spool COMPARE_FILE_HOLD_S, and as long again after each op that read
    it (the first time on a whole sample, the other columns can be counted for minutes); a newer comparison of the
    run takes over the hold."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for _ in range(30):
        run.add_frame("request", size=100_000_000)
    m = manager(tmp_path, run, clock=clock)

    def play(seconds: float):                                 # the run goes on: a pass and a prune now and then
        clock.now += seconds
        run.add_frame("request", size=100_000_000)
        m.live_seq["r1"] = len(run.frames) - 1
        m._prune("r1")
    frame = m.compare_frame("r1", 12)
    path = Path(frame["counts_file"])
    play(A.COMPARE_HOLD_S + 1)
    assert path.exists()
    m._hold_compared({"run": "r1", "columns": [{"frame": frame}]})   # its first op, 61 s in: held 600 s from now
    play(A.COMPARE_FILE_HOLD_S - 1)
    assert path.exists(), "its second op still to come"
    play(2.0)
    assert not path.exists(), "the hold ran out: not one of the newest nor of the reserve (100 MB frames)"
    # One comparison per run: a newer one's frame is held, the one before is not, even after an op of the old one.
    first, second = m.compare_frame("r1", 30), m.compare_frame("r1", 33)
    assert set(m.compare_files["r1"]) == {33} and (first["seq"], second["seq"]) == (30, 33)
    m._hold_compared({"run": "r1", "columns": [{"frame": first}]})
    assert set(m.compare_files["r1"]) == {33}
    m._hold_compared({"run": "r1", "columns": [{"frame": {"run": "r1", "seq": [33]}}, "x"]})   # nothing it can read
    assert set(m.compare_files["r1"]) == {33}


def test_go_deeper_on_a_frame_goes_as_its_counted_frame(tmp_path):
    run = CountedRun(tmp_path / "r1")
    run.add_frame(None)
    run.add_frame("request")
    m = checked(tmp_path, run)
    assert m.request_deeper("r1", 1, 7) == 0
    job, _ = m._next_job()
    assert job == ("r1", 0) and job.ops == ("qc", "deeper") and job.deeper["cell"] == 7
    m.request_deeper("r1", 2, 7)
    run.count_path(2).unlink()                                # gone while it waited: snapped again
    job, _ = m._next_job()
    assert job == ("r1", 0) and job.kind == "request"
    # No counted frame with a file left: dropped, its reason handed over.
    run.count_path(0).unlink()
    m.deeper_requested["r1"] = (1, 7)
    job, _ = m._next_job()
    assert m.deeper_dropped == [{"run": "r1", "seq": 1, "cell": 7, "model": MODEL[0], "error": NO_COUNT,
                                 "reason": "no_counts"}]


# ---- backfill and the spool --------------------------------------------------------------------------------------------
def test_backfill_comes_from_the_forced_files_and_the_spool_s_reserve(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for seq in range(1, 101):
        run.add_frame("upscale" if seq == 30 else "request" if seq % 2 == 0 else None)
    reserve = {10, 50, 70}
    for seq in range(2, 100, 2):                              # the spool's reserve: the rest were pruned (frame 30,
        if seq not in reserve and seq != 30:                  # an upscale, is in the run folder)
            run.count_path(seq).unlink()
    m = manager(tmp_path, run, clock=clock)
    m.configure("r1", enabled=False)                          # the check alone (a tool's switch)
    m.attach_qc(Provider())
    m.configure_qc("r1", enabled=True)
    jobs = []
    while (job := m._next_job()[0]) is not None:
        jobs.append((job.kind, job.seq))
        finish(m, job)
    assert jobs[0] == ("live", 100)
    assert sorted(seq for kind, seq in jobs if kind == "backfill") == [0, 10, 30, 50, 70]
    # A backfill frame whose file went before its turn is skipped (the reserve moved on).
    other = CountedRun(tmp_path / "r2", run_id="r2", clock=clock)
    for seq in range(1, 9):
        other.add_frame("request")
    m = manager(tmp_path, other, clock=clock)
    m.configure("r2", enabled=False)
    m.attach_qc(Provider())
    m.configure_qc("r2", enabled=True)
    job, _ = m._next_job()
    assert job == ("r2", 8)
    planned = list(m.qc_runs["r2"].backfill)
    assert sorted(planned) == list(range(8))
    finish(m, job)
    for seq in planned[1:4]:
        other.count_path(seq).unlink()
    done = []
    while (job := m._next_job()[0]) is not None:
        done.append(job.seq)
        finish(m, job)
    assert sorted(done) == sorted(set(planned) - set(planned[1:4]))


def test_pruning_never_removes_a_file_a_job_needs(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock, start_every=4)   # cells-only start: a start block every 4th frame
    for seq in range(1, 41):
        run.add_frame("request", has_start=seq % 4 == 0 and seq <= 28, size=1000)
    m = manager(tmp_path, run, clock=clock)
    m.attach_qc(Provider())
    m.configure_qc("r1", enabled=True)
    m.live_seq["r1"] = 30                                     # the last live pass took 30: 31 .. 40 still to analyse
    m.requested["r1"] = 3                                     # a frame asked for
    m.qc_requested["r1"] = 4
    m.deeper_requested["r1"] = (5, 1)
    qc = m._qc_state("r1", m._qc_models())
    qc.backfill = [6]                                         # waiting for backfill
    run.wheres = [{"token": 1, "seq": 7, "file": "where-1.bin.zst"}]   # a where export of frame 7
    m.compare_frame("r1", 11)                                 # taken by Compare
    request = {"run": "r1", "columns": [{"frame": {"run": "r1", "seq": 12, "counts_file": str(run.count_path(12))}}]}
    m.compare_queue.append((request, None))                   # a Compare op queued
    m.spool_cap_mb = 0.012                                    # 12 kB: what is held leaves no room for a reserve
    store = m.runs
    m._prune("r1")
    (run_id, keep, removed), = store.spool.pruned
    held = {3, 4, 5, 6, 7, 11, 12, 28}                        # asked for, backfill, a where export, Compare, start block
    assert {int(path.name[:6]) for path in run.spool_dir.iterdir()} == held | set(range(31, 41))   # and 8 newest, due
    assert (run.live / "frames" / "000000.counts.zst").exists(), "the forced files stay in the run folder"
    assert len(removed) == 40 - 18 and not removed & keep
    # Room for a reserve: evenly spaced inside the run, BACKFILL_MAX at most.
    fresh = CountedRun(tmp_path / "r2", run_id="r2", clock=clock)
    for seq in range(1, 141):
        fresh.add_frame("request")
    m = manager(tmp_path, fresh, clock=clock)
    m.live_seq["r2"] = 140
    m._prune("r2")
    kept = sorted(int(path.name[:6]) for path in fresh.spool_dir.iterdir())
    reserve = [s for s in kept if s < 133]
    assert kept[-KEEP_NEWEST:] == list(range(133, 141)) and len(reserve) == BACKFILL_MAX
    assert reserve[0] == 1 and reserve[-1] == 132 and np.diff(reserve).max() <= 2 * 131 / (BACKFILL_MAX - 1), \
        "evenly spaced over the others"
    run, store = fresh, m.runs
    # A test's own run, a run without a spool and a deleted run are never pruned here.
    for seq in range(141, 201):
        run.add_frame("request")
    m.live_seq["r2"] = 200
    store.spool.pruned.clear()
    run.meta["qc_test"] = {"of": "parent", "role": "A1", "side": "A", "test": "t1"}
    m._prune("r2")
    run.meta.pop("qc_test")
    run.info["counts"]["spool"] = None
    m._prune("r2")
    m._prune("gone")
    assert store.spool.pruned == []
    run.info["counts"]["spool"] = str(run.spool_dir)
    m._prune("r2")
    assert len(store.spool.pruned) == 1, "with its spool again: pruned"


def test_a_frame_counted_while_the_spool_is_pruned_stays_until_it_is_analysed(tmp_path):
    """The run store notes a spooled file as soon as its counts line merges, and its prune removes what is not kept: a
    frame counted after `_keep` read the frames (while the prune runs), or after the spool was listed and before `_keep`
    read the run (it looks only at the frames a prune could remove: one counted since the listing is one), must not go
    before it is analysed."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for _ in range(20):
        run.add_frame("request")
    m = manager(tmp_path, run, clock=clock)
    m.live_seq["r1"] = 20
    m.spool_cap_mb = 0.002                                    # room for nothing a job does not need
    spool, landed = m.runs.spool, []
    prune = spool.prune

    def merging(run_id, keep):                                # proseg's next counted frame merges meanwhile
        landed.append(run.add_frame("request"))
        return prune(run_id, keep)
    spool.prune = merging
    m._prune("r1")
    assert landed == [21] and run.count_path(21).exists()
    assert spooled(run) == [21], "the others analysed: gone"
    spool.prune, keep = prune, m._keep
    m.live_seq["r1"] = 21                                     # analysed: its file goes at the next prune

    def listed(*args, **kwargs):                              # merged after the listing, before _keep reads the run:
        landed.extend(run.add_frame("request") for _ in range(KEEP_NEWEST + 2))      # more than its newest, even
        return keep(*args, **kwargs)
    m._keep = listed
    m._prune("r1")
    assert landed == list(range(21, 32)) and spooled(run) == list(range(22, 32))
    del m._keep
    # Long after the newest counted frame (tokens stopped meanwhile), a file proseg wrote before the prune looked, whose
    # line merges while it runs.
    late = CountedRun(tmp_path / "late", run_id="late", clock=clock)
    for _ in range(100):
        late.add_frame(None)
    late.spool_dir.mkdir(parents=True, exist_ok=True)
    (late.spool_dir / "000050.counts.zst").write_bytes(b"PLC1")
    m.runs.runs["late"] = late
    m.live_seq["late"] = 0
    spool.prune = lambda run_id, keep: (late.count(50), prune(run_id, keep))[1]
    m._prune("late")
    assert spooled(late) == [50]


def test_a_failed_prune_never_stops_the_scheduler(tmp_path):
    run = CountedRun(tmp_path / "r1")
    m = manager(tmp_path, run)

    def failing(run_id, run=None):
        raise RuntimeError("the spool's folder went")
    m._prune = failing
    m._prune_runs({"r1", "gone"})
    assert [text for _, run_id, text in m.events if run_id == "r1"] == \
        ["Proseg's spooled counts could not be pruned: RuntimeError: the spool's folder went"]


def test_a_run_started_again_keeps_its_new_frames_until_they_are_analysed(tmp_path):
    """Proseg started again (engine/runs.py: its frames from 0, its spool emptied): the frame the last pass took
    belongs to the run before, so none of the new frames counts as analysed, also once they pass its number."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for _ in range(30):
        run.add_frame("request")
    m = manager(tmp_path, run, clock=clock, min_interval_s=600)
    m.live_seq["r1"], m.started["r1"] = 30, clock.now        # its last pass took frame 30; the next one is far off
    m.spool_cap_mb = 0.002                                    # room for nothing a job does not need
    for path in run.spool_dir.iterdir():
        path.unlink()
    with run.lock:
        run.frames = []
    run.add_frame("init")
    for _ in range(3):
        run.add_frame("request")
    m._prune("r1")                                            # before the scheduler looked at it again
    assert spooled(run) == [1, 2, 3]
    assert m._next_job()[0] is None and "r1" not in m.live_seq
    for _ in range(36):
        run.add_frame("request")                              # 4 .. 39: past the frame the last pass took
    m._prune("r1")
    assert spooled(run) == list(range(1, 40)), "all still to analyse"


def test_the_reserve_stays_within_the_spool_s_cap(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for seq in range(1, 61):
        run.add_frame("request", size=55_000_000)             # a whole sample: 55 MB a counted frame
    m = manager(tmp_path, run, clock=clock)
    m.live_seq["r1"] = 60
    there = {os.path.normpath(path) for path in m.runs.spool.files("r1")}
    keep = m._keep("r1", run, there)
    kept = sorted(int(Path(path).name[:6]) for path in keep if path in there)
    # 1,024 MB: the 8 newest (440 MB) and room for 2 more (110 MB) leave 474 MB: 8 frames of the reserve.
    assert len(kept) == KEEP_NEWEST + 8 and kept[-KEEP_NEWEST:] == list(range(53, 61))
    assert sum(55 for _ in kept) + 110 <= 1024
    # The other runs hold 1,500 MB of the engine's 2,048: 548 MB is its share, room for 7 newest beside two more.
    keep = m._keep("r1", run, there, others=1500.0)
    assert sorted(int(Path(path).name[:6]) for path in keep if path in there) == list(range(54, 61))
    assert m._keep("r1", run, there, others=2100.0) & there == set(), "no share left: only what a job needs"


def keep_walking_every_frame(m, run_id, run, there, others=0.0, lean=False) -> set[str]:
    """`_keep` as it was until it looked only at the frames a prune could remove: every counted frame of the run walked
    on each prune (20 ms a pass at 5,000 counted frames). The keep set it gave is the one held to here."""
    with run.lock:
        counted = [(s, frame["counts_file"], frame.get("counts") or {}) for s, frame in enumerate(run.frames)
                   if frame.get("counts_file")]
        last = len(run.frames) - 1
        held = {w.get("seq") for w in (getattr(run, "wheres", None) or []) if isinstance(w, dict)}
    ended = run.meta.get("state") in A.TERMINAL
    folder = ((run.info or {}).get("counts") or {}).get("spool")
    for asked in (m.requested.get(run_id), m.qc_requested.get(run_id), (m.deeper_requested.get(run_id) or (None,))[0]):
        if isinstance(asked, int) and not isinstance(asked, bool):
            held.add(asked)
    qc = m.qc_runs.get(run_id)
    if qc is not None and m.qc_enabled.get(run_id, False):
        held.update(qc.backfill)
    held.update(seq for seq, until in m.compare_files.get(run_id, {}).items() if until > m.clock())
    queued = {os.path.normpath(str(column["frame"]["counts_file"]))
              for request, _ in m.compare_queue for column in request.get("columns") or []
              if isinstance(column, dict) and isinstance(column.get("frame"), dict)
              and column["frame"].get("run") == run_id and column["frame"].get("counts_file")}
    on = m.enabled.get(run_id, not ended) or m.qc_enabled.get(run_id, False)
    live = m.live_seq.get(run_id, -1)
    due_after = (live if live <= last else -1) if on else math.inf
    path = {s: os.path.normpath(str(run.live / name)) for s, name, _ in counted}
    keep = set(queued)
    keep.update(path[s] for s, _, _ in counted if s in held or s > due_after)
    start = next((s for s, _, line in reversed(counted) if line.get("has_start")), None)
    if start is not None:
        keep.add(path[start])
    top = counted[-1][0] if counted else -1
    keep.update(name for name in there if A._spooled_seq(name) > top)
    if folder:
        keep.update(os.path.normpath(os.path.join(str(folder), f"{s:06d}.counts.zst"))
                    for s in range(max(top, last - A.AHEAD_FRAMES) + 1, last + A.AHEAD_FRAMES + 1))
    if lean:
        return keep
    size = {s: max(0, int(line.get("bytes") or 0)) for s, _, line in counted}
    frame = max((size[s] for s, _, _ in counted[-KEEP_NEWEST:]), default=0)
    room = min(m.spool_cap_mb, 2 * m.spool_cap_mb - others) * 1e6 - 2 * frame \
        - sum(size[s] for s, _, _ in counted if path[s] in there and path[s] in keep)
    for s, _, _ in reversed(counted[-KEEP_NEWEST:]):
        if path[s] in keep or path[s] not in there:
            continue
        if size[s] > room:
            break
        keep.add(path[s])
        room -= size[s]
    rest = [s for s, _, _ in counted if path[s] in there and path[s] not in keep]
    if rest and room > 0:
        mean = sum(size[s] for s in rest) / len(rest)
        count = min(BACKFILL_MAX, len(rest), int(room // mean) if mean > 0 else BACKFILL_MAX)
        if count > 0:
            wanted = np.linspace(rest[0], rest[-1], count) if count > 1 else [(rest[0] + rest[-1]) / 2]
            keep.update(path[s] for s in nearest_each(rest, wanted))
    return keep


def test_a_prune_keeps_the_files_walking_every_counted_frame_kept(tmp_path):
    """`_keep` looks only at the frames whose file a prune could remove (the spool's files, the frames held, the newest
    and the newest with the start block): of the spool's files it keeps those the walk over every counted frame kept
    (keep_walking_every_frame), and nothing that walk did not keep, whatever is held, the run's share of the spool and
    its state: forced files beside spooled ones, files gone already, Clusters & UMAP off, a run started again, a
    squeezed (lean) or an ended run, and a held or ended run whose newest counted frames are forced ones, larger than
    the spooled ones (in its folder, not in the spool: the room kept for the next two counted frames is theirs)."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock, start_every=3)
    for seq in range(1, 61):
        reason = None if seq % 4 == 0 else "upscale" if seq == 21 else "hold" if seq == 45 else "request"
        run.add_frame(reason, has_start=seq % 3 == 0 and seq < 50, size=1000 + 37 * seq)
    for seq in (1, 2, 3, 5, 6, 9, 10, 11, 13, 23):            # pruned before
        run.count_path(seq).unlink()
    m = checked(tmp_path, run, clock=clock)
    m.requested["r1"], m.qc_requested["r1"], m.deeper_requested["r1"] = 6, 14, (15, 1)
    m._qc_state("r1", m._qc_models()).backfill = [17, 18, 19]
    run.wheres = [{"token": 1, "seq": 22, "file": "where-1.bin.zst"}]
    m.compare_files["r1"] = {26: clock.now + 60, 27: clock.now - 1}   # the hold of 27 ran out
    m.compare_queue.append(({"run": "r1", "columns": [{"frame": {"run": "r1", "seq": 29,
                                                                 "counts_file": str(run.count_path(29))}}]}, None))
    there = {os.path.normpath(path) for path in m.runs.spool.files("r1")}
    compared = 0
    for state, umap, live in (("running", True, 50), ("running", True, -1), ("running", True, 99),
                              ("running", False, 50), ("done", False, 58)):
        run.meta["state"] = state
        m.configure("r1", enabled=umap)
        m.configure_qc("r1", enabled=umap)
        m.live_seq["r1"] = live
        for cap, others, lean in ((1024, 0.0, False), (0.03, 0.0, False), (0.03, 0.035, False), (0.002, 0.0, False),
                                  (0.03, 0.0, True)):
            m.spool_cap_mb = cap
            new, old = m._keep("r1", run, there, others, lean), keep_walking_every_frame(m, "r1", run, there, others, lean)
            case = (state, umap, live, cap, others, lean)
            assert new & there == old & there and new <= old, case
            compared += len(old & there) > 0
    assert compared >= 20, "the cases keep files"
    # Held at the end (its hold frame) or ended (hold and end frames): 3 MB each, durable, beside 1 MB spooled ones.
    for state, forced in (("paused", ("hold",)), ("done", ("hold", "end"))):
        large = CountedRun(tmp_path / state, run_id=state, clock=clock, state=state)
        for _ in range(37):
            large.add_frame("request", size=1_000_000)
        for reason in forced:
            large.add_frame(reason, size=3_000_000)
        m = manager(tmp_path / f"m-{state}", large, clock=clock)
        m.spool_cap_mb, m.live_seq[state] = 20.0, large.counted[-1]       # every frame analysed
        there = {os.path.normpath(path) for path in m.runs.spool.files(state)}
        new, old = m._keep(state, large, there), keep_walking_every_frame(m, state, large, there)
        assert len(there) == 37 and len(old & there) == 14, (state, len(old & there))   # 20 MB less 2 x 3 MB
        assert new & there == old & there and new <= old, state


class Reads(list):
    """A run's frame records that count how many of them are read (by index, slice or iteration)."""
    reads = 0

    def __getitem__(self, index):
        found = super().__getitem__(index)
        self.reads += len(found) if isinstance(index, slice) else 1
        return found

    def __iter__(self):
        for record in super().__iter__():
            self.reads += 1
            yield record

    def __reversed__(self):
        for record in super().__reversed__():
            self.reads += 1
            yield record


def long_run(folder, frames: int, spooled: int = 30):
    """A real engine Run (engine/runs.py) of `frames` frames, every other one counted with the start block (a run
    from cells alone): frame 0 by itself in its folder, the others in its spool, which holds the `spooled` newest
    counted files and those of frames 6 and 8 (the rest pruned)."""
    from sopa_companion.engine.runs import Run
    live, spool = folder / "live", folder / "spool"
    (live / "frames").mkdir(parents=True)
    spool.mkdir()
    records = [{"seq": seq, "iter": seq, "phase": "burnin", "t": 0.1 * seq, "file": f"frames/{seq:06d}.bin.zst",
                **GEOMETRY} for seq in range(frames)]
    lines = [{"seq": seq, "iter": seq, "counted_file": f"frames/{seq:06d}.counted.zst", "token": None, "where": None,
              "file": f"frames/{seq:06d}.counts.zst" if seq == 0 else str(spool / f"{seq:06d}.counts.zst"),
              "reason": "init" if seq == 0 else "request", "nnz": 1, "sum_n": 1, "sum_e": 1.0, "sum_f": None,
              "has_f": False, "has_start": True, "saturated": 0, "bytes": 1000, "ms": {"extract": 0.0, "write": 0.0}}
             for seq in range(0, frames, 2)]
    (live / "frames.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (live / "counts.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (live / "info.json").write_text(json.dumps({"cells": 3000, "counts": {
        "mode": "request", "format": "PLC1", "version": 1, "e_scale": 32, "start": True, "start_every": 1,
        "spool": str(spool), "where": 1}}))
    (live / "frames" / "000000.counts.zst").write_bytes(b"PLC1")
    for seq in [line["seq"] for line in lines[-spooled:]] + [6, 8]:
        (spool / f"{seq:06d}.counts.zst").write_bytes(b"PLC1")
    run = Run(folder, {"id": "r1", "state": "running", "kind": "proseg"})
    run.read_new()
    return run


def test_a_prune_and_the_newest_counted_frame_read_only_the_frames_they_need(tmp_path):
    """After every job of a run a prune works out the files to keep (`_keep`), and each new frame has the scheduler
    look for the run's newest counted frame with a file (`_newest_counted`): both read the frames of the spool's files,
    of the frames held and of the newest few (from the end of run.counted), never every counted frame of a long run (a
    prune took 20 ms at 5,000 counted frames, 100 ms at 20,000): as many reads at 4,000 frames as at 1,000. The keep
    set is still the walk's."""
    reads = {}
    for frames in (1000, 4000):
        run = long_run(tmp_path / f"r{frames}", frames)
        counted = list(run.counted)
        assert len(counted) == frames // 2 and counted[-1] == frames - 2
        m = manager(tmp_path / f"m{frames}", run)
        m.requested["r1"], m.compare_files["r1"] = 6, {8: m.clock() + 60}      # asked for, taken by Compare
        run.wheres = [{"token": 1, "seq": 10, "file": "where-1.bin.zst"}]       # a where export (its file gone)
        m.live_seq["r1"] = counted[-4]                                          # the 3 newest still to analyse
        there = {os.path.normpath(str(path)) for path in (run.dir / "spool").iterdir()}
        run.frames = Reads(run.frames)
        keep = m._keep("r1", run, there)
        found = [run.frames.reads]
        assert keep & there == keep_walking_every_frame(m, "r1", run, there) & there
        assert {path for path in there if Path(path).name in ("000006.counts.zst", "000008.counts.zst")} <= keep
        run.frames.reads = 0
        assert m._newest_counted(run) == counted[-1]
        found.append(run.frames.reads)
        for seq in counted[-(NEWEST_LOOK + 4):]:                  # the newest gone: looked for further back
            (run.dir / "spool" / f"{seq:06d}.counts.zst").unlink()
        run.frames.reads = 0
        assert m._newest_counted(run) == counted[-(NEWEST_LOOK + 5)]
        found.append(run.frames.reads)
        reads[frames] = found
    assert reads[1000] == reads[4000], reads
    keep_reads, newest_reads, back_reads = reads[4000]
    assert keep_reads <= 32 + 3 + KEEP_NEWEST + 1 and newest_reads <= NEWEST_LOOK and back_reads <= 2 * NEWEST_LOOK, \
        reads


def snap_walking_every_frame(run, seq, kept=None) -> int | None:
    """`_snap` as it was until it looked from `seq` in run.counted: every frame of the run walked on each request."""
    with run.lock:
        counted = [(s, frame["counts_file"]) for s, frame in enumerate(run.frames) if frame.get("counts_file")]
    i = bisect.bisect_right([s for s, _ in counted], seq)
    for index in [*range(i - 1, -1, -1), *range(i, len(counted))]:
        s, name = counted[index]
        if (kept is not None and kept(s)) or (run.live / name).exists():
            return s
    return None


def start_counts_walking_every_frame(run, seq) -> str | None:
    """`_start_counts` as it was: every frame up to `seq` walked on each pass of a run with a start line."""
    with run.lock:
        names = [run.frames[s]["counts_file"] for s in range(min(seq, len(run.frames) - 1), -1, -1)
                 if run.frames[s].get("counts_file") and (run.frames[s].get("counts") or {}).get("has_start")]
    return next((str(run.live / name) for name in names if (run.live / name).exists()), None)


def counted_with_files_walking_every_frame(run, below) -> list[int]:
    """`_counted_with_files` as it was: every frame before `below` walked, the file of each counted one looked at."""
    with run.lock:
        counted = [(s, frame["counts_file"]) for s, frame in enumerate(run.frames[:max(0, below)])
                   if frame.get("counts_file")]
    return [s for s, name in counted if (run.live / name).exists()]


def test_a_request_s_frame_its_start_block_and_backfill_s_files_are_those_the_walk_found(tmp_path):
    """An explicit request's frame (`_snap`: down from it, then up, a kept result counting as a file), the start block a
    pass pairs its frame with (`_start_counts`) and the files a check's backfill chooses from (`_counted_with_files`)
    are looked for in run.counted from where they are wanted: at every frame they are those the walk over every frame
    found, with forced files beside spooled ones, frames not counted, long stretches of files gone and kept results."""
    run = CountedRun(tmp_path / "r1", start_every=3)
    for seq in range(1, 61):
        reason = None if seq % 4 == 0 else "upscale" if seq == 21 else "hold" if seq == 45 else "request"
        run.add_frame(reason, has_start=seq % 3 == 0 and seq < 50)
    for seq in (0, 1, 2, 3, 5, 6, 9, 10, 11, 13, *range(14, 44)):  # pruned before: frame 0's too
        if run.frames[seq].get("counts_file"):
            run.count_path(seq).unlink()
    m = manager(tmp_path, run)
    snap, start = A.AnalysisManager._snap, A.AnalysisManager._start_counts
    for kept in (None, {7, 31}.__contains__, {58}.__contains__, {2}.__contains__):
        for seq in range(-2, 64):
            assert snap(run, seq, kept) == snap_walking_every_frame(run, seq, kept), (seq, kept)
    for seq in range(-2, 64):
        assert start(run, seq) == start_counts_walking_every_frame(run, seq), seq
        assert m._counted_with_files(run, seq) == counted_with_files_walking_every_frame(run, seq), seq
    assert (snap(run, 2), snap(run, 40), snap(run, 40, {31}.__contains__)) == (7, 7, 31)   # the oldest after; far down
    assert (start(run, 44), start(run, 60)) == (None, str(run.count_path(45)))           # none; the hold's
    assert m._counted_with_files(run, 61) == [7, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59]


def test_a_request_s_frame_its_start_block_and_backfill_s_files_read_only_what_they_need(tmp_path, monkeypatch):
    """An explicit request's frame (`_snap`) and a pass's start block (`_start_counts`) read the frames from where they
    are wanted on in run.counted, never every frame of a long run (a call cost about 1 and 3 ms at 5,000 counted frames,
    8 and 14 ms at 20,000); the files a check's backfill chooses from (`_counted_with_files`) are looked at by one
    listing of the spool and the forced frames' few files one by one (each file looked at: about 100 ms at 20,000). As
    many at 4,000 frames as at 1,000."""
    found = {}
    for frames in (1000, 4000):
        run = long_run(tmp_path / f"r{frames}", frames)
        counted, m = list(run.counted), manager(tmp_path / f"m{frames}", run)
        newest, spool = counted[-1], run.dir / "spool"
        run.frames = Reads(run.frames)
        reads = []
        for seq, snapped in ((newest, newest), (newest + 50, newest), (9, 8)):     # 6 and 8 kept in the spool
            run.frames.reads = 0
            assert A.AnalysisManager._snap(run, seq) == snapped, seq
            reads.append(run.frames.reads)
        for seq, block in ((newest, newest), (9, 8)):
            run.frames.reads = 0
            assert A.AnalysisManager._start_counts(run, seq) == str(spool / f"{block:06d}.counts.zst"), seq
            reads.append(run.frames.reads)
        looks = {"exists": 0, "listdir": 0}
        exists, listdir = type(run.live).exists, os.listdir

        def looked(self, *args, **kwargs):
            looks["exists"] += 1
            return exists(self, *args, **kwargs)

        def listed(*args, **kwargs):
            looks["listdir"] += 1
            return listdir(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(type(run.live), "exists", looked)
            patch.setattr(os, "listdir", listed)
            assert m._counted_with_files(run, newest) == [0, 6, 8, *counted[-30:-1]]
        found[frames] = (reads, looks)
    assert found[1000] == found[4000], found
    reads, looks = found[4000]
    assert max(reads) <= NEWEST_LOOK and looks == {"exists": 1, "listdir": 1}, found


def test_large_frames_keep_only_the_newest_that_fit_so_tokens_never_stop(tmp_path):
    """Frames of 150 MB (a whole sample with its start block): the newest stay only while they fit with two more
    frames in the run's cap, so its spool never reaches the cap and every pass asks for the next counted frame."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock)
    m.runs.spool.cap_mb = 1024
    issued = drive(m, ["r1"], 40, clock, 150_000_000)
    assert [step["r1"] for step in issued] == list(range(1, 41)), "a token with every pass"
    newest = len(run.frames) - 1
    assert spooled(run) == list(range(newest - 4, newest + 1)), "the 4 newest (600 MB) and the one counted since"
    assert not [text for _, run_id, text in m.events if run_id == "r1"], "never stopped: nothing said"


def test_an_ended_run_s_spool_gives_way_to_the_runs_that_sample(tmp_path):
    """An ended run's spool (its newest and a reserve, 880 MB) gets no new frame and no job: once it leaves two
    sampling runs less than their cap it keeps only what a job needs, so both keep asking for counted frames within
    the engine's 2,048 MB."""
    clock = Clock()
    ended = CountedRun(tmp_path / "A", run_id="A", clock=clock)
    for _ in range(60):
        ended.add_frame("request", size=55_000_000)
    ended.add_frame("end", size=55_000_000)
    ended.meta["state"] = "done"
    b, c = (CountedRun(tmp_path / name, run_id=name, clock=clock) for name in ("B", "C"))
    m = manager(tmp_path, ended, b, c, clock=clock)
    store = m.runs
    store.spool.cap_mb = 1024
    m.live_seq["A"] = 61
    m._prune("A")                                             # its last prune, alone: its newest and a reserve
    assert store.spool.usage_mb("A") == pytest.approx(880.0) and "A" in m.spooled and not m.squeezed
    issued = drive(m, ["B", "C"], 150, clock, 55_000_000)
    assert min(issued[-1].values()) >= 74, issued[-1]          # a token every other step each: they take turns
    assert "A" in m.squeezed and spooled(ended) == [], "no frame of it asked for or still to analyse"
    assert store.spool.usage_mb() + 55 <= 2 * 1024
    assert all(len(spooled(run)) >= KEEP_NEWEST for run in (b, c)), "each keeps its newest"
    # The engine's spool full (with files of other runs than these, say): every run with a spool is pruned, the ended
    # ones first and to what a job needs.
    late = CountedRun(tmp_path / "D", run_id="D", clock=clock)
    for _ in range(20):
        late.add_frame("request", size=1000)
    late.meta["state"] = "stopped"
    store.runs["D"] = late
    m._prune("D")                                             # no pressure from it: it keeps its frames
    assert len(spooled(late)) == 20 and "D" not in m.squeezed
    order, prune = [], m._prune
    m._prune = lambda run_id, run=None: (order.append(run_id), prune(run_id, run))[1]
    m.to_prune.clear()
    with m.lock:
        m._counts_wait("B", "engine_cap", clock.now + 5.0)
    assert m.to_prune == {"A", "B", "C", "D"} and "D" in m.squeezed
    m._prune_runs(m.to_prune)
    assert order[:2] == ["A", "D"] and sorted(order[2:]) == ["B", "C"], order
    assert spooled(late) == []
    late.meta["state"] = "running"                            # proseg started again: it samples, it is not squeezed
    m._prune("D")
    assert "D" not in m.squeezed


def test_a_full_spool_is_said_again_only_after_tokens_went_for_a_while(tmp_path):
    """A spool at its cap that lets one token go and stops the next is said once in the log, not with every swing;
    once tokens went COUNTS_NOTE_S since its last stop, a new stop is said again."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = manager(tmp_path, run, clock=clock)
    store = m.runs
    full = "Proseg's count is not asked for while this run's spool is full"

    def step(stopped: bool):
        clock.now += 1.0
        if stopped:
            store.spool.why["r1"] = "run_cap"
        else:
            store.spool.why.pop("r1", None)
        job, _ = m._next_job()
        if job is not None:
            finish(m, job)
        ask(m)
        if store.counts_status("r1")["outstanding"]:
            store.serve("r1")
    for turn in range(20):
        step(turn % 2 == 0)
    assert store.tokens["r1"][0] == 10, "every other step a token went"
    assert [text for _, run_id, text in m.events if run_id == "r1"] == [full]
    for _ in range(int(A.COUNTS_NOTE_S) + 1):
        step(False)
    step(True)
    assert [text for _, run_id, text in m.events if run_id == "r1"] == [full, full]


def test_a_watched_run_s_live_passes_go_before_backfill(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    for seq in range(1, 21):
        run.add_frame("request")
    m = checked(tmp_path, run, clock=clock)
    m.configure("r1", enabled=False)                          # the check alone
    store = m.runs
    order = []
    for _ in range(4):
        job, _ = m._next_job()
        order.append(job.kind)
        finish(m, job)
        store.serve("r1")                                     # a new counted frame is always due
    assert order == ["live", "backfill", "live", "backfill"], order
    m.qc_provider.watched.add("r1")                           # a Watch runs: its frames first
    order = []
    for turn in range(4):
        job, _ = m._next_job()
        order.append(job.kind)
        finish(m, job)
        if turn < 3:
            store.serve("r1")
    assert order == ["live"] * 4, order
    job, _ = m._next_job()                                    # caught up: backfill goes meanwhile
    assert job.kind == "backfill"
    assert m._look(m._qc_models(locked=True), locked=True)["r1"]["watched"] is True
    # Beside another run: while the watched run has a pass due, no background pass goes; live passes take turns.
    other = CountedRun(tmp_path / "r2", run_id="r2", clock=clock)
    for seq in range(1, 21):
        other.add_frame("request")
    m = checked(tmp_path, run, other, clock=clock, watched={"r1"})
    for run_id in ("r1", "r2"):
        m.configure(run_id, enabled=False)
    order = []
    for _ in range(4):
        clock.now += 0.01
        job, _ = m._next_job()
        order.append((job.kind, job.run))
        finish(m, job)
        m.runs.serve(job.run)
    assert order == [("live", "r1"), ("live", "r2"), ("live", "r1"), ("live", "r2")], order
    m.qc_provider.watched.clear()                             # the Watch ended: a background pass after each live one
    clock.now += 0.01
    job, _ = m._next_job()
    assert job.kind == "backfill"
    finish(m, job)
    clock.now += 0.01
    assert m._next_job()[0].kind == "live"


# ---- the start line and the requests' count files -----------------------------------------------------------------
def start_run(tmp_path, run_id="r1", meta=CELLS_START, start_every=1, **kwargs) -> CountedRun:
    """A run that started from cells alone (its prior exported), counted with proseg's start block."""
    run = CountedRun(tmp_path / run_id, run_id=run_id, meta=meta, start_every=start_every, **kwargs)
    run.info["prior"] = {"file": "prior.bin.zst", "format": "dense", **GEOMETRY}
    (run.live / "prior.bin.zst").write_bytes(b"")
    return run


def test_the_start_line_is_proseg_s_count_at_iteration_0_for_cells_only_starts(tmp_path):
    run = start_run(tmp_path)
    run.add_frame("request")
    m = checked(tmp_path, run)
    m.configure("r1", enabled=False)
    kinds = []
    while (job := m._next_job()[0]) is not None:
        kinds.append((job.kind, job.seq, job.ops))
        finish(m, job)
    assert ("start", None, ("qc_start",)) in kinds
    calls = []
    stub_worker(m, calls)
    qc = m.qc_runs["r1"]
    qc.start = "pending"
    assert m._process(Job("r1", None, ("qc_start",), "start", qc)) is None and qc.start == "done"
    request = calls[-1]
    assert request["op"] == "qc_start" and request["counts_file"] == str(run.live / "frames" / "000000.counts.zst")
    assert request["prior"] == {"file": str(run.live / "prior.bin.zst"), "format": "dense", **GEOMETRY}
    assert request["qc"]["out_start"] == str(qc_folder(run.dir, MODEL[0]) / "start.npz")
    # Starts with nuclei (nuclei only, or cells with their nuclei: a Xenium bundle's) have no start line.
    for name, meta in (("paired", {"sample_kind": "zarr", "prior": {"cells": {"kind": "shapes", "key": "cells"},
                                                                     "nuclei": {"kind": "shapes", "key": "nuclei"}}}),
                       ("nuclei", {"sample_kind": "zarr", "prior": {"cells": None, "nuclei": {"key": "nuclei"}}}),
                       ("bundle", {"sample_kind": "xenium"})):
        other = start_run(tmp_path, name, meta=meta, start_every=None)
        m = checked(tmp_path, other)
        while (job := m._next_job()[0]) is not None:
            assert job.kind != "start", name
            finish(m, job)
        assert m.describe_qc(name)["start"] == "none", name
    # Frame 0 not counted yet (its line has not come), or its file gone: no start line yet, nor a failure.
    late = start_run(tmp_path, "late", init=False)
    late.add_frame(None)
    m = checked(tmp_path, late)
    m._next_job()
    assert m.qc_runs["late"].start == "pending"
    late.count(0, "init")
    late.count_path(0).unlink()
    assert m._background("late", late, m.qc_runs["late"], 0.0) is None
    late.count(0, "init")
    assert m._background("late", late, m.qc_runs["late"], 0.0).ops == ("qc_start",)


def test_the_start_line_waits_for_frame_0_s_own_count_and_the_prior(tmp_path):
    """The start line is proseg's own count at iteration 0: frame 0's counts line of reason "init" (a frame 0 counted
    for another reason is no start), and the prior's footprints place the starting cells (it waits for that file)."""
    run = start_run(tmp_path, init=False)
    run.add_frame(None)
    run.count(0, "request")
    m = checked(tmp_path, run)
    qc = m._qc_state("r1", m._qc_models())
    assert m._init_count(run) is None and m._background("r1", run, qc, 0.0) is None and qc.start == "pending"
    run.count(0, "init")
    (run.live / "prior.bin.zst").unlink()                     # not written yet
    assert m._init_count(run) is not None and m._background("r1", run, qc, 0.0) is None and qc.start == "pending"
    (run.live / "prior.bin.zst").write_bytes(b"")
    job = m._background("r1", run, qc, 0.0)
    assert job.kind == "start" and job.ops == ("qc_start",)


def test_requests_name_the_frame_s_count_and_the_start_block_it_pairs_with(tmp_path):
    run = start_run(tmp_path, start_every=None)               # start blocks only where a line says it has one
    run.count(0, "init", has_start=True)
    run.add_frame("request", has_start=True)                  # 1
    run.add_frame("request")                                  # 2: pairs with 1's block
    run.add_frame(None)                                       # 3
    m = checked(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    assert m._process(Job("r1", 2, ("umap", "qc"), "live", qc)) is None
    request = calls[-1]
    assert request["counts_file"] == str(run.count_path(2))
    assert request["init_counts_file"] == str(run.count_path(0))
    assert request["start_counts_file"] == str(run.count_path(1))
    assert "prior" not in request["qc"]
    assert request["qc"]["out_cells"] == str(qc_folder(run.dir, MODEL[0]) / "000002.qc.bin")
    assert m._process(Job("r1", 1, ("qc",), "live", qc)) is None
    assert calls[-1]["start_counts_file"] == str(run.count_path(1)), "its own block"
    run.count_path(1).unlink()
    run.add_frame("request")                                  # 4: 1's file gone, so frame 0's block
    assert m._process(Job("r1", 4, ("qc",), "live", qc)) is None
    assert calls[-1]["start_counts_file"] == str(run.count_path(0))
    # A UMAP pass alone names no start; a start with nuclei names none either.
    m._process(Job("r1", 4, ("umap",), "request"))
    assert calls[-1]["ops"] == ["umap"] and "init_counts_file" not in calls[-1]
    paired = start_run(tmp_path, "paired", meta={"sample_kind": "xenium"})
    paired.add_frame("request")
    m.runs.runs["paired"] = paired
    m.configure_qc("paired", enabled=True)
    pqc = m._qc_state("paired", m._qc_models())
    m._process(Job("paired", 1, ("qc",), "live", pqc))
    assert calls[-1]["run"] == "paired" and "init_counts_file" not in calls[-1] and "start_counts_file" not in calls[-1]
    # A frame not counted: its ops fail with no_counts, the worker is not asked.
    count = len(calls)
    assert m._process(Job("r1", 3, ("qc",), "request", qc)) == NO_COUNT and len(calls) == count
    assert m.qc_status["r1"]["error"] == NO_COUNT


# ---- the count each frame is read on (counts_file.count_of) ------------------------------------------------------------
SAVED = ("pause", "hold", "finish", "end")                  # proseg counts these with f, the count it saves


def test_each_request_names_the_count_its_frame_is_read_on(tmp_path):
    """A frame proseg counted for a pause, the hold, a finish or the end (its line holds f) is read on the count proseg
    saves: its request says count "sampled" for every op, and the check also writes the frame's bytes on the averaged
    count (out_cells_e: <seq>.qc.averaged.bin, which the test reads), both handed to the provider; a frame counted at
    the start, an upscale, when asked or every n iterations is read on the averaged count, with no second file. Each
    result is kept for the count it was read on."""
    run = CountedRun(tmp_path / "r1")                         # 0: init
    reasons = ("init", "upscale", "request", "every") + SAVED
    for reason in reasons[1:]:
        run.add_frame(reason)                                 # 1 .. 7
    m = checked(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    for seq, reason in enumerate(reasons):
        assert m._process(Job("r1", seq, ("umap", "qc"), "request", qc)) is None, reason
        sampled = reason in SAVED
        count = "sampled" if sampled else "averaged"
        request, record = calls[-1], m.qc_provider.records[-1]
        assert (request["count"], record["count"], qc.done[seq]) == (count, count, count), reason
        e_bytes = qc.dir / f"{seq:06d}.qc.averaged.bin"
        assert request["qc"].get("out_cells_e") == record["cells_e"] == (str(e_bytes) if sampled else None), reason
        assert e_bytes.exists() is sampled and record["cells"] == str(qc.dir / f"{seq:06d}.qc.bin"), reason
        assert m.summary("r1", seq) == {"seq": seq, **({"count": "sampled"} if sampled else {})}, reason
    assert m.result_counts("r1") == {seq: "sampled" if reason in SAVED else "averaged"
                                     for seq, reason in enumerate(reasons)}


def test_a_frame_counted_again_is_analysed_again_on_the_count_proseg_saves(tmp_path):
    """A frame analysed on a token's count (averaged) that the hold then counts again (the same seq, its line now
    holding f) is analysed again by itself on f, Clusters & UMAP and the check both: the provider gets the check again
    with count "sampled", the result is kept and announced on that count (result_counts) and the averaged one is never
    served again; nothing more is due after. A frame whose new count's file is gone before it is read again keeps its
    result as it was (it cannot be read again): asked for, it is served without a worker."""
    run = CountedRun(tmp_path / "r1")                         # 0: init
    run.add_frame("request")                                  # 1: a token's count
    m = checked(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    for seq in (0, 1):
        assert m._process(Job("r1", seq, ("umap", "qc"), "request", qc)) is None
    assert m.summary("r1", 1) == {"seq": 1} and qc.done == {0: "averaged", 1: "averaged"}
    assert m._next_job() == (None, POLL_S)
    run.meta["state"] = "paused"
    run.count(1, "hold")                                      # the hold counts frame 1 again: durably, with f
    job, _ = m._next_job()
    assert job == ("r1", 1) and job.kind == "live" and set(job.ops) == {"umap", "qc"}
    assert m._process(job) is None and calls[-1]["count"] == "sampled"
    assert m.summary("r1", 1) == {"seq": 1, "count": "sampled"} and m.result_counts("r1") == {0: "averaged", 1: "sampled"}
    assert json.loads(m._paths(run, 1)[0].read_text()) == {"seq": 1, "count": "sampled"}
    assert [(r["seq"], r["count"]) for r in m.qc_provider.records if r["kind"] == "frame"] == \
        [(0, "averaged"), (1, "averaged"), (1, "sampled")]
    assert qc.done[1] == "sampled" and m._next_job() == (None, POLL_S)
    # Counted again, its file gone before it was read: the result kept stands, served as it is.
    run.meta["state"] = "running"
    run.add_frame("request")                                  # 2
    assert m._process(Job("r1", 2, ("umap", "qc"), "request", qc)) is None
    run.meta["state"] = "paused"
    run.count(2, "hold")
    run.count_path(2).unlink()
    asked = len(calls)
    assert m.configure("r1", seq=2)["requested"] == 2
    job, _ = m._next_job()
    assert job == ("r1", 2) and job.ops == ("umap",), "its check stands: not scored again"
    assert m._process(job) is None and len(calls) == asked and m.summary("r1", 2) == {"seq": 2}
    assert m.result_counts("r1")[2] == "averaged" and qc.done[2] == "averaged"
    assert m.configure_qc("r1", seq=2)["requested"] == 2
    assert m._next_job()[0] is None and len(calls) == asked, "its check kept stands: nothing to score again"


def test_a_frame_counted_again_is_scored_again_however_it_is_asked_for(tmp_path):
    """A frame scored on a token's count (averaged) that the hold then counts again (its line now holds f, its file
    there) is scored again on f by whichever request reaches it: the check's own (configure_qc: the check alone), the
    Clusters & UMAP request (configure: the check rides along) and Go deeper's (request_deeper: the card and the window
    together). A check an earlier session left on f (its <seq>.qc.json says "sampled") stands for a frame read on f:
    a new session never scores it again."""
    run = CountedRun(tmp_path / "r1")                         # 0: init
    run.add_frame("request")                                  # 1: a token's count
    m = checked(tmp_path, run)
    calls = []
    stub_worker(m, calls)
    qc = m._qc_state("r1", m._qc_models())
    for seq in (0, 1):
        assert m._process(Job("r1", seq, ("umap", "qc"), "request", qc)) is None
    assert qc.done == {0: "averaged", 1: "averaged"} and m._next_job() == (None, POLL_S)
    run.meta["state"] = "paused"
    run.count(1, "hold")                                      # the hold counts frame 1 again: durably, with f
    for asked, ops in ((lambda: m.configure_qc("r1", seq=1), ("qc",)),
                       (lambda: m.configure("r1", seq=1), ("umap", "qc")),
                       (lambda: m.request_deeper("r1", 1, 3), ("qc", "deeper"))):
        asked()
        job, _ = m._next_job()
        assert job == ("r1", 1) and job.kind == "request" and tuple(job.ops) == ops, (job.kind, job.ops)
    assert m._process(job) is None and calls[-1]["count"] == "sampled" and qc.done[1] == "sampled"
    # A new session finds the check on f the first left (its summary written beside its bytes, as the check writes
    # them): it stands, the frame's UMAP alone is worked out (this session holds no UMAP result of it).
    (qc.dir / "000001.qc.json").write_text(json.dumps({"kind": "frame", "seq": 1, "count": "sampled"}))
    again = checked(tmp_path / "again", run)
    stub_worker(again, [])
    found = again._qc_state("r1", again._qc_models())
    assert found.dir == qc.dir and found.done == {1: None}
    job, _ = again._next_job()
    assert job == ("r1", 1) and tuple(job.ops) == ("umap",) and found.done[1] == "sampled"


def test_a_reply_on_another_count_than_asked_is_refused_and_not_asked_again(tmp_path):
    """A worker that read another count than the one asked for (an older one that knows no `count` reads the averaged
    one) has its results refused: nothing kept (its files removed, nothing handed to the provider), each op failed
    with the reason count_mismatch and the count asked for, the check backing off as after any failure; the same frame
    is not asked again on that count by itself, a frame counted after it is."""
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)            # 0: init
    m = checked(tmp_path, run, clock=clock)
    calls, sent = [], []
    stub_worker(m, calls)
    current = m._call

    def older(request):                                       # reads e whatever it is asked
        sent.append(request["count"])
        return current({key: value for key, value in request.items() if key != "count"})
    m._call = older
    qc = m._qc_state("r1", m._qc_models())
    assert m._process(Job("r1", 0, ("umap", "qc"), "request", qc)) is None and sent == ["averaged"]
    run.meta["state"] = "paused"
    paused = run.add_frame("pause")                           # 1: read on f
    job, _ = m._next_job()
    assert job == ("r1", paused) and set(job.ops) == {"umap", "qc"}
    failure = m._process(job)
    assert failure == "the worker read the averaged count, not the sampled one asked for" and sent[-1] == "sampled"
    assert m.status["r1"] == {"state": "error", "seq": 1, "error": failure, "reason": "count_mismatch",
                              "count": "sampled"}
    assert m.qc_status["r1"] == {"state": "error", "seq": 1, "error": failure, "count": "sampled"}
    assert 1 not in qc.done and 1 not in m.results.get("r1", {}) and qc.retry_at > clock.now
    assert not any(path.exists() for path in m._paths(run, 1))
    assert [(r["kind"], r["seq"]) for r in m.qc_provider.records] == [("frame", 0), ("error", 1)]
    clock.now += 1.0
    assert m._next_job()[0] is None and len(sent) == 2, "not asked again on the same count"
    clock.now = qc.retry_at + 1.0
    run.meta["state"] = "running"
    run.add_frame("request")                                  # 2: a newer frame
    job, _ = m._next_job()
    assert job == ("r1", 2) and set(job.ops) == {"umap", "qc"}


# ---- Go deeper's where box, a test's runs, the scheduler's thread -------------------------------------------------------
def test_a_followed_cell_s_box_rides_with_the_tokens(tmp_path):
    clock = Clock()
    run = CountedRun(tmp_path / "r1", clock=clock)
    m = checked(tmp_path, run, clock=clock)
    m.follow_where("r1", [1, 2, 3, 4])                        # no cell followed: ignored
    assert "r1" not in m.where_box
    m.follow_deeper("r1", 5)
    m.follow_where("r1", (110.0, 60.0, 130.5, 80.25))
    assert m.where_box == {"r1": [110.0, 60.0, 130.5, 80.25]}
    job, _ = m._next_job()
    assert job.kind == "live" and "deeper" in job.ops and m.to_ask == {"r1": [110.0, 60.0, 130.5, 80.25]}
    ask(m)
    assert m.runs.wheres == [("r1", [110.0, 60.0, 130.5, 80.25], 0)] and m.runs.tokens["r1"] == [1, 0]
    for bad in ([1, 2, 3], [0, 0, 0, 1], [0, 0, float("nan"), 1], "box", [1, 2, 0, 4]):
        with pytest.raises(AnalysisError) as refused:
            m.follow_where("r1", bad)
        assert refused.value.status == 422
    assert m.where_box["r1"] == [110.0, 60.0, 130.5, 80.25]
    m.follow_where("r1", None)
    assert "r1" not in m.where_box
    m.follow_where("r1", [1, 2, 3, 4])
    m.stop_deeper("r1")                                       # cleared with the follow
    assert "r1" not in m.where_box
    m.follow_deeper("r1", 5)
    m.follow_where("r1", [1, 2, 3, 4])
    m.follow_deeper("r1", None)
    assert "r1" not in m.where_box
    m.follow_deeper("r1", 5)
    m.follow_where("r1", [1, 2, 3, 4])
    clock.now += A.DEEPER_FOLLOW_TTL_S                        # its lease ran out
    m._next_job()
    assert "r1" not in m.where_box and "r1" not in m.deeper_follow
    # The box is a lease of its own: renewed by each box handed over, it goes alone once none came for
    # WHERE_BOX_TTL_S (the view turned off while the window, still following the cell, renews its follow).
    m.follow_deeper("r1", 5)
    m.follow_where("r1", [1, 2, 3, 4])
    clock.now += A.WHERE_BOX_TTL_S - 1.0
    m.follow_where("r1", [1, 2, 3, 5])                        # the next reply's box: renewed
    clock.now += A.WHERE_BOX_TTL_S - 1.0
    m.follow_deeper("r1", 5)                                  # the window renews its follow, not the box
    m._next_job()
    assert m.where_box == {"r1": [1.0, 2.0, 3.0, 5.0]}
    clock.now += 1.0
    m._next_job()
    assert "r1" not in m.where_box and m.deeper_follow == {"r1": 5}
    asks = m.to_ask.get("r1", "none asked")
    assert asks is None or asks == "none asked", asks          # a token asked now goes without a where box
    # A box the store refuses (engine/runs.py checks it as a view): the count alone is asked for, said once.
    def refused(run_id, box, wait_s=2.0):
        raise ValueError("the box is not on the run's raster")
    m.runs.request_where = refused
    tokens = m.runs.tokens["r1"]
    tokens[1] = tokens[0]
    for _ in range(2):
        m._ask_counts({"r1": [1, 2, 3, 4]})
        tokens[1] = tokens[0]                                 # served
    assert tokens == [3, 3]
    said = [text for _, run_id, text in m.events if run_id == "r1"]
    assert said == ["Proseg's where export could not be asked for: ValueError: the box is not on the run's raster"]


def test_a_test_s_own_runs_are_left_to_the_test(tmp_path):
    """The corner's view loop asks for A1's and B1's counts (engine/qc_test.py) and the test prunes its runs' spools:
    the scheduler does neither, even for a child a tool switched on."""
    child = CountedRun(tmp_path / "c1", run_id="c1",
                       meta={"qc_test": {"of": "r1", "role": "A2", "side": "A", "test": "t"}})
    m = manager(tmp_path, child)
    m.configure("c1", enabled=True)
    job, _ = m._next_job()
    assert job == ("c1", 0) and not m.to_ask
    m.runs.serve("c1")
    m._prune("c1")
    assert m.runs.spool.pruned == [] and m.runs.tokens.get("c1", [0, 0])[0] == 0


def test_the_scheduler_s_thread_reads_each_counted_frame_once_with_one_token_out(tmp_path):
    """The loop end to end: a pass on the newest counted frame asks for the next token (after the lock), proseg counts
    the frame after it for that token while frames between stay uncounted, the next pass takes it; a token is never
    asked for while one is out; each pass is followed by a prune of the run's spool, which stays within what it
    keeps."""
    run = CountedRun(tmp_path / "r1")
    m = manager(tmp_path, run)
    store = m.runs
    passes, out_before, pruned = [], [], []
    lock = threading.Lock()

    def call(request):
        passes.append(request["seq"])
        time.sleep(0.005)
        Path(request["out_json"]).write_text(json.dumps({"seq": request["seq"]}))
        Path(request["out_cells"]).write_bytes(b"")
        return {"run": "r1", "seq": request["seq"], "ok": True, "summary": {"seq": request["seq"]}}
    m._call = call
    asked, prune = store.request_counts, m._prune

    def counting(run_id):
        with lock:
            issued, served = store.tokens.get(run_id, [0, 0])
            out_before.append(issued - served)
            return asked(run_id)
    store.request_counts = counting
    m._prune = lambda run_id, run=None: (pruned.append(run_id), prune(run_id, run))[1]
    stop = threading.Event()

    def proseg():                                             # an iteration every 2 ms; counts when a token waits
        while not stop.is_set():
            with lock:
                issued, served = store.tokens.get("r1", [0, 0])
                if issued > served:
                    store.serve("r1")
                else:
                    run.add_frame(None)
            time.sleep(0.002)
    sampler = threading.Thread(target=proseg, daemon=True)
    m.thread.start()
    sampler.start()
    try:
        wait_for(lambda: len(passes) >= KEEP_NEWEST + BACKFILL_MAX + 12, timeout=60)
    finally:
        stop.set()
        sampler.join(5)
        m.close()
        m.thread.join(5)
    counted = {frame["seq"] for frame in run.frames if frame.get("counts_file")}
    assert set(passes) <= counted and len(set(passes)) == len(passes), "each counted frame once, never an uncounted one"
    assert out_before and max(out_before) == 0, "never asked while one is out"
    assert len(pruned) >= len(passes) - 1 and store.spool.pruned, "pruned after its passes"
    assert len(store.spool.files("r1")) <= KEEP_NEWEST + BACKFILL_MAX + 2
