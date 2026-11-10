"""Test a setting here, the engine's side (engine/qc_test.py), on stand-ins: the one input a test's runs share; the small
rules (roles, factorisation, the schedule's end, the pool route's query, the parent's defaults, the hold, how a run
started, the crop); the POST's checks in their order, with tests/fake_pools.py's FakePools; a waiting test (the app's
own steps, its loop, its start once nothing holds, its cancel); and the test's thread on a fake Runs and clock: its
input, matching and runs (all at once from the shared input, the setting on side B only, made paused and given the
parent's hold at their own scale before they sample, then resumed), its following (progress,
pushes, a run that failed or stopped, the time limit, every run made again without factorisation), the answer as
engine/qc_rule.py gives it with its caveats, and its runs and input cleared away. The corner view on hand-made runs
(test_deeper_where's) under the engine's real Runs. The readout and the rule are stand-ins here (their own tests are
test_qc_readout.py's and test_qc_rule.py's); a test's life through the API, with the real Runs and the fake proseg, is
test_qc_test.py's."""
from __future__ import annotations

import base64
import dataclasses
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from conftest import wait_for
import counts_fixtures as CF
from fake_pools import FakePools
from test_qc_pool import measured_on_proseg  # noqa: F401  (test_qc_pool's world: no caveat)
from sopa_companion.engine import qc_pool, qc_readout, qc_rule
from sopa_companion.engine import qc_rules as R
from sopa_companion.engine import qc_test as T
from sopa_companion.engine.qc import QcError
from sopa_companion.engine.runs import ACTIVE, RunError

# The schedule proseg-live reports in info.json for its defaults (the one the test was measured on).
SCHEDULE = {"init_iterations": 20, "burnin": 200, "samples": 200, "recorded": 100, "hillclimb": 50,
            "morphology_steps": 4000}
# The parent: a detail run of a Xenium bundle (started from nuclei and cells), 700 µm across, paused at the end of
# proseg's default schedule (iteration 450, frame 451): where a test may answer, on the square around a cell.
REGION = [100.0, 100.0, 800.0, 800.0]
PARENT = {"id": "p", "kind": "proseg", "sample": "s", "sample_name": "s", "sample_kind": "xenium", "sample_path": None,
          "state": "paused", "paused": True, "mode": "detail", "options": {}, "live_values": {}, "prior": None,
          "box": REGION, "region": {"kind": "box", "box": REGION, "bbox": REGION},
          "schedule_done": {"iter": 450, "seq": 451}}
INFO = {"cells": 50, "genes": 2, "transcripts": 5000, "schedule": SCHEDULE,
        "counts": {"mode": "request", "format": "PLC1", "version": 1, "e_scale": 32, "start": False,
                   "start_every": None, "spool": None, "where": 1}}     # proseg-live keeping its own counts
# A test run's way, one step of its world at a time: loading, its first frame, sampling, past sampling, done.
DONE = [("running", None), ("running", 0), ("running", 200), ("running", 400), ("finishing", 450), ("done", 450)]
# What a test's items and their order look like once matched (engine/qc_pool.py match_input); the readout and the rule
# are stand-ins here, so only their passing on matters.
ITEMS = qc_readout.Items(kind=np.array([0, 1, 2], np.int8), cell=np.zeros(3, np.int32),
                         giver=np.array([1, 1, -1], np.int32), target=np.array([-1, 0, -1], np.int64),
                         gene=np.array([0, 1, 1], np.int32), transcript_id=np.array([7, 8, 9], np.uint64),
                         x=np.array([350.25, 351.5, 352.5], np.float32), y=np.array([350.5, 351.5, 352.5], np.float32),
                         cell_names=("a-1", "b-1"), gene_names=("G1", "G2"))
ORDER = qc_rule.PoolOrder(seed=0, cells=(0,), distance_um=(0.0,), parent_cells=(7,))
COUNTS = {"target": {"ok": 186, "not_found": 1, "unsure": 2}, "matched": {"ok": 540, "not_found": 0, "unsure": 0},
          "uniform": {"ok": 1150, "not_found": 0, "unsure": 0}}


# ---- stand-ins ---------------------------------------------------------------------------------------------------------
def area(rule="square", box=(200.0, 200.0, 700.0, 700.0)) -> qc_pool.Area:
    """A test's area as engine/qc_pool.py gives it: its box, inset by 25 µm, and the region its runs are made over."""
    x0, y0, x1, y1 = (float(v) for v in box)
    spec = {"kind": "box", "box": [x0, y0, x1, y1]}
    return qc_pool.Area(rule=rule, box=(x0, y0, x1, y1), inner=(x0 + 25, y0 + 25, x1 - 25, y1 - 25),
                        spec=spec if rule == "square" else {**spec, "bbox": [x0, y0, x1, y1]})


def pool(**fields) -> qc_pool.Pool:
    """A cell's pool as engine/qc_pool.py fixes it (cell 7, a-1, of the run "p" at its last frame), `fields` changed."""
    values = dict(v=qc_pool.POOL_V, run="p", seq=451, iter=450, model="f" * 32, constants=qc_rule.CONSTANTS_ID,
                  start="paired", area=area(), k=160, seed=7, seed_name="a-1", cells=(7, 9, 12), members=2,
                  distance_um=(0.0, 11.2, 14.0), names={7: "a-1", 9: "b-1", 12: "c-1", 3: "d-1"}, items=(),
                  suspects=187, available=None, record_duplicates=0,
                  this_cell={"likely": 12, "suspects": 9, "pattern": "own",
                             "line": {"likely": 12, "own": 9, "none": 3, "neighbour": 0, "other": 0}},
                  view_suspects=({"x": 350.25, "y": 350.5, "gene": "G1", "p": 0.912},), view_sources=(3,),
                  row=("paired", "square"))
    values.update(fields)
    return qc_pool.Pool(**values)


def matched(of: qc_pool.Pool | None = None, **fields) -> qc_pool.Matched:
    values = dict(pool=of or pool(), items=ITEMS, order=ORDER, counts=COUNTS, input="", seconds=0.1)
    values.update(fields)
    return qc_pool.Matched(**values)


def made(meta: dict, cell: int) -> tuple[qc_pool.Pool, qc_pool.Matched]:
    """FakePools' make: the cell's pool in that run (its start as qc_rules says, its row, its area by
    engine/qc_pool.py's rule around the place (450, 450)), and its matching."""
    row = qc_pool.row_of(meta)
    of = pool(run=meta["id"], seed=cell, start=R.start_of(meta), area=qc_pool.area_of(meta["region"], (450.0, 450.0)),
              row=row, k=qc_rule.constants_for(*row)["k"])
    return of, matched(of)


def write_input(path: Path, x=None, y=None, ids=None) -> pa.Table:
    """A run's input/transcripts.parquet in the schema the app's helper writes (zarr_bridge/helper.py prepare):
    `ids` (None among them for a missing one) as given, else 0..n-1."""
    rng = np.random.default_rng(0)
    x = np.asarray(rng.uniform(100, 800, 400) if x is None else x, np.float32)
    y = np.asarray(rng.uniform(100, 800, x.size) if y is None else y, np.float32)
    n = x.size
    table = pa.table({
        "transcript_id": pa.array(np.arange(n, dtype=np.uint64) if ids is None else ids, pa.uint64()),
        "cell_id": pa.array([f"c{i % 9}-1" for i in range(n)], pa.large_string()),
        "overlaps_nucleus": pa.array((np.arange(n) % 2).astype(np.uint8)),
        "feature_name": pa.array([f"G{i % 2 + 1}" for i in range(n)], pa.large_string()),
        "x_location": pa.array(x), "y_location": pa.array(y), "z_location": pa.array(np.full(n, 5.0, np.float32)),
        "qv": pa.array(np.full(n, 40.0, np.float32)), "fov_name": pa.array(["0"] * n, pa.large_string()),
        "nucleus_distance": pa.array(np.zeros(n, np.float32)), "codeword_index": pa.array(np.zeros(n, np.int32)),
        "codeword_category": pa.array(["predesigned_gene"] * n, pa.large_string()),
        "is_gene": pa.array(np.ones(n, bool)),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return table


class Clock:
    """The test's clock: `t`, moved on by hand (and by `tick` at each reading: a wait that takes time)."""

    def __init__(self, t: float = 1000.0, tick: float = 0.0):
        self.t, self.tick = t, tick

    def __call__(self) -> float:
        now = self.t
        self.t += self.tick
        return now


class FakeRun:
    """A run as a test looks at it: meta, info and frames (under its lock), its folder; `plan`: the (state, newest
    iteration or None[, meta]) it goes through, one per step of its world; `stubborn`: an abort leaves it running.
    `start`: the (state, info or None[, meta]) a test's run made paused goes through while it waits, one per step, before
    its plan (proseg loading its input, then paused with its info.json); its plan only once it is resumed."""

    def __init__(self, folder: Path, meta: dict, info: dict | None = None, plan=(), start=()):
        self.dir, self.meta, self.info = Path(folder), meta, info
        self.frames, self.lock, self.plan, self.start = [], threading.RLock(), list(plan), list(start)
        self.stubborn, self.touched, self.room = False, 0, None

    @property
    def live(self) -> Path:
        return self.dir / "live"

    def touch(self):
        self.touched += 1

    def step(self):
        if self.meta.get("state") not in ACTIVE:
            return
        if self.meta.get("qc_test") and self.meta.get("paused"):       # made paused: nothing samples before Resume
            if self.start:
                state, info, *more = self.start.pop(0)
                with self.lock:
                    self.meta.update(state=state, **(more[0] if more else {}))
                    if info is not None:
                        self.info = dict(info)
            return
        if self.plan:
            state, newest, *more = self.plan.pop(0)
            with self.lock:
                self.meta.update(state=state, **(more[0] if more else {}))
                if newest is not None:
                    self.frames.append({"seq": len(self.frames), "iter": newest})


class FakeZarr:
    """The SpatialData samples' side: whether a sample is open (`known`), and its opening again, noted in `events`."""

    def __init__(self, events: list):
        self.events, self.is_known = events, True

    def known(self, sample: str) -> bool:
        return self.is_known

    def open(self, path, recent=True):
        self.events.append(("open", path, recent))
        self.is_known = True


class FakeRuns:
    """The engine's Runs as a test sees them: the parent "p", paused at the end of its schedule with its input, and the
    runs a test makes (`create`, called as Runs.create is: on the test's _WithRoom, whose max_active each made run keeps
    as `room`), each going through `start(role)` while it waits paused (by default: paused at once, with `child` as its
    info.json) and, once resumed, `plan(role, options)`, one step at a time (`step`). An abort stops a run at once
    (`slow`: that many seconds later; never a `stubborn` one). Every create, set (with its values), resume, abort and
    delete is in `events`; a set refused as Runs.control refuses it (proseg_options' bounds) is a RunError."""
    max_active = 2
    max_region = 1200.0                 # [proseg] max_region_um, which the run's own reasons say

    def __init__(self, root: Path, meta: dict | None = None, plan=None):
        self.root = Path(root) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.parent = FakeRun(self.root / "p", json.loads(json.dumps({**PARENT, **(meta or {})})),
                              info=json.loads(json.dumps(INFO)))
        self.runs: dict[str, FakeRun] = {"p": self.parent}
        self.made: list[FakeRun] = []
        self.events: list[tuple] = []
        self.plan = plan or (lambda role, options: list(DONE))
        self.child = {key: INFO[key] for key in ("cells", "genes", "transcripts", "schedule", "counts")}
        self.start = lambda role: [("paused", self.child)]
        self.slow = 0.0
        self.zarr = FakeZarr(self.events)

    def get(self, run_id) -> FakeRun:
        run = self.runs.get(run_id)
        if run is None:
            raise RunError("Unknown run", 404)
        return run

    def add(self, run_id: str, **meta) -> FakeRun:
        run = self.runs[run_id] = FakeRun(self.root / run_id, {"id": run_id, "kind": "proseg", **meta})
        return run

    def active(self) -> list[FakeRun]:
        return [run for run in self.runs.values() if run.meta.get("state") in ACTIVE]

    def held(self, folders) -> set:
        return set()

    def frame_path(self, run_id, seq):
        raise RunError("Unknown frame", 404)

    def create(self, sample_id, box=None, options=None, region=None, prior=None, kind="proseg", stain=None,
               extra=None):
        return self.register(self.max_active, sample_id, options, region, prior, extra)   # self: the _WithRoom

    def register(self, room, sample_id, options, region, prior, extra) -> dict:
        run_id = f"t{len(self.made) + 1}"
        role = extra["qc_test"]["role"]
        meta = {"id": run_id, "kind": "proseg", "sample": sample_id, "state": "preparing", "options": dict(options),
                "region": region, "prior": prior, **json.loads(json.dumps(extra))}
        run = FakeRun(self.root / run_id, meta, plan=self.plan(role, options), start=self.start(role))
        run.room = room
        run.dir.mkdir(parents=True)
        self.runs[run_id] = run
        self.made.append(run)
        self.events.append(("create", role, run_id))
        return {"id": run_id, **meta}

    def control(self, run_id, action, speed=None, values=None):
        run = self.get(run_id)
        if action == "set":                              # as Runs.control: the live values checked, then kept
            from sopa_companion.engine import proseg_options
            if run.meta.get("state") not in ACTIVE:
                raise RunError("This run is not active", 409)
            merged = {**run.meta.get("live_values", {}), **(values or {})}
            try:
                proseg_options.live_lines(merged)
            except ValueError as exc:
                raise RunError(str(exc)) from None
            run.meta["live_values"] = merged
            self.events.append(("set", run_id, dict(values or {})))
            return
        self.events.append((action, run_id, time.monotonic()) if action == "abort" else (action, run_id))
        if action == "abort" and not run.stubborn:
            stop = lambda: run.meta.update(state="stopped", message="Stopped after 3 snapshots, without results")
            threading.Timer(self.slow, stop).start() if self.slow else stop()
        elif action in ("pause", "resume"):              # as Runs.control: a run still preparing stays preparing
            if run.meta.get("state") not in ("preparing", "running", "paused"):
                raise RunError("This run is not sampling", 409)
            run.meta["paused"] = action == "pause"
            if run.meta["state"] != "preparing":
                run.meta["state"] = "paused" if action == "pause" else "running"

    def delete(self, run_id, later=False):
        run = self.get(run_id)
        if run.meta.get("state") in ACTIVE:
            raise RunError("Stop the run before deleting it", 409)
        self.events.append(("delete", run_id, later))
        del self.runs[run_id]

    def step(self):
        for run in list(self.runs.values()):
            if run is not self.parent:
                run.step()


class Steps:
    """A test's cancel event in the tests of its thread: each wait of FOLLOW_S moves its world one step on (`before`
    called first, with how many waits there were) and the clock by that much; `set` cancels, as DELETE does. A wait
    past `most` fails (AssertionError): a loop that would wait for ever fails at once instead of hanging the suite."""

    def __init__(self, runs: FakeRuns, clock: Clock, before=None, most: int = 1000):
        self.runs, self.clock, self.before, self.cancelled, self.waits = runs, clock, before, False, 0
        self.most = most

    def is_set(self) -> bool:
        return self.cancelled

    def set(self):
        self.cancelled = True

    def wait(self, timeout=None) -> bool:
        self.waits += 1
        if self.waits > self.most:
            raise AssertionError(f"still waiting after {self.most} waits")
        if self.before is not None:
            self.before(self.waits)
        self.clock.t += timeout or 0.0
        self.runs.step()
        return self.cancelled


class Looks:
    """A waiting test's cancel in its loop on the engine's own pools (real time: the run's worker works meanwhile):
    each wait noted (its timeout, in `timeouts`), `before` called with how many there were, then a nap of `nap` s;
    never set. A wait past `most` fails (AssertionError), as Steps'."""

    def __init__(self, before=None, nap: float = 0.0, most: int = 200):
        self.before, self.nap, self.most, self.timeouts = before, nap, most, []

    def is_set(self) -> bool:
        return False

    def wait(self, timeout=None) -> bool:
        self.timeouts.append(timeout)
        if len(self.timeouts) > self.most:
            raise AssertionError(f"still waiting after {self.most} waits")
        if self.before is not None:
            self.before(len(self.timeouts))
        time.sleep(self.nap)
        return False


class Where:
    """Go deeper's where engine as the corner's crop asks it: the cell's footprint in the parent's frame (µm)."""

    def __init__(self, box=(345.0, 345.0, 357.0, 356.0)):
        self.box, self.asked = box, []

    def footprints(self, run_id, seq, cells):
        self.asked.append((run_id, seq, list(cells)))
        return {cell: list(self.box) for cell in cells}


class Analysis:
    """The scheduler's Clusters & UMAP switch, which a test turns off for each of its runs (`fails`: refused, as it is
    wherever Clusters & UMAP are unavailable)."""

    def __init__(self, fails: bool = False):
        self.calls, self.fails = [], fails

    def configure(self, run_id, enabled=None, seq=None, params=None):
        self.calls.append((run_id, enabled))
        if self.fails:
            raise QcError("Clusters & UMAP are unavailable here", 409)


class FakeQc:
    """The mixing check as a waiting test asks it (QcManager): its run's switch, its sample's state and Go deeper's
    details' (describe), its switch (configure) and its learning asked again (models.ensure), each call in `calls`. A
    retry of a failed sample, or of failed details, sets them going again. `waits` (a list, given to FakePools as its
    `extra`) always holds the waits the engine would find in it (`set` changes it as the check moves on)."""

    def __init__(self, enabled: bool = True, sample_state: str = "ready", detail: str = "ready"):
        self.enabled, self.sample_state, self.detail, self.calls = enabled, sample_state, detail, []
        self.models = SimpleNamespace(ensure=self.ensure)
        self.waits: list = []
        self.set()

    def set(self, **values):
        for key, value in values.items():
            setattr(self, key, value)
        self.waits[:] = ([qc_pool.reason("check_off")] if not self.enabled else
                         [qc_pool.reason("check_wait")] if self.sample_state != "ready" else
                         [qc_pool.reason("details")] if self.detail != "ready" else [])

    def describe(self, run_id):
        state = "off" if not self.enabled else self.sample_state
        return {"enabled": self.enabled, "state": state, "sample_state": self.sample_state,
                "detail": {"state": self.detail}}

    def configure(self, run_id, enabled=None, seq=None):
        self.calls.append(("configure", run_id, enabled))
        if enabled is not None:
            self.set(enabled=enabled)

    def ensure(self, sample, retry=False, detail=False):
        self.calls.append(("ensure", sample, retry, detail))
        if retry and self.sample_state == "failed":
            self.set(sample_state="waiting")
        if detail and self.detail in ("failed", "none"):
            self.set(detail="queued")


def world(tmp_path, meta=None, plan=None, clock=None, analysis=None, qc=None,
          **pools) -> tuple[FakeRuns, T.QcTests, Clock]:
    """The fake world: the parent with its input (400 transcripts over its region) and its cells.txt (c0-1, c1-1, ...),
    its tests' engine on a clock of the test's own, the pools from FakePools (`pools`: its options), the mixing check
    `qc` (none: no step of its own is taken)."""
    runs = FakeRuns(tmp_path, meta, plan)
    write_input(runs.parent.dir / "input" / "transcripts.parquet")
    runs.parent.live.mkdir(parents=True, exist_ok=True)
    (runs.parent.live / "cells.txt").write_text("".join(f"c{k}-1\n" for k in range(INFO["cells"])))
    clock = clock or Clock()
    tests = T.QcTests(runs, qc, analysis or Analysis(), clock=clock, where=Where(),
                      pools=FakePools(runs, pools.pop("make", made), **pools))
    return runs, tests, clock


def begun(runs: FakeRuns, tests: T.QcTests, clock: Clock, *, flag="cell-compactness", value=0.02, before=None,
          **fields) -> T.Test:
    """A test of the parent's cell 7, as start() makes it, in place, its waits stepping the world (not started)."""
    test = tests._new(runs.parent, tests.pools.pool("p", 7), flag, value, tests._current(runs.parent, flag))
    for key, found in fields.items():
        setattr(test, key, found)
    test.cancel = Steps(runs, clock, before)
    tests.tests["p"] = test
    return test


def following(tmp_path, plan=None, clock=None, before=None) -> tuple[FakeRuns, T.QcTests, T.Test]:
    """A test of the fake world with its input and its runs made (`_make`), given their hold and resumed (`_hold_runs`:
    its waits on a clock of their own and its pushes left out, so the test's clock, its pushes and `before` see the
    following alone), to follow."""
    runs, tests, clock = world(tmp_path, plan=plan, clock=clock)
    test = begun(runs, tests, clock, before=before)
    tests._input(test)
    test.matched = matched(test.pool)
    tests._make(test)
    steps, test.cancel, tests._push = test.cancel, Steps(runs, Clock()), lambda test, force=False: None
    tests._hold_runs(test)
    test.cancel = steps
    del tests._push                                       # QcTests' own again
    return runs, tests, test


def refused(tests: T.QcTests, run_id: str, body, status: int, words: str):
    """A POST start() refuses with `status` and these words; the refusal."""
    with pytest.raises((QcError, RunError)) as caught:
        tests.start(run_id, body)
    assert (caught.value.status, str(caught.value)) == (status, words), (body, str(caught.value))
    return caught.value


def ready(tmp_path, meta=None, **pools) -> tuple[FakeRuns, T.QcTests, threading.Event]:
    """POSTs on the fake world: a test, once started, waits in its thread for `go` (on its way meanwhile)."""
    runs, tests, _ = world(tmp_path, meta, **pools)
    go = threading.Event()
    tests._drive = lambda test: go.wait(10)
    return runs, tests, go


def pressed(tmp_path, meta=None, body=None, before=None, **options) -> tuple[FakeRuns, T.QcTests, T.Test, Clock]:
    """A POST on the fake world as the API makes it (start(): its checks and its own steps), its thread held back: the
    test, its cancel the world's steps (`before`: called first at each), to drive here (`tests._drive`)."""
    runs, tests, clock = world(tmp_path, meta, **options)
    tests._drive = lambda test: None
    tests.start("p", body or BODY)
    del tests._drive                                     # the real one, from here on
    test = tests.tests["p"]
    test.thread.join(5)
    test.cancel = Steps(runs, clock, before)
    return runs, tests, test, clock


BODY = {"cell": 7, "flag": "cell-compactness", "value": 0.02}


# ---- the shared input --------------------------------------------------------------------------------------------------
def test_a_region_re_run_as_it_is_reads_the_parents_own_input(tmp_path):
    parent = tmp_path / "p" / "input" / "transcripts.parquet"
    table = write_input(parent)
    folder = tmp_path / "qc-tests" / "abcd1234"
    found = T.shared_input(parent, area("as_is", (100, 100, 550, 550)), folder)
    assert {key: value for key, value in found.items() if key != "seconds"} == \
        {"path": str(parent), "owned": False, "transcripts": table.num_rows, "ids": "own", "bytes": 0}
    assert 0 <= found["seconds"] < 5 and not folder.exists()        # nothing written
    with pytest.raises(ValueError) as missing:
        T.shared_input(tmp_path / "gone" / "input" / "transcripts.parquet", area("as_is"), folder)
    assert str(missing.value) == "your run's input is missing" and not folder.exists()
    with pytest.raises(ValueError, match="^your run's input is missing$"):
        T.shared_input(tmp_path / "gone" / "input" / "transcripts.parquet", area("square"), folder)


@pytest.mark.parametrize("ids", [[0, 1, 1, 2, 3, 4], [5, None, 6, 7, 8, 9]], ids=["repeated", "missing"])
def test_ids_that_repeat_or_are_missing_are_renumbered_in_a_copy_of_the_tests_own(tmp_path, ids):
    """The ids join each run's per-transcript file to the targets: two transcripts may not share one, nor one have
    none. Renumbered 0..n-1 (uint64, as proseg reads them) in the test's own copy; the parent's file is left as it is."""
    parent = tmp_path / "p" / "input" / "transcripts.parquet"
    table = write_input(parent, x=[110, 200, 300, 400, 500, 540], y=[120, 220, 320, 420, 520, 545], ids=ids)
    folder = tmp_path / "qc-tests" / "abcd1234"
    found = T.shared_input(parent, area("as_is", (100, 100, 550, 550)), folder)
    path = folder / "transcripts.parquet"
    assert (found["path"], found["owned"], found["ids"], found["transcripts"]) == (str(path), True, "rows", 6)
    assert found["bytes"] == path.stat().st_size > 0
    copy = pq.read_table(path)
    assert copy.column("transcript_id").to_pylist() == [0, 1, 2, 3, 4, 5]
    assert copy.schema.equals(table.schema) and copy.drop_columns(["transcript_id"]).equals(
        table.drop_columns(["transcript_id"]))
    assert pq.read_table(parent).equals(table)
    # A square of such a parent: its rows, renumbered too.
    square = T.shared_input(parent, area("square", (100, 100, 400, 400)), tmp_path / "qc-tests" / "efgh5678")
    assert (square["ids"], square["transcripts"]) == ("rows", 3)
    assert pq.read_table(square["path"]).column("transcript_id").to_pylist() == [0, 1, 2]


def test_a_square_takes_the_parents_rows_inside_it_half_open(tmp_path):
    """x0 <= x < x1 and y0 <= y < y1, as the helper crops a region (zarr_bridge/helper.py): a row on the square's left
    or lower edge is in, one on its right or upper edge is out. Unique ids are kept as the parent's."""
    parent = tmp_path / "p" / "input" / "transcripts.parquet"
    x = [100.0, 600.0, 350.0, 350.0, 99.99, 350.0, 599.99]
    y = [350.0, 350.0, 100.0, 600.0, 350.0, 599.99, 100.0]
    write_input(parent, x=x, y=y, ids=[11, 12, 13, 14, 15, 16, 17])
    folder = tmp_path / "qc-tests" / "abcd1234"
    found = T.shared_input(parent, area("square", (100, 100, 600, 600)), folder)
    copy = pq.read_table(folder / "transcripts.parquet")
    assert copy.column("transcript_id").to_pylist() == [11, 13, 16, 17]
    assert (found["owned"], found["ids"], found["transcripts"]) == (True, "own", 4)
    assert copy.column("x_location").to_pylist() == [np.float32(v) for v in (100.0, 350.0, 350.0, 599.99)]


def test_the_tests_own_input_keeps_every_column_and_type_and_appears_whole(tmp_path, monkeypatch):
    """Every column with its type as proseg reads the Xenium schema (uint8, large_string, float32 ...), zstd, under a
    name ending .parquet (proseg tells the format by it); written aside, then moved in place, so a run never finds half
    of it, and a write that fails leaves nothing. A square without a transcript is refused."""
    parent = tmp_path / "p" / "input" / "transcripts.parquet"
    table = write_input(parent)
    folder = tmp_path / "qc-tests" / "abcd1234"
    path = folder / "transcripts.parquet"
    written, real = [], pq.write_table

    def write(data, where, **kwargs):
        written.append((Path(where).name, path.exists(), kwargs))
        return real(data, where, **kwargs)
    monkeypatch.setattr(pq, "write_table", write)
    found = T.shared_input(parent, area("square", (200, 200, 700, 700)), folder)
    copy = pq.read_table(found["path"])
    assert found["path"] == str(path) and path.name.endswith(".parquet") and copy.schema.equals(table.schema)
    assert [str(copy.schema.field(name).type) for name in ("overlaps_nucleus", "cell_id", "feature_name", "x_location",
                                                           "transcript_id", "is_gene")] == \
        ["uint8", "large_string", "large_string", "float", "uint64", "bool"]
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
    assert [(name != path.name, there, kwargs) for name, there, kwargs in written] == \
        [(True, False, {"compression": "zstd"})]
    assert [entry.name for entry in folder.iterdir()] == ["transcripts.parquet"]
    x, y = table["x_location"], table["y_location"]
    inside = table.filter(pc.and_(pc.and_(pc.greater_equal(x, 200.0), pc.less(x, 700.0)),
                                  pc.and_(pc.greater_equal(y, 200.0), pc.less(y, 700.0))))
    assert copy.equals(inside) and found["transcripts"] == inside.num_rows > 0 and found["bytes"] == path.stat().st_size

    def broken(data, where, **kwargs):
        Path(where).write_bytes(b"PAR1 and no more")
        raise OSError("No space left on device")
    monkeypatch.setattr(pq, "write_table", broken)
    other = tmp_path / "qc-tests" / "efgh5678"
    with pytest.raises(OSError, match="No space left"):
        T.shared_input(parent, area("square", (200, 200, 700, 700)), other)
    assert list(other.iterdir()) == []
    with pytest.raises(ValueError, match="no transcripts"):
        T.shared_input(parent, area("square", (2000, 2000, 2500, 2500)), tmp_path / "qc-tests" / "far")
    assert not (tmp_path / "qc-tests" / "far" / "transcripts.parquet").exists()


# ---- the small rules ---------------------------------------------------------------------------------------------------
def test_the_roles_are_a_side_as_now_then_a_side_with_the_setting():
    assert T.roles_for(3) == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert T.roles_for(4) == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    # Runs per side as measured (engine/qc_rule.py): 3 + 3 from nuclei and cells, 4 + 4 from cells only.
    assert [len(T.roles_for(qc_rule.CONSTANTS[start]["runs_per_side"])) for start in ("paired", "cells")] == [6, 8]


def test_a_parent_without_factorisation_has_its_test_runs_made_without_it():
    assert not T.no_factorization_of({}, None)
    assert not T.no_factorization_of({"options": {"no-factorization": False}}, {"factorization": True})
    assert T.no_factorization_of({"retried_without_factorization": True}, {"factorization": True})   # runs.py's retry
    assert T.no_factorization_of({"options": {"no-factorization": True}}, None)                      # asked for
    assert T.no_factorization_of({"options": {}}, {"factorization": False})                          # as proseg says


def test_a_runs_schedule_ends_at_burn_in_plus_samples_plus_hill_climb():
    assert T.schedule_total(None, {}) == T.schedule_total({}, {}) == 450
    assert T.schedule_total({"schedule": SCHEDULE}, {"burnin-samples": 7}) == 450      # what proseg reported first
    assert T.schedule_total(None, {"burnin-samples": 100, "samples": 60, "hillclimb": 0}) == 160


def test_the_pool_routes_query_is_read_by_the_engine_in_its_own_words():
    assert T.parse_preview("3", "2.5") == (3, 2.5) and T.parse_preview("3", None) == (3, 0.0)
    assert T.parse_preview(" 0 ", "10") == (0, 10.0) and T.parse_preview("12", "0") == (12, 0.0)
    for cell in (None, "", "3.0", "-1", "x", "1e3", "+3", "1234567890"):
        with pytest.raises(QcError) as refusal:
            T.parse_preview(cell, None)
        assert (refusal.value.status, str(refusal.value)) == (422, "cell must be a cell number (label - 1)"), cell
    for wait in ("11", "-1", "nan", "x", "inf", "", "10.001"):
        with pytest.raises(QcError) as refusal:
            T.parse_preview("3", wait)
        assert (refusal.value.status, str(refusal.value)) == (422, "wait must be 0 to 10 seconds"), wait


def test_a_parent_at_proseg_defaults_is_told_from_one_with_other_settings():
    from sopa_companion.engine import proseg_options
    defaults = {o["flag"]: o["default"] for o in proseg_options.OPTIONS if o["default"] is not None}
    assert T.parent_defaults({}) and T.parent_defaults(T._asked({"options": defaults}))
    # A value as a frame carries it (float32) is the default it shows.
    assert T.parent_defaults({"cell-compactness": 0.03999999910593033, "prior-weight": 1.0})
    assert not T.parent_defaults(T._asked({"options": {"cell-compactness": 0.08}}))
    assert not T.parent_defaults(T._asked({"options": {}, "live_values": {"prior-weight": 2}}))      # a hold set live
    assert not T.parent_defaults({"not-a-setting": 1})
    # What a test run takes of its parent: its options with the live values since, without the threads, the
    # temperature, the retry's flag, the outputs and unset values.
    asked = T._asked({"options": {"nthreads": 32, "temperature": 0.5, "no-factorization": True, "voxel-layers": None,
                                  "output-counts-fmt": "csv", "exclude-spatialdata-transcripts": True,
                                  "cell-compactness": 0.05, "min-qv": 25.0},
                      "live_values": {"prior-weight": 2.0, "cell-compactness": 0.06}})
    assert asked == {"cell-compactness": 0.06, "min-qv": 25.0, "prior-weight": 2.0}
    # A Temperature set (live since, else as an option), which the test runs leave out: not proseg's defaults.
    for meta, temperature in (({"options": {"temperature": 0.5}}, 0.5),
                              ({"options": {"temperature": 0.5}, "live_values": {"temperature": 0.1}}, 0.1),
                              ({"options": {}, "live_values": {"temperature": None}}, None), ({}, None)):
        assert T.temperature_of(meta) == temperature, meta
        assert T.parent_defaults(T._asked(meta), T.temperature_of(meta)) is (temperature is None), meta


def test_how_a_run_started():
    """The test's info says how its run's cells started: from cells only every transcript inside one holds its spot to
    it, so the hold is not neutral there."""
    zarr, cells = {"sample_kind": "zarr"}, {"kind": "shapes", "key": "cellpose_boundaries"}
    nuclei = {"kind": "shapes", "key": "nucleus_boundaries"}
    assert T.start_of({"sample_kind": "xenium", "prior": None}) == T.start_of({}) == "paired"     # a Xenium bundle
    assert T.start_of({**zarr, "prior": {"cells": None, "nuclei": nuclei}}) == "nuclei"
    assert T.start_of({**zarr, "prior": {"cells": cells, "nuclei": None}}) == "cells"
    assert T.start_of({**zarr, "prior": {"cells": cells, "nuclei": nuclei}}) == "paired"
    # As the input was prepared (zarr_bridge/helper.py): nuclei that flag nothing inside the cells seed from the cells.
    assert T.start_of({**zarr, "prior": {"cells": cells, "nuclei": nuclei}, "prepared": {"seeded_from": "cells"}}) == "cells"
    assert T.start_of({**zarr, "prior": {"cells": cells, "nuclei": nuclei}, "prepared": {"seeded_from": "nuclei"}}) == \
        "paired"
    assert T.start_of is R.start_of                    # one rule: the mixing check's R5 reads it too


def test_the_hold_is_the_weight_proseg_applies():
    """A test says how firmly its run holds the starting cells, its prior-weight and that value as proseg applies it
    (times the transcripts per cell over 100, above 100 a cell, in float32: main.rs:955-956), with that scale: the
    weight the run's log prints while the value is the one it started with. As in the run started from cells only of
    section D: 1,387 transcripts a cell, 0.5 printed as 6.94, then 1 set live (13.87); about 0.072 brings it to 1."""
    run = {"transcripts": 1_387_000, "cells": 1000, "prior_weight": float(np.float32(0.5) * np.float32(13.87))}
    assert T.hold_of(run, 0.5) == {"value": 0.5, "effective": 6.935, "scale": 13.87}
    assert T.hold_of(run, 0.5)["effective"] == pytest.approx(run["prior_weight"], rel=1e-6)   # info.json's own
    assert T.hold_of(run, 1.0) == {"value": 1.0, "effective": 13.87, "scale": 13.87}          # set live since
    hold = T.hold_of(run, 0.5)
    aimed = float(f"{hold['value'] / hold['effective']:.2g}")                                 # the page's aim
    assert aimed == 0.072 and T.hold_of(run, aimed)["effective"] == pytest.approx(1.0, abs=0.01)
    assert T.hold_of({"transcripts": 3, "cells": 2}, 2) == {"value": 2, "effective": 2.0, "scale": 1.0}   # as set
    assert T.hold_of({"transcripts": 300, "cells": 0}, 0.5) == {"value": 0.5, "effective": 1.5, "scale": 3.0}  # 1 cell
    assert T.hold_of(run, 0) == {"value": 0, "effective": 0.0, "scale": 13.87}
    for info, value in ((None, 1.0), ({}, 1.0), ({"transcripts": 3}, 1.0), ({"transcripts": 3, "cells": 2}, None),
                        ({"transcripts": 3, "cells": 2}, True), ({"transcripts": "3", "cells": 2}, 1.0)):
        assert T.hold_of(info, value) is None, (info, value)


def test_the_crop_is_the_cells_box_grown_made_square_and_kept_inside_the_area():
    square = [0.0, 0.0, 500.0, 500.0]
    assert T.crop_around([100, 100, 110, 120], None, square) == [81.0, 86.0, 129.0, 134.0]    # 20 + 24: 48 at least
    assert T.crop_around([100, 100, 200, 160], None, square) == [88.0, 68.0, 212.0, 192.0]    # 100 + 24
    assert T.crop_around([100, 100, 300, 150], None, square) == [130.0, 55.0, 270.0, 195.0]   # 140 at most
    assert T.crop_around([2, 490, 12, 498], None, square) == [0.0, 452.0, 48.0, 500.0]        # moved inside
    assert T.crop_around(None, (250.0, 250.0), square) == [226.0, 226.0, 274.0, 274.0]        # no footprint: its place
    assert T.crop_around(None, (10.0, 200.0), [0, 0, 30, 400]) == [0.0, 185.0, 30.0, 215.0]   # a narrow area
    assert T.crop_around([345, 345, 357, 356], None, area().box) == [327.0, 326.5, 375.0, 374.5]


def test_the_crop_is_around_the_cells_suspects_when_its_outline_cannot_be_read(tmp_path):
    """The pool gives the cell's suspects, never a place of its own: when the where engine cannot give the cell's
    outline in the parent's frame, the corner's crop goes around the middle of the cell's suspects (inside the area);
    without a suspect, there is no crop (the views show none)."""
    class Blind:
        def footprints(self, run_id, seq, cells):
            raise RunError("The frame's file is incomplete", 500)

    runs, tests, clock = world(tmp_path)
    tests.where = Blind()
    test = begun(runs, tests, clock)
    assert test.place == (350.25, 350.5)                       # the pool's one suspect of the cell
    tests._crop(test)
    assert test.crop == T.crop_around(None, (350.25, 350.5), test.area.box) == [326.25, 326.5, 374.25, 374.5]
    two = T.Test("p", pool(view_suspects=({"x": 300.0, "y": 410.0, "gene": "G1", "p": 0.9},
                                          {"x": 310.0, "y": 420.0, "gene": "G1", "p": 0.8},
                                          {"x": True, "y": 1.0, "gene": "G1", "p": 0.7})), "cell-compactness", 0.02,
                 0.04, clock, runs_per_side=3)
    assert two.place == (305.0, 415.0)                         # a suspect without a number for x is left out
    none = T.Test("p", pool(view_suspects=()), "cell-compactness", 0.02, 0.04, clock, runs_per_side=3)
    assert none.place is None
    tests._crop(none)
    assert none.crop is None


# ---- the POST ----------------------------------------------------------------------------------------------------------
def test_a_posts_body_is_checked_first_in_the_engines_words(tmp_path):
    runs, tests, go = ready(tmp_path)
    drift = {**BODY, "flag": "diffusion-probability"}
    for body, words in (
            ("x", "A test is {cell, flag, value}"), ([BODY], "A test is {cell, flag, value}"),
            ({**BODY, "cell": -1}, "cell must be a cell number (label - 1)"),
            ({**BODY, "cell": True}, "cell must be a cell number (label - 1)"),
            ({**BODY, "cell": "7"}, "cell must be a cell number (label - 1)"),
            ({**BODY, "cell": 7.0}, "cell must be a cell number (label - 1)"),
            ({"flag": "cell-compactness", "value": 0.02}, "cell must be a cell number (label - 1)"),
            ({**BODY, "changes": [{"flag": "prior-weight", "value": 0.5}]},
             "A test changes one setting: send flag and value"),
            ({**BODY, "flag": "temperature"}, "A test changes one of the settings the test form offers"),
            ({**BODY, "flag": None}, "A test changes one of the settings the test form offers"),
            ({**BODY, "value": "0.02"}, "value must be a number"), ({**BODY, "value": True}, "value must be a number"),
            ({**BODY, "value": None}, "value must be a number"),
            ({**BODY, "value": float("nan")}, "value must be a number"),
            ({**BODY, "value": float("inf")}, "value must be a number"),
            ({**BODY, "value": 50}, "Choose a value from 0.001 to 10"),
            ({**BODY, "value": 0}, "Choose a value from 0.001 to 10"),
            ({**BODY, "flag": "prior-weight", "value": 1001}, "Choose a value from 0 to 1000"),
            ({**BODY, "flag": "prior-weight", "value": -0.5}, "Choose a value from 0 to 1000"),
            ({**drift, "value": 0.05}, "Choose a value from 0.1 to 0.3"),
            ({**drift, "value": 0.35}, "Choose a value from 0.1 to 0.3"),
            ({**drift, "value": 0.5}, "Choose a value from 0.1 to 0.3")):
        refused(tests, "nope", body, 422, words)              # the body first: the run is not even looked up
    assert tests.pools.calls["pool"] == 0 and runs.made == [] and tests.tests == {}
    # Far drift at either end of its range, and in it; suspects and sources sent are ignored (the pool's are the
    # engine's).
    for value in (0.1, 0.3, 0.2):
        assert T.QcTests._body({**drift, "value": value, "suspects": "x", "sources": [1]}) == \
            (7, "diffusion-probability", value)
    assert T.QcTests._body({**BODY, "flag": "prior-weight", "value": 0}) == (7, "prior-weight", 0)


def test_a_post_is_refused_in_order_and_nothing_is_made(tmp_path, monkeypatch):
    """The POST's refusals in their order: closing (503), an unknown run (404), a test of this run waiting or on its way
    (409), then the pool's refusals (409 {detail, status, reasons}: only what stops a test, never a wait or a caveat;
    an unknown cell 404), then the run's own value and far drift with drift off (422). None of them makes a run,
    keeps a test or changes the parent."""
    runs, tests, go = ready(tmp_path)
    tests.closed = True
    refused(tests, "p", BODY, 503, "The app is closing")
    tests.closed = False
    refused(tests, "nope", BODY, 404, "Unknown run")
    # Ended before its end: no end frame will ever come (refused before the pool is worked out).
    runs.parent.meta.update(state="stopped", schedule_done=None)
    found = refused(tests, "p", BODY, 409, "This run ended before its last iteration")
    assert found.extra == {"reasons": [{"code": "ended", "words": qc_pool.REASON_WORDS["ended"]}]}
    # Being saved, with a wait and caveats beside it: the refusal alone.
    runs.parent.meta.update(state="finishing",
                            region={"kind": "box", "box": [0, 0, 250, 250], "bbox": [0, 0, 250, 250]})
    found = refused(tests, "p", BODY, 409, "This run is being saved")
    assert [r["code"] for r in found.extra["reasons"]] == ["ending"]
    runs.parent.meta.update(state="paused", schedule_done=dict(PARENT["schedule_done"]), region=dict(PARENT["region"]))
    # A refusal of the cell's own, past the run's (here make's: the cell does not look mixed).
    def no_suspects(meta, cell):
        raise qc_pool.OutsideDomain([qc_pool.reason("no_suspects")])
    tests.pools.make = no_suspects
    found = refused(tests, "p", BODY, 409, "No suspects to test in this cell: none glows, or none could be placed")
    assert found.extra == {"reasons": [qc_pool.reason("no_suspects")]}
    tests.pools.make = lambda meta, cell: (_ for _ in ()).throw(RunError("Unknown cell", 404))
    refused(tests, "p", {**BODY, "cell": 999}, 404, "Unknown cell")
    tests.pools.make = made
    # The run's own value (here proseg's default, also as a frame's float32 shows it): 422; far drift while drift is
    # switched off in the run: 422.
    refused(tests, "p", {**BODY, "value": 0.04}, 422, "0.04 is what this run uses now: choose another value to test")
    refused(tests, "p", {**BODY, "value": 0.03999999910593033}, 422,
            "0.04 is what this run uses now: choose another value to test")
    runs.parent.meta["live_values"] = {"no-diffusion": True}
    refused(tests, "p", {**BODY, "flag": "diffusion-probability", "value": 0.25}, 422,
            "Far drift does nothing while drift is switched off in this run")
    # Nothing refused made a run, kept a test, or changed the parent.
    assert runs.made == [] and tests.tests == {} and runs.parent.meta["state"] == "paused"
    assert runs.parent.touched == 0 and not tests.going()
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == []
    # One test at a time per run: a second POST while it waits, or while it is on its way.
    runs.parent.meta.pop("live_values")
    tests.start("p", BODY)
    refused(tests, "p", BODY, 409, "A test of this run is on its way: wait for its answer, or cancel it")
    go.set()


def test_a_post_while_a_wait_holds_makes_a_waiting_test_after_the_apps_own_steps(tmp_path):
    """A press while the run has not reached its end (paused before it), the mixing check off: 202 with a "waiting"
    test, after the app's own steps (the run resumed, the check switched on: then learning); its record with its waits
    now, its row, runs per side and cell's name (cells.txt), and nothing of its pool yet (area, square, pool null; its
    runs' ids null). Pushed at once."""
    meta = {"state": "paused", "paused": True, "schedule_done": None}
    qc = FakeQc(enabled=False, sample_state="learning")
    runs, tests, go = ready(tmp_path, meta, qc=qc, extra=qc.waits)
    record = tests.start("p", BODY)
    go.set()
    assert record["state"] == "waiting" and runs.made == [] and tests.pools.calls["pool"] == 1
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p")]
    assert runs.parent.meta["state"] == "running" and qc.calls == [("configure", "p", True)]
    test = tests.tests["p"]
    assert (test.resumed, test.met) == (True, {"resume", "check_off"})
    assert [w["code"] for w in record["waiting"]] == ["not_at_end", "check_wait"]     # after the steps: as they are
    assert (record["area"], record["square"], record["pool"], record["result"], record["reasons"], record["caveats"]) \
        == (None, None, None, None, [], [])
    assert record["runs"] == dict.fromkeys(["A1", "A2", "A3", "B1", "B2", "B3"]) and record["runs_per_side"] == 3
    assert (record["cell"], record["cell_name"], record["start"], record["row"]) == \
        (7, "c7-1", "paired", {"start": "paired", "situation": "square"})
    assert record["progress"]["phase"] == "waiting" and record["seconds"] == {} and record["was_running"] is False
    assert tests.brief("p")["state"] == "waiting" and runs.parent.touched == 1 and not tests.going()
    assert tests.view("p") == {"state": "waiting", "box": None, "suspects": [], "runs": []}


def test_a_post_while_another_run_samples_waits_for_it_and_its_caveats_never_stop_it(tmp_path, monkeypatch):
    """Another run sampling (not paused): the press waits (other_run), the parent at its end left as it is. Where the
    test was not measured (a region under 300 µm), it runs at once, its caveats in its record; a cells-only start on a
    region over 500 µm (measured on proseg's own count) runs with none, judged with its own square row."""
    runs, tests, go = ready(tmp_path)
    other = runs.add("o", state="running")
    record = tests.start("p", BODY)
    assert record["state"] == "waiting" and [w["code"] for w in record["waiting"]] == ["other_run"]
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [] and runs.parent.meta["state"] == "paused"
    go.set()
    other.meta["state"] = "paused"
    box = [0.0, 0.0, 250.0, 250.0]
    runs, tests, go = ready(tmp_path / "small", {"region": {"kind": "box", "box": box, "bbox": box}, "box": box})
    record = tests.start("p", BODY)
    go.set()
    assert record["state"] == "preparing" and record["caveats"] == [qc_pool.reason("region_small", min_um=300.0)]
    assert (record["area"]["rule"], record["row"], record["waiting"]) == \
        ("as_is", {"start": "paired", "situation": "as_is"}, [])
    monkeypatch.setattr(R, "start_of", lambda meta: "cells")
    runs, tests, go = ready(tmp_path / "cells")
    record = tests.start("p", BODY)
    go.set()
    assert (record["state"], [c["code"] for c in record["caveats"]]) == ("preparing", [])
    assert (record["row"], record["runs_per_side"], record["area"]["rule"]) == \
        ({"start": "cells", "situation": "square"}, 4, "square")


def test_a_pool_not_taken_at_the_runs_end_makes_the_press_wait(tmp_path, monkeypatch):
    """The pool's frame must be the run's last (schedule_done): one taken before (a pool computed while the run still
    sampled, kept by a stand-in that keeps make()'s frame) makes the test wait, and its loop takes the pool again;
    it starts with the one at the run's end."""
    def older(meta, cell):
        of, found = made(meta, cell)
        return dataclasses.replace(of, seq=300, iter=299), found
    answered(monkeypatch)
    runs, tests, test, clock = pressed(tmp_path, make=older, seq_from_meta=False,
                                       before=lambda waits: waits == 2 and setattr(tests.pools, "make", made))
    assert test.state == "waiting" and test.waiting == [] and tests.pools.calls["pool"] == 1 and runs.made == []
    tests._drive(test)
    assert test.state == "done" and test.pool.seq == 451 and tests.pools.calls["pool"] == 4


def test_a_test_starts_with_its_record_in_the_apis_order(tmp_path):
    runs, tests, go = ready(tmp_path)
    record = tests.start("p", {**BODY, "suspects": [{"x": 1}], "sources": [3]})
    go.set()
    assert list(record) == ["id", "state", "cell", "cell_name", "flag", "name", "value", "current", "tried", "start",
                            "area", "square", "runs_per_side", "threads", "runs", "progress", "pool", "message",
                            "result", "was_running", "seconds", "hold", "no_factorization", "reruns", "estimate_s",
                            "waiting", "reasons", "caveats", "row"]
    assert len(record["id"]) == 8 and int(record["id"], 16) >= 0
    assert {key: value for key, value in record.items() if key != "id"} == {
        "state": "preparing", "cell": 7, "cell_name": "a-1", "flag": "cell-compactness", "name": "compactness",
        "value": 0.02, "current": 0.04, "tried": True, "start": "paired",
        "area": {"rule": "square", "box": [200.0, 200.0, 700.0, 700.0], "inner": [225.0, 225.0, 675.0, 675.0]},
        "square": [200.0, 200.0, 700.0, 700.0], "runs_per_side": 3, "threads": 8,
        "runs": {"A1": None, "A2": None, "A3": None, "B1": None, "B2": None, "B3": None},
        "progress": {"iter": 0, "of": 450, "runs": 6, "done": 0, "sides": {"A": 0, "B": 0}, "marks": [200, 400],
                     "phase": "preparing", "retry": False},
        "pool": {"cells": 2, "suspects": 187, "k": 160, "matched": None, "uniform": None, "unsure": None,
                 "not_found": None},
        "message": None, "result": None, "was_running": False, "seconds": {},
        "hold": {"value": 1.0, "effective": 1.0, "scale": 1.0, "seq": 451, "given": None},
        "no_factorization": False, "reruns": 0, "estimate_s": qc_pool.ESTIMATE_S["paired"], "waiting": [],
        "reasons": [], "caveats": [], "row": {"start": "paired", "situation": "square"}}
    assert json.loads(json.dumps(record)) == record
    assert tests.pools.calls["pool"] == 1 and runs.parent.touched == 1          # its run detail went out at once
    assert tests.get("p") == record and tests.going()
    assert tests.brief("p") == {"state": "preparing", "progress": record["progress"]}
    assert tests.detail({"id": "p", "state": "paused"}) == {"id": "p", "state": "paused", "qc_test": tests.brief("p")}
    assert tests.detail({"id": "o"}) == {"id": "o"}
    # One test at a time per run.
    refused(tests, "p", BODY, 409, "A test of this run is on its way: wait for its answer, or cancel it")
    # Once matched, the record counts what was found.
    test = tests.tests["p"]
    test.matched = matched(pool(members=3, suspects=201))
    assert tests.public(test)["pool"] == {"cells": 3, "suspects": 201, "k": 160, "matched": 540, "uniform": 1150,
                                          "unsure": 2, "not_found": 1}


def test_a_run_started_from_cells_only_is_tested_on_four_runs_a_side(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "start_of", lambda meta: "cells")
    box = [100.0, 100.0, 550.0, 550.0]
    runs, tests, go = ready(tmp_path, {"region": {"kind": "box", "box": box, "bbox": box}, "box": box})
    record = tests.start("p", {**BODY, "flag": "prior-weight", "value": 0.2})
    go.set()
    assert (record["start"], record["runs_per_side"], list(record["runs"]), record["progress"]["runs"]) == \
        ("cells", 4, ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"], 8)
    assert (record["area"]["rule"], record["area"]["box"], record["square"], record["estimate_s"]) == \
        ("as_is", box, None, qc_pool.ESTIMATE_S["cells"])
    assert (record["name"], record["current"], record["tried"]) == ("hold", 1.0, True)


def test_the_pool_preview_adds_busy_or_other_run_to_the_pools_own_reasons(tmp_path):
    """The preview: the pools' reply, with other_run among its waits while another run samples or another run's test
    goes (offer stays true: a press waits for it), and busy {why: "test"} among its refusals while a test of this run
    waits or is on its way (offer false); and the run's hold at the frame a test reads (`hold`)."""
    runs, tests, go = ready(tmp_path)
    reply = tests.preview("p", 7, wait=2.5)
    assert {k: v for k, v in reply.items() if k != "settings"} == {
        "v": 1, "state": "ready", "run": "p", "cell": 7, "offer": True, "reasons": [], "waits": [], "caveats": [],
        "hold": {"value": 1.0, "effective": 1.0, "scale": 1.0, "seq": 451}}
    assert list(reply["settings"]) == list(T.FLAGS) and reply["settings"]["prior-weight"]["now"] == 1.0
    busy = {"code": "busy", "words": "A test of this run is on its way", "why": "test"}
    # Another run sampling: other_run, after the run's own waits, in the page's order.
    other = runs.add("o", state="running")
    runs.parent.meta.update(state="running", schedule_done=None)
    reply = tests.preview("p", 7)
    assert (reply["offer"], reply["reasons"]) == (True, [])
    assert reply["waits"] == [qc_pool.reason("not_at_end"), qc_pool.reason("other_run")]
    other.meta["state"] = "paused"
    runs.parent.meta.update(state="paused", schedule_done=dict(PARENT["schedule_done"]))
    assert tests.preview("p", 7)["waits"] == []
    # Another run's test going (its runs not made yet): other_run too; a waiting one is not going.
    for state, waits in (("preparing", ["other_run"]), ("comparing", ["other_run"]), ("waiting", []),
                         ("done", [])):
        tests.tests["o"] = T.Test("o", pool(run="o"), "cell-compactness", 0.02, 0.04, tests.clock, runs_per_side=3)
        tests.tests["o"].state = state
        assert [w["code"] for w in tests.preview("p", 7)["waits"]] == waits, state
    del tests.tests["o"]
    # A test of this run on its way: busy {why: test}.
    tests.start("p", BODY)
    reply = tests.preview("p", 7)
    assert (reply["offer"], reply["reasons"]) == (False, [busy])
    go.set()
    assert tests.pools.calls["preview"] == 8
    with pytest.raises(RunError):
        tests.preview("nope", 7)


def test_a_cancel_dismissal_and_closing(tmp_path):
    """DELETE cancels a test on its way (kept as "cancelled") and dismisses one that ended; closing the app cancels
    every test on its way, waits for its thread, and refuses any new one."""
    runs, tests, go = ready(tmp_path)
    tests._drive = lambda test: test.cancel.wait(10)          # on its way until cancelled
    tests.start("p", BODY)
    test = tests.tests["p"]
    cancelled = tests.cancel("p")
    assert (cancelled["state"], cancelled["message"]) == ("cancelled", None) and not test.thread.is_alive()
    assert not tests.going() and tests.get("p")["state"] == "cancelled" and "cancelled" in test.marks
    assert tests.cancel("p")["state"] == "cancelled"           # dismissed
    for ask in (tests.get, tests.cancel, tests.view):
        with pytest.raises(QcError, match="No test for this run"):
            ask("p")
    tests.start("p", {**BODY, "value": 0.03})
    test = tests.tests["p"]
    tests.close()
    assert (test.state, test.message) == ("cancelled", "The app closed") and not test.thread.is_alive()
    refused(tests, "p", BODY, 503, "The app is closing")
    tests.forget("p")
    assert tests.brief("p") is None and tests.tests == {}


# ---- a waiting test ----------------------------------------------------------------------------------------------------
def reached(runs: FakeRuns):
    """The parent reaches the end of its schedule: proseg holds there, and the run pauses (runs.py _schedule_done)."""
    runs.parent.meta.update(state="paused", paused=True, schedule_done=dict(PARENT["schedule_done"]))


BEFORE_END = {"state": "paused", "paused": True, "schedule_done": None}       # paused before the end of its schedule


def test_a_waiting_test_starts_once_its_run_reaches_its_end(tmp_path, monkeypatch):
    """A press before the run's end: the run resumed once (the test is not going: the run may resume and step), the
    test waiting, its waits pushed when they change; it asks every WAIT_S (the pools' prerequisites) and once nothing
    holds takes its pool, with the parent's settings read again (a hold set live meanwhile), and goes on: "waited" in
    its seconds (the time limit counts from it), then its input, runs and answer, as a test pressed at the end."""
    asked = answered(monkeypatch)
    pushed = []

    def happens(waits):
        pushed.append((waits, runs.parent.touched, [w["code"] for w in test.waiting]))
        if waits == 2:
            runs.parent.meta["live_values"] = {"prior-weight": 2.0}
        if waits == 3:
            reached(runs)
    runs, tests, test, clock = pressed(tmp_path, BEFORE_END, before=happens)
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p")]
    assert (test.state, [w["code"] for w in test.waiting], test.resumed) == ("waiting", ["not_at_end"], True)
    assert not tests.going() and runs.parent.meta["state"] == "running"
    # Its hold as pressed: the value now, no frame tested yet (the pool's comes with it), nothing given.
    assert test.hold == {"value": 1.0, "effective": 1.0, "scale": 1.0, "seq": None, "given": None}
    tests._drive(test)
    assert (test.state, test.message, test.result) == ("done", None, asked["result"])
    assert [codes for _, _, codes in pushed[:3]] == [["not_at_end"]] * 3 and test.waiting == []
    assert pushed[0][1] == 1 and pushed[2][1] == 1          # nothing new to push while its waits stay
    assert list(tests.public(test)["seconds"]) == \
        ["waited", "input", "matched", "made", "running", "sampled", "comparing", "done"]
    assert test.marks["waited"] - test.began == 3 * T.WAIT_S
    assert tests.pools.calls["prerequisites"] >= 4 and tests.pools.calls["pool"] == 2
    assert (test.asked, test.parent_defaults, test.hold) == ({"prior-weight": 2.0}, False, {
        "value": 2.0, "effective": 2.0, "scale": 1.0, "seq": 451,
        "given": {"scale": 1.0, "as_now": 2.0, "tried": None}})          # 100 transcripts a cell: proseg's scale 1
    assert (test.pool.seq, test.area.rule, test.row, test.was_running) == (451, "square", ("paired", "square"), False)
    assert [e for e in runs.events if e[0] in ("pause", "resume") and e[1] == "p"] == [("resume", "p")]
    assert [e[1] for e in runs.events if e[0] == "create"] == ["A1", "B1", "A2", "B2", "A3", "B3"]


def test_a_waiting_test_cancelled_pauses_again_the_run_its_press_resumed(tmp_path):
    """DELETE of a waiting test: cancelled; the run its press resumed, still sampling before its end, paused again;
    one it did not resume (another test was going), or one the user paused meanwhile, left as it is."""
    runs, tests, test, clock = pressed(tmp_path, BEFORE_END)
    assert runs.parent.meta["state"] == "running" and test.resumed
    cancelled = tests.cancel("p")
    assert (cancelled["state"], cancelled["message"], cancelled["waiting"]) == ("cancelled", None, [])
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p"), ("pause", "p")]
    assert runs.parent.meta["state"] == "paused" and runs.made == []
    tests._drive(test)                                    # its thread sees the cancel: nothing more
    assert test.state == "cancelled" and runs.made == []
    # Paused by the user meanwhile: left paused.
    runs, tests, test, clock = pressed(tmp_path / "paused", BEFORE_END)
    runs.control("p", "pause")
    tests.cancel("p")
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p"), ("pause", "p")]
    # Another run's test going at the press: the run is not resumed (then nothing to pause again).
    runs, tests, clock = world(tmp_path / "busy", BEFORE_END)
    tests.tests["o"] = T.Test("o", pool(run="o"), "cell-compactness", 0.02, 0.04, clock, runs_per_side=3)
    tests._drive = lambda test: None
    record = tests.start("p", BODY)
    assert [w["code"] for w in record["waiting"]] == ["not_at_end", "other_run"]
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [] and not tests.tests["p"].resumed
    tests.cancel("p")
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == []


def test_a_waiting_test_that_ends_before_its_runs_pauses_again_the_run_its_press_resumed(tmp_path, monkeypatch):
    """A waiting test that ends failed before its runs start (a refusal while it waits; the sample's learning failing
    again after its retry): the run its press resumed, still sampling before its end, is paused again, as a Cancel
    pauses it (the page's Close says the run goes on as it was before the test). One that reached its end meanwhile,
    where proseg holds it paused, is left as it is."""
    answered(monkeypatch)
    holder = {}
    runs, tests, test, clock = pressed(tmp_path, BEFORE_END, extra=[],
                                       before=lambda waits: waits == 2 and holder["go"]())
    holder["go"] = lambda: tests.pools.extra.append(qc_pool.reason("check_unavailable"))
    assert runs.parent.meta["state"] == "running" and test.resumed
    tests._drive(test)
    assert (test.state, [r["code"] for r in test.reasons], runs.made) == ("failed", ["check_unavailable"], [])
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p"), ("pause", "p")]
    assert runs.parent.meta["state"] == "paused"
    # The sample's learning failing again after the press asked for it once more.
    qc = FakeQc(sample_state="failed")
    runs, tests, test, clock = pressed(tmp_path / "learn", BEFORE_END, qc=qc, extra=qc.waits,
                                       before=lambda waits: waits == 2 and qc.set(sample_state="failed"))
    tests._drive(test)
    assert (test.state, test.message, runs.made) == ("failed", T.CHECK_FAILED, [])
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p"), ("pause", "p")]
    # At its end meanwhile (held there, paused), then the cell refused: nothing to pause again.
    holder = {}
    runs, tests, test, clock = pressed(tmp_path / "end", BEFORE_END, extra=[],
                                       before=lambda waits: waits == 2 and holder["go"]())
    holder["go"] = lambda: (reached(runs), setattr(tests.pools, "make", lambda meta, cell: (_ for _ in ()).throw(
        qc_pool.OutsideDomain([qc_pool.reason("no_suspects")]))))
    tests._drive(test)
    assert (test.state, [r["code"] for r in test.reasons], runs.made) == ("failed", ["no_suspects"], [])
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("resume", "p")]


def test_waiting_tests_go_one_at_a_time(tmp_path, monkeypatch):
    """While another run's test goes, a waiting test waits (other_run), its run not resumed; once that one is done, the
    run is resumed (once), and the test starts once its run reaches its end."""
    answered(monkeypatch)
    other = {}

    def happens(waits):
        if waits == 3:
            other["test"].state = "done"
        if waits == 6:
            reached(runs)
    runs, tests, clock = world(tmp_path, BEFORE_END)
    other["test"] = tests.tests["o"] = T.Test("o", pool(run="o"), "cell-compactness", 0.02, 0.04, clock,
                                              runs_per_side=3)
    other["test"].state = "running"
    tests._drive = lambda test: None
    tests.start("p", BODY)
    del tests._drive
    test = tests.tests["p"]
    test.thread.join(5)
    test.cancel = Steps(runs, clock, happens)
    seen = []
    real = tests._meet
    monkeypatch.setattr(tests, "_meet", lambda test, run, waits: (seen.append([w["code"] for w in test.waiting]),
                                                                   real(test, run, waits)))
    tests._drive(test)
    assert test.state == "done" and seen[0] == ["not_at_end", "other_run"]
    assert [e for e in runs.events if e[0] in ("pause", "resume") and e[1] == "p"] == [("resume", "p")]
    assert test.marks["waited"] - test.began == 6 * T.WAIT_S


def test_a_waiting_test_switches_the_check_on_and_starts_once_it_learned(tmp_path, monkeypatch):
    """The mixing check off at the press: switched on (its own switch, once); while it learns the test waits; once the
    sample is learned and F scored, it starts. Switched off again by the user meanwhile: never switched on again."""
    answered(monkeypatch)
    qc = FakeQc(enabled=False, sample_state="learning")

    def happens(waits):
        if waits == 2:
            qc.set(sample_state="ready")
    runs, tests, test, clock = pressed(tmp_path, qc=qc, extra=qc.waits, before=happens)
    assert qc.calls == [("configure", "p", True)] and [w["code"] for w in test.waiting] == ["check_wait"]
    tests._drive(test)
    assert test.state == "done" and qc.calls == [("configure", "p", True)]
    # Off again while it waits: the test waits on (until Cancel), the switch left as the user set it.
    qc = FakeQc(enabled=False, sample_state="learning")
    runs, tests, test, clock = pressed(tmp_path / "off", qc=qc, extra=qc.waits, before=lambda waits: (
        waits == 1 and qc.set(enabled=False)) or (waits == 4 and tests.cancel("p")))
    tests._drive(test)
    assert test.state == "cancelled" and qc.calls == [("configure", "p", True)] and runs.made == []


def test_a_waiting_test_asks_a_failed_sample_once_and_fails_when_it_fails_again(tmp_path, monkeypatch):
    """A sample whose learning failed: asked again once (at the press); a second failure ends the test, in its own
    words, before any run. One that fails only after the press is asked again once too."""
    answered(monkeypatch)
    qc = FakeQc(sample_state="failed")
    runs, tests, test, clock = pressed(tmp_path, qc=qc, extra=qc.waits,
                                       before=lambda waits: waits == 2 and qc.set(sample_state="failed"))
    assert qc.calls == [("ensure", "s", True, False)] and test.met == {"check_retry"}
    tests._drive(test)
    assert (test.state, test.message, test.reasons) == ("failed", T.CHECK_FAILED, [])
    assert qc.calls == [("ensure", "s", True, False)] and runs.made == []
    qc = FakeQc(sample_state="learning")
    fails = {1: "failed", 3: "failed"}
    runs, tests, test, clock = pressed(tmp_path / "later", qc=qc, extra=qc.waits,
                                       before=lambda waits: waits in fails and qc.set(sample_state=fails[waits]))
    assert qc.calls == []
    tests._drive(test)
    assert (test.state, test.message) == ("failed", T.CHECK_FAILED)
    assert qc.calls == [("ensure", "s", True, False)]


def test_a_waiting_test_asks_for_go_deepers_details_and_fails_when_they_fail(tmp_path, monkeypatch):
    """Go deeper's details missing: asked for once (at the press, with a retry: one that failed before is asked
    again); learned, the test starts; their failure after that ends the test, in its own words."""
    answered(monkeypatch)
    qc = FakeQc(detail="failed")
    runs, tests, test, clock = pressed(tmp_path, qc=qc, extra=qc.waits,
                                       before=lambda waits: waits == 2 and qc.set(detail="ready"))
    assert qc.calls == [("ensure", "s", True, True)] and [w["code"] for w in test.waiting] == ["details"]
    tests._drive(test)
    assert test.state == "done" and qc.calls == [("ensure", "s", True, True)]
    qc = FakeQc(detail="none")
    runs, tests, test, clock = pressed(tmp_path / "fails", qc=qc, extra=qc.waits,
                                       before=lambda waits: waits == 2 and qc.set(detail="failed"))
    tests._drive(test)
    assert (test.state, test.message, runs.made) == ("failed", T.DETAILS_FAILED, [])
    assert qc.calls == [("ensure", "s", True, True)]


def test_a_waiting_test_ends_on_what_stops_it(tmp_path, monkeypatch):
    """While it waits: its run ending before its end fails it with that reason; a refusal of the pools (here the
    check can't run) fails it with its reasons; the cell's own refusal, known once nothing holds, too; its run being
    saved or deleted cancels it; the run's own value by then fails it in the POST's words. No run is made."""
    answered(monkeypatch)
    cases = [(lambda runs, tests: runs.parent.meta.update(state="stopped"), "failed", ["ended"],
              "This run ended before its last iteration"),
             (lambda runs, tests: tests.pools.extra.append(qc_pool.reason("check_unavailable")), "failed",
              ["check_unavailable"], "The mixing check can't run on this sample"),
             (lambda runs, tests: (reached(runs), setattr(tests.pools, "make", lambda meta, cell: (_ for _ in ()).throw(
                 qc_pool.OutsideDomain([qc_pool.reason("no_suspects")])))), "failed", ["no_suspects"],
              "No suspects to test in this cell: none glows, or none could be placed"),
             (lambda runs, tests: runs.parent.meta.update(state="finishing", finish_requested=True), "cancelled", [],
              "Your run is being saved, so the test stopped"),
             (lambda runs, tests: runs.runs.pop("p"), "cancelled", [], "Your run was deleted"),
             (lambda runs, tests: (reached(runs), runs.parent.meta.update(live_values={"cell-compactness": 0.02})),
              "failed", [], "0.02 is what this run uses now: choose another value to test")]
    for n, (happen, state, reasons, message) in enumerate(cases):
        holder = {}
        runs, tests, test, clock = pressed(tmp_path / str(n), BEFORE_END, extra=[],
                                           before=lambda waits, holder=holder: waits == 2 and holder["happen"]())
        holder["happen"] = lambda runs=runs, tests=tests, happen=happen: happen(runs, tests)
        tests._drive(test)
        assert (test.state, [r["code"] for r in test.reasons], test.message) == (state, reasons, message), n
        assert runs.made == [] and tests.public(test)["waiting"] == []
        if reasons:
            assert tests.public(test)["reasons"] == [qc_pool.reason(code) for code in reasons]


def test_a_parent_sampling_past_its_end_is_paused_as_the_runs_start(tmp_path, monkeypatch):
    """A run sampling on past its end (its pool taken at the end: past_end, a caveat) is paused as its test's runs
    start, at the press or once its wait is over (another run sampling meanwhile), and the test says so
    (was_running: the page resumes it after); a run at its end, paused, is left as it is (was_running false)."""
    answered(monkeypatch)
    sampling = {"state": "running", "paused": False, "continued": True}
    runs, tests, go = ready(tmp_path, sampling)
    record = tests.start("p", BODY)
    go.set()
    assert (record["state"], record["was_running"], [c["code"] for c in record["caveats"]]) == \
        ("preparing", True, ["past_end"])
    assert [e for e in runs.events if e[0] in ("pause", "resume")] == [("pause", "p")]
    assert runs.parent.meta["state"] == "paused"
    other = {}
    runs, tests, clock = world(tmp_path / "queued", sampling)
    other["run"] = runs.add("o", state="running")
    tests._drive = lambda test: None
    tests.start("p", BODY)
    del tests._drive
    test = tests.tests["p"]
    test.thread.join(5)
    assert [w["code"] for w in test.waiting] == ["other_run"] and runs.events == []
    test.cancel = Steps(runs, clock, lambda waits: waits == 2 and other["run"].meta.update(state="paused"))
    tests._drive(test)
    assert (test.state, test.was_running) == ("done", True)
    assert [e for e in runs.events if e[0] in ("pause", "resume") and e[1] == "p"] == [("pause", "p")]
    runs, tests, go = ready(tmp_path / "at-end")
    assert tests.start("p", BODY)["was_running"] is False and runs.events == []
    go.set()


def test_a_waiting_test_with_the_engines_own_pools(tmp_path, monkeypatch):
    """With engine/qc_pool.py's own pools (test_qc_pool's synthetic run, its cells' answers made up there): a press
    while the run's last frame is not scored yet waits (check_scoring), the check asked once to score it; once it is,
    the loop takes the cell's pool at the run's end, with its caveats (here the changed schedule's), and the test is on
    its way. A press on a cell that owns nothing at the end, or on a run that ended before its end, is refused."""
    import test_qc_pool as QP
    world = QP.World(tmp_path / "w", meta={"options": {"samples": 300}})
    try:
        monkeypatch.setitem(qc_rule.CONSTANTS["paired"], "k", 6)
        QP.fabricate({c: {"key": "OWN", "pattern": "OWN", "n": 3, "giver": 10} for c in range(12)}, monkeypatch)
        world.runs.root = tmp_path / "runs"
        world.runs.active = lambda: [world.run]
        world.run.touch = lambda: None
        scored = world.qc_dir / f"{QP.SEQ:06d}.qc.bin"
        scored.rename(world.qc_dir / "kept.bin")
        tests = T.QcTests(world.runs, world.qc, world.analysis, where=world.where, pools=world.pools)
        tests._drive = lambda test: None
        record = tests.start("p1", {"cell": 5, "flag": "cell-compactness", "value": 0.02})
        del tests._drive
        assert record["state"] == "waiting" and [w["code"] for w in record["waiting"]] == ["check_scoring"]
        assert (record["row"], record["cell_name"], world.qc.configured) == \
            ({"start": "paired", "situation": "as_is"}, QP.D.RUN_IDS[5], [("p1", QP.SEQ)])
        test = tests.tests["p1"]
        test.thread.join(5)
        # The test's cancel: each of its loop's waits; the check scored the run's last frame at the second.
        test.cancel = Looks(before=lambda waits: waits == 2 and (world.qc_dir / "kept.bin").rename(scored))
        tests._wait(test)
        assert (test.state, test.waiting, test.cancel.timeouts) == ("preparing", [], [T.WAIT_S] * 2)
        assert (test.pool.seq, test.pool.row, test.area.rule, test.progress["of"]) == \
            (QP.SEQ, ("paired", "as_is"), "as_is", 550)
        assert [c["code"] for c in tests.public(test)["caveats"]] == ["schedule"]
        assert world.qc.configured == [("p1", QP.SEQ)]                         # asked once
        del tests.tests["p1"]
        with pytest.raises(qc_pool.OutsideDomain) as refused:
            tests.start("p1", {"cell": 11, "flag": "cell-compactness", "value": 0.02})    # vanished at the end
        assert refused.value.reasons == [qc_pool.reason("not_judged")]
        world.meta.update(state="stopped", schedule_done=None)
        with pytest.raises(qc_pool.OutsideDomain) as refused:
            tests.start("p1", {"cell": 5, "flag": "cell-compactness", "value": 0.02})
        assert refused.value.reasons == [qc_pool.reason("ended")] and tests.tests == {}
    finally:
        world.pools.close()


def test_a_press_whose_pool_takes_longer_than_its_wait_makes_the_test_wait_for_it(tmp_path, monkeypatch):
    """A pool still being worked out when the press's own wait for it ends (PRESS_WAIT_S; a whole sample's can take
    minutes): never "try again" (503), a waiting test with nothing to wait for but its pool (the page says "Starting…"),
    which its loop takes once it is there (the same work going on, as in the engine's run worker), and goes on."""
    answered(monkeypatch)
    monkeypatch.setattr(T, "PRESS_WAIT_S", 0.05)
    runs, tests, test, clock = pressed(tmp_path, delay_s=0.5)
    record = tests.public(test)
    assert (record["state"], record["waiting"], record["pool"], record["area"]) == ("waiting", [], None, None)
    assert (tests.pools.calls["pool"], runs.made, test.resumed) == (1, [], False)
    tests._drive(test)
    assert (test.state, test.pool.seq, len(runs.made)) == ("done", 451, 6) and "waited" in test.marks


def test_a_pool_slower_than_every_wait_with_the_engines_own_pools(tmp_path, monkeypatch):
    """With engine/qc_pool.py's own pools, a pool slower than every wait for it (cells of 0.3 s each against waits of
    0.2 s, as a whole sample's can take minutes against the press's few seconds and pool()'s two minutes): the press
    makes a waiting test, never "try again"; its loop asks again past each wait that ends first (StillWorking: the same
    request, each cell worked out once) and goes on once the pool is there."""
    import test_qc_pool as QP
    world = QP.World(tmp_path / "w")
    try:
        monkeypatch.setitem(qc_rule.CONSTANTS["paired"], "k", 6)
        calls = []
        QP.fabricate({c: {"key": "OWN", "pattern": "OWN", "n": 3, "giver": 10} for c in range(12)}, monkeypatch, calls)
        quick = qc_pool.QcPools._cell

        def slow(self, context, weights, c, **count):
            time.sleep(0.3)
            return quick(self, context, weights, c, **count)
        monkeypatch.setattr(qc_pool.QcPools, "_cell", slow)
        real, asked = qc_pool.QcPools.pool, []

        def short(self, run_id, cell, *, timeout=120.0, check=None):
            asked.append(timeout)
            return real(self, run_id, cell, timeout=0.2, check=check)
        monkeypatch.setattr(qc_pool.QcPools, "pool", short)
        world.runs.root = tmp_path / "runs"
        world.runs.active = lambda: [world.run]
        world.run.touch = lambda: None
        tests = T.QcTests(world.runs, world.qc, world.analysis, where=world.where, pools=world.pools)
        tests._drive = lambda test: None
        record = tests.start("p1", {"cell": 5, "flag": "cell-compactness", "value": 0.02})
        del tests._drive
        assert (record["state"], record["waiting"], record["pool"], record["area"]) == ("waiting", [], None, None)
        assert (asked, record["cell_name"]) == ([T.PRESS_WAIT_S], QP.D.RUN_IDS[5])
        test = tests.tests["p1"]
        test.thread.join(5)
        test.cancel = Looks(nap=0.05)
        tests._wait(test)
        assert (test.state, test.waiting, test.pool.seed, test.area.rule) == ("preparing", [], 5, "as_is")
        assert len(asked) > 2 and set(asked[1:]) == {120.0}                  # asked again past a wait that ended
        assert calls and len(calls) == len(set(calls))                        # the same request: each cell once
    finally:
        world.pools.close()


def test_the_time_limit_counts_from_the_end_of_the_wait_and_grows_with_the_schedule(tmp_path):
    """TIMEOUT_S for the measured schedule's 450 iterations, longer in proportion for a longer one (never shorter);
    counted from the end of the test's wait when it waited."""
    assert [T.time_limit(of) for of in (450, 300, 900, 675)] == [600.0, 600.0, 1200.0, 900.0]
    runs, tests, test = following(tmp_path, plan=lambda role, options: DONE[:3])     # sampling on, never ending
    test.marks["waited"] = test.began + 100.0
    test.progress["of"] = 900
    test.cancel.most = int((100.0 + 1200.0) / T.FOLLOW_S) + 1                        # its time limit's looks
    with pytest.raises(ValueError, match=r"^no answer within 20 minutes$"):
        tests._follow(test)
    assert test.clock() - test.began > 100.0 + 1200.0
    assert test.cancel.waits == int((100.0 + 1200.0) / T.FOLLOW_S) + 1


def test_an_engine_that_stopped_mid_test_leaves_no_input_behind(tmp_path):
    runs = FakeRuns(tmp_path)
    left = tmp_path / "qc-tests" / "abcd1234" / "transcripts.parquet"
    left.parent.mkdir(parents=True)
    left.write_bytes(b"PAR1")
    tests = T.QcTests(runs, None, None, where=Where(), pools=FakePools(runs, made))
    assert tests.inputs == tmp_path / "qc-tests"
    wait_for(lambda: not left.parent.exists(), timeout=10)


# ---- the test's thread -------------------------------------------------------------------------------------------------
def answered(monkeypatch, at: int = 450) -> dict:
    """The readout and the rule as stand-ins: every run read at iteration `at` (asked["read"]: the folders, items and
    inner box given), and the result made of what it was given (asked["answer"]; asked["result"], set beforehand or
    not, carrying the caveats given as answer() does)."""
    asked = {}

    def read_runs(dirs, items, inner, **kwargs):
        asked["read"] = (dict(dirs), items, list(inner), kwargs)
        return {role: SimpleNamespace(iter=at, role=role) for role in dirs}

    def answer(items, order, readouts, sides, start, **kwargs):
        asked["answer"] = (items, order, readouts, sides, start, kwargs)
        result = asked.setdefault("result", {"v": 2, "verdict": "not_clear"})
        result.setdefault("caveats", [dict(c) for c in kwargs.get("caveats") or ()])
        return result
    monkeypatch.setattr(qc_readout, "read_runs", read_runs)
    monkeypatch.setattr(qc_rule, "answer", answer)
    return asked


def test_a_test_goes_from_its_input_to_its_answer_and_leaves_nothing_behind(tmp_path, monkeypatch):
    asked = answered(monkeypatch)
    runs, tests, clock = world(tmp_path)
    test = begun(runs, tests, clock)
    parent_input = runs.parent.dir / "input" / "transcripts.parquet"
    tests._drive(test)
    assert (test.state, test.message) == ("done", None) and test.result is asked["result"]
    # The shared input: the square's rows of the parent's own, the test's own file; matched there with the settings
    # every run takes.
    own = tests.inputs / test.id / "transcripts.parquet"
    assert test.input["path"] == str(own) and test.input["owned"] and test.input["ids"] == "own"
    assert tests.pools.calls["match_input"] == 1 and test.matched.input == str(own)
    # 2N runs from it (I10), all deleted later, and the test's own input removed; the parent's input and state stay.
    ids = [run.meta["id"] for run in runs.made]
    assert [e[1] for e in runs.events if e[0] == "create"] == ["A1", "B1", "A2", "B2", "A3", "B3"]
    assert [e for e in runs.events if e[0] == "delete"] == [("delete", test.runs[role], True) for role in test.roles]
    assert set(runs.runs) == {"p"} and not own.parent.exists() and parent_input.exists()
    assert runs.parent.meta["state"] == "paused" and runs.parent.meta["schedule_done"] == PARENT["schedule_done"]
    # Read: each run's folder by its role; the items found; the area's inner box. Judged: the sides in role order, the
    # start, the setting, the situation, the runs' last iteration, the matching's counts, the parent's defaults.
    dirs, items, inner, _ = asked["read"]
    assert dirs == {run.meta["qc_test"]["role"]: run.dir for run in runs.made} and items is ITEMS
    assert inner == [225.0, 225.0, 675.0, 675.0]
    items, order, readouts, sides, start, kwargs = asked["answer"]
    assert (items, order, start) == (ITEMS, ORDER, "paired") and set(readouts) == set(test.roles)
    assert sides == {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]}
    assert kwargs == {"setting": {"flag": "cell-compactness", "value": 0.02, "current": 0.04}, "situation": "square",
                      "at_iter": 450, "matched_counts": COUNTS, "parent_defaults": True, "row": ("paired", "square"),
                      "caveats": []}
    assert test.result == {"v": 2, "verdict": "not_clear", "caveats": [], "hold": {
        "value": 1.0, "effective": 1.0, "scale": 1.0, "seq": 451, "given": {"scale": 1.0, "as_now": 1.0, "tried": None}}}
    # Its steps in turn (seconds since the POST), its progress at the end, the crop around the cell's footprint.
    record = tests.public(test)
    assert list(record["seconds"]) == ["input", "matched", "made", "running", "sampled", "comparing", "done"]
    assert record["progress"] == {"iter": 450, "of": 450, "runs": 6, "done": 6, "sides": {"A": 450, "B": 450},
                                  "marks": [200, 400], "phase": "reading", "retry": False}
    assert record["runs"] == dict(zip(["A1", "B1", "A2", "B2", "A3", "B3"], ids)) and record["result"] is test.result
    assert test.crop == [327.0, 326.5, 375.0, 374.5] and tests.where.asked == [("p", 451, [7])]
    assert not tests.going() and tests.brief("p")["state"] == "done"


def test_a_region_re_run_as_it_is_reads_the_parents_input_and_keeps_it(tmp_path, monkeypatch):
    answered(monkeypatch)
    box = [100.0, 100.0, 550.0, 550.0]
    runs, tests, clock = world(tmp_path, {"region": {"kind": "box", "box": box, "bbox": box}, "box": box})
    test = begun(runs, tests, clock)
    tests._drive(test)
    parent_input = runs.parent.dir / "input" / "transcripts.parquet"
    assert test.state == "done" and test.input["path"] == str(parent_input) and not test.input["owned"]
    assert all(run.meta["input"]["path"] == str(parent_input) for run in runs.made)
    assert all(run.meta["region"] == {"kind": "box", "box": box, "bbox": box} for run in runs.made)
    assert parent_input.exists() and not (tests.inputs / test.id).exists()


def test_the_runs_are_made_all_at_once_from_the_shared_input_with_the_setting_on_side_b(tmp_path):
    analysis = Analysis()
    runs, tests, clock = world(tmp_path, {"options": {"cell-compactness": 0.05, "nthreads": 32, "min-qv": 25.0,
                                                      "output-counts-fmt": "csv", "no-factorization": False},
                                          "live_values": {"prior-weight": 2.0}, "prior": {"cells": {"key": "c"}},
                                          "prepared": {"seeded_from": "nuclei", "transcripts": 400}},
                               analysis=analysis)
    test = begun(runs, tests, clock, value=0.02)
    tests._input(test)
    tests._make(test)
    roles = ["A1", "B1", "A2", "B2", "A3", "B3"]
    assert [e[1] for e in runs.events] == roles and list(test.runs.values()) == ["t1", "t3", "t5", "t2", "t4", "t6"]
    shared = {"path": test.input["path"], "transcripts": test.input["transcripts"], "ids": "own", "owned": True,
              "prepared": {"seeded_from": "nuclei", "transcripts": 400, "source": "parent"}}
    asked = {"cell-compactness": 0.05, "min-qv": 25.0, "prior-weight": 2.0, "nthreads": 8}
    for role, run in zip(roles, runs.made):
        meta = run.meta
        assert meta["options"] == ({**asked, "cell-compactness": 0.02} if role[0] == "B" else asked), role
        assert meta["qc_test"] == {"of": "p", "role": role, "side": role[0], "test": test.id}
        assert meta["input"] == shared and (meta["paused"], meta["speed"]) == (True, "max")     # given its hold first
        assert meta["region"] == test.area.spec and meta["region"] is not test.area.spec
        assert (meta["sample"], meta["prior"]) == ("s", {"cells": {"key": "c"}})
        assert run.room == FakeRuns.max_active + 6          # 2N more places: the parent waits paused among them
    assert analysis.calls == [(run.meta["id"], False) for run in runs.made]
    assert "made" in test.marks and test.marks["made"] == clock.t
    # Without factorisation (the parent's, or the test's rerun): every run alike.
    test.no_factorization = True
    tests._make(test)
    assert all(run.meta["options"]["no-factorization"] is True for run in runs.made[6:])


# ---- the runs' hold (proseg's scale: the input's transcripts per cell over 100) ---------------------------------------
# A case as measured on a sample: proseg's scale 10.5 in the parent's 1.2 mm box, 4.52 in its test's 500 µm square.
PARENT_SCALE = {"transcripts": 1_050_000, "cells": 1000}       # 1,050 transcripts a cell: scale 10.5
SQUARE_SCALE = {"transcripts": 452_000, "cells": 1000}         # 452 a cell: scale 4.52
AS_NOW, TRIED = 1.1615043878555298, 4.646017551422119          # 0.5 and 2.0 at 10.5, given at 4.52


def scaled(tmp_path, at_f, meta=None, after=None, **child) -> tuple[FakeRuns, T.QcTests, Clock]:
    """The fake world with the parent at proseg's scale 10.5 and its frames up to F (451, its schedule's end), F's
    record holding the prior-weight proseg sampled it with (`at_f`), then, for a parent that sampled on past its end
    (`after`: the hold it was set to there), four frames more with that one, and its test's runs at scale 4.52
    (`child`: more of their info.json)."""
    runs, tests, clock = world(tmp_path, meta)
    runs.parent.info.update(PARENT_SCALE)
    runs.parent.frames = [{"seq": seq, "iter": max(0, seq - 1)} for seq in range(451)] + \
        [{"seq": 451, "iter": 450, "live": {"prior-weight": at_f}}]
    if after is not None:
        runs.parent.frames += [{"seq": seq, "iter": seq - 1, "live": {"prior-weight": after}}
                               for seq in range(452, 456)]
        runs.parent.meta["continued"] = True
    runs.child.update(SQUARE_SCALE, **child)
    return runs, tests, clock


def calls(runs: FakeRuns) -> list:
    """Every set and resume of a test's run, as it was asked: (action, role, values, the run's state and frames then)."""
    seen, real = [], runs.control

    def control(run_id, action, speed=None, values=None):
        run = runs.get(run_id)
        if action in ("set", "resume") and run.meta.get("qc_test"):
            seen.append((action, run.meta["qc_test"]["role"], dict(values or {}), run.meta["state"], len(run.frames)))
        return real(run_id, action, speed, values)
    runs.control = control
    return seen


def test_the_runs_are_made_paused_and_given_the_parents_hold_at_their_own_scale(tmp_path, monkeypatch):
    """Proseg multiplies the hold by the input's transcripts per cell over 100: 10.5 in the parent's box, 4.52 in the
    test's square, so the parent's 0.5 (5.25 as proseg applies it there) would be 2.26 in its runs. Each run is made
    paused and, once its info.json is there and it is paused, given as a live setting the value whose hold there is the
    parent's at the frame tested, 1.1615043878555298 (times 4.52 in float32: 5.25 exactly), then resumed: before its
    first iteration. The weight is F's alone: not the 3.0 the parent sampled with on past its end, nor the 2.0 set live
    while it rests since. Only the value given changes: each command line carries the parent's settings, its hold F's
    (as the answer's defaults say it). The record and the answer say what was given."""
    asked = answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path, 0.5, {"live_values": {"prior-weight": 2.0}}, after=3.0)
    seen = calls(runs)
    test = begun(runs, tests, clock)
    assert test.hold == {"value": 0.5, "effective": 5.25, "scale": 10.5, "seq": 451, "given": None}
    assert (test.asked, test.parent_defaults) == ({"prior-weight": 0.5}, False)
    tests._drive(test)
    assert (test.state, test.message) == ("done", None)
    assert T.scale_of(runs.child) == float(np.float32(4.52)) and T.scale_of(runs.parent.info) == 10.5
    assert np.float32(AS_NOW) * np.float32(T.scale_of(runs.child)) == np.float32(0.5) * np.float32(10.5) == 5.25
    assert seen == [(action, role, values, "paused", 0) for role in test.roles
                    for action, values in (("set", {"prior-weight": AS_NOW}), ("resume", {}))]
    assert all(run.meta["live_values"] == {"prior-weight": AS_NOW} for run in runs.made)
    assert all(run.meta["options"]["prior-weight"] == 0.5 for run in runs.made)       # its command line: F's hold
    given = {"value": 0.5, "effective": 5.25, "scale": 10.5, "seq": 451,
             "given": {"scale": 4.52, "as_now": AS_NOW, "tried": None}}
    assert test.hold == given and tests.public(test)["hold"] == given and test.result["hold"] == given
    assert asked["answer"][5]["setting"] == {"flag": "cell-compactness", "value": 0.02, "current": 0.04}
    assert asked["answer"][5]["parent_defaults"] is False


def test_a_test_of_the_hold_gives_side_b_the_value_at_the_parents_scale(tmp_path, monkeypatch):
    """A test of the hold (2.0 against the parent's 0.5): side A as now (1.1615043878555298), side B the value picked
    applied at the parent's scale, 2.0 x 10.5 = 21.0, the strength a run of the parent with 2.0 would have:
    4.646017551422119 here (times 4.52 in float32: 21.0). B's command line carries 2.0, as before."""
    asked = answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path, 0.5)
    seen = calls(runs)
    test = begun(runs, tests, clock, flag="prior-weight", value=2.0)
    tests._drive(test)
    assert (test.state, test.message) == ("done", None)
    assert np.float32(TRIED) * np.float32(T.scale_of(runs.child)) == np.float32(2.0) * np.float32(10.5) == 21.0
    assert [(role, values) for action, role, values, _, _ in seen if action == "set"] == \
        [(role, {"prior-weight": TRIED if role[0] == "B" else AS_NOW}) for role in test.roles]
    assert [run.meta["options"].get("prior-weight") for run in runs.made] == [None, 2.0] * 3      # A1, B1, A2, ...
    given = {"value": 0.5, "effective": 5.25, "scale": 10.5, "seq": 451,
             "given": {"scale": 4.52, "as_now": AS_NOW, "tried": TRIED}}
    assert test.hold == given and test.result["hold"] == given
    assert asked["answer"][5]["setting"] == {"flag": "prior-weight", "value": 2.0, "current": 0.5}


USED_AT_F = "{} is what this run used at its last iteration: choose another value to test"


def test_a_test_of_the_hold_is_against_its_value_at_the_frame_tested(tmp_path, monkeypatch):
    """The hold is read at the frame tested alone: F sampled with 0.5, then the parent sampled on past its end with 3.0
    and rests with 2.0 set live since. A test of the hold at F's 0.5 would give both sides 0.5's strength: refused at
    the press, in words that say the run used it at its last iteration. 2.0, its value now, is a test against 0.5: side
    A given 0.5's strength, side B 2.0's at the parent's scale, `current` 0.5 in the record and the answer, the command
    lines' hold F's 0.5 (B's 2.0), the parent's settings not proseg's defaults."""
    asked = answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path, 0.5, {"live_values": {"prior-weight": 2.0}}, after=3.0)
    hold = {**BODY, "flag": "prior-weight"}
    refused(tests, "p", {**hold, "value": 0.5}, 422, USED_AT_F.format("0.5"))
    refused(tests, "p", {**hold, "value": 0.49999998}, 422, USED_AT_F.format("0.5"))     # float32's precision
    assert runs.made == [] and tests.tests == {} and runs.parent.touched == 0
    assert [tests._current(runs.parent, "prior-weight", seq) for seq in (None, 451, 455)] == [2.0, 0.5, 3.0]
    seen = calls(runs)
    test = begun(runs, tests, clock, flag="prior-weight", value=2.0)
    assert (test.current, test.asked, test.parent_defaults) == (0.5, {"prior-weight": 0.5}, False)
    tests._drive(test)
    assert (test.state, test.message) == ("done", None)
    assert [(role, values) for action, role, values, _, _ in seen if action == "set"] == \
        [(role, {"prior-weight": TRIED if role[0] == "B" else AS_NOW}) for role in test.roles]
    assert [run.meta["options"]["prior-weight"] for run in runs.made] == [0.5, 2.0] * 3   # A1, B1, A2, ...
    assert tests.public(test)["current"] == 0.5 and test.result["hold"]["given"]["tried"] == TRIED
    assert asked["answer"][5]["setting"] == {"flag": "prior-weight", "value": 2.0, "current": 0.5}
    assert asked["answer"][5]["parent_defaults"] is False


def test_a_hold_changed_since_the_frame_tested_is_neither_the_runs_nor_the_defaults(tmp_path, monkeypatch):
    """F sampled with proseg's default hold (1.0), the parent then sampled on with 2.0 set live: a test of another
    setting gives its runs F's hold (their command lines 1.0, each given 1.0's strength at its own scale) and says the
    parent's settings were proseg's defaults, as they were at F; a test of the hold at 2.0 is not refused (its runs as
    now hold as F did), one at 1.0 is, in its words."""
    asked = answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path, 1.0, {"live_values": {"prior-weight": 2.0}}, after=2.0)
    current = tests._current(runs.parent, "prior-weight", 451)
    assert current == 1.0 and tests._current(runs.parent, "prior-weight") == 2.0
    tests._refuse_value(runs.parent, "prior-weight", 2.0, current)                       # 2.0 against 1.0: taken
    with pytest.raises(QcError) as same:
        tests._refuse_value(runs.parent, "prior-weight", 1.0, current)
    assert (same.value.status, str(same.value)) == (422, USED_AT_F.format("1"))
    seen = calls(runs)
    test = begun(runs, tests, clock)                                                     # compactness
    assert (test.hold["value"], test.asked, test.parent_defaults) == (1.0, {"prior-weight": 1.0}, True)
    tests._drive(test)
    as_f = T.given_hold(10.5, T.scale_of(runs.child))
    assert test.state == "done" and as_f == float(np.float32(10.5 / np.float32(4.52)))
    assert {values["prior-weight"] for action, _, values, _, _ in seen if action == "set"} == {as_f}
    assert all(run.meta["options"]["prior-weight"] == 1.0 for run in runs.made)
    assert asked["answer"][5]["parent_defaults"] is True


def test_the_pool_reply_says_the_hold_at_the_frame_tested_and_a_press_is_checked_there(tmp_path, monkeypatch):
    """The pool reply carries the parent's hold at the frame a test reads (`hold`: F's record's 0.5, with 2.0 set live
    since, at proseg's scale 10.5 there), which the page's form checks a value of the hold with, as the press does: a
    test of 0.5 is refused in the words for its last iteration and one of 2.0, its value now, is taken, also while the
    cell's pool is still to come (a wait holds: the press checks the value at F all the same, never the value now), and
    its waiting record says that one hold, F's, in its `current` and its `hold` alike. Before the run's end: its value
    now, no frame, in the reply and in a waiting record. Without the run's counts in its info.json: the value alone."""
    runs, tests, clock = scaled(tmp_path / "f", 0.5, {"live_values": {"prior-weight": 2.0}}, after=3.0)
    assert tests.preview("p", 7)["hold"] == {"value": 0.5, "effective": 5.25, "scale": 10.5, "seq": 451}
    tests.pools.extra[:] = [qc_pool.reason("check_scoring", wait=True)]
    with pytest.raises(qc_pool.NotYet):
        tests.pools.pool("p", 7)
    hold = {**BODY, "flag": "prior-weight"}
    refused(tests, "p", {**hold, "value": 0.5}, 422, USED_AT_F.format("0.5"))
    assert tests.tests == {}
    tests._drive = lambda test: None
    record = tests.start("p", {**hold, "value": 2.0})
    assert (record["state"], record["current"], record["value"]) == ("waiting", 0.5, 2.0)
    assert record["hold"] == {**tests.preview("p", 7)["hold"], "given": None} and record["hold"]["value"] == \
        record["current"]
    runs, tests, clock = scaled(tmp_path / "before", 0.5, {**BEFORE_END, "live_values": {"prior-weight": 2.0}})
    assert tests.preview("p", 7)["hold"] == {"value": 2.0, "effective": 21.0, "scale": 10.5, "seq": None}
    tests._drive = lambda test: None
    record = tests.start("p", {**hold, "value": 0.5})
    assert (record["state"], record["current"], record["hold"]) == \
        ("waiting", 2.0, {"value": 2.0, "effective": 21.0, "scale": 10.5, "seq": None, "given": None})
    del runs.parent.info["transcripts"]
    assert tests.preview("p", 7)["hold"] == {"value": 2.0, "effective": None, "scale": None, "seq": None}


def test_a_waiting_test_of_the_hold_is_taken_against_its_value_at_the_frame_tested(tmp_path, monkeypatch):
    """Pressed before the run's end, a test of the hold is against its value now (no frame tested yet); once the run
    reached its end, its value at F decides as the pool is taken: F sampled with 0.5 and 2.0 set live since, a test of
    0.5 fails in the POST's words for its last iteration (no run made), and one of 2.0 goes on against 0.5."""
    answered(monkeypatch)

    def at_end(runs):
        reached(runs)
        runs.parent.frames = [{"seq": seq, "iter": max(0, seq - 1)} for seq in range(451)] + \
            [{"seq": 451, "iter": 450, "live": {"prior-weight": 0.5}}]
        runs.parent.meta["live_values"] = {"prior-weight": 2.0}
    for value, state, message, current in ((0.5, "failed", USED_AT_F.format("0.5"), 1.0), (2.0, "done", None, 0.5)):
        holder = {}
        runs, tests, test, clock = pressed(tmp_path / str(value), BEFORE_END, {**BODY, "flag": "prior-weight",
                                                                                "value": value}, extra=[],
                                           before=lambda waits, holder=holder: waits == 2 and at_end(holder["runs"]))
        holder["runs"] = runs
        assert (test.state, test.current) == ("waiting", 1.0)                          # its value now, as pressed
        tests._drive(test)
        assert (test.state, test.message, test.current) == (state, message, current), value
        assert (runs.made == []) == (state == "failed")


HERE = "Nothing can be tested here: matching your run's hold here would need {}, above the most proseg takes (1000)"
AT = "Nothing can be tested at {} here: matching it here would need {}, above the most proseg takes (1000)"


def test_a_hold_past_what_proseg_takes_refuses_the_test_before_any_run_samples(tmp_path, monkeypatch):
    """The parent holding at 100 (1050 as proseg applies it) tested in a square of fewer than 100 transcripts a cell
    (proseg's scale 1) would need 1050 there, past the most proseg takes (1000, proseg_options' bound, which a live
    value is kept to): nothing can be tested here, said in plain words before any run is given a value or resumed, and
    its runs are cleared away. A value just past it reads past it too: 430.4762268066406 at 10.5 needs the float32
    1000.0001220703125 at 4.52, "1000" in Python's `g`, said as its float32's shortest decimal."""
    from sopa_companion.engine import proseg_options
    answered(monkeypatch)
    assert T.HOLD_MAX == next(o["max"] for o in proseg_options.OPTIONS if o["flag"] == "prior-weight") == 1000.0
    for at_f, child, said in ((100.0, {"transcripts": 50_000}, "1050"), (430.4762268066406, {}, "1000.0001")):
        runs, tests, clock = scaled(tmp_path / said, at_f, **child)
        seen = calls(runs)
        test = begun(runs, tests, clock)
        tests._drive(test)
        assert (test.state, test.message, test.reasons) == ("failed", HERE.format(said), [])
        assert seen == [] and [e[0] for e in runs.events if e[0] != "create"] == ["abort"] * 6 + ["delete"] * 6
        assert set(runs.runs) == {"p"} and test.hold["given"] is None
    needs = T.given_hold(float(np.float32(430.4762268066406) * np.float32(10.5)), T.scale_of(runs.child))
    assert (needs, f"{needs:g}") == (1000.0001220703125, "1000")    # what `g` alone would have said


def test_a_value_of_the_hold_past_what_proseg_takes_here_is_refused_and_another_is_taken(tmp_path, monkeypatch):
    """A test of the hold at 500 on the parent held at 0.5 (proseg's scale 10.5 there: 5250 as applied) would need
    1161.5 in its square (scale 4.52), past the most proseg takes: nothing can be tested at 500 here, said in plain
    words (another value can), before any run is given a value or resumed, its runs cleared away; its runs as now
    would have taken 1.1615 there. A test of 400 there (928.9) goes on as any other."""
    answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path / "500", 0.5)
    seen = calls(runs)
    test = begun(runs, tests, clock, flag="prior-weight", value=500.0)
    tests._drive(test)
    need = T.given_hold(float(np.float32(500.0) * np.float32(10.5)), T.scale_of(runs.child))
    assert f"{need:g}" == "1161.5" and T.given_hold(5.25, T.scale_of(runs.child)) == AS_NOW
    assert (test.state, test.message) == ("failed", AT.format("500", "1161.5"))
    assert seen == [] and set(runs.runs) == {"p"} and test.hold["given"] is None
    runs, tests, clock = scaled(tmp_path / "400", 0.5)
    seen = calls(runs)
    test = begun(runs, tests, clock, flag="prior-weight", value=400.0)
    tests._drive(test)
    assert (test.state, test.message) == ("done", None)
    tried = T.given_hold(float(np.float32(400.0) * np.float32(10.5)), T.scale_of(runs.child))
    assert [(role, values) for action, role, values, _, _ in seen if action == "set"] == \
        [(role, {"prior-weight": tried if role[0] == "B" else AS_NOW}) for role in test.roles]


def test_a_hold_past_what_proseg_takes_is_found_once_one_run_has_read_its_input(tmp_path, monkeypatch):
    """Every run of a test reads one input from one start the same way, so one run's scale is every run's: the first
    to have read its input (its info.json) shows the hold is past the bound, and the test is refused at once, never
    waiting for the others to load (here they never would)."""
    answered(monkeypatch)
    runs, tests, clock = scaled(tmp_path, 100.0, transcripts=50_000)
    runs.start = lambda role: [("paused", runs.child)] if role == "B2" else [("preparing", None)] * 3000
    seen = calls(runs)
    test = begun(runs, tests, clock)
    test.cancel.most = 5
    tests._drive(test)
    assert (test.state, test.message) == ("failed", HERE.format("1050"))
    assert seen == [] and test.cancel.waits <= 2 and set(runs.runs) == {"p"}


def test_a_run_that_fails_or_never_loads_before_its_hold_fails_the_test(tmp_path, monkeypatch):
    """While the runs load (made paused, waiting for their info.json): one that fails fails the test with its role and
    its words, and runs that never come up paused fail it at its time limit; nothing is given or resumed."""
    answered(monkeypatch)
    runs, tests, clock = world(tmp_path / "failed")
    runs.start = lambda role: [("failed", None, {"message": "proseg exited with code 101"})] if role == "B2" else \
        [("paused", runs.child)]
    seen = calls(runs)
    test = begun(runs, tests, clock)
    tests._drive(test)
    assert (test.state, test.message) == ("failed", "The test could not finish: its B2 run failed: proseg exited with "
                                                    "code 101")
    assert seen == [] and set(runs.runs) == {"p"}
    runs, tests, clock = world(tmp_path / "loading")
    runs.start = lambda role: [("paused", None)] * 3000                     # proseg never writes its info.json
    seen = calls(runs)
    test = begun(runs, tests, clock)
    test.cancel.most = int(T.TIMEOUT_S / T.FOLLOW_S) + 1                    # its time limit's looks
    tests._drive(test)
    assert (test.state, test.message) == ("failed", "The test could not finish: no answer within 10 minutes")
    assert test.cancel.waits == int(T.TIMEOUT_S / T.FOLLOW_S) + 1 and test.progress["phase"] == "loading"
    assert seen == [] and set(runs.runs) == {"p"}


def test_a_temperature_set_in_the_parent_is_left_out_of_its_runs_and_its_answer_says_so(tmp_path, monkeypatch):
    """A parent whose Temperature was set (here live, lowered to settle its outlines): tested all the same (its changed
    schedule a caveat), its runs on both sides without it (from their start it would settle them at once), and its
    answer says the parent's settings were not proseg's defaults (parent_defaults false: the tooltip's line)."""
    asked = answered(monkeypatch)
    runs, tests, clock = world(tmp_path, {"options": {"temperature": 1.0}, "live_values": {"temperature": 0.1}})
    test = begun(runs, tests, clock)
    tests._drive(test)
    assert test.state == "done" and len(runs.made) == 6 and test.parent_defaults is False
    assert all("temperature" not in run.meta["options"] for run in runs.made)
    assert asked["answer"][5]["parent_defaults"] is False
    assert [c["code"] for c in test.result["caveats"]] == ["schedule"]


def test_a_forgotten_sample_is_opened_again_and_clusters_umap_may_be_unavailable(tmp_path):
    """A SpatialData parent whose sample was closed since: opened again (not added to the recent list) before the first
    run is made, else Runs.create would take it for a Xenium bundle. Clusters & UMAP refusing to be switched off
    (unavailable here) does not stop the test."""
    runs, tests, clock = world(tmp_path, {"sample_kind": "zarr", "sample_path": "/data/sample.zarr"},
                               analysis=Analysis(fails=True))
    runs.zarr.is_known = False
    test = begun(runs, tests, clock)
    tests._input(test)
    tests._make(test)
    assert runs.events[0] == ("open", "/data/sample.zarr", False)
    assert [e[0] for e in runs.events[1:]] == ["create"] * 6 and all(test.runs.values())
    assert len(tests.analysis.calls) == 6
    tests._make(test)                                   # open now: not opened again
    assert [e[0] for e in runs.events].count("open") == 1


def test_following_waits_for_every_run_and_keeps_the_progress(tmp_path):
    """I9's progress while the runs go: iter the lowest run's newest iteration (the end's for a run past sampling), each
    side's lowest, how many are done, the phase (loading until every run has a frame, then sampling, then writing once
    every run is past sampling); "running" once every run has a frame. The test waits until every run is done."""
    slow = [("running", None), ("running", 0), ("running", 190), ("running", 400), ("running", 449), ("finishing", 450),
            ("done", 450)]
    fast = [("running", None), ("running", 0), ("running", 210), ("running", 420), ("finishing", 450), ("done", 450)]
    seen = []
    runs, tests, test = following(tmp_path, plan=lambda role, options: list(slow if role == "B1" else fast),
                                  before=lambda waits: seen.append((test.state, T._progress(test))))
    tests._follow(test)
    assert test.cancel.waits == 7 and all(run.meta["state"] == "done" for run in runs.made)
    fixed = {"of": 450, "runs": 6, "marks": [200, 400], "retry": False}
    assert seen == [
        ("preparing", {**fixed, "iter": 0, "done": 0, "sides": {"A": 0, "B": 0}, "phase": "loading"}),
        ("preparing", {**fixed, "iter": 0, "done": 0, "sides": {"A": 0, "B": 0}, "phase": "loading"}),
        ("running", {**fixed, "iter": 0, "done": 0, "sides": {"A": 0, "B": 0}, "phase": "sampling"}),
        ("running", {**fixed, "iter": 190, "done": 0, "sides": {"A": 210, "B": 190}, "phase": "sampling"}),
        ("running", {**fixed, "iter": 400, "done": 0, "sides": {"A": 420, "B": 400}, "phase": "sampling"}),
        ("running", {**fixed, "iter": 449, "done": 0, "sides": {"A": 450, "B": 449}, "phase": "sampling"}),
        ("running", {**fixed, "iter": 450, "done": 5, "sides": {"A": 450, "B": 450}, "phase": "writing"})]
    assert T._progress(test) == {**fixed, "iter": 450, "done": 6, "sides": {"A": 450, "B": 450}, "phase": "writing"}
    # When each step was reached: running at the third look (every run framed), sampled at the seventh.
    assert list(test.marks) == ["input", "made", "running", "sampled"]
    assert (test.marks["running"] - test.began, test.marks["sampled"] - test.began) == (0.5, 1.5)


def test_the_run_goes_out_at_once_for_a_new_state_and_at_most_once_a_second_for_progress(tmp_path):
    seen = []
    runs, tests, test = following(tmp_path, before=lambda waits: seen.append(runs.parent.touched))
    tests._follow(test)
    # Looks every 0.25 s: the progress at the first look, the new state at the third (at once), then a second later.
    assert seen == [1, 1, 2, 2, 2, 2] and runs.parent.touched == 3
    clock = [100.0]
    tests = T.QcTests(runs, None, None, clock=lambda: clock[0], where=Where(), pools=FakePools(runs, made))
    test = T.Test("p", pool(), "cell-compactness", 0.02, 0.04, tests.clock, runs_per_side=3)
    tests.tests["p"] = test
    runs.parent.touched = 0
    tests._set(test, state="running")
    assert runs.parent.touched == 1
    for n in range(1, 6):                                # the progress, every 0.25 s as the runs are looked at
        test.progress["iter"] = n * 10
        tests._push(test)
        clock[0] += 0.25
    assert runs.parent.touched == 2 and tests.brief("p")["progress"]["iter"] == 50
    tests._set(test, state="comparing")
    assert runs.parent.touched == 3 and test.marks == {"running": 100.0, "comparing": 101.25}
    # A test that ended keeps its end: an answer that comes after a cancel never replaces it.
    tests._set(test, state="cancelled", message="Your run ended, so the test stopped")
    tests._set(test, state="done", result={"v": 2})
    assert (test.state, test.message, test.result) == ("cancelled", "Your run ended, so the test stopped", None)
    assert tests.public(test)["seconds"] == {"running": 0.0, "comparing": 1.2, "cancelled": 1.2}


@pytest.mark.parametrize("ending", ["stopped by the disk guard", "failed", "deleted"])
def test_a_run_that_ends_without_its_file_fails_the_test_at_once_with_its_role(tmp_path, ending):
    words = {"stopped by the disk guard": "only 12 GB free on the data disk",
             "failed": "proseg exited with code 101: thread 'main' panicked"}

    def plan(role, options):
        if role != "B2" or ending == "deleted":
            return list(DONE)
        state = "stopped" if ending.startswith("stopped") else "failed"
        return DONE[:2] + [(state, 0, {"message": words[ending]})]
    runs, tests, test = following(tmp_path, plan=plan,
                                  before=lambda waits: waits == 3 and ending == "deleted" and runs.runs.pop("t4"))
    with pytest.raises(ValueError) as failed:
        tests._follow(test)
    assert str(failed.value) == {"stopped by the disk guard": "its B2 run stopped: only 12 GB free on the data disk",
                                 "failed": "its B2 run failed: proseg exited with code 101: thread 'main' panicked",
                                 "deleted": "its B2 run was deleted"}[ending]
    assert test.cancel.waits == 3                         # the look right after: not one more


def test_a_test_without_an_answer_by_its_timeout_fails(tmp_path):
    runs, tests, test = following(tmp_path, plan=lambda role, options: DONE[:3])     # sampling on, never ending
    test.cancel.most = int(T.TIMEOUT_S / T.FOLLOW_S) + 1                              # its time limit's looks
    with pytest.raises(ValueError, match=r"^no answer within 10 minutes$"):
        tests._follow(test)
    # The first look past TIMEOUT_S from the POST (the runs are looked at every FOLLOW_S).
    assert test.cancel.waits == int(T.TIMEOUT_S / T.FOLLOW_S) + 1 and test.clock() - test.began > T.TIMEOUT_S


def test_a_run_that_needs_the_retry_has_every_run_made_again_without_factorisation(tmp_path):
    """runs.py retries a run whose first start failed without factorisation; the test then makes all its runs again
    without it, both sides alike, once, in this order: every old run stopped before any new one is made, the new
    ones made before the old ones are deleted (so no id is taken again), the old runs' views and names let go, its
    progress back to its start (retry true), pushed at once."""
    retried = {"retried_without_factorization": True,
               "message": "Proseg could not start on this region with factorisation; retrying without it…"}

    def plan(role, options):
        if role == "A2" and not options.get("no-factorization"):
            return [("running", None, retried)] + DONE[1:]
        return list(DONE)
    at_make = []
    runs, tests, test = following(tmp_path, plan=plan)
    real = tests._make

    def make(again):
        at_make.append((T._progress(again), again.state, runs.parent.touched, dict(again.marks)))
        return real(again)
    tests._make = make
    test.views["A1"] = {"key": ("t1", 1, 0, None), "order": (0, 1), "final": False, "view": {}}
    test.names["A1"] = {"run": "t1", "cells": {}}
    old = [run.meta["id"] for run in runs.made]
    touched = runs.parent.touched
    tests._follow(test)
    new = [run.meta["id"] for run in runs.made[6:]]
    assert len(new) == 6 and not set(new) & set(old) and set(test.runs.values()) == set(new)
    kinds = [(e[0], e[1]) for e in runs.events]
    stopped = [i for i, (kind, _) in enumerate(kinds) if kind == "abort"]
    remade = [i for i, (kind, _) in enumerate(kinds) if kind == "create"][6:]
    deleted = [i for i, (kind, _) in enumerate(kinds) if kind == "delete"]
    assert sorted(kinds[i][1] for i in stopped) == sorted(old) and max(stopped) < min(remade)
    assert max(remade) < min(deleted) and sorted(kinds[i][1] for i in deleted) == sorted(old)
    assert all(e[2] is True for e in runs.events if e[0] == "delete")
    assert (test.reruns, test.no_factorization, test.views, test.names) == (1, True, {}, {})
    assert all(run.meta["options"]["no-factorization"] is True for run in runs.made[6:])
    progress, state, pushed, marks = at_make[0]
    assert progress == {"iter": 0, "of": 450, "runs": 6, "done": 0, "sides": {"A": 0, "B": 0}, "marks": [200, 400],
                        "phase": "preparing", "retry": True}
    assert state == "preparing" and pushed == touched + 2 and list(marks) == ["input"]   # the new runs' still to come
    assert T._progress(test)["retry"] is True and T._progress(test)["done"] == 6
    assert list(test.marks) == ["input", "made", "running", "sampled"]
    assert all(run.meta["state"] == "done" for run in runs.made[6:])
    # The new runs, made paused, are given their hold and resumed in turn once the old ones are deleted.
    held = [i for i, (kind, run_id) in enumerate(kinds) if kind in ("set", "resume") and run_id in new]
    assert [kinds[i] for i in held] == [(kind, test.runs[role]) for role in test.roles for kind in ("set", "resume")]
    assert min(held) > max(deleted) and all(run.meta["live_values"] == {"prior-weight": 1.0} for run in runs.made[6:])


def test_an_old_run_that_does_not_stop_fails_the_test_and_nothing_is_made_again(tmp_path):
    def plan(role, options):
        return [("running", None, {"retried_without_factorization": True})] + DONE[1:] if role == "A2" else list(DONE)
    runs, tests, test = following(tmp_path, plan=plan)
    runs.made[0].stubborn = True                    # A1: an abort leaves it running
    test.clock.tick = 5.0                           # each look at the clock while it waits: 5 s
    with pytest.raises(ValueError, match=r"^a test run did not stop$"):
        tests._follow(test)
    assert [e[0] for e in runs.events].count("create") == 6 and [e[0] for e in runs.events].count("abort") == 6
    assert (test.reruns, test.no_factorization) == (0, False) and list(test.runs.values()) == ["t1", "t3", "t5",
                                                                                                "t2", "t4", "t6"]


def test_a_run_that_fails_in_the_new_batch_fails_the_test(tmp_path):
    """The second failure: a run made again without factorisation that still cannot start (runs.py retries no run that
    already carries the flag) fails the test like any other."""
    def plan(role, options):
        if options.get("no-factorization"):
            return [("running", None), ("failed", None, {"message": "proseg exited with code 101"})] if role == "B1" \
                else list(DONE)
        return [("running", None, {"retried_without_factorization": True})] + DONE[1:] if role == "A3" else list(DONE)
    runs, tests, test = following(tmp_path, plan=plan)
    with pytest.raises(ValueError, match=r"^its B1 run failed: proseg exited with code 101$"):
        tests._follow(test)
    assert test.reruns == 1 and [e[0] for e in runs.events].count("create") == 12


def test_every_run_is_aborted_before_any_is_waited_for(tmp_path, monkeypatch):
    """8 runs that each take 0.5 s to stop after their abort stop in about 0.5 s together, not 4 s; 8 that never stop
    are waited for until one deadline, STOP_S from the first look (not 8 x STOP_S), and said. A run that ended is not
    aborted, and one gone is passed over."""
    runs = FakeRuns(tmp_path)
    tests = T.QcTests(runs, None, None, where=Where(), pools=FakePools(runs, made))
    test = T.Test("p", pool(), "cell-compactness", 0.02, 0.04, time.monotonic, runs_per_side=4)
    for role in test.roles:
        test.runs[role] = runs.add(f"r{role}", state="running").meta["id"]
    runs.slow = 0.5
    began = time.monotonic()
    assert tests._stop_runs(test) is True
    took = time.monotonic() - began
    aborts = [e for e in runs.events if e[0] == "abort"]
    assert len(aborts) == 8 and max(e[2] for e in aborts) - began < 0.2 and 0.5 <= took < 2.0
    monkeypatch.setattr(T, "STOP_S", 0.5)
    runs.events.clear()
    for run in runs.runs.values():
        run.meta["state"], run.stubborn = "running", True
    began = time.monotonic()
    assert tests._stop_runs(test) is False
    assert 0.5 <= time.monotonic() - began < 2.0 and [e[0] for e in runs.events] == ["abort"] * 8
    runs.events.clear()
    runs.get("rA1").meta["state"] = "done"
    for role in ("A2", "A3", "A4", "B1", "B2", "B3", "B4"):
        runs.get(f"r{role}").meta["state"] = "stopped"
    del runs.runs["rB4"]
    assert tests._stop_runs(test) is True and runs.events == []


def test_clearing_deletes_the_runs_later_and_removes_the_tests_own_input_only(tmp_path):
    runs, tests, test = following(tmp_path)
    own = Path(test.input["path"])
    parent_input = runs.parent.dir / "input" / "transcripts.parquet"
    assert own.parent == tests.inputs / test.id and own.exists() and test.input["owned"]
    tests._clear(test)
    assert [e[0] for e in runs.events if e[0] != "create"] == ["set", "resume"] * 6 + ["abort"] * 6 + ["delete"] * 6
    assert [e for e in runs.events if e[0] == "delete"] == [("delete", test.runs[role], True) for role in test.roles]
    assert not own.parent.exists() and parent_input.exists() and set(runs.runs) == {"p"}
    # A run that does not stop is left in the list (aborted, never sampling on); the others go; so does the input.
    runs, tests, test = following(tmp_path / "again")
    runs.made[2].stubborn = True
    test.clock.tick = 5.0
    tests._clear(test)
    assert set(runs.runs) == {"p", runs.made[2].meta["id"]} and not (tests.inputs / test.id).exists()
    # Re-run as it is, the parent's own file is the input: nothing of the parent's is removed.
    box = [100.0, 100.0, 550.0, 550.0]
    runs, tests, clock = world(tmp_path / "as-is", {"region": {"kind": "box", "box": box, "bbox": box}, "box": box})
    test = begun(runs, tests, clock)
    tests._input(test)
    tests._clear(test)
    assert Path(test.input["path"]).exists() and not test.input["owned"]


def test_a_parent_that_is_saved_left_or_deleted_stops_its_test_one_that_merely_ended_does_not(tmp_path, monkeypatch):
    """While its runs go: its run being finished, saved or left, or deleted, cancels the test (its runs aborted and
    deleted); one that merely ended (stopped, done or failed at or past its end: its input and frames stay until it is
    left) keeps it, and the test answers."""
    answered(monkeypatch)
    for change, words in (({"finish_requested": True, "state": "finishing"},
                           "Your run is being saved, so the test stopped"),
                          ({"leaving": {"answer": "discard"}}, "Your run is being saved, so the test stopped"),
                          ({"zarr_save": {"state": "saving"}}, "Your run is being saved, so the test stopped"),
                          (None, "Your run was deleted"),
                          ({"state": "stopped"}, None), ({"state": "failed"}, None),
                          ({"state": "done", "finish_requested": True, "zarr_save": {"state": "saved"}}, None)):
        runs, tests, clock = world(tmp_path / str(len(list(tmp_path.iterdir()))))

        def happens(waits, runs=runs, change=change):
            if waits == 3:
                runs.parent.meta.update(change) if change else runs.runs.pop("p")
        test = begun(runs, tests, clock, before=happens)
        tests._drive(test)
        if words is None:
            assert (test.state, test.message) == ("done", None), change
            assert [e[0] for e in runs.events if e[0] not in ("create", "delete")] == ["set", "resume"] * 6, change
        else:
            assert (test.state, test.message, test.result) == ("cancelled", words, None), words
            assert [e[0] for e in runs.events if e[0] != "create"] == \
                ["set", "resume"] * 6 + ["abort"] * 6 + ["delete"] * 6
        assert not (tests.inputs / test.id).exists()


def test_a_cancel_while_the_runs_sample_stops_and_deletes_every_run(tmp_path, monkeypatch):
    answered(monkeypatch)
    runs, tests, clock = world(tmp_path)
    # The first wait: the runs paused with their info.json (`_hold_runs`); then they sample.
    test = begun(runs, tests, clock, before=lambda waits: waits == 4 and tests.cancel("p"))
    tests._drive(test)
    assert (test.state, test.message, test.result) == ("cancelled", None, None) and not tests.going()
    assert [e[0] for e in runs.events if e[0] != "create"] == ["set", "resume"] * 6 + ["abort"] * 6 + ["delete"] * 6
    assert set(runs.runs) == {"p"} and not (tests.inputs / test.id).exists()
    assert list(tests.public(test)["seconds"]) == ["input", "matched", "made", "running", "cancelled"]


def test_too_few_found_in_the_shared_input(tmp_path, monkeypatch):
    """None of the clicked cell's suspects found in the shared input: the test fails before any run, in its own words.
    A pool short of k once matched (or before) goes on: its runs are judged, and the answer carries too_few with the
    numbers judged (the pool's own too_few replaced; its other caveats kept, in order) only while under k."""
    answered(monkeypatch)
    runs, tests, clock = world(tmp_path)
    test = begun(runs, tests, clock)

    def none(*args, **kwargs):
        raise qc_pool.TooFewFound()
    tests.pools.match_input = none
    tests._drive(test)
    assert (test.state, test.message) == ("failed", "None of this cell's suspects could be found in the area's "
                                                    "transcripts")
    assert runs.made == [] and not (tests.inputs / test.id).exists()
    for judged, said in ((140, [qc_pool.reason("schedule"), qc_pool.reason("too_few", have=140, need=160)]),
                         (160, [qc_pool.reason("schedule")])):
        asked = answered(monkeypatch)
        asked["result"] = {"v": 2, "verdict": "not_clear", "pool": {"suspects": judged}}
        runs, tests, clock = world(tmp_path / str(judged))
        test = begun(runs, tests, clock)
        short = (qc_pool.reason("too_few", have=150, need=160), qc_pool.reason("schedule"))
        test.pool = dataclasses.replace(test.pool, caveats=short)
        real = tests.pools.match_input
        tests.pools.match_input = lambda pool, *args, **kwargs: dataclasses.replace(
            real(pool, *args, **kwargs), pool=dataclasses.replace(pool, caveats=short))
        tests._drive(test)
        assert test.state == "done" and len(runs.made) == 6
        assert asked["answer"][5]["caveats"] == [qc_pool.reason("schedule")]          # given: the pool's but too_few
        assert test.result["caveats"] == said and tests.public(test)["caveats"] == list(short)


def test_a_test_that_fails_says_why_and_leaves_no_runs(tmp_path, monkeypatch):
    answered(monkeypatch)

    def unreadable(dirs, items, inner, **kwargs):
        raise qc_readout.ReadoutError("the test runs did not end at the same iteration")
    monkeypatch.setattr(qc_readout, "read_runs", unreadable)
    runs, tests, clock = world(tmp_path)
    test = begun(runs, tests, clock)
    tests._drive(test)
    assert (test.state, test.result) == ("failed", None)
    assert test.message == "The test could not finish: the test runs did not end at the same iteration"
    assert set(runs.runs) == {"p"} and not (tests.inputs / test.id).exists()
    # The parent's input gone: no other source, said in plain words.
    runs, tests, clock = world(tmp_path / "gone")
    (runs.parent.dir / "input" / "transcripts.parquet").unlink()
    test = begun(runs, tests, clock)
    tests._drive(test)
    assert (test.state, test.message) == ("failed", "The test could not finish: your run's input is missing")
    assert runs.made == []


# ---- the corner view ---------------------------------------------------------------------------------------------------
KEEP_VIEW = 0.75                # proseg keeps this share of every gene in the hand-made runs' counts


def frame_layers(live: Path, record: dict) -> np.ndarray:
    """A hand-made frame's per-layer labels, flat [nz * ny * nx] (a tiled frame's: its consensus, its one layer)."""
    from sopa_companion.engine.deeper_where import _decode       # (not the engine's, which the tests count)
    nx, ny, nz = int(record["nx"]), int(record["ny"]), int(record["nz"])
    if record.get("format") == "tiles":
        from sopa_companion.engine.tiles import Container
        return Container(live / record["file"]).full("consensus").ravel()
    return _decode(live / record["file"])[2 * nx * ny:(2 + nz) * nx * ny]


def destinations(live: Path, record: dict, moved=None) -> np.ndarray:
    """Where proseg put each transcript of a hand-made run at that frame: the label at its observed voxel, those of
    `moved` ({record: label}) elsewhere."""
    import gzip
    from sopa_companion.engine.transcript_index import RECORD as RECORDS
    records = np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORDS)
    info = json.loads((live / "info.json").read_text())
    grid = {key: record[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
    i, j, k, inside = CF.observed_voxels(records, grid, info["z"]["grid_origin"], info["z"]["grid_spacing"])
    labels = frame_layers(live, record)
    dest = np.zeros(len(records), np.int64)
    dest[inside] = labels[(k[inside] * grid["ny"] + j[inside]) * grid["nx"] + i[inside]]
    for r, label in (moved or {}).items():
        dest[int(r)] = int(label)
    return records, dest, grid, info


def count_frame(live: Path, record: dict, moved=None) -> dict:
    """Proseg's count of a hand-made frame (`destinations`, KEEP_VIEW of each gene kept), as proseg-live writes it: the
    counts file, its sidecar and its line (returned)."""
    records, dest, _, info = destinations(live, record, moved)
    found = CF.counts_of(dest, records["gene"], int(info["cells"]), int(info["genes"]))
    e = found["n"] * KEEP_VIEW
    return CF.write_frame_counts(live, int(record["seq"]), int(record["iter"]), ncells=int(info["cells"]),
                                 ngenes=int(info["genes"]), indptr=found["indptr"], genes=found["genes"], n=found["n"],
                                 e=e, f=np.floor(e).astype(np.int64))


def export_frame(run, seq: int, box, token: int, moved=None) -> dict:
    """Proseg's where export of `box` at frame `seq` of a hand-made run (from `destinations`), as proseg-live writes
    it, read in as the engine reads it."""
    record = dict(run.frames[seq])
    records, dest, grid, info = destinations(run.live, record, moved)
    i0, j0, i1, j1 = CF.where_span(box, grid)
    i, j, k, inside = CF.observed_voxels(records, grid, info["z"]["grid_origin"], info["z"]["grid_spacing"])
    span = inside & (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
    entries = {"i": i[span], "j": j[span], "k": k[span], "gene": records["gene"][span].astype(np.int64),
               "dest": dest[span], "n": np.ones(int(span.sum()), np.int64)}
    found = CF.write_where_export(run.live, token, seq, int(record["iter"]), box, grid, entries)
    run.read_new()
    return found


def keep_window(live: Path, layers) -> None:
    """A hand-made tiled frame 0 with its per-layer window over the whole frame (`layers`, [nz, ny, nx]) or none (None),
    as proseg-live keeps the layers of the view it last showed: its container (test_qc_pool_tiled's writer, the planes
    as they were) and its record."""
    from sopa_companion.engine.tiles import Container
    from test_qc_pool_tiled import write_plt1
    record = json.loads((live / "frames.jsonl").read_text().splitlines()[0])
    container = Container(live / record["file"])
    planes = {name: container.full(name) for name in ("consensus", "last_change")}
    info = None if layers is None else {"i0": 0, "j0": 0, "nx": int(layers.shape[2]), "ny": int(layers.shape[1]),
                                        "nz": int(layers.shape[0])}
    write_plt1(live / record["file"], planes, int(container.header["tile"]), int(record["nz"]),
               None if info is None else (info, layers))
    record["window"] = None if info is None else {key: info[key] for key in ("i0", "j0", "nx", "ny")}
    (live / "frames.jsonl").write_text(json.dumps(record) + "\n")


@pytest.fixture
def hand_runs(config):
    """test_deeper_where's hand-made runs (write_run) under the engine's Runs, frame 0 of each counted by proseg
    (count_frame) and read in with its count: make(run_id=(layers, records, options), ...) -> Runs; their transcript
    indexes let go of after. A tiled run's option "window" (True or False) gives its frame its per-layer window over the
    whole frame or none (`keep_window`), whatever write_run keeps by default."""
    from sopa_companion.engine.runs import Runs
    from test_deeper_where import write_run
    data, made_ = Path(config["paths"]["data_dir"]), []
    # Runs written before the engine starts are loaded as they are (not set aside as older runs or crash leftovers).
    config["proseg"]["keep_unsaved"] = True

    def make(**runs):
        for run_id, (layers, records, options) in runs.items():
            options = dict(options)
            window = options.pop("window", None)
            live = write_run(data, run_id, layers, records, **options) / "live"
            if options.get("tiled") and window is not None:
                keep_window(live, layers if window else None)
            count_frame(live, json.loads((live / "frames.jsonl").read_text().splitlines()[0]))
        engine = Runs(config, None)
        made_.append(engine)
        return engine
    yield make
    for engine in made_:
        for run in list(engine.runs.values()):
            engine._release(run)


def add_frame(run, layers: np.ndarray, iteration: int, moved=None) -> dict:
    """One more dense frame of a hand-made run (its consensus: layer 0), read in as proseg-live's next, counted."""
    import zstandard
    seq = len(run.frames)
    record = {**run.frames[-1], "seq": seq, "iter": iteration, "file": f"frames/{seq:06d}.bin.zst"}
    for key in ("counts_file", "counted_file", "counts"):
        record.pop(key, None)
    nz, ny, nx = layers.shape
    raw = np.concatenate([layers[0].ravel(), np.zeros(nx * ny), layers.ravel()]).astype("<u4")
    (run.live / record["file"]).write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.tobytes()))
    with open(run.live / "frames.jsonl", "a") as stream:
        stream.write(json.dumps(record) + "\n")
    count_frame(run.live, record, moved)
    run.read_new()
    return dict(run.frames[-1])


def hand_pool(suspects=(), sources=(), names=None, box=(0.0, 0.0, 400.0, 400.0)) -> qc_pool.Pool:
    """The pool of test_deeper_where's cell A (a-1, cell 0) at the hand-made parent's frame 0, as the corner view takes
    it: its suspects, and the neighbours they come from (parent cell numbers, named in `names`)."""
    return pool(seq=0, iter=40, seed=0, seed_name="a-1", area=area("as_is", box), view_suspects=tuple(suspects),
                view_sources=tuple(sources), names={0: "a-1", **(names or {})})


def hand_test(runs, suspects=(), **roles) -> tuple[T.QcTests, T.Test]:
    """A test of the hand-made run "p" (test_deeper_where's cell A, a-1) as QcTests holds it while its runs (`roles`:
    role -> run id) sample, the parent waiting paused; the crop around A, and the box its runs export."""
    from sopa_companion.engine.deeper_where import DeeperWhere
    runs.get("p").meta["state"] = "paused"
    tests = T.QcTests(runs, None, None, where=DeeperWhere(runs))
    test = T.Test("p", hand_pool(suspects), "prior-weight", 0.5, 2.0, time.monotonic, runs_per_side=3)
    test.place, test.matched = (115.0, 65.0), matched(test.pool)
    test.crop, test.state = [100.0, 50.0, 140.0, 90.0], "running"
    test.where_box = T.view_box(test.crop, test.suspects, test.area.box)
    test.runs.update(roles)
    tests.tests["p"] = test
    return tests, test


def shares(run, seq: int, cell: int, records, suspects, how: str, box=None) -> list:
    """Each suspect's share proseg counts for the cell (label) at frame `seq`, worked out here: its own transcript among
    `records` (the run's, by gene and place within 0.01 µm, one only, else None), times the cell's keep (KEEP_VIEW of
    every gene) and: approx, whether the cell's label holds its observed voxel; exact, the share of its bucket's
    transcripts proseg put in the cell (`destinations`, the export's)."""
    record = dict(run.frames[seq])
    _, dest, grid, info = destinations(run.live, record, None if how == "approx" else box)
    found = []
    labels = frame_layers(run.live, record)
    i, j, k, inside = CF.observed_voxels(records, grid, info["z"]["grid_origin"], info["z"]["grid_spacing"])
    genes = (run.live / "genes.txt").read_text().split("\n")[:-1]
    for s in suspects:
        here = (np.abs(records["x"] - s["x"]) <= 0.01 + 1e-9) & (np.abs(records["y"] - s["y"]) <= 0.01 + 1e-9)
        close = np.flatnonzero(here & (np.asarray(genes, object)[records["gene"]] == s["gene"]) if s["gene"] in genes
                               else np.zeros(len(records), bool))
        if close.size != 1:
            found.append(None)
            continue
        r = int(close[0])
        if how == "approx":
            label = labels[(k[r] * grid["ny"] + j[r]) * grid["nx"] + i[r]] if inside[r] else 0
            found.append(round(KEEP_VIEW * float(label == cell), 3))
        else:
            mates = np.flatnonzero(inside & (i == i[r]) & (j == j[r]) & (k == k[r])
                                   & (records["gene"] == records["gene"][r]))
            found.append(round(KEEP_VIEW * float((dest[mates] == cell).mean()), 3))
    return found


@pytest.mark.parametrize("kind", ["tiled", "window", "dense"])
def test_a_runs_view_holds_each_suspect_by_proseg_s_count(hand_runs, monkeypatch, kind):
    """A run's view of its newest counted frame (test_deeper_where's hand-made cells: A = a-1, its neighbours B = b-1
    and D = d-1): the cell and the neighbours by their Xenium ids; each suspect's share proseg counts for the cell there
    (held, 3 decimals), approx (held_by "approx") from the frame's per-layer labels times the cell's keep: B holds A's
    two right columns in the top layer only of the dense run, so a deep transcript there is not A's; a tiled frame
    reads its window's layers, and without a window places nothing (null); a suspect without its transcript within
    0.01 µm (or of a gene the run lacks) null; exact with the frame's where export of the view's box (held_by "exact"):
    the share of its bucket proseg put in A, every transcript of the bucket counted, suspect or not (some of A's EDGE
    moved to B: a bucket of two suspects and a third transcript, two of the three B's; a suspect at the suspects' edge
    sharing its bucket with a transcript past that edge, A's); the cell's number proseg's count of it (rounded); the
    crop's labels the frame's consensus, cut to the raster; each of them paired with the label of its name in the
    parent. One decode per frame, the view kept; once the run is deleted, the view kept is given."""
    from sopa_companion.engine.deeper_where import DeeperWhere
    from test_deeper_where import GENES, RECORD, layout, placed
    tiled, windowless = kind != "dense", kind == "tiled"
    layers = layout()[None] if tiled else np.stack([layout()] * 4)
    if not tiled:
        layers[3, 10:20, 18:20] = 2
    edge, layer = GENES.index("EDGE"), GENES.index("LAYER")
    # Two of A's EDGE more, sharing one voxel (one bucket: proseg shares it out between them), and a third there that is
    # no suspect; a LAYER beside the suspects' last one (x 149.5, y 51.5, in the background), in its voxel, no suspect.
    others = [(117.9, 63.9), (149.9, 51.5)]
    records = np.concatenate([placed(), np.array([(118.5, 64.25, 0.9, edge), (118.5, 66.25, 0.1, edge),
                                                  (117.2, 63.2, 0.5, edge), (117.6, 63.6, 0.5, edge),
                                                  (117.9, 63.9, 0.5, edge), (149.9, 51.5, 0.5, layer)], RECORD)])
    options = {"tiled": True, "window": kind == "window"} if tiled else {}
    runs = hand_runs(p=(layout()[None], placed(), {"tiled": True, "cells": ["d-1", "zz-1", "a-1", "b-1", "c-1"]}),
                     t1=(layers, records, options))
    run, where = runs.get("t1"), DeeperWhere(runs)
    asked = json.loads(where.reply("t1", 0, 0, [1, 3], ["EDGE", "SPREAD", "LAYER", "FEW"], ["OWN1", "OWN2"]))
    points = asked["points"]
    suspects = [{"x": x, "y": y, "gene": asked["genes"][g], "p": 0.5}
                for x, y, g in zip(points["x"], points["y"], points["gene"]) if (x, y) not in others]
    assert len(suspects) == len(points["x"]) - 2 and max(s["x"] for s in suspects) == 149.5
    suspects += [{"x": 115.0, "y": 65.0, "gene": "NOPE", "p": 0.5},          # a gene the run lacks: no share
                 {"x": 118.52, "y": 64.25, "gene": "EDGE", "p": 0.5},        # 0.02 µm off its transcript: none either
                 {"x": 118.505, "y": 64.255, "gene": "EDGE", "p": 0.5}]      # within 0.01 µm: its own (B's, deep)
    tests = T.QcTests(runs, None, None, where=where)
    test = T.Test("p", hand_pool(suspects, (1, 3, 4), {1: "b-1", 3: "d-1", 4: "zz-1"}), "prior-weight", 0.5, 2.0,
                  time.monotonic, runs_per_side=3)
    test.place, test.runs["A1"] = (115.0, 65.0), "t1"
    tests._crop(test)
    # The crop around A; the box its runs export: the crop and every suspect (B's reach to 149.5 µm), 1 µm around.
    assert test.crop == [91.0, 41.0, 139.0, 89.0] and test.where_box == [91.0, 41.0, 150.5, 89.0]
    decoded = []
    real = T._decode
    monkeypatch.setattr(T, "_decode", lambda path: decoded.append(path.name) or real(path))
    frame = T.newest_counted(run)
    assert frame["seq"] == 0 and frame["counts_file"] == "frames/000000.counts.zst"
    view = tests._view(test, "A1", run, frame)
    assert set(view) == {"role", "side", "seq", "iter", "voxel", "x0", "y0", "w", "h", "labels", "cell", "sources",
                         "held", "held_by", "unsure", "cell_total", "parents", "ms"}
    assert (view["role"], view["side"], view["seq"], view["iter"], view["cell"], view["sources"]) == \
        ("A1", "A", 0, 40, 1, [2, 4, None])
    # The crop [91, 41, 139, 89] cut to the raster [100, 160) x [50, 90): its columns and rows 0-38.
    assert (view["voxel"], view["x0"], view["y0"], view["w"], view["h"]) == (1.0, 100.0, 50.0, 39, 39)
    assert np.array_equal(np.frombuffer(base64.b64decode(view["labels"]), "<u4").reshape(39, 39), layout()[:39, :39])
    approx = shares(run, 0, 1, records, suspects, "approx")
    assert view["held_by"] == "approx" and view["unsure"] == [0] * len(suspects)        # each alone at its place
    if windowless:                                           # no window: nothing placed without the export
        assert view["held"] == [None] * len(suspects)
    else:                                # a tiled frame's window: its layers (here one, the consensus, A's at its edge)
        deep = 0.0 if kind == "dense" else KEEP_VIEW         # the last suspect, deep: B's in the dense top layer
        assert view["held"] == approx and approx[-3:] == [None, None, deep] and 0.75 in approx and 0.0 in approx
    _, dest, _, _ = destinations(run.live, dict(run.frames[0]))
    assert view["cell_total"] == int(np.floor(KEEP_VIEW * (dest == 1).sum() + 0.5)) > 0
    assert view["parents"] == [1, 3, 2, 4, 3, 5, 4, 1], "a-1 b-1 c-1 d-1 here; 3, 4, 5 and 1 in the parent"
    # Kept: the same frame is not looked at again.
    assert tests._view(test, "A1", run, frame) is view
    assert decoded == ([] if tiled else ["000000.bin.zst"])
    # proseg's export of the view's box with that frame: exact, some of A's EDGE in B's (one alone in its bucket; one of
    # the two suspects sharing a bucket, and the third transcript there), the LAYER past the suspects' edge in A's.
    def at(x, y):
        return int(np.flatnonzero((records["x"] == np.float32(x)) & (records["y"] == np.float32(y)))[0])
    a_edge = np.flatnonzero((records["gene"] == edge) & (records["x"] < 120) & (records["x"] > 119))[:6:2]
    moved = {**{int(r): 2 for r in a_edge}, at(117.2, 63.2): 2, at(117.9, 63.9): 2, at(149.9, 51.5): 1}
    export_frame(run, 0, test.where_box, 1, moved)
    exact = tests._run_view(test, "A1", run, frame)
    want = shares(run, 0, 1, records, suspects, "exact", moved)
    assert exact["held_by"] == "exact" and exact["held"] == want and exact["cell_total"] == view["cell_total"]
    shared = {(suspects[k]["x"], suspects[k]["y"]): h for k, h in enumerate(want) if h in (0.25, 0.375)}
    assert shared == {(117.2, 63.2): 0.25, (117.6, 63.6): 0.25, (149.5, 51.5): 0.375}     # A's: 1 in 3; 1 in 2
    assert sum(h == 0 for h in want if h is not None) > sum(h == 0 for h in approx if h is not None) or windowless
    if not tiled:
        moved_layers = np.roll(layers, 1, axis=2)                       # every cell a column to the right
        decoded.clear()
        view = tests._view(test, "A1", run, add_frame(run, moved_layers, 41))
        assert (view["seq"], view["iter"], view["held_by"]) == (1, 41, "approx")
        assert decoded == ["000001.bin.zst"]
        assert np.array_equal(np.frombuffer(base64.b64decode(view["labels"]), "<u4").reshape(39, 39),
                              moved_layers[0, :39, :39])
        assert view["held"] == shares(run, 1, 1, records, suspects, "approx")
    # Through GET /qc/test/view: the run's newest counted frame (the one kept); once the run is deleted (the test's
    # end), the view kept, whatever the test's state; a newer frame asked of the run that went leaves it as it is.
    tests.tests["p"], test.state = test, "running"
    assert tests.view("p") == {"state": "running", "box": test.crop, "suspects": suspects, "runs": [view]}
    runs.delete("t1")
    for state in ("running", "done"):
        test.state = state
        assert tests.view("p") == {"state": state, "box": test.crop, "suspects": suspects, "runs": [view]}
    assert tests._view(test, "A1", run, {**run.frames[-1], "seq": 9, "t": 1.0}) is view


def test_the_view_asks_its_runs_for_their_exports_and_keeps_their_newest_counts(hand_runs, monkeypatch):
    """While the runs sample, each view asks runs A1 and B1 (never the others), without waiting, for proseg's export of
    the view's box with their next counted frame (Runs.request_where; a refusal is let be), looks at their newest
    counted frame only (a newer frame without proseg's count, or one whose count was let go, is skipped), and keeps each
    one's spool to its newest SPOOL_KEPT counted files; nothing is asked before the box is known or once the runs are
    done."""
    from test_deeper_where import layout, placed
    layers = layout()[None]
    runs = hand_runs(p=(layers, placed(), {}), t1=(layers, placed(), {}), t2=(layers, placed(), {}),
                     t3=(layers, placed(), {}))
    tests, test = hand_test(runs, [{"x": 119.5, "y": 60.5, "gene": "EDGE", "p": 0.9}], A1="t1", B1="t2", A2="t3")
    asked, pruned = [], []

    def request_where(run_id, box, wait_s=2.0):
        asked.append((run_id, list(box), wait_s))
        if run_id == "t2":
            raise RunError("This run is not active", 409)
        return None
    files = {run_id: [f"/spool/{run_id}/{seq:06d}.counts.zst" for seq in (3, 1, 2)] for run_id in ("t1", "t2", "t3")}
    monkeypatch.setattr(runs, "request_where", request_where, raising=False)
    monkeypatch.setattr(runs, "spool", SimpleNamespace(files=lambda run_id: list(files[run_id]),
                                                       prune=lambda run_id, keep: pruned.append((run_id, keep))),
                        raising=False)
    views = tests.view("p")["runs"]
    assert [v["role"] for v in views] == ["A1", "B1"]
    assert asked == [("t1", test.where_box, 0), ("t2", test.where_box, 0)]
    assert pruned == [(run_id, {f"/spool/{run_id}/000002.counts.zst", f"/spool/{run_id}/000003.counts.zst"})
                      for run_id in ("t1", "t2")]
    # A newer frame without its count, and one whose count was let go: the newest counted one stays shown.
    run = runs.get("t1")
    newer = add_frame(run, layers, 60)
    (run.live / newer["counts_file"]).unlink()
    with run.lock:
        run.frames.append({**run.frames[-1], "seq": 2, "iter": 70, "file": newer["file"], "counts_file": None})
    assert T.newest_counted(run)["seq"] == 0
    assert [v["seq"] for v in tests.view("p")["runs"]] == [0, 0]
    # No box yet: nothing asked; the runs done (the test comparing): nothing asked either.
    asked.clear()
    test.where_box = None
    tests.view("p")
    test.where_box, test.state = [100.0, 50.0, 140.0, 90.0], "comparing"
    tests.view("p")
    assert asked == []


def test_a_view_of_the_largest_area_is_quick(hand_runs, capsys):
    """The corner view's cost on a test run of the largest area a test re-runs (a 500 µm square, as a detail run's grid
    of 560 x 560 at 1 µm with 4 layers, decoded whole; 10 µm cells; 1.33 transcripts per µm², as in a dense sample;
    400 suspects around the cell), each frame with proseg's count (read for the cell's keep and its number): the first
    view (the run's transcript index built once), a new frame (one decode: the best of three), the same frame again
    (kept). Budgets: a new frame within 150 ms (the page asks at most 3 times a second, for two runs), a kept one within
    20 ms."""
    import gzip
    from sopa_companion.engine.deeper_where import DeeperWhere
    from test_deeper_where import GENES, RECORD, X0, Y0
    side = 560
    j, i = np.mgrid[0:side, 0:side]
    plane = ((j // 10) * (side // 10) + i // 10 + 1).astype(np.uint32)
    plane[(i % 10 == 0) | (j % 10 == 0)] = 0                  # a column of background between cells
    layers = np.stack([plane, np.roll(plane, 1, 1), plane, np.roll(plane, 1, 0)])
    cells = [f"c{k}" for k in range(int(plane.max()))]
    rng = np.random.default_rng(0)
    n = int(1.33 * 500 * 500)
    records = np.zeros(n, RECORD)
    records["x"], records["y"] = rng.uniform(X0 + 30, X0 + 530, n), rng.uniform(Y0 + 30, Y0 + 530, n)
    records["z"], records["gene"] = rng.uniform(0, 1, n), rng.integers(0, len(GENES), n)
    runs = hand_runs(big=(layers, records[:10], {"cells": cells}))
    run = runs.get("big")
    (run.live / "transcripts.bin.gz").write_bytes(gzip.compress(records.tobytes(), compresslevel=1))
    count_frame(run.live, dict(run.frames[0]))
    run.read_new()
    where = DeeperWhere(runs)
    cell = int(plane[285, 285]) - 1
    near = records[(np.abs(records["x"] - (X0 + 285)) < 20) & (np.abs(records["y"] - (Y0 + 285)) < 20)][:400]
    suspects = [{"x": round(float(r["x"]), 2), "y": round(float(r["y"]), 2), "gene": GENES[r["gene"]], "p": 0.5}
                for r in near]
    tests = T.QcTests(runs, None, None, where=where)
    test = T.Test("p", pool(seq=0, seed=cell, seed_name=cells[cell], view_suspects=tuple(suspects), view_sources=(),
                            area=area("square", (X0 + 30, Y0 + 30, X0 + 530, Y0 + 530))),
                  "prior-weight", 0.5, 2.0, time.monotonic, runs_per_side=3)
    test.runs["A1"] = "big"
    test.crop = T.crop_around(where.footprints("big", 0, [cell])[cell], None, test.area.box)
    test.where_box = T.view_box(test.crop, test.suspects, test.area.box)
    began = time.perf_counter()
    first = tests._view(test, "A1", run, T.newest_counted(run))
    cold = (time.perf_counter() - began) * 1e3
    times = []
    for step in (1, 2, 3):
        frame = add_frame(run, np.roll(layers, step, axis=2), 40 + step)
        began = time.perf_counter()
        view = tests._view(test, "A1", run, frame)
        times.append((time.perf_counter() - began) * 1e3)
    began = time.perf_counter()
    assert tests._view(test, "A1", run, frame) is view
    kept = (time.perf_counter() - began) * 1e3
    # (The suspects lie within 20 µm of the cell's centre: some of them in it, as proseg counts them.)
    held = [h for h in first["held"] if h is not None]
    assert len(suspects) == 400 and 0 < sum(h > 0 for h in held) < 400 and set(held) <= {0.0, KEEP_VIEW}
    _, dest, _, _ = destinations(run.live, dict(run.frames[0]))
    assert first["cell_total"] == int(np.floor(KEEP_VIEW * (dest == cell + 1).sum() + 0.5)) > 50
    assert (view["w"], view["h"]) == (48, 48) and view["cell_total"] > 50
    size = len(json.dumps(view, separators=(",", ":")))
    with capsys.disabled():
        print(f"\n  test view of a 500 µm square: first {cold:.0f} ms (index built), new frame "
              f"{', '.join(f'{t:.0f}' for t in times)} ms, kept {kept:.2f} ms; {size / 1024:.1f} KB a run")
    assert min(times) <= 150 and kept <= 20


def test_a_poll_that_comes_late_never_replaces_the_final_views(hand_runs, monkeypatch):
    """A poll of the view takes run A1's newest counted frame (iteration 40) while the runs sample, and reaches the
    views only after the answer kept the runs' last frames' views (iteration 450). Those are final (marked under the
    view lock): the poll leaves them. Once the runs are deleted, the views kept are still the last frames', which the answer
    read (the poll used to leave iteration 40 there for good)."""
    from test_deeper_where import layout, placed
    layers = layout()[None]
    runs = hand_runs(p=(layers, placed(), {}), t1=(layers, placed(), {}), t2=(layers, placed(), {}))
    first, second = runs.get("t1"), runs.get("t2")
    tests, test = hand_test(runs, [{"x": 119.5, "y": 60.5, "gene": "EDGE", "p": 0.9}], A1="t1", B1="t2")
    taken, go = threading.Event(), threading.Event()
    real = tests._view

    def late_view(*args, **kwargs):                         # the late poll: its frame taken, the views reached late
        if threading.current_thread().name == "late-poll":
            taken.set()
            assert go.wait(5)
        return real(*args, **kwargs)
    monkeypatch.setattr(tests, "_view", late_view)
    asked = answered(monkeypatch)
    replies = []
    poll = threading.Thread(target=lambda: replies.append(tests.view("p")), name="late-poll")
    poll.start()
    try:
        assert taken.wait(5)
        for run in (first, second):                         # both runs reach the end of their schedule meanwhile
            add_frame(run, layers, 450)
        test.state = "comparing"
        result = tests._answer(test)
    finally:
        go.set()
        poll.join(5)
    assert not poll.is_alive() and len(replies) == 1 and result is asked["result"]
    for run_id in ("t1", "t2"):                             # the test's end: its runs go
        runs.delete(run_id)
    test.state = "done"
    kept = tests.view("p")["runs"]
    assert [(view["role"], view["side"], view["iter"]) for view in kept] == [("A1", "A", 450), ("B1", "B", 450)]
    assert replies[0]["runs"] == kept and asked["answer"][5]["at_iter"] == 450
    assert [view["held"] for view in kept] == [[KEEP_VIEW], [KEEP_VIEW]]


def test_while_a_test_runs_a_runs_view_only_moves_forward(hand_runs):
    """Two polls of the view at once, the one with the newer frame (iteration 100)
    first: the other (40), coming last, leaves the newer view kept, and gives it; a frame newer still replaces it."""
    from test_deeper_where import layout, placed
    layers = layout()[None]
    runs = hand_runs(p=(layers, placed(), {}), t1=(layers, placed(), {}))
    run = runs.get("t1")
    tests, test = hand_test(runs, A1="t1")
    older = dict(run.frames[0])
    newer = tests._view(test, "A1", run, add_frame(run, layers, 100))
    assert (newer["seq"], newer["iter"]) == (1, 100)
    assert tests._view(test, "A1", run, older) is newer and test.views["A1"]["view"] is newer
    assert tests.view("p")["runs"] == [newer]
    assert tests._view(test, "A1", run, add_frame(run, layers, 120))["iter"] == 120


def test_a_view_of_a_run_made_again_is_never_kept(hand_runs):
    """Once the test made its runs again (without factorisation: `_rerun`), a poll that took the old run's frame and
    came late leaves nothing: only the test's own run of the role is shown, whatever its frames."""
    from test_deeper_where import layout, placed
    layers = layout()[None]
    runs = hand_runs(p=(layers, placed(), {}), t1=(layers, placed(), {}), t2=(layers, placed(), {}))
    old, new = runs.get("t1"), runs.get("t2")
    tests, test = hand_test(runs, A1="t2")                  # A1 was t1, made again as t2
    assert tests._view(test, "A1", old, add_frame(old, layers, 300)) is None and test.views == {} and test.names == {}
    kept = tests._view(test, "A1", new, dict(new.frames[0]))
    assert (kept["role"], kept["iter"]) == ("A1", 40) and test.names["A1"]["run"] == "t2"
    assert tests._view(test, "A1", old, add_frame(old, layers, 400)) is kept and test.views["A1"]["view"] is kept


def test_a_suspect_where_another_transcript_lies_is_told_by_its_depth_or_left_unsure(hand_runs, monkeypatch):
    """Two EDGE transcripts at one x, y in a detail run, A's (z 0.1) and B's (z 0.9: B
    holds A's two right columns in the top layer), alike in the parent and both test runs. The pool keeps each suspect
    with its rank by depth among the where reply's points of its gene there, and how many (of): A's, rank 0 of 2, is
    found in each run and proseg counts it for A (the nearest in the plane, B's, was taken before: no change read as
    one); B's, rank 1, is not A's. Depth as the reply gives it: two at z 0.2501 (B's, layer 1) and 0.2499 (A's, layer 0)
    both show 0.25, so the page ranks them in the reply's order, and so does the run. One there without its rank, or
    of a count the run does not hold, is unsure: no share (null), 1 in the view's unsure, never counted. One alone
    needs no rank. The answer's final views are the same."""
    from sopa_companion.engine.deeper_where import DeeperWhere
    from test_deeper_where import GENES, RECORD, layout
    layers = np.stack([layout()] * 4)
    layers[3, 10:20, 18:20] = 2
    layers[1, 16, 13] = 2                                   # B's in layer 1 at (113.5, 66.5)
    edge = GENES.index("EDGE")
    records = np.array([(118.5, 64.25, 0.9, edge), (118.5, 64.25, 0.1, edge), (112.5, 62.5, 0.5, edge),
                        (113.5, 66.5, 0.2501, edge), (113.5, 66.5, 0.2499, edge)], RECORD)
    runs = hand_runs(**{run_id: (layers, records, {}) for run_id in ("p", "t1", "t2")})
    # The suspects as the pool keeps them from the parent's where reply: of the points of a gene at one x, y (to 0.01
    # µm, whoever holds them), how many, and each one's place among them by depth (a stable sort: ties in its order).
    where = json.loads(DeeperWhere(runs).reply("p", 0, 0, [1], ["EDGE"], []))
    points = where["points"]
    assert (points["x"], points["z"], points["owner"]) == \
        ([118.5, 118.5, 112.5, 113.5, 113.5], [0.9, 0.1, 0.5, 0.25, 0.25], [2, 1, 1, 2, 1])
    places, sent = list(zip(points["gene"], points["x"], points["y"])), []
    for k, (gene, x, y) in enumerate(places):
        there = sorted((j for j, place in enumerate(places) if place == (gene, x, y)), key=lambda j: points["z"][j])
        suspect = {"x": x, "y": y, "gene": where["genes"][gene], "p": 0.9}
        sent.append({**suspect, "rank": there.index(k), "of": len(there)} if len(there) > 1 else suspect)
    a, b = {"x": 118.5, "y": 64.25, "gene": "EDGE", "p": 0.9}, {"x": 113.5, "y": 66.5, "gene": "EDGE", "p": 0.9}
    alone = {"x": 112.5, "y": 62.5, "gene": "EDGE", "p": 0.9}
    assert sent == [{**a, "rank": 1, "of": 2}, {**a, "rank": 0, "of": 2}, alone, {**b, "rank": 0, "of": 2},
                    {**b, "rank": 1, "of": 2}]
    sent += [a, {**a, "rank": 0, "of": 3}]                  # no rank, or a count the runs do not hold
    tests, test = hand_test(runs, sent, A1="t1", B1="t2")
    _, dest, _, _ = destinations(runs.get("t1").live, dict(runs.get("t1").frames[0]))
    assert dest.tolist() == [2, 1, 1, 2, 1]                 # proseg's own assignment: the voxel's label
    views = tests.view("p")["runs"]
    assert [view["role"] for view in views] == ["A1", "B1"]
    for view in views:
        assert view["held"] == [0.0, KEEP_VIEW, KEEP_VIEW, 0.0, KEEP_VIEW, None, None], view["role"]
        assert view["unsure"] == [0, 0, 0, 0, 0, 1, 1], view["role"]
    answered(monkeypatch, at=40)
    test.state = "comparing"
    tests._answer(test)
    assert tests.view("p")["runs"] == views and all(test.views[role]["final"] for role in ("A1", "B1"))
