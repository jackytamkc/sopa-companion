"""Test a setting here through its API (engine/qc_test.py QcTests behind server/app.py: GET /runs/{id}/qc/test/pool,
POST, GET and DELETE /runs/{id}/qc/test, GET /runs/{id}/qc/test/view, and "qc_test" {state, progress} in the run
detail and its pushes), with the real Runs and the fake proseg (tests/fake_proseg.py) on a sample of its own: 700 x 700
µm without an image, so that a region re-run as it is (300 to 500 µm), a 500 µm square around a cell in a larger one,
and the regions a test was not measured on (narrow: a crop; small) can all be drawn.

The pool comes from tests/fake_pools.py with this file's `make` (the real pool's own tests are tests/test_qc_pool.py),
except in the pool route's tests, which keep the app's own. The fake world (FAKE_PROSEG_WORLD) gives every run the same
cells and transcripts, and moves the pool's suspects out of their cells in the runs that carry the tested value only, so
that the real readout and rule (engine/qc_readout.py, engine/qc_rule.py) answer "clearly fewer" or "not clear" as
planted. A test's own parts (its runs followed, stopped and made again, its views, its checks) are tested in
tests/test_qc_test_engine.py. The last tests hold README.md and docs/ARCHITECTURE.md to what the engine decides: where
and how a test runs, and which of its runs the corner draws."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from fake_pools import FakePools
from sopa_companion.engine import qc_constants, qc_pool, qc_readout, qc_rule, qc_rules
from sopa_companion.engine import qc_test as T
from sopa_companion.engine.config import ROOT
from sopa_companion.engine.qc import QcError
from sopa_companion.engine.runs import ACTIVE, RunError

BASE = "http://127.0.0.1:8799"
WS = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
PROGRESS = {"iter", "of", "runs", "done", "sides", "marks", "phase", "retry"}         # a test's progress (the run detail)
RECORD = ["id", "state", "cell", "cell_name", "flag", "name", "value", "current", "tried", "start", "area", "square",
          "runs_per_side", "threads", "runs", "progress", "pool", "message", "result", "was_running", "seconds", "hold",
          "no_factorization", "reruns", "estimate_s", "waiting", "reasons", "caveats", "row"]   # the {test} record
POOL_REPLY = ["v", "state", "run", "cell", "cell_name", "seq", "iter", "count", "model", "constants", "start",
              "runs_per_side", "k", "area", "offer", "reasons", "pattern", "line", "line_shown", "this_cell", "pool",
              "estimate_s", "ms", "message", "waits", "caveats", "row", "hold", "settings"]


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


# ---- the sample --------------------------------------------------------------------------------------------------------
def big_bundle(root: Path) -> Path:
    """conftest's Xenium bundle (write_xenium_bundle's schema) over 700 x 700 µm, without its image: the sample's extent
    is then its transcripts' own maxima (engine/samples.py), 700 x 700 µm from the one transcript placed at the corner.
    4,000 transcripts, each in the cell its 40 µm block names (c{x // 40}_{y // 40}-1)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder = root / "output-XETG00000__0000000__BIGREGION-1__20260101__000000"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(0, 700, n).astype(np.float32)
    y = rng.uniform(0, 700, n).astype(np.float32)
    x[0] = y[0] = 700.0
    table = pa.table({
        "transcript_id": pa.array(np.arange(n, dtype=np.uint64) + 281474976710656),
        "cell_id": pa.array([f"c{int(a // 40)}_{int(b // 40)}-1" for a, b in zip(x, y)], pa.large_string()),
        "overlaps_nucleus": pa.array(rng.integers(0, 2, n).astype(np.uint8)),
        "feature_name": pa.array([f"G{i % 7}" for i in range(n)], pa.large_string()),
        "x_location": pa.array(x), "y_location": pa.array(y),
        "z_location": pa.array(rng.uniform(5, 15, n).astype(np.float32)),
        "qv": pa.array(np.full(n, 40, np.float32)),
        "fov_name": pa.array(["A1"] * n, pa.large_string()),
        "nucleus_distance": pa.array(np.zeros(n, np.float32)),
        "codeword_index": pa.array(np.zeros(n, np.int32)),
        "codeword_category": pa.array(["predesigned_gene"] * n, pa.large_string()),
        "is_gene": pa.array(np.ones(n, bool)),
    })
    pq.write_table(table, folder / "transcripts.parquet")
    (folder / "experiment.xenium").write_text(json.dumps({"region_name": "BIGREGION-1", "num_cells": 289}))
    return folder


# ---- the fake world ----------------------------------------------------------------------------------------------------
# What every fake proseg run of a test that asks for it reads instead of the tiny sample (tests/fake_proseg.py
# FAKE_PROSEG_WORLD): the sample's 700 µm on a grid of 10 µm voxels, 2 layers, every frame alike; cells of 3 x 3 voxels
# every 4 (c{a}_{b}-1 from (40a, 40b) on, a and b 0 to 16, as the sample names its own), a voxel of no cell between
# them. The clicked cell c8_8-1 (its centre at (335, 335)) and the 5 cells nearest it hold 40 suspects each: transcripts
# of FROM in their right-hand column, from the cell on their right (their giver); each suspect has 3 own controls and
# its cell 50 more of its own RNA (OWN); every other cell holds 5 OWN. A run that carries --cell-compactness 0.02 has
# every suspect one voxel further right: in no cell.
VOXEL, GRID, CELLS, BLOCK, PITCH = 10.0, 70, 17, 3, 4
GENES, FROM, OWN = ["FROM", "OWN"], 0, 1
SUSPECTS, MATCHED, UNIFORM, FILLER = 40, 3, 50, 5
MOVED = 0.02                                   # the compactness at which the world moves the suspects (side B's)


def number(a: int, b: int) -> int:
    """The parent cell number (label - 1) of the world's cell c{a}_{b}-1."""
    return b * CELLS + a


def centre(cell: int) -> tuple[float, float]:
    """A cell's centre as proseg-live gives it (its columns' mean + 0.5, times the voxel)."""
    a, b = cell % CELLS, cell // CELLS
    return (PITCH * a + 1.5) * VOXEL, (PITCH * b + 1.5) * VOXEL


SEED = number(8, 8)                             # the cell every test clicks
POOLED = [number(a, b) for a, b in ((8, 8), (8, 7), (7, 8), (9, 8), (8, 9), (9, 9))]


class World:
    """The fake world's file content (`json`) and which of its transcripts each pooled cell's targets and controls are
    (row indices of transcripts.bin.gz: `targets`, `matched` (3 per target), `uniform`), with each pooled cell's giver."""

    def __init__(self):
        rng = np.random.default_rng(20261029)
        plane = np.zeros((GRID, GRID), np.int64)
        for b in range(CELLS):
            for a in range(CELLS):
                plane[PITCH * b:PITCH * b + BLOCK, PITCH * a:PITCH * a + BLOCK] = number(a, b) + 1
        self.names = [f"c{c % CELLS}_{c // CELLS}-1" for c in range(CELLS * CELLS)]
        self.rows: list[tuple[float, float, float, int]] = []
        self.targets, self.matched, self.uniform, self.giver = {}, {}, {}, {}

        def put(cell: int, columns, gene: int) -> int:
            a, b = cell % CELLS, cell // CELLS
            column, row = PITCH * a + int(rng.choice(list(columns))), PITCH * b + int(rng.integers(0, BLOCK))
            x = np.float32(VOXEL * column + rng.uniform(0.5, VOXEL - 0.5))
            y = np.float32(VOXEL * row + rng.uniform(0.5, VOXEL - 0.5))
            self.rows.append((float(x), float(y), float(np.float32(rng.uniform(0.05, 0.95))), gene))
            return len(self.rows) - 1

        for cell in POOLED:
            self.giver[cell] = cell + 1
            self.targets[cell] = [put(cell, [BLOCK - 1], FROM) for _ in range(SUSPECTS)]
            self.matched[cell] = [[put(cell, range(BLOCK - 1), OWN) for _ in range(MATCHED)] for _ in range(SUSPECTS)]
            self.uniform[cell] = [put(cell, range(BLOCK - 1), OWN) for _ in range(UNIFORM)]
        for cell in range(CELLS * CELLS):
            if cell not in self.targets:
                for _ in range(FILLER):
                    put(cell, range(BLOCK), OWN)
        self.ids = [100_000 + 3 * row for row in range(len(self.rows))]        # never the row: the join is by id
        suspects = np.array([self.rows[r][:2] for rows in self.targets.values() for r in rows])
        apart = np.abs(suspects[:, None, :] - suspects[None, :, :]).max(axis=2) + np.eye(len(suspects)) * 1e9
        assert apart.min() > 0.05, "two suspects at one place: the views could not tell them apart"
        self.json = {"cells": self.names, "genes": GENES, "nx": GRID, "ny": GRID, "nz": 2, "x0": 0.0, "y0": 0.0,
                     "voxel": VOXEL, "labels": plane.ravel().tolist(),
                     "transcripts": [[tid, *row] for tid, row in zip(self.ids, self.rows)],
                     "moves": {"flag": "cell-compactness", "value": MOVED, "dx": 1,
                               "rows": sorted(r for rows in self.targets.values() for r in rows)}}


WORLD = World()


def make(meta: dict, cell: int) -> tuple[qc_pool.Pool, qc_pool.Matched]:
    """The pool of `cell` in the fake world, for the parent `meta` (its region, its end, and its start as qc_rules reads
    it when called), as engine/qc_pool.py fixes it (the clicked cell, then the pooled cells nearest it, ties by cell
    number, until their suspects reach the start's k; then RESERVE_CELLS more) and as match_input finds it in the shared
    input: every target and control found. The world's other cells hold no pool (no_suspects); a cell past its cells:
    404."""
    if not 0 <= cell < len(WORLD.names):
        raise RunError("Unknown cell", 404)
    if cell != SEED:
        raise qc_pool.OutsideDomain([qc_pool.reason("no_suspects")])
    start, measured = qc_rules.start_of(meta), qc_pool.row_of(meta)       # the row it is judged with
    area = qc_pool.area_of(meta["region"], centre(SEED))
    k = qc_rule.constants_for(*measured)["k"]
    distance = {c: math.dist(centre(c), centre(SEED)) for c in POOLED}
    order = sorted(POOLED, key=lambda c: (c != SEED, distance[c], c))
    members, total = [], 0
    for c in order:
        if total >= k:
            break
        members.append(c)
        total += len(WORLD.targets[c])
    cells = order[:len(members) + qc_pool.RESERVE_CELLS]
    names = list(dict.fromkeys(WORLD.names[c] for c in [*cells, *(WORLD.giver[c] for c in cells)]))
    at = {name: i for i, name in enumerate(names)}
    rows, found = [], []          # (Items fields) and (qc_pool.Item fields) of every target, then control, in order

    def add(kind: int, cell_: int, giver: int | None, target: int, gene: int, row: int):
        rows.append(row)
        found.append((kind, cell_, giver, target, gene))
    for c in cells:
        for row in WORLD.targets[c]:
            add(qc_readout.SUSPECT, c, WORLD.giver[c], -1, FROM, row)
    item_of = {row: i for i, row in enumerate(rows)}
    for c in cells:
        for row, controls in zip(WORLD.targets[c], WORLD.matched[c]):
            for control in controls:
                add(qc_readout.MATCHED, c, WORLD.giver[c], item_of[row], OWN, control)
    for c in cells:
        for row in WORLD.uniform[c]:
            add(qc_readout.UNIFORM, c, None, -1, OWN, row)
    kinds = np.array([f[0] for f in found], np.int8)
    items = qc_readout.Items(
        kind=kinds, cell=np.array([at[WORLD.names[f[1]]] for f in found], np.int32),
        giver=np.array([qc_readout.NO_GIVER if f[2] is None else at[WORLD.names[f[2]]] for f in found], np.int32),
        target=np.array([f[3] for f in found], np.int64), gene=np.array([f[4] for f in found], np.int32),
        transcript_id=np.array([WORLD.ids[r] for r in rows], np.uint64),
        x=np.array([WORLD.rows[r][0] for r in rows], np.float32), y=np.array([WORLD.rows[r][1] for r in rows], np.float32),
        cell_names=tuple(names), gene_names=tuple(GENES))
    items.check()
    word = {qc_readout.SUSPECT: "target", qc_readout.MATCHED: "matched", qc_readout.UNIFORM: "uniform"}
    pool_items = tuple(qc_pool.Item(kind=word[f[0]], cell=f[1], target=None if f[3] < 0 else f[3], giver=f[2],
                                    gene=GENES[f[4]], record=r, x=WORLD.rows[r][0], y=WORLD.rows[r][1],
                                    zn=WORLD.rows[r][2], near=None if f[2] is None else True,
                                    p=0.9 if f[0] == qc_readout.SUSPECT else None, primary=f[0] == qc_readout.SUSPECT,
                                    start_class="OWN" if f[0] == qc_readout.SUSPECT else None)
                       for f, r in zip(found, rows))
    done = meta.get("schedule_done") or {}
    pool = qc_pool.Pool(
        v=qc_pool.POOL_V, run=meta["id"], seq=done.get("seq"), iter=done.get("iter"), model="f" * 32,
        constants=qc_rule.CONSTANTS_ID, start=start, area=area, k=k, seed=SEED, seed_name=WORLD.names[SEED],
        cells=tuple(cells), members=len(members), distance_um=tuple(round(distance[c], 1) for c in cells),
        names={c: WORLD.names[c] for c in {*cells, *(WORLD.giver[c] for c in cells)}}, items=pool_items,
        suspects=total, available=None, record_duplicates=0,
        this_cell={"likely": SUSPECTS, "suspects": SUSPECTS, "pattern": "own",
                   "line": {"likely": SUSPECTS, "own": SUSPECTS, "none": 0, "neighbour": 0, "other": 0}},
        view_suspects=tuple({"x": round(WORLD.rows[r][0], 2), "y": round(WORLD.rows[r][1], 2), "gene": "FROM", "p": 0.9}
                            for r in WORLD.targets[SEED]),
        view_sources=(WORLD.giver[SEED],), row=measured)
    counts = {kind: {"ok": int((kinds == code).sum()), "not_found": 0, "unsure": 0}
              for kind, code in (("target", qc_readout.SUSPECT), ("matched", qc_readout.MATCHED),
                                 ("uniform", qc_readout.UNIFORM))}
    order_ = qc_rule.PoolOrder(seed=at[WORLD.names[SEED]], cells=tuple(at[WORLD.names[c]] for c in cells),
                               distance_um=tuple(distance[c] for c in cells), parent_cells=tuple(cells))
    return pool, qc_pool.Matched(pool=pool, items=items, order=order_, counts=counts, input="", seconds=0.01)


# ---- the app -----------------------------------------------------------------------------------------------------------
class Switch:
    """Clusters & UMAP's switch, as QcTests turns it off for each run it makes (engine/analysis.py configure)."""

    def __init__(self):
        self.calls = []

    def configure(self, run_id, enabled=None, seq=None, params=None):
        self.calls.append((run_id, enabled))


@pytest.fixture(scope="session")
def big_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("big-sample")
    big_bundle(root)
    return root


def fake_world(tmp_path: Path) -> Path:
    """The fake world's file (what FAKE_PROSEG_WORLD names: tests/fake_proseg.py) written in `tmp_path`: its path."""
    path = tmp_path / "world.json"
    path.write_text(json.dumps(WORLD.json))
    return path


@pytest.fixture
def world(tmp_path, monkeypatch) -> Path:
    """The fake world (fake_world), for every fake proseg run the test starts (the parent's too)."""
    path = fake_world(tmp_path)
    monkeypatch.setenv("FAKE_PROSEG_WORLD", str(path))
    return path


def started(config, root, *, fake_pools: bool):
    """The app with the real Runs and the fake proseg on the 700 µm sample alone (conftest's settings otherwise); its
    tests' pools tests/fake_pools.py's over `make` (`fake_pools`), set once the app runs, and a stand-in for Clusters &
    UMAP's switch."""
    from sopa_companion.server import create_app
    config["paths"]["sample_roots"] = [str(root)]
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        tests = application.state.qc_tests
        if fake_pools:
            tests.pools = FakePools(application.state.runs, make)
        tests.analysis = Switch()
        yield application, client


@pytest.fixture
def api(config, big_root):
    yield from started(config, big_root, fake_pools=True)


@pytest.fixture
def api_own_pools(config, big_root):
    yield from started(config, big_root, fake_pools=False)


def parent(application, client, box, *, options=None, play=True) -> str:
    """A run of the 700 µm sample over `box`, with proseg's default settings unless `options` and nothing set live:
    played at full speed (`play`) until proseg pauses it at the end of its schedule (at_end), else left paused on its
    first frame."""
    sample = application.state.runs.samples.list()[0]["id"]
    made = client.post("/api/v1/runs", json={"sample": sample, "box": box, "options": options or {}}, headers=auth())
    assert made.status_code == 200, made.text
    run_id = made.json()["id"]
    if play:
        for body in ({"action": "speed", "speed": "max"}, {"action": "resume"}):
            assert client.post(f"/api/v1/runs/{run_id}/control", json=body, headers=auth()).status_code == 200
    return run_id


def at_end(application, *run_ids, timeout=90.0):
    """Until each run waits paused at the end of its schedule (meta schedule_done: proseg holds there)."""
    runs = application.state.runs

    def ended(run_id):
        meta = runs.get(run_id).meta
        return meta.get("state") == "paused" and (meta.get("schedule_done") or {}).get("iter") is not None
    wait_for(lambda: all(ended(run_id) for run_id in run_ids), timeout=timeout)


def ask(client, run_id, **body):
    """POST a test of the clicked cell: compactness at MOVED unless `body` says otherwise."""
    return client.post(f"/api/v1/runs/{run_id}/qc/test",
                       json={"cell": SEED, "flag": "cell-compactness", "value": MOVED, **body}, headers=auth())


def ended(client, run_id, timeout=120.0) -> dict:
    """The run's test once it ended (done, failed or cancelled)."""
    url = f"/api/v1/runs/{run_id}/qc/test"
    wait_for(lambda: client.get(url, headers=auth()).json()["state"] not in T.ON_ITS_WAY, timeout=timeout,
             interval=0.1)
    return client.get(url, headers=auth()).json()


def all_made(client, run_id, timeout=30.0) -> dict:
    """The run's test's runs once every one is made: {role: run id}."""
    url = f"/api/v1/runs/{run_id}/qc/test"
    wait_for(lambda: all(client.get(url, headers=auth()).json()["runs"].values()), timeout=timeout, interval=0.05)
    return client.get(url, headers=auth()).json()["runs"]


def flags(command: list[str]) -> dict:
    """A command line's --flag: value (True for a flag without one)."""
    found = {}
    for n, token in enumerate(command):
        if token.startswith("--"):
            following = command[n + 1] if n + 1 < len(command) else None
            found[token] = following if following is not None and not following.startswith("--") else True
    return found


def spy_commands(application, monkeypatch) -> dict:
    """Every proseg command line the engine starts: {"runs": {run id: [{command, meta (as then), own_input (whether
    the run had its own input/transcripts.parquet)}]} (a run proseg retried has two), "inputs": {path: its table as
    the first command found it}}."""
    import pyarrow.parquet as pq
    runs = application.state.runs
    real = runs._command
    seen = {"runs": {}, "inputs": {}}

    def command(run, *args, **kwargs):
        line = real(run, *args, **kwargs)
        seen["runs"].setdefault(run.meta["id"], []).append(
            {"command": list(line), "meta": json.loads(json.dumps(run.meta)),
             "own_input": (run.dir / "input" / "transcripts.parquet").exists()})
        path = line[line.index("--xenium") + 1]
        if path not in seen["inputs"]:
            seen["inputs"][path] = pq.read_table(path) if Path(path).is_file() else None
        return line
    monkeypatch.setattr(runs, "_command", command)
    return seen


def spy_controls(application, monkeypatch) -> list[tuple]:
    """Every set and resume of a test's run as the engine asks for it: [(role, action, values, its state then, whether
    it had sampled no iteration then)]."""
    runs = application.state.runs
    real = runs.control
    seen = []

    def control(run_id, action, speed=None, values=None):
        run = runs.get(run_id)
        if action in ("set", "resume") and run.meta.get("qc_test"):
            with run.lock:
                unsampled = not run.frames or run.frames[-1].get("iter") == 0
            seen.append((run.meta["qc_test"]["role"], action, dict(values or {}), run.meta.get("state"), unsampled))
        return real(run_id, action, speed, values)
    monkeypatch.setattr(runs, "control", control)
    return seen


def spy_states(application, monkeypatch) -> list[dict]:
    """The test's runs and own input as each state of a test is set: [{state, runs: {role: {state, metadata (its
    output/transcript-metadata.parquet there), save_defaults} | None (gone)}, input (the test's own folder there)}]."""
    tests, runs = application.state.qc_tests, application.state.runs
    real = tests._set
    marks = []

    def set_state(test, **values):
        if values.get("state") is not None:
            found = {}
            for role, run_id in dict(test.runs).items():
                try:
                    run = runs.get(run_id) if run_id is not None else None
                except RunError:
                    run = None
                found[role] = None if run is None else {
                    "state": run.meta.get("state"), "save_defaults": run.meta.get("save_defaults", "unset"),
                    "metadata": (run.dir / "output" / "transcript-metadata.parquet").is_file()}
            marks.append({"state": values["state"], "runs": found, "input": (tests.inputs / test.id).exists()})
        return real(test, **values)
    monkeypatch.setattr(tests, "_set", set_state)
    return marks


class Watch:
    """The engine's runs looked at every millisecond while a test goes: the runs listed without "qc_test" besides the
    parents (the page would list them), and the most test runs active at once."""

    def __init__(self, runs, parents):
        self.runs, self.parents = runs, set(parents)
        self.untagged, self.most, self.going = set(), 0, True
        self.thread = threading.Thread(target=self._look, daemon=True)
        self.thread.start()

    def _look(self):
        while self.going:
            listed = list(self.runs.runs.values())
            self.untagged.update(r.meta["id"] for r in listed if r.meta["id"] not in self.parents
                                 and "qc_test" not in r.meta)
            self.most = max(self.most, sum(1 for r in listed if r.meta.get("qc_test")
                                           and r.meta.get("state") in ACTIVE))
            time.sleep(0.001)

    def stop(self) -> "Watch":
        self.going = False
        self.thread.join(5)
        return self


def gone(client, *run_ids) -> bool:
    return all(client.get(f"/api/v1/runs/{run_id}", headers=auth()).status_code == 404 for run_id in run_ids)


# ---- a test's life -----------------------------------------------------------------------------------------------------
def test_a_test_of_a_square_runs_3_and_3_fresh_runs_from_its_own_input_and_answers(api, world, monkeypatch):
    """A parent started from nuclei and cells over the whole 700 µm sample, paused at the end of its schedule: a test of
    compactness 0.02 on the clicked cell re-runs the 500 µm square around it (shifted to stay inside the region) from one
    input of its own, the parent's rows inside the square (half-open, every column and type kept, the parent's ids): 3
    runs as now and 3 with 0.02 alone, all at once, 8 threads each, made paused and given the parent's hold at their
    own scale before their first iteration, then to the end of proseg's schedule and plainly ended (proseg's
    per-transcript file; no --live-hold), each tagged with the test from its first moment, Clusters & UMAP off. Every
    run is done with its file before the test reads them, and gone with the test's input before it says "done"; the
    parent waits at its end throughout. The world moves the suspects out of their cells in the runs with 0.02 only: the
    real readout and rule answer "clearly fewer". Meanwhile a second POST and the parent's resume are refused, its pool
    says busy, and the corner's view follows runs A1 and B1 and keeps their last frames once the runs are gone."""
    application, client = api
    runs, tests = application.state.runs, application.state.qc_tests
    seen, marks = spy_commands(application, monkeypatch), spy_states(application, monkeypatch)
    controls = spy_controls(application, monkeypatch)
    p = parent(application, client, [0, 0, 700, 700])
    at_end(application, p)
    done_at = dict(runs.get(p).meta["schedule_done"])
    parent_input = runs.get(p).dir / "input" / "transcripts.parquet"
    watch = Watch(runs, {p})
    url = f"/api/v1/runs/{p}/qc/test"
    answer = ask(client, p)
    assert answer.status_code == 202, answer.text
    test = answer.json()
    assert list(test) == RECORD
    box, inner = [85.0, 85.0, 585.0, 585.0], [110.0, 110.0, 560.0, 560.0]      # around (335, 335), inside [0, 700]
    assert (test["state"], test["result"], test["message"], test["was_running"]) == ("preparing", None, None, False)
    assert (test["cell"], test["cell_name"], test["flag"], test["name"], test["value"], test["current"], test["tried"]) \
        == (SEED, "c8_8-1", "cell-compactness", "compactness", MOVED, 0.04, True)
    assert (test["start"], test["area"], test["square"]) == ("paired", {"rule": "square", "box": box, "inner": inner}, box)
    assert (test["runs_per_side"], test["threads"], sorted(test["runs"]), test["estimate_s"]) == \
        (3, 8, ["A1", "A2", "A3", "B1", "B2", "B3"], qc_pool.ESTIMATE_S["paired"])
    assert set(test["progress"]) == PROGRESS and (test["progress"]["of"], test["progress"]["runs"]) == (450, 6)
    assert test["progress"]["marks"] == [200, 400] and test["progress"]["retry"] is False
    hold = {"value": 1.0, "effective": 1.0, "scale": 1.0, "seq": done_at["seq"], "given": None}   # 9 transcripts a
    assert test["hold"] == hold                                                            # cell: the weight as set
    assert (test["no_factorization"], test["reruns"]) == (False, 0)
    # On its way: one test at a time, the parent waits, and its pool says why another cannot start.
    again = ask(client, p)
    assert again.status_code == 409
    assert again.json()["detail"] == "A test of this run is on its way: wait for its answer, or cancel it"
    busy = client.get(f"{url}/pool", params={"cell": SEED}, headers=auth()).json()
    assert busy["offer"] is False and busy["reasons"] == [qc_pool.reason("busy", why="test")]
    for action in ("resume", "step"):
        refused = client.post(f"/api/v1/runs/{p}/control", json={"action": action}, headers=auth())
        assert refused.status_code == 409 and "wait for its answer" in refused.json()["detail"]
    assert client.get(f"/api/v1/runs/{p}", headers=auth()).json()["qc_test"]["state"] in T.ON_ITS_WAY
    views = []

    def look():
        found = client.get(f"{url}/view", headers=auth())
        assert found.status_code == 200 and found.headers["cache-control"] == "private, no-cache", found.text
        views.append(found.json())
        return views[-1]["state"] not in T.ON_ITS_WAY
    wait_for(look, timeout=120, interval=0.1)
    test = client.get(url, headers=auth()).json()
    assert test["state"] == "done", test["message"]
    assert watch.stop().untagged == set()
    ids = test["runs"]

    # The input: one, the test's own, the parent's rows inside the square.
    import pyarrow.parquet as pq
    shared = tests.inputs / test["id"] / "transcripts.parquet"
    whole, table = pq.read_table(parent_input), seen["inputs"][str(shared)]
    x, y = whole["x_location"].to_numpy(), whole["y_location"].to_numpy()
    inside = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
    assert [(f.name, f.type) for f in table.schema] == [(f.name, f.type) for f in whole.schema]
    assert table.num_rows == int(inside.sum()) < whole.num_rows
    assert sorted(table["transcript_id"].to_pylist()) == sorted(whole["transcript_id"].filter(inside).to_pylist())
    # The runs: every one as I10 says, B alone with the setting.
    for role, run_id in ids.items():
        (call,) = seen["runs"][run_id]
        line, meta = call["command"], call["meta"]
        given = flags(line)
        assert given["--xenium"] == str(shared) and call["own_input"] is False
        assert given["--nthreads"] == "8" and meta["threads"] == 8
        assert given.get("--cell-compactness") == ("0.02" if role[0] == "B" else None), role
        assert "--live-hold" not in given and "--live-tiled" not in given and "--no-factorization" not in given
        assert (given["--output-transcript-metadata"], given["--output-transcript-metadata-fmt"]) == \
            ("transcript-metadata.parquet", "parquet")
        assert meta["qc_test"] == {"of": p, "role": role, "side": role[0], "test": test["id"]}
        assert {key: meta["input"][key] for key in ("path", "owned", "ids")} == \
            {"path": str(shared), "owned": True, "ids": "own"}
        assert meta["input"]["transcripts"] == table.num_rows and meta["input"]["prepared"]["source"] == "parent"
        assert (meta["speed"], meta["paused"], meta["region"]["box"]) == ("max", True, box)    # paused: its hold first
    # Each given the parent's hold at its own scale (1 here: the value as set) while paused on its first frame, then
    # resumed: before its first iteration.
    assert sorted(controls) == sorted((role, action, values, "paused", True) for role in ids
                                      for action, values in (("set", {"prior-weight": 1.0}), ("resume", {})))
    assert test["hold"] == {**hold, "given": {"scale": 1.0, "as_now": 1.0, "tried": None}} == test["result"]["hold"]
    assert sorted(tests.analysis.calls) == sorted((run_id, False) for run_id in ids.values())
    # Every run done with its file before the test read them; every run and the input gone before it said "done".
    comparing = next(mark for mark in marks if mark["state"] == "comparing")
    assert comparing["runs"] == dict.fromkeys(ids, {"state": "done", "save_defaults": None, "metadata": True})
    assert marks[-1] == {"state": "done", "runs": dict.fromkeys(ids), "input": False}
    assert gone(client, *ids.values()) and not shared.exists()
    assert [r["id"] for r in client.get("/api/v1/runs", headers=auth()).json()] == [p]
    meta = runs.get(p).meta
    assert (meta["state"], meta["schedule_done"], meta.get("continued")) == ("paused", done_at, None)
    assert test["progress"] == {**test["progress"], "iter": 450, "of": 450, "runs": 6, "done": 6, "phase": "reading"}
    assert {"input", "matched", "made", "running", "sampled", "comparing", "done"} <= set(test["seconds"])
    assert test["seconds"]["input"] <= test["seconds"]["made"] <= test["seconds"]["comparing"] <= test["seconds"]["done"]
    assert test["pool"] == {"cells": 4, "suspects": 160, "k": 160, "matched": 6 * SUSPECTS * MATCHED,
                            "uniform": 6 * UNIFORM, "unsure": 0, "not_found": 0}

    # The answer: as now every suspect stays in its cell; with 0.02 every one leaves it, its own RNA stays.
    result = test["result"]
    json.dumps(result, allow_nan=False)
    assert (result["v"], result["verdict"], result["why"], result["rule"], result["start"], result["situation"],
            result["at_iter"]) == (2, "clearly_fewer", None, "excess", "paired", "square", 450)
    assert result["setting"] == {"flag": "cell-compactness", "value": MOVED, "current": 0.04}
    assert result["runs"] == {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]} and result["parent_defaults"] is True
    constants = result["constants"]
    assert (constants["id"], constants["situation"], constants["rule"], constants["k"], constants["runs_per_side"]) == \
        (qc_rule.CONSTANTS_ID, "square", "excess", 160, 3)
    assert constants["rates"]["areas"] == 3 and constants["tau"] == pytest.approx(0.0894, abs=1e-4)
    assert result["come_back"] == {"held": [1.0] * 3, "mean": 1.0, "gate": 0.5, "passed": True}
    fewer = result["fewer"]
    assert (fewer["form"], fewer["a"], fewer["b"], fewer["gap"], fewer["separated"], fewer["fires"],
            fewer["why_not"]) == ("excess", [0.0] * 3, [-1.0] * 3, 1.0, True, True, None)
    assert fewer["suspects"]["mean"] == {"a": 1.0, "b": 0.0} and fewer["own"]["mean"] == {"a": 1.0, "b": 1.0}
    assert fewer["forms"]["suspects"]["fires"] is True and fewer["forms"]["excess"]["fires"] is True
    assert result["cost"] == {"own_same_spot": {"a": 1.0, "b": 1.0, "change": 0.0},
                              "own_all": {"a": 1.0, "b": 1.0, "change": 0.0},
                              "transcripts_per_cell": {"a": 5.0, "b": 5.0, "change": 0.0}, "shrinking": False}
    assert result["went"] == {"A": {"stay": 1.0, "neighbour": 0.0, "other": 0.0, "background": 0.0},
                              "B": {"stay": 0.0, "neighbour": 0.0, "other": 0.0, "background": 1.0}}
    assert [c["cell"] for c in result["pool"]["cells"]] == [SEED, number(8, 7), number(7, 8), number(9, 8)]
    assert (result["pool"]["suspects"], result["pool"]["available"], result["pool"]["k"]) == (160, 240, 160)
    assert result["pool"]["dropped"] == {"unsure": 0, "not_found": 0, "crosscheck_failed": 0, "cell_absent": 0}
    assert result["cell"] == {"cell": SEED, "name": "c8_8-1", "suspects": SUSPECTS, "judged": False}
    assert len(result["points"]["x"]) == len(result["points"]["cell"]) == 160
    assert result["points"]["A"][0] == [1.0, 0.0, 0.0, 0.0] and result["points"]["B"][0] == [0.0, 0.0, 0.0, 1.0]
    per_run = result["reading"]["per_run"]
    assert sorted(per_run) == sorted(ids) and {(r["iter"], r["rows"], r["tx"]) for r in per_run.values()} == \
        {(450, len(WORLD.rows), 5.0)}

    # The corner's view: runs A1 (as now) and B1 (with 0.02), the cell's suspects as the pool keeps them.
    suspects = [dict(s) for s in make(runs.get(p).meta, SEED)[0].view_suspects]
    assert any(view["state"] == "running" and view["runs"] for view in views)
    final = views[-1]
    assert final["suspects"] == suspects and final["state"] == "done"
    assert [(run["role"], run["side"], run["iter"]) for run in final["runs"]] == [("A1", "A", 450), ("B1", "B", 450)]
    x0, y0, x1, y1 = final["box"]
    assert x0 <= 320 and y0 <= 320 and x1 >= 350 and y1 >= 350 and x1 - x0 == y1 - y0 >= 48       # the cell's block
    for view in final["runs"]:
        assert (view["cell"], view["sources"]) == (SEED + 1, [WORLD.giver[SEED] + 1])
        assert view["unsure"] == [0] * SUSPECTS and view["held_by"] in ("exact", "approx")
    # Each view's shares are proseg's: as now its suspects stay in their cell; with 0.02 proseg moved them out, exactly
    # (its export of the view: none goes to the cell) or approximately (the frame's labels, which do not move here, keep
    # them, but proseg counts none of their gene for the cell any more: its keep of it is 0).
    mine = SUSPECTS * (1 + MATCHED) + UNIFORM                  # every transcript of the cell: proseg keeps them all
    for view in [run for found in views for run in found["runs"]] + final["runs"]:
        assert view["held"] == [0.0 if view["side"] == "B" else 1.0] * SUSPECTS, (view["side"], view["iter"])
        # Proseg's count of the cell: less the suspects it moved into the gap beside the cell with 0.02.
        assert view["cell_total"] == mine - (SUSPECTS if view["side"] == "B" else 0), view["side"]
    assert {view["held_by"] for found in views for view in found["runs"]} <= {"exact", "approx"}
    # While the runs sample, each view asks A1 and B1 for proseg's export of its box with their next counted frame:
    # some of their views read it (exact).
    assert any(view["held_by"] == "exact" for found in views if found["state"] in ("preparing", "running")
               for view in found["runs"])
    # Kept after the runs went, until the test is dismissed.
    assert client.get(f"{url}/view", headers=auth()).json() == final
    assert client.get(f"/api/v1/runs/{p}", headers=auth()).json()["qc_test"]["state"] == "done"
    dismissed = client.delete(url, headers=auth())
    assert dismissed.status_code == 200 and dismissed.json()["state"] == "done"
    for path in (url, f"{url}/view"):
        missing = client.get(path, headers=auth())
        assert missing.status_code == 404 and missing.json()["detail"] == "No test for this run"
    assert "qc_test" not in client.get(f"/api/v1/runs/{p}", headers=auth()).json()
    assert client.delete(url, headers=auth()).status_code == 404


def test_a_region_re_run_as_it_is_reads_the_parents_own_input_and_says_when_it_is_not_clear(api, world, monkeypatch):
    """A parent over a 450 µm region (re-run as it is): every run reads the parent's own input (not the test's: its ids
    are the parent's), which is still there after the test. Compactness 0.03 moves nothing in the world: both sides alike,
    "not clear" because the runs with it are not all apart from the runs as now. The page watching the parent hears the
    test with it: at once for each new state, at most once a second for its progress, which carries I9's fields."""
    application, client = api
    runs = application.state.runs
    seen = spy_commands(application, monkeypatch)
    p = parent(application, client, [100, 100, 550, 550])
    at_end(application, p)
    own = runs.get(p).dir / "input" / "transcripts.parquet"
    pushes = []
    with client.websocket_connect("/api/v1/ws", headers=WS) as ws:
        ws.send_json({"op": "watch", "run": p, "since": len(runs.get(p).frames)})
        answer = ask(client, p, value=0.03)
        assert answer.status_code == 202, answer.text
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            event = ws.receive_json()
            if event["ev"] == "run" and event["run"]["id"] == p and "qc_test" in event["run"]:
                pushes.append((time.monotonic(), event["run"]["qc_test"]))
                if pushes[-1][1]["state"] not in T.ON_ITS_WAY:
                    break
    test = client.get(f"/api/v1/runs/{p}/qc/test", headers=auth()).json()
    assert test["state"] == "done", test["message"]
    assert (test["area"], test["square"]) == ({"rule": "as_is", "box": [100.0, 100.0, 550.0, 550.0],
                                               "inner": [125.0, 125.0, 525.0, 525.0]}, None)
    for role, run_id in test["runs"].items():
        (call,) = seen["runs"][run_id]
        assert flags(call["command"])["--xenium"] == str(own) and call["own_input"] is False
        assert {key: call["meta"]["input"][key] for key in ("path", "owned", "ids")} == \
            {"path": str(own), "owned": False, "ids": "own"}
        assert call["meta"]["region"] == runs.get(p).meta["region"]
        assert flags(call["command"]).get("--cell-compactness") == ("0.03" if role[0] == "B" else None)
    assert own.is_file() and gone(client, *test["runs"].values())
    result = test["result"]
    assert (result["verdict"], result["situation"], result["constants"]["rates"]["areas"]) == ("not_clear", "as_is", 2)
    assert result["come_back"]["passed"] is True and result["setting"]["value"] == 0.03
    fewer = result["fewer"]
    assert (fewer["a"], fewer["b"], fewer["gap"], fewer["separated"], fewer["fires"], fewer["why_not"]) == \
        ([0.0] * 3, [0.0] * 3, 0.0, False, False, "apart")
    assert result["went"]["A"] == result["went"]["B"] == {"stay": 1.0, "neighbour": 0.0, "other": 0.0,
                                                          "background": 0.0}
    assert result["cost"]["shrinking"] is None and result["cost"]["transcripts_per_cell"]["a"] == 5.0
    # The pushes: the states in their order, each change at once; progress at most once a second in a state.
    states = [push["state"] for _, push in pushes]
    changes = [state for n, state in enumerate(states) if n == 0 or state != states[n - 1]]
    assert changes[0] == "preparing" and changes[-1] == "done" and "running" in changes
    assert changes == [state for state in ("preparing", "running", "comparing", "done") if state in changes]
    for _, push in pushes:
        assert set(push) == {"state", "progress"} and set(push["progress"]) == PROGRESS
        assert (push["progress"]["of"], push["progress"]["runs"], push["progress"]["marks"]) == (450, 6, [200, 400])
        assert push["progress"]["phase"] in ("preparing", "loading", "sampling", "writing", "reading")
    running = [at for at, push in pushes if push["state"] == "running"]
    assert len(running) - 1 <= (running[-1] - running[0]) / T.PUSH_S + 1, running
    assert pushes[-1][1]["progress"]["done"] == 6


def test_a_start_from_cells_only_runs_4_and_4_runs_all_at_once(api, world, monkeypatch):
    """A parent started from cells only (the start rule patched, as qc_test and qc_pool read it when called) has 4 runs
    on each side, all 8 active at once: they hold 8 places past [proseg] max_active_runs (the parent waits paused among
    them), while a run asked for by hand still finds the limit. Compactness 0.02: "clearly fewer", with the cells-only
    constants of a region re-run as it is (measured on proseg's own count: tau 0.0656)."""
    application, client = api
    runs = application.state.runs
    p = parent(application, client, [100, 100, 550, 550])
    at_end(application, p)
    monkeypatch.setattr(qc_rules, "start_of", lambda meta: "cells")
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "2")        # each test run waits before its first frame: all made by then
    watch = Watch(runs, {p})
    answer = ask(client, p)
    assert answer.status_code == 202, answer.text
    assert (answer.json()["start"], answer.json()["runs_per_side"], answer.json()["progress"]["runs"]) == ("cells", 4, 8)
    assert sorted(answer.json()["runs"]) == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    ids = all_made(client, p)
    assert len(runs.active()) == 9 == runs.max_active + 7
    sample = runs.samples.list()[0]["id"]
    refused = client.post("/api/v1/runs", json={"sample": sample, "box": [0, 0, 100, 100]}, headers=auth())
    assert refused.status_code == 409
    test = ended(client, p)
    assert watch.stop().most == 8 and watch.untagged == set()
    assert test["state"] == "done", test["message"]
    result = test["result"]
    assert (result["verdict"], result["start"], result["situation"]) == ("clearly_fewer", "cells", "as_is")
    assert result["runs"] == {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    assert (result["constants"]["runs_per_side"], result["constants"]["tau"]) == (4, pytest.approx(0.0656, abs=1e-4))
    assert result["come_back"]["held"] == [1.0] * 4 and result["fewer"]["b"] == [-1.0] * 4
    assert test["progress"]["done"] == 8 and gone(client, *ids.values())


def test_a_run_that_cannot_start_with_factorisation_has_every_run_made_again_without_it(api, world, monkeypatch, tmp_path):
    """One test run fails to start with factorisation (FAKE_PROSEG_FAIL_ONCE, set once the parent waits at its end: proseg
    takes the engine's settings as it starts, so the parent never sees it); runs.py's own retry starts it without, and
    the test makes all of its runs again without it, both sides alike, once: every run of the answer carries
    --no-factorization, the first ones are gone, and the test answers."""
    application, client = api
    seen = spy_commands(application, monkeypatch)
    p = parent(application, client, [100, 100, 550, 550])
    at_end(application, p)
    monkeypatch.setenv("FAKE_PROSEG_FAIL_ONCE", str(tmp_path / "failed-once"))
    assert ask(client, p).status_code == 202
    test = ended(client, p)
    assert test["state"] == "done", test["message"]
    assert (test["reruns"], test["no_factorization"], test["progress"]["retry"]) == (1, True, True)
    assert test["result"]["verdict"] == "clearly_fewer" and (tmp_path / "failed-once").exists()
    final = set(test["runs"].values())
    for run_id in final:
        (call,) = seen["runs"][run_id]
        assert "--no-factorization" in call["command"] and call["meta"]["qc_test"]["test"] == test["id"]
    first = [run_id for run_id, calls in seen["runs"].items() if run_id not in final and run_id != p]
    assert len(first) == 6 and gone(client, *first)
    retried = [run_id for run_id in first if len(seen["runs"][run_id]) == 2]       # runs.py's own retry
    assert len(retried) == 1 and "--no-factorization" in seen["runs"][retried[0]][1]["command"]
    assert [r["id"] for r in client.get("/api/v1/runs", headers=auth()).json()] == [p]


def test_a_parent_that_ran_without_factorisation_has_its_runs_start_without_it(api, world, monkeypatch):
    """A parent whose own start needed proseg's retry without factorisation (FAKE_PROSEG_FAIL=startup while it starts,
    then removed): every run of its test starts without it, and none needs a retry."""
    application, client = api
    seen = spy_commands(application, monkeypatch)
    monkeypatch.setenv("FAKE_PROSEG_FAIL", "startup")
    p = parent(application, client, [100, 100, 550, 550])
    wait_for(lambda: application.state.runs.get(p).meta.get("retried_without_factorization"), timeout=30)
    monkeypatch.delenv("FAKE_PROSEG_FAIL")
    at_end(application, p)
    answer = ask(client, p)
    assert answer.status_code == 202 and answer.json()["no_factorization"] is True
    test = ended(client, p)
    assert test["state"] == "done", test["message"]
    assert (test["reruns"], test["progress"]["retry"]) == (0, False)
    for run_id in test["runs"].values():
        (call,) = seen["runs"][run_id]
        assert "--no-factorization" in call["command"]


# ---- stopping ----------------------------------------------------------------------------------------------------------
def test_cancelling_a_test_aborts_its_runs_at_once_and_leaves_its_parent_as_it_was(api, monkeypatch):
    """DELETE while the 6 runs still read their input (FAKE_PROSEG_LOAD_S=30, set once the parent waits at its end):
    every run is aborted at once, all waited for together, and moved away (runs/.trash) before the answer, within a few
    seconds; the test's own input goes too, the test stays as "cancelled" until dismissed, and the parent waits at its
    end as before."""
    application, client = api
    runs, tests = application.state.runs, application.state.qc_tests
    p = parent(application, client, [0, 0, 700, 700])
    at_end(application, p)
    done_at = dict(runs.get(p).meta["schedule_done"])
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "30")
    answer = ask(client, p)
    assert answer.status_code == 202
    test_id = answer.json()["id"]
    ids = all_made(client, p)
    assert (tests.inputs / test_id / "transcripts.parquet").is_file()
    trash = Path(runs.root) / ".trash"                  # made by a delete that empties the folder later, only
    assert not trash.exists()
    began = time.monotonic()
    cancelled = client.delete(f"/api/v1/runs/{p}/qc/test", headers=auth())
    took = time.monotonic() - began
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled", cancelled.text
    assert took < 3.0, f"DELETE took {took:.1f} s"
    assert cancelled.json()["message"] is None
    assert gone(client, *ids.values()) and not any((Path(runs.root) / run_id).exists() for run_id in ids.values())
    assert trash.is_dir() and not (tests.inputs / test_id).exists()
    meta = runs.get(p).meta
    assert (meta["state"], meta["schedule_done"], meta.get("continued"), meta.get("finish_requested")) == \
        ("paused", done_at, None, None)
    assert client.get(f"/api/v1/runs/{p}/qc/test", headers=auth()).json()["state"] == "cancelled"
    assert client.delete(f"/api/v1/runs/{p}/qc/test", headers=auth()).status_code == 200      # dismissed
    assert client.get(f"/api/v1/runs/{p}/qc/test", headers=auth()).status_code == 404


def test_a_test_that_gets_no_answer_in_time_fails_and_one_whose_run_is_saved_stops(api, monkeypatch):
    """With its time patched short, a test whose runs never get going fails with "no answer within …", its runs and
    input cleared away; a test whose parent is then saved (Finish & save: being saved while proseg writes its results,
    here a few seconds) stops with that reason, its runs cleared."""
    application, client = api
    runs, tests = application.state.runs, application.state.qc_tests
    monkeypatch.setenv("FAKE_PROSEG_FINISH_S", "4")
    p = parent(application, client, [0, 0, 700, 700])
    at_end(application, p)
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "30")
    monkeypatch.setattr(T, "TIMEOUT_S", 3.0)
    answer = ask(client, p)
    assert answer.status_code == 202
    failed = ended(client, p, timeout=30)
    assert failed["state"] == "failed" and failed["result"] is None
    assert failed["message"].startswith("The test could not finish: no answer within "), failed["message"]
    assert gone(client, *failed["runs"].values()) and not (tests.inputs / failed["id"]).exists()
    assert runs.get(p).meta["state"] == "paused"
    monkeypatch.setattr(T, "TIMEOUT_S", 600.0)
    assert ask(client, p).status_code == 202                  # a new test replaces the one that failed
    ids = all_made(client, p)
    assert client.post(f"/api/v1/runs/{p}/control", json={"action": "finish"}, headers=auth()).status_code == 200
    stopped = ended(client, p, timeout=30)
    assert (stopped["state"], stopped["message"]) == ("cancelled", "Your run is being saved, so the test stopped")
    assert gone(client, *ids.values())


def test_closing_the_app_cancels_a_test_and_clears_its_runs_first(api, monkeypatch):
    application, client = api
    runs, tests = application.state.runs, application.state.qc_tests
    p = parent(application, client, [0, 0, 700, 700])
    at_end(application, p)
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "30")
    assert ask(client, p).status_code == 202
    ids = all_made(client, p)
    tests.close()
    closed = client.get(f"/api/v1/runs/{p}/qc/test", headers=auth()).json()
    assert (closed["state"], closed["message"]) == ("cancelled", "The app closed")
    assert gone(client, *ids.values()) and not (tests.inputs / closed["id"]).exists()
    assert [r["id"] for r in runs.pending()] == [p]                        # nothing of the test left to save
    refused = ask(client, p)
    assert refused.status_code == 503 and refused.json()["detail"] == "The app is closing"


# ---- refusals, waits and caveats ---------------------------------------------------------------------------------------
def test_a_test_is_refused_only_where_it_cannot_run_and_waits_or_runs_elsewhere(api, monkeypatch):
    """The POST's refusals in their order and words: the body (FastAPI's own 422 for one that is not an object; the
    engine's words for the rest), an unknown run or cell (404), a run being saved or one that ended before its end (409
    {detail: the first reason's words, status, reasons}), the run's own value and far drift with drift off (422); none
    of them pauses, resumes or makes a run. Where the test was not measured (a narrow or small region, a changed
    schedule, a lasso, a whole sample or a box run in tiles, a start from nuclei only) the pool route offers it with its
    caveats, and a press starts it (a cells-only start on a large region, measured on proseg's own count, has none); a
    press before the run's end, or while
    another run samples, makes a waiting test (its run resumed by the press, paused again by its DELETE)."""
    application, client = api
    runs, tests = application.state.runs, application.state.qc_tests
    runs.one_run, runs.keep_unsaved, runs.max_active = False, True, 8     # several parents at once
    monkeypatch.setenv("FAKE_PROSEG_FINISH_S", "3")                      # a run being saved stays so a while
    square = parent(application, client, [0, 0, 700, 700])
    narrow = parent(application, client, [0, 0, 400, 700])
    small = parent(application, client, [0, 0, 250, 250])
    longer = parent(application, client, [100, 100, 550, 550], options={"burnin-samples": 100})
    dry = parent(application, client, [100, 100, 550, 550], options={"no-diffusion": True})
    waiting = parent(application, client, [100, 100, 550, 550], play=False)          # on its first frame
    at_end(application, square, narrow, small, longer, dry)
    wait_for(lambda: runs.get(waiting).meta["state"] == "paused" and runs.get(waiting).frames, timeout=30)
    assert client.post(f"/api/v1/runs/{waiting}/control", json={"action": "speed", "speed": "slow"},
                       headers=auth()).status_code == 200

    def refused(run_id, *codes, **body):
        """POST: 409 with exactly these reasons."""
        answer = ask(client, run_id, **body)
        expected = [qc_pool.reason(code) for code in codes]
        assert answer.status_code == 409, (codes, answer.text)
        assert answer.headers["content-type"] == "application/problem+json"
        assert answer.json() == {"detail": expected[0]["words"], "status": 409, "reasons": expected}, codes

    def offered(run_id, *codes):
        """The pool route: offered (no refusal, no wait), with exactly these caveats."""
        reply = client.get(f"/api/v1/runs/{run_id}/qc/test/pool", params={"cell": SEED}, headers=auth()).json()
        assert (reply["offer"], reply["reasons"], reply["waits"]) == (True, [], []), (codes, reply)
        data = {"region_small": {"min_um": 300.0}, "tiled_box": {"max_um": runs.max_region}}
        assert reply["caveats"] == [qc_pool.reason(code, **data.get(code, {})) for code in codes], codes

    # The body, then the run and the cell.
    url = f"/api/v1/runs/{square}/qc/test"
    assert client.post(url, json=[SEED, "cell-compactness", 0.02], headers=auth()).status_code == 422
    with pytest.raises(QcError) as direct:
        tests.start(square, [SEED, "cell-compactness", 0.02])
    assert (str(direct.value), direct.value.status) == ("A test is {cell, flag, value}", 422)
    good = {"cell": SEED, "flag": "cell-compactness", "value": MOVED}
    for body, words in (({**good, "cell": -1}, "cell must be a cell number (label - 1)"),
                        ({**good, "cell": 1.5}, "cell must be a cell number (label - 1)"),
                        ({**good, "cell": "3"}, "cell must be a cell number (label - 1)"),
                        ({**good, "changes": [{"flag": "prior-weight", "value": 0.5}]},
                         "A test changes one setting: send flag and value"),
                        ({**good, "flag": "temperature"},
                         "A test changes one of the settings the test form offers"),
                        ({**good, "value": True}, "value must be a number"),
                        ({**good, "value": "0.02"}, "value must be a number"),
                        ({**good, "value": 50}, "Choose a value from 0.001 to 10"),
                        ({**good, "flag": "prior-weight", "value": 5000}, "Choose a value from 0 to 1000"),
                        ({**good, "flag": "diffusion-probability", "value": 0.05}, "Choose a value from 0.1 to 0.3")):
        answer = client.post(url, json=body, headers=auth())
        assert answer.status_code == 422 and answer.json() == {"detail": words, "status": 422}, body
    for run_url, words in (("/api/v1/runs/nope/qc/test", "Unknown run"), (url, "Unknown cell")):
        answer = client.post(run_url, json={**good, "cell": 99999}, headers=auth())
        assert answer.status_code == 404 and answer.json()["detail"] == words
    # Where it was not measured: offered, with its caveats.
    offered(square)
    offered(narrow, "area_narrow")
    offered(small, "region_small")
    offered(longer, "schedule")
    run = runs.get(square)
    lasso = {"kind": "polygon", "points": [[100, 100], [600, 100], [350, 600]], "bbox": [100, 100, 600, 600]}
    whole = {"kind": "full", "bbox": [0.0, 0.0, 700.0, 700.0]}
    cells = run.info["cells"]
    # One rule for whole samples, the mixing check's (qc_constants.run_mode): WHOLE_CELLS cells or more, tiled or not;
    # a run of the whole sample's region with fewer is a tiled box (its region's kind alone decides nothing).
    for changes, count, codes in (({"mode": "tiled", "region": whole}, cells, ["tiled_box"]),
                                  ({"mode": "tiled", "region": whole}, qc_constants.WHOLE_CELLS, ["whole_sample"]),
                                  ({}, qc_constants.WHOLE_CELLS, ["whole_sample"]),
                                  ({"mode": "tiled"}, cells, ["tiled_box"]),   # a box run in tiles: over the limit
                                  ({"mode": "tiled", "region": lasso}, cells, ["tiled_box", "lasso"]),
                                  ({"region": lasso}, cells, ["lasso"])):
        kept = {key: run.meta[key] for key in changes}
        run.meta.update(changes)
        run.info["cells"] = count
        try:
            offered(square, *codes)
        finally:
            run.meta.update(kept)
            run.info["cells"] = cells
    with monkeypatch.context() as patched:
        patched.setattr(qc_rules, "start_of", lambda meta: "nuclei")
        offered(square, "nuclei_start")
        offered(small, "nuclei_start", "region_small")              # judged as nuclei and cells: its region's own
        patched.setattr(qc_rules, "start_of", lambda meta: "cells")
        offered(square)                                             # its square measured on proseg's own count
        offered(small, "region_small")                              # 300 µm for cells-only starts too
    # Past the pool: the run's own value, and far drift with drift off.
    same = ask(client, square, value=0.04)
    assert same.status_code == 422
    assert same.json()["detail"] == "0.04 is what this run uses now: choose another value to test"
    drift = ask(client, dry, flag="diffusion-probability", value=0.25)
    assert drift.status_code == 422
    assert drift.json()["detail"] == "Far drift does nothing while drift is switched off in this run"
    assert sorted(r["id"] for r in client.get("/api/v1/runs", headers=auth()).json()) == \
        sorted([square, narrow, small, longer, dry, waiting]) and tests.tests == {}
    # A press where it was not measured starts the test, its caveats in its record (then cancelled here).
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "30")
    started = ask(client, narrow)
    assert started.status_code == 202 and started.json()["state"] == "preparing", started.text
    assert (started.json()["caveats"], started.json()["area"]["rule"], started.json()["row"]) == \
        ([qc_pool.reason("area_narrow")], "crop", {"start": "paired", "situation": "square"})
    assert client.delete(f"/api/v1/runs/{narrow}/qc/test", headers=auth()).json()["state"] == "cancelled"
    assert client.delete(f"/api/v1/runs/{narrow}/qc/test", headers=auth()).status_code == 200      # dismissed
    # Another run sampling: the press waits for it (and the pool route says so); the parent at its end stays paused.
    assert client.post(f"/api/v1/runs/{waiting}/control", json={"action": "resume"}, headers=auth()).status_code == 200
    wait_for(lambda: runs.get(waiting).meta["state"] == "running", timeout=30)
    pool = client.get(f"{url}/pool", params={"cell": SEED}, headers=auth()).json()
    assert (pool["offer"], pool["reasons"], pool["waits"]) == (True, [], [qc_pool.reason("other_run")])
    other = ask(client, square)
    assert other.status_code == 202 and other.json()["state"] == "waiting", other.text
    assert other.json()["waiting"] == [qc_pool.reason("other_run")] and runs.get(square).meta["state"] == "paused"
    again = ask(client, square)                                 # one test at a time, waiting or on its way
    assert again.status_code == 409
    assert again.json()["detail"] == "A test of this run is on its way: wait for its answer, or cancel it"
    busy = client.get(f"{url}/pool", params={"cell": SEED}, headers=auth()).json()
    assert busy["offer"] is False and busy["reasons"] == [qc_pool.reason("busy", why="test")]
    assert client.delete(url, headers=auth()).json()["state"] == "cancelled"
    assert runs.get(square).meta["state"] == "paused" and client.delete(url, headers=auth()).status_code == 200
    # Before the run's end: the press resumes it (here paused again first), and DELETE pauses it again.
    assert client.post(f"/api/v1/runs/{waiting}/control", json={"action": "pause"}, headers=auth()).status_code == 200
    wait_for(lambda: runs.get(waiting).meta["state"] == "paused", timeout=30)
    pressed = ask(client, waiting)
    assert pressed.status_code == 202 and pressed.json()["state"] == "waiting", pressed.text
    assert pressed.json()["waiting"] == [qc_pool.reason("not_at_end")] and pressed.json()["pool"] is None
    assert runs.get(waiting).meta["state"] == "running"
    assert client.get(f"/api/v1/runs/{waiting}", headers=auth()).json()["qc_test"]["state"] == "waiting"
    cancelled = client.delete(f"/api/v1/runs/{waiting}/qc/test", headers=auth())
    assert cancelled.json()["state"] == "cancelled" and runs.get(waiting).meta["state"] == "paused"
    assert client.delete(f"/api/v1/runs/{waiting}/qc/test", headers=auth()).status_code == 200
    # Refused: a run without proseg's own counts (made before they were kept, or by a proseg-live that can't keep
    # them), whatever else holds.
    run = runs.get(square)
    info = run.info
    run.info = {key: value for key, value in info.items() if key != "counts"}
    try:
        refused(square, "no_counts")
        run.meta["live_counts"] = "old_proseg"
        refused(square, "old_proseg")
    finally:
        run.info = info
        run.meta["live_counts"] = "request"
    # Refused: a run being saved; one that ended before its end; one that ended at its end is tested.
    assert client.post(f"/api/v1/runs/{longer}/control", json={"action": "finish"}, headers=auth()).status_code == 200
    refused(longer, "ending")
    wait_for(lambda: runs.get(longer).meta["state"] == "done", timeout=30)
    offered(longer, "schedule")                                 # done at its end (a Xenium bundle: nothing to save)
    assert client.post(f"/api/v1/runs/{waiting}/control", json={"action": "abort"}, headers=auth()).status_code == 200
    wait_for(lambda: runs.get(waiting).meta["state"] == "stopped", timeout=30)
    refused(waiting, "ended")
    # Nothing was made by a refusal or a wait; the test that ran was cleared away.
    assert sorted(r["id"] for r in client.get("/api/v1/runs", headers=auth()).json()) == \
        sorted([square, narrow, small, longer, dry, waiting])
    assert {run_id: runs.get(run_id).meta["state"] for run_id in (small, dry)} == dict.fromkeys((small, dry), "paused")
    assert tests.tests == {} and not (tests.inputs.exists() and any(tests.inputs.iterdir()))


# ---- the pool route ----------------------------------------------------------------------------------------------------
def test_the_pool_route_lists_the_runs_reasons_by_kind_in_the_engines_words(api_own_pools):
    """GET /runs/{id}/qc/test/pool?cell=&wait= with the app's own pools: a run that has not reached its end, of a
    Xenium bundle (whose mixing check can't run: no counts table, and no analysis worker here), is refused at once,
    the check's refusal among its reasons and its end among its waits (state ready, offer false; nothing of the cell
    or its pool; the area its region gives; the row it would be judged with), however long `wait` lets it wait. Its
    query is the engine's to read: 422 in its words as problem JSON (never FastAPI's own list), 404 for an unknown run
    or cell or a run that is not proseg's; private, never cached."""
    application, client = api_own_pools
    runs = application.state.runs
    p = parent(application, client, [100, 100, 550, 550], play=False)
    wait_for(lambda: runs.get(p).meta["state"] == "paused" and runs.get(p).info, timeout=30)
    url = f"/api/v1/runs/{p}/qc/test/pool"
    began = time.monotonic()
    answer = client.get(url, params={"cell": "0", "wait": "10"}, headers=auth())
    assert time.monotonic() - began < 2.0
    assert answer.status_code == 200 and answer.headers["cache-control"] == "private, no-cache"
    reply = answer.json()
    assert list(reply) == POOL_REPLY
    assert (reply["v"], reply["state"], reply["run"], reply["cell"], reply["offer"]) == (1, "ready", p, 0, False)
    assert reply["reasons"] == [{"code": "check_unavailable", "words": "The mixing check can't run on this sample"}]
    assert reply["waits"] == [{"code": "not_at_end", "words": "Starts once your run reaches its last iteration"}]
    assert reply["caveats"] == [qc_pool.reason("count_basis")]           # until the constants are proseg's
    assert (reply["constants"], reply["start"], reply["runs_per_side"], reply["k"], reply["row"]) == \
        (qc_rule.CONSTANTS_ID, "paired", 3, 160, {"start": "paired", "situation": "as_is"})
    assert reply["area"] == {"rule": "as_is", "box": [100.0, 100.0, 550.0, 550.0], "inner": [125.0, 125.0, 525.0, 525.0]}
    assert (reply["pattern"], reply["line"], reply["line_shown"], reply["this_cell"], reply["pool"]) == (None,) * 5
    assert reply["count"] == "averaged"                                  # F's: its first frame, counted at the start
    assert client.get(url, params={"cell": "1"}, headers=auth()).status_code == 200      # wait: 0 unless given
    for query, words in (({"cell": "x"}, "cell must be a cell number (label - 1)"),
                         ({"cell": "-1"}, "cell must be a cell number (label - 1)"),
                         ({"cell": "1.5"}, "cell must be a cell number (label - 1)"),
                         ({}, "cell must be a cell number (label - 1)"),
                         ({"cell": "0", "wait": "11"}, "wait must be 0 to 10 seconds"),
                         ({"cell": "0", "wait": "-1"}, "wait must be 0 to 10 seconds"),
                         ({"cell": "0", "wait": "nan"}, "wait must be 0 to 10 seconds"),
                         ({"cell": "0", "wait": "x"}, "wait must be 0 to 10 seconds")):
        answer = client.get(url, params=query, headers=auth())
        assert answer.status_code == 422 and answer.headers["content-type"] == "application/problem+json", query
        assert answer.json() == {"detail": words, "status": 422}, query
    for path, query, words in (("/api/v1/runs/nope/qc/test/pool", {"cell": "0"}, "Unknown run"),
                               (url, {"cell": "2"}, "Unknown cell")):         # the fake's sample has 2 cells
        answer = client.get(path, params=query, headers=auth())
        assert answer.status_code == 404 and answer.json() == {"detail": words, "status": 404}
    run = runs.get(p)
    run.meta["kind"] = "stain"
    try:
        answer = client.get(url, params={"cell": "0"}, headers=auth())
        assert answer.status_code == 404 and answer.json()["detail"] == "Not a proseg run"
    finally:
        run.meta["kind"] = "proseg"


def test_the_pool_route_and_the_post_refuse_where_the_check_cannot_run(api_own_pools):
    """A run at the end of its schedule, of a Xenium bundle (the mixing check can't run on it): the cheap reasons alone
    answer, at once, offer false, with the run's end as the pool's frame; a press is refused with that reason (a test
    starts from the check's suspects: it would wait for ever), nothing paused or made."""
    application, client = api_own_pools
    runs = application.state.runs
    p = parent(application, client, [100, 100, 550, 550])
    at_end(application, p)
    reply = client.get(f"/api/v1/runs/{p}/qc/test/pool", params={"cell": "0", "wait": "5"}, headers=auth()).json()
    assert (reply["state"], reply["offer"], reply["waits"], reply["caveats"]) == \
        ("ready", False, [], [qc_pool.reason("count_basis")])
    assert reply["reasons"] == [qc_pool.reason("check_unavailable")]
    done = runs.get(p).meta["schedule_done"]
    assert (reply["seq"], reply["iter"]) == (done["seq"], done["iter"]) == (451, 450)
    answer = client.post(f"/api/v1/runs/{p}/qc/test", json={"cell": 0, "flag": "cell-compactness", "value": 0.02},
                         headers=auth())
    assert answer.status_code == 409 and answer.json()["reasons"] == [qc_pool.reason("check_unavailable")]
    assert runs.get(p).meta["state"] == "paused" and [r["id"] for r in client.get("/api/v1/runs",
                                                                                   headers=auth()).json()] == [p]


# ---- the hold ----------------------------------------------------------------------------------------------------------
def test_proseg_s_hold_scale_and_the_value_that_gives_a_run_a_hold():
    """scale_of is proseg's own float32 scale, max(1, transcripts / cells / 100) (main.rs:955-956): on the crop
    (2,601,519 transcripts, 3,439 cells) 7.564754, the prior_weight its info.json says at 1.0; hold_of carries it (6
    significant digits) beside the float32 product proseg applies; given_hold is the float32 value whose product with
    a run's scale is the hold asked for: the parent's 0.5 at 10.5 (5.25) given at 4.52 is 1.1615043878555298, and that
    times 4.52 in float32 is 5.25 exactly, whether proseg reads it from its command line or as a live setting."""
    crop = {"transcripts": 2_601_519, "cells": 3_439, "prior_weight": 7.564754}       # the crop's info.json at 1.0
    assert T.scale_of(crop) == float(np.float32(crop["prior_weight"])) == 7.564754009246826
    assert T.hold_of(crop, 1.0) == {"value": 1.0, "effective": 7.56475, "scale": 7.56475}
    assert T.scale_of({"transcripts": 50_000, "cells": 1000}) == 1.0 == T.scale_of({"transcripts": 7, "cells": 0})
    for info in (None, {}, {"transcripts": 5}, {"cells": 5}, {"transcripts": "5", "cells": 1}):
        assert T.scale_of(info) is None, info
    parent, square = T.scale_of({"transcripts": 1_050_000, "cells": 1000}), T.scale_of({"transcripts": 452_000,
                                                                                         "cells": 1000})
    assert (parent, square) == (10.5, float(np.float32(4.52)))
    assert T.hold_of({"transcripts": 1_050_000, "cells": 1000}, 0.5) == {"value": 0.5, "effective": 5.25,
                                                                          "scale": 10.5}
    as_now = T.given_hold(float(np.float32(0.5) * np.float32(parent)), square)
    tried = T.given_hold(float(np.float32(2.0) * np.float32(parent)), square)
    assert (as_now, tried) == (1.1615043878555298, 4.646017551422119)
    assert np.float32(as_now) * np.float32(square) == 5.25 and np.float32(tried) * np.float32(square) == 21.0
    for value in (as_now, tried):                      # its shortest decimal reads back as the same float32
        assert float(np.float32(repr(value))) == value and float(str(value)) == value
    assert T.given_hold(5.25, 1.0) == 5.25 and T.HOLD_MAX == 1000.0


# ---- the docs ----------------------------------------------------------------------------------------------------------
def paragraph(name: str, start: str) -> str:
    """The paragraph or list item of a doc that ships (README.md, docs/ARCHITECTURE.md) beginning with `start`, up to
    the next blank line or list item, as one line (markdown's line breaks inside it are spaces)."""
    text = (ROOT / name).read_text()
    at = text.index(start)
    ends = [end for end in (text.find("\n\n", at), text.find("\n- ", at)) if end != -1]
    return re.sub(r"\s+", " ", text[at:min(ends, default=len(text))])


def test_the_docs_say_where_a_test_runs_as_the_engine_decides():
    """engine/qc_pool.py re-runs a region as it is when its box fits in 500 x 500 µm, else a 500 µm window around the
    cell, shifted to stay inside the region and cut to it where the region is narrower (a crop); a test the
    measurements did not cover runs all the same, judged with the nearest measured row (qc_rule.row_for) and carrying
    caveats; a press before the run's end waits for it, and one with the mixing check off switches it on first.
    README.md and docs/ARCHITECTURE.md say so, and no longer that a test answers only where it was measured."""
    assert qc_pool.area_of({"kind": "box", "bbox": [0, 0, 500, 500]}, (math.nan, math.nan)).rule == "as_is"
    assert qc_pool.area_of({"kind": "box", "bbox": [0, 0, 700, 700]}, (100.0, 100.0)).box == (0.0, 0.0, 500.0, 500.0)
    narrow = qc_pool.area_of({"kind": "box", "bbox": [0, 0, 400, 1200]}, (100.0, 1100.0))
    assert (narrow.rule, narrow.box) == ("crop", (0.0, 700.0, 400.0, 1200.0))
    assert (qc_rule.row_for("nuclei", [0, 0, 400, 1200]), qc_rule.row_for("cells", [0, 0, 700, 700])) == \
        (("paired", "square"), ("cells", "square"))
    assert [qc_pool.KIND[code] for code in ("not_at_end", "check_off")] == ["wait", "wait"]
    assert {qc_pool.KIND[code] for code in ("area_narrow", "lasso", "square_not_measured", "whole_sample",
                                            "nuclei_start", "too_few")} == {"caveat"}
    readme = paragraph("README.md", "**Test a setting here**")
    for sentence in ("go to the end of proseg's schedule, all at once: 3 as now and 3 with your value",
                     "A corner of the image shows one of them as now and one with your value, side by side.",
                     "Fresh runs of your region (or, when it is over 500 µm, of a 500 µm square around the cell, cut "
                     "to your region where it is narrower)",
                     "A press before your run reaches its last iteration waits for it, and one with the mixing check "
                     "off switches it on first.",
                     "Where the test was not measured, it runs all the same and its answer says so in its tooltip."):
        assert sentence in readme, sentence
    assert "It answers only where it was measured" not in readme
    architecture = paragraph("docs/ARCHITECTURE.md", "- **Test a setting here**")
    for sentence in ("the area is the region as it is when it fits in 500 × 500 µm, else a 500 µm square around the "
                     "cell, shifted to stay inside the region and cut to it where the region is narrower",
                     "where the measurements did not cover a situation, the test runs all the same, judged with the "
                     "nearest measured row (`constants.start`, `constants.situation`), and its answer carries a caveat "
                     "(`caveats`)"):
        assert sentence in architecture, sentence
    assert "answers only where it was measured" not in architecture


def test_the_readme_says_the_corner_shows_one_run_as_now_and_one_with_the_value():
    """A test's 6 or 8 runs go at once, and the corner draws two of them, the first of each side (qc_test.VIEW_ROLES;
    the square's test above gets A1 and B1 alone): README.md says so, never that the runs go side by side in it."""
    assert T.VIEW_ROLES == ("A1", "B1")
    readme = paragraph("README.md", "**Test a setting here**")
    assert "go to the end of proseg's schedule, all at once: 3 as now and 3 with your value" in readme
    assert "A corner of the image shows one of them as now and one with your value, side by side." in readme
    assert "schedule side by side" not in readme
