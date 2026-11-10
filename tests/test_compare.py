"""Compare's engine (engine/compare.py): cell centres, the one area every column covers, and the comparison jobs
(one per run, counted columns scored as they come, results kept beside the run, each column by its own count: the
run's proseg's of its nearest counted frame, a segmentation's its table's or its outlines') against a fake sample, model
and GPU worker. The helper's counting and the segmentation catalogue are tested on SpatialData in test_zarr.py."""
from __future__ import annotations

from concurrent.futures import Future
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zstandard
from fastapi.testclient import TestClient

from conftest import wait_for

from sopa_companion.analysis import counts_file as CF
from sopa_companion.analysis import worker as W
from sopa_companion.engine import compare as C
from sopa_companion.engine import zarr_samples as Z
from sopa_companion.engine.analysis import AnalysisError
from sopa_companion.engine.runs import Run, RunError
from sopa_companion.engine.samples import SampleError
from sopa_companion.engine.zarr_samples import HelperFailed

BASE = "http://127.0.0.1:8799"


def auth():
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}


def outlines(rings: list, owners: list | None = None, count: int | None = None) -> dict:
    """Exported outlines (ZarrSamples._outlines arrays) from rings [[x, y], ...]: part k belongs to owners[k]."""
    owners = list(range(len(rings))) if owners is None else owners
    offsets = np.zeros(len(rings) + 1, np.uint32)
    offsets[1:] = np.cumsum([len(r) for r in rings])
    coords = np.array([v for r in rings for v in r], np.float32).reshape(-1, 2)
    bounds = np.array([[*np.min(r, 0), *np.max(r, 0)] if len(r) else [0, 0, 0, 0] for r in rings], np.float32)
    count = max(owners) + 1 if count is None else count
    return {"ids": [f"cell-{k}" for k in range(count)], "bounds": bounds, "offsets": offsets,
            "owner": np.array(owners, np.uint32), "coords": coords}


def square(cx, cy, half=4.0, closed=True):
    ring = [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]]
    return ring + ring[:1] if closed else ring


# ---- centres and the area -------------------------------------------------------------------------------------------
def test_outline_centres_are_the_area_weighted_centres_of_their_parts():
    ell = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]                      # area 3: (2 x (1, .5) + (.5, 1.5)) / 3
    far = square(10_000.5, 9_000.25, half=3)                                    # far from the origin: still exact
    clockwise = square(5, 7, closed=False)[::-1]
    rings = [square(1, 1, 1), ell, far, clockwise,
             square(1, 1, 1), [[10, 0], [11, 0], [11, 1], [10, 1]],             # one outline of two parts
             [[3, 3], [4, 4], [5, 5]],                                          # no area: the mean of its vertices
             []]                                                                # an empty part
    x, y = C.outline_centres(outlines(rings, [0, 1, 2, 3, 4, 4, 5, 6], count=8))
    expected = [(1, 1), (2.5 / 3, 2.5 / 3), (10_000.5, 9_000.25), (5, 7), (2.9, 0.9), (4, 4)]
    assert np.allclose(np.c_[x[:6], y[:6]], expected, atol=1e-3)
    assert np.isnan(x[6]) and np.isnan(y[6]) and np.isnan(x[7])                # owners without vertices


def test_frame_centres_read_the_cell_statistics(tmp_path):
    n = 4
    stats = np.zeros((7, n), np.uint32)
    stats[0] = [5, 0, 3, 1]                                                     # cell 1 holds no column: no centre
    stats[5], stats[6] = np.array([1.5, 9, 30.25, -4], "<f4").view("<u4"), np.array([2.5, 9, 40, 7], "<f4").view("<u4")
    path = tmp_path / "000003.cells.zst"
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(stats.astype("<u4").tobytes()))
    x, y = C.frame_centres(path, n)
    assert np.allclose(x[[0, 2, 3]], [1.5, 30.25, -4]) and np.allclose(y[[0, 2, 3]], [2.5, 40, 7])
    assert np.isnan(x[1]) and np.isnan(y[1])
    with pytest.raises(ValueError):
        C.frame_centres(path, 5)


def cells_in(tiles: dict) -> tuple[np.ndarray, np.ndarray]:
    """Centres: tiles {(i, j): cells} of 100 µm, spread inside each tile."""
    xs, ys = [], []
    for (i, j), count in tiles.items():
        for k in range(count):
            xs.append(i * 100 + 10 + 20 * (k % 4))
            ys.append(j * 100 + 15 + 20 * (k // 4))
    return np.array(xs, float), np.array(ys, float)


def test_the_area_is_the_tiles_holding_three_cells_of_every_column():
    a = cells_in({(0, 0): 3, (1, 0): 5, (2, 0): 4, (3, 0): 2, (-1, 0): 3})    # (3, 0) holds 2: not covered by a
    b = cells_in({(1, 0): 3, (2, 0): 9, (3, 0): 6, (-1, 0): 4})
    area = C.common_area({"a": a, "b": b})
    assert C.tile_of(area["tiles"])[0].tolist() == [-1, 1, 2] and area["count"] == 3
    assert area["box"] == [-100.0, 0.0, 300.0, 100.0] and area["size_um"] == [400.0, 100.0]
    assert area["own"] == {"a": 4, "b": 4}
    inside = C.in_area(*a, area)
    assert inside.sum() == 3 + 5 + 4 and not inside[:3].any()                  # (0, 0) is a's alone
    with_missing = C.in_area(np.array([np.nan, 150.0]), np.array([50.0, 50.0]), area)
    assert with_missing.tolist() == [False, True]
    # nothing shared: no area
    apart = C.common_area({"a": cells_in({(0, 0): 5}), "b": cells_in({(4, 4): 5})})
    assert apart["box"] is None and apart["count"] == 0 and apart["size_um"] == [0.0, 0.0]
    # a tile edge belongs to the tile it opens
    assert C.tile_of(C.tile_keys([100.0, 99.999], [0.0, 0.0]))[0].tolist() == [1, 0]


def test_a_region_runs_area_stays_inside_its_box():
    """Its box limits every column and starts the grid: other columns' cells beyond the region never count, and a
    tile cut by the box's far edge is kept as cut."""
    grid = np.mgrid[5:600:10, 5:200:10].reshape(2, -1).astype(float)          # 10 µm apart over 600 x 200 µm
    run = grid[:, (grid[0] > 150) & (grid[0] < 420)]                           # the run's cells: its region only
    limit = [150.0, 0.0, 420.0, 200.0]
    area = C.common_area({"run": (run[0], run[1]), "xenium": (grid[0], grid[1])}, limit=limit)
    assert area["box"] == [150.0, 0.0, 420.0, 200.0] and area["size_um"] == [270.0, 200.0] and area["count"] == 6
    assert area["own"] == {"run": 6, "xenium": 6} and area["origin"] == (150.0, 0.0)
    inside = C.in_area(grid[0], grid[1], area)
    assert inside.sum() == 27 * 20 and (grid[0][inside] > 150).all() and (grid[0][inside] < 420).all()
    loose = C.common_area({"run": (run[0], run[1]), "xenium": (grid[0], grid[1])})   # on the frame's grid instead,
    assert C.in_area(grid[0], grid[1], loose).sum() > inside.sum()             # Xenium's cells beside the region count


def test_an_element_is_counted_whole_unless_the_area_is_a_small_part_of_it():
    assert C.count_box([0, 0, 100, 100], [0, 0, 120, 120]) is None
    assert C.count_box([0, 0, 100, 100], [0, 0, 1000, 1000]) == [0, 0, 100, 100]
    assert C.count_box([0, 0, 100, 100], None) == [0, 0, 100, 100]


# ---- jobs with a fake sample, model and worker ----------------------------------------------------------------------
GRID = [(20 + 40 * i, 20 + 40 * j) for i in range(10) for j in range(5)]       # 50 cells over 400 x 200 µm


class FakeRuns:
    def __init__(self):
        self.runs, self.lock = {}, threading.Lock()

    def get(self, run_id):
        if run_id not in self.runs:
            raise RunError("Unknown run", 404)
        return self.runs[run_id]


class FakeZarr:
    """A sample with shapes layers {key: centres}, each at a version (element_stamp; write() replaces a layer as a
    notebook's write_element does: another version); counts written on request as the helper writes them, of the layer
    as it is when counted (gates hold a count, outline_gates an outline read, until set or stopped; before_count and
    after_outlines run once, as another writer lands). As ZarrSamples: the outlines are kept per version (exports), and
    an outline read or a count during which the layer was written again is refused (409); `stamps[key]` answers the
    next stamp reads in turn (None: its folder did not answer)."""

    def __init__(self, folder: Path, layers: dict, extents: dict | None = None):
        self.folder, self.layers, self.extents = folder, layers, extents or {}
        self.versions: dict[str, str] = {}
        self.stamps: dict[str, list] = {}
        self.exports: dict[tuple, dict] = {}
        self.cached: set[str] = set()
        self.gates: dict[str, threading.Event] = {}
        self.outline_gates: dict[str, threading.Event] = {}
        self.before_count: dict[str, callable] = {}
        self.after_outlines: dict[str, callable] = {}
        self.fail: dict[str, str] = {}        # outlines that cannot be read
        self.fail_counts: dict[str, str] = {}  # elements whose counting fails
        self.counted: list[tuple] = []
        self.stopped: list[str] = []
        self.held: list[str] = []             # outline reads held by their gate
        self.read: list[tuple] = []           # (key, version) of each outline read
        self.rules: list[dict | None] = []    # the counting rule each lookup or count was asked for
        self.own: dict[str, tuple] = {}       # key -> (count_by, table) it was asked to read (none: its outlines')
        self.sources: dict[str, dict] = {}    # key -> {count_by, count_table} the listing says (none: its outlines)
        self.table_versions: dict[str, str] = {}
        self.listings = 0                     # the sample's segmentations listed (each may run the sopa helper)

    def known(self, sid):
        return True

    def element_stamp(self, sid, key):
        if self.stamps.get(key):
            return self.stamps[key].pop(0)
        return self.versions.get(key, "v1") if key in self.layers else None

    def write(self, key, centres):
        self.layers[key] = list(centres)
        self.versions[key] = f"v{int(self.versions.get(key, 'v1')[1:]) + 1}"

    def segmentations(self, sid):
        self.listings += 1
        return [{"key": k, "name": f"Layer {k}", "source": "other", "version": self.element_stamp(sid, k),
                 "extent": self.extents.get(k, [0, 0, 400, 200]), **self.sources.get(k, {})} for k in self.layers]

    def table_stamp(self, sid, table):
        return self.table_versions.get(table, "t1")

    def _outlines(self, sid, key, should_stop=None):
        if key in self.fail:
            raise SampleError(self.fail[key])
        version = self.versions.get(key, "v1")
        gate = self.outline_gates.get(key)
        if gate is not None:
            self.held.append(key)
        while gate is not None and not gate.wait(0.02):
            if should_stop and should_stop():
                self.stopped.append(f"outlines {key}")
                raise SampleError("Stopped", 409)
        if self.versions.get(key, "v1") != version:
            raise SampleError(f"'{key}' was written again while its outlines were read: try again", 409)
        self.read.append((key, version))
        value = self.exports.setdefault((key, version), outlines([square(x, y) for x, y in self.layers[key]]))
        hook = self.after_outlines.pop(key, None)
        if hook is not None:
            hook()
        return value

    def counts(self, sid, key, box=None, progress=None, should_stop=None, cached_only=False, rule=None,
               count_by="outlines", table=None):
        self.rules.append(rule)
        self.own[key] = (count_by, table)
        path = self.folder / f"{key}.npz"
        if key not in self.cached:
            if cached_only:
                return None
            self.counted.append((key, box))
            version = self.versions.get(key, "v1")
            hook = self.before_count.pop(key, None)
            if hook is not None:
                hook()
            gate = self.gates.get(key)
            while gate is not None and not gate.wait(0.02):
                if should_stop and should_stop():
                    self.stopped.append(key)
                    raise SampleError("Stopped", 409)
            if key in self.fail_counts:
                raise SampleError(self.fail_counts[key])
            if self.versions.get(key, "v1") != version:
                raise SampleError(f"'{key}' was written again while it was counted: compare again", 409)
        write_counts(path, self.layers[key], box)
        return path


def write_counts(path: Path, centres: list, box=None, ids: list | None = None):
    """A counts file as zarr_bridge/helper.py count writes it, rows in their own order (here: reversed): each row's
    position in the element (`index`), its id and its outline's centre; `ids` other names for the element's cells."""
    keep = [k for k, (x, y) in enumerate(centres) if box is None or (box[0] <= x <= box[2] and box[1] <= y <= box[3])]
    keep = np.array(keep[::-1], np.int64)
    names = ids if ids is not None else [f"cell-{k}" for k in range(len(centres))]
    xy = np.array(centres, np.float64).reshape(-1, 2)
    np.savez(path, index=keep.astype("<i4"), shape=np.array([len(keep), 3]),
             ids=np.array([names[k] for k in keep], dtype=str), x=xy[keep, 0].astype("<f4"), y=xy[keep, 1].astype("<f4"))


class SizedZarr(FakeZarr):
    """A FakeZarr whose sample detail gives its extent (the transcripts' bounding box)."""

    def __init__(self, folder: Path, layers: dict, extent: list):
        super().__init__(folder, layers, {key: extent for key in layers})
        self.extent = extent

    def detail(self, sid):
        return {"extent": list(self.extent)}


class FakeModels:
    def model(self, sid):
        return {"id": "6ccbac0d4a1dad54ef1cdceabc413bbb", "dir": "/models/m1", "table": "xenium_table"}


class FakeQc:
    def __init__(self, deep=True):
        self.model = {"id": "6ccbac0d4a1dad54ef1cdceabc413bbb", "path": "/models/m1", "deep": "/models/m1/deep.npz" if deep else None}
        self.models = FakeModels()

    def qc_model(self, run):
        return dict(self.model) if self.model else None


class FakeAnalysis:
    """AnalysisManager's submit_compare as the engine uses it: records each request and answers
    a Future of the worker's reply {ok, run, seq, compare: {...}}, each column's points written where it was told. With
    `hold`, the Futures stay unanswered until the test answers them (`futures`)."""
    available = True

    def __init__(self, runs=None):
        self.runs = runs
        self.framed: list[tuple] = []      # (run, seq asked) of each compare_frame
        self.after_frame = None            # run once a frame is chosen (a count cleared from the spool then)
        self.requests: list[dict] = []
        self.scored: list[dict] = []       # per request, each shapes column's cells handed over: their centres (x, y)
        self.futures: list[Future] = []
        self.hook = None
        self.reply = None
        self.hold = False
        self.own: set[int] = set()         # the columns given a map of their own ("Each its own"), by index

    def qc_thresholds(self):
        return {"mixed_at": .3, "judge_min": 100, "fit_min": 20}

    def compare_frame(self, run_id, seq=None):
        """AnalysisManager.compare_frame as the engine's contract gives it: the run's column at the newest frame with
        proseg's count (its file there) at or before `seq` (the newest frame without one), else the oldest after it;
        a pass's fields, that frame's seq and iter, and `counts_file`. 404 an unknown frame; 409 none."""
        self.framed.append((run_id, seq))
        run = self.runs.get(run_id)
        with run.lock:
            frames = [dict(f) for f in run.frames]
        if seq is not None and not 0 <= seq < len(frames):
            raise AnalysisError("Unknown frame", 404)
        counted = [k for k, f in enumerate(frames) if f.get("counts_file") and (run.live / f["counts_file"]).exists()]
        if not counted:
            raise AnalysisError("This run has no frame with proseg's count", 409)
        top = len(frames) - 1 if seq is None else seq
        at = max((k for k in counted if k <= top), default=None)
        at = min(k for k in counted if k > top) if at is None else at
        frame, info = frames[at], run.info
        column = {"run": run_id, "seq": at, "iter": frame.get("iter"), "transcripts": str(run.live / "transcripts.bin.gz"),
                  "genes": str(run.live / "genes.txt"), "frame": str(run.live / frame["file"]),
                  "format": frame.get("format", "dense"), **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                  "z_origin": info["z"]["grid_origin"], "z_spacing": info["z"]["grid_spacing"], "ncells": info["cells"],
                  "cells_file": str(run.live / frame["cells_file"]), "counts_file": str(run.live / frame["counts_file"])}
        if self.after_frame is not None:
            self.after_frame(column)
        return column

    def submit_compare(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        self.scored.append({c["key"]: handed(c) for c in request["columns"] if c.get("counts")})
        future = Future()
        self.futures.append(future)
        if self.hook is not None:
            self.hook(request)
        if not self.hold:
            future.set_running_or_notify_cancel()
            future.set_result(self.answer(request))
        return future

    def answer(self, request):
        if self.reply is not None:
            return self.reply
        columns, made = [], False
        for k, column in enumerate(request["columns"]):
            cells = np.load(column["cells"])
            points = np.zeros(len(cells), dtype=[("x", "<f4"), ("y", "<f4"), ("kind", "u1"), ("share", "u1")])
            points["kind"], points["share"] = 1, 40
            path = Path(column["out_points"])
            path.write_bytes(points.tobytes())
            own = {}
            if request.get("own") and column["index"] in self.own:   # its own map and clusters, as the worker writes them
                mine, ids = path.with_name(path.stem + ".own.bin"), path.with_name(path.stem + ".clusters.bin")
                points["x"] = 1.0
                mine.write_bytes(points.tobytes())
                ids.write_bytes(np.arange(len(cells), dtype="<u2").tobytes())
                own = {"own_points_file": str(mine), "clusters_file": str(ids), "clusters": 4,
                       "clusters_matched": 4 - column["index"]}
                made = True
            columns.append({**own, "key": column["key"], "name": column["name"], "state": "ready", "error": None,
                            "cells_in_area": int(len(cells)), "tiles": None, "judged": int(len(cells)), "mixed": 1,
                            "mixed_per100": 10.0 + column["index"], "like100_per100": 8.5, "like100_sd": .2,
                            "like100_cells": int(len(cells)), "histogram": [len(cells), 0, 0, 0, 0, 0, 0],
                            "median_tx": 512.0, "under100_share": .05, "umap_points": int(len(cells)), "sampled": False,
                            "points_file": column["out_points"]})
        return {"ok": True, "run": request["run"], "seq": None,
                "compare": {"model": request["model_id"], "area": request["area"],
                            "kinds": [{"index": 0, "name": "ALB · APOA1 · TTR", "color": "#0072B2", "share": .6},
                                      {"index": 17, "name": "PTPRC · CD3E"}],
                            "kinds_reason": None, "kinds_error": None, "columns": columns, "layout": "pca",
                            "layout_error": None, "bounds": [-10, -10, 10, 10],
                            "own_layout": "pca" if made else None, "own_bounds": [-5, -5, 5, 5] if made else None,
                            "own_error": None if made or not request.get("own") else "too_few_cells"}}


def handed(column: dict) -> np.ndarray:
    """The centres (n x 2, as its counts file says them) of the cells the engine hands the worker for a shapes column:
    the counts file's rows it names (cells-<index>.npy), as the worker reads them (analysis/worker.py area_rows)."""
    with np.load(column["counts"]) as counted:
        rows = np.load(column["cells"])
        return np.c_[counted["x"][rows], counted["y"][rows]].astype(float)


def write_stats(path: Path, centres: list):
    n = len(centres)
    stats = np.zeros((7, n), np.uint32)
    stats[0] = 9
    xy = np.array(centres, "<f4").reshape(n, 2)
    stats[5], stats[6] = xy[:, 0].view("<u4"), xy[:, 1].view("<u4")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(stats.astype("<u4").tobytes()))


def write_plc(path: Path, ncells: int, iteration: int, f: bool = False) -> dict:
    """Proseg's count of a frame (PLC1, analysis/counts_file.py): one transcript of gene 0 in each cell, half of it
    kept; with `f`, proseg's own draw of it too (none kept). Its counts line's fields (write's stats)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return CF.write(path, ncells=ncells, ngenes=3, iteration=iteration, indptr=np.arange(ncells + 1),
                    genes=np.zeros(ncells, np.int64), n=np.ones(ncells, np.int64), e=np.full(ncells, .5),
                    f=np.zeros(ncells, np.int64) if f else None)


def add_run(runs: FakeRuns, root: Path, run_id="r1", centres=GRID, kind="proseg", frames=2, counted=None, spool=None,
            **meta) -> Run:
    """A run of `frames` frames, each with its cell statistics; those in `counted` (default: every one) with proseg's
    count, in the run's own folder (frames/<seq>.counts.zst, a lasting one: proseg's own draw too), or in `spool` (a
    folder outside the run: the engine's spool) for the seqs in it, as the engine's merge of counts.jsonl records
    them (counts_file, counts)."""
    run = Run(root / run_id, {"id": run_id, "sample": "zsample", "sample_kind": "zarr", "sample_path": "/x.zarr",
                              "kind": kind, "state": "running", **meta})
    run.live.mkdir(parents=True)
    (run.live / "transcripts.bin.gz").write_bytes(b"")
    spooled = spool[1] if spool else ()
    for seq in range(frames):
        write_stats(run.live / f"frames/{seq:06d}.cells.zst", centres)
        run.frames.append({"seq": seq, "iter": 10 * seq + 5, "file": f"frames/{seq:06d}.tiles", "format": "tiles",
                           "nx": 400, "ny": 200, "nz": 4, "x0": 0.0, "y0": 0.0, "voxel": 1.0,
                           "cells_file": f"frames/{seq:06d}.cells.zst"})
        if seq in (range(frames) if counted is None else counted):
            name = str(spool[0] / f"{seq:06d}.counts.zst") if seq in spooled else f"frames/{seq:06d}.counts.zst"
            line = write_plc(run.live / name, len(centres), 10 * seq + 5, f=seq not in spooled)
            run.frames[-1].update(counts_file=name, counted_file=f"frames/{seq:06d}.counted.zst",
                                  counts={"seq": seq, "iter": 10 * seq + 5, "file": name, **line})
    run.info = {"cells": len(centres), "z": {"grid_origin": 0.0, "grid_spacing": 0.25}}
    runs.runs[run_id] = run
    return run


@pytest.fixture
def setup(tmp_path):
    runs = FakeRuns()
    run = add_run(runs, tmp_path / "runs")
    zarr = FakeZarr(tmp_path, {"xenium": GRID, "cellpose": [c for c in GRID if c[0] < 200]})
    analysis = FakeAnalysis(runs)
    manager = C.CompareManager(runs, zarr, FakeQc(), analysis)
    yield manager, run, zarr, analysis
    manager.close()


def finished(manager, run_id="r1", timeout=20):
    wait_for(lambda: manager.describe(run_id)["state"] != "working", timeout=timeout)
    return manager.describe(run_id)


def test_the_run_and_counted_layers_are_scored_at_once_then_all_together(setup):
    manager, run, zarr, analysis = setup
    zarr.cached.add("cellpose")                     # counted before
    zarr.gates["xenium"] = threading.Event()        # counted now: held until the first answer is in
    analysis.hook = lambda request: zarr.gates["xenium"].set()
    first = manager.start("r1", {"columns": ["run", "xenium", "cellpose"]})
    assert first["state"] == "working" and [c["key"] for c in first["columns"]] == ["run", "xenium", "cellpose"]
    assert first["columns"][0]["name"] == "This run · iteration 15" and first["columns"][1]["name"] == "Layer xenium"
    assert first["layout_kind"] is None                                  # no layout yet
    done = finished(manager)
    assert done["state"] == "ready" and done["error"] is None and done["layout"] == 2
    assert done["layout_kind"] == "pca"            # the worker's word for its layout (the fake's numpy stand-in)
    assert [r["stage"] for r in analysis.requests] == [1, 2]
    assert [c["key"] for c in analysis.requests[0]["columns"]] == ["run", "cellpose"]
    assert [c["key"] for c in analysis.requests[1]["columns"]] == ["run", "xenium", "cellpose"]
    # the area: where all three hold 3 cells per 100 µm square (cellpose's x < 200)
    assert done["area"] == {"box": [0.0, 0.0, 200.0, 200.0], "size_um": [200.0, 200.0], "tiles": 4, "tile_um": 100.0}
    in_area = sum(1 for x, _ in GRID if x < 200)
    for column in done["columns"]:
        assert column["state"] == "ready" and column["cells_in_area"] == in_area and column["judged"] == in_area
        assert column["histogram"] == [in_area, 0, 0, 0, 0, 0, 0] and column["ready_s"] is not None
    assert [c["tiles"] for c in done["columns"]] == [8, 8, 4]
    assert done["kinds"] == [{"index": 0, "name": "ALB · APOA1 · TTR", "color": "#0072b2"},       # the worker's colour,
                             {"index": 17, "name": "PTPRC · CD3E", "color": C.KIND_COLOURS[1]}]  # else the engine's
    assert done["kinds_reason"] is None and done["bounds"] == [-10, -10, 10, 10] and done["layout_error"] is None
    assert done["columns"][0]["like100_sd"] == .2 and done["columns"][0]["mixed"] == 1
    assert done["progress"]["done"] == done["progress"]["of"] and done["step"] == "done"
    assert zarr.counted == [("xenium", None)]       # half the element or more: counted whole (any later area too)
    # what the worker was handed
    request = analysis.requests[1]
    assert request["model"] == "/models/m1" and request["model_id"] == FakeQc().model["id"]
    assert request["kinds"] == {"path": "/models/m1/kinds.npz", "table": {"zarr": "/x.zarr", "key": "xenium_table"}}
    assert request["max_points"] == C.MAX_POINTS and "op" not in request
    assert request["thresholds"] == {"mixed_at": .3, "judge_min": 100, "fit_min": 20}
    assert request["seed"] == C.draw_seed("r1", FakeQc().model) and request["area"] == done["area"]
    # The draws never depend on the columns chosen: a column keeps its like-for-like number when another is added.
    assert C.draw_seed("r1", FakeQc().model) == C.draw_seed("r1", {**FakeQc().model, "path": "/elsewhere"})
    assert C.draw_seed("r2", FakeQc().model) != C.draw_seed("r1", FakeQc().model)
    frame = request["columns"][0]["frame"]
    assert frame["seq"] == 1 and frame["format"] == "tiles" and frame["ncells"] == 50 and frame["run"] == "r1"
    assert frame["cells_file"].endswith("frames/000001.cells.zst") and frame["z_spacing"] == .25
    assert frame["frame"].endswith("frames/000001.tiles")
    assert frame["counts_file"] == str(run.live / "frames/000001.counts.zst")      # proseg's count of the frame
    run_cells = np.load(request["columns"][0]["cells"])
    assert sorted(np.array(GRID)[run_cells][:, 0].tolist()) == sorted(x for x, _ in GRID if x < 200)
    xenium = request["columns"][1]
    rows = np.load(xenium["cells"])                 # rows of the counts file (its own order) whose outline is inside
    index = np.load(xenium["counts"])["index"]
    assert sorted(np.array(GRID)[index[rows]][:, 0].tolist()) == sorted(x for x, _ in GRID if x < 200)
    # each column's newest points, and the result beside the run
    path = manager.points("r1", 1)
    assert path.name == "points-2-1.bin" and path.stat().st_size == 10 * in_area
    saved = json.loads((run.dir / "compare" / done["job"] / "result.json").read_text())
    assert saved["files"] == {"0": "points-2-0.bin", "1": "points-2-1.bin", "2": "points-2-2.bin"}
    with pytest.raises(RunError):
        manager.points("r1", 3)


def test_the_same_request_is_answered_from_the_comparison_it_made(setup, tmp_path):
    manager, run, zarr, analysis = setup
    manager.start("r1", {"columns": ["xenium", "run"]})
    done = finished(manager)
    assert done["state"] == "ready" and len(analysis.requests) == 2        # the run at once, then both together
    assert manager.start("r1", {"columns": ["xenium", "run"]})["job"] == done["job"]
    assert len(analysis.requests) == 2
    # another engine (a restart): found beside the run, the same numbers, no worker call
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)
    try:
        found = again.start("r1", {"columns": ["xenium", "run"]})
        assert found["state"] == "ready" and found["columns"] == done["columns"] and found["kinds"] == done["kinds"]
        assert found["area"] == done["area"] and again.points("r1", 1).name == "points-2-1.bin"
        assert len(analysis.requests) == 2
        # another frame, another model, other columns: new comparisons
        assert again.start("r1", {"columns": ["xenium", "run"], "seq": 0})["job"] != done["job"]
    finally:
        again.close()


def test_a_newer_request_replaces_the_one_on_its_way(setup):
    manager, run, zarr, analysis = setup
    zarr.gates["xenium"] = threading.Event()        # never released: only a stop ends it
    first = manager.start("r1", {"columns": ["xenium"]})
    wait_for(lambda: zarr.counted)
    second = manager.start("r1", {"columns": ["cellpose", "run"]})
    assert second["job"] != first["job"]
    done = finished(manager)
    assert done["job"] == second["job"] and done["state"] == "ready"
    assert zarr.stopped == ["xenium"]               # its helper was stopped
    assert all(c["key"] != "xenium" for r in analysis.requests for c in r["columns"])


def test_requests_that_cannot_be_compared_are_refused(setup, tmp_path):
    manager, run, zarr, analysis = setup

    def status(run_id, body):
        with pytest.raises(RunError) as caught:
            manager.start(run_id, body)
        return caught.value.status

    for bad in ({"columns": []}, {"columns": ["run"] * 2}, {"columns": ["run", "xenium", "cellpose", "a", "b"]},
                {"columns": [1]}, {"columns": "run"}, {}, {"columns": ["run"], "seq": "1"}):
        assert status("r1", bad) == 422, bad
    assert status("r1", {"columns": ["nope"]}) == 404
    listing = zarr.segmentations
    zarr.segmentations = lambda sid: [{**s, "version": C.UNREAD} if s["key"] == "xenium" else s for s in listing(sid)]
    assert status("r1", {"columns": ["run", "xenium"]}) == 503      # its folder briefly unreadable: no comparison kept
    zarr.segmentations = listing
    assert status("r1", {"columns": ["run"], "seq": 9}) == 404
    assert status("missing", {"columns": ["run"]}) == 404
    add_run(manager.runs, tmp_path / "runs", "stain", kind="stain")
    assert status("stain", {"columns": ["run"]}) == 404
    add_run(manager.runs, tmp_path / "runs", "early", frames=0)
    assert status("early", {"columns": ["run"]}) == 409
    assert manager.start("early", {"columns": ["xenium"]})["state"] == "working"   # the layers alone need no frame
    manager.qc.model = None
    assert status("r1", {"columns": ["run"]}) == 409
    manager.qc.model = FakeQc().model
    analysis.available = False
    assert status("r1", {"columns": ["run"]}) == 503
    # Where Clusters and the UMAP cannot run, the refusal (on screen) says so in the engine's words, one colon.
    analysis.unavailable_why = lambda: "this Python environment lacks igraph and umap-learn"
    with pytest.raises(RunError) as caught:
        manager.start("r1", {"columns": ["run"]})
    assert caught.value.status == 503 and str(caught.value) == (
        "Compare needs Clusters and the UMAP, which can't run here: this Python environment lacks igraph and umap-learn.")
    with pytest.raises(RunError) as caught:
        manager.describe("r1")
    assert caught.value.status == 404


def test_a_column_that_cannot_be_read_or_counted_leaves_the_others(setup):
    manager, run, zarr, analysis = setup
    zarr.layers["broken"] = GRID
    zarr.fail["broken"] = "Could not read the outlines"
    zarr.layers["bad"] = GRID
    manager.start("r1", {"columns": ["run", "broken", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and [c["state"] for c in done["columns"]] == ["ready", "failed", "ready"]
    assert done["columns"][1]["error"] == "Could not read the outlines"
    zarr.fail_counts["bad"] = "the helper failed"   # outlines read, its counting fails
    manager.start("r1", {"columns": ["run", "bad"]})
    done = finished(manager)
    assert [c["state"] for c in done["columns"]] == ["ready", "failed"] and done["columns"][1]["error"] == "the helper failed"
    assert done["state"] == "ready" and done["layout"] == 1


def test_failures_reach_the_page_in_plain_words(setup, capsys):
    """compare.mjs shows a column's error as it comes: an engine exception or the GPU worker's technical text is said
    plainly there (the app's own errors keep their words, above), and its own words go to the engine's log."""
    manager, run, zarr, analysis = setup
    zarr.layers["odd"] = GRID
    outlines_of, answer = zarr._outlines, analysis.answer

    def outlines_failing(sid, key, should_stop=None):
        if key == "odd":
            raise KeyError("offsets")
        return outlines_of(sid, key, should_stop=should_stop)

    def worker_failing(request):
        reply = answer(request)
        for column in reply["compare"]["columns"]:
            if column["key"] == "cellpose":
                column.update(state="failed", error="FileNotFoundError: /models/m1/encoder.npz")
        return reply

    zarr._outlines, analysis.answer = outlines_failing, worker_failing
    manager.start("r1", {"columns": ["run", "odd", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready"
    assert {c["key"]: c["error"] for c in done["columns"]} == {
        "run": None, "odd": "Could not read its outlines", "cellpose": "The GPU worker could not read it"}
    log = capsys.readouterr().out
    assert "KeyError: 'offsets'" in log and "encoder.npz" in log


def test_columns_with_no_tissue_in_common_and_worker_failures_fail_the_comparison(setup):
    manager, run, zarr, analysis = setup
    zarr.layers["far"] = [(5000 + x, 5000 + y) for x, y in GRID]
    manager.start("r1", {"columns": ["run", "far"]})
    done = finished(manager)
    assert done["state"] == "failed" and "share no tissue" in done["error"]
    assert all(c["state"] == "failed" for c in done["columns"]) and analysis.requests == []
    analysis.reply = {"ok": False, "errors": {"compare": "ValueError: no column could be counted"},
                      "error": "ValueError: no column could be counted"}
    manager.start("r1", {"columns": ["run"]})
    done = finished(manager)                                   # the worker's technical words stay in the log
    assert done["state"] == "failed" and done["error"] == "The GPU worker could not compare these"
    analysis.reply, analysis.hold = None, True                 # no reply came (the worker stopped): the Future says so
    manager.start("r1", {"columns": ["run"]})
    wait_for(lambda: analysis.futures and not analysis.futures[-1].done())
    analysis.futures[-1].set_exception(AnalysisError("the GPU worker timed out"))
    done = finished(manager)
    assert done["state"] == "failed" and done["error"] == "the GPU worker timed out"
    # The first of two calls failing (the run alone, while cellpose is counted) fails the run's column alone:
    # cellpose is still counted and scored in the next call.
    asked = len(analysis.futures)
    manager.start("r1", {"columns": ["run", "cellpose"]})
    wait_for(lambda: len(analysis.futures) > asked and not analysis.futures[-1].done())
    analysis.futures[-1].set_exception(AnalysisError("the GPU worker timed out"))
    wait_for(lambda: len(analysis.futures) > asked + 1)
    assert [c["key"] for c in analysis.requests[-1]["columns"]] == ["cellpose"]
    second = analysis.futures[-1]
    second.set_running_or_notify_cancel()
    second.set_result(analysis.answer(analysis.requests[-1]))
    done = finished(manager)
    assert done["state"] == "ready" and [(c["state"], c["error"]) for c in done["columns"]] == [
        ("failed", "the GPU worker timed out"), ("ready", None)]
    assert not (run.dir / "compare" / done["job"] / "result.json").exists()        # a part failed: never kept
    analysis.hold = False
    assert manager.start("r1", {"columns": ["run", "cellpose"]})["job"] == done["job"]   # asked again: done again
    assert finished(manager)["state"] == "ready" and all(c["state"] == "ready" for c in finished(manager)["columns"])
    assert (run.dir / "compare" / done["job"] / "result.json").exists()            # whole now: kept
    # a failed comparison is tried again when asked again
    analysis.reply = None
    manager.start("r1", {"columns": ["run"]})
    assert finished(manager)["state"] == "ready"


def test_a_model_without_details_has_no_kinds_and_says_why(setup):
    manager, run, zarr, analysis = setup
    manager.qc = FakeQc(deep=False)
    original = analysis.answer

    def without_kinds(request):
        reply = original(request)
        return {**reply, "compare": {**reply["compare"], "kinds": None, "kinds_reason": "needs_detail"}}
    analysis.answer = without_kinds
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and done["kinds"] is None and done["kinds_reason"] == "needs_detail"
    first = manager.start("r1", {"columns": ["run", "cellpose"]})["job"]
    manager.qc = FakeQc(deep=True)                             # the details learned: another comparison, with kinds
    analysis.answer = original
    assert manager.start("r1", {"columns": ["run", "cellpose"]})["job"] != first
    assert finished(manager)["kinds"] is not None


def test_a_region_run_is_compared_inside_its_region(setup, tmp_path):
    manager, run, zarr, analysis = setup
    add_run(manager.runs, tmp_path / "runs", "region", mode="detail", box=[0, 0, 150, 200])
    manager.start("region", {"columns": ["run", "xenium"]})
    done = finished(manager, "region")
    assert done["state"] == "ready" and done["area"]["box"] == [0.0, 0.0, 150.0, 200.0]
    assert [c["cells_in_area"] for c in done["columns"]] == [20, 20]         # x 20, 60, 100, 140: Xenium's too
    request = analysis.requests[-1]
    xenium = request["columns"][1]
    rows = np.load(xenium["cells"])
    assert (np.array(GRID)[np.load(xenium["counts"])["index"][rows]][:, 0] < 150).all()
    assert zarr.counted[-1] == ("xenium", [0.0, 0.0, 150.0, 200.0])           # its box: under half of the element


def test_every_kind_of_region_limits_every_column_lassos_by_their_polygon(setup, tmp_path):
    """A tiled rectangle (over 1,200 µm) limits the columns like a detail one, and a lasso
    by its polygon, not only its bounds; a full-sample run by nothing."""
    manager, run, zarr, analysis = setup
    add_run(manager.runs, tmp_path / "runs", "tiled", mode="tiled", box=[0, 0, 150, 200],
            region={"kind": "box", "box": [0, 0, 150, 200], "bbox": [0, 0, 150, 200]})
    manager.start("tiled", {"columns": ["run", "xenium"]})
    done = finished(manager, "tiled")
    assert done["area"]["box"] == [0.0, 0.0, 150.0, 200.0] and [c["cells_in_area"] for c in done["columns"]] == [20, 20]
    # A triangle over the left half: (0, 0), (300, 0), (0, 200): a cell counts when its centre is inside it.
    triangle = [[0, 0], [300, 0], [0, 200]]
    add_run(manager.runs, tmp_path / "runs", "lasso", mode="detail", box=[0, 0, 300, 200],
            region={"kind": "polygon", "points": triangle, "bbox": [0, 0, 300, 200]})
    manager.start("lasso", {"columns": ["run", "xenium"]})
    done = finished(manager, "lasso")
    xs, ys = np.array(GRID, float).T
    inside = C.in_polygon(xs, ys, np.array(triangle, float))
    assert 0 < inside.sum() < len(GRID)
    request = analysis.requests[-1]
    for column in request["columns"]:
        rows = np.load(column["cells"])
        labels = rows if column["key"] == "run" else np.load(column["counts"])["index"][rows]
        assert inside[labels].all(), column["key"]                     # nothing outside the lasso
    add_run(manager.runs, tmp_path / "runs", "full", mode="tiled", box=[0, 0, 400, 200],
            region={"kind": "full", "bbox": [0, 0, 400, 200]})
    assert C.region_limit(manager.runs.get("full")) == (None, None)
    assert C.region_limit(manager.runs.get("tiled"))[0] == [0.0, 0.0, 150.0, 200.0]


def test_the_area_is_never_said_larger_than_the_sample(tmp_path):
    """A full-sample run's area was whole 100 µm tiles from (0, 0), so Compare stated more than the sample
    itself. Its box and size stay within the sample's extent and the run's own bounds; the cells in it stay the tiles'."""
    grid = np.mgrid[5:246:10, 5:246:10].reshape(2, -1).T.astype(float)        # 25 x 25 cells over 0-250 µm
    centres = (grid[:, 0], grid[:, 1])
    extent = [0.0, 0.0, 246.3, 246.25]
    assert C.common_area({"a": centres, "b": centres})["box"] == [0.0, 0.0, 300.0, 300.0]   # whole tiles
    area = C.common_area({"a": centres, "b": centres}, clip=extent)
    assert area["box"] == extent and area["size_um"] == [246.3, 246.25] and area["count"] == 9
    assert C.in_area(*centres, area).all()
    assert C.common_area({"a": centres}, clip=[1000, 1000, 1100, 1100])["box"] == [0.0, 0.0, 300.0, 300.0]
    runs = FakeRuns()
    run = add_run(runs, tmp_path / "runs", "full", centres=grid.tolist(), mode="tiled", box=extent,
                  region={"kind": "full", "bbox": extent})
    assert C.area_clip(run, None) == extent and C.area_clip(run, [0, 0, 200, 300]) == [0.0, 0.0, 200.0, 246.25]
    assert C.area_clip(run, [0, 0, 200, 300], with_run=False) == [0.0, 0.0, 200.0, 300.0]
    assert C.area_clip(SimpleNamespace(meta={}), None) is None and C.area_clip(run, [0, 0, float("nan"), 1]) == extent
    zarr = SizedZarr(tmp_path, {"xenium": [tuple(c) for c in grid]}, extent)
    manager = C.CompareManager(runs, zarr, FakeQc(), FakeAnalysis(runs))
    try:
        manager.start("full", {"columns": ["run", "xenium"]})
        done = finished(manager, "full")
        assert done["state"] == "ready" and done["area"] == {"box": extent, "size_um": [246.3, 246.25], "tiles": 9,
                                                             "tile_um": 100.0}
        assert [c["cells_in_area"] for c in done["columns"]] == [len(grid)] * 2
        manager.start("full", {"columns": ["xenium"]})                 # the layers alone: within the sample still
        assert finished(manager, "full")["area"]["box"] == extent
    finally:
        manager.close()


def test_compare_says_why_the_mixing_check_has_no_model(setup):
    """Compare answered "turn it on and wait until it has learned this sample" when the mixing check had failed
    to learn it, or could not learn it at all: waiting never helps then, so the refusal names why."""
    manager, run, zarr, analysis = setup
    manager.qc.model = None

    class Check:
        def __init__(self, state, reason=None):
            self.state, self.reason = state, reason

        def describe(self, sid):
            if self.state is None:
                raise KeyError(sid)
            return {"sample": sid, "state": self.state, "reason": self.reason}

    for state, reason, words in (
            ("failed", "not_settled", "which could not learn this sample: the learning did not settle. Try again from "
                                      "the mixing check."),
            ("failed", "a_new_reason", "which could not learn this sample: it ran into a problem."),
            ("unavailable", "no_table", "which cannot learn this sample: this sample has no table of cells with "
                                        "transcript counts."),
            # a table whose cells cannot be placed on the transcripts (the trainer's refusal): said in words
            ("unavailable", "positions", "which cannot learn this sample: the sample's cell table doesn't say where "
                                         "its cells lie on the transcripts."),
            ("failed", "too_few_cells", "which could not learn this sample: it has too few cells to learn from."),
            # what this computer lacks to learn on (its packages, a graphics card it can use, a Mac GPU this PyTorch can
            # use, PyTorch's CUDA build, a driver new enough): what to set up, in plain words, never naming the method
            ("unavailable", "software", "Compare needs the mixing check, which isn't set up on this computer: it needs "
                                        "scvi-tools and scviva-tools, in this Python environment or in one named as "
                                        "python under [qc] in the app's config.toml."),
            ("unavailable", "no_gpu", "Compare needs the mixing check, which isn't set up on this computer: it learns on "
                                      "an NVIDIA graphics card or a Mac's own GPU, and this computer has none it can "
                                      "use."),
            ("unavailable", "apple_gpu", "Compare needs the mixing check, which isn't set up on this computer: it learns "
                                         "on this Mac's GPU, which failed a short test with this computer's PyTorch "
                                         "(the Log says why; on macOS 14, installing the app again brings PyTorch "
                                         "2.14, which passes it)."),
            ("unavailable", "torch_cpu", "Compare needs the mixing check, which isn't set up on this computer: it learns "
                                         "on an NVIDIA graphics card, which this computer's PyTorch can't use until its "
                                         "CUDA build is installed."),
            ("unavailable", "driver", "Compare needs the mixing check, which isn't set up on this computer: it learns on "
                                      "an NVIDIA graphics card, whose driver is too old for this computer's PyTorch and "
                                      "needs updating."),
            ("learning", None, "mixing check: wait until it has learned this sample"),
            ("off", None, "turn it on and wait until it has learned this sample"),
            (None, None, "turn it on and wait until it has learned this sample")):
        manager.qc.models = Check(state, reason)
        with pytest.raises(RunError) as refused:
            manager.start("r1", {"columns": ["run"]})
        assert refused.value.status == 409 and words in str(refused.value), (state, str(refused.value))
        if state == "unavailable" and reason in C.CHECK_SETUP:      # the whole sentence: one colon, no method named
            assert str(refused.value) == words and str(refused.value).count(":") == 1
            assert not any(word in str(refused.value).lower() for word in ("resolvi", "latent", "posterior",
                                                                            "diffusion", "embedding", "prior"))
    assert not set(C.CHECK_SETUP) & set(C.CHECK_UNAVAILABLE)       # said by its own branch only: no second copy


def test_a_forgotten_sample_is_opened_again_without_listing_it(setup):
    """Opening a run's forgotten sample again for Compare put it back on top of the open dialog's list."""
    manager, run, zarr, analysis = setup
    opened = []
    zarr.known = lambda sid: False
    zarr.open = lambda path, recent=True: opened.append((path, recent))
    manager.start("r1", {"columns": ["run"]})
    assert finished(manager)["state"] == "ready" and opened == [("/x.zarr", False)]


def test_points_inside_a_polygon_by_the_even_odd_rule():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    got = C.in_polygon(np.array([5, 15, -1, 5, np.nan]), np.array([5, 5, 5, 11, 5]), square)
    assert got.tolist() == [True, False, False, False, False]
    ring = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [3, 3], [3, 7], [7, 7], [7, 3], [3, 3]], float)
    assert C.in_polygon(np.array([5.0, 1.0]), np.array([5.0, 2.0]), ring).tolist() == [False, True]   # its hole


def test_every_column_is_counted_by_the_runs_own_input_filters(setup, tmp_path):
    """The run's column is proseg's own input (its min-qv, its excluded genes): every shapes
    column its outlines count is counted by the same rule, the answer names it, and another rule is another
    comparison."""
    manager, run, zarr, analysis = setup
    assert C.input_rule(run) == {"min_qv": 20.0, "exclude": None} == C.DEFAULT_RULE
    manager.start("r1", {"columns": ["run", "xenium"]})
    default = finished(manager)
    assert default["rule"] == {"min_qv": 20.0, "exclude": None} and set(map(json.dumps, zarr.rules)) == {
        json.dumps({"min_qv": 20.0, "exclude": None})}
    strict = add_run(manager.runs, tmp_path / "runs", "strict", options={"min-qv": 30, "excluded-genes": "^MALAT1$"})
    assert C.input_rule(strict) == {"min_qv": 30.0, "exclude": "^MALAT1$"}
    zarr.rules.clear()
    manager.start("strict", {"columns": ["run", "xenium"]})
    done = finished(manager, "strict")
    assert done["rule"] == {"min_qv": 30.0, "exclude": "^MALAT1$"} and zarr.rules
    assert all(r == {"min_qv": 30.0, "exclude": "^MALAT1$"} for r in zarr.rules)
    plain = add_run(manager.runs, tmp_path / "runs", "plain", options={"excluded-genes": C.DEFAULT_EXCLUDED, "min-qv": "x"})
    assert C.input_rule(plain) == C.DEFAULT_RULE                       # proseg's own default pattern; a bad value
    keys = [C.job_key(FakeQc().model, ["run"], {}, 1, rule) for rule in (C.DEFAULT_RULE, {"min_qv": 30.0, "exclude": None})]
    assert C.digest_of(keys[0]) != C.digest_of(keys[1])


# ---- each column by its own count --------------------------------------------------------------------------------------
def test_each_column_reads_its_own_count_and_says_whose(setup, tmp_path):
    """Each segmentation by its own count: the run's column is proseg's own count, a saved proseg result's its table's
    X, Xenium's cells Xenium's table, a layer with no count of its own the transcripts inside its outlines. Each column
    says whose (`count`) from the start, kept with the result; a table's column is asked of the sample by its table
    (ZarrSamples.counts reads that table's own count), and fails in words of its own."""
    manager, run, zarr, analysis = setup
    zarr.layers["saved"] = GRID
    zarr.sources = {"xenium": {"count_by": "xenium", "count_table": "xenium_table"},
                    "saved": {"count_by": "proseg", "count_table": "proseg_table"}}
    first = manager.start("r1", {"columns": ["run", "xenium", "cellpose", "saved"]})
    assert [c["count"] for c in first["columns"]] == ["proseg", "xenium", "outlines", "proseg"]
    done = finished(manager)
    assert done["state"] == "ready" and all(c["state"] == "ready" for c in done["columns"])
    assert [c["count"] for c in done["columns"]] == ["proseg", "xenium", "outlines", "proseg"]
    assert zarr.own == {"xenium": ("xenium", "xenium_table"), "cellpose": ("outlines", None),
                        "saved": ("proseg", "proseg_table")}
    request = analysis.requests[-1]
    assert [c.get("counts", "").endswith(f"{c['key']}.npz") for c in request["columns"][1:]] == [True] * 3
    assert "count" not in request["columns"][1]                # the worker reads every counts file the same way
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)      # kept beside the run: each column says whose
    try:
        found = again.start("r1", {"columns": ["run", "xenium", "cellpose", "saved"]})
        assert found["state"] == "ready" and [c["count"] for c in found["columns"]] == ["proseg", "xenium", "outlines",
                                                                                        "proseg"]
    finally:
        again.close()
    # A listing that names no table, or a count no table holds, is its outlines'.
    zarr.sources = {"xenium": {"count_by": "xenium", "count_table": None}, "saved": {"count_by": "other"}}
    manager.start("r1", {"columns": ["xenium", "saved"]})
    assert [c["count"] for c in finished(manager)["columns"]] == ["outlines", "outlines"]
    assert zarr.own["xenium"] == zarr.own["saved"] == ("outlines", None)
    # A table's own count that cannot be read fails its column alone, in words of its own.
    zarr.sources = {"saved": {"count_by": "proseg", "count_table": "proseg_table"}}
    counts_of = zarr.counts

    def failing(sid, key, *a, **kw):
        if key == "saved" and not kw.get("cached_only"):
            raise HelperFailed("ValueError: whole counts")
        return counts_of(sid, key, *a, **kw)
    zarr.counts = failing
    manager.start("r1", {"columns": ["run", "saved", "cellpose"]})
    done = finished(manager)
    assert [(c["state"], c["error"]) for c in done["columns"]] == [("ready", None), ("failed", "Could not read its counts"),
                                                                   ("ready", None)]


def test_the_run_column_is_proseg_count_of_the_nearest_counted_frame(setup, tmp_path):
    """The run's column reads proseg's own count of a frame: the frame asked for, or the newest, at the nearest frame
    with that count (AnalysisManager.compare_frame), whose seq and iteration the comparison says. A count in the
    engine's spool (outside the run's folder: only the newest are kept) is held beside the comparison for the worker,
    and goes with it; a lasting one is read where it is. Without a frame with proseg's count the run is refused in the
    scheduler's words."""
    manager, _, zarr, analysis = setup
    spool = tmp_path / "spool"
    run = add_run(manager.runs, tmp_path / "runs", "live", frames=4, counted=[0, 2], spool=(spool, [2]))
    manager.start("live", {"columns": ["run", "cellpose"], "seq": 1})        # frame 1 has none: frame 0's
    done = finished(manager, "live")
    assert done["state"] == "ready" and (done["seq"], done["iter"]) == (0, 5) and analysis.framed[-1] == ("live", 1)
    assert done["columns"][0]["name"] == "This run · iteration 5" and done["columns"][0]["count"] == "proseg"
    frame = analysis.requests[-1]["columns"][0]["frame"]
    assert frame["seq"] == 0 and frame["counts_file"] == str(run.live / "frames/000000.counts.zst")   # read where it is
    assert not list((run.dir / "compare" / done["job"]).glob("*counts*"))
    # The newest: frame 2, its count in the spool. Held before the spool clears it, the worker reads the copy.
    zarr.gates["cellpose"] = gate = threading.Event()       # counted now: the run's column scored first
    original = (spool / "000002.counts.zst").read_bytes()
    held = []

    def kept(request):
        path = Path(request["columns"][0]["frame"]["counts_file"])
        held.append((path, path.read_bytes() if path.exists() else None))
        gate.set()
    analysis.hook = kept
    started = manager.start("live", {"columns": ["run", "cellpose"]})
    assert (started["seq"], started["iter"]) == (2, 25) and started["columns"][0]["name"] == "This run · iteration 25"
    wait_for(lambda: held)
    (spool / "000002.counts.zst").unlink()                  # cleared from the spool: the comparison holds its own
    done = finished(manager, "live")
    wait_for(lambda: not manager.jobs["live"].thread.is_alive())
    assert done["state"] == "ready" and all(c["state"] == "ready" for c in done["columns"])
    assert all(path.parent == run.dir / "compare" / done["job"] and path.name.startswith("run-") and data == original
               for path, data in held) and len(held) == 2
    assert not held[0][0].exists() and not list((run.dir / "compare" / done["job"]).glob("*counts*"))   # gone with it
    # Cleared before it could be held: the run's column fails in plain words, the other columns are compared.
    add_run(manager.runs, tmp_path / "runs", "cleared", frames=2, spool=(tmp_path / "spool2", [1]))
    analysis.hook, analysis.after_frame = None, lambda column: Path(column["counts_file"]).unlink()
    manager.start("cleared", {"columns": ["run", "cellpose"]})
    done = finished(manager, "cleared")
    analysis.after_frame = None
    assert done["state"] == "ready" and [(c["state"], c["error"]) for c in done["columns"]] == [
        ("failed", C.GONE), ("ready", None)]
    assert not (manager.runs.get("cleared").dir / "compare" / done["job"] / "result.json").exists()
    # No frame with proseg's count (a run from before it was kept): refused in the scheduler's words.
    add_run(manager.runs, tmp_path / "runs", "older", frames=3, counted=[])
    with pytest.raises(RunError) as refused:
        manager.start("older", {"columns": ["run", "cellpose"]})
    assert refused.value.status == 409 and str(refused.value) == "This run has no frame with proseg's count"
    assert manager.start("older", {"columns": ["cellpose"]})["state"] == "working"   # the layers alone need no frame
    finished(manager, "older")


def test_a_comparison_is_keyed_by_whose_count_each_column_reads(setup, tmp_path):
    """What decides a column's numbers is in the comparison's key: whose count it reads, the table and its version
    (a table written again is compared again), whether the run's frame's count holds proseg's own draw (the like for
    like reads it, else draws its own), the live form of proseg's count (LIVE_COUNT, one switch), and RESULT_VERSION 4:
    no comparison of the outline count era is reused."""
    manager, run, zarr, analysis = setup
    assert C.RESULT_VERSION == 4 and C.job_key(FakeQc().model, ["run"], {}, 1)["count"] == CF.LIVE_COUNT
    base = dict(model=FakeQc().model, columns=["run", "xenium"], stamps={"xenium": "v1"}, seq=1)
    keys = {name: C.digest_of(C.job_key(**base, counted=counted)) for name, counted in {
        "plain": {"run": ["proseg", True], "xenium": ["xenium", "xenium_table", "t1"]},
        "drawn": {"run": ["proseg", False], "xenium": ["xenium", "xenium_table", "t1"]},
        "rewritten": {"run": ["proseg", True], "xenium": ["xenium", "xenium_table", "t2"]},
        "outlines": {"run": ["proseg", True], "xenium": ["outlines", None, None]}}.items()}
    assert len(set(keys.values())) == 4
    zarr.sources = {"xenium": {"count_by": "xenium", "count_table": "xenium_table"}}
    first = manager.start("r1", {"columns": ["run", "xenium"]})["job"]
    assert finished(manager)["state"] == "ready" and manager.start("r1", {"columns": ["run", "xenium"]})["job"] == first
    zarr.table_versions["xenium_table"] = "t2"                       # Xenium's table written again
    second = manager.start("r1", {"columns": ["run", "xenium"]})["job"]
    assert second != first and finished(manager)["state"] == "ready"
    with run.lock:
        run.frames[1]["counts"] = {**run.frames[1]["counts"], "has_f": False}   # its count without proseg's draw
    assert manager.start("r1", {"columns": ["run", "xenium"]})["job"] not in (first, second)
    finished(manager)


def test_the_key_says_whether_the_count_the_worker_reads_holds_proseg_draw(setup, tmp_path):
    """A seq's count in the spool can be counted again for good while the comparison reads its frame (a hold, the
    finish, the end: the same count with proseg's draw, in the run's own folder, its counts line then naming that
    file). The frame is asked for again until its line names the file given, so the key says what the worker reads:
    the like for like reads proseg's draw, and the next request (naming that file) is answered from the comparison.
    A line naming yet another file each time is refused after FRAME_TRIES, in plain words, nothing asked of the
    worker."""
    manager, _, zarr, analysis = setup
    spool = tmp_path / "spool"
    run = add_run(manager.runs, tmp_path / "runs", "live", frames=3, counted=[2], spool=(spool, [2]))
    durable = write_plc(run.live / "frames/000002.counts.zst", len(GRID), 25, f=True)
    zarr.cached.add("cellpose")

    def recount(column):                                    # merged right after the frame was chosen, once
        analysis.after_frame = None
        with run.lock:
            run.frames[2].update(counts_file="frames/000002.counts.zst",
                                 counts={**run.frames[2]["counts"], "file": "frames/000002.counts.zst", **durable})
    analysis.after_frame = recount
    drawn = []
    analysis.hook = lambda request: drawn.append(CF.read(request["columns"][0]["frame"]["counts_file"]).f is not None)
    first = manager.start("live", {"columns": ["run", "cellpose"]})
    assert analysis.framed == [("live", None)] * 2 and first["seq"] == 2   # asked again: its line names another file
    assert finished(manager, "live")["state"] == "ready" and drawn == [True]
    assert analysis.requests[-1]["columns"][0]["frame"]["counts_file"] == str(run.live / "frames/000002.counts.zst")
    again = manager.start("live", {"columns": ["run", "cellpose"]})
    assert again["job"] == first["job"] and len(analysis.requests) == 1
    names = itertools.count()

    def moving(column):                                     # another file each time it is read
        name = f"frames/again-{next(names)}.counts.zst"
        shutil.copyfile(run.live / "frames/000002.counts.zst", run.live / name)
        with run.lock:
            run.frames[2].update(counts_file=name, counts={**run.frames[2]["counts"], "file": name})
    analysis.after_frame, asked = moving, len(analysis.framed)
    with pytest.raises(RunError) as refused:
        manager.start("live", {"columns": ["run", "cellpose"], "seq": 2})
    assert refused.value.status == 409 and str(refused.value) == C.RECOUNTED == (
        "Proseg's count of this iteration changed while it was read: compare again")
    assert len(analysis.framed) - asked == C.FRAME_TRIES and len(analysis.requests) == 1
    analysis.after_frame = None                             # settled: the same count, the same comparison
    assert manager.start("live", {"columns": ["run", "cellpose"], "seq": 2})["job"] == first["job"]


def test_the_run_column_reads_the_count_its_frame_is_read_on_and_says_whose_draw(setup, tmp_path, monkeypatch):
    """The run's column reads the count its frame is read on (counts_file.count_of of its counts line, the frame the
    worker gets saying it: `count`): a frame proseg counted for a pause, the hold, a finish or the end is read on f, the
    count proseg saves (`form` "sampled"), any other on e ("averaged"); `like_from` says whose draw the like-for-like
    cut starts from: "proseg" (its own f, held in the frame's file) or "app" (one the app draws from proseg's count).
    An averaged frame's key is as it was (the run's entry [RUN_COUNT, drawn]); the hold counting the frame again on f
    is another comparison. A comparison kept before these fields says them too (its key holds both)."""
    manager, _, zarr, analysis = setup
    zarr.cached.add("cellpose")
    counted = []
    keyed = C.job_key
    monkeypatch.setattr(C, "job_key", lambda *a, **k: counted.append(a[7]) or keyed(*a, **k))
    run = add_run(manager.runs, tmp_path / "runs", "live", frames=3)            # durable counts, with proseg's draw
    first = manager.start("live", {"columns": ["run", "cellpose"]})
    done = finished(manager, "live")
    column = done["columns"][0]
    assert (column["form"], column["like_from"], column["count"]) == ("averaged", "proseg", "proseg")
    assert analysis.requests[-1]["columns"][0]["frame"]["count"] == "averaged" and counted[-1]["run"] == ["proseg", True]
    assert "form" not in done["columns"][1] and "like_from" not in done["columns"][1]
    with run.lock:                                          # the hold counts frame 2 again, with f
        run.frames[2] = {**run.frames[2], "counts": {**run.frames[2]["counts"], "reason": "hold", "has_f": True}}
    held = manager.start("live", {"columns": ["run", "cellpose"]})
    done = finished(manager, "live")
    assert held["job"] != first["job"] and counted[-1]["run"] == ["proseg", True, "sampled"]
    assert (done["columns"][0]["form"], done["columns"][0]["like_from"]) == ("sampled", "proseg")
    assert analysis.requests[-1]["columns"][0]["frame"]["count"] == "sampled"
    # A comparison kept before these fields (an earlier version's result.json): found again, and says them.
    saved = run.dir / "compare" / held["job"] / "result.json"
    stored = json.loads(saved.read_text())
    stored["columns"] = [{k: v for k, v in c.items() if k not in C.RUN_FIELDS} for c in stored["columns"]]
    saved.write_text(json.dumps(stored))
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)
    try:
        asked = len(analysis.requests)
        found = again.start("live", {"columns": ["run", "cellpose"]})
        assert found["job"] == held["job"] and found["state"] == "ready" and len(analysis.requests) == asked
        assert (found["columns"][0]["form"], found["columns"][0]["like_from"]) == ("sampled", "proseg")
    finally:
        again.close()
    # A count in the spool without proseg's draw: the app draws one.
    add_run(manager.runs, tmp_path / "runs", "asked", frames=2, spool=(tmp_path / "spool", [1]))
    manager.start("asked", {"columns": ["run", "cellpose"]})
    done = finished(manager, "asked")
    assert (done["columns"][0]["form"], done["columns"][0]["like_from"]) == ("averaged", "app")
    assert counted[-1]["run"] == ["proseg", False]


def test_a_frame_without_proseg_count_is_never_sent_to_the_worker(setup):
    """A frame given without proseg's count (or without its cell statistics) is refused (409) before any comparison
    starts: the run's column is never handed to the worker without proseg's count."""
    manager, run, zarr, analysis = setup
    framed = analysis.compare_frame
    for change in ({"counts_file": None}, {"counts_file": ""}, {"counts_file": ...}, {"cells_file": None}):
        def without(run_id, seq=None, change=change):    # ...: the field left out
            column = framed(run_id, seq)
            for name, value in change.items():
                if value is ...:
                    del column[name]
                else:
                    column[name] = value
            return column
        analysis.compare_frame = without
        with pytest.raises(RunError) as refused:
            manager.start("r1", {"columns": ["run", "cellpose"]})
        assert refused.value.status == 409 and str(refused.value) == (
            "This iteration cannot be compared (no cell statistics or proseg's count)"), change
    assert analysis.requests == [] and "r1" not in manager.jobs and not (run.dir / "compare").exists()


def test_a_comparison_waiting_for_the_runs_previous_one_holds_its_frame_first(setup, tmp_path):
    """A comparison replacing one whose worker call is running waits for that call (on the CPU or a whole sample: a
    while). Its frame's count, in the engine's spool, is held as its work starts, before that wait: the spool clearing
    it meanwhile never fails the run's column, and the copy goes with the comparison."""
    manager, _, zarr, analysis = setup
    spool = tmp_path / "spool"
    add_run(manager.runs, tmp_path / "runs", "live", frames=3, counted=[2], spool=(spool, [2]))
    zarr.cached.update({"xenium", "cellpose"})
    analysis.hold = True
    manager.start("live", {"columns": ["run", "cellpose"]})                  # its worker call runs, unanswered
    wait_for(lambda: analysis.futures)
    analysis.futures[0].set_running_or_notify_cancel()
    started = manager.start("live", {"columns": ["run", "xenium"]})
    job = manager.jobs["live"]
    assert started["seq"] == 2 and job.digest == started["job"]
    wait_for(lambda: job.held is not None, timeout=5)                        # held before the wait
    assert job.thread.is_alive() and not analysis.futures[0].done() and len(analysis.requests) == 1
    assert job.held.parent == job.folder and job.held.read_bytes() == (spool / "000002.counts.zst").read_bytes()
    (spool / "000002.counts.zst").unlink()                                   # cleared from the spool meanwhile
    analysis.hold = False
    analysis.futures[0].set_result(analysis.answer(analysis.requests[0]))
    done = finished(manager, "live")
    assert done["job"] == started["job"] and [(c["key"], c["state"], c["error"]) for c in done["columns"]] == [
        ("run", "ready", None), ("xenium", "ready", None)]
    assert analysis.requests[-1]["columns"][0]["frame"]["counts_file"] == str(job.held)
    wait_for(lambda: not job.thread.is_alive())
    assert not job.held.exists()                                            # gone with the comparison


def write_table_counts(path: Path, centres: list, named: list):
    """A table's own counts as zarr_bridge/helper.py count writes them: a row only for each outline (by its position)
    the table names by instance id."""
    keep = np.array(sorted(named), np.int64)
    xy = np.array(centres, np.float64).reshape(-1, 2)
    np.savez(path, index=keep.astype("<i4"), shape=np.array([len(keep), 3]),
             ids=np.array([f"cell-{k}" for k in keep], dtype=str).reshape(-1), x=xy[keep, 0].astype("<f4"),
             y=xy[keep, 1].astype("<f4"))


def test_a_table_naming_none_of_the_areas_cells_fails_its_column(setup, capsys):
    """A table's own count holds the outlines it names by instance id. One naming none of the area's (its ids written
    in another form) fails its column in plain words: never scored as a column of no cells, never kept. One naming
    some of them is scored on those; the engine log says the share."""
    manager, run, zarr, analysis = setup
    zarr.sources = {"xenium": {"count_by": "xenium", "count_table": "xenium_table"}}
    counts_of, named = zarr.counts, []

    def table_counts(sid, key, *a, **kw):
        path = counts_of(sid, key, *a, **kw)
        if key == "xenium" and path is not None:
            write_table_counts(path, zarr.layers[key], named)
        return path
    zarr.counts = table_counts
    manager.start("r1", {"columns": ["run", "xenium"]})
    done = finished(manager)
    assert done["state"] == "ready" and [(c["key"], c["state"], c["error"]) for c in done["columns"]] == [
        ("run", "ready", None), ("xenium", "failed", C.NOT_HELD)]
    assert all("xenium" not in [c["key"] for c in r["columns"]] for r in analysis.requests)
    assert not (run.dir / "compare" / done["job"] / "result.json").exists()
    assert "its table 'xenium_table' holds 0 of the 50 cells in the area" in capsys.readouterr().out
    named[:] = range(0, len(GRID), 2)                       # its table names every other one
    manager.start("r1", {"columns": ["run", "xenium"]})
    done = finished(manager)
    assert [(c["state"], c["cells_in_area"]) for c in done["columns"]] == [("ready", 50), ("ready", 25)]
    assert sorted(map(tuple, analysis.scored[-1]["xenium"].tolist())) == sorted(GRID[k] for k in named)
    assert "its table 'xenium_table' holds 25 of the 50 cells in the area" in capsys.readouterr().out
    named[:] = range(len(GRID))                             # all of them: nothing said
    zarr.table_versions["xenium_table"] = "t2"
    manager.start("r1", {"columns": ["run", "xenium"]})
    assert [c["cells_in_area"] for c in finished(manager)["columns"]] == [50, 50]
    assert "its table" not in capsys.readouterr().out


def test_the_checks_thresholds_are_part_of_what_a_comparison_is(setup):
    """Other [qc] thresholds give other numbers, so another comparison (never one kept from
    before)."""
    manager, run, zarr, analysis = setup
    first = manager.start("r1", {"columns": ["run"]})["job"]
    finished(manager)
    analysis.qc_thresholds = lambda: {"mixed_at": .5, "judge_min": 250, "fit_min": 50}
    assert manager.start("r1", {"columns": ["run"]})["job"] != first


def test_the_sopa_helpers_own_words_never_reach_the_page(setup, capsys):
    """The helper's failure (its exception's words, or its last lines) is technical: the page
    says it plainly, the log keeps it."""
    manager, run, zarr, analysis = setup
    zarr.layers["odd"] = GRID
    counts_of = zarr.counts

    def failing(sid, key, *a, **kw):
        if key == "odd" and not kw.get("cached_only"):
            raise HelperFailed("ArrowInvalid: Float value 1e+40 was truncated converting to int64")
        return counts_of(sid, key, *a, **kw)
    zarr.counts = failing
    manager.start("r1", {"columns": ["run", "odd"]})
    done = finished(manager)
    assert done["columns"][1] == {**done["columns"][1], "state": "failed", "error": "Could not count its transcripts"}
    assert "ArrowInvalid" in capsys.readouterr().out
    assert C.said(HelperFailed("x"), "plain") == "plain" and C.said(SampleError("Its own words"), "plain") == "Its own words"


def test_points_a_worker_answers_inline_are_kept_as_its_files_would_be(setup):
    manager, run, zarr, analysis = setup

    def inline(request):
        columns = [{"key": c["key"], "judged": 2, "points": {"x": [1.5, -2.0], "y": [3.0, 4.25], "share": [12, 255]}}
                   for c in request["columns"]]
        return {"ok": True, "compare": {"kinds": None, "columns": columns}}
    analysis.answer = inline
    manager.start("r1", {"columns": ["run"]})
    assert finished(manager)["state"] == "ready"
    points = np.frombuffer(manager.points("r1", 0).read_bytes(), C.POINT)
    assert points["x"].tolist() == [1.5, -2.0] and points["y"].tolist() == [3.0, 4.25]
    assert points["kind"].tolist() == [255, 255] and points["share"].tolist() == [12, 255] and C.POINT.itemsize == 10


def test_a_replaced_comparison_takes_back_its_queued_worker_call(setup):
    manager, run, zarr, analysis = setup
    analysis.hold = True
    first = manager.start("r1", {"columns": ["run"]})
    wait_for(lambda: len(analysis.futures) == 1)
    analysis.hold = False
    second = manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["job"] == second["job"] != first["job"] and done["state"] == "ready"
    assert analysis.futures[0].cancelled()                    # never ran: the scheduler drops it
    assert all(f.done() for f in analysis.futures)


def test_a_deleted_run_stops_its_comparison(setup):
    manager, run, zarr, analysis = setup
    zarr.gates["xenium"] = threading.Event()
    manager.start("r1", {"columns": ["xenium"]})
    wait_for(lambda: zarr.counted)
    job = manager.jobs["r1"]
    manager.forget("r1")
    wait_for(lambda: not job.thread.is_alive())
    assert zarr.stopped == ["xenium"] and "r1" not in manager.jobs


# ---- a compared layer written again meanwhile -------------------------------------------------------------------------
LEFT = [c for c in GRID if c[0] < 200]          # 25 cells over x 0-200 µm
RIGHT = [c for c in GRID if c[0] >= 200][:20]    # 20 cells over x 220-340 µm: the same ids (cell-0 ...), other places
DENSE = [(5 + 10 * i, 5 + 10 * j) for i in range(20) for j in range(20)]   # 400 cells over x and y 0-200 µm


def in_box(points, box) -> np.ndarray:
    points = np.asarray(points, float).reshape(-1, 2)
    return (points[:, 0] >= box[0]) & (points[:, 0] < box[2]) & (points[:, 1] >= box[1]) & (points[:, 1] < box[3])


def scored_right(done: dict, analysis, layer: list, key="cellpose"):
    """The worker's last call scored for `key` exactly the cells of `layer` (the element as it is now) in the area: the
    tiles holding 3 cells of the run's (GRID) and of `layer`."""
    box = done["area"]["box"]
    handed_over = analysis.scored[-1][key]
    assert in_box(handed_over, box).all(), (handed_over, box)       # never a cell outside the area
    xy = np.array(layer, float)
    area = C.common_area({"run": np.array(GRID, float).T, key: xy.T})
    assert area["box"] == box
    expected = xy[C.in_area(xy[:, 0], xy[:, 1], area)]
    assert sorted(map(tuple, handed_over.tolist())) == sorted(map(tuple, expected.tolist()))
    column = next(c for c in done["columns"] if c["key"] == key)
    assert column["state"] == "ready" and column["cells_in_area"] == len(expected) > 0


def test_counts_rows_are_the_outlines_at_their_positions():
    """A counts row is the outline at its position only with that outline's id and its centre within the outline's
    bounds or at the outline's own centre: ids alone match any element as long (a RangeIndex), and a centroid lies within
    its outline's bounds whatever its shape (a hole, a crescent, several parts) as long as its ring never crosses
    itself."""
    rings = [square(10, 10), square(50, 10), square(90, 10, 2), square(95, 12, 2)]   # the last two: one outline
    data = outlines(rings, [0, 1, 2, 2], count=3)
    names, bounds, centre = cells = C.outline_cells(data)
    assert names.tolist() == ["cell-0", "cell-1", "cell-2"]
    assert np.allclose(bounds, [[6, 6, 14, 14], [46, 6, 54, 14], [88, 8, 97, 14]])
    assert np.allclose(centre, np.c_[C.outline_centres(data)])
    position = np.array([2, 0])
    assert C.same_cells(cells, position, ["cell-2", "cell-0"], [93.0, 10.0], [11.0, 10.0])
    assert C.same_cells(cells, position, ["cell-2", "cell-0"], [97.9, 14.9], [13.9, 5.2])   # float32 rounding: 1 µm
    assert not C.same_cells(cells, position, ["cell-0", "cell-2"], [93.0, 10.0], [11.0, 10.0])   # other ids
    assert not C.same_cells(cells, position, ["cell-2", "cell-0"], [10.0, 93.0], [10.0, 11.0])   # other places
    assert not C.same_cells(cells, position, ["cell-2", "cell-0"], [93.0, 16.0], [11.0, 10.0])   # 2 µm beyond
    assert not C.same_cells(cells, position, ["cell-2"], [93.0], [11.0])                         # not one per row
    assert not C.same_cells(cells, [3], ["cell-3"], [10.0], [10.0])                              # no such outline
    assert C.same_cells(cells, np.zeros(0, int), [], [], [])
    assert C.misfits(cells, [2, 0, 1], ["cell-2", "cell-1", "cell-1"], [93, 10, 50], [11, 10, 10]).tolist() == [1]
    names, bounds, centre = cells = C.outline_cells(outlines([square(10, 10)], [0], count=2))   # one without parts
    assert np.isnan(bounds[1]).all() and np.isnan(centre[1]).all()
    assert C.same_cells(cells, [1], ["cell-1"], [np.nan], [np.nan])
    assert not C.same_cells(cells, [1], ["cell-1"], [10.0], [10.0])


@pytest.mark.parametrize("origin", [(132.0, 292.0), (12_345.0, 8_765.0)], ids=["near", "far"])
def test_an_outline_whose_ring_crosses_itself_is_matched_at_its_own_centre(origin):
    """A ring that crosses itself (two lobes of nearly opposite signed area) has its
    centroid, as the sopa helper counts it (shapely.centroid, float32), 75 µm outside its bounds: the bounds alone
    refused an unchanged layer holding one for good. The engine's own centre of the exported ring is that same point,
    near the origin or far from it; a row elsewhere outside the bounds still is not the outline."""
    ox, oy = origin
    ring = [[ox, oy], [ox + 16, oy + 16], [ox + 16, oy], [ox, oy + 15]]
    helper = {(132.0, 292.0): (222.6666717529297, 302.3333435058594),       # the sopa env's GEOS 3.13, float32
              (12_345.0, 8_765.0): (12435.6669921875, 8775.3330078125)}[origin]
    cells = C.outline_cells(outlines([square(ox - 50, oy), ring + ring[:1]]))
    assert helper[0] > cells[1][1, 2] + 70                                   # far beyond its bounds
    assert np.hypot(*(cells[2][1] - helper)) < 0.01
    assert C.same_cells(cells, [1, 0], ["cell-1", "cell-0"], [helper[0], ox - 50], [helper[1], oy])
    assert not C.same_cells(cells, [1, 0], ["cell-1", "cell-0"], [helper[0] + 2, ox - 50], [helper[1], oy])


@pytest.mark.parametrize("when", ["after_outlines", "before_count"])
def test_a_layer_written_again_while_compared_is_compared_again_as_it_is_now(setup, when):
    """A notebook writes the compared layer back (as many cells elsewhere, the same ids) after Compare read
    its outlines, or as its counting starts. The comparison starts again at the new version (the same job, its layouts
    counting on) and scores the layer's cells in the area as it is now: never the rows of one version taken for the
    cells of the other. Its key names the version before: it is not kept."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)
    (zarr.after_outlines if when == "after_outlines" else zarr.before_count)["cellpose"] = \
        lambda: zarr.write("cellpose", RIGHT)
    first = manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["job"] == first["job"] and done["state"] == "ready" and done["error"] is None
    assert [c["state"] for c in done["columns"]] == ["ready", "ready"]
    assert [v for k, v in zarr.read if k == "cellpose"] == ["v1", "v2"]          # read again at its new version
    assert done["area"]["box"] == [200.0, 0.0, 400.0, 200.0]                    # where the run and the layer now meet
    scored_right(done, analysis, RIGHT)
    assert done["layout"] == len(analysis.requests)                              # every layout of both tries counted
    assert manager.points("r1", 1, job=done["job"], layout=done["layout"]).exists()
    assert not (run.dir / "compare" / done["job"] / "result.json").exists()
    again = manager.start("r1", {"columns": ["run", "cellpose"]})                # asked again: at v2, and kept
    assert again["job"] != done["job"] and finished(manager)["state"] == "ready"
    assert (run.dir / "compare" / again["job"] / "result.json").exists()


@pytest.mark.parametrize("layer", [RIGHT, LEFT[::-1], LEFT[5:]], ids=["elsewhere", "reversed", "fewer"])
def test_counts_of_other_cells_under_the_same_version_are_never_scored(setup, layer, capsys):
    """The element's stamp cannot tell every write (written back within its resolution; counts kept under
    a version they are not of): each counts row must also be its position's outline (its id and place). Here the layer
    changes under the same version as its counting starts: as many cells elsewhere, the same cells reversed, the first
    five dropped. Its column fails at once in plain words, nothing of it scored, the rows that differ in the log: compared
    again, the same outlines and counts kept for that version would be read, and the column would fail
    saying it was written again."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)
    zarr.before_count["cellpose"] = lambda: zarr.layers.update(cellpose=list(layer))     # no new version
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and [v for k, v in zarr.read if k == "cellpose"] == ["v1"]   # read once: no restart
    assert [(c["state"], c["error"]) for c in done["columns"]] == [("ready", None), ("failed", C.DISAGREE)]
    assert all("cellpose" not in scored for scored in analysis.scored) and done["layout"] == 1
    assert "rows of its counts (cellpose.npz) are not its outlines at version v1" in capsys.readouterr().out
    assert not (run.dir / "compare" / done["job"] / "result.json").exists()


@pytest.mark.parametrize("off", [4, 5], ids=["left-out", "refused"])
def test_a_few_counts_rows_unlike_their_outlines_are_left_out_more_fail_the_column(setup, off, capsys):
    """A cell of several parts, one of whose rings crosses itself, has no centre the sopa helper and the engine agree on
    (the helper's centroid can lie tens of µm outside it). Up to MISFITS_LEFT_OUT of a counts file's rows (4 of 400) may
    misfit their outlines: they are left out, never scored, and named in the log; the column is compared. More (5 of
    400), and the file is not of its outlines: the column fails."""
    manager, run, zarr, analysis = setup
    zarr.layers["dense"] = list(DENSE)
    counts_of = zarr.counts

    def counts(sid, key, *args, **kwargs):           # the first `off` rows' centres 60 µm from their outlines
        path = counts_of(sid, key, *args, **kwargs)
        if key == "dense" and path is not None:
            with np.load(path) as counted:
                data = dict(counted)
            data["x"] = (data["x"] + np.where(np.arange(len(data["x"])) < off, 60, 0)).astype("<f4")
            np.savez(path, **data)
        return path
    zarr.counts = counts
    manager.start("r1", {"columns": ["run", "dense"]})
    done = finished(manager)
    column, log = done["columns"][1], capsys.readouterr().out
    assert done["state"] == "ready" and f"{off} of 400 rows of its counts (dense.npz) are not its outlines" in log
    if off == 4:
        assert column["state"] == "ready" and column["cells_in_area"] == 396 and ": left out" in log
        handed_over = analysis.scored[-1]["dense"]
        assert len(handed_over) == 396 and (handed_over[:, 0] < 200).all()       # none of the rows that misfit
    else:
        assert (column["state"], column["error"]) == ("failed", C.DISAGREE) and ": left out" not in log
        assert all("dense" not in scored for scored in analysis.scored)


def test_a_layer_written_again_during_its_second_try_fails_its_column_alone(setup):
    """Written again once more while compared again, the layer's column fails in words that say so; the
    other columns are compared, and nothing of the layer reaches the worker."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)
    writes = []

    def writer():
        writes.append(1)
        zarr.write("cellpose", RIGHT if len(writes) % 2 else LEFT)
        if len(writes) < 2:
            zarr.after_outlines["cellpose"] = writer
    zarr.after_outlines["cellpose"] = writer
    manager.start("r1", {"columns": ["run", "xenium", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and len(writes) == 2
    assert [(c["state"], c["error"]) for c in done["columns"]] == [("ready", None), ("ready", None),
                                                                   ("failed", C.WRITTEN_AGAIN)]
    assert all(c["key"] != "cellpose" for r in analysis.requests for c in r["columns"])
    assert not (run.dir / "compare" / done["job"] / "result.json").exists()
    with pytest.raises(RunError):
        manager.points("r1", 2)


def test_a_layer_written_again_while_its_numbers_come_is_compared_again(setup):
    """The versions are read once more as the comparison is ready: a write while the worker scored it
    starts it again, so a ready comparison is of the layers as they are."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)
    zarr.cached.add("cellpose")                     # counted before: one call of the worker per try
    analysis.hook = lambda request: zarr.write("cellpose", RIGHT) if len(analysis.requests) == 1 else None
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and len(analysis.requests) == 2 and done["layout"] == 2
    assert [v for k, v in zarr.read if k == "cellpose"] == ["v1", "v2"]
    scored_right(done, analysis, RIGHT)


def test_a_layer_written_again_while_its_outlines_are_read_is_compared_again(setup):
    """The sample refuses the outlines of a layer written over while it exported them (ZarrSamples: 409),
    as it refuses counts of one written over while counted (the before_count case above). Its element no longer the
    version pinned, the comparison starts again at the new one: the refusal is never taken for the column's failure."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)
    zarr.outline_gates["cellpose"] = gate = threading.Event()
    manager.start("r1", {"columns": ["run", "cellpose"]})
    wait_for(lambda: "cellpose" in zarr.held)
    zarr.write("cellpose", RIGHT)
    del zarr.outline_gates["cellpose"]
    gate.set()
    done = finished(manager)
    assert done["state"] == "ready" and [(c["state"], c["error"]) for c in done["columns"]] == [("ready", None)] * 2
    assert [v for k, v in zarr.read if k == "cellpose"] == ["v2"]                # the refused read is none
    scored_right(done, analysis, RIGHT)


def test_a_restart_pins_the_versions_without_listing_the_sample_again(setup):
    """Compared again, each layer is pinned to its version on disk (its stamp), the sample not listed
    again: a listing can run the sopa helper (what made each element), which nothing stops, so a newer comparison and the
    engine's close waited for it. A layer whose folder does not answer then (twice, STAMP_RETRY_S apart) fails alone, in
    words that say so; the others are compared."""
    manager, run, zarr, analysis = setup
    zarr.layers["cellpose"] = list(LEFT)

    def writer():
        zarr.write("cellpose", RIGHT)
        zarr.stamps["cellpose"] = ["v2", None, None]   # seen written, then its folder does not answer at the restart
    zarr.after_outlines["cellpose"] = writer
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    assert done["state"] == "ready" and zarr.listings == 1 and zarr.stamps["cellpose"] == []
    assert [(c["state"], c["error"]) for c in done["columns"]] == [
        ("ready", None), ("failed", "'cellpose' could not be read just now: try again in a moment")]
    assert all("cellpose" not in scored for scored in analysis.scored)


# The same race with the real ZarrSamples (its element stamps, the outlines' export cache and the stamp-named counts),
# and the worker's own readers: only the sopa helper is stood in for, in the engine's process, reading the element as
# it is on disk when it reads it.
WRITES = itertools.count()


def write_element(sample: Path, key: str, centres: list, ids: list | None = None):
    """A shapes element as SpatialData keeps it, in a folder of its own: written again, the folder is replaced (as
    write_element(overwrite=True)), another stamp (each write leaves a file of its own name). Its cells: ids (by
    default named by place, as Xenium's are) and centres, each an 8 µm square."""
    folder = sample / "shapes" / key
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    ids = [f"{key}-{x:g}-{y:g}" for x, y in centres] if ids is None else ids
    (folder / "cells.json").write_text(json.dumps({"ids": ids, "centres": [[float(x), float(y)] for x, y in centres]}))
    (folder / f"write-{next(WRITES)}").touch()


def read_element(sample: Path, key: str) -> tuple[list, np.ndarray]:
    data = json.loads((sample / "shapes" / key / "cells.json").read_text())
    return data["ids"], np.array(data["centres"], float).reshape(-1, 2)


class SopaStandIn:
    """zarr_bridge/helper.py's `shapes` and `count` as their files come (the export: bounds, offsets, owner, coords and
    its .json with ids; the counts: CSR, ids, index, x, y, genes, rule; a box by centre, 1 µm around), of the element as
    it is on disk when read. `before_count[key]` runs once as a count starts (another writer landing then)."""

    def __init__(self, sample: Path):
        self.sample, self.before_count = sample, {}

    def __call__(self, command, args, progress=None, should_stop=None, timeout=3600):
        if command == "count" and args["key"] in self.before_count:
            self.before_count.pop(args["key"])()
        ids, centres = read_element(self.sample, args["key"])
        out = Path(args["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if command == "shapes":
            data = outlines([square(x, y) for x, y in centres])
            with open(f"{out}.bin", "wb") as stream:
                for name in ("bounds", "offsets", "owner", "coords"):
                    stream.write(np.ascontiguousarray(data[name]).astype(data[name].dtype.newbyteorder("<")).tobytes())
            Path(f"{out}.json").write_text(json.dumps({"schema": 1, "key": args["key"], "polygons": len(ids),
                                                       "parts": len(ids), "vertices": len(data["coords"]), "ids": ids}))
            return {"polygons": len(ids)}
        assert command == "count", command
        index = np.arange(len(ids))
        if args.get("box") is not None:
            x0, y0, x1, y1 = args["box"]
            x, y = centres[:, 0], centres[:, 1]
            index = np.flatnonzero((x >= x0 - 1) & (x <= x1 + 1) & (y >= y0 - 1) & (y <= y1 + 1))
        n = len(index)
        np.savez(out, data=np.full(n, 3, "<u4"), indices=np.zeros(n, "<i4"), indptr=np.arange(n + 1).astype("<i8"),
                 shape=np.array([n, 1], "<i8"), ids=np.array([ids[i] for i in index], dtype=str),
                 index=index.astype("<i4"), x=centres[index, 0].astype("<f4"), y=centres[index, 1].astype("<f4"),
                 genes=np.array(["ALB"]), rule=np.array("rule"))
        return {"cells": n}


class Samples(Z.ZarrSamples):
    """The real ZarrSamples, `after_outlines[key]` run once after an outline read (another writer landing then). Its
    reads are not stopped here (test_closing_and_replacing_kill_the_outline_export_helper's business)."""
    after_outlines: dict

    def _outlines(self, sid, key, should_stop=None):
        value = super()._outlines(sid, key)
        hook = self.after_outlines.pop(key, None)
        if hook is not None:
            hook()
        return value


def cells_in_tiles(tiles: dict) -> list:
    return [tuple(c) for c in np.array(cells_in(tiles)).T.tolist()]


A1 = [(10, 15), (30, 15), (50, 15), (70, 15), (10, 35), (120, 50), (140, 50)]       # 5 in tile (0, 0), 2 in (1, 0)
B1 = [(120, 20), (125, 40), (130, 60), (135, 80), (140, 30), (40, 50), (60, 50)]     # as many: 5 in (1, 0)
SCENARIOS = {    # (the run's cells, xenium's, the layer's before and after, when the write lands, ids by place or row)
    "S1-other-cells": (cells_in_tiles({(0, 0): 8, (1, 0): 8}), None, A1, B1, "count", "named"),
    "S1b-between-outlines-and-counts": (cells_in_tiles({(0, 0): 8, (1, 0): 8}), None, A1, B1, "outlines", "named"),
    "S2-first-five-dropped": (LEFT, GRID, GRID, GRID[5:], "count", "named"),
    "S3-reversed": (LEFT, GRID, GRID, GRID[::-1], "count", "named"),
    "S3-reversed-row-ids": (LEFT, GRID, GRID, GRID[::-1], "count", "rows"),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_a_layer_replaced_on_disk_while_compared_is_scored_as_it_is_now(tmp_path, name):
    """End to end: another writer replaces the compared element (a notebook's write_element) as its count
    starts, or between its outlines and its counts. The comparison ends ready with the layer's cells in the area as the
    element is now, as the worker reads them (read_counts_file, area_rows): before, the counts' positions in the new
    element were taken for cells of the old outlines (cells outside the area, or the wrong ones)."""
    run_cells, xenium, before, after, when, ids = SCENARIOS[name]
    xenium = run_cells if xenium is None else xenium
    named = (lambda c: None) if ids == "named" else (lambda c: [str(k) for k in range(len(c))])
    sample, key = tmp_path / "sample.zarr", "proseg_boundaries"
    write_element(sample, "xenium", xenium, named(xenium))
    write_element(sample, key, before, named(before))
    zarr = Samples({"zarr": {"python": ""}, "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "data")},
                    "proseg": {"max_region_um": 2000.0}})
    zarr.recent = [{"id": "zsample", "name": "toy", "path": str(sample), "kind": "zarr"}]
    extent = [0.0, 0.0, 3000.0, 400.0]

    def extent_of(cells):
        c = np.array(cells, float)
        return [float(c[:, 0].min() - 4), float(c[:, 1].min() - 4), float(c[:, 0].max() + 4), float(c[:, 1].max() + 4)]
    detail = {"id": "zsample", "points_key": "transcripts", "extent": extent,           # as inspected when it opened
              "shapes": [{"key": k, "count": len(c), "geometry": {"Polygon": len(c)}, "extent": extent_of(c)}
                         for k, c in (("xenium", xenium), (key, before))]}
    zarr.detail, zarr._made_by = (lambda sid: detail), (lambda sid: None)
    zarr.helper = helper = SopaStandIn(sample)
    zarr.after_outlines = {}
    runs = FakeRuns()
    run = add_run(runs, tmp_path / "runs", centres=run_cells)
    analysis = FakeAnalysis(runs)
    manager = C.CompareManager(runs, zarr, FakeQc(), analysis)
    try:
        zarr.segmentation_outlines("zsample", key)                 # shown on the tissue once: exported, in memory
        writer = lambda: write_element(sample, key, after, named(after))   # noqa: E731
        (helper.before_count if when == "count" else zarr.after_outlines)[key] = writer
        manager.start("r1", {"columns": ["run", "xenium", key]})
        done = finished(manager)
        assert done["state"] == "ready", done
        column = done["columns"][2]
        assert column["state"] == "ready" and not helper.before_count and not zarr.after_outlines
        _, now = read_element(sample, key)
        area = C.common_area({"run": np.array(run_cells, float).T, "xenium": np.array(xenium, float).T, key: now.T},
                             clip=extent)
        assert done["area"]["box"] == area["box"]
        expected = now[C.in_area(now[:, 0], now[:, 1], area)]
        request = analysis.requests[-1]["columns"][2]
        matrix, names, x, y, _ = W.read_counts_file(request["counts"])
        rows = W.area_rows(request["cells"], matrix.shape[0])
        assert sorted(zip(x[rows].tolist(), y[rows].tolist())) == sorted(map(tuple, expected.tolist()))
        assert column["cells_in_area"] == len(expected) > 0
        assert not (run.dir / "compare" / done["job"] / "result.json").exists()     # its key names the version before
    finally:
        manager.close()
        zarr.close()


# ---- stopping an outline read -------------------------------------------------------------------------------------------
def test_closing_stops_an_outline_read_at_once(setup, capsys):
    """Compare's outline export is stopped with the comparison (ZarrSamples._outlines' should_stop): the
    engine's close returns at once, its thread is gone, and the stop is said as no failure of the layer's."""
    manager, run, zarr, analysis = setup
    zarr.outline_gates["xenium"] = threading.Event()          # a slow export: only a stop ends it
    manager.start("r1", {"columns": ["run", "xenium"]})
    wait_for(lambda: "xenium" in zarr.held)
    job = manager.jobs["r1"]
    began = time.monotonic()
    manager.close()
    assert time.monotonic() - began < 1.0 and not job.thread.is_alive()
    assert zarr.stopped == ["outlines xenium"] and job.public()["error"] == "replaced by a newer comparison"
    assert "Could not read" not in capsys.readouterr().out and job.columns[1]["error"] is None


def test_a_newer_comparison_starts_at_once_while_the_one_it_replaces_reads_outlines(setup):
    """The comparison it replaces stops its outline export, so the newer one starts at once (it waited
    for the export to end)."""
    manager, run, zarr, analysis = setup
    zarr.outline_gates["xenium"] = threading.Event()
    first = manager.start("r1", {"columns": ["xenium"]})
    wait_for(lambda: "xenium" in zarr.held)
    began = time.monotonic()
    second = manager.start("r1", {"columns": ["cellpose", "run"]})
    wait_for(lambda: ("cellpose", "v1") in zarr.read, timeout=5)
    assert time.monotonic() - began < 1.0
    done = finished(manager)
    assert done["job"] == second["job"] != first["job"] and done["state"] == "ready"
    assert zarr.stopped == ["outlines xenium"]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_closing_and_replacing_kill_the_outline_export_helper(tmp_path, monkeypatch):
    """End to end: the real ZarrSamples runs a slow outline export (a stand-in for the sopa python that takes
    30 s). A newer comparison starts at once and the engine's close returns at once, the export's process gone each time
    (killed and reaped: it can write nothing afterwards). Before, close gave up after 5 s and the helper, orphaned,
    wrote into the cache after the engine had gone."""
    pids, script = tmp_path / "pids", tmp_path / "python"
    script.write_text(f"#!{sys.executable}\nimport json, os, sys, time\n"
                      "open(os.environ['STAND_IN_PIDS'], 'a').write(f'{os.getpid()} {sys.argv[3]}\\n')\n"
                      "time.sleep(30)\nprint(json.dumps({'error': 'too late'}))\nsys.exit(1)\n")
    script.chmod(0o755)
    monkeypatch.setenv("STAND_IN_PIDS", str(pids))
    sample, extent = tmp_path / "sample.zarr", [0.0, 0.0, 400.0, 200.0]
    for key in ("cells", "nuclei"):
        write_element(sample, key, GRID)
    zarr = Z.ZarrSamples({"zarr": {"python": str(script)}, "proseg": {"max_region_um": 2000.0},
                          "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "data")}})
    zarr.recent = [{"id": "zsample", "name": "toy", "path": str(sample), "kind": "zarr"}]
    detail = {"id": "zsample", "points_key": "transcripts", "extent": extent,
              "shapes": [{"key": k, "count": 50, "geometry": {"Polygon": 50}, "extent": extent} for k in ("cells", "nuclei")]}
    zarr.detail, zarr._made_by = (lambda sid: detail), (lambda sid: None)
    runs = FakeRuns()
    add_run(runs, tmp_path / "runs")
    manager = C.CompareManager(runs, zarr, FakeQc(), FakeAnalysis(runs))

    def exports() -> list[int]:
        return [int(line.split()[0]) for line in pids.read_text().splitlines()] if pids.exists() else []
    try:
        manager.start("r1", {"columns": ["run", "cells"]})
        wait_for(lambda: len(exports()) == 1)
        began = time.monotonic()
        manager.start("r1", {"columns": ["run"]})
        assert finished(manager)["state"] == "ready" and time.monotonic() - began < 2.0
        assert not alive(exports()[0])
        manager.start("r1", {"columns": ["run", "nuclei"]})
        wait_for(lambda: len(exports()) == 2)
        job, began = manager.jobs["r1"], time.monotonic()
        manager.close()
        assert time.monotonic() - began < 1.0 and not job.thread.is_alive() and not alive(exports()[1])
    finally:
        manager.close()
        zarr.close()
        for pid in exports():
            if alive(pid):
                os.kill(pid, 9)


# ---- points of the comparison and layout drawn --------------------------------------------------------------------------
def test_points_are_only_of_the_comparison_and_layout_asked_for(setup):
    """A points request names the comparison (its job) and the layout the page draws: another comparison of
    the run (from another window too: one slot per run) or another layout is refused (409), so a column never draws the
    points of another comparison, column or layout."""
    manager, run, zarr, analysis = setup
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    job, layout = done["job"], done["layout"]
    assert layout == 2 and manager.points("r1", 1, job=job, layout=layout) == manager.points("r1", 1)
    for asked in ({"job": "0123456789abcdef"}, {"job": job, "layout": layout - 1}, {"layout": layout + 1}):
        with pytest.raises(RunError) as refused:
            manager.points("r1", 1, **asked)
        assert refused.value.status == 409, asked
    # Another window compares the same columns the other way round: its column 0 is cellpose. The first window's next
    # request, naming its own comparison, is refused rather than answered with cellpose's points for its run column.
    manager.start("r1", {"columns": ["cellpose", "run"]})
    after = finished(manager)
    with pytest.raises(RunError) as refused:
        manager.points("r1", 0, job=job, layout=layout)
    assert refused.value.status == 409
    assert manager.points("r1", 0, job=after["job"], layout=after["layout"]).name == f"points-{after['layout']}-0.bin"
    # Kept and found again (another engine): the same comparison, layout and points.
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)
    try:
        found = again.start("r1", {"columns": ["cellpose", "run"]})
        assert found["job"] == after["job"] and found["layout"] == after["layout"]
        assert found["layout_kind"] == after["layout_kind"] == "pca"
        assert again.points("r1", 0, job=found["job"], layout=found["layout"]).name == f"points-{after['layout']}-0.bin"
    finally:
        again.close()
    manager.forget("r1")
    with pytest.raises(RunError) as refused:
        manager.points("r1", 0, job=after["job"])
    assert refused.value.status == 409                         # its comparison went
    with pytest.raises(RunError) as missing:
        manager.points("r1", 0)
    assert missing.value.status == 404


def test_each_columns_own_map_is_kept_served_and_found_again(setup):
    """Compare's "Each its own": the worker writes each column's own map (<stem>.own.bin: its points laid out alone) and
    its own clusters (<stem>.clusters.bin, u16 per point) beside its shared-map points; the engine keeps both with the
    comparison, says each column's clusters, serves them for the comparison and layout the page draws, keeps them beside
    the run and finds them again. A column with no map of its own (too few cells) has none to serve; a comparison kept
    before own maps existed is made again."""
    manager, run, zarr, analysis = setup
    analysis.own = {0, 1}
    manager.start("r1", {"columns": ["run", "cellpose"]})
    done = finished(manager)
    job, layout = done["job"], done["layout"]
    assert [r["own"] for r in analysis.requests] == [False, True], "own maps in the call with every column only"
    assert (done["own_layout"], done["own_bounds"], done["own_error"]) == ("pca", [-5, -5, 5, 5], None)
    assert [(c["own"], c["clusters"], c["clusters_matched"]) for c in done["columns"]] == [(True, 4, 4), (True, 4, 3)]
    own = manager.points("r1", 1, job=job, layout=layout, own=True)
    clusters = manager.points("r1", 1, job=job, layout=layout, what="clusters")
    shared = manager.points("r1", 1, job=job, layout=layout)
    assert (own.name, clusters.name) == (f"points-{layout}-1.own.bin", f"points-{layout}-1.clusters.bin")
    assert own.stat().st_size == shared.stat().st_size == 5 * clusters.stat().st_size and own.read_bytes() != shared.read_bytes()
    for asked in ({"job": "0123456789abcdef"}, {"job": job, "layout": layout - 1}):      # another comparison or layout
        with pytest.raises(RunError) as refused:
            manager.points("r1", 1, own=True, **asked)
        assert refused.value.status == 409, asked
    folder = run.dir / "compare" / job
    saved = json.loads((folder / "result.json").read_text())
    assert saved["own_files"] == {str(i): {"points": f"points-{layout}-{i}.own.bin", "clusters": f"points-{layout}-{i}.clusters.bin"}
                                  for i in (0, 1)}
    assert sorted(p.name for p in folder.glob("points-*.bin")) == sorted(
        f"points-{layout}-{i}{end}" for i in (0, 1) for end in (".bin", ".own.bin", ".clusters.bin"))   # the first layout's gone
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)
    try:
        asked = len(analysis.requests)
        found = again.start("r1", {"columns": ["run", "cellpose"]})
        assert found["job"] == job and found["state"] == "ready" and len(analysis.requests) == asked
        assert (found["own_layout"], found["own_bounds"]) == ("pca", [-5, -5, 5, 5]) and found["columns"][1]["own"] is True
        assert again.points("r1", 1, job=job, layout=layout, own=True) == own
        assert again.points("r1", 1, job=job, layout=layout, what="clusters") == clusters
    finally:
        again.close()
    # A comparison kept before "Each its own" (no own_files): made again, so every column has its own map.
    saved.pop("own_files")
    (folder / "result.json").write_text(json.dumps(saved))
    again = C.CompareManager(manager.runs, zarr, FakeQc(), analysis)
    try:
        asked = len(analysis.requests)
        assert again.start("r1", {"columns": ["run", "cellpose"]})["state"] == "working"
        assert finished(again)["columns"][1]["own"] is True and len(analysis.requests) > asked
    finally:
        again.close()
    # A column too small for a map of its own: none, nor one left from before; the others keep theirs.
    analysis.own = {0}
    manager.start("r1", {"columns": ["run", "cellpose", "xenium"]})
    done = finished(manager)
    assert [c["own"] for c in done["columns"]] == [True, False, False] and done["columns"][1]["clusters"] == 0
    with pytest.raises(RunError) as missing:
        manager.points("r1", 1, job=done["job"], layout=done["layout"], own=True)
    assert missing.value.status == 404
    assert manager.points("r1", 0, job=done["job"], layout=done["layout"], what="clusters").exists()
    analysis.own = set()
    manager.start("r1", {"columns": ["cellpose", "run"]})
    done = finished(manager)
    assert (done["own_layout"], done["own_error"]) == (None, "too_few_cells") and not any(c["own"] for c in done["columns"])


# ---- routes -------------------------------------------------------------------------------------------------------------
def test_routes_answer_with_the_managers_and_their_problems(config, tmp_path, monkeypatch):
    from sopa_companion.server import create_app
    with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
        manager = client.app.state.compare
        assert client.get("/api/v1/runs/nope/compare", headers=auth()).status_code == 404
        assert client.post("/api/v1/runs/nope/compare", json={"columns": ["run"]}, headers=auth()).status_code == 404
        monkeypatch.setattr(manager, "describe", lambda run_id: {"state": "ready", "run": run_id})
        answer = client.get("/api/v1/runs/r1/compare", headers=auth())
        assert answer.json() == {"state": "ready", "run": "r1"} and answer.headers["cache-control"] == "private, no-cache"

        def refuse(run_id, body):
            raise C.CompareError("turn it on", 409)
        monkeypatch.setattr(manager, "start", refuse)
        refused = client.post("/api/v1/runs/r1/compare", json={"columns": ["run"]}, headers=auth())
        assert refused.status_code == 409 and refused.json()["detail"] == "turn it on"
        monkeypatch.setattr(manager, "start", lambda run_id, body: {"state": "working", "columns": body["columns"]})
        started = client.post("/api/v1/runs/r1/compare", json={"columns": ["run", "x"]}, headers=auth())
        assert started.status_code == 202 and started.json()["columns"] == ["run", "x"]
        points = tmp_path / "points.bin"
        points.write_bytes(bytes(range(20)))
        monkeypatch.setattr(manager, "points", lambda run_id, index, job=None, layout=None, own=False, what="points":
                            points if index == 1 else None)
        got = client.get("/api/v1/runs/r1/compare/points/1.bin", headers=auth())
        assert got.content == bytes(range(20)) and got.headers["cache-control"] == "private, no-cache"
        assert client.get("/api/v1/runs/r1/compare/points/x.bin", headers=auth()).status_code == 404
        assert client.get("/api/v1/samples/nope/segmentations", headers=auth()).status_code == 404
        assert client.get("/api/v1/samples/nope/segmentations/cells/outlines", headers=auth()).status_code == 404


def test_the_points_route_hands_on_the_comparison_and_layout_the_page_draws(config, monkeypatch):
    """Through the server: the page names the comparison and the layout it draws (api.js
    comparePoints: ?job=&layout=), and the route hands both to CompareManager.points (the layout as a number), so the
    points of a replaced comparison or layout are refused (409) there too: server/app.py's route declares them."""
    from sopa_companion.server import create_app
    with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
        manager, asked = client.app.state.compare, []

        def points(run_id, index, job=None, layout=None, own=False, what="points"):
            asked.append((run_id, index, job, layout, own, what))
            raise C.CompareError("This comparison was replaced", 409)
        monkeypatch.setattr(manager, "points", points)
        refused = client.get("/api/v1/runs/r1/compare/points/1.bin?job=0123456789abcdef&layout=3", headers=auth())
        assert refused.status_code == 409 and refused.json()["detail"] == "This comparison was replaced"
        assert asked == [("r1", 1, "0123456789abcdef", 3, False, "points")], "the route drops ?job= and ?layout="
        # "Each its own": the column's own map (?own=1) and its own clusters, refused the same way
        assert client.get("/api/v1/runs/r1/compare/points/1.bin?job=0123456789abcdef&layout=3&own=1",
                          headers=auth()).status_code == 409
        assert client.get("/api/v1/runs/r1/compare/clusters/2.bin?job=0123456789abcdef&layout=3",
                          headers=auth()).status_code == 409
        assert asked[1:] == [("r1", 1, "0123456789abcdef", 3, True, "points"),
                             ("r1", 2, "0123456789abcdef", 3, False, "clusters")]
