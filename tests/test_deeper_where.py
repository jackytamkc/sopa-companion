"""Go deeper's "where it spills" (engine/deeper_where.py; GET /runs/{id}/qc/deeper/where), version 3: proseg's share
of each transcript for the cell (w), exact from proseg's where export of the box (|B| from the run's transcripts, a
bucket across the box's edge whole), approximate from the frame's 3-D labels x the cell's keep e / n (a tiled frame's
window layers, never its 2-D column), or none (a tiled frame outside its window, counts gone), with the counts as sums
of w (over a box holding the cell's pools the exact w add up to its averaged count) and the tissue around weighed at
the cell's keep; where the transcripts lie stays positional (owners by the frame's labels: per-layer in detail runs,
the 2-D projection in whole-sample runs, the prior for the start line); the box; the points; each gene's numbers and
word (edge, spread, layer, few); "into"; replies kept by where their shares came from (an export gone meanwhile:
approximate); the export asked of a paused run and the box a playing run's frames export (the live cell's newest
reply's, never the start line's); a cell whose pink dots lie far from the neighbour that would be named (the
strip case); the refusals; and the time it takes on a crop-sized frame. Small runs written by hand, with numbers
worked out by hand; the invariant against the worker's own numpy code."""
from __future__ import annotations

import gzip
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from counts_fixtures import counts_of, observed_voxels, where_span, write_frame_counts, write_where_export

BASE = "http://127.0.0.1:8799"
RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
GENES = ["EDGE", "SPREAD", "LAYER", "FEW", "OWN1", "OWN2", "OTHER", "UNASKED"]
X0, Y0, NX, NY = 100.0, 50.0, 60, 40               # 1 µm columns: the raster is [100, 160) x [50, 90)
CELLS = ["a-1", "b-1", "c-1", "d-1", "e-1"]          # label k + 1: A (the cell), B (a source), C, D (a source), E


def auth():
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE}


def layout() -> np.ndarray:
    """The cell A (label 1): columns 10-19, rows 10-19 (x 110-120, y 60-70); its source B (2) right beside it (x
    120-130); another cell C (3) below it (y 74-84); a second source D (4) four columns to its left (x 100-106)."""
    plane = np.zeros((NY, NX), np.uint32)
    for label, (c0, c1), (r0, r1) in ((1, (10, 20), (10, 20)), (2, (20, 30), (10, 20)), (3, (10, 20), (24, 34)),
                                      (4, (0, 6), (10, 20))):
        plane[r0:r1, c0:c1] = label
    return plane


def placed() -> np.ndarray:
    """The transcripts the numbers below are worked out from (z 0.5), in this order."""
    rows = []

    def put(gene, xs, ys):
        rows.extend((x, y, 0.5, GENES.index(gene)) for x, y in zip(xs, ys))
    tenth = [60.5 + i for i in range(10)]
    # In A: EDGE along the edge it shares with B (0.5 µm away), SPREAD and LAYER away from it, three FEW, its own
    # genes (two OWN1 1.5 µm from B, the rest far), and five OTHER (not asked: counted, never shown).
    put("EDGE", [119.5] * 10, tenth)
    put("SPREAD", [112.5] * 10, tenth)
    put("LAYER", [113.5] * 10, tenth)
    put("FEW", [119.8] * 3, [61.5, 62.5, 63.5])
    put("OWN1", [118.5, 118.5] + [111.5] * 9 + [110.5] * 9, [65.5, 66.5] + tenth[:9] + tenth[:9])
    put("OWN2", [115.5] * 10, tenth)
    put("OTHER", [116.5] * 5, [61.5, 62.5, 63.5, 64.5, 65.5])
    # In B: EDGE throughout; A's OWN1 just inside it (0.5, 1.0, 1.9 µm from A) and one far in; its own genes (UNASKED),
    # two of them near A.
    put("EDGE", [125.5] * 10 + [126.5] * 10, tenth + tenth)
    put("OWN1", [120.5, 121.0, 121.9, 128.0], [65.5] * 4)
    put("UNASKED", [120.2, 121.5] + [127.5] * 8, [61.5, 62.5] + tenth[:8])
    # SPREAD also in B, far from A: more of it than in D (8 against 6), but less densely (8 in 100 µm², 6 in 60).
    put("SPREAD", [128.5] * 8, tenth[:8])
    # In D: SPREAD, two OWN2 4.5 µm from A, three of its own.
    put("SPREAD", [103.0] * 6, [61.5, 62.5, 63.5, 64.5, 65.5, 66.5])
    put("OWN2", [105.5, 105.5], [62.5, 63.5])
    put("UNASKED", [104.0] * 3, [61.5, 62.5, 63.5])
    # LAYER all around: 50 in C (another cell) and 50 in the background, inside the box; EDGE past the box.
    put("LAYER", [110.5 + i for i in range(10) for _ in range(5)], [75.5 + j for _ in range(10) for j in range(5)])
    put("OTHER", [115.5] * 4, [80.5, 81.5, 82.5, 83.5])
    put("LAYER", [140.5 + i for i in range(10) for _ in range(5)], [51.5 + j for _ in range(10) for j in range(5)])
    put("EDGE", [155.5] * 7, [60.5 + i for i in range(7)])
    return np.array(rows, dtype=RECORD)


def cell_stats(consensus, layers, x0, y0, voxel, ncells):
    """A frame's cell statistics as proseg-live writes them (live.rs): area, voxels, assigned, foreground, changed
    (u32), then the centroid x, y (f32, NaN without a column)."""
    stats = np.zeros((7, ncells), np.uint32)
    labels = consensus.ravel().astype(np.int64)
    rows, cols = np.divmod(np.arange(labels.size), consensus.shape[1])
    keep = (labels > 0) & (labels <= ncells)
    area = np.bincount(labels[keep], minlength=ncells + 1)[1:]
    sx = np.bincount(labels[keep], weights=cols[keep], minlength=ncells + 1)[1:]
    sy = np.bincount(labels[keep], weights=rows[keep], minlength=ncells + 1)[1:]
    flat = layers.ravel().astype(np.int64)
    stats[0], stats[1] = area, np.bincount(flat[(flat > 0) & (flat <= ncells)], minlength=ncells + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.where(area > 0, x0 + ((sx / area).astype(np.float32) + 0.5) * voxel, np.nan).astype(np.float32)
        cy = np.where(area > 0, y0 + ((sy / area).astype(np.float32) + 0.5) * voxel, np.nan).astype(np.float32)
    stats[5], stats[6] = cx.view(np.uint32), cy.view(np.uint32)
    return stats


def write_tiles(path, planes: dict, tile: int, window=None):
    """proseg-live's tile container (native/proseg/src/live.rs): tiles row-major, planes fastest, then the window:
    `window` = (layers [nz, h, w], first column, first row) of per-layer labels, or None (none kept)."""
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
    info = None
    if window is None:
        blobs.append(b"")
    else:
        layers, i0, j0 = window
        info = {"i0": int(i0), "j0": int(j0), "nx": int(layers.shape[2]), "ny": int(layers.shape[1]),
                "nz": int(layers.shape[0])}
        blobs.append(zstandard.ZstdCompressor(level=1).compress(np.ascontiguousarray(layers, "<u4").tobytes()))
    header = json.dumps({"nx": nx, "ny": ny, "nz": 1 if window is None else info["nz"], "tile": tile,
                         "tiles_x": tiles_x, "tiles_y": tiles_y, "planes": names, "window": info}).encode()
    offset, head = 8 + len(header) + 12 * len(blobs), b"PLT1" + struct.pack("<I", len(header)) + header
    for blob in blobs:
        head += struct.pack("<QI", offset, len(blob))
        offset += len(blob)
    path.write_bytes(head + b"".join(blobs))
    return info


def frame_counts(live, seq, iteration, labels, records, grid, z, ncells, ngenes, keep=1.0, dest=None):
    """A counted frame: each cell's count per gene of the transcripts whose destination it owns (`dest`, a label per
    record; by default the label of its observed voxel in the per-layer `labels` [nz, ny, nx]: no drift), its e = n x
    keep (a share, or a [ncells, ngenes] table of them), written as proseg-live does (tests/counts_fixtures.py). Returns
    the counts line."""
    if dest is None:
        dest = destinations(records, labels, grid, z)
    found = counts_of(dest, records["gene"], ncells, ngenes)
    rows = np.repeat(np.arange(ncells), np.diff(found["indptr"]))
    share = np.broadcast_to(np.asarray(keep, np.float64), (ncells, ngenes))
    e = found["n"] * share[rows, found["genes"]]
    return write_frame_counts(live, seq, iteration, ncells=ncells, ngenes=ngenes, indptr=found["indptr"],
                              genes=found["genes"], n=found["n"], e=e)


def destinations(records, labels, grid, z):
    """Each record's label at its observed voxel in the per-layer `labels` [nz, ny, nx] (0 off the grid)."""
    i, j, k, inside = observed_voxels(records, grid, *z)
    out = np.zeros(len(records), np.int64)
    out[inside] = np.asarray(labels)[k[inside], j[inside], i[inside]]
    return out


def write_run(data, run_id, layers, records, *, tiled=False, consensus=None, voxel=1.0, x0=X0, y0=Y0, prior=None,
              state="done", stats=True, tile=16, genes=GENES, cells=CELLS, chosen="cells", kind="proseg", counts=1.0,
              window=True, first=None):
    """A run folder as proseg-live leaves it: run.json, live/info.json, genes.txt, cells.txt, transcripts.bin.gz, one
    frame (seq 0, iteration 40) with its cell statistics, frames.jsonl; `prior`: {"layers": [nz, ny, nx]} (dense,
    prior.bin.zst) or {"plane": [ny, nx]} (tiled, prior.tiles), with its "voxel", "x0", "y0". `layers` [nz, ny, nx]:
    the frame's per-layer labels; its 2-D projection is `consensus` (default: layer 0); a tiled frame keeps them as its
    window (whole) unless `window` is False. `counts`: the keep of proseg's count of the frame (counted from its layers:
    the counts file and its line), None for a frame not counted. `first`: per-layer labels of the frame's counts instead
    (a start line's frame 0 that differs from the prior). `chosen`: "cells" (a cell prior) or "nuclei" (nuclei only: no
    start line)."""
    import zstandard
    folder = Path(data) / "runs" / run_id
    live = folder / "live"
    (live / "frames").mkdir(parents=True)
    nz, ny, nx = layers.shape
    consensus = layers[0] if consensus is None else consensus
    meta = {"id": run_id, "kind": kind, "sample": "s", "sample_name": "s",
            "box": [x0, y0, x0 + nx * voxel, y0 + ny * voxel], "mode": "tiled" if tiled else "detail", "state": state,
            "created": "2026-10-21T12:00:00+00:00",
            "options": {}, "speed": "max", "paused": False,
            "prior": {"cells": {"kind": "shapes", "key": "cells"} if chosen == "cells" else None,
                      "nuclei": {"kind": "shapes", "key": "nuclei"}}}
    (folder / "run.json").write_text(json.dumps(meta))
    z = {"grid_origin": 0.0, "grid_spacing": 1 / nz}
    info = {"cells": len(cells), "genes": len(genes), "layers": nz, "z": z}
    if prior is not None:
        geometry = {"x0": prior.get("x0", x0), "y0": prior.get("y0", y0), "voxel": prior.get("voxel", voxel)}
        if "plane" in prior:
            plane = prior["plane"]
            write_tiles(live / "prior.tiles", {"label": plane, "permille": np.where(plane > 0, 800, 0)}, tile)
            info["prior"] = {"file": "prior.tiles", "format": "tiles", "tile": tile, "nx": plane.shape[1],
                             "ny": plane.shape[0], "nz": 1, **geometry}
        else:
            p = prior["layers"]
            raw = np.concatenate([p.ravel(), np.where(p > 0, 800, 0).ravel()]).astype("<u4")
            (live / "prior.bin.zst").write_bytes(zstandard.ZstdCompressor(level=3).compress(raw.tobytes()))
            info["prior"] = {"file": "prior.bin.zst", "nx": p.shape[2], "ny": p.shape[1], "nz": p.shape[0], **geometry}
    (live / "info.json").write_text(json.dumps(info))
    (live / "genes.txt").write_text("".join(g + "\n" for g in genes))
    (live / "cells.txt").write_text("".join(c + "\n" for c in cells))
    (live / "transcripts.bin.gz").write_bytes(gzip.compress(np.asarray(records, RECORD).tobytes()))
    record = {"seq": 0, "iter": 40, "phase": "sampling", "nx": nx, "ny": ny, "nz": nz, "x0": x0, "y0": y0,
              "voxel": voxel, "cells": len(cells)}
    if tiled:
        planes = {"consensus": consensus, "last_change": np.zeros_like(consensus)}
        held = write_tiles(live / "frames/000000.tiles", planes, tile, (layers, 0, 0) if window else None)
        record.update(file="frames/000000.tiles", format="tiles", tile=tile,
                      window=None if held is None else {k: held[k] for k in ("i0", "j0", "nx", "ny")})
    else:
        record["file"] = "frames/000000.bin.zst"
        raw = np.concatenate([consensus.ravel(), np.zeros(nx * ny), layers.ravel()]).astype("<u4")
        (live / record["file"]).write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.tobytes()))
    if stats:
        record["cells_file"] = "frames/000000.cells.zst"
        values = cell_stats(consensus, layers, x0, y0, voxel, len(cells)).astype("<u4")
        (live / record["cells_file"]).write_bytes(zstandard.ZstdCompressor(level=1).compress(values.tobytes()))
    (live / "frames.jsonl").write_text(json.dumps(record) + "\n")
    if counts is not None:
        grid = {"x0": x0, "y0": y0, "voxel": voxel, "nx": nx, "ny": ny, "nz": nz}
        frame_counts(live, 0, 40, layers if first is None else first, np.asarray(records, RECORD), grid,
                     (z["grid_origin"], z["grid_spacing"]), len(cells), len(genes), counts)
    return folder


# ---- proseg's counts and where exports as engine/runs.py serves them (its API: INTERFACES §5.3-5.4) -----------------
def merge_counts(run):
    """The counts lines of the run's live/counts.jsonl merged into its frame records, as the engine's Runs does when
    it reads them (the newest line of a seq wins): counts_file, counted_file, counts."""
    path = run.live / "counts.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    with run.lock:
        for line in lines:
            if line["seq"] < len(run.frames):
                run.frames[line["seq"]].update(counts_file=line["file"], counted_file=line["counted_file"], counts=line)


class Exports:
    """The engine's where exports of a set of runs (Runs.where_export, Runs.request_where) for the tests: the exports
    written (`keep`), and what a paused run's proseg serves when asked (`serve(run_id, box)` -> W or None)."""

    def __init__(self, runs):
        self.runs, self.kept, self.asked, self.serve = runs, {}, [], None

    def keep(self, run_id, meta):
        self.kept.setdefault(run_id, []).append(meta)
        return meta

    def where_export(self, run_id, seq, box):
        from sopa_companion.engine import held
        for meta in sorted(self.kept.get(run_id, []), key=lambda m: -m["token"]):
            if meta["seq"] == seq and held.covers(meta, box) and Path(meta["path"]).exists():
                return meta
        return None

    def request_where(self, run_id, box, wait_s=2.0):
        self.asked.append((run_id, list(box), wait_s))
        return None if self.serve is None else self.serve(run_id, box)


def attach(runs) -> Exports:
    """Runs (the engine's, or the scoring inputs' stand-in) with the counts merged and the tests' where exports."""
    exports = Exports(runs)
    runs.where_export, runs.request_where = exports.where_export, exports.request_where
    for run in list(runs.runs.values()):
        merge_counts(run)
    return exports


def export_of(run, box, records, dest, *, token=1, seq=0, iteration=40, grid=None, pools=None) -> dict:
    """A where export of `box` as proseg-live writes it (tests/counts_fixtures.py): one record per transcript whose
    observed voxel lies in the box's span, its destination `dest` (a label per record), n 1 each; or `pools` given
    whole ({i, j, k, gene, dest, n}). Returns W with its file's "path"."""
    if grid is None:
        with run.lock:
            frame = dict(run.frames[seq])
        grid = {key: frame[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
    if pools is None:
        z = run.info["z"]
        i, j, k, inside = observed_voxels(records, grid, z["grid_origin"], z["grid_spacing"])
        i0, j0, i1, j1 = where_span(box, grid)
        sel = inside & (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
        pools = {"i": i[sel], "j": j[sel], "k": k[sel], "gene": np.asarray(records["gene"])[sel].astype(np.int64),
                 "dest": np.asarray(dest)[sel], "n": np.ones(int(sel.sum()), np.int64)}
    meta = write_where_export(run.live, token, seq, iteration, box, grid, pools)
    return {**meta, "path": str(run.live / meta["file"])}


@pytest.fixture
def make_runs(config):
    """write_run's runs, then the engine's Runs over them (counts merged, where exports as the tests write them) and a
    DeeperWhere; make(...) -> (runs, where), runs.exports the tests' exports."""
    from sopa_companion.engine.deeper_where import DeeperWhere
    from sopa_companion.engine.runs import Runs
    data = Path(config["paths"]["data_dir"])
    made = []
    # Runs written before the engine starts are loaded as they are (not set aside as older runs or crash leftovers).
    config["proseg"]["keep_unsaved"] = True

    def make(analysis=None, **runs):
        for run_id, (layers, records, options) in runs.items():
            write_run(data, run_id, layers, records, **options)
        made.append(Runs(config, None))
        made[-1].exports = attach(made[-1])
        return made[-1], DeeperWhere(made[-1], analysis)
    yield make
    for engine in made:                              # their transcript indexes let go of
        for run in list(engine.runs.values()):
            engine._release(run)


def reply(where, run_id, seq, cell, sources=(), genes=(), own=(), givers=None):
    return json.loads(where.reply(run_id, seq, cell, list(sources), list(genes), list(own), givers=givers))


ASK = {"sources": [1, 3], "genes": ["EDGE", "SPREAD", "LAYER", "FEW", "NOPE"], "own": ["OWN1", "OWN2", "EDGE"]}


def stats_by_gene(found):
    return {s["gene"]: s for s in found["gene_stats"]}


# ---- the measurement, by hand ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("tiled", [True, False])
def test_the_numbers_of_a_hand_made_frame(make_runs, tiled):
    """Whole-sample (2-D columns) and detail (one layer everywhere: the same labels) runs whose proseg keeps every
    transcript it counts (keep 1: w is 1 for the cell's own) give the numbers worked out by hand: the box, owners, the
    cell's count, areas, each gene's giver, near counts, densities and word, and into."""
    runs, where = make_runs(r1=(layout()[None] if tiled else np.stack([layout()] * 4), placed(), {"tiled": tiled}))
    found = reply(where, "r1", 0, 0, **ASK)
    assert found["v"] == 3 and found["run"] == "r1" and (found["seq"], found["iter"], found["cell"]) == (0, 40, 0)
    assert found["basis"] == ("column" if tiled else "voxel") and found["sources"] == [1, 3] and found["near_um"] == 2.0
    # No export kept and none asked (not paused): the frame's per-layer labels place proseg's share (a tiled frame's
    # window holds them here).
    assert (found["held"], found["held_seq"]) == ("approx", None) and runs.exports.asked == []
    # The box: A, B and D (columns 0-29, rows 10-19) + 20 µm, clipped to the raster.
    assert found["box"] == [100.0, 50.0, 150.0, 90.0]
    assert found["genes"] == ["EDGE", "SPREAD", "LAYER", "FEW", "OWN1", "OWN2"]       # NOPE: not the run's
    assert found["roles"] == ["from"] * 4 + ["own"] * 2                                 # EDGE once, as "from"
    assert found["cell_total"] == 68                 # 10 + 10 + 10 + 3 + 20 + 10 + 5 OTHER (never shown)
    assert found["cell_area_um2"] == 100.0 and found["source_area_um2"] == [100.0, 60.0]
    stats = stats_by_gene(found)
    assert stats["EDGE"] == {"gene": "EDGE", "role": "from", "giver": 0, "giver_by": "density", "in_cell": 10, "count": 10,
                             "near_giver": 10, "own_near": [2, 35],
                             "density": {"cell": 10.0, "giver": 20.0, "around": 0.0}, "word": "edge"}
    assert stats["SPREAD"] == {"gene": "SPREAD", "role": "from", "giver": 1, "giver_by": "density", "in_cell": 10, "count": 10,
                               "near_giver": 0, "own_near": [0, 35],
                               "density": {"cell": 10.0, "giver": 10.0, "around": 0.0}, "word": "spread"}
    # LAYER: proseg's count of the cells around, C's 50 (the 50 in no cell are no cell's count), per 100 µm² of C's own
    # footprint in the box (100 µm²: never the gaps between cells nor the box's empty edges): 50, at least half its 10
    # in the cell.
    assert stats["LAYER"] == {"gene": "LAYER", "role": "from", "giver": None, "giver_by": None, "in_cell": 10, "count": 10,
                              "near_giver": None, "own_near": None,
                              "density": {"cell": 10.0, "giver": None, "around": 50.0}, "word": "layer"}
    assert stats["FEW"]["in_cell"] == 3 and stats["FEW"]["word"] == "few" and stats["FEW"]["giver"] is None
    # Each gene's own count, proseg's for the cell in the frame's counts (here every transcript inside it, keep 1).
    assert {g: s["count"] for g, s in stats.items()} == {g: s["in_cell"] for g, s in stats.items()}
    # Own genes have no giver: no edge was measured for them, so no word (few and layer are measured, and still said).
    assert stats["OWN1"] == {"gene": "OWN1", "role": "own", "giver": None, "giver_by": None, "in_cell": 20, "count": 20,
                             "near_giver": None, "own_near": None,
                             "density": {"cell": 20.0, "giver": None, "around": 0.0}, "word": None}
    assert (stats["OWN2"]["in_cell"], stats["OWN2"]["word"]) == (10, None)
    # Into: proseg's count for A of its own genes inside each source's outline (Σ w). Placed approximately, a transcript
    # inside a source is the source's by its 3-D owner, so A holds none of them here (the page shows into on exact
    # replies only); the reply says it reads the averaged count (a frame counted for a token).
    assert found["into"] == [
        {"source": 0, "gene": "OWN1", "inside": 0.0, "near_cell": 0.0, "source_own_near": [0.0, 0.0]},
        {"source": 0, "gene": "OWN2", "inside": 0.0, "near_cell": 0.0, "source_own_near": [0.0, 0.0]},
        {"source": 1, "gene": "OWN1", "inside": 0.0, "near_cell": 0.0, "source_own_near": [0.0, 0.0]},
        {"source": 1, "gene": "OWN2", "inside": 0.0, "near_cell": 0.0, "source_own_near": [0.0, 0.0]}]
    assert found["count"] == "averaged"
    # Every transcript of the genes asked in the box, with its owner: 0 none, 1 A, 2 B, 3 D, 255 another cell (C),
    # whether it lies near (version 2's "near") and proseg's share of it for A (version 3's "w").
    points = found["points"]
    assert set(points) == {"x", "y", "z", "gene", "owner", "near", "w"} and not found["truncated"]
    assert len(points["x"]) == 203 and all(len(v) == 203 for v in points.values())
    records = placed()
    asked = [g for g, name in enumerate(GENES) if name in found["genes"]]
    inside = (records["x"] < 150) & np.isin(records["gene"], asked)
    assert points["x"] == [round(float(v), 2) for v in records["x"][inside]]      # one 100 µm cell: the run's order
    assert points["y"] == [round(float(v), 2) for v in records["y"][inside]] and points["z"] == [0.5] * 203
    assert [found["genes"][g] for g in points["gene"]] == [GENES[g] for g in records["gene"][inside]]
    owners = dict(zip(*np.unique(points["owner"], return_counts=True)))
    assert owners == {1: 63, 2: 32, 3: 8, 255: 50, 0: 50}
    assert points["w"] == [1.0 if o == 1 else 0.0 for o in points["owner"]]
    # Near: A's 10 EDGE (0.5 µm from their giver B) and B's three OWN1 within 2 µm of A; SPREAD's giver D lies 6.5 µm
    # away, LAYER and FEW have no giver (FEW hugs B all the same), A's own OWN1 beside B is A's own: none of those.
    shown = records[inside]
    assert points["near"] == [int((GENES[g] == "EDGE" and 110 <= x < 120) or (GENES[g] == "OWN1" and 120 <= x < 122))
                              for x, g in zip(shown["x"], shown["gene"])]
    assert sum(points["near"]) == 13


def test_proseg_keeps_a_share_each_number_is_its_count(make_runs):
    """Proseg keeps a share of each gene of A (its averaged count over what it assigns: q = e / n), so each of A's
    transcripts is w = q of its gene, and every number of the reply is proseg's count: the cell's total, each gene's in
    the cell, near its giver and the rest of the cell's RNA, the densities; the sources' own counts (their keeps) set
    their densities and so which one is a gene's giver. Owners, the box and into stay where the transcripts lie."""
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, GENES.index("EDGE")], keep[0, GENES.index("SPREAD")], keep[0, GENES.index("OWN1")] = 0.75, 0.5, 0.25
    keep[1, GENES.index("SPREAD")], keep[3, GENES.index("SPREAD")] = 0.5, 0.25    # B 8 x 0.5 / 100, D 6 x 0.25 / 60
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {"counts": keep}))
    found = reply(where, "r1", 0, 0, **ASK)
    assert found["held"] == "approx"
    points = found["points"]
    q = {g: keep[0, GENES.index(g)] for g in GENES}
    assert points["w"] == [q[found["genes"][g]] if o == 1 else 0.0 for o, g in zip(points["owner"], points["gene"])]
    # 10 x 0.75 EDGE + 10 x 0.5 SPREAD + 10 LAYER + 3 FEW + 20 x 0.25 OWN1 + 10 OWN2 + 5 OTHER.
    assert found["cell_total"] == 7.5 + 5 + 10 + 3 + 5 + 10 + 5
    stats = stats_by_gene(found)
    assert (stats["EDGE"]["in_cell"], stats["EDGE"]["near_giver"]) == (7.5, 7.5)
    # The rest of A's RNA near B: its two OWN1 there (0.25 each); in all OWN1, OWN2 and OTHER (5 + 10 + 5).
    assert stats["EDGE"]["own_near"] == [0.5, 20.0] and stats["EDGE"]["word"] == "edge"
    assert stats["EDGE"]["density"] == {"cell": 7.5, "giver": 20.0, "around": 0.0}
    # SPREAD's giver by density in proseg's counts: B 4 in 100 µm² against D 1.5 in 60.
    assert (stats["SPREAD"]["giver"], stats["SPREAD"]["in_cell"], stats["SPREAD"]["density"]["giver"]) == (0, 5.0, 4.0)
    assert stats["OWN1"]["in_cell"] == 5.0 and stats["OWN1"]["density"]["cell"] == 5.0
    # A's OWN1 inside B is B's (approximately placed): proseg counts none of it for A here.
    assert found["into"][0] == {"source": 0, "gene": "OWN1", "inside": 0.0, "near_cell": 0.0,
                                "source_own_near": [0.0, 0.0]}
    assert len(points["x"]) == 203 and found["box"] == [100.0, 50.0, 150.0, 90.0]


def test_the_cells_around_are_proseg_s_count_of_them_per_their_own_footprints(make_runs):
    """H1: "around" is proseg's count of the cells around (every cell but the cell and its sources), summed over the box
    as the cell's own count is, per 100 µm² of those cells' own footprints in the box (their voxels whose column lies in
    it, x v² / layers): never the gaps between cells nor the box's empty edges, and never RNA in no cell. C's 50 LAYER
    in its 100 µm², and E's 13 in the 30 µm² of E inside the box (its other half, past the box's edge, and the LAYER
    there count in neither): 63 in 130 µm². Proseg keeping a share of C's LAYER (C's own keep) scales it; A's keep moves
    only A's side; RNA proseg moved into A from no cell counts in A, never around."""
    layer = GENES.index("LAYER")
    plane = layout()
    plane[30:36, 45:55] = 5                                           # E: x 145-155, y 80-86 (half past the box)
    layers = np.stack([plane] * 4)
    extra = np.array([(146.5, 80.5 + i % 6, 0.5, layer) for i in range(13)]       # E's LAYER in the box ...
                     + [(152.5, 81.5, 0.5, layer)] * 4, RECORD)                  # ... and past it
    records = np.concatenate([placed(), extra])
    # (C's keep of LAYER, A's, E's, the word): 48.46 / 12.40 per 100 µm² read "layer" against A's 10 (or 5); 3.03 does
    # not (under half of 10).
    for n, (c_share, a_share, e_share, word) in enumerate(((1.0, 1.0, 1.0, "layer"), (1.0, 0.5, 1.0, "layer"),
                                                           (0.0625, 1.0, 1.0, "layer"), (0.0625, 1.0, 0.0625, None))):
        keep = np.ones((len(CELLS), len(GENES)))
        keep[2, layer], keep[0, layer], keep[4, layer] = c_share, a_share, e_share
        runs, where = make_runs(**{f"k{n}": (layers, records, {"counts": keep})})
        found = stats_by_gene(reply(where, f"k{n}", 0, 0, sources=[1, 3], genes=["LAYER"]))["LAYER"]
        assert (found["in_cell"], found["density"]["cell"]) == (10 * a_share, 10 * a_share)
        assert found["density"]["around"] == round(100 * (50 * c_share + 13 * e_share) / 130, 2), found
        assert found["word"] == word, found
    # No other cell in the box (C and E elsewhere): nothing around, no density, no layer word.
    alone = layout()
    alone[24:34, 10:20] = 0
    runs, where = make_runs(alone=(np.stack([alone] * 4), placed(), {}))
    found = stats_by_gene(reply(where, "alone", 0, 0, sources=[1, 3], genes=["LAYER"]))["LAYER"]
    assert found["density"]["around"] is None and found["word"] is None and found["in_cell"] == 10
    # Exact: 10 of the 50 LAYER transcripts lying in no cell went to A (proseg's pools): A's count is 20, the cells
    # around still hold C's 50 (the other 40 lie in no cell: nobody's count).
    dest = destinations(placed(), np.stack([layout()] * 4), {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY,
                                                             "nz": 4}, (0.0, 0.25))
    moved = np.flatnonzero((placed()["gene"] == layer) & (placed()["x"] < 142) & (placed()["y"] < 60))
    assert moved.size == 10 and not dest[moved].any()
    dest[moved] = 1
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {"counts": None}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, placed(), None, None, len(CELLS), len(GENES), 1.0, dest=dest)
    merge_counts(run)
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=2))
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=["LAYER"])
    stats = stats_by_gene(found)["LAYER"]
    assert found["held"] == "exact" and (stats["in_cell"], stats["density"]["cell"]) == (20, 20.0)
    assert stats["density"]["around"] == 50.0 and stats["word"] == "layer"
    assert sum(w for w, o in zip(found["points"]["w"], found["points"]["owner"]) if o == 0) == 10, "drawn in no cell"


def counts_table(table: dict):
    """A frame's counts (counts_file.Counts) of {cell: {gene: (n, e)}} over 4 cells and 2 genes, read back."""
    from sopa_companion.analysis import counts_file
    import tempfile
    indptr, genes, n, e = [0], [], [], []
    for c in range(4):
        for g, (nn, ee) in sorted(table.get(c, {}).items()):
            genes.append(g)
            n.append(nn)
            e.append(ee)
        indptr.append(len(genes))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "000000.counts.zst"
        counts_file.write(path, ncells=4, ngenes=2, iteration=1, indptr=indptr, genes=genes, n=n, e=e)
        return counts_file.read(path)


def test_others_is_every_other_cells_share_of_each_transcript():
    """held.others, by weights' rules, for every cell but those left out at once (the cell and its sources): exact from
    the export's pools of the record's bucket whose destination is another cell (min(1, its pools / |B|) x its keep q =
    e / n of the gene), never background (dest 0); approximate from the 3-D owner at the record's voxel x its keep;
    none without either. Cells 0 (the cell) and 3 (a source) are left out; 1 keeps half of gene 0, 2 all of it."""
    from sopa_companion.engine import held
    grid = {"x0": 0.0, "y0": 0.0, "voxel": 1.0, "nx": 4, "ny": 1, "nz": 1}
    records = np.array([(0.5, 0.5, 0.5, 0), (1.5, 0.5, 0.5, 0), (1.2, 0.5, 0.5, 0), (2.5, 0.5, 0.5, 0),
                        (3.5, 0.5, 0.5, 0), (3.5, 0.5, 0.5, 1)], RECORD)
    counts = counts_table({0: {0: (1, 1.0)}, 1: {0: (2, 1.0), 1: (1, 0.25)}, 2: {0: (1, 1.0)}, 3: {0: (2, 2.0)}})
    assert np.allclose(held.keep_at(counts, [1, 2, 1, 3, 0], [0, 0, 1, 0, 1]), [0.5, 1.0, 0.25, 1.0, 0.0])
    # Pools by bucket: (0) cell 1 x 1; (1) cell 2 x 1 and background x 1 over its two records; (2) the cell itself;
    # (3) source 3 x 1 and cell 1 x 2 (more than its one record: d at most 1); gene 1 at (3): cell 1 x 1.
    pools = {"i": np.array([0, 1, 1, 2, 3, 3, 3]), "j": np.zeros(7, np.int64), "k": np.zeros(7, np.int64),
             "gene": np.array([0, 0, 0, 0, 0, 0, 1]), "dest": np.array([2, 3, 0, 1, 4, 2, 2]),
             "n": np.array([1, 1, 1, 1, 1, 2, 1])}
    export = held.WhereExport(token=1, seq=0, iter=1, box=(0.0, 0.0, 4.0, 1.0), margin_um=24.0, x0=0.0, y0=0.0,
                              voxel=1.0, nx=4, ny=1, nz=1, i0=0, j0=0, i1=3, j1=0,
                              **{k: np.asarray(v, np.int64) for k, v in pools.items()})
    found, how = held.others(records, [0, 3], counts, grid, 0.0, 1.0, export=export)
    assert how == "exact" and np.allclose(found, [0.5, 0.5, 0.5, 0.0, 0.5, 0.25])
    # Approximate: the owner at each record's voxel (cells 1, 2, 2, the cell, none, none) x its keep.
    layers = (np.array([[[2, 3, 1, 0]]], np.uint32), 0, 0)
    found, how = held.others(records, [0, 3], counts, grid, 0.0, 1.0, layers=layers)
    assert how == "approx" and np.allclose(found, [0.5, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert held.others(records, [0, 3], counts, grid, 0.0, 1.0)[1] == "none"
    # On f (the count proseg saves) the keep is f / n; a frame without f has none to give.
    with pytest.raises(ValueError):
        held.others(records, [0, 3], counts, grid, 0.0, 1.0, "f", layers=layers)


def test_into_is_proseg_s_count_for_the_cell_inside_each_neighbours_outline(make_runs):
    """H12: into reads proseg's count for the cell (Σ w, to 3 decimals) of its own genes lying inside each source's
    drawn outline, of them within NEAR_UM of the cell, and of the other genes there [near the cell, inside]: five OWN1
    at one voxel inside B, 1.5 µm from A, of which proseg gives A one (its pools there: 1 to A, 4 to B) read 1.0 and
    1.0, never the five the outline holds; B's own RNA there is B's (0). A's count of OWN1 holds that one too."""
    own = GENES.index("OWN1")
    records = np.concatenate([placed(), np.array([(121.5, 66.5, 0.5, own)] * 5, RECORD)])
    grid = {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 4}
    dest = destinations(records, np.stack([layout()] * 4), grid, (0.0, 0.25))
    added = np.arange(len(placed()), len(records))
    assert (dest[added] == 2).all()
    dest[added[0]] = 1                                          # proseg's pools at that voxel: 1 to A, 4 to B
    runs, where = make_runs(r1=(np.stack([layout()] * 4), records, {"counts": None}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), 1.0, dest=dest)
    merge_counts(run)
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], records, dest, token=3))
    found = reply(where, "r1", 0, 0, **ASK)
    assert found["held"] == "exact"
    assert found["into"][0] == {"source": 0, "gene": "OWN1", "inside": 1.0, "near_cell": 1.0,
                                "source_own_near": [0.0, 0.0]}
    assert stats_by_gene(found)["OWN1"]["in_cell"] == 21.0
    assert all(isinstance(entry[key], float) for entry in found["into"] for key in ("inside", "near_cell"))


def test_a_frame_read_on_the_count_proseg_saves_shares_by_f(make_runs):
    """A frame proseg counted for a pause, the hold, a finish or the end, with f (counts_file.count_of "sampled"): w =
    d_B x f / n, so over a box holding the cell's pools the exact w add up to its f (cell_total Σ f, each gene's in_cell
    its f), and the reply says "sampled"; the test's measurement (measure: count "averaged" unless asked) reads e / n of
    the same frame. Counted first for a token (averaged), then again for the hold: its replies on e and on f are kept
    apart, each served for the count the frame is read on."""
    grid = {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 4}
    dest = destinations(placed(), np.stack([layout()] * 4), grid, (0.0, 0.25))
    found = counts_of(dest, placed()["gene"], len(CELLS), len(GENES))
    e, f = found["n"] * 0.75, np.maximum(found["n"] - 1, 0)       # A: Σ e 68 x 0.75 = 51, Σ f 68 - 7 = 61
    fields = {key: found[key] for key in ("indptr", "genes", "n")}
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {"counts": None}))
    run = runs.get("r1")
    write_frame_counts(run.live, 0, 40, ncells=len(CELLS), ngenes=len(GENES), e=e, reason="request", token=1,
                       **fields)
    merge_counts(run)
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=2))
    averaged = reply(where, "r1", 0, 0, **ASK)
    assert (averaged["held"], averaged["count"], averaged["cell_total"]) == ("exact", "averaged", 51.0)
    write_frame_counts(run.live, 0, 40, ncells=len(CELLS), ngenes=len(GENES), e=e, f=f, reason="hold", **fields)
    merge_counts(run)                                       # the hold counts the same frame again, with f
    sampled = reply(where, "r1", 0, 0, **ASK)
    assert (sampled["held"], sampled["count"], sampled["cell_total"]) == ("exact", "sampled", 61.0)
    assert stats_by_gene(sampled)["EDGE"]["in_cell"] == 9.0 and stats_by_gene(averaged)["EDGE"]["in_cell"] == 7.5
    assert {key[7] for key in where.replies} == {"averaged", "sampled"}      # kept apart
    measured, _ = where.measure("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"])
    assert (measured["count"], measured["cell_total"]) == ("averaged", 51.0)
    on_f, _ = where.measure("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"], count="sampled")
    assert {k: v for k, v in on_f.items() if k != "ms"} == {k: v for k, v in sampled.items() if k != "ms"}


def test_w_is_exact_from_proseg_s_where_export_with_each_bucket_s_transcripts_from_the_run(make_runs):
    """With proseg's export of the box, w = d_B x q: d_B the share of the pooled transcripts of the transcript's bucket
    (its gene at its observed voxel) that A owns, the denominator the bucket's transcripts in the run (|B|), held at 1;
    the pools come from where the transcripts went (drift), not from where they lie: RNA lying in B that proseg moved
    into A is A's (it glows outside A's outline), A's pools give nothing to a bucket of no transcript."""
    edge, own = GENES.index("EDGE"), GENES.index("OWN1")
    records = np.array([(119.2, 62.5, 0.5, edge)] * 2 + [(119.7, 62.5, 0.5, edge)] * 2      # b1: column 19, row 12
                       + [(121.5, 62.5, 0.5, edge)] * 3                                      # b2: column 21 (in B)
                       + [(112.5, 65.5, 0.5, edge)] * 2                                      # b3: column 12, row 15
                       + [(115.5, 65.5, 0.5, edge)]                                          # b4: column 15, row 15
                       + [(113.5, 63.5, 0.5, own)] * 5                                       # b5: OWN1 at (13, 13)
                       + [(125.5, 65.5, 0.5, edge)] * 4, RECORD)                             # B's own EDGE
    # Proseg's pools (per bucket, by destination): b1 A 2, B 2; b2 A 3 (moved into A); b3 A 1, no cell 1; b4 A 5 (more
    # than its 1 transcript: d at most 1); b5 A 5; B's own 4; and a pool of A's in a bucket no transcript lies in.
    pools = {"i": [19, 19, 21, 12, 12, 15, 13, 25, 30], "j": [12, 12, 12, 15, 15, 15, 13, 15, 15],
             "k": [2] * 9, "gene": [edge] * 6 + [own, edge, edge], "dest": [1, 2, 1, 1, 0, 1, 1, 2, 1],
             "n": [2, 2, 3, 1, 1, 5, 5, 4, 7]}
    dest = np.array([1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])   # n per cell and gene as the pools
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, edge], keep[0, own] = 0.75, 0.5                 # (keeps proseg's file holds exactly: e in 1/32 steps)
    runs, where = make_runs(r1=(np.stack([layout()] * 4), records, {"counts": None}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), keep, dest=dest)
    merge_counts(run)
    meta = runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], records, None, token=4,
                                             pools={k: np.asarray(v) for k, v in pools.items()}))
    found = reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"], own=["OWN1"])
    assert (found["held"], found["held_seq"]) == ("exact", 0) and found["box"] == [100.0, 50.0, 150.0, 90.0]
    assert meta["box"] == [100.0, 50.0, 150.0, 90.0]
    q_edge, q_own = 0.75, 0.5
    want = [0.5 * q_edge] * 4 + [1.0 * q_edge] * 3 + [0.5 * q_edge] * 2 + [q_edge] + [q_own] * 5 + [0.0] * 4
    assert found["points"]["w"] == pytest.approx(want, abs=1e-12)
    assert found["points"]["owner"] == [1] * 4 + [2] * 3 + [1] * 8 + [2] * 4
    assert found["cell_total"] == pytest.approx(sum(want), abs=1e-3)
    stats = stats_by_gene(found)
    assert stats["EDGE"]["in_cell"] == pytest.approx(sum(want[:10]), abs=1e-3)
    assert stats["OWN1"]["in_cell"] == pytest.approx(5 * q_own, abs=1e-3)
    # B's share by where the transcripts went: half of each of b1's 4, all of its own 4: its density of EDGE is
    # proseg's count of it.
    assert stats["EDGE"]["density"]["giver"] == pytest.approx(100 * (4 * 0.5 + 4) / 100, abs=0.01)
    # Near: b1's (0.3 and 0.8 µm from B) and b2's (in B: proseg moved them into A) are A's and near B; b3, b4 not.
    assert found["points"]["near"][:10] == [1] * 7 + [0] * 3
    assert stats["EDGE"]["near_giver"] == pytest.approx(4 * 0.5 * q_edge + 3 * q_edge, abs=1e-3)


def test_a_bucket_across_the_box_edge_is_shared_among_all_its_transcripts(make_runs):
    """On a 1.5 µm grid the box's edge (20 µm past the cell) falls inside a column: a bucket there holds transcripts on
    both sides of it. Exact w = the pooled share over every transcript of the bucket (|B| from the run, the box's
    columns and one more on each side), so the two inside the box get 2 of 4 x q, never 2 of the 2 shown."""
    edge, own = GENES.index("EDGE"), GENES.index("OWN1")
    layers = np.zeros((2, 40, 60), np.uint32)               # 1.5 µm: x 100-190, y 50-110
    layers[:, 10:16, 20:26] = 1                              # A: x 130-139, y 65-74; its box [110, 50, 159, 94]
    records = np.array([(109.2, 73.0, 0.1, edge), (109.7, 73.0, 0.1, edge),     # column 6 (x 109-110.5), row 15:
                        (110.1, 73.0, 0.1, edge), (110.4, 73.0, 0.1, edge)]     # two outside the box, two inside
                       + [(133.0, 70.0, 0.1, own)] * 5, RECORD)                  # A's own, column 22, row 13
    dest = np.array([1, 0, 1, 0] + [1] * 5)                  # two of the bucket's four went to A (n 2)
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, edge] = 0.75
    runs, where = make_runs(r1=(layers, records, {"counts": None, "voxel": 1.5}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), keep, dest=dest)
    merge_counts(run)
    pools = {"i": np.array([6, 6, 22]), "j": np.array([15, 15, 13]), "k": np.array([0, 0, 0]),
             "gene": np.array([edge, edge, own]), "dest": np.array([1, 0, 1]), "n": np.array([2, 2, 5])}
    meta = runs.exports.keep("r1", export_of(run, [110.0, 50.0, 159.0, 94.0], records, None, token=5, pools=pools))
    assert (meta["i0"], meta["i1"]) == (6, 39)               # the export's span starts with the edge's column
    found = reply(where, "r1", 0, 0, genes=["EDGE"], own=["OWN1"])
    assert (found["held"], found["held_seq"], found["box"]) == ("exact", 0, [110.0, 50.0, 159.0, 94.0])
    shown = {(x, found["genes"][g]): w for x, g, w in zip(found["points"]["x"], found["points"]["gene"],
                                                         found["points"]["w"])}
    assert len(found["points"]["x"]) == 7, "the points are the box's: the two outside it only share their bucket"
    assert shown == {(110.1, "EDGE"): 0.375, (110.4, "EDGE"): 0.375, (133.0, "OWN1"): 1.0}
    assert found["cell_total"] == pytest.approx(2 * 0.375 + 5, abs=1e-9)


def test_w_is_approximate_from_the_frames_3d_labels(make_runs):
    """Without an export, w = [the transcript's observed voxel is A's in the frame's per-layer labels] x q: a voxel A
    holds only in its top layer counts for the transcripts there (and only for them), whatever the flat projection
    says; q per gene."""
    edge = GENES.index("EDGE")
    layers = np.stack([layout()] * 4)
    layers[3, 10:20, 18:20] = 2                              # B holds A's two right columns in the top layer
    records = np.array([(118.5, 65.5, 0.9, edge), (118.5, 65.5, 0.1, edge), (111.5, 62.5, 0.3, edge)], RECORD)
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, edge] = 0.625
    runs, where = make_runs(r1=(layers, records, {"counts": keep}))
    found = reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])
    assert found["held"] == "approx" and found["points"]["owner"] == [2, 1, 1]
    assert found["points"]["w"] == [0.0, 0.625, 0.625]
    assert found["cell_total"] == 1.25 and found["gene_stats"][0]["in_cell"] == 1.25


def test_too_few_is_proseg_s_own_count_of_the_gene_where_its_shares_are_placed_approximately(make_runs):
    """Placed approximately (the frame's layers x the keep), a gene's sum of w misses RNA proseg moved into the cell and
    keeps RNA it moved out, so it is not proseg's count of the gene: each gene's "count" is (the frame's counts), and
    "too few to tell" reads it, and what was placed too (the other words are judged on that). Proseg moved seven of A's
    SPREAD out (it counts 3, its layers show 10: too few counted, under FEW_IN_CELL 4) and four FEW lying in no cell
    into A (it counts 7, its layers show 3: too few placed); exact, with its export, the sums are those counts, and 7
    FEW are not too few."""
    few, spread = GENES.index("FEW"), GENES.index("SPREAD")
    records = np.concatenate([placed(), np.array([(135.5, 60.5 + i, 0.5, few) for i in range(4)], RECORD)])
    layers = np.stack([layout()] * 4)
    grid = {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 4}
    dest = destinations(records, layers, grid, (0.0, 0.25))
    moved_in = np.flatnonzero((records["gene"] == few) & (records["x"] > 130))
    moved_out = np.flatnonzero((records["gene"] == spread) & (dest == 1))[:7]
    assert moved_in.size == 4 and not dest[moved_in].any() and moved_out.size == 7
    dest[moved_in], dest[moved_out] = 1, 0
    runs, where = make_runs(r1=(layers, records, {"counts": None}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), 1.0, dest=dest)
    merge_counts(run)
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=["FEW", "SPREAD"])
    stats = stats_by_gene(found)
    assert found["held"] == "approx"
    assert (stats["FEW"]["in_cell"], stats["FEW"]["count"]) == (3.0, 7.0) and stats["FEW"]["word"] == "few"
    assert (stats["SPREAD"]["in_cell"], stats["SPREAD"]["count"]) == (10.0, 3.0) and stats["SPREAD"]["word"] == "few"
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], records, dest, token=2))
    exact = stats_by_gene(reply(where, "r1", 0, 0, sources=[1, 3], genes=["FEW", "SPREAD"]))
    assert [(exact[g]["in_cell"], exact[g]["count"]) for g in ("FEW", "SPREAD")] == [(7.0, 7.0), (3.0, 3.0)]
    assert exact["FEW"]["word"] != "few" and exact["SPREAD"]["word"] == "few"


@pytest.mark.parametrize("case", ["spread", "layer", "edge"])
def test_too_few_placed_is_too_few_to_tell_however_many_proseg_counts(make_runs, case):
    """Placed approximately, the words are judged on what the frame's layers place, so a gene with a giver that proseg
    counts 4 or more of, with fewer than 4 placed, is "few", never a word judged on those few: proseg counts 7 of A's
    SPREAD (3 lying in A, 4 lying in no cell moved in), its layers place the 3; spread through A (its giver D, by
    density, 6.5 µm away), with 50 more around it (as common around, on 3), or along B's edge (B named by the window's
    answer). Exact, with proseg's export, all 7 are placed and judged."""
    spread = GENES.index("SPREAD")
    at = 119.5 if case == "edge" else 112.5
    records = placed()
    records = np.concatenate([records[~((records["gene"] == spread) & (records["x"] == 112.5))],
                              np.array([(at, 60.5 + i, 0.5, spread) for i in range(3)]
                                       + [(135.5, 60.5 + i, 0.5, spread) for i in range(4)]
                                       + [(140.5 + i, 51.5 + j, 0.5, spread) for i in range(10) for j in range(5)
                                          if case == "layer"], RECORD)])
    layers = np.stack([layout()] * 4)
    grid = {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 4}
    dest = destinations(records, layers, grid, (0.0, 0.25))
    moved_in = np.flatnonzero((records["gene"] == spread) & (records["x"] == 135.5))
    assert moved_in.size == 4 and not dest[moved_in].any()
    dest[moved_in] = 1
    runs, where = make_runs(r1=(layers, records, {"counts": None}))
    run = runs.get("r1")
    frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), 1.0, dest=dest)
    merge_counts(run)
    givers = {"SPREAD": 1} if case == "edge" else None
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=["SPREAD"], givers=givers)
    stats = stats_by_gene(found)["SPREAD"]
    assert found["held"] == "approx"
    assert (stats["giver"], stats["giver_by"]) == ((0, "result") if case == "edge" else (1, "density"))
    assert (stats["in_cell"], stats["count"], stats["word"]) == (3.0, 7.0, "few")
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], records, dest, token=2))
    exact = reply(where, "r1", 0, 0, sources=[1, 3], genes=["SPREAD"], givers=givers)
    stats = stats_by_gene(exact)["SPREAD"]
    assert exact["held"] == "exact" and (stats["in_cell"], stats["count"]) == (7.0, 7.0) and stats["word"] == "spread"


def test_the_page_tells_too_few_placed_by_the_same_floor():
    """The page says why a gene is "few" (too few counted, or too few placed of what proseg counts) by the engine's
    floor: frontend/src/qc-deeper.js WHERE_FEW."""
    import re
    from sopa_companion.engine.deeper_where import FEW_IN_CELL
    page = (Path(__file__).resolve().parents[1] / "frontend/src/qc-deeper.js").read_text()
    assert re.search(rf"\bexport const WHERE_FEW = {FEW_IN_CELL};", page)


def test_a_tiled_frames_window_shares_by_its_layers_never_by_its_column(make_runs):
    """A whole-sample frame whose window holds the box: w from the window's per-layer labels, never from the flat
    column it is drawn by. B holds A's two right columns in the top layer only, the column says A: a transcript there
    in the top layer is drawn in A, and is B's."""
    edge = GENES.index("EDGE")
    layers = np.stack([layout()] * 4)
    layers[3, 10:20, 18:20] = 2
    records = np.array([(118.5, 65.5, 0.9, edge), (118.5, 65.5, 0.1, edge), (111.5, 62.5, 0.3, edge)], RECORD)
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, edge] = 0.625
    runs, where = make_runs(r1=(layers, records, {"counts": keep, "tiled": True, "consensus": layout()}))
    found = reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])
    assert (found["held"], found["held_seq"], found["basis"]) == ("approx", None, "column")
    assert found["points"]["owner"] == [1, 1, 1], "drawn by the column"
    assert found["points"]["w"] == [0.0, 0.625, 0.625], "shared by the window's 3-D labels"
    assert found["cell_total"] == 1.25 and found["gene_stats"][0]["in_cell"] == 1.25


def test_an_export_gone_before_its_reply_is_worked_out_is_approximate_and_says_so(make_runs, monkeypatch):
    """Proseg keeps only its 4 newest where exports: one gone between finding it and reading it gives the frame's
    layers' share, said "approx" and kept as such; once an export is readable again the reply is exact."""
    from sopa_companion.engine import held
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    dest = destinations(placed(), np.stack([layout()] * 4), {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY,
                                                             "nz": 4}, (0.0, 0.25))
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=3))
    real = held.read_where

    def gone(meta, path=None):
        raise FileNotFoundError("deleted by proseg meanwhile")
    monkeypatch.setattr(held, "read_where", gone)
    first = reply(where, "r1", 0, 0, **ASK)
    assert (first["held"], first["held_seq"]) == ("approx", None)
    assert [k[-2:] for k in where.replies] == [("approx", None)], "kept as what it came from"
    monkeypatch.setattr(held, "read_where", real)
    second = reply(where, "r1", 0, 0, **ASK)
    assert (second["held"], second["held_seq"]) == ("exact", 0) and second["points"]["w"] == first["points"]["w"]


def test_counts_gone_before_the_reply_is_worked_out_hold_nothing(make_runs, monkeypatch):
    """The frame's counts pruned from the spool between finding them and reading them: no keep, so nothing is held
    (held "none", every w 0), never a guess, and no gene gets a word."""
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    monkeypatch.setattr(where, "_counts", lambda *a, **k: None)
    found = reply(where, "r1", 0, 0, **ASK)
    assert (found["held"], found["held_seq"], found["cell_total"]) == ("none", None, 0)
    assert set(found["points"]["w"]) == {0.0}
    assert all(s["in_cell"] == 0 and s["word"] is None for s in found["gene_stats"])


def test_a_frame_whose_counts_are_gone_holds_nothing_and_gives_no_word(make_runs):
    """A detail frame whose counts file the spool pruned (it keeps the 8 newest counted files and a reserve), with no
    reply of it kept: nothing held (held "none", every w 0), and no gene gets a word, never "few" (that proseg counts
    under 4 of them for the cell, where it counts 10 of EDGE). Where the transcripts lie stays: owners, the giver the
    window's answer names, near flags, into."""
    givers = {"EDGE": 1, "SPREAD": 1, "LAYER": 3, "OWN1": 1}
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    before = reply(where, "r1", 0, 0, **ASK, givers=givers)
    assert before["held"] == "approx" and stats_by_gene(before)["EDGE"]["in_cell"] == 10
    assert stats_by_gene(before)["EDGE"]["word"] == "edge" and stats_by_gene(before)["FEW"]["word"] == "few"
    with run.lock:
        name = run.frames[0]["counts_file"]
    (run.live / name).unlink()
    where.replies.clear()                                   # no reply of the frame kept (else it stands: below)
    found = reply(where, "r1", 0, 0, **ASK, givers=givers)
    assert (found["held"], found["held_seq"], found["cell_total"]) == ("none", None, 0)
    assert set(found["points"]["w"]) == {0.0}
    assert [(s["gene"], s["in_cell"], s["word"]) for s in found["gene_stats"]] == \
        [(s["gene"], 0, None) for s in before["gene_stats"]]
    assert [(s["giver"], s["giver_by"]) for s in found["gene_stats"]] == \
        [(s["giver"], s["giver_by"]) for s in before["gene_stats"]]
    for key in ("x", "y", "z", "gene", "owner", "near"):
        assert found["points"][key] == before["points"][key], key
    assert (found["box"], found["into"], found["basis"]) == (before["box"], before["into"], "voxel")


def test_an_approximate_reply_kept_stands_once_its_frames_counts_are_gone(make_runs):
    """A frame's layers and keep do not change: its approximate reply kept is given again once its counts file is
    pruned (never one holding nothing in its place); a reply not kept (another cell) holds nothing, with no word."""
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    first = where.reply("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"])
    assert json.loads(first)["held"] == "approx"
    with run.lock:
        name = run.frames[0]["counts_file"]
    (run.live / name).unlink()
    assert where.reply("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"]) == first
    assert [k[-2:] for k in where.replies] == [("approx", None)]
    other = reply(where, "r1", 0, 2, sources=[0], genes=["LAYER"])
    assert (other["held"], [s["word"] for s in other["gene_stats"]]) == ("none", [None])
    assert reply(where, "r1", 0, 0, **ASK, givers={})["held"] == "none", "another reply of the cell: not kept"


def test_counts_gone_while_an_exact_reply_is_worked_out_leave_the_approximate_reply_kept(make_runs, monkeypatch):
    """An export of the frame arrives after its approximate reply was kept, and the frame's counts are pruned between
    the plan and its reading: nothing would be held, so the approximate reply kept stands (and nothing else is kept);
    with the counts there, the exact reply."""
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    first = where.reply("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"])
    dest = destinations(placed(), np.stack([layout()] * 4), {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY,
                                                             "nz": 4}, (0.0, 0.25))
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=3))
    with monkeypatch.context() as m:
        m.setattr(where, "_counts", lambda *a, **k: None)
        assert where.reply("r1", 0, 0, ASK["sources"], ASK["genes"], ASK["own"]) == first
    assert [k[-2:] for k in where.replies] == [("approx", None)]
    exact = reply(where, "r1", 0, 0, **ASK)
    assert (exact["held"], exact["held_seq"]) == ("exact", 0)
    assert exact["points"]["w"] == json.loads(first)["points"]["w"], "no drift here: the layers' share is exact"


@pytest.mark.parametrize("why", ["no window", "a window elsewhere", "no counts"])
def test_without_an_export_a_tiled_frame_outside_its_window_holds_nothing_never_its_column(make_runs, why):
    """A past tiled frame whose window misses the box (or with none), or a frame proseg did not count: w is 0 for every
    point (no pink dots), held "none"; the flat projection's owner is still drawn, never taken as proseg's share."""
    options = {"tiled": True}
    if why == "no window":
        options["window"] = False
    elif why == "no counts":
        options["counts"] = None
    runs, where = make_runs(r1=(layout()[None], placed(), options))
    if why == "a window elsewhere":                          # the window: columns 40-59 only (x 140-160)
        run = runs.get("r1")
        plane = layout()
        write_tiles(run.live / "frames/000000.tiles", {"consensus": plane, "last_change": np.zeros_like(plane)}, 16,
                    (plane[None, :, 40:], 40, 0))
    found = reply(where, "r1", 0, 0, **ASK)
    assert (found["held"], found["held_seq"]) == ("none", None) and found["basis"] == "column"
    assert found["points"]["w"] == [0.0] * 203 and found["cell_total"] == 0
    assert dict(zip(*np.unique(found["points"]["owner"], return_counts=True)))[1] == 63, "still drawn in its outline"
    assert all(s["in_cell"] == 0 and s["word"] is None for s in found["gene_stats"]), "nothing held: no word"
    assert sum(found["points"]["near"]) == 3, "only the own genes inside B near A (positional)"


def test_over_a_box_holding_the_cells_pools_the_exact_w_add_up_to_its_count(make_runs):
    """The Σ w identity: transcripts of every gene drifted between A, B, D and no cell (a quarter of them anywhere),
    proseg's pools per bucket and destination exported whole: over the box, A's w add up to its averaged count (e from
    its counts file), gene by gene, and so do each source's."""
    rng = np.random.default_rng(11)
    n = 1500
    records = np.zeros(n, RECORD)
    records["x"], records["y"] = rng.uniform(101.0, 148.0, n), rng.uniform(52.0, 88.0, n)
    records["z"], records["gene"] = rng.uniform(0, 1, n), rng.integers(0, len(GENES), n)
    layers = np.stack([layout()] * 4)
    grid = {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 4}
    dest = destinations(records, layers, grid, (0.0, 0.25))
    drift = rng.random(n) < 0.25
    dest[drift] = rng.choice([0, 1, 2, 4], int(drift.sum()))
    keep = rng.uniform(0.3, 1.0, (len(CELLS), len(GENES)))
    runs, where = make_runs(r1=(layers, records, {"counts": None}))
    run = runs.get("r1")
    line = frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), keep, dest=dest)
    merge_counts(run)
    runs.exports.keep("r1", export_of(run, [100.0, 50.0, 150.0, 90.0], records, dest, token=2))
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=GENES, own=[])
    assert found["held"] == "exact" and found["box"] == [100.0, 50.0, 150.0, 90.0]
    from sopa_companion.analysis import counts_file
    table = counts_file.rows(counts_file.read(run.live / line["file"]), [0, 1, 3])["e"].astype(np.float64)
    assert found["cell_total"] == pytest.approx(table[0].sum(), abs=2e-3)
    assert [s["in_cell"] for s in found["gene_stats"]] == pytest.approx(table[0], abs=1e-3)
    assert sum(found["points"]["w"]) == pytest.approx(table[0].sum(), abs=1500 * 5e-4)    # the points' w: 3 decimals
    # Each source's own count gives its density: Σ of its shares per its footprint's area.
    for k, area in ((0, 100.0), (1, 60.0)):
        for g, s in enumerate(found["gene_stats"]):
            if s["giver"] == k:
                assert s["density"]["giver"] == pytest.approx(100 * table[1 + k][g] / area, abs=0.01)


def test_each_genes_giver_is_the_one_the_windows_answer_names(make_runs):
    runs, where = make_runs(r1=(layout()[None], placed(), {"tiled": True}))
    # B is cell 1, D cell 3; the answer names cell 7 (not a source here) for FEW and nothing for OWN1.
    found = reply(where, "r1", 0, 0, **ASK, givers={"EDGE": 1, "SPREAD": 1, "LAYER": 3, "FEW": 7, "OWN1": 1})
    stats = stats_by_gene(found)
    assert [(stats[g]["giver"], stats[g]["giver_by"]) for g in ("EDGE", "SPREAD", "LAYER", "FEW", "OWN1")] == \
        [(0, "result"), (0, "result"), (1, "result"), (None, None), (None, None)]
    assert stats["EDGE"]["word"] == "edge" and stats["EDGE"]["near_giver"] == 10
    assert stats["SPREAD"]["near_giver"] == 0 and stats["SPREAD"]["own_near"] == [2, 35]
    assert stats["SPREAD"]["density"]["giver"] == 8.0 and stats["SPREAD"]["word"] == "spread"
    assert (stats["LAYER"]["near_giver"], stats["LAYER"]["own_near"], stats["LAYER"]["word"]) == (0, [0, 35], "layer")
    assert stats["FEW"]["near_giver"] is None and stats["FEW"]["word"] == "few"
    # An answer without givers among the genes asked: none, never the densest source instead.
    found = reply(where, "r1", 0, 0, **ASK, givers={})
    assert all(s["giver"] is None and s["giver_by"] is None for s in found["gene_stats"])
    assert stats_by_gene(found)["EDGE"]["word"] is None                # no giver: no edge measured, no word


def test_the_edge_needs_twice_the_rest_of_the_cells_share_and_enough_transcripts(make_runs):
    """The rule for "edge": from >= 2.5 near the giver, at least a third of the gene's transcripts in the
    cell, and a near share >= 2 x that of the rest of the cell's RNA (OWN1 + OWN2 + OTHER: 35 in A), smoothed by one."""
    def run_with(own_near: int, edge_near: int):
        in_a = (placed()["x"] >= 110) & (placed()["x"] < 120) & (placed()["y"] >= 60) & (placed()["y"] < 70)
        records = [tuple(r) for r, a in zip(placed(), in_a) if not (a and GENES[r["gene"]] in ("EDGE", "OWN1"))]
        records += [(119.5, 60.5 + i, 0.5, 0) for i in range(edge_near)]                      # EDGE near B
        records += [(112.5, 60.5 + i, 0.5, 0) for i in range(10 - edge_near)]                 # ... and far from it
        records += [(118.5 + i % 2, 60.5 + i // 2, 0.5, 4) for i in range(own_near)]          # OWN1 near B
        records += [(110.5 + i % 2, 60.5 + i // 2, 0.5, 4) for i in range(20 - own_near)]     # ... and far from it
        return np.array(records, RECORD)
    # EDGE's share near B is edge_near / 10, the rest's (own_near + 1) / 37 smoothed: "edge" when edge_near >= 4 (2.5,
    # and a third of 10) and edge_near x 37 >= 2 x (own_near + 1) x 10. Three of ten with nothing else near is not
    # enough.
    for case, (own_near, edge_near, word) in enumerate([(2, 10, "edge"), (17, 10, "edge"), (18, 10, "spread"),
                                                        (20, 10, "spread"), (2, 2, "spread"), (0, 3, "spread"),
                                                        (0, 4, "edge"), (10, 6, "edge"), (11, 6, "spread")]):
        run_id = f"r{case}"
        runs, where = make_runs(**{run_id: (layout()[None], run_with(own_near, edge_near), {"tiled": True})})
        edge = stats_by_gene(reply(where, run_id, 0, 0, **ASK))["EDGE"]
        assert (edge["giver"], edge["in_cell"], edge["near_giver"]) == (0, 10, edge_near)
        assert edge["own_near"] == [own_near, 35]
        assert edge["word"] == word, (own_near, edge_near, edge)


def test_the_edge_needs_2_5_of_proseg_s_count_near_the_giver_a_sum_of_fractional_w(make_runs):
    """EDGE_NEAR on proseg's count (2.5: three transcripts' worth, 2026-10-31), where w is a fraction: A holds 17 EDGE,
    ten along B, and proseg counts e of them (its count file keeps e in steps of 1/32), each placed by the frame's layers
    at w = e / 17. At e 4.25 the ten sum to 2.5 near the giver, "edge"; one step less, e 4.21875, they sum to 2.48,
    "spread", and at e 4.0625 to 2.39. Every case passes the other rules (not few: e 4 or more; a third of the gene's
    count near the giver; twice the rest of the cell's near share), so the near sum alone decides."""
    from sopa_companion.engine import deeper_where as DW
    in_a = (placed()["x"] >= 110) & (placed()["x"] < 120) & (placed()["y"] >= 60) & (placed()["y"] < 70)
    records = [tuple(r) for r, a in zip(placed(), in_a) if not (a and GENES[r["gene"]] == "EDGE")]
    records += [(119.5, 60.5 + i, 0.5, 0) for i in range(10)]                         # EDGE near B
    records += [(112.5, 60.5 + i, 0.5, 0) for i in range(7)]                          # ... and far from it
    records = np.array(records, RECORD)
    for case, (e, word) in enumerate([(4.25, "edge"), (4.21875, "spread"), (4.0625, "spread")]):
        keep = np.ones((len(CELLS), len(GENES)))
        keep[0, GENES.index("EDGE")] = e / 17
        run_id = f"r{case}"
        runs, where = make_runs(**{run_id: (layout()[None], records, {"tiled": True, "counts": keep})})
        found = reply(where, run_id, 0, 0, **ASK)
        edge = stats_by_gene(found)["EDGE"]
        near, n, (rest_near, rest) = edge["near_giver"], edge["in_cell"], edge["own_near"]
        assert found["held"] == "approx" and edge["giver"] == 0 and (rest_near, rest) == (2, 35)
        assert (near, n, edge["count"]) == pytest.approx((10 * e / 17, e, e), abs=1e-3)
        assert min(n, edge["count"]) >= DW.FEW_IN_CELL and near >= DW.EDGE_SHARE * n
        assert near * (rest + 2) >= DW.EDGE_RATIO * (rest_near + 1) * n
        assert (near >= DW.EDGE_NEAR) == (word == "edge") and edge["word"] == word, (e, edge)


def test_without_own_genes_asked_the_rest_of_the_cell_is_the_baseline(make_runs):
    """The most mixed cells have no gene that is mostly their own (the window asks none): the edge is still judged,
    against every gene of the cell but EDGE (58 transcripts; FEW's 3 and OWN1's 2 lie near B)."""
    runs, where = make_runs(r1=(layout()[None], placed(), {"tiled": True}))
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=["EDGE"], own=[])
    edge = found["gene_stats"][0]
    assert edge["own_near"] == [5, 58] and edge["near_giver"] == 10 and edge["word"] == "edge"
    assert found["into"] == []


def test_a_points_near_flag_is_the_measurement_its_counts_are_made_of(make_runs):
    """Near is 1 for a "from" gene's transcript proseg counts for the cell (or inside its outline) within NEAR_UM of
    that gene's giver (whichever source the answer names), and for a source's transcript of an own gene within NEAR_UM
    of the cell; 0 for every other. Weighted by w and summed per gene, the flags are gene_stats' near_giver; per source
    and gene, into's near_cell (proseg's count for the cell of them)."""
    runs, where = make_runs(r1=(layout()[None], placed(), {"tiled": True}), r2=(np.stack([layout()] * 4), placed(), {}))
    for run_id, givers in (("r1", None), ("r1", {"EDGE": 3, "SPREAD": 1}),
                           ("r2", {"EDGE": 1, "SPREAD": 3, "LAYER": 1})):
        found = reply(where, run_id, 0, 0, **ASK, givers=givers)
        points, roles = found["points"], found["roles"]
        flags = list(zip(points["near"], points["owner"], points["gene"], points["w"]))
        for g, s in enumerate(found["gene_stats"]):
            assert sum(n * w for n, o, gene, w in flags if gene == g) == (s["near_giver"] or 0), (run_id, s)
        for entry in found["into"]:
            g = found["genes"].index(entry["gene"])
            assert round(sum(n * w for n, o, gene, w in flags if o == 2 + entry["source"] and gene == g), 3) == \
                entry["near_cell"]
        # Never near: the tissue around, the cell's own genes, a source's "from" genes.
        assert not any(n for n, o, gene, w in flags if o in (0, 255) or (o == 1) == (roles[gene] == "own"))
        assert sum(points["near"]) == (3 if givers == {"EDGE": 3, "SPREAD": 1} else 13)
    # EDGE named as D's (13.5 µm away): none of A's EDGE is near, though all of it hugs B.
    found = reply(where, "r1", 0, 0, **ASK, givers={"EDGE": 3})
    edge = found["genes"].index("EDGE")
    assert stats_by_gene(found)["EDGE"]["near_giver"] == 0
    assert not any(n for n, gene in zip(found["points"]["near"], found["points"]["gene"]) if gene == edge)


def test_near_is_measured_to_the_nearest_point_of_the_footprint():
    """Within 2 µm of a footprint's squares (exactly, corners included); 0 inside it; 1 and 2 µm columns."""
    from sopa_companion.engine.deeper_where import near
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True                                       # the square [5, 6) x [5, 6) (origin 0, 1 µm columns)
    px = np.array([5.5, 6.5, 8.0, 8.01, 7.5, 7.4, 3.0, 2.99, 5.5, 5.5])
    py = np.array([5.5, 5.5, 5.5, 5.5, 7.5, 7.4, 5.0, 5.0, 8.0, 3.0])
    assert near(px, py, mask, 0, 0, 0.0, 0.0, 1.0).tolist() == \
        [True, True, True, False, False, True, True, False, True, True]
    # The same footprint in a window that starts elsewhere, and 2 µm columns (the burn-in grid, a prior).
    moved = near(px + 10, py + 20, mask, 0, 0, 10.0, 20.0, 1.0)
    assert moved.tolist() == near(px, py, mask, 0, 0, 0.0, 0.0, 1.0).tolist()
    big = np.zeros((5, 5), bool)
    big[2, 2] = True                                        # [4, 6) x [4, 6) in µm
    assert near(np.array([7.9, 8.1, 3.0, 5.0]), np.array([5.0, 5.0, 3.0, 8.0]), big, 0, 0, 0.0, 0.0, 2.0).tolist() == \
        [True, False, True, True]
    assert near(np.zeros(0), np.zeros(0), big, 0, 0, 0.0, 0.0, 2.0).tolist() == []


def test_owners_follow_the_layers_in_detail_runs(make_runs):
    """Detail runs place a transcript in the cell owning its voxel (worker voxel_lookup: z clipped into the layers):
    B holds A's two right columns in the top layer, and A one voxel far away in the bottom layer only. The box holds
    every voxel; distances go to the columns where a cell owns any layer; areas are voxels x v² / layers."""
    layers = np.stack([layout()] * 4)
    layers[3, 10:20, 18:20] = 2
    layers[0, 35, 45] = 1
    layers[:, 30, 50] = 9                                    # a label past the run's 5 cells: no cell (the worker's)
    edge, own, other = GENES.index("EDGE"), GENES.index("OWN1"), GENES.index("OTHER")
    records = np.array([(118.5, 65.5, 0.9, edge), (118.5, 65.5, 0.1, edge), (145.5, 85.5, 0.1, own),
                        (145.5, 85.5, 0.6, own), (117.5, 64.5, -0.4, edge), (117.5, 64.5, 1.6, edge),
                        (111.5, 62.5, 0.3, edge), (125.5, 62.5, 0.3, other), (150.5, 80.5, 0.5, edge)], RECORD)
    runs, where = make_runs(r1=(layers, records, {}))
    found = reply(where, "r1", 0, 0, sources=[1, 3], genes=["EDGE"], own=["OWN1"])
    assert found["basis"] == "voxel" and found["box"] == [100.0, 50.0, 160.0, 90.0]
    assert found["points"]["owner"] == [2, 1, 1, 0, 1, 1, 1, 0]
    assert found["points"]["w"] == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    # Near B's footprint (its top layer's columns 18-19 included): A's EDGE in column 18 and the two 0.5 µm from it.
    assert found["points"]["near"] == [0, 1, 0, 0, 1, 1, 0, 0]
    assert found["cell_total"] == 5
    assert found["cell_area_um2"] == 95.25 and found["source_area_um2"] == [105.0, 60.0]   # 381, 420, 240 voxels
    edge_stats = found["gene_stats"][0]
    # B's footprint takes in columns 18-19 (its top layer): A's EDGE at 118.5 and 117.5 are near it, 111.5 is not.
    assert (edge_stats["giver"], edge_stats["in_cell"], edge_stats["near_giver"]) == (0, 4, 3)
    # 4 in the cell is not too few to tell (FEW_IN_CELL 4: what 5 transcripts are worth in proseg's count), and 3 of
    # them near B, far more than the rest of its RNA, is B's edge.
    assert edge_stats["own_near"] == [0, 1] and edge_stats["word"] == "edge"


@pytest.mark.parametrize("tiled", [True, False])
def test_the_start_line_places_on_the_prior_and_shares_by_the_first_frame(make_runs, tiled):
    """seq "start": the transcripts placed by the prior's labels (tiled: prior.tiles' 2-D label plane; dense:
    prior.bin.zst's layers) on their own 2 µm grid; proseg's share from its first frame (its layers and counts:
    proseg's count at iteration 0), approximate; iteration 0."""
    prior = np.zeros((2, 20, 30), np.uint32)                 # 2 µm: x 100-160, y 50-90
    prior[:, 5:10, 5:10] = 1                                  # A: x 110-120, y 60-70
    prior[:, 5:10, 10:15] = 2                                 # B: x 120-130
    prior[1, 5:10, 10] = 1                                    # dense: A takes B's first column in the upper layer
    edge = GENES.index("EDGE")
    records = np.array([(120.5, 62.5, 0.1, edge), (120.5, 62.5, 0.7, edge), (115.0, 65.0, 0.9, edge),
                        (125.0, 65.0, 0.9, edge), (70.0, 60.0, 0.5, edge)], RECORD)
    # The first frame, on its 1 µm grid: the prior's cells as they start (a tiled run's 2-D, in both layers).
    start = np.repeat(np.repeat(prior if not tiled else np.stack([prior[0]] * 2), 2, axis=1), 2, axis=2)
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, edge] = 0.5
    spec = {"plane": prior[0], "voxel": 2.0} if tiled else {"layers": prior, "voxel": 2.0}
    runs, where = make_runs(r1=(start, records, {"tiled": tiled, "prior": spec, "tile": 8, "counts": keep}))
    found = reply(where, "r1", "start", 0, sources=[1], genes=["EDGE"])
    assert (found["seq"], found["iter"], found["basis"]) == ("start", 0, "start")
    assert (found["held"], found["held_seq"]) == ("approx", None)
    assert found["box"] == [100.0, 50.0, 150.0, 90.0]        # A and B (x 110-130, y 60-70) + 20, clipped
    # Layers of 0.5 (z grid_spacing): z 0.1 is layer 0 (B's), z 0.7 layer 1 (A's in the dense prior; the tiled prior is
    # 2-D: B's); the transcript at x 70 lies outside the grid and the box.
    assert found["points"]["owner"] == [2, 2 if tiled else 1, 1, 2]
    assert found["points"]["w"] == [0.0, 0.0 if tiled else 0.5, 0.5, 0.0]
    assert found["cell_total"] == (0.5 if tiled else 1.0)
    assert found["cell_area_um2"] == (100.0 if tiled else 110.0)      # 25 columns x 4 µm²; dense (25 + 30) / 2 x 4
    # No first frame counted: nothing held at the start line.
    run = runs.get("r1")
    with run.lock:
        frame = run.frames[0]
        for key in ("counts_file", "counted_file", "counts"):
            frame.pop(key)
    where.replies.clear()
    found = reply(where, "r1", "start", 0, sources=[1], genes=["EDGE"])
    assert found["held"] == "none" and found["points"]["w"] == [0.0] * 4
    assert found["points"]["owner"] == [2, 2 if tiled else 1, 1, 2]


def test_a_frame_whose_cells_own_nothing_has_no_box(make_runs):
    frame = np.zeros((2, NY, NX), np.uint32)
    runs, where = make_runs(r1=(frame, placed(), {"tiled": False}))
    found = reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])
    assert found["box"] is None and found["held"] == "none" and found["points"]["w"] == []


def test_points_are_cut_at_their_limit_proseg_s_count_of_the_cell_first(make_runs, monkeypatch):
    """Past POINTS_MAX: the points proseg counts for the cell (w > 0) first, then the cell's and its sources' by their
    outlines, then the tissue around, each in the box's order (the run's, here shuffled), and the points kept stay in
    that order."""
    from sopa_companion.engine import deeper_where
    records = placed()[np.random.default_rng(4).permutation(len(placed()))]
    runs, where = make_runs(r1=(layout()[None], records, {"tiled": True}))
    whole = reply(where, "r1", 0, 0, **ASK)
    w, owner = whole["points"]["w"], whole["points"]["owner"]
    counted = [i for i, v in enumerate(w) if v > 0]
    ours = [i for i, o in enumerate(owner) if o in (2, 3)]
    rest = [i for i, o in enumerate(owner) if o in (0, 255)]
    assert (len(counted), len(ours), len(rest)) == (63, 40, 100) and min(ours) < max(counted[:40])
    assert all(owner[i] == 1 for i in counted)
    for limit, kept in ((40, counted[:40]), (90, sorted(counted + ours[:27])),
                        (120, sorted(counted + ours + rest[:17]))):
        monkeypatch.setattr(deeper_where, "POINTS_MAX", limit)
        where.replies.clear()
        cut = reply(where, "r1", 0, 0, **ASK)
        assert cut["truncated"] is True and len(cut["points"]["x"]) == limit
        assert {k: v for k, v in cut["points"].items()} == {k: [v[i] for i in kept] for k, v in whole["points"].items()}
        assert cut["gene_stats"] == whole["gene_stats"] and cut["into"] == whole["into"]   # counted on every one
        assert cut["cell_total"] == whole["cell_total"]


@pytest.mark.parametrize("grow", [(1, 4), (1,)])
def test_a_cell_found_away_from_its_centre_and_without_cell_statistics(make_runs, monkeypatch, grow):
    """Tiled frames: a cell is looked for around its centre and taken once the window holds as many columns as its
    statistics count. A far piece (34 columns past the first window) is found by the wider window, or, past that, by
    scanning every tile; and so is every cell of a frame without cell statistics."""
    from sopa_companion.engine import deeper_where
    monkeypatch.setattr(deeper_where, "SEED_HALF", 5)        # A's first window: columns 0-47 (half 22: its size)
    monkeypatch.setattr(deeper_where, "SEED_GROW", grow)
    plane = layout()
    plane[2:4, 56:58] = 1                                     # A's far piece: x 156-158, y 52-54
    records = np.concatenate([placed(), np.array([(156.5, 52.5, 0.5, GENES.index("EDGE"))], RECORD)])
    runs, where = make_runs(r1=(plane[None], records, {"tiled": True}),
                            r2=(plane[None], records, {"tiled": True, "stats": False}))
    scans = []
    real = deeper_where._tiles
    monkeypatch.setattr(deeper_where, "_tiles", lambda *a: scans.append(1) or real(*a))
    first = reply(where, "r1", 0, 0, **ASK)
    assert bool(scans) == (grow == (1,))                      # found by the wider window, or by the scan
    second = reply(where, "r2", 0, 0, **ASK)
    for found in (first, second):
        assert found["box"] == [100.0, 50.0, 160.0, 90.0] and found["cell_total"] == 69
        assert found["cell_area_um2"] == 104.0 and found["source_area_um2"] == [100.0, 60.0]
    assert first["gene_stats"] == second["gene_stats"] and first["points"] == second["points"]
    monkeypatch.setattr(deeper_where, "SEED_HALF", 40)      # the default: the whole of this small raster at once
    scans.clear()
    assert reply(deeper_where.DeeperWhere(runs), "r1", 0, 0, **ASK)["cell_total"] == 69 and not scans


@pytest.mark.parametrize("tiled", [True, False])
def test_footprints_are_the_boxes_a_reply_starts_from(make_runs, tiled):
    """The test's corner view centres on the cell's footprint (engine/qc_test.py): its columns' box in µm (in a detail
    run the columns where it owns any layer), None for a cell that owns nothing; a reply's refusals."""
    from sopa_companion.engine.runs import RunError
    layers = layout()[None] if tiled else np.stack([layout()] * 4)
    if not tiled:
        layers[3, 30, 50] = 1                                  # A owns one far voxel in its top layer only
    runs, where = make_runs(r1=(layers, placed(), {"tiled": tiled}))
    assert where.footprints("r1", 0, [0, 1, 2, 4]) == {
        0: [110.0, 60.0, 120.0, 70.0] if tiled else [110.0, 60.0, 151.0, 81.0], 1: [120.0, 60.0, 130.0, 70.0],
        2: [110.0, 74.0, 120.0, 84.0], 4: None}
    for args, status in ((("r1", 1, [0]), 404), (("r1", 0, [5]), 404), (("nope", 0, [0]), 404)):
        with pytest.raises(RunError) as refused:
            where.footprints(*args)
        assert refused.value.status == status


def test_cells_with_no_footprint(make_runs):
    plane = layout()
    plane[plane == 1] = 0                                    # A vanished from this frame
    runs, where = make_runs(r1=(plane[None], placed(), {"tiled": True}))
    found = reply(where, "r1", 0, 0, **ASK)
    assert found["box"] == [100.0, 50.0, 150.0, 90.0] and found["cell_total"] == 0 and found["cell_area_um2"] == 0
    assert all(s["in_cell"] == 0 and s["word"] == "few" and s["density"]["cell"] is None for s in found["gene_stats"])
    nothing = reply(where, "r1", 0, 0, sources=[], genes=["EDGE"], own=["OWN1"])
    assert nothing["box"] is None and nothing["held"] == "none"
    assert nothing["points"] == {"x": [], "y": [], "z": [], "gene": [], "owner": [], "near": [], "w": []}
    assert nothing["cell_total"] == 0 and nothing["gene_stats"][0]["in_cell"] == 0 and nothing["into"] == []
    assert nothing["gene_stats"][0]["word"] is None, "nothing held: no word"


def test_a_reply_without_genes_still_counts_the_cell(make_runs):
    """No gene asked (the test's corner view asks for the cell's count alone): the box and the cell's count, nothing
    per gene, no points (this once raised IndexError on the empty gene rows)."""
    runs, where = make_runs(r1=(layout()[None], placed(), {"tiled": True}), r2=(np.stack([layout()] * 4), placed(), {}))
    for run_id in ("r1", "r2"):
        found = reply(where, run_id, 0, 0, sources=[1, 3])
        assert (found["genes"], found["gene_stats"], found["into"], found["box"]) == ([], [], [], [100.0, 50.0, 150.0, 90.0])
        assert found["cell_total"] == 68
        assert found["points"] == dict.fromkeys(("x", "y", "z", "gene", "owner", "near", "w"), [])


def test_replies_are_kept_and_the_frames_labels_too(make_runs, monkeypatch):
    from sopa_companion.engine import deeper_where
    decoded = []
    real_decode = deeper_where._decode
    monkeypatch.setattr(deeper_where, "_decode", lambda path: decoded.append(path.name) or real_decode(path))
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    first = where.reply("r1", 0, 0, [1, 3], ["EDGE"], ["OWN1"])
    calls = []
    real = where._work
    monkeypatch.setattr(where, "_work", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert where.reply("r1", 0, 0, [1, 3], ["EDGE"], ["OWN1"]) is first and calls == []
    other = json.loads(where.reply("r1", 0, 2, [], ["EDGE"], []))        # another cell: the frame decoded once
    assert calls == [1] and decoded == ["000000.bin.zst"] and other["cell"] == 2
    named = json.loads(where.reply("r1", 0, 0, [1, 3], ["EDGE"], ["OWN1"], givers={"EDGE": 3}))
    assert calls == [1, 1] and named["gene_stats"][0]["giver"] == 1      # another answer: worked out again


def test_replies_are_kept_by_where_their_shares_came_from(make_runs, monkeypatch):
    """An approximate reply is kept until proseg's export of the box is there; then the exact one is worked out and
    kept for good (the frame's snapshot never changes, so it stands even once the export has gone)."""
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    approx = reply(where, "r1", 0, 0, **ASK)
    calls = []
    real = where._work
    monkeypatch.setattr(where, "_work", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert reply(where, "r1", 0, 0, **ASK) == approx and calls == []
    dest = destinations(placed(), np.stack([layout()] * 4), {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY,
                                                             "nz": 4}, (0.0, 0.25))
    meta = runs.exports.keep("r1", export_of(run, approx["box"], placed(), dest, token=3))
    exact = reply(where, "r1", 0, 0, **ASK)
    assert calls == [1] and (exact["held"], exact["held_seq"]) == ("exact", 0)
    assert exact["points"]["w"] == approx["points"]["w"]      # no drift here: each pool is its voxel's owner's
    Path(meta["path"]).unlink()
    assert reply(where, "r1", 0, 0, **ASK) == exact and calls == [1], "kept for good"
    # An export whose file is gone, or one that does not hold the box, is not a reply's: approximate.
    assert reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])["held"] == "approx"
    runs.exports.keep("r1", export_of(run, [110.0, 60.0, 130.0, 70.0], placed(), dest, token=4))
    assert reply(where, "r1", 0, 0, sources=[3], genes=["EDGE"])["held"] == "approx" and calls == [1, 1, 1]


def test_paused_at_the_frame_proseg_is_asked_to_export_the_box(make_runs):
    """While the run is paused (or held) at the frame asked, proseg is asked for an export of the reply's box (its
    paused state serves it at once: within EXPORT_WAIT_S), and the reply is exact; an answer of another frame, or one
    that does not hold the box, is not taken; nothing is asked of a frame the run is not paused at."""
    from sopa_companion.engine import deeper_where
    runs, where = make_runs(r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    dest = destinations(placed(), np.stack([layout()] * 4), {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY,
                                                             "nz": 4}, (0.0, 0.25))
    assert reply(where, "r1", 0, 0, **ASK)["held"] == "approx" and runs.exports.asked == []
    with run.lock:
        run.ack = {"state": "paused", "iter": 40, "seq": 0}
    where.replies.clear()
    answers = iter([export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=5, seq=0)])
    runs.exports.serve = lambda run_id, box: next(answers, None)
    found = reply(where, "r1", 0, 0, **ASK)
    assert runs.exports.asked == [("r1", [100.0, 50.0, 150.0, 90.0], deeper_where.EXPORT_WAIT_S)]
    assert (found["held"], found["held_seq"]) == ("exact", 0)
    # Answers that do not fit: of another frame, or of a smaller box.
    for answer in (export_of(run, [100.0, 50.0, 150.0, 90.0], placed(), dest, token=6, seq=0) | {"seq": 3},
                   export_of(run, [110.0, 60.0, 130.0, 70.0], placed(), dest, token=7)):
        runs.exports.serve = lambda run_id, box, answer=answer: answer
        where.replies.clear()
        assert reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])["held"] == "approx"
    with run.lock:
        run.ack = {"state": "paused", "iter": 50, "seq": 1}            # paused at another frame: nothing asked
    asked = len(runs.exports.asked)
    where.replies.clear()
    assert reply(where, "r1", 0, 0, sources=[3], genes=["EDGE"])["held"] == "approx"
    assert len(runs.exports.asked) == asked


def test_a_playing_run_exports_the_box_the_window_follows(make_runs):
    """While the run plays, each reply hands the analysis scheduler its box (FOLLOW_PAD_UM wider) to have proseg export
    with each counted frame (engine/analysis.py follow_where). Paused or held (D.1), a reply of the frame proseg rests
    on arms its box all the same and keeps its cell as the one followed, so the first counted frame after Resume carries
    the box's export; a reply of another frame then arms nothing new; done, none."""
    from sopa_companion.engine import deeper_where

    class Scheduler:
        def __init__(self):
            self.boxes = []

        def follow_where(self, run_id, box):
            self.boxes.append((run_id, box))
    scheduler = Scheduler()
    runs, where = make_runs(analysis=scheduler, r1=(np.stack([layout()] * 4), placed(), {}))
    run = runs.get("r1")
    run.meta["state"] = "running"
    found = reply(where, "r1", 0, 0, **ASK)
    pad = deeper_where.FOLLOW_PAD_UM
    grown = lambda box: [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]    # noqa: E731
    assert scheduler.boxes == [("r1", grown([100.0, 50.0, 150.0, 90.0]))]
    assert found["box"] == [100.0, 50.0, 150.0, 90.0] and where.followed["r1"] == (0, 0)
    with run.lock:
        run.ack = {"state": "paused", "iter": 40, "seq": 0}             # held at the end of its schedule
    reply(where, "r1", 0, 0, **ASK)
    assert scheduler.boxes[-1] == ("r1", grown([100.0, 50.0, 150.0, 90.0])) and where.followed["r1"] == (0, 0)
    run.meta["state"] = "paused"
    c = reply(where, "r1", 0, 2, genes=["EDGE"])                        # another cell of the frame it rests on
    assert scheduler.boxes[-1] == ("r1", grown(c["box"])) and where.followed["r1"] == (2, 0)
    add_frame(run, np.stack([layout()] * 4), 50)
    calls = len(scheduler.boxes)
    reply(where, "r1", 1, 0, **ASK)                                     # another frame while paused: nothing new
    assert len(scheduler.boxes) == calls and where.followed["r1"] == (2, 0)
    where.measure("r1", 0, 0, [1], ["EDGE"], [])                        # the test's measurement follows nothing
    assert len(scheduler.boxes) == calls
    run.meta["state"] = "done"
    with run.lock:
        run.ack = {"state": "done", "iter": 50, "seq": 1}
    reply(where, "r1", 1, 2, genes=["EDGE"])
    assert scheduler.boxes[-1] == ("r1", None) and "r1" not in where.followed
    where.analysis = object()                                           # a scheduler that cannot: nothing asked
    assert reply(where, "r1", 0, 0, **ASK)["box"] == [100.0, 50.0, 150.0, 90.0]


def add_frame(run, layers, iteration, keep=1.0):
    """The next frame of a detail run as proseg-live appends it: its labels file (the consensus its first layer), its
    record, and proseg's count of it (each transcript to the label of its observed voxel, x keep), merged."""
    import zstandard
    seq = len(run.frames)
    nz, ny, nx = layers.shape
    with run.lock:
        first = dict(run.frames[0])
    record = {**{key: first[key] for key in ("phase", "nx", "ny", "nz", "x0", "y0", "voxel", "cells")}, "seq": seq,
              "iter": iteration, "file": f"frames/{seq:06d}.bin.zst"}
    raw = np.concatenate([layers[0].ravel(), np.zeros(nx * ny), layers.ravel()]).astype("<u4")
    (run.live / record["file"]).write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.tobytes()))
    with run.lock:
        run.frames.append(record)
    records = np.frombuffer(gzip.decompress((run.live / "transcripts.bin.gz").read_bytes()), RECORD)
    grid = {key: first[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
    z = run.info["z"]
    frame_counts(run.live, seq, iteration, layers, records, grid, (z["grid_origin"], z["grid_spacing"]),
                 int(run.info["cells"]), int(run.info["genes"]), keep)
    merge_counts(run)
    return record


def test_only_the_live_cells_newest_reply_moves_the_box_a_playing_run_exports(make_runs):
    """While the run plays, the box its counted frames export is the live cell's: the start line's reply never hands
    its box over (the window asks for it beside its live replies, and the starting footprints lie elsewhere), nor does
    an older frame's reply of the same cell (a view scrubbed back); a newer frame's, another cell's, a kept exact
    reply's and a restarted run's do."""
    from sopa_companion.engine import deeper_where

    class Scheduler:
        def __init__(self):
            self.boxes = []

        def follow_where(self, run_id, box):
            self.boxes.append((run_id, box))
    pad = deeper_where.FOLLOW_PAD_UM
    grown = lambda box: [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]    # noqa: E731
    prior = np.zeros((2, 20, 30), np.uint32)                 # 2 µm: x 100-160, y 50-90
    prior[:, 5:10, 5:10], prior[:, 5:10, 10:15] = 1, 2       # A started at x 110-120, B at x 120-130
    edge = GENES.index("EDGE")
    records = np.array([(115.0, 65.0, 0.5, edge), (125.0, 65.0, 0.5, edge), (136.0, 65.0, 0.5, edge)], RECORD)
    now = np.zeros((2, 40, 60), np.uint32)                   # frame 0: A at x 130-140, B at x 140-150
    now[:, 10:20, 30:40], now[:, 10:20, 40:50] = 1, 2
    scheduler = Scheduler()
    runs, where = make_runs(analysis=scheduler, r1=(now, records, {"prior": {"layers": prior, "voxel": 2.0}}))
    run = runs.get("r1")
    run.meta["state"] = "running"
    live = reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"])
    start = reply(where, "r1", "start", 0, sources=[1], genes=["EDGE"])
    assert live["box"] == [110.0, 50.0, 160.0, 90.0] and start["box"] == [100.0, 50.0, 150.0, 90.0]
    assert scheduler.boxes == [("r1", grown(live["box"]))], "the start line's box is never exported"
    later = np.zeros_like(now)                               # frame 1: A at x 135-145, B at x 145-155
    later[:, 10:20, 35:45], later[:, 10:20, 45:55] = 1, 2
    add_frame(run, later, 50)
    newer = reply(where, "r1", 1, 0, sources=[1], genes=["EDGE"])
    assert newer["box"] == [115.0, 50.0, 160.0, 90.0] and scheduler.boxes[-1] == ("r1", grown(newer["box"]))
    calls = len(scheduler.boxes)
    reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"], own=["OWN1"])      # A scrubbed back to frame 0
    assert len(scheduler.boxes) == calls, "an older frame of the same cell leaves the box as it is"
    other = reply(where, "r1", 0, 1, sources=[0], genes=["EDGE"])            # B at frame 0: another cell
    assert scheduler.boxes[-1] == ("r1", grown(other["box"]))
    # A kept exact reply hands its box over again (A at frame 1, after B's).
    dest = destinations(records, later, {"x0": X0, "y0": Y0, "voxel": 1.0, "nx": NX, "ny": NY, "nz": 2}, (0.0, 0.5))
    runs.exports.keep("r1", export_of(run, newer["box"], records, dest, token=2, seq=1, iteration=50))
    assert reply(where, "r1", 1, 0, sources=[1], genes=["EDGE"])["held"] == "exact"
    exact = where.reply("r1", 1, 0, [1], ["EDGE"], [])
    assert json.loads(exact)["held"] == "exact" and where.reply("r1", 1, 0, [1], ["EDGE"], []) is exact
    reply(where, "r1", 0, 1, sources=[0], genes=["EDGE"])                    # B again, then A's kept exact reply
    where.reply("r1", 1, 0, [1], ["EDGE"], [])
    assert scheduler.boxes[-1] == ("r1", grown(newer["box"]))
    # A restarted run begins with frame 0 again: its reply hands its box over.
    with run.lock:
        del run.frames[1:]
    reply(where, "r1", 0, 0, sources=[1], genes=["EDGE"], own=["OWN2"])
    assert scheduler.boxes[-1] == ("r1", grown(live["box"]))


@pytest.mark.parametrize("kind", ["detail", "tiled", "start"])
def test_measure_is_the_reply_with_each_points_record(make_runs, monkeypatch, kind):
    """measure() (the test's targets, engine/qc_pool.py): the reply's object, its bytes reply()'s but "ms", and the
    run's record of each point in the points' order (x, y and z round to the points'; its gene is the point's), also
    when the points are cut at their limit; worked out afresh, never kept among the replies; reply()'s refusals."""
    from sopa_companion.engine import deeper_where
    from sopa_companion.engine.runs import RunError
    prior = np.zeros((2, 20, 30), np.uint32)
    prior[:, 5:10, 5:10], prior[:, 5:10, 10:15], prior[:, 5:10, 0:3] = 1, 2, 4
    layers = layout()[None] if kind == "tiled" else np.stack([layout()] * 4)
    options = {"tiled": kind == "tiled", "prior": {"layers": prior, "voxel": 2.0}}
    records = placed()[np.random.default_rng(3).permutation(len(placed()))]
    runs, where = make_runs(r1=(layers, records, options))
    seq = "start" if kind == "start" else 0
    asked = {"sources": [1, 3], "genes": ["EDGE", "SPREAD", "LAYER", "FEW"], "own": ["OWN1", "OWN2"],
             "givers": {"EDGE": 1, "SPREAD": 3}}
    for limit in (deeper_where.POINTS_MAX, 40):
        monkeypatch.setattr(deeper_where, "POINTS_MAX", limit)
        where.replies.clear()
        where.kept_bytes = 0
        out, held = where.measure("r1", seq, 0, **asked)
        assert where.replies == {} and where.kept_bytes == 0           # never kept
        data = where.reply("r1", seq, 0, **asked)
        shown = json.loads(data)
        assert "ms" not in out and {k: v for k, v in shown.items() if k != "ms"} == out
        assert data == json.dumps({**out, "ms": shown["ms"]}, separators=(",", ":"), allow_nan=False).encode()
        points = out["points"]
        assert held.dtype == RECORD and len(held) == len(points["x"]) > 0
        assert out["truncated"] == (limit == 40) and (limit != 40 or len(held) == 40)
        assert [round(float(v), 2) for v in held["x"]] == points["x"]
        assert [round(float(v), 2) for v in held["y"]] == points["y"]
        assert [round(float(v), 3) for v in held["z"]] == points["z"]
        assert [GENES[g] for g in held["gene"]] == [out["genes"][g] for g in points["gene"]]
        assert set(map(tuple, held.tolist())) <= set(map(tuple, records.tolist()))
    for args, status in ((("nope", seq, 0), 404), (("r1", seq, 9), 404), (("r1", 7, 0), 404)):
        with pytest.raises(RunError) as refused:
            where.measure(*args, sources=[1], genes=["EDGE"], own=[])
        assert refused.value.status == status
    monkeypatch.setattr(deeper_where, "BOX_MAX_UM", 10.0)
    with pytest.raises(RunError) as refused:
        where.measure("r1", seq, 0, **asked)
    assert refused.value.status == 422


def test_parameters_are_checked():
    from sopa_companion.engine.deeper_where import parse
    from sopa_companion.engine.runs import RunError
    assert parse("12", "3", "4, 5", "A,B, ,C", "D") == (12, 3, [4, 5], ["A", "B", "C"], ["D"])
    assert parse(" start", "0", "", "", "") == ("start", 0, [], [], [])
    for args in (("x", "1", "", "", ""), ("-1", "1", "", "", ""), ("1.5", "1", "", "", ""), (None, "1", "", "", ""),
                 ("1", "", "", "", ""), ("1", "-2", "", "", ""), ("1", "1", "2,x", "", ""), ("1", "1", "2,2", "", ""),
                 ("1", "1", "1,2", "", ""), ("1", "1", ",".join(map(str, range(2, 13))), "", ""),
                 ("1", "1", "", ",".join(f"G{i}" for i in range(40)), ",".join(f"H{i}" for i in range(25))),
                 ("²", "1", "", "", ""), ("1" * 12, "1", "", "", "")):
        with pytest.raises(RunError) as refused:
            parse(*args)
        assert refused.value.status == 422, args


# ---- a cell whose pink dots lie far from the neighbour that would be named (the strip case, A2) --------------
F_GENES = ["GZF", "OWN1", "OWN2"]
F_CELLS = ["cell-a", "cell-f", "cell-b"]


def strip_case(adjacent: bool = False):
    """A case seen on a real sample, made small: cell-a (A, label 1: x 130-145, y 65-80) holds a strip of 13
    transcripts of GZF, a gene of another kind, along its left edge (x 130.5); the only outlined cell of that kind,
    cell-f (F, label 2, all GZF), lies on the far side, 15 µm past A's right edge (x 160-170: about 30 µm from
    the strip); cell-b (B, label 3, A's own kind) lies left of A. `adjacent`: F right beside A instead, the strip
    along A's right edge (x 144.5), within 2 µm of F. Returns (layers [4, 50, 80], records)."""
    plane = np.zeros((50, 80), np.uint32)
    plane[15:30, 30:45] = 1                                   # A
    plane[15:30, 45:55 if adjacent else 60:70] = 2           # F
    plane[15:30, 18:30] = 3                                   # B
    strip_x = 144.5 if adjacent else 130.5
    rows = [(strip_x, 65.5 + i, 0.5, 0) for i in range(13)]                          # the strip: A's GZF
    fx = (146.0, 154.0) if adjacent else (161.0, 169.0)
    rows += [(fx[0] + (fx[1] - fx[0]) * (i % 9) / 8, 66.0 + i // 9, 0.5, 0) for i in range(36)]   # F's own GZF
    rows += [(133.5 + i % 10, 67.5 + i // 10, 0.5, 1) for i in range(60)]           # A's own genes
    rows += [(134.5 + i % 8, 72.5 + i // 8, 0.5, 2) for i in range(24)]
    rows += [(119.5 + i % 9, 66.5 + i // 9, 0.5, 1) for i in range(45)]             # B's (A's kind)
    return np.stack([plane] * 4), np.array(rows, RECORD)


def strip_case_answer(seq=0, cell=0) -> dict:
    """Go deeper's answer for cell-a (a ready v 1 result): GZF, 12 of its 13 from neighbours, given by
    cell-f (row 0: the only cell of its kind among the neighbours), B the same kind as A."""
    return {"v": 1, "run": "r1", "seq": seq, "iter": 40, "model": "m", "cell": cell, "state": "ready",
            "transcripts": 97, "outside": 0, "code": 40, "parts": [0.82, 0.16, 0.02], "rough": False,
            "genes": [["GZF", 13, 1.0, 12.0, 1, 0], ["OWN1", 60, 59.0, 1.0, 0], ["OWN2", 24, 23.5, 0.5, 0]],
            "genes_total": 3, "kind": ["OWN1", "OWN2"], "own_genes": [["OWN1", 59.0, 60], ["OWN2", 23.5, 24]],
            "same_kind_share": 0.1,
            "neighbours": [{"cell": 1, "name": F_CELLS[1], "um": 27.5, "tx": 12.0, "genes": [["GZF", 12.0]],
                            "kind": ["GZF"], "similar": 0.1, "same": False, "known": True},
                           {"cell": 2, "name": F_CELLS[2], "um": 13.5, "tx": 2.0, "genes": [["OWN1", 1.0]],
                            "kind": ["OWN1", "OWN2"], "similar": 0.95, "same": True, "known": True}],
            "learned": "yes", "ms": 1.25}


@pytest.mark.parametrize("adjacent", [False, True])
def test_pink_dots_far_from_the_named_neighbour_are_near_none_of_it(make_runs, adjacent):
    """The where engine on the strip case: A's 13 GZF are proseg's (w 0.875 each: it keeps most of them for A),
    their giver is cell-f, and none lies within 2 µm of it (it is 30 µm away): their near flags are 0, and so the
    page's scoring (engine/qc_scoring.py, the page's twin) gives them glow but cell-f no support: the window says
    they likely come from a GZF cell with no outline here, never that neighbour. With cell-f right beside the
    strip, the flags are set and it is their source as before."""
    from sopa_companion.engine import qc_scoring as S
    layers, records = strip_case(adjacent)
    keep = np.ones((3, 3))
    keep[0, 0] = 0.875
    runs, where = make_runs(r1=(layers, records, {"counts": keep, "genes": F_GENES, "cells": F_CELLS}))
    result = strip_case_answer()
    params = S.page_params(result, 0)
    q = params["req"]["params"]
    assert (q["sources"], q["genes"], q["own"]) == ([1, 2], ["GZF"], ["OWN1", "OWN2"])
    found = reply(where, "r1", 0, 0, q["sources"], q["genes"], q["own"], givers={"GZF": 1})
    strip = [i for i, (g, o) in enumerate(zip(found["points"]["gene"], found["points"]["owner"])) if g == 0 and o == 1]
    assert len(strip) == 13 and all(found["points"]["w"][i] == 0.875 for i in strip)
    gzf = stats_by_gene(found)["GZF"]
    assert (gzf["giver"], gzf["giver_by"], gzf["in_cell"]) == (0, "result", 11.375)
    assert gzf["near_giver"] == (11.375 if adjacent else 0)
    assert [found["points"]["near"][i] for i in strip] == [int(adjacent)] * 13
    scores = S.page_scores(result, params["req"], found)
    glowing = [p for p in scores["points"] if p["gene"] == "GZF"]
    assert len(glowing) == 13 and all(p["owner"] == 1 and p["P"] >= S.GLOW_MIN for p in glowing)
    assert all(p["pi"] >= S.LIKELY_MIN and p["w"] >= S.LIKELY_W for p in glowing), "likely: proseg keeps most for A"
    assert scores["support"] == ([[0, 13 * 0.875]] if adjacent else [])


# ---- the invariant: proseg's count -----------------------------------------------------------------------------------
@pytest.mark.parametrize("tiled", [False, True])
def test_the_cells_exact_count_is_go_deepers_transcripts_plus_outside(config, tmp_path, tiled):
    """For every cell of a frame, with proseg's export of its box (every transcript's pool its voxel's owner: dense
    layers, or a whole-sample frame's columns) and its count of the frame (keep 1), the cell's w here add up to the
    worker's Go deeper result's `transcripts + outside` (the worker's own numpy code, test_worker_deeper's model), and
    each gene's to its row."""
    from sopa_companion.analysis import worker as W
    from sopa_companion.engine.deeper_where import DeeperWhere
    from sopa_companion.engine.runs import Runs
    from test_analysis import MODEL_GENES, NCELLS, RUN_GENES, frame_plane, write_run as write_worker_run
    from test_worker_deeper import JUDGE, K, RUN_IDS, make_deep
    data = Path(config["paths"]["data_dir"])
    folder = write_worker_run(data / "runs" / "w1", tiled=tiled)
    live = folder / "live"
    (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    (folder / "run.json").write_text(json.dumps({"id": "w1", "kind": "proseg", "state": "done", "created": "x",
                                                 "prior": {"cells": {"kind": "shapes"}, "nuclei": None}}))
    model = make_deep(tmp_path / "model", folder)
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    grid = {key: frame[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
    labels = np.stack([frame_plane(2)] * frame["nz"])                    # the worker's: the voxel's, or its column's
    records = np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORD)
    dest = destinations(records, labels, grid, (0.0, 0.5))
    line = frame_counts(live, 2, 2, None, records, None, None, NCELLS, len(RUN_GENES), 1.0, dest=dest)
    common = {"run": "w1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    block = {"model": str(model), "model_id": "feedface", "start": False, "k": K, "fit_min": 5, "judge_min": JUDGE,
             "codes": None, "cell_ids": str(live / "cells.txt"), "v": 1}
    request = {"op": "analyze", "ops": ["deeper"], **common, "seq": 2, "iter": 2, "frame": str(live / frame["file"]),
               "format": frame.get("format", "dense"), **grid, "cells_file": str(live / frame["cells_file"]),
               "counts_file": str(live / line["file"])}
    config["proseg"]["keep_unsaved"] = True                  # the hand-made run is loaded as it is
    worker, runs = W.Worker(xp=np), Runs(config, None)
    exports = attach(runs)
    where = DeeperWhere(runs)
    run = runs.get("w1")
    whole = [frame["x0"], frame["y0"], frame["x0"] + frame["nx"] * frame["voxel"],
             frame["y0"] + frame["ny"] * frame["voxel"]]
    exports.keep("w1", export_of(run, whole, records, dest, seq=2, iteration=2, grid=grid))
    states = set()
    try:
        for c in range(NCELLS):
            answer = worker.handle({**request, "deeper": {**block, "cell": c}})
            assert answer["ok"], answer
            result = answer["deeper"]
            states.add(result["state"])
            mine = json.loads(where.reply("w1", 2, c, [], RUN_GENES, []))
            counted = {s["gene"]: s["in_cell"] for s in mine["gene_stats"]}
            assert mine["basis"] == ("column" if tiled else "voxel") and mine["held"] in ("exact", "none")
            if mine["box"] is None:                          # the cell holds nothing in this frame
                assert result["state"] == "empty" and mine["held"] == "none"
                continue
            assert mine["held"] == "exact"
            assert sum(counted.values()) == pytest.approx(mine["cell_total"], abs=1e-6)  # every gene asked
            if result["state"] == "empty":                   # no transcript of the model's genes (and no `outside`)
                assert not any(n for g, n in counted.items() if g in MODEL_GENES), (c, counted)
                continue
            assert mine["cell_total"] == pytest.approx(result["transcripts"] + result["outside"]), (c, result)
            assert sum(n for g, n in counted.items() if g not in MODEL_GENES) == pytest.approx(result["outside"])
            rows = {row[0]: row[1] for row in result.get("genes", [])}
            assert all(counted[g] == pytest.approx(n) for g, n in rows.items())    # each gene's row
    finally:
        for run in list(runs.runs.values()):
            runs._release(run)
    assert "ready" in states and states & {"too_few", "empty"}


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("x0, y0, voxel", [(2800.015625, 1234.5, 1.0), (0.1, 2.0, 1.5)])
def test_ownership_is_the_workers_lookup_transcript_by_transcript(tmp_path, tiled, x0, y0, voxel):
    """Random labels and transcripts (many on column edges, some outside the grid and the z range): each owner is the
    label the worker's own lookup gives (voxel_lookup on the layers; tile_lookup + tile_labels on the plane), in its
    float32 arithmetic: with an origin and a column float32 cannot hold exactly, float64 would place some elsewhere."""
    from sopa_companion.analysis import worker as W
    from sopa_companion.engine.deeper_where import Dense, Plane
    from sopa_companion.engine.tiles import Container
    rng = np.random.default_rng(3)
    nx, ny, nz, tile = 37, 29, 4, 8
    layers = rng.integers(0, 9, (nz, ny, nx)).astype(np.uint32)
    n = 20000
    x = np.where(rng.random(n) < 0.5, x0 + rng.integers(-2, nx + 2, n) * voxel,
                 rng.uniform(x0 - 3, x0 + nx * voxel + 3, n)).astype(np.float32)
    y = np.where(rng.random(n) < 0.5, y0 + rng.integers(-2, ny + 2, n) * voxel,
                 rng.uniform(y0 - 3, y0 + ny * voxel + 3, n)).astype(np.float32)
    z = np.where(rng.random(n) < 0.3, rng.integers(-1, 6, n) * 0.25, rng.uniform(-0.3, 1.3, n)).astype(np.float32)
    records = np.zeros(n, RECORD)
    records["x"], records["y"], records["z"] = x, y, z
    if tiled:
        write_tiles(tmp_path / "f.tiles", {"consensus": layers[0], "last_change": layers[1]}, tile)
        labels = Plane("column", Container(tmp_path / "f.tiles"), "consensus", x0, y0, voxel, None)
        header, present, buffer = W.plane_tiles(str(tmp_path / "f.tiles"), "consensus")
        tiles = np.zeros((1 + len(present), tile, tile), np.uint32)
        tiles[1:] = buffer[:len(present)]
        ids, offsets = W.tile_lookup(np, x, y, x0, y0, voxel, nx, ny, tile, header["tiles_x"], header["tiles_y"])
        want = W.tile_labels(np, tiles, present, header["tiles_x"] * header["tiles_y"], ids, offsets)
    else:
        labels = Dense("voxel", layers, x0, y0, voxel, 0.0, 0.25)
        want = np.append(layers.ravel(), 0)[W.voxel_lookup(np, x, y, z, x0, y0, voxel, nx, ny, nz, 0.0, 0.25)]
    window = labels.window(0, 0, nx, ny)
    assert np.array_equal(labels.lookup(records, window, 0, 0), want) and (want > 0).any() and (want == 0).any()
    wide = np.floor((x.astype(np.float64) - x0) / voxel) != W.grid_columns(np, x, y, x0, y0, voxel, nx, ny)[0]
    assert wide.any() == (voxel != 1.0)                      # the second geometry tells float32 from float64


def test_a_tiled_priors_cells_are_found_where_the_worker_puts_them(tmp_path):
    """The start line of a tiled run: each cell's columns and centre over the whole label plane (the seeds its window
    starts from), as the worker's footprint() works them out: partial edge tiles, a label past the run's cells."""
    from sopa_companion.analysis import worker as W
    from sopa_companion.engine.deeper_where import plane_stats
    from sopa_companion.engine.tiles import Container
    rng = np.random.default_rng(5)
    plane = rng.integers(0, 7, (23, 29)).astype(np.uint32)
    plane[0, 0] = 99
    write_tiles(tmp_path / "prior.tiles", {"label": plane, "permille": plane}, 8)
    found = plane_stats(Container(tmp_path / "prior.tiles"), "label", 6, 10.0, 20.0, 2.0)
    header, present, buffer = W.plane_tiles(str(tmp_path / "prior.tiles"), "label")
    tiles = np.zeros((1 + len(present), 8, 8), np.uint32)
    tiles[1:] = buffer[:len(present)]
    area, cx, cy = W.footprint(np, ("tiles", header, present, tiles), 6, 10.0, 20.0, 2.0)
    assert np.array_equal(found["area"] * 4.0, area) and (found["area"] > 0).all()
    np.testing.assert_allclose(found["cx"], cx, rtol=1e-6)
    np.testing.assert_allclose(found["cy"], cy, rtol=1e-6)


# ---- through the API -------------------------------------------------------------------------------------------------
@pytest.fixture
def api(config):
    config["proseg"]["keep_unsaved"] = True                  # the hand-made runs are loaded as they are
    data = Path(config["paths"]["data_dir"])
    write_run(data, "run-col", layout()[None], placed(), tiled=True)
    write_run(data, "run-vox", np.stack([layout()] * 4), placed())
    write_run(data, "run-nuclei", layout()[None], placed(), tiled=True, chosen="nuclei",
              prior={"plane": layout()[::2, ::2], "voxel": 2.0})
    write_run(data, "run-going", layout()[None], placed(), tiled=True)
    write_run(data, "run-stain", layout()[None], placed(), tiled=True, kind="stain")
    from sopa_companion.server import create_app
    application = create_app(config, token="t" * 40)
    with TestClient(application, base_url=BASE) as client:
        for run_id in ("run-col", "run-vox", "run-nuclei", "run-going"):    # the counts as the engine merges them
            merge_counts(application.state.runs.get(run_id))
        yield application, client


def get(client, run_id, status=200, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    found = client.get(f"/api/v1/runs/{run_id}/qc/deeper/where?{query}", headers=auth())
    assert found.status_code == status, found.text
    return found


def test_where_through_the_api(api, monkeypatch):
    application, client = api
    asked = {"seq": 0, "cell": 0, "sources": "1,3", "genes": "EDGE,SPREAD,LAYER,FEW,NOPE", "own": "OWN1,OWN2,EDGE"}
    found = get(client, "run-col", **asked)
    assert found.headers["content-type"] == "application/json" and "no-cache" in found.headers["cache-control"]
    body = found.json()
    assert body["v"] == 3 and body["held"] == "approx"
    assert body["cell_total"] == 68 and body["box"] == [100.0, 50.0, 150.0, 90.0] and body["ms"] >= 0
    assert [(s["giver"], s["giver_by"]) for s in body["gene_stats"][:2]] == [(0, "density"), (1, "density")]
    assert get(client, "run-vox", **asked).json()["gene_stats"] == body["gene_stats"]
    # The window's kept answer names the givers (deeper.cached; here a stand-in): EDGE and SPREAD from B (cell 1).
    seen = []

    def cached(run_id, seq, cell, model=None):
        seen.append((run_id, seq, cell))
        return {"state": "ready", "neighbours": [{"cell": 1}, {"cell": 3}],
                "genes": [["EDGE", 30, 9.0, 21.0, 1, 0], ["SPREAD", 10, 1.0, 9.0, 1, 0], ["X", 2, 2, 0, 0]]}
    monkeypatch.setattr(application.state.deeper, "cached", cached)
    named = get(client, "run-col", **asked).json()
    assert seen == [("run-col", 0, 0)]
    assert [(s["giver"], s["giver_by"]) for s in named["gene_stats"][:2]] == [(0, "result"), (0, "result")]
    def missing(*args, **kwargs):
        seen.append(args)
        raise KeyError("not worked out")
    monkeypatch.setattr(application.state.deeper, "cached", missing)
    assert get(client, "run-col", **asked).json() == body                             # the kept reply again
    seen.clear()
    # Refusals: 404 never there, 409 not written yet (a run still going), 422 malformed; none asks for the kept answer.
    for run_id, params, status in (
            ("nope", asked, 404), ("run-stain", asked, 404), ("run-col", {**asked, "seq": 5}, 404),
            ("run-col", {**asked, "cell": 5}, 404), ("run-col", {**asked, "sources": "1,9"}, 404),
            ("run-col", {**asked, "seq": "start"}, 404), ("run-nuclei", {**asked, "seq": "start"}, 404),
            ("run-col", {**asked, "seq": "last"}, 422), ("run-col", {**asked, "cell": "a"}, 422),
            ("run-col", {**asked, "sources": "1,1"}, 422), ("run-col", {**asked, "sources": "0"}, 422),
            ("run-col", {"cell": 0}, 422), ("run-col", {"seq": 0}, 422)):
        get(client, run_id, status, **params)
    going = application.state.runs.get("run-going")
    going.meta["state"] = "running"
    get(client, "run-going", 409, **{**asked, "seq": 3})
    (going.live / "transcripts.bin.gz").rename(going.live / "transcripts.bin.gz.away")
    get(client, "run-going", 409, **asked)
    going.meta["state"] = "done"
    get(client, "run-going", 404, **asked)
    (going.live / "transcripts.bin.gz.away").rename(going.live / "transcripts.bin.gz")
    (going.live / "frames" / "000000.tiles").unlink()
    get(client, "run-going", 404, **asked)
    going.meta["state"] = "paused"
    get(client, "run-going", 409, **asked)
    assert seen == []
    # A cell and a source too far apart to show together.
    from sopa_companion.engine import deeper_where
    monkeypatch.setattr(deeper_where, "BOX_MAX_UM", 40.0)
    assert "too far apart" in get(client, "run-col", 422, **{**asked, "cell": 2}).json()["detail"]


# ---- the time it takes on a crop-sized frame ------------------------------------------------------------------------
@pytest.mark.parametrize("tiled", [False, True])
def test_a_crop_sized_frame_answers_within_300_ms(make_runs, tiled, capsys):
    """Budget: <= 300 ms on the crop. A detail run's largest frame (1,200 x 1,200 µm at 1 µm, 4 layers,
    decoded whole) and a 1,500 µm whole-sample crop (tiles of 512, its window holding the box); 10 µm cells; crop
    density (1.33 per µm²) around the cell; proseg's count of the frame read whole. Cold: nothing kept but the run's
    transcript index (built once per run). The best of three."""
    from sopa_companion.engine import tiles
    side = 1500 if tiled else 1200
    j, i = np.mgrid[0:side, 0:side]
    plane = ((j // 10) * (side // 10) + i // 10 + 1).astype(np.uint32)
    plane[(i % 10 == 0) | (j % 10 == 0)] = 0                  # a column of background between cells
    cells = [f"c{k}" for k in range(int(plane.max()))]
    rng = np.random.default_rng(0)
    n = int(1.33 * 240 * 240)                                 # a 240 µm square around the cell (column 605, row 605)
    records = np.zeros(n, RECORD)
    records["x"], records["y"] = rng.uniform(X0 + 485, X0 + 725, n), rng.uniform(Y0 + 485, Y0 + 725, n)
    records["z"], records["gene"] = rng.uniform(0, 1, n), rng.integers(0, len(GENES), n)
    layers = plane[None] if tiled else np.stack([plane, np.roll(plane, 1, 1), plane, np.roll(plane, 1, 0)])
    runs, where = make_runs(big=(layers, records, {"tiled": tiled, "cells": cells, "tile": 512}))
    runs.transcript_index("big")
    c = int(plane[605, 605]) - 1
    sources = [c + 1, c - 1, c + side // 10, c - side // 10, c + side // 10 + 1]
    times = []
    for _ in range(3):
        where.replies.clear()
        where.dense.clear()
        where.counts.clear()
        where.boxes.clear()
        tiles.CACHE.forget(str(runs.get("big").dir))
        began = time.perf_counter()
        found = json.loads(where.reply("big", 0, c, sources, ["EDGE", "SPREAD", "LAYER", "FEW"], ["OWN1", "OWN2"]))
        times.append((time.perf_counter() - began) * 1e3)
    assert found["held"] == "approx" and found["cell_total"] > 0 and len(found["points"]["x"]) > 1000
    with capsys.disabled():
        print(f"\n  where on a {side} µm {'tiled' if tiled else 'detail'} frame: "
              f"{', '.join(f'{t:.0f}' for t in times)} ms (reply {found['ms']} ms, {len(found['points']['x'])} points)")
    assert min(times) <= 300
