"""Test a setting here on a whole-sample (tiled) run (engine/qc_pool.py `_TiledContext`, `_RecordIndex`): its last
frame F read in its column basis for positions only, for `answers` and the pool's own work. A tiled run whose frame and
starting cells hold the planes of a one-layer detail run (its twin: the same records, cells, stored bytes, model and
proseg's count) answers as the twin does while both are held at F (each transcript's share from proseg's export of the
test area: never its column), cell by cell, and `_take` gives the same pools; record keys are np.unique's; a target's
start is prior.tiles' at the worker's own column; a giver's footprint and "near" are the where reply's; a per-layer
window in the frame changes nothing while held; past F, the window places the shares when it covers the area
(approx_targets), else they are taken from the parent's state now (targets_now), else, the parent ended, the window
places them over the part of the area it holds (approx_targets), else no record has a share: nothing glows, or a cell
Go deeper's where engine still places (an export of its box kept) is refused, having no control (no_controls); such a
run is tested like its twin, with its caveat (whole_sample); and the containers here are written as proseg-live writes
them.

The world (`layout`): 12 cells on a 1 µm grid (tiles of 16, partial at both edges), some grown into a neighbour's start
(held there at 800, 650, 649 or 500 permille), into another cell's or where no cell started; several whose boxes take
in other cells' columns; a wide and short one; one with a fragment 200 columns from its centre (found only by scanning
every tile) and one with a fragment near it; one partly off the starting cells' 2 µm grid; and a record held twice.
Proseg's count: each transcript goes to the cell owning its column (one share kept of every gene). Go deeper and the
where measurement are the real ones, each point's w the world's (test_qc_pool's HeldWhere), or, for a parent that
ended past F, placed as Go deeper's where engine places a box (WindowWhere; R5's own engine as well, once merged); the
page's scoring is test_qc_pool's stand-in, each cell asking its own sources and genes (PLAN)."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

import counts_fixtures as CF
from sopa_companion.analysis import counts_file
from sopa_companion.analysis import worker as W
from sopa_companion.engine import held
from sopa_companion.engine import qc_pool as P
from sopa_companion.engine import qc_rule as Q
from sopa_companion.engine import qc_scoring as S
from sopa_companion.engine.deeper_where import NEAR_UM, SEED_GROW, SEED_HALF, DeeperWhere, near
from sopa_companion.engine.tiles import Container
from sopa_companion.engine.transcript_index import RECORD
from test_analysis import RUN_GENES, write_container, write_run as write_model_run
from test_deeper_where import cell_stats
from test_qc_pool import (COUNTS, MIXED, PLAIN, Analysis, HeldWhere, Page, Qc, Run, Runs,  # noqa: F401
                          measured_on_proseg, write_codes)
from test_worker_deeper import RUN_IDS, make_deep

# The package these tests belong to, not another copy found first on the path.
assert Path(P.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1], P.__file__

X0, Y0, VOXEL, NX, NY = 100.0, 50.0, 1.0, 211, 45     # F's grid, [100, 311) x [50, 95) µm (13 x 16 + 3, 2 x 16 + 13)
TILE, NZ = 16, 4                                      # the tiled frame's tiles and layers (its twin: one layer)
# The starting cells' grid, [99.2, 299.2) x [49, 95) µm in tiles of 8: its x0 is no float32 value, so float32 columns
# (the worker's) and float64 ones part on some points.
PX0, PY0, PVOXEL, PNX, PNY, PTILE = 99.2, 49.0, 2.0, 100, 23, 8
SEQ, ITER, NCELLS, K = 0, 450, 12, 100                # F, its iteration, the cells, the pools' k here
STRONG = Q.MEASURED["strong_permille"]
MIXED_AT = Q.MEASURED["thresholds"]["mixed_at"]
INSET = float(Q.MEASURED["inset_um"])
REGION = (X0 - 200.0, Y0 - 300.0, X0 + 400.0, Y0 + 300.0)                  # 600 µm: a square around the cell
SQUARE = (X0 - 13.0, Y0 - 225.0, X0 + 487.0, Y0 + 275.0)                   # 500 µm; its inner area leaves cell 10 out
AREA = P.Area(rule="square", box=SQUARE, inner=(SQUARE[0] + INSET, SQUARE[1] + INSET, SQUARE[2] - INSET,
                                                SQUARE[3] - INSET), spec={"kind": "box", "box": list(SQUARE)})

# F's cells (label = cell + 1) over columns [c0, c1) and rows [r0, r1) of its grid, painted in this order.
FRAME = [(11, 0, 10, 10, 40), (11, 206, 210, 20, 22),       # cell 10, and its fragment 200 columns away
         (1, 10, 25, 10, 25), (2, 25, 40, 10, 25), (1, 25, 30, 20, 25),     # cell 0 grew into cell 1's start
         (3, 40, 55, 10, 25),
         (4, 10, 25, 25, 40), (5, 25, 40, 25, 40), (5, 20, 25, 25, 30),     # cell 4 into cell 3's
         (6, 40, 55, 25, 40), (6, 45, 55, 22, 25),                          # cell 5 into cell 2's
         (7, 55, 70, 10, 40),
         (8, 10, 70, 40, 43),                                               # cell 7: wide and short
         (9, 70, 85, 10, 25), (9, 85, 88, 10, 25),                          # cell 8 where no cell started
         (10, 70, 85, 25, 40), (10, 100, 103, 30, 33),                      # cell 9, and a fragment 23 columns away
         (12, 180, 206, 15, 27)]                                            # cell 11, its last columns off the start
# The starting cells: label, F's columns and rows, permille; each 2 µm column takes them at its centre.
START = [(11, 0, 10, 10, 40, 800), (1, 10, 25, 10, 25, 800),
         (2, 25, 40, 10, 25, 800), (2, 25, 27, 20, 25, 650), (2, 27, 29, 20, 25, 649),     # under cell 0's growth
         (3, 40, 55, 10, 25, 800), (3, 45, 55, 22, 25, 500),                               # under cell 5's
         (4, 10, 25, 25, 40, 800), (5, 25, 40, 25, 40, 800), (6, 40, 55, 25, 40, 800), (7, 55, 70, 10, 40, 800),
         (8, 10, 40, 40, 43, 800),                                         # cell 7 started half as wide
         (9, 70, 85, 10, 25, 800), (0, 85, 88, 10, 25, 300),               # no cell there (a stray permille under it)
         (10, 70, 85, 25, 40, 800), (12, 180, 206, 15, 27, 800)]
# Transcripts planted one per column at its centre (gene, columns, rows), where the start classes are decided.
PLANTED = [("G1", 25, 30, 20, 25),          # cell 0's growth over cell 1's start: 650, 649 and 800 permille
           ("G5", 45, 55, 22, 25),          # cell 5's over cell 2's: 500
           ("G3", 20, 25, 25, 30),          # cell 4's over cell 3's (its G3 comes from cell 1: another cell's)
           ("G3", 85, 88, 10, 25),          # cell 8's where no cell started
           ("G9", 10, 25, 39, 40),          # cell 3's last row, whose start columns hold cell 7's start
           ("G1", 180, 206, 15, 27)]        # cell 11, on the start's grid and off it
# Each cell's where request (the page's, here a stand-in): {gene asked as a neighbour's: the source it comes from}.
PLAN = {0: {"G1": 1, "G2": 3, "G3": 10}, 1: {"G4": 0, "G5": 2, "G6": 4}, 2: {"G1": 1, "G7": 5, "G8": 6},
        3: {"G2": 0, "G4": 4, "G9": 7}, 4: {"G3": 1, "G6": 5, "G9": 7}, 5: {"G5": 2, "G8": 6, "G9": 7},
        6: {"G7": 5, "G1": 8, "G2": 9}, 7: {"G4": 3, "G5": 4, "G6": 5}, 8: {"G3": 6, "G4": 9}, 9: {"G5": 6, "G6": 8},
        10: {"G7": 0, "G8": 3}, 11: {"G1": 10, "G2": 10}}


def layout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F's consensus plane [NY, NX], the starting cells' label and permille [PNY, PNX])."""
    plane = np.zeros((NY, NX), np.uint32)
    for label, c0, c1, r0, r1 in FRAME:
        plane[r0:r1, c0:c1] = label
    j, i = np.mgrid[0:PNY, 0:PNX]
    col, row = (PX0 + (i + 0.5) * PVOXEL - X0) / VOXEL, (PY0 + (j + 0.5) * PVOXEL - Y0) / VOXEL
    start, permille = np.zeros((PNY, PNX), np.uint32), np.zeros((PNY, PNX), np.uint32)
    for label, c0, c1, r0, r1, level in START:
        at = (col >= c0) & (col < c1) & (row >= r0) & (row < r1)
        start[at], permille[at] = label, level
    return plane, start, permille


def world_records(plane: np.ndarray) -> np.ndarray:
    """The run's records: each cell's at random over its columns (one per column on average; genes G0-G9 alike, a few
    control probes), some anywhere (in no cell, off F's grid), the planted ones; then a record held twice (cell 0's
    planted G1 at column 27, row 22) and two at a planted record's place that are other records (another gene, another
    depth)."""
    rng = np.random.default_rng(7)
    xs, ys = [], []
    for label in range(1, NCELLS + 1):
        rows, cols = np.nonzero(plane == label)
        pick = rng.integers(0, rows.size, rows.size)
        xs.append(X0 + (cols[pick] + rng.uniform(0.05, 0.95, pick.size)) * VOXEL)
        ys.append(Y0 + (rows[pick] + rng.uniform(0.05, 0.95, pick.size)) * VOXEL)
    xs.append(rng.uniform(X0 - 9.0, X0 + NX * VOXEL, 400))
    ys.append(rng.uniform(Y0, Y0 + NY * VOXEL + 5.0, 400))
    records = np.zeros(sum(len(v) for v in xs), RECORD)
    records["x"], records["y"] = np.concatenate(xs), np.concatenate(ys)
    records["z"] = rng.uniform(0.0, 1.0, len(records))
    records["gene"] = rng.choice(len(RUN_GENES), len(records), p=np.r_[np.full(10, 0.097), 0.03])
    planted = np.array([(X0 + (col + 0.5) * VOXEL, Y0 + (row + 0.5) * VOXEL, 0.0, RUN_GENES.index(gene))
                        for gene, c0, c1, r0, r1 in PLANTED for row in range(r0, r1) for col in range(c0, c1)], RECORD)
    planted["z"] = rng.uniform(0.0, 1.0, len(planted))

    def at(col, row):
        return planted[(planted["x"] == np.float32(X0 + col + 0.5)) & (planted["y"] == np.float32(Y0 + row + 0.5))]
    other_gene, other_depth = at(26, 21), at(28, 23)
    other_gene["gene"], other_depth["z"] = RUN_GENES.index("G2"), other_depth["z"] / 2
    return np.concatenate([records, planted, at(27, 22), other_gene, other_depth])


def write_plt1(path: Path, planes: dict, tile: int, nz: int, window=None):
    """A proseg-live tile container as live.rs writes one: "PLT1", the u32 length of its compact JSON header {nx, ny,
    nz (a frame's layers; a prior's 1), tile, tiles_x, tiles_y, planes, window}, an index of (u64 offset, u32 length)
    entries (tiles row-major, planes fastest, then the window's), the blobs: each tile at its own size (an edge tile is
    smaller), zstd level 1 with its content size (worker.plane_tiles decodes a whole tile straight into its buffer),
    length 0 where all zero. `window`: ({i0, j0, nx, ny, nz}, layers [nz, ny, nx]) or None."""
    import zstandard
    names = list(planes)
    ny, nx = planes[names[0]].shape
    tiles_x, tiles_y = max(1, -(-nx // tile)), max(1, -(-ny // tile))
    blobs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            for name in names:
                block = np.ascontiguousarray(planes[name][ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile], "<u4")
                blobs.append(zstandard.ZstdCompressor(level=1).compress(block.tobytes()) if block.any() else b"")
    info, layers = window if window is not None else (None, None)
    blobs.append(zstandard.ZstdCompressor(level=1).compress(np.ascontiguousarray(layers, "<u4").tobytes())
                 if layers is not None and np.any(layers) else b"")
    header = json.dumps({"nx": nx, "ny": ny, "nz": nz, "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
                         "planes": names, "window": info}, separators=(",", ":")).encode()
    offset, head = 8 + len(header) + 12 * len(blobs), b"PLT1" + struct.pack("<I", len(header)) + header
    for blob in blobs:
        head += struct.pack("<QI", offset, len(blob))
        offset += len(blob)
    path.write_bytes(head + b"".join(blobs))


def write_world(folder: Path, *, tiled: bool, window=None) -> Path:
    """The layout's run folder (live/ as proseg-live leaves it; F = seq 0): tiled (F and the start as tile containers;
    `window`: F's per-layer window, as write_plt1 takes it, or None) or its one-layer detail twin (zstd: F's consensus,
    last change and its one layer, the consensus; the start's label and permille)."""
    import zstandard
    plane, start, permille = layout()
    live = folder / "live"
    (live / "frames").mkdir(parents=True)
    (live / "transcripts.bin.gz").write_bytes(gzip.compress(world_records(plane).tobytes()))
    (live / "genes.txt").write_text("".join(g + "\n" for g in RUN_GENES))
    (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    stats = cell_stats(plane, plane[None], X0, Y0, VOXEL, NCELLS).astype("<u4")
    (live / "frames/000000.cells.zst").write_bytes(zstandard.ZstdCompressor(level=1).compress(stats.tobytes()))
    last = np.where(plane > 0, ITER, 0).astype(np.uint32)
    geometry = {"nx": PNX, "ny": PNY, "nz": 1, "x0": PX0, "y0": PY0, "voxel": PVOXEL}
    record = {"seq": SEQ, "iter": ITER, "phase": "recorded", "nx": NX, "ny": NY, "x0": X0, "y0": Y0, "voxel": VOXEL,
              "cells": NCELLS, "cells_file": "frames/000000.cells.zst"}
    if tiled:
        write_plt1(live / "prior.tiles", {"label": start, "permille": permille}, PTILE, 1)
        prior = {"file": "prior.tiles", "format": "tiles", "tile": PTILE, **geometry}
        write_plt1(live / "frames/000000.tiles", {"consensus": plane, "last_change": last}, TILE, NZ, window)
        info = None if window is None else window[0]
        record.update(file="frames/000000.tiles", format="tiles", tile=TILE, nz=NZ,
                      window=None if info is None else {k: info[k] for k in ("i0", "j0", "nx", "ny")},
                      view=None if info is None else [X0 + info["i0"] * VOXEL, Y0 + info["j0"] * VOXEL,
                                                      X0 + (info["i0"] + info["nx"]) * VOXEL,
                                                      Y0 + (info["j0"] + info["ny"]) * VOXEL])
    else:
        raw = np.concatenate([start.ravel(), permille.ravel()]).astype("<u4")
        (live / "prior.bin.zst").write_bytes(zstandard.ZstdCompressor(level=3).compress(raw.tobytes()))
        prior = {"file": "prior.bin.zst", **geometry}
        raw = np.concatenate([plane.ravel(), last.ravel(), plane.ravel()]).astype("<u4")
        (live / "frames/000000.bin.zst").write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.tobytes()))
        record.update(file="frames/000000.bin.zst", nz=1)
    (live / "info.json").write_text(json.dumps({"cells": NCELLS, "genes": len(RUN_GENES), "layers": record["nz"],
                                                "z": {"grid_origin": 0.0, "grid_spacing": 1.0 / NZ}, "prior": prior}))
    (live / "frames.jsonl").write_text(json.dumps(record) + "\n")
    return folder


class Twin:
    """The layout's run p1, tiled or its one-layer detail twin, paused at the end of its schedule and held there
    (proseg's ack "paused" at F), with QcPools over it and test_qc_pool's stand-ins (the mixing check serves F's stored
    bytes: every cell mixed but cell 9). Proseg's count at F: every transcript goes to the cell owning its column
    (`dest`), `keep` of each gene kept; the parent exports a box from those destinations while paused (`export`).
    `where_held`: how HeldWhere places w ("exact"; "approx" from `window_layers`, the frame's window; "none")."""

    def __init__(self, folder: Path, model: Path, *, tiled: bool, window=None):
        write_world(folder, tiled=tiled, window=window)
        self.live = folder / "live"
        info = json.loads((self.live / "info.json").read_text())
        (self.live / "info.json").write_text(json.dumps({**info, "counts": dict(COUNTS)}))
        self.records = np.frombuffer(gzip.decompress((self.live / "transcripts.bin.gz").read_bytes()), RECORD).copy()
        self.grid = {"x0": X0, "y0": Y0, "voxel": VOXEL, "nx": NX, "ny": NY, "nz": NZ if tiled else 1}
        self.z0, self.dz = 0.0, 1.0 / NZ
        self.plane = layout()[0]
        i, j, _, inside = CF.observed_voxels(self.records, self.grid, self.z0, self.dz)
        self.dest = np.zeros(len(self.records), np.int64)
        self.dest[inside] = self.plane[j[inside], i[inside]]
        self.keep = np.full(len(RUN_GENES), 0.75)
        self.write_counts(SEQ, ITER, "hold")
        self.codes = np.full(NCELLS, MIXED, np.uint8)
        self.codes[9] = PLAIN
        self.qc_dir = folder / "qc"
        write_codes(self.qc_dir, SEQ, self.codes)
        self.meta = {"id": "p1", "kind": "proseg", "state": "paused", "schedule_done": {"seq": SEQ, "iter": ITER},
                     "region": {"kind": "box", "box": list(REGION), "bbox": list(REGION)},
                     "mode": "tiled" if tiled else "detail", "options": {},
                     "prior": {"cells": {"kind": "shapes", "key": "cells"},
                               "nuclei": {"kind": "shapes", "key": "nuclei"}},
                     "sample_kind": "zarr", "prepared": {"seeded_from": "nuclei"}, "live_counts": "request"}
        self.run = Run(folder, self.meta)
        self.run.ack, self.run.export = {"state": "paused", "seq": SEQ, "iter": ITER}, self.export
        self.runs = Runs(self.run)
        self.where_held, self.window_layers, self.tokens, self.kept_w = "exact", None, 0, {}
        self.where = HeldWhere(self.runs, self)
        self.pools = P.QcPools(self.runs, Qc(model, self.qc_dir), Analysis(), self.where)

    def write_counts(self, seq: int, iteration: int, reason: str = "request") -> dict:
        found = CF.counts_of(self.dest, self.records["gene"], NCELLS, len(RUN_GENES))
        e = found["n"] * self.keep[found["genes"]]
        return CF.write_frame_counts(self.live, seq, iteration, ncells=NCELLS, ngenes=len(RUN_GENES),
                                     indptr=found["indptr"], genes=found["genes"], n=found["n"], e=e,
                                     f=np.floor(e).astype(np.int64), reason=reason)

    def export(self, box, seq: int) -> dict:
        """Proseg's export of `box` at `seq`: one record per transcript whose observed voxel lies in its span."""
        self.tokens += 1
        i0, j0, i1, j1 = CF.where_span(box, self.grid)
        i, j, k, inside = CF.observed_voxels(self.records, self.grid, self.z0, self.dz)
        span = inside & (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
        entries = {"i": i[span], "j": j[span], "k": k[span], "gene": self.records["gene"][span].astype(np.int64),
                   "dest": self.dest[span], "n": np.ones(int(span.sum()), np.int64)}
        return CF.write_where_export(self.live, self.tokens, seq, ITER, box, self.grid, entries)

    def q(self, cell: int, which: str = "e") -> np.ndarray:
        found = counts_file.read(self.live / self.run.frames[SEQ]["counts_file"])
        rows = counts_file.rows(found, [cell], (which, "n"))
        e, n = rows[which][0].astype(np.float64), rows["n"][0].astype(np.float64)
        return np.divide(e, n, out=np.zeros_like(e), where=n > 0)

    def w_all(self, cell: int, how: str = "exact", which: str = "e") -> np.ndarray:
        key = (cell, how, which)
        if key not in self.kept_w:
            if how == "exact":
                i, j, k, inside = CF.observed_voxels(self.records, self.grid, self.z0, self.dz)
                export = held.WhereExport(token=0, seq=SEQ, iter=ITER, box=(0.0, 0.0, 1.0, 1.0), margin_um=0.0,
                                          x0=X0, y0=Y0, voxel=VOXEL, nx=NX, ny=NY, nz=self.grid["nz"], i0=0, j0=0,
                                          i1=NX - 1, j1=NY - 1, i=i[inside], j=j[inside], k=k[inside],
                                          gene=self.records["gene"][inside].astype(np.int64), dest=self.dest[inside],
                                          n=np.ones(int(inside.sum()), np.int64))
                w, _ = held.weights(self.records, cell, self.q(cell, which), self.grid, self.z0, self.dz,
                                    export=export)
            elif how == "approx":
                w, _ = held.weights(self.records, cell, self.q(cell, which), self.grid, self.z0, self.dz,
                                    layers=self.window_layers)
            else:
                w = np.zeros(len(self.records))
            self.kept_w[key] = w
        return self.kept_w[key]

    def w_of(self, cell: int, records: np.ndarray, how: str = "exact", which: str = "e") -> np.ndarray:
        keys, first = np.unique(P._void(self.records), return_index=True)
        at = np.searchsorted(keys, P._void(np.ascontiguousarray(records, RECORD)))
        return self.w_all(cell, how, which)[first[at]]

    def layers_at(self, seq: int) -> np.ndarray:
        """The footprints' plane: F's consensus (the twin's one layer)."""
        return self.plane[None]

    def context(self) -> P._Context:
        """A frame context of F made here, outside the run's worker, as QcPools makes it for the area AREA: the class
        its frame asks for, on the area's basis."""
        facts = self.pools._facts_at_end(self.run)
        basis = self.pools._basis(self.run, facts, tuple(AREA.box))
        tiled = facts.frame.get("format") == "tiles"
        return (P._TiledContext if tiled else P._Context)((*facts.key(), basis.seq, basis.held), self.run, facts,
                                                          basis)

    def judged_mixed(self) -> list[int]:
        return [c for c in range(NCELLS) if self.codes[c] != P.NOT_JUDGED and self.codes[c] >= W.code_at(MIXED_AT)]

    def owned(self, cell: int) -> np.ndarray:
        """The records of the cell's columns (its own transcripts in the drawing), in file order."""
        return np.flatnonzero(self.dest == cell + 1)


class WindowWhere(HeldWhere):
    """Go deeper's where engine placing each cell's box at a tiled frame as INTERFACES §8.2 says the engine's own does
    (a stand-in beside it), never the world's say: exact from an export of that frame holding the reply's box
    that the engine keeps (Runs.where_export; the world's own shares); else approx from the frame's per-layer window
    when it holds every column and row of the box, one to spare each side (its label at each record's own layer times
    the cell's keep); else none. (An export asked of a parent paused at that frame is not made here: these worlds are
    never paused there.)"""

    def placed(self, run_id, seq, cell, out, records, count="averaged"):
        run, box = self.runs.get(run_id), out["box"]
        which = "f" if count == "sampled" else "e"
        nothing = np.zeros(len(records))
        if box is None:
            return "none", nothing
        if self.runs.where_export(run_id, seq, box) is not None:
            return "exact", self.world.w_of(cell, records, "exact", which)
        frame = run.frames[seq]
        container = Container(run.live / frame["file"])
        if not container.header.get("window"):
            return "none", nothing
        info, layers = container.window()
        v = float(frame["voxel"])
        c0, c1 = max(0, math.floor((box[0] - X0) / v) - 1), min(NX - 1, math.floor((box[2] - X0) / v) + 1)
        r0, r1 = max(0, math.floor((box[1] - Y0) / v) - 1), min(NY - 1, math.floor((box[3] - Y0) / v) + 1)
        i0, j0 = int(info["i0"]), int(info["j0"])
        if not (i0 <= c0 and c1 < i0 + int(info["nx"]) and j0 <= r0 and r1 < j0 + int(info["ny"])):
            return "none", nothing
        w, how = held.weights(records, cell, self.world.q(cell, which), self.world.grid, self.world.z0, self.world.dz,
                              layers=(layers, i0, j0))
        assert how == "approx"
        return "approx", w


def where_engines():
    """The where engines a test runs with: the stand-in, and the engine's own (R5's)."""
    return ["stand-in", "R5"]


def use_where(twin, which: str):
    """The twin's where engine for Go deeper and the pools: the stand-in (WindowWhere) or R5's own."""
    twin.where = twin.pools.where = WindowWhere(twin.runs, twin) if which == "stand-in" else DeeperWhere(twin.runs)


class PlannedPage(Page):
    """test_qc_pool's page stand-in, each cell asking its own sources and genes (PLAN); P 0.7 near the gene's giver,
    else 0.55 (G9: 0.3, a target that is not likely)."""

    def __init__(self):
        super().__init__()
        self.p = lambda gene, near: 0.7 if near else (0.3 if gene == "G9" else 0.55)

    def params(self, raw, cell):
        page = Page()                            # one per call: two runs' worker threads never share one
        page.sources, page.genes = list(dict.fromkeys(PLAN[cell].values())), list(PLAN[cell])
        return page.params(raw, cell)


@pytest.fixture
def make_twin(tmp_path, monkeypatch):
    """make(name, tiled=..., window=...) -> a Twin; one model (test_worker_deeper.make_deep), PLAN's page and givers,
    the pools' k at K, and the frame contexts kept until the end."""
    model = make_deep(tmp_path / "model", write_model_run(tmp_path / "model-run"))
    page = PlannedPage()
    monkeypatch.setattr(S, "page_params", page.params)
    monkeypatch.setattr(S, "page_scores", page.scores)
    monkeypatch.setattr(P, "givers_of", lambda result: dict(PLAN[result["cell"]]))
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", K)
    monkeypatch.setattr(P, "POOL_HOLD_S", 3600.0)
    made = []

    def make(name, **options):
        made.append(Twin(tmp_path / name, model, **options))
        return made[-1]
    yield make
    for twin in made:
        twin.pools.close()


def answers(twin) -> dict:
    """Every cell's answer (QcPools.answers), without its wall-clock seconds."""
    found = twin.pools.answers("p1", range(NCELLS))
    return {cell: {k: v for k, v in answer.items() if k != "seconds"} for cell, answer in found.items()}


def take(twin, cell: int, area: P.Area = AREA) -> tuple:
    """(reasons, pool, the reply's cell fields) of `_take` for the cell at F with this fixed area: pool()'s own job,
    through the run's worker, the area given instead of the cheap tier's."""
    pools = twin.pools
    facts = pools._facts_at_end(twin.run)
    facts.area = area
    while True:                                  # asked again when a newer request replaced it before it began
        job = pools._submit("p1", P._Job(pools._pool_key(facts, cell), facts, cell=cell))
        assert job.done.wait(120)
        if not job.dropped:
            break
    if isinstance(job.outcome, BaseException):
        raise job.outcome
    return job.outcome.reasons, job.outcome.pool, job.outcome.fields


def footprint_near(plane: np.ndarray, label: int, x, y) -> np.ndarray:
    """Whether each point lies within NEAR_UM of a column the label owns in the plane, every column measured."""
    rows, cols = np.nonzero(plane == label)
    lx, ly = X0 + cols * VOXEL, Y0 + rows * VOXEL
    px, py = np.asarray(x, np.float64)[:, None], np.asarray(y, np.float64)[:, None]
    dx = np.maximum(np.maximum(lx - px, px - (lx + VOXEL)), 0.0)
    dy = np.maximum(np.maximum(ly - py, py - (ly + VOXEL)), 0.0)
    return (dx * dx + dy * dy <= NEAR_UM * NEAR_UM + 1e-9).any(axis=1)


# ---- the tiled branch against its one-layer detail twin ----------------------------------------------------------
def test_a_tiled_frame_answers_as_its_one_layer_dense_twin(make_twin):
    """The same F in either basis (the twin's one layer is the tiled frame's consensus, its start the tiled start):
    every cell's answer is the same, and so is `_take`'s pool for every judged, mixed cell with one fixed square: its
    reasons, the reply's fields, the Pool with every Item. The worlds hold what the comparison has to see."""
    dense, tiled = make_twin("dense", tiled=False), make_twin("tiled", tiled=True)
    for name in ("transcripts.bin.gz", "cells.txt", "genes.txt", "frames/000000.cells.zst"):
        assert (tiled.live / name).read_bytes() == (dense.live / name).read_bytes()
    plane, start, permille = layout()
    assert np.array_equal(W.read_tiled_consensus(str(tiled.live / "frames/000000.tiles")), plane)
    assert np.array_equal(W.read_frame(str(dense.live / "frames/000000.bin.zst"))[2 * NX * NY:].reshape(NY, NX), plane)
    prior = Container(tiled.live / "prior.tiles")
    assert np.array_equal(prior.full("label"), start) and np.array_equal(prior.full("permille"), permille)
    got = answers(tiled)
    assert got == answers(dense)
    assert type(tiled.pools.work["p1"].context) is P._TiledContext
    assert type(dense.pools.work["p1"].context) is P._Context
    # What the answers hold: every start class (650 strong, 649 and 500 weak, no cell there and off the start's grid),
    # near and far, likely and not, the record held twice, controls drawn by their near flag.
    targets = [t for answer in got.values() for t in answer["targets"]]
    assert {t["start_class"] for t in targets} == {"OWN", "GIVER_STRONG", "GIVER_WEAK", "NONE", "OTHER"}
    assert {("GIVER_STRONG", 650), ("GIVER_WEAK", 649), ("GIVER_WEAK", 500), ("NONE", 0)} <= \
        {(t["start_class"], t["permille"]) for t in targets}
    assert {t["start_label"] for t in targets if t["start_class"] == "NONE"} == {0, -1}
    assert {t["near"] for t in targets} == {True, False} and {t["likely"] for t in targets} == {True, False}
    assert sum(t["copies"] == 2 for t in got[0]["targets"]) == 2 and got[9]["mixed"] is False
    assert any(answer["controls"]["matched"] for answer in got.values())
    taken = {}
    for cell in tiled.judged_mixed():
        taken[cell] = take(tiled, cell)
        assert taken[cell] == take(dense, cell), cell
    reached = [cell for cell, (reasons, _, fields) in taken.items() if not reasons and fields["pool"]["reached"]]
    assert reached and all(taken[cell][1].k == K and taken[cell][1].suspects >= K for cell in reached)
    short = [cell for cell, (reasons, _, fields) in taken.items() if not reasons and not fields["pool"]["reached"]]
    assert all(taken[cell][1].caveats[-1] == P.reason("too_few", have=taken[cell][1].suspects, need=K)
               for cell in short)
    assert taken[11][0] == [P.reason("no_primary")]                      # its suspects' giver lies outside
    pool = taken[0][1]
    assert pool is not None and pool.record_duplicates == 1 and len(pool.cells) > pool.members
    assert {item.kind for item in pool.items} == {"target", "matched", "uniform"}


def test_a_frame_with_a_window_reads_as_one_without(make_twin):
    """A tiled frame carrying a per-layer window (in its record and its container, labels unlike its consensus) reads
    as the same frame without one: the same answers and pools."""
    plane, _, _ = layout()
    info = {"i0": 20, "j0": 5, "nx": 40, "ny": 30, "nz": NZ}
    layers = np.stack([np.roll(plane[5:35, 20:60], k + 1, axis=1) for k in range(NZ)])     # not the consensus
    plain, windowed = make_twin("plain", tiled=True), make_twin("windowed", tiled=True, window=(info, layers))
    assert windowed.run.frames[SEQ]["window"] == {"i0": 20, "j0": 5, "nx": 40, "ny": 30}
    assert plain.run.frames[SEQ]["window"] is None
    found = Container(windowed.live / "frames/000000.tiles").window()
    assert found[0] == info and np.array_equal(found[1], layers)
    assert answers(windowed) == answers(plain)
    for cell in plain.judged_mixed():
        assert take(windowed, cell) == take(plain, cell), cell


# ---- past F: where the shares come from ----------------------------------------------------------------------------
def moved_on(twin, seq: int | None = None):
    """The tiled parent sampled on past F: paused again at `seq` (None: ended, nothing to ask), no export kept."""
    twin.meta["continued"] = True
    twin.run.ack = None if seq is None else {"state": "paused", "seq": seq, "iter": ITER + seq}
    twin.run.wheres = []


def add_frame(twin, seq: int):
    """Frame `seq` of the tiled twin: F's planes and statistics again, counted (proseg's count of its state)."""
    live = twin.live
    for kind in ("tiles", "cells.zst"):
        (live / f"frames/{seq:06d}.{kind}").write_bytes((live / f"frames/000000.{kind}").read_bytes())
    twin.run.frames.append({**twin.run.frames[SEQ], "seq": seq, "iter": ITER + seq, "file": f"frames/{seq:06d}.tiles",
                            "cells_file": f"frames/{seq:06d}.cells.zst"})
    twin.write_counts(seq, ITER + seq, "pause")
    twin.run.read_counts()


def test_past_f_a_window_over_the_area_places_the_shares_approximately(make_twin):
    """A tiled parent past F whose frame kept a per-layer window over the whole test area: nothing is asked of it, and
    each record's share is the window's label at its own layer times the cell's keep (never the consensus column: a
    layer that gives a cell's columns to another counts for that one), with the caveat approx_targets."""
    plane, _, _ = layout()
    layers = np.stack([plane] * NZ)
    layers[3][plane == 4] = 5                                          # cell 3's top layer is cell 4's
    info = {"i0": 0, "j0": 0, "nx": NX, "ny": NY, "nz": NZ}
    twin = make_twin("windowed", tiled=True, window=(info, layers))
    moved_on(twin, SEQ + 4)
    twin.where_held, twin.window_layers = "approx", (layers, 0, 0)
    context = twin.context()
    try:
        assert (context.basis.held, context.basis.caveat, twin.runs.asked) == ("approx", "approx_targets", [])
        weights = context.weights(tuple(AREA.box), context.basis)
        for c in range(NCELLS):
            idx, w = weights.of(c, context.q(c))
            whole = twin.w_all(c, "approx")
            assert idx.tolist() == np.flatnonzero(whole > 0).tolist() and np.allclose(w, whole[idx]), c
        idx, _ = weights.of(4, context.q(4))
        top = np.clip(np.floor(twin.records["z"] / twin.dz), 0, NZ - 1) == 3
        assert set(idx.tolist()) >= set(np.flatnonzero(top & (twin.dest == 4)).tolist())    # cell 3's top layer
    finally:
        context.close()
    _, pool, _ = take(twin, 0)
    assert pool is not None and "approx_targets" in [c["code"] for c in pool.caveats]


def test_past_f_without_a_window_over_the_area_the_shares_are_taken_where_the_run_is_now(make_twin, monkeypatch):
    """A tiled parent past F whose frame has no window over the test area: its export of the area is asked of its
    state now (paused at a later counted frame), and the frame work is done there, exactly: Go deeper's answer on
    proseg's count of that frame and the where measurement there, F's stored bytes, centres and pool kept, with the
    caveat targets_now."""
    twin = make_twin("now", tiled=True)
    add_frame(twin, 1)
    moved_on(twin, 1)
    asked = []
    real = DeeperWhere.measure

    def measure(self, run_id, seq, *args, **kwargs):
        asked.append(seq)
        return real(self, run_id, seq, *args, **kwargs)
    monkeypatch.setattr(DeeperWhere, "measure", measure)
    reasons, pool, fields = take(twin, 0)
    context = twin.pools.work["p1"].context
    assert (context.key[-2:], context.seq, context.f_seq) == ((1, "exact"), 1, SEQ)
    assert context.request["counts_file"] == str(twin.live / twin.run.frames[1]["counts_file"])
    assert set(asked) == {1} and [box for _, box, _ in twin.runs.asked] == [[X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL]]
    assert reasons == [] and pool.seq == SEQ and "targets_now" in [c["code"] for c in pool.caveats]


def ended_twin(make_twin, name: str, where: str, window_nx: int | None = None):
    """The tiled twin sampled on past F and ended there (state done: nothing can be asked of it, no export kept), F
    keeping the per-layer window of the view last shown over its first `window_nx` columns (every row; None: no
    window), with the where engine `where` (use_where)."""
    window = None
    if window_nx is not None:
        plane, _, _ = layout()
        window = ({"i0": 0, "j0": 0, "nx": window_nx, "ny": NY, "nz": NZ}, np.stack([plane[:, :window_nx]] * NZ))
    twin = make_twin(name, tiled=True, window=window)
    moved_on(twin, None)
    twin.meta["state"] = "done"
    use_where(twin, where)
    return twin


def codes_of(reasons) -> list[str]:
    return [r["code"] for r in reasons]


@pytest.mark.parametrize("where", where_engines())
def test_an_ended_parent_without_a_window_or_an_export_has_nothing_glowing(make_twin, where):
    """Ended past F, F without a window and no export of it kept: the area's basis is none, and Go deeper's where
    engine (the stand-in, or R5's own) places no cell either, so nothing glows: no_suspects, after the one ask of the
    parent's state now (which nothing answers)."""
    ended = ended_twin(make_twin, "ended", where)
    reasons, pool, _ = take(ended, 0)
    assert reasons == [P.reason("no_suspects")] and pool is None
    context = ended.pools.work["p1"].context
    assert context.key[-2:] == (SEQ, "none") and ended.pools.answers_kept[(*context.key, 0)].held == "none"
    assert [box for _, box, _ in ended.runs.asked] == [[X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL]]


@pytest.mark.parametrize("where", where_engines())
def test_an_ended_parent_s_window_over_part_of_the_area_places_the_shares_there(make_twin, where):
    """Ended past F (nothing can be asked of it), F keeping the per-layer window of the view last shown over columns
    0-99 of its grid: over cell 1 and its sources, not the whole test area. Go deeper's where engine (the stand-in,
    or R5's own) places cell 1's box from that window, so it glows; the area's shares come from the same window where
    it holds the area (approx_targets), a record outside it no cell's: each cell's are its label there at the record's
    own layer times its keep, the cells inside have their controls, matched as anywhere, all inside the window, and the
    test is offered."""
    window_nx, seed = 100, 1
    ended = ended_twin(make_twin, "window", where, window_nx)
    reasons, pool, _ = take(ended, seed)
    context = ended.pools.work["p1"].context
    assert context.key[-2:] == (SEQ, "approx") and ended.pools.answers_kept[(*context.key, seed)].held == "approx"
    assert context.basis.caveat == "approx_targets" and context.basis.window[0]["nx"] == window_nx
    weights = context.held[tuple(float(v) for v in AREA.box)]
    i, _, _, on = CF.observed_voxels(ended.records, ended.grid, ended.z0, ended.dz)
    within = np.flatnonzero(on & (i < window_nx))
    layers = context.basis.window[1]
    for c in range(NCELLS):
        idx, w = weights.of(c, context.q(c))
        want, how = held.weights(ended.records[within], c, ended.q(c), ended.grid, ended.z0, ended.dz,
                                 layers=(layers, 0, 0))
        assert how == "approx" and idx.tolist() == within[want > 0].tolist() and np.allclose(w, want[want > 0]), c
    assert weights.of(11, context.q(11))[0].size == 0 and weights.of(seed, context.q(seed))[0].size > 0
    assert reasons == [] and pool is not None and pool.cells[0] == seed
    kinds = [item.kind for item in pool.items]
    assert {k: kinds.count(k) > 0 for k in ("target", "matched", "uniform")} == \
        {"target": True, "matched": True, "uniform": True}
    controls = np.array([item.record for item in pool.items if item.kind != "target"])
    assert np.isin(controls, within).all()
    assert "approx_targets" in codes_of(pool.caveats)
    reply = ended.pools.preview("p1", seed, wait=30)
    assert (reply["offer"], reply["reasons"]) == (True, []) and "approx_targets" in codes_of(reply["caveats"])


@pytest.mark.parametrize("where", where_engines())
def test_a_cell_that_glows_with_no_control_to_judge_it_by_is_refused(make_twin, monkeypatch, where):
    """Ended past F, F without a window: the area's basis is none. But Go deeper still holds its export of cell 1's
    box at F (asked while the parent was held there), so its where engine (the stand-in, or R5's own) places that box
    exactly and the cell glows. No record of the area has a share, so no control can be drawn for any cell: under the
    rule "excess" (the suspects judged against their own RNA at the same spot) the test could never be judged, so it is
    refused (no_controls, never no_suspects), at once (no other cell worked out); the rule "suspects" (the suspects
    alone) takes the pool, with no control."""
    seed = 1
    ended = ended_twin(make_twin, "kept", where)
    box = [X0, Y0, X0 + 80.0, Y0 + NY * VOXEL]                # cell 1's reply box (columns 0-75) and more, not the area
    kept = ended.export(box, SEQ)
    ended.run.wheres = [{**kept, "path": str(ended.live / kept["file"])}]
    reasons, pool, fields = take(ended, seed)
    context = ended.pools.work["p1"].context
    answer = ended.pools.answers_kept[(*context.key, seed)]
    assert context.key[-2:] == (SEQ, "none") and answer.held == "exact" and answer.acting()
    assert (reasons, pool, fields["this_cell"]["suspects"] > 0) == ([P.reason("no_controls")], None, True)
    assert ended.pools.computed == 1 and not any(answer.matched.values()) and answer.uniform == ()
    reply = ended.pools.preview("p1", seed, wait=30)
    assert (reply["offer"], codes_of(reply["reasons"])) == (False, ["no_controls"])
    with pytest.raises(P.OutsideDomain) as refused:
        ended.pools.pool("p1", seed)
    assert refused.value.reasons == [P.reason("no_controls")]
    # Judged by the suspects alone: taken, with no control at all.
    monkeypatch.setitem(Q.CONSTANTS["paired"], "rule", "suspects")
    ended.pools.forget("p1")
    reasons, pool, _ = take(ended, seed)
    assert reasons == [] and {item.kind for item in pool.items} == {"target"}


def test_a_window_counts_only_where_it_holds_the_whole_area(make_twin):
    """`_window_over`: a tiled frame's window places shares over a test area only when it holds every column and row of
    the area on the frame's grid (one short on any side: none), or, asked for part of it (`part`: the parent cannot be
    asked), when it holds any column and row of it (none past its last one on any side); `on_grid` cuts an area to the
    frame's grid (none when they do not meet)."""
    plane, _, _ = layout()
    layers = np.stack([plane[5:35, 20:60]] * NZ)
    twin = make_twin("partial", tiled=True, window=({"i0": 20, "j0": 5, "nx": 40, "ny": 30, "nz": NZ}, layers))
    frame = twin.run.frames[SEQ]
    inside = (X0 + 20, Y0 + 5, X0 + 59.5, Y0 + 34.5)                      # columns 20-59, rows 5-34: the window's own
    found = P._window_over(twin.run, frame, inside)
    assert found is not None and found[0]["i0"] == 20 and np.array_equal(found[1], layers)
    for box in ((X0 + 19.5, Y0 + 5, X0 + 59.5, Y0 + 34.5), (X0 + 20, Y0 + 5, X0 + 60.0, Y0 + 34.5),
                (X0 + 20, Y0 + 4.9, X0 + 59.5, Y0 + 34.5), (X0 + 20, Y0 + 5, X0 + 59.5, Y0 + 35.0)):
        assert P._window_over(twin.run, frame, box) is None, box
        found = P._window_over(twin.run, frame, box, part=True)          # each holds part of it
        assert found is not None and np.array_equal(found[1], layers), box
    # A column or a row of it, at any side: part of it; past its last one: none.
    for box in ((X0, Y0, X0 + 20.5, Y0 + NY), (X0 + 59.5, Y0, X0 + NX, Y0 + 5.5), (X0, Y0 + 34.5, X0 + 20.5, Y0 + NY)):
        assert P._window_over(twin.run, frame, box, part=True) is not None, box
    for box in ((X0, Y0, X0 + 19.9, Y0 + NY), (X0 + 60.0, Y0, X0 + NX, Y0 + NY), (X0, Y0, X0 + NX, Y0 + 4.9),
                (X0, Y0 + 35.0, X0 + NX, Y0 + NY)):
        assert P._window_over(twin.run, frame, box, part=True) is None, box
    assert P._window_over(make_twin("bare", tiled=True).run, frame, inside) is None          # no window at all
    assert P._window_over(twin.run, frame, (X0, Y0, X0 + NX, Y0 + NY), part=True) is not None       # the whole grid
    assert P._window_over(make_twin("bare2", tiled=True).run, frame, inside, part=True) is None
    assert P.on_grid((0.0, 0.0, 1e4, 1e4), frame) == (X0, Y0, X0 + NX * VOXEL, Y0 + NY * VOXEL)
    assert P.on_grid((X0 + 10, Y0 + 10, X0 + 20, Y0 + 20), frame) == (X0 + 10, Y0 + 10, X0 + 20, Y0 + 20)
    assert P.on_grid((0.0, 0.0, X0, 1e4), frame) is None


# ---- its parts ---------------------------------------------------------------------------------------------------
def test_record_keys_are_uniques():
    """`_RecordIndex.find` gives np.unique's lowest index and count of every record's 16 bytes (`_Context`'s keys) and
    a plain grouping's: many records at one place, records held several times, x of +0.0 and -0.0 (other bytes, other
    records) and NaN; nothing asked, nothing found; a record the parent does not hold, at a place it holds or not,
    refused."""
    rng = np.random.default_rng(3)
    n = 6000
    records = np.zeros(n, RECORD)
    few = rng.random(n) < 0.7                     # most at a few places, depths and genes: many per place
    records["x"] = np.where(few, rng.choice(np.array([0.0, -0.0, 1.5, 2.25, 4096.5, np.nan], np.float32), n),
                            rng.uniform(-1e4, 1e4, n))
    records["y"] = np.where(few, rng.choice(np.array([0.0, 3.5, -7.25], np.float32), n), rng.uniform(-1e4, 1e4, n))
    records["z"] = np.where(few, rng.choice(np.array([0.125, 0.5], np.float32), n), rng.random(n))
    records["gene"] = np.where(few, rng.integers(0, 3, n), rng.integers(0, 400, n))
    records[rng.integers(0, n, 200)] = records[rng.integers(0, n, 200)].copy()        # held more than once
    signed = np.zeros(2, RECORD)
    signed["x"], signed["y"], signed["z"], signed["gene"] = [0.0, -0.0], [9.5, 9.5], [0.25, 0.25], [7, 7]
    records = np.concatenate([records, signed])
    index = P._RecordIndex(records)
    keys, first, copies = np.unique(P._void(records), return_index=True, return_counts=True)
    at = np.searchsorted(keys, P._void(records))
    got_first, got_copies = index.find(records)
    assert got_first.dtype == got_copies.dtype == np.int64
    assert got_first.tolist() == first[at].tolist() and got_copies.tolist() == copies[at].tolist()
    groups: dict = {}
    for i in range(len(records)):
        groups.setdefault(records[i:i + 1].tobytes(), []).append(i)
    assert [groups[records[i:i + 1].tobytes()][0] for i in range(len(records))] == got_first.tolist()
    assert [len(groups[records[i:i + 1].tobytes()]) for i in range(len(records))] == got_copies.tolist()
    assert got_copies.max() > 2 and len(groups) < len(records)
    # +0.0 and -0.0: one place to the eye, two records here as in np.unique (other bytes, other x bits).
    assert [v.tolist() for v in index.find(records[-2:])] == [[n, n + 1], [1, 1]]
    nan = np.flatnonzero(np.isnan(records["x"]))
    found = index.find(records[nan[:3]])
    assert found[1].min() >= 2 and np.isnan(records["x"][found[0]]).all() and (found[0] <= nan[:3]).all()
    assert [v.tolist() for v in index.find(records[:0])] == [[], []]
    for change in ({"gene": 999}, {"z": 0.75}, {"x": np.nextafter(np.float32(1.5), np.float32(2))}):
        absent = np.zeros(1, RECORD)
        absent["x"], absent["y"], absent["z"], absent["gene"] = 1.5, 3.5, 0.125, 1
        for key, value in change.items():
            absent[key] = value
        with pytest.raises(RuntimeError, match="^a suspect's transcript is not among your run's transcripts$"):
            index.find(np.concatenate([records[5:6], absent]))


def test_tiled_start_classes_by_column(make_twin):
    """`start_at` is prior.tiles' label and permille at each point's column, the worker's own (its tile_lookup on the
    prior's tiles, in float32: a point on a column's left edge as float32 has it is in that column, where float64
    would put it in the one before); -1 and 0 off the grid (either side; its last, partial tiles inside); permille 0
    where no cell started; and the detail twin's voxel lookup says the same. Planted points of every start class."""
    tiled, dense = make_twin("tiled", tiled=True), make_twin("dense", tiled=False)
    context, twin = tiled.context(), dense.context()
    try:
        # (x, y) µm, the cell and giver the class is asked for, the class and permille wanted.
        planted = [((X0 + 15.5, Y0 + 15.5), 0, 1, "OWN", 800), ((X0 + 25.5, Y0 + 21.5), 0, 1, "GIVER_STRONG", 650),
                   ((X0 + 27.5, Y0 + 21.5), 0, 1, "GIVER_WEAK", 649),
                   ((X0 + 29.5, Y0 + 21.5), 0, 1, "GIVER_STRONG", 800),
                   ((PX0 + 26.0, Y0 + 21.5), 0, 1, "GIVER_STRONG", 650),        # column 13's left edge (float64: 12)
                   ((X0 + 50.5, Y0 + 23.5), 5, 2, "GIVER_WEAK", 500), ((X0 + 22.5, Y0 + 27.5), 4, 1, "OTHER", 800),
                   ((X0 + 86.5, Y0 + 15.5), 8, 6, "NONE", 0),                    # no cell, a stray permille: 0
                   ((X0 + 202.5, Y0 + 20.5), 11, 10, "NONE", 0), ((PX0 - 0.5, Y0 + 15.5), 0, 1, "NONE", 0),
                   ((X0 + 15.5, PY0 + PNY * PVOXEL), 3, 7, "NONE", 0), ((X0 + 15.5, PY0 - 0.25), 0, 1, "NONE", 0),
                   ((PX0 + PNX * PVOXEL - 0.5, PY0 + PNY * PVOXEL - 0.5), 11, 10, "NONE", 0)]
        x = np.array([p[0][0] for p in planted], np.float32)
        y = np.array([p[0][1] for p in planted], np.float32)
        label, permille = context.start_at(x, y, np.full(x.size, 0.5, np.float32))
        assert label.dtype == permille.dtype == np.int64
        assert [P.start_class(int(lab), int(per), cell, giver, STRONG)
                for lab, per, (_, cell, giver, _, _) in zip(label, permille, planted, strict=True)] == \
            [p[3] for p in planted]
        assert permille.tolist() == [p[4] for p in planted]
        assert label[8:12].tolist() == [-1] * 4 and label[7] == label[12] == 0 and layout()[2][8, 43] == 300
        assert math.floor((float(x[4]) - PX0) / PVOXEL) == 12 and layout()[1][10, 12] == 1      # float64: its own
        # Anywhere, on and off the grid: the worker's own lookup on the prior's tiles, and the detail twin's.
        rng = np.random.default_rng(4)
        x = np.concatenate([x, rng.uniform(PX0 - 20, PX0 + PNX * PVOXEL + 20, 3000).astype(np.float32)])
        y = np.concatenate([y, rng.uniform(PY0 - 20, PY0 + PNY * PVOXEL + 20, 3000).astype(np.float32)])
        z = rng.uniform(0.0, 1.0, x.size).astype(np.float32)
        label, permille = context.start_at(x, y, z)
        want = {}
        for name in ("label", "permille"):
            header, present, tiles = W.plane_tiles(str(tiled.live / "prior.tiles"), name)
            ids, offsets = W.tile_lookup(np, x, y, PX0, PY0, PVOXEL, PNX, PNY, header["tile"], header["tiles_x"],
                                         header["tiles_y"])
            stack = np.zeros((1 + len(present), header["tile"], header["tile"]), np.uint32)
            stack[1:] = tiles[:len(present)]
            want[name] = W.tile_labels(np, stack, present, header["tiles_x"] * header["tiles_y"], ids, offsets)
        inside = W.grid_columns(np, x, y, PX0, PY0, PVOXEL, PNX, PNY)[2]
        assert 0 < inside.sum() < x.size
        assert label.tolist() == np.where(inside, want["label"].astype(np.int64), -1).tolist()
        assert permille.tolist() == np.where(label > 0, want["permille"].astype(np.int64), 0).tolist()
        assert [v.tolist() for v in twin.start_at(x, y, z)] == [label.tolist(), permille.tolist()]
        assert [v.tolist() for v in context.start_at(x[:0], y[:0], z[:0])] == [[], []]
    finally:
        context.close()
        twin.close()


def test_tiled_near_is_the_where_replys_near(make_twin):
    """`near` on a cell's records at F is the where reply's near flag for the same transcripts (every gene asked as
    the giver's), and every owned column measured says the same: for a giver whose box is wide and short (its axes
    swapped, the flags differ), one with a fragment 200 columns from its centre (its box found by scanning every
    tile), one whose box takes in the cell's own columns. A giver that owns nothing: its footprint has no column, so
    nothing is near, not even at F's corner, where that empty footprint is placed."""
    tiled = make_twin("tiled", tiled=True)
    plane, _, _ = layout()
    context = tiled.context()
    try:
        stats = DeeperWhere._cellstats(tiled.run, tiled.run.frames[SEQ], NCELLS)
        centre = math.floor((float(stats["cx"][10]) - X0) / VOXEL)
        half = max(SEED_HALF, 2 * math.ceil(math.sqrt(int(stats["area"][10]))))
        assert 209 > centre + SEED_GROW[-1] * half                       # past the widest window around its centre
        for cell, giver in ((3, 7), (11, 10), (0, 1)):
            own = tiled.owned(cell)
            x, y = context.records["x"][own], context.records["y"][own]
            got = context.near(giver, x, y)
            want = footprint_near(plane, giver + 1, x, y)
            assert got.tolist() == want.tolist() and want.any() and not want.all(), (cell, giver)
            out, held = tiled.where.measure("p1", SEQ, cell, [giver], list(RUN_GENES), [],
                                            givers=dict.fromkeys(RUN_GENES, giver))
            mine = np.asarray(out["points"]["owner"]) == 1
            assert not out["truncated"]
            assert sorted(v.tobytes() for v in P._void(held[mine])) == \
                sorted(v.tobytes() for v in P._void(context.records[own]))
            assert context.near(giver, held["x"][mine], held["y"][mine]).tolist() == \
                (np.asarray(out["points"]["near"])[mine] == 1).tolist()
            mask, c0, r0 = context.feet[giver]
            rows, cols = np.nonzero(plane == giver + 1)
            assert (c0, r0, c0 + mask.shape[1], r0 + mask.shape[0]) == \
                (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1) and mask.sum() == rows.size
        mask, c0, r0 = context.feet[7]
        assert mask.shape == (3, 60) and context.feet[10][0].shape == (30, 210)
        own = tiled.owned(3)                                                            # cell 3's
        x, y = context.records["x"][own], context.records["y"][own]
        swapped = near(x.astype(np.float64), y.astype(np.float64), mask.T, r0, c0, X0, Y0, VOXEL)
        assert swapped.tolist() != footprint_near(plane, 8, x, y).tolist()
        own = tiled.owned(11)                                                           # cell 11's: near the fragment
        close = context.near(10, context.records["x"][own], context.records["y"][own])
        assert close.any() and (context.records["x"][own][close] >= X0 + 204).all()
        assert not context.near(NCELLS + 5, context.records["x"][own], context.records["y"][own]).any()
        assert not context.near(NCELLS + 5, np.array([X0 + 0.5]), np.array([Y0 + 0.5])).any()     # F's corner column
        assert not context.feet[NCELLS + 5][0].any()
    finally:
        context.close()


# ---- what stays as it was ----------------------------------------------------------------------------------------
def test_a_tiled_parent_is_tested_with_the_whole_sample_caveat(make_twin, monkeypatch):
    """A whole-sample run is tested now: its own reasons are the caveat whole_sample alone (a paired start in a
    square, paused at its end); a preview offers the test with it, and pool() takes the pool through the tiled frame
    context, in the 500 µm square the region gives around the cell: its dense twin's pool cell for cell, but the
    caveat. A whole sample by its cells (H11: qc_constants.run_mode "whole", WHOLE_CELLS or more, which the twin is
    too small to hold: it stands for one here); the same tiled run of its own few cells is a tiled box."""
    from sopa_companion.engine import qc_constants
    tiled, dense = make_twin("tiled", tiled=True), make_twin("dense", tiled=False)
    tiled.meta["region"] = {"kind": "full", "bbox": list(REGION)}     # the region a whole-sample run has
    assert [r["code"] for r in P.run_reasons(tiled.meta, tiled.run.info, max_um=1200.0)] == ["tiled_box"]
    real = qc_constants.run_mode
    monkeypatch.setattr(qc_constants, "run_mode",
                        lambda meta, info: "whole" if meta.get("mode") == "tiled" else real(meta, info))
    assert P.run_reasons(tiled.meta, tiled.run.info, max_um=1200.0) == [P.reason("whole_sample")]
    assert P.run_reasons(dense.meta, dense.run.info, max_um=1200.0) == []
    offered = []
    for cell in tiled.judged_mixed():
        try:
            pool = tiled.pools.pool("p1", cell)
        except P.OutsideDomain as refused:                         # the cell's own refusals, as its twin's
            with pytest.raises(P.OutsideDomain) as twin_refused:
                dense.pools.pool("p1", cell)
            assert refused.reasons == twin_refused.value.reasons, cell
            reply = tiled.pools.preview("p1", cell)
            assert (reply["offer"], reply["reasons"], reply["caveats"][:1]) == \
                (False, refused.reasons, [P.reason("whole_sample")]), cell
            continue
        twin = dense.pools.pool("p1", cell)
        assert (pool.area.rule, pool.row, pool.start) == ("square", ("paired", "square"), "paired")
        assert pool.caveats == (P.reason("whole_sample"), *twin.caveats)
        assert P.Pool(**{**pool.__dict__, "caveats": ()}) == P.Pool(**{**twin.__dict__, "caveats": ()}), cell
        reply = tiled.pools.preview("p1", cell)                    # the pool kept: at once
        assert (reply["state"], reply["offer"], reply["reasons"], reply["waits"]) == ("ready", True, [], [])
        assert reply["caveats"] == list(pool.caveats) and reply["area"] == pool.area.public()
        offered.append(cell)
    assert offered and type(tiled.pools.work["p1"].context) is P._TiledContext


def test_an_unknown_frame_format_is_refused(make_twin):
    """A frame of another format reaches `_Context`, which refuses it in its own words; `_TiledContext` takes tiled
    frames only, with starting cells stored per column."""
    tiled, dense = make_twin("tiled", tiled=True), make_twin("dense", tiled=False)
    tiled.run.frames[SEQ]["format"] = "other"
    with pytest.raises(RuntimeError, match="^only a detail run's frames can be tested$"):
        tiled.pools.answers("p1", [0])
    facts = dense.pools._facts_at_end(dense.run)
    with pytest.raises(RuntimeError, match="^only a whole-sample run's frames are read by their columns$"):
        P._TiledContext(facts.key(), dense.run, facts, P._Basis("none", SEQ, facts.frame))
    tiled.run.frames[SEQ]["format"] = "tiles"
    tiled.run.info["prior"] = dict(dense.run.info["prior"])
    facts = tiled.pools._facts_at_end(tiled.run)
    with pytest.raises(RuntimeError, match="^this run's starting cells are not stored per column$"):
        P._TiledContext(facts.key(), tiled.run, facts, P._Basis("none", SEQ, facts.frame))


# ---- the containers ----------------------------------------------------------------------------------------------
def test_plt1_writer_is_the_suites(tmp_path):
    """write_plt1 at 4 layers without a window writes test_analysis.write_container's container (the headers equal as
    JSON, ours compact as proseg-live's; the same blobs: empty tiles, partial edge tiles at their own size); a
    container with a window has its planes read unchanged by tiles.Container and the worker, and its window as
    written."""
    import zstandard
    rng = np.random.default_rng(11)
    tile, nx, ny = 8, 30, 21                                  # partial edge tiles both ways
    consensus = rng.integers(1, 50, (ny, nx)).astype(np.uint32)
    consensus[0:8, 0:16], consensus[16:, 24:] = 0, 0          # two empty tiles and an empty edge tile
    last = rng.integers(0, 9, (ny, nx)).astype(np.uint32)
    planes = {"consensus": consensus, "last_change": last}
    write_plt1(tmp_path / "ours.tiles", planes, tile, 4)
    write_container(tmp_path / "suite.tiles", planes, tile)
    (ours, ours_index, ours_raw), (suite, suite_index, suite_raw) = (W.read_container(str(tmp_path / name))
                                                                     for name in ("ours.tiles", "suite.tiles"))
    assert ours == suite and b" " not in ours_raw[8:8 + int.from_bytes(ours_raw[4:8], "little")]

    def blob(raw, entry) -> bytes:
        return raw[int(entry["offset"]):int(entry["offset"]) + int(entry["length"])]
    assert [blob(ours_raw, e) for e in ours_index] == [blob(suite_raw, e) for e in suite_index]
    assert (ours_index["length"] == 0).sum() == 4                       # three tiles and the window
    edge = (2 * ours["tiles_x"] + 2) * 2                                # consensus of tile (row 2, column 2): 5 x 8
    assert len(zstandard.ZstdDecompressor().decompress(blob(ours_raw, ours_index[edge]))) == 5 * 8 * 4
    layers = rng.integers(0, 50, (3, 5, 7)).astype(np.uint32)
    info = {"i0": 4, "j0": 2, "nx": 7, "ny": 5, "nz": 3}
    write_plt1(tmp_path / "window.tiles", planes, tile, 3, (info, layers))
    container = Container(tmp_path / "window.tiles")
    assert container.header["nz"] == 3 and container.header["window"] == info
    assert np.array_equal(container.full("consensus"), consensus)
    assert np.array_equal(container.full("last_change"), last)
    assert np.array_equal(W.read_tiled_consensus(str(tmp_path / "window.tiles")), consensus)
    found_info, found_layers = container.window()
    assert found_info == info and np.array_equal(found_layers, layers)
