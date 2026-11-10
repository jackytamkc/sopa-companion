"""Go deeper: where it spills (the reply's version 3: DeeperWhere.reply, below; docs/ARCHITECTURE.md).

For one cell of a frame (or of the start line) and the neighbours its Go deeper window shows as sources: every
transcript of the genes asked inside the box around them, with proseg's share of it for the cell and the cell whose
outline it lies in, and per gene a measurement of where proseg's count of it sits: at the edge the cell shares with the
neighbour it seems to come from, spread through the cell, or as common in the cells around (a layer of expression
that no outline removes); plus proseg's count for the cell of its own genes lying inside each source's outline ("into").
Each point says whether it
lies near: a "from" gene's transcript proseg counts for the cell (or inside its outline) within NEAR_UM of its giver, an
own gene's inside a source within NEAR_UM of the cell (the page places the model's per-gene split by these, a rough
guide per transcript).

Proseg's share of each transcript for the cell, w (0 to 1, engine/held.py), is proseg's own count placed: exact from
proseg's where export of the box (the paused or held state's, asked for here, or the counted frame's own while the run
plays: the analysis scheduler asks proseg for one of the box the window follows with each counted frame), else
approximate from that frame's per-layer labels (a dense frame's, or a tiled frame's window when it covers the box) times
the cell's keep e / n in the frame's counts (f / n on a frame read on the count proseg saves: one counted for a pause,
the hold, a finish or the end, the reply's `count` "sampled"), else none (a past tiled frame outside its window, or a
frame whose counts are gone): never from a frame's flat 2-D projection. The reply says which (`held`). Over a box holding
the cell's pools the cell's exact w add up to its count in that frame, gene by gene, so the counts here (`cell_total`,
each gene's `in_cell`, near the giver and the rest of the cell's RNA, and `into`) are proseg's own, as sums of w; the
cells around are weighed the same way (proseg's share of each transcript of the box for every other cell, held.others,
per area of their own footprints in the box). With nothing held ("none") those sums are of nothing, so no gene gets a
word: they never stand for proseg's count. A frame's approximate reply kept stands once its counts are gone (pruned from the counts spool): that frame's
layers and keep do not change. The start line's w come from proseg's first frame (its layers and counts: proseg's count
at iteration 0).

Which cell's outline a transcript lies in (`owner`) stays positional, for drawing, the near flags and the areas: the
label at its measured position in the frame's per-layer labels (basis "voxel", detail runs), in its 2-D projection
(basis "column", whole-sample runs), or in the prior's labels for the start line (basis "start"), by the worker's own
grid_columns and voxel_lookup. The model splits a cell's RNA per gene, never per transcript: positions are a separate,
model-free measurement beside its estimate.

Plain numpy: the box's transcripts come from the run's spatial index (engine/transcript_index.py); the labels from the
frame's own file (a dense frame or prior is decoded whole, a tiled one composed over the box, its cells found from the
frame's cell statistics). Distances are exact in the plane: from a transcript's measured x, y to the nearest point of a
cell's footprint, its owned columns taken as squares (in the voxel basis the columns where it owns any layer); 0 inside.
"""
from __future__ import annotations

from collections import OrderedDict
import gzip
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import NamedTuple

import numpy as np

from ..analysis import counts_file
from ..analysis.worker import grid_columns, voxel_lookup
from . import held as H
from .runs import ACTIVE, RunError
from .tiles import Container
from .transcript_index import RECORD

WHERE_V = 3                 # 3: each point's proseg share "w" and the reply's "held" (2 added the points' "near")
MARGIN_UM = 20.0            # the box: the cell's and the sources' footprints grown by this, clipped to the run's raster
NEAR_UM = 2.0               # "near": within this distance in the plane of a footprint
POINTS_MAX = 30_000         # transcripts sent at most, the cell's and the sources' first ("truncated" when cut)
SOURCES_MAX = 10            # Go deeper's neighbours (worker DEEP_K)
NAMES_MAX = 64              # gene names asked, both lists together
BOX_MAX_UM = 1000.0         # a box wider than this: the cell and these sources lie too far apart to show together
QUERY_MAX = 4_000_000       # transcripts the box may hold (a dense sample's densest 400 µm window holds well under 1 M)
# The words (a rough guide; every count is proseg's, a sum of w). The counts were first guesses in transcripts (5 and
# 3), carried over to proseg's count (2026-10-31) at what that many transcripts are worth in it (5 are 4.0-4.4, 3 near
# the giver 2.4-2.6); the ratios are unitless and kept (EDGE_SHARE and EDGE_RATIO read about the same on both counts;
# LAYER_RATIO's reads about 0.9 of the outline's, as the RNA proseg moves in is counted in the cell, not around).
FEW_IN_CELL = 4             # under this many in proseg's count of the cell, or placed of it where its shares are placed
                            # approximately: "few" (too few to tell; the page's WHERE_FEW)
# "edge": >= 2.5 near the giver, at least a third of the gene's count in the cell, and a near share >= 2 x that of the
# rest of the cell's RNA (every gene but the "from" genes asked: a baseline every cell has, as most cells hold many
# transcripts near their edge), both shares smoothed by one so a handful never decides it.
EDGE_NEAR, EDGE_SHARE, EDGE_RATIO = 2.5, 1 / 3, 2.0
LAYER_RATIO = 0.5           # "layer": not at the edge, and in the cells around (proseg's count of them per area of
                            # their own footprints in the box) at least half as dense as in the cell's count
SEED_HALF = 40              # a tiled frame's cells are first looked for this many columns around their centres ...
SEED_GROW = (1, 4)          # ... then in a window this many times as wide; still not whole: every tile is scanned
REPLIES_KEPT_BYTES = 32 << 20    # replies kept (their JSON), most recently used first
DENSE_KEPT = 2              # decoded dense label volumes (or tiled windows) kept (a 1,200 µm detail frame's are 23 MB)
PRIORS_KEPT = 4             # tiled priors whose cells' areas and centres are kept
NAMES_KEPT = 8              # runs whose gene names are kept
COUNTS_KEPT = 2             # frames' counts kept decoded (a whole sample's are tens of MB)
BOXES_KEPT = 256            # replies' boxes kept (a reply kept is found again without its frame's labels)
EXPORT_WAIT_S = 2.0         # paused or held at the frame: how long proseg is given to export the box (served at once)
FOLLOW_PAD_UM = 5.0         # the box the scheduler asks a playing run to export: the reply's, this much wider on each
                            # side, so the next counted frame's export still holds it as the cell moves a little
_WHOLE = re.compile(r"[0-9]{1,9}")
_MISSING = object()


class _Plan(NamedTuple):
    """What a reply is measured over: its box (None: neither the cell nor a source holds anything in that frame), and
    where proseg's share comes from (held, held_seq; the where export's record when exact; the frame whose counts and
    layers give the keep and the 3-D owner: the asked frame, or the first frame for the start line), and which of that
    frame's counts the keep reads (`count`: "averaged", e, or "sampled", f, the count proseg saves)."""
    path: object                # the asked frame's labels file (or the prior's)
    box: list | None
    held: str
    held_seq: int | None
    where: dict | None
    frame: dict | None
    frame_path: Path | None
    counts: Path | None
    count: str = "averaged"


def parse(seq, cell, sources, genes, own) -> tuple:
    """The query (text as it comes) -> (seq: n | "start", cell, [sources], [genes], [own]); RunError 422 for anything
    malformed: a source twice, the cell among its own sources, more than SOURCES_MAX sources or NAMES_MAX names. Cells
    are label - 1. A name repeated, or in both lists, counts once (as "from")."""
    seq = (seq or "").strip()
    if seq != "start":
        if not _WHOLE.fullmatch(seq):
            raise RunError('seq must be a frame number or "start"')
        seq = int(seq)
    cell = (cell or "").strip()
    if not _WHOLE.fullmatch(cell):
        raise RunError("cell must be a cell number (label - 1)")
    cell = int(cell)
    found = [v.strip() for v in (sources or "").split(",") if v.strip()]
    if not all(_WHOLE.fullmatch(v) for v in found):
        raise RunError("sources must be cell numbers (label - 1), separated by commas")
    found = [int(v) for v in found]
    if len(found) > SOURCES_MAX:
        raise RunError(f"At most {SOURCES_MAX} sources")
    if len(set(found)) != len(found) or cell in found:
        raise RunError("Each source once, and never the cell itself")
    genes = [v.strip() for v in (genes or "").split(",") if v.strip()]
    own = [v.strip() for v in (own or "").split(",") if v.strip()]
    if len(genes) + len(own) > NAMES_MAX:
        raise RunError(f"At most {NAMES_MAX} gene names")
    return seq, cell, found, genes, own


def givers_of(result) -> dict | None:
    """{gene name: its main giving neighbour (label - 1)} from a kept Go deeper answer (v1 rows [gene, tx, own, nb,
    tag, giver?], giver a row of `neighbours`); None without a ready answer."""
    if not isinstance(result, dict) or result.get("state") != "ready":
        return None
    neighbours, out = result.get("neighbours") or [], {}
    for row in result.get("genes") or []:
        if isinstance(row, list) and len(row) >= 6 and isinstance(row[5], int) and 0 <= row[5] < len(neighbours):
            cell = (neighbours[row[5]] or {}).get("cell")
            if isinstance(cell, int) and not isinstance(cell, bool):
                out[str(row[0])] = cell
    return out


def _decode(path) -> np.ndarray:
    """A dense frame or prior file (zstd, or gzip for runs made by older versions) as u32 values."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(raw)
    else:
        import zstandard
        data = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    return np.frombuffer(data, "<u4")


def _tiles(container: Container, plane: str):
    """Every non-empty tile of a plane as (row, column, array), decoded without the shared tile cache: whole scans
    (a tiled prior's cells, a cell not found near its centre) would push out the tiles the viewer is using."""
    p, h = container.plane_index(plane), container.header
    for ty in range(h["tiles_y"]):
        for tx in range(h["tiles_x"]):
            entry = (ty * h["tiles_x"] + tx) * container.nplanes + p
            if int(container.index[entry]["length"]) == 0:
                continue
            r, c = ty * container.tile, tx * container.tile
            rows, cols = min(container.tile, container.ny - r), min(container.tile, container.nx - c)
            yield r, c, np.frombuffer(container._blob(entry), "<u4").reshape(rows, cols)


def _bounds(rows, cols, found, wanted, out: dict, r0: int = 0, c0: int = 0):
    """Grow out[label] (column box [c0, r0, c1, r1)) by the cells' voxels at rows/cols (found: their labels)."""
    for label in wanted:
        hit = found == label
        if hit.any():
            box = [int(cols[hit].min()) + c0, int(rows[hit].min()) + r0, int(cols[hit].max()) + c0 + 1,
                   int(rows[hit].max()) + r0 + 1]
            old = out.get(label)
            out[label] = box if old is None else [min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]),
                                                  max(old[3], box[3])]


class Dense:
    """Per-layer labels held whole: a detail run's frame (basis "voxel") or its prior (the start line)."""

    def __init__(self, basis: str, layers: np.ndarray, x0: float, y0: float, voxel: float, z_origin: float,
                 z_spacing: float):
        self.basis, self.layers = basis, layers
        self.nz, self.ny, self.nx = layers.shape
        self.x0, self.y0, self.voxel, self.z_origin, self.z_spacing = x0, y0, voxel, z_origin, z_spacing

    def boxes(self, wanted: list[int]) -> dict:
        """{label: column box of every voxel it owns, or None}: the whole volume, exactly."""
        flat = self.layers.reshape(-1)
        hit = np.flatnonzero(np.isin(flat, np.asarray(wanted, np.uint32), kind="table"))
        plane = hit % (self.ny * self.nx)
        out: dict = {}
        _bounds(plane // self.nx, plane % self.nx, flat[hit], wanted, out)
        return {label: out.get(label) for label in wanted}

    def window(self, c0, r0, c1, r1) -> np.ndarray:
        return self.layers[:, r0:r1, c0:c1]

    def lookup(self, records, window, c0, r0) -> np.ndarray:
        """Each transcript's label: its voxel, with the worker's own voxel_lookup (0 outside the grid)."""
        index = voxel_lookup(np, records["x"], records["y"], records["z"], self.x0, self.y0, self.voxel, self.nx,
                             self.ny, self.nz, self.z_origin, self.z_spacing)
        flat, out = self.layers.reshape(-1), np.zeros(len(records), np.uint32)
        inside = index < flat.size
        out[inside] = flat[index[inside]]
        return out


class Plane:
    """A tiled run's 2-D labels (basis "column": a frame's projection; the start line: prior.tiles' "label"), composed
    over the box only. `area`, `cx`, `cy` (per cell; NaN centre without a column): the frame's cell statistics, or the
    prior's worked out once, or None (every tile is then scanned)."""

    def __init__(self, basis: str, container: Container, plane: str, x0: float, y0: float, voxel: float,
                 stats: dict | None):
        self.basis, self.container, self.plane, self.stats = basis, container, plane, stats
        self.nx, self.ny, self.nz = container.nx, container.ny, 1
        self.x0, self.y0, self.voxel = x0, y0, voxel

    def boxes(self, wanted: list[int]) -> dict:
        """{label: column box of its columns, or None}: looked for around the cell's centre (a window, then one four
        times as wide) and taken once the window holds as many columns as the cell statistics count; a cell not
        complete by then (a far fragment, no statistics) is found by scanning every tile."""
        out = {label: None for label in wanted}
        todo = list(wanted)
        if self.stats is not None:
            area, cx, cy = self.stats["area"], self.stats["cx"], self.stats["cy"]
            todo = [label for label in wanted if label <= len(area) and area[label - 1] > 0]
            seeds = [label for label in todo if math.isfinite(cx[label - 1]) and math.isfinite(cy[label - 1])]
            if seeds:
                cols = [(float(cx[label - 1]) - self.x0) / self.voxel for label in seeds]
                rows = [(float(cy[label - 1]) - self.y0) / self.voxel for label in seeds]
                half = max(SEED_HALF, 2 * math.ceil(math.sqrt(max(int(area[label - 1]) for label in seeds))))
                for grow in SEED_GROW:
                    c0, r0 = max(0, math.floor(min(cols)) - grow * half), max(0, math.floor(min(rows)) - grow * half)
                    c1 = min(self.nx, math.floor(max(cols)) + grow * half + 1)
                    r1 = min(self.ny, math.floor(max(rows)) + grow * half + 1)
                    if c1 <= c0 or r1 <= r0:
                        break
                    labels = self.container.compose(self.plane, c0, r0, c1, r1)
                    for label in [label for label in todo if label in seeds]:
                        rows_, cols_ = np.nonzero(labels == label)
                        if rows_.size == int(area[label - 1]):          # every column of the cell is in the window
                            _bounds(rows_, cols_, np.full(rows_.size, label), [label], out, r0, c0)
                            todo.remove(label)
                    if not any(label in seeds for label in todo):
                        break
        if todo:
            for r, c, tile in _tiles(self.container, self.plane):
                hit = np.isin(tile, np.asarray(todo, np.uint32), kind="table")
                if hit.any():
                    rows_, cols_ = np.nonzero(hit)
                    _bounds(rows_, cols_, tile[rows_, cols_], todo, out, r, c)
        return out

    def window(self, c0, r0, c1, r1) -> np.ndarray:
        return self.container.compose(self.plane, c0, r0, c1, r1)[None]

    def lookup(self, records, window, c0, r0) -> np.ndarray:
        """Each transcript's label: its column, with the worker's own grid_columns (0 outside the grid)."""
        i, j, inside = grid_columns(np, records["x"], records["y"], self.x0, self.y0, self.voxel, self.nx, self.ny)
        i, j = i.astype(np.int64) - c0, j.astype(np.int64) - r0
        ok = inside & (i >= 0) & (i < window.shape[2]) & (j >= 0) & (j < window.shape[1])
        if (inside & ~ok).any():
            raise RuntimeError("a transcript of the box lies outside its labels")
        out = np.zeros(len(records), np.uint32)
        out[ok] = window[0, j[ok], i[ok]]
        return out


def plane_stats(container: Container, plane: str, ncells: int, x0: float, y0: float, voxel: float) -> dict:
    """Each cell's columns and centre in a whole tiled plane (a tiled prior: no cell statistics), as proseg-live works
    out a frame's: centre = origin + (mean column + 0.5) x voxel, NaN without a column."""
    area, sx, sy = (np.zeros(ncells + 1, np.float64) for _ in range(3))
    for r, c, tile in _tiles(container, plane):
        rows, cols = np.nonzero((tile > 0) & (tile <= ncells))
        labels = tile[rows, cols].astype(np.int64)
        area += np.bincount(labels, minlength=ncells + 1)
        sx += np.bincount(labels, weights=cols + c, minlength=ncells + 1)
        sy += np.bincount(labels, weights=rows + r, minlength=ncells + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx, cy = x0 + (sx / area + 0.5) * voxel, y0 + (sy / area + 0.5) * voxel
    return {"area": area[1:].astype(np.int64), "cx": cx[1:], "cy": cy[1:]}


def near(px, py, mask: np.ndarray, c0: int, r0: int, x0: float, y0: float, voxel: float) -> np.ndarray:
    """Whether each point (µm) lies within NEAR_UM of a footprint: mask[row - r0, col - c0] true where it owns the
    column (col, row), the square [x0 + col v, x0 + (col + 1) v) x [...]. Exact: the squares close enough are each
    measured."""
    hit = np.zeros(px.size, bool)
    if not px.size:
        return hit
    # Columns up to `reach` away can hold a square within NEAR_UM (a point on its column's left edge is exactly 2 µm
    # from the square three 1 µm columns to its left); only offsets whose nearest possible distance is within it.
    reach = math.floor(NEAR_UM / voxel) + 1
    col, row = np.floor((px - x0) / voxel).astype(np.int64), np.floor((py - y0) / voxel).astype(np.int64)
    h, w = mask.shape
    for dj in range(-reach, reach + 1):
        r = row + dj
        low = y0 + r * voxel
        dy = np.maximum(np.maximum(low - py, py - (low + voxel)), 0.0)
        for di in range(-reach, reach + 1):
            if (max(abs(di) - 1, 0) ** 2 + max(abs(dj) - 1, 0) ** 2) * voxel * voxel > NEAR_UM * NEAR_UM + 1e-9:
                continue
            c = col + di
            low = x0 + c * voxel
            dx = np.maximum(np.maximum(low - px, px - (low + voxel)), 0.0)
            inside = (r >= r0) & (r < r0 + h) & (c >= c0) & (c < c0 + w)
            owned = np.zeros(px.size, bool)
            owned[inside] = mask[r[inside] - r0, c[inside] - c0]
            hit |= owned & (dx * dx + dy * dy <= NEAR_UM * NEAR_UM + 1e-9)
    return hit


def _per100(n, area):
    return round(100.0 * float(n) / area, 2) if area > 0 else None


class DeeperWhere:
    """GET /runs/{id}/qc/deeper/where (server/app.py): `reply` -> the reply's JSON (bytes). Thread-safe; replies,
    decoded dense volumes, frames' counts and tiled priors' cells are kept (bounded), and the files are read outside the
    lock. `analysis`: the analysis scheduler (engine/analysis.py follow_where: the box a playing run's counted frames
    export), or None."""

    def __init__(self, runs, analysis=None):
        self.runs, self.analysis = runs, analysis
        self.lock = threading.Lock()
        self.replies: OrderedDict = OrderedDict()   # (run, seq, cell, sources, genes, own, givers, held, held_seq)
                                                    # -> JSON
        self.kept_bytes = 0
        self.dense: OrderedDict = OrderedDict()     # (file, mtime, size) -> layers [nz, ny, nx]; (file, mtime, size,
                                                    # "window") -> a tiled frame's window (layers, i0, j0)
        self.priors: OrderedDict = OrderedDict()    # (file, mtime) -> plane_stats of a tiled prior
        self.genes: OrderedDict = OrderedDict()     # (genes.txt, mtime) -> _names()
        self.counts: OrderedDict = OrderedDict()    # (file, mtime, size) -> counts_file.Counts
        self.boxes: OrderedDict = OrderedDict()     # (run, seq, cell, sources) -> the reply's box (or None)
        self.followed: dict = {}                    # run -> (cell, seq) of the reply whose box a playing run exports

    def reply(self, run_id: str, seq, cell: int, sources: list[int], genes: list[str], own: list[str],
              givers: dict | None = None) -> bytes:
        """The reply's JSON for parse()'s values. `givers`: givers_of() the window's kept answer, which names each
        gene's giver; None: each gene's giver is the source holding it most densely (giver_by "density"). 404 not a
        proseg run, no such frame or cell (or no start line); 409 not written yet (a run still going); 422 the cell
        and its sources lie too far apart. Kept by what proseg's share came from: a frame's exact reply stays its
        reply; an approximate one (or none) is worked out again once proseg's export of the box is there; and once
        nothing is held for that frame (its counts pruned from the spool) its approximate reply kept stands: the
        frame's layers and keep do not change. Each reply reads the count its frame is read on now (`count`:
        counts_file.count_of of its counts line, "sampled" for a frame proseg counted for a pause, the hold, a finish or
        the end: q = f / n then, the count proseg saves; the start line's "averaged"), and is kept by it: a frame counted
        again is measured again."""
        run = self.runs.get(run_id)
        record, path = self._check(run, seq, cell, sources)
        count = "averaged" if record is None else counts_file.count_of(record.get("counts"))
        base = (run_id, seq, cell, tuple(sources), tuple(genes), tuple(own),
                None if givers is None else tuple(givers.get(name) for name in genes), count)
        if seq != "start":
            with self.lock:
                hit = self.replies.get(base + ("exact", seq))
                if hit is not None:
                    self.replies.move_to_end(base + ("exact", seq))
                    box = self.boxes.get((run_id, seq, cell, tuple(sources)), _MISSING)
            if hit is not None:
                if box is not _MISSING:
                    self._follow(run, seq, cell, box)
                return hit
        plan = self._plan(run, seq, record, path, cell, sources, count)
        if seq != "start":                              # the start line's footprints are not the live cell's
            self._follow(run, seq, cell, plan.box)
        hit = self._kept(base, plan.held, plan.held_seq)
        if hit is not None:
            return hit
        began = time.perf_counter()
        out, _ = self._work(run, seq, record, plan, cell, sources, genes, own, givers)
        out["ms"] = round((time.perf_counter() - began) * 1e3, 1)
        data = json.dumps(out, separators=(",", ":"), allow_nan=False).encode()
        key = base + (out["held"], out["held_seq"])          # what it came from in the end (an export gone: approx)
        with self.lock:
            if out["held"] == "none":                       # the counts gone meanwhile: an approximate reply stands
                hit = self.replies.get(base + ("approx", None))
                if hit is not None:
                    self.replies.move_to_end(base + ("approx", None))
                    return hit
            if key not in self.replies:
                self.replies[key] = data
                self.kept_bytes += len(data)
                while self.kept_bytes > REPLIES_KEPT_BYTES and len(self.replies) > 1:
                    self.kept_bytes -= len(self.replies.popitem(last=False)[1])
        return data

    def _kept(self, base: tuple, held: str, held_seq) -> bytes | None:
        """The reply kept of what proseg's share comes from now (`reply`'s key `base` + (held, held_seq)), None without
        one; with nothing held ("none") the frame's approximate reply kept comes first."""
        keys = [base + ("approx", None)] if held == "none" else []
        keys.append(base + (held, held_seq))
        with self.lock:
            for key in keys:
                hit = self.replies.get(key)
                if hit is not None:
                    self.replies.move_to_end(key)
                    return hit
        return None

    def measure(self, run_id: str, seq, cell: int, sources: list[int], genes: list[str], own: list[str],
                givers: dict | None = None, count: str = "averaged") -> tuple[dict, np.ndarray]:
        """`reply`'s measurement for the test of a setting (engine/qc_pool.py): (the reply's object without "ms", the
        run's records (engine/transcript_index.py RECORD: x, y, z, gene as the run holds them) of its points, one per
        point in the points' order), on `count` of the frame's counts ("averaged": the test's targets read e / n, as
        they were measured; "sampled": f / n, as Go deeper shows a frame read on the count proseg saves). Worked out
        afresh every time, never kept among the replies (the decoded frames, counts, gene names and the run's index are
        shared with them); the same refusals as `reply` (404, 409, 422)."""
        run = self.runs.get(run_id)
        record, path = self._check(run, seq, cell, sources)
        plan = self._plan(run, seq, record, path, cell, sources, count)
        return self._work(run, seq, record, plan, cell, sources, genes, own, givers)

    # ---- what is asked, and whether it is there -----------------------------------------------------------------
    def check(self, run_id: str, seq, cell: int, sources: list[int]):
        """reply's refusals alone (404, 409), for the route to make before it asks anything else of the run."""
        self._check(self.runs.get(run_id), seq, cell, sources)

    def footprints(self, run_id: str, seq, cells: list[int]) -> dict:
        """{cell (label - 1): the box [x0, y0, x1, y1] µm of its footprint in that frame (the columns where it owns
        any layer, as a reply's box starts from), None where it owns nothing}: the test's corner view centres on the
        cell's (engine/qc_test.py). reply's refusals (404, 409)."""
        run = self.runs.get(run_id)
        record, path = self._check(run, seq, cells[0], list(cells[1:]))
        labels = self._labels(run, seq, record, path, int(run.info["cells"]))
        found, v = labels.boxes([c + 1 for c in cells]), labels.voxel
        return {c: None if found[c + 1] is None else [labels.x0 + found[c + 1][0] * v, labels.y0 + found[c + 1][1] * v,
                                                     labels.x0 + found[c + 1][2] * v, labels.y0 + found[c + 1][3] * v]
                for c in cells}

    def _check(self, run, seq, cell: int, sources: list[int]):
        """(frame record or None, its labels' file) after the refusals: 404 never there, 409 not yet (run going)."""
        if run.meta.get("kind") in ("stain", "patches"):
            raise RunError("Not a proseg run", 404)
        going = run.meta.get("state") in ACTIVE         # what is missing may still come

        def missing(later: str, never: str):
            return RunError(later, 409) if going else RunError(never, 404)
        info = run.info
        if not info:
            raise missing("This run has not written its first frame yet", "This run has no frames")
        cells = int(info.get("cells") or 0)
        if not all(0 <= c < cells for c in (cell, *sources)):
            raise RunError("Unknown cell", 404)
        if seq == "start":
            prior, chosen = info.get("prior") or {}, run.meta.get("prior")
            if not prior.get("file") or (isinstance(chosen, dict) and not chosen.get("cells")):
                raise RunError("This run has no start line (it started from nuclei only)", 404)
            if not re.fullmatch(r"prior\.(bin\.zst|tiles)", str(prior["file"])):
                raise RunError("Malformed prior", 500)
            record, path = None, run.live / prior["file"]
        else:
            with run.lock:
                record = dict(run.frames[seq]) if seq < len(run.frames) else None
            if record is None:
                raise missing("This frame is not written yet", "Unknown frame")
            path = self.runs.frame_path(run.meta["id"], seq)
        if not path.exists():
            raise missing("This frame is not written yet", "This frame's file is missing")
        if not (run.live / "transcripts.bin.gz").exists():
            raise missing("This run's transcripts are not written yet",
                          "This run has no exported transcripts (made before live analysis existed)")
        return record, path

    def _labels(self, run, seq, record, path, ncells: int):
        info = run.info
        z = info.get("z") or {}
        z_origin, z_spacing = float(z.get("grid_origin", 0.0)), float(z.get("grid_spacing", 1.0))
        if seq == "start":
            prior = info["prior"]
            x0, y0, voxel = float(prior["x0"]), float(prior["y0"]), float(prior["voxel"])
            if prior.get("format") == "tiles":
                container = Container(path)
                with self.lock:
                    stats = self.priors.get(container.key)
                if stats is None:
                    stats = plane_stats(container, "label", ncells, x0, y0, voxel)
                    with self.lock:
                        self.priors[container.key] = stats
                        while len(self.priors) > PRIORS_KEPT:
                            self.priors.popitem(last=False)
                return Plane("start", container, "label", x0, y0, voxel, stats)
            layers = self._volume(path, 0, int(prior.get("nz", 1)), int(prior["ny"]), int(prior["nx"]))
            return Dense("start", layers, x0, y0, voxel, z_origin, z_spacing)
        x0, y0, voxel = float(record["x0"]), float(record["y0"]), float(record["voxel"])
        if record.get("format") == "tiles":
            return Plane("column", Container(path), "consensus", x0, y0, voxel, self._cellstats(run, record, ncells))
        layers = self._volume(path, 2, int(record["nz"]), int(record["ny"]), int(record["nx"]))
        return Dense("voxel", layers, x0, y0, voxel, z_origin, z_spacing)

    def _volume(self, path, skip: int, nz: int, ny: int, nx: int) -> np.ndarray:
        """A dense file's per-layer labels [nz, ny, nx] (after `skip` planes: a frame's consensus and last change)."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self.lock:
            layers = self.dense.get(key)
            if layers is not None:
                self.dense.move_to_end(key)
                return layers
        values = _decode(path)
        plane = ny * nx
        if values.size < (skip + nz) * plane:
            raise RunError("The frame's file is incomplete", 500)
        layers = values[skip * plane:(skip + nz) * plane].reshape(nz, ny, nx)
        with self.lock:
            self.dense[key] = layers
            while len(self.dense) > DENSE_KEPT:
                self.dense.popitem(last=False)
        return layers

    @staticmethod
    def _cellstats(run, record: dict, ncells: int) -> dict | None:
        """A tiled frame's cell statistics (areas and centres), None without them (every tile is then scanned)."""
        name = record.get("cells_file")
        if not name or not re.fullmatch(r"frames/[0-9]{6,9}\.cells\.zst", str(name)):
            return None
        try:
            stats = _decode(run.live / name)
        except OSError:
            return None
        if stats.size != 7 * ncells:
            return None
        stats = stats.reshape(7, ncells)
        return {"area": stats[0].astype(np.int64), "cx": stats[5].view("<f4").astype(np.float64),
                "cy": stats[6].view("<f4").astype(np.float64)}

    def _names(self, run) -> dict:
        """{"n": the run's genes, "index": {gene name: its run gene indices}} from genes.txt (line g names gene g)."""
        path = run.live / "genes.txt"
        key = (str(path), path.stat().st_mtime_ns)
        with self.lock:
            found = self.genes.get(key)
        if found is None:
            names = path.read_text().split("\n")[:-1]
            found = {}
            for g, name in enumerate(names):
                found.setdefault(name, []).append(g)
            found = {"n": len(names), "index": {k: np.asarray(v, np.int64) for k, v in found.items()}}
            with self.lock:
                self.genes[key] = found
                while len(self.genes) > NAMES_KEPT:
                    self.genes.popitem(last=False)
        return found

    # ---- where proseg's share comes from ----------------------------------------------------------------------------
    def _plan(self, run, seq, record, path, cell: int, sources: list[int], count: str = "averaged") -> _Plan:
        """The reply's box (kept per frame, cell and sources: a kept reply is found again without decoding the frame's
        labels; 422 when wider than BOX_MAX_UM) and where proseg's share of its transcripts comes from (`_source`), on
        `count` of the frame's counts."""
        key = (run.meta["id"], seq, cell, tuple(sources))
        with self.lock:
            box = self.boxes.get(key, _MISSING)
            if box is not _MISSING:
                self.boxes.move_to_end(key)
        if box is _MISSING:
            box = self._box(self._labels(run, seq, record, path, int(run.info["cells"])), cell, sources)
            with self.lock:
                self.boxes[key] = box
                while len(self.boxes) > BOXES_KEPT:
                    self.boxes.popitem(last=False)
        if box is not None and max(box[2] - box[0], box[3] - box[1]) > BOX_MAX_UM:
            raise RunError("The cell and these neighbours lie too far apart to show together")
        return self._source(run, seq, record, path, box)._replace(count=count)

    @staticmethod
    def _box(labels, cell: int, sources: list[int]) -> list | None:
        """The cell's and the sources' footprints grown by MARGIN_UM, clipped to the raster; None when none of them
        owns anything in that frame."""
        found = labels.boxes([cell + 1, *(s + 1 for s in sources)])
        present = [b for b in found.values() if b is not None]
        if not present:
            return None
        v, x0, y0 = labels.voxel, labels.x0, labels.y0
        c0, r0 = min(b[0] for b in present), min(b[1] for b in present)
        c1, r1 = max(b[2] for b in present), max(b[3] for b in present)
        return [max(x0, x0 + c0 * v - MARGIN_UM), max(y0, y0 + r0 * v - MARGIN_UM),
                min(x0 + labels.nx * v, x0 + c1 * v + MARGIN_UM), min(y0 + labels.ny * v, y0 + r1 * v + MARGIN_UM)]

    def _source(self, run, seq, record, path, box) -> _Plan:
        """Where proseg's share of the box's transcripts comes from: "exact", proseg's where export of that frame
        holding the box (kept, or asked for now while the run is paused or held at that frame); "approx", that frame's
        per-layer labels cover the box (a dense frame's whole layers, a tiled frame's window); "none" otherwise, and
        always without that frame's counts (the cell's keep). The start line's share is its first frame's (approx or
        none)."""
        if seq == "start":
            with run.lock:
                frame = dict(run.frames[0]) if run.frames else None
            try:
                frame_path = None if frame is None else self.runs.frame_path(run.meta["id"], 0)
            except RunError:
                frame_path = None
        else:
            frame, frame_path = record, path
        counts = self._counts_path(run, frame)
        plan = _Plan(path, box, "none", None, None, frame, frame_path, counts)
        if box is None or counts is None or frame_path is None:
            return plan
        if seq != "start":
            meta = self._export(run, seq, box)
            if meta is not None:
                return plan._replace(held="exact", held_seq=int(meta["seq"]), where=meta)
        if self._layers_cover(frame, frame_path, box):
            return plan._replace(held="approx")
        return plan

    @staticmethod
    def _counts_path(run, frame: dict | None) -> Path | None:
        """The frame's counts file (its counts line's `file`: in the run's frames, or an absolute path in the counts
        spool), None when the frame has none (not counted, a run made before proseg's counts were kept) or it is
        gone (pruned from the spool)."""
        name = (frame or {}).get("counts_file")
        if not isinstance(name, str) or not name:
            return None
        found = run.live / name
        return found if found.is_file() else None

    def _export(self, run, seq: int, box) -> dict | None:
        """Proseg's where export of frame `seq` holding the box: one kept (engine/runs.py where_export), or, while the
        run is paused or held at that frame, one asked for now (request_where: the paused state serves it at once)."""
        run_id = run.meta["id"]
        kept = getattr(self.runs, "where_export", None)
        meta = kept(run_id, seq, box) if kept is not None else None
        if meta is None and self._paused_at(run, seq):
            ask = getattr(self.runs, "request_where", None)
            meta = ask(run_id, box, wait_s=EXPORT_WAIT_S) if ask is not None else None
            if meta is not None and (meta.get("seq") != seq or not H.covers(meta, box)):
                meta = None
        return meta

    @staticmethod
    def _paused_at(run, seq) -> bool:
        """Whether proseg is paused (or held) at frame `seq`: its ack (engine/runs.py Run.ack) says so."""
        with run.lock:
            ack = getattr(run, "ack", None)
            ack = dict(ack) if isinstance(ack, dict) else {}
        return ack.get("state") == "paused" and ack.get("seq") == seq

    @staticmethod
    def _layers_cover(frame: dict, frame_path: Path, box) -> bool:
        """Whether the frame's per-layer labels hold every transcript of the box (a column and a row to spare): a
        dense frame's always; a tiled frame's only inside its window (none: its 2-D projection never stands in)."""
        if frame.get("format") != "tiles":
            return frame_path.is_file()
        try:
            window = Container(frame_path).header.get("window")
        except (OSError, ValueError):
            return False
        if not window:
            return False
        x0, y0, v = float(frame["x0"]), float(frame["y0"]), float(frame["voxel"])
        nx, ny = int(frame["nx"]), int(frame["ny"])
        c0, c1 = max(0, math.floor((box[0] - x0) / v) - 1), min(nx - 1, math.floor((box[2] - x0) / v) + 1)
        r0, r1 = max(0, math.floor((box[1] - y0) / v) - 1), min(ny - 1, math.floor((box[3] - y0) / v) + 1)
        i0, j0 = int(window["i0"]), int(window["j0"])
        return i0 <= c0 and i0 + int(window["nx"]) - 1 >= c1 and j0 <= r0 and j0 + int(window["ny"]) - 1 >= r1

    def _counts(self, path: Path, ncells: int, ngenes: int):
        """A frame's counts read whole (counts_file.read: checked against the run's cells and genes), kept; None when
        it cannot be read (gone meanwhile, or not this run's)."""
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self.lock:
            found = self.counts.get(key)
            if found is not None:
                self.counts.move_to_end(key)
                return found
        try:
            found = counts_file.read(path, ncells, ngenes)
        except (OSError, counts_file.CountsError):
            return None
        with self.lock:
            self.counts[key] = found
            while len(self.counts) > COUNTS_KEPT:
                self.counts.popitem(last=False)
        return found

    def _frame_layers(self, plan: _Plan):
        """(per-layer labels, first column, first row) of the plan's frame: a dense frame's whole layers, a tiled
        frame's window; None without them."""
        frame = plan.frame
        if frame.get("format") != "tiles":
            return self._volume(plan.frame_path, 2, int(frame["nz"]), int(frame["ny"]), int(frame["nx"])), 0, 0
        stat = plan.frame_path.stat()
        key = (str(plan.frame_path), stat.st_mtime_ns, stat.st_size, "window")
        with self.lock:
            found = self.dense.get(key)
            if found is not None:
                self.dense.move_to_end(key)
                return found
        window = Container(plan.frame_path).window()
        if window is None:
            return None
        info, layers = window
        found = (layers, int(info["i0"]), int(info["j0"]))
        with self.lock:
            self.dense[key] = found
            while len(self.dense) > DENSE_KEPT:
                self.dense.popitem(last=False)
        return found

    def _weights(self, run, plan: _Plan, records, cells: list[int], ngenes: int):
        """(w [len(cells), records] float64, held, the other cells' share of each record or None, the cell's own count
        per run gene or None): proseg's share of each record for each of `cells` (the cell, then its sources) by the
        plan (engine/held.py weights_many), and every other cell's at once (held.others: the cells around), on the plan's
        count (q = e / n, or f / n on a frame read on the count proseg saves); an export gone meanwhile is the frame's
        layers' share instead, counts gone none (and no other share or count)."""
        none = np.zeros((len(cells), len(records)), np.float64)
        if plan.held == "none" or not len(records):
            return none, plan.held, None, None
        counts = self._counts(plan.counts, int(run.info["cells"]), ngenes)
        if counts is None:
            return none, "none", None, None
        which = counts_file.ARRAY.get(plan.count, "e")
        try:
            q = H.keeps(counts, cells, which)
        except counts_file.CountsError:                 # that count is not in the frame's file: nothing held
            return none, "none", None, None
        own = counts_file.rows(counts, cells[:1], (which,))[which][0].astype(np.float64)
        z = run.info.get("z") or {}
        z_origin, z_spacing = float(z.get("grid_origin", 0.0)), float(z.get("grid_spacing", 1.0))
        grid = {key: plan.frame[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
        if plan.held == "exact":
            try:
                export = H.read_where(plan.where)
            except (OSError, ValueError):
                export = None
            if export is not None:
                w, held = H.weights_many(records, cells, q, grid, z_origin, z_spacing, export=export)
                around, _ = H.others(records, cells, counts, grid, z_origin, z_spacing, which, export=export)
                return w, held, around, own
            if not self._layers_cover(plan.frame, plan.frame_path, plan.box):
                return none, "none", None, None
        try:
            layers = self._frame_layers(plan)
            w, held = H.weights_many(records, cells, q, grid, z_origin, z_spacing, layers=layers)
            around = None if held == "none" else \
                H.others(records, cells, counts, grid, z_origin, z_spacing, which, layers=layers)[0]
        except (OSError, ValueError, RunError):         # the frame's file gone or not its record's: nothing held
            return none, "none", None, None
        return (w, held, None, None) if held == "none" else (w, held, around, own)

    def _follow(self, run, seq: int, cell: int, box):
        """While the run plays, the analysis scheduler asks proseg to export the box of frame `seq`'s reply
        (FOLLOW_PAD_UM wider) with each counted frame, so the window following the cell gets exact shares frame after
        frame. Only a frame's reply moves it (never the start line's), and for the same cell never back to an older
        frame's (a view scrubbed back while the run plays): the box stays the live cell's. Kept or worked out, each
        reply of a playing run's frame hands its box over again. Paused or held, the reply of the frame proseg rests on
        arms its box the same way and is kept as the one followed, so the first counted frame after Resume carries the
        box's export (the scheduler keeps both leases while the run rests: engine/analysis.py _expire_follows); a reply
        of another frame then arms nothing new. Done, finishing or stopped: no box."""
        follow = getattr(self.analysis, "follow_where", None)
        if follow is None:
            return
        run_id = run.meta["id"]
        with run.lock:
            ack = getattr(run, "ack", None) or {}
            state = run.meta.get("state")
            playing = state == "running" and ack.get("state") != "paused"
            resting = state in ("running", "paused") and ack.get("state") == "paused"
            frames = len(run.frames)
        here = resting and ack.get("seq") == seq        # the frame proseg is paused or held at
        if resting and not here:
            return
        with self.lock:
            if playing:
                last = self.followed.get(run_id)
                if last is not None and last[0] == cell and seq < last[1] < frames:   # (frames: a restart begins anew)
                    return
                self.followed[run_id] = (cell, seq)
            elif here:
                self.followed[run_id] = (cell, seq)
            else:
                self.followed.pop(run_id, None)
        wanted = None if not (playing or here) or box is None else \
            [box[0] - FOLLOW_PAD_UM, box[1] - FOLLOW_PAD_UM, box[2] + FOLLOW_PAD_UM, box[3] + FOLLOW_PAD_UM]
        try:
            follow(run_id, wanted)
        except Exception:                               # the scheduler's bookkeeping never fails a reply
            pass

    # ---- the measurement ------------------------------------------------------------------------------------------
    def _work(self, run, seq, record, plan: _Plan, cell: int, sources: list[int], genes: list[str], own: list[str],
              givers: dict | None) -> tuple[dict, np.ndarray]:
        """(the reply's object without "ms", the records of its points: `measure`)."""
        info = run.info
        ncells = int(info["cells"])
        known = self._names(run)
        names, roles = [], []
        for name, role in [(n, "from") for n in genes] + [(n, "own") for n in own]:
            if name in known["index"] and name not in names:
                names.append(name)
                roles.append(role)
        G, K = len(names), len(sources)
        lut = np.full(known["n"] + 1, -1, np.int64)          # run gene -> its row in `names`, -1 not asked
        for g, name in enumerate(names):
            lut[known["index"][name]] = g
        labels = self._labels(run, seq, record, plan.path, ncells)
        cell_label, source_labels = cell + 1, [s + 1 for s in sources]
        out = {"v": WHERE_V, "run": run.meta["id"], "seq": seq,
               "iter": 0 if record is None else int(record.get("iter") or 0), "cell": cell, "sources": list(sources),
               "basis": labels.basis, "held": plan.held, "held_seq": plan.held_seq, "count": plan.count,
               "near_um": NEAR_UM, "box": None, "genes": names, "roles": roles}
        v, x0, y0 = labels.voxel, labels.x0, labels.y0
        box = plan.box
        if box is not None:
            # An exact share needs every transcript of the box's buckets (|B|): the box's columns and rows, one more
            # on each side; the points are those inside the box itself, in the run's order.
            exact = plan.held == "exact"
            asked = H.query_box(plan.where, box) if exact else box
            found, _ = self.runs.transcript_index(run.meta["id"]).query(asked, QUERY_MAX)
            if found is None:
                raise RunError(f"More than {QUERY_MAX:,} transcripts around the cell and these neighbours")
            if exact:
                in_box = (found["x"] >= box[0]) & (found["x"] < box[2]) & (found["y"] >= box[1]) & (found["y"] < box[3])
                records = found[in_box]
            else:
                in_box, records = None, found
            pad = math.floor(NEAR_UM / v) + 3           # past the box: rounding at its edge, and the near squares
            wc0, wr0 = max(0, math.floor((box[0] - x0) / v) - pad), max(0, math.floor((box[1] - y0) / v) - pad)
            wc1 = min(labels.nx, math.ceil((box[2] - x0) / v) + pad)
            wr1 = min(labels.ny, math.ceil((box[3] - y0) / v) + pad)
            window = labels.window(wc0, wr0, wc1, wr1)
            label = labels.lookup(records, window, wc0, wr0)
            out["box"] = [round(float(b), 3) for b in box]
            shares, held, others, own_count = self._weights(run, plan, found if exact else records, [cell, *sources],
                                                            known["n"])
            if in_box is not None:
                shares = shares[:, in_box]
                others = None if others is None else others[in_box]
            out["held"], out["held_seq"] = held, plan.held_seq if held == "exact" else None
        else:                                           # neither the cell nor a source owns anything in this frame
            box, wc0, wr0, others, own_count = None, 0, 0, None, None
            records, label, window = np.zeros(0, RECORD), np.zeros(0, np.uint32), np.zeros((labels.nz, 0, 0), np.uint32)
            shares = np.zeros((1 + K, 0), np.float64)
        w = shares[0]                                   # proseg's share of each point for the cell
        owner = np.zeros(len(records), np.uint8)       # 0 no cell, 1 the cell, 2 + k sources[k], 255 another cell
        owner[(label > 0) & (label <= ncells)] = 255
        owner[label == cell_label] = 1
        for k, s in enumerate(source_labels):
            owner[label == s] = 2 + k
        row = lut[np.minimum(records["gene"].astype(np.int64), known["n"])]   # each transcript's gene in `names`
        # Footprints in the plane (the columns where a cell owns any layer) and areas (owned voxels x v^2 / layers:
        # the area a flat cell of that volume covers, so the cell, its sources and the cells around share one unit).
        # The cells around: every other cell's voxels whose column lies in the box (its centre), never the gaps between
        # cells nor the box's empty edges.
        footprint = {s: (window == s).any(axis=0) for s in (cell_label, *source_labels)}
        area = {s: float((window == s).sum()) * v * v / window.shape[0] for s in (cell_label, *source_labels)}
        cell_area, source_area = area[cell_label], [area[s] for s in source_labels]
        around_area = 0.0
        if box is not None and window.size:
            cx = x0 + (np.arange(wc0, wc0 + window.shape[2]) + 0.5) * v
            cy = y0 + (np.arange(wr0, wr0 + window.shape[1]) + 0.5) * v
            columns = ((cy >= box[1]) & (cy < box[3]))[:, None] & ((cx >= box[0]) & (cx < box[2]))[None, :]
            theirs = (window > 0) & (window <= ncells) & ~np.isin(window, [cell_label, *source_labels])
            around_area = float((theirs & columns[None]).sum()) * v * v / window.shape[0]
        out.update(cell_total=round(float(w.sum()), 3), cell_area_um2=round(cell_area, 2),
                   source_area_um2=[round(a, 2) for a in source_area])
        px, py = records["x"].astype(np.float64), records["y"].astype(np.float64)
        asked = row >= 0

        def held_by(share):                             # proseg's count of each gene asked: its shares summed
            return np.bincount(row[asked], weights=share[asked], minlength=G) if G else np.zeros(0)
        in_cell = held_by(w)
        # The cells around: proseg's count of the box's transcripts for every cell but the cell and its sources
        # (held.others), summed over the box as the cell's own count and the givers' are, per area of their own
        # footprints in the box: "as common around" compares proseg's count with proseg's count, each per the area of
        # the cells it belongs to (background and the gaps between cells in neither). Nothing held: nothing around.
        around = np.zeros(G) if others is None else held_by(others)
        kept = [held_by(shares[1 + k]) for k in range(K)]    # each source's own count of the genes (its density)
        # The cell's own genes inside each source's outline (into): proseg's count of them for the cell there.
        inside = [held_by(np.where(owner == 2 + k, w, 0.0)) for k in range(K)]
        own_row = np.array([r == "own" for r in roles], bool)

        # Each gene's giver: the one the window's answer names (when among the sources), else the source holding it
        # most densely in proseg's count.
        giver, giver_by = [None] * G, [None] * G
        for g in range(G):
            if roles[g] != "from":
                continue
            if givers is not None:
                named = givers.get(names[g])
                if named in sources:
                    giver[g], giver_by[g] = sources.index(named), "result"
                continue
            dense = [kept[k][g] / source_area[k] if source_area[k] > 0 and kept[k][g] > 0 else 0.0 for k in range(K)]
            if dense and max(dense) > 0:
                giver[g], giver_by[g] = int(np.argmax(dense)), "density"

        # The cell's RNA: every transcript proseg counts for it (w > 0), and those inside its outline (w 0 or not:
        # they are measured for the near flags, and add nothing to the sums).
        mine = np.flatnonzero((w > 0) | (owner == 1))
        rows_mine, w_mine = row[mine], w[mine]
        # The baseline: the rest of the cell's RNA, every gene but the "from" genes asked (own genes and genes not asked).
        rest = (rows_mine < 0) | np.isin(rows_mine, np.flatnonzero(own_row))     # (no gene asked: every one)
        # Each transcript's near flag (the points' "near"): set below for a "from" gene's of the cell near its giver
        # and an own gene's in a source near the cell, from the same measurements as the sums.
        close_by = np.zeros(len(records), np.uint8)
        near_giver, own_near = {}, {}
        for k in sorted({k for k in giver if k is not None}):
            close = near(px[mine], py[mine], footprint[source_labels[k]], wc0, wr0, x0, y0, v)
            known_gene = close & (rows_mine >= 0)
            near_giver[k] = np.bincount(rows_mine[known_gene], weights=w_mine[known_gene], minlength=G) if G \
                else np.zeros(0)
            own_near[k] = [float(w_mine[close & rest].sum()), float(w_mine[rest].sum())]
            close_by[mine[close & np.isin(rows_mine, [g for g in range(G) if giver[g] == k])]] = 1
        stats = []
        measured = out["held"] != "none"                # nothing held: the sums are of nothing, so no word
        placed = out["held"] == "approx"                # placed by the frame's layers: the sums are not proseg's count
        for g in range(G):
            k = giver[g]
            n, close = float(in_cell[g]), None if k is None else float(near_giver[k][g])
            # Proseg's own count of the gene for the cell, from the frame's counts (the exact sum of w where the shares
            # are exact; where they are placed approximately, the sum of w misses RNA proseg moved in): "few" reads it,
            # and, placed approximately, the sum of w too: the other words are judged on what was placed, so under
            # FEW_IN_CELL placed is too few to tell where they sit, however many proseg counts.
            count = None if own_count is None else float(own_count[known["index"][names[g]]].sum())
            cell_d = n / cell_area if cell_area > 0 else None
            around_d = around[g] / around_area if around_area > 0 else None
            if not measured:
                word = None
            elif (n if count is None else count) < FEW_IN_CELL or (placed and round(n, 3) < FEW_IN_CELL):
                word = "few"
            elif k is not None and close >= EDGE_NEAR and close >= EDGE_SHARE * n and \
                    close * (own_near[k][1] + 2) >= EDGE_RATIO * (own_near[k][0] + 1) * n:
                word = "edge"
            elif around_d is not None and cell_d is not None and around_d >= LAYER_RATIO * cell_d:
                word = "layer"
            else:                                       # "spread" only where the edge was measured: no giver, no word
                word = "spread" if k is not None else None
            stats.append({"gene": names[g], "role": roles[g], "giver": k, "giver_by": giver_by[g],
                          "in_cell": round(n, 3), "count": None if count is None else round(count, 3),
                          "near_giver": None if close is None else round(close, 3),
                          "own_near": None if k is None else [round(x, 3) for x in own_near[k]],
                          "density": {"cell": _per100(n, cell_area),
                                      "giver": None if k is None else _per100(kept[k][g], source_area[k]),
                                      "around": _per100(around[g], around_area)},
                          "word": word})
        out["gene_stats"] = stats

        # Into each source: proseg's count for the cell of its own genes lying inside the source's outline (Σ w), of
        # them within NEAR_UM of the cell, and of the other genes' transcripts there [near the cell, inside]: the outline
        # only says where (on an approximate reply every w inside a source is 0: the page shows it on exact ones only).
        into = []
        for k in range(K):
            theirs = np.flatnonzero(owner == 2 + k)
            close = near(px[theirs], py[theirs], footprint[cell_label], wc0, wr0, x0, y0, v)
            rows_, w_ = row[theirs], w[theirs]
            near_cell = np.bincount(rows_[close & (rows_ >= 0)], weights=w_[close & (rows_ >= 0)], minlength=G) \
                if G else []
            other = rows_ < 0
            baseline = [round(float(w_[close & other].sum()), 3), round(float(w_[other].sum()), 3)]
            close_by[theirs[close & np.isin(rows_, np.flatnonzero(own_row))]] = 1
            into += [{"source": k, "gene": names[g], "inside": round(float(inside[k][g]), 3),
                      "near_cell": round(float(near_cell[g]), 3), "source_own_near": list(baseline)}
                     for g in range(G) if roles[g] == "own"]
        out["into"] = into

        shown = np.flatnonzero(asked)
        out["truncated"] = bool(shown.size > POINTS_MAX)
        if out["truncated"]:                            # proseg's count of the cell first, then the cell's and the
            counted = shown[w[shown] > 0]               # sources' by their outlines, then the tissue around
            ours = shown[(w[shown] <= 0) & (owner[shown] >= 1) & (owner[shown] < 255)]
            tissue = shown[(w[shown] <= 0) & ((owner[shown] == 0) | (owner[shown] == 255))]
            take = counted[:POINTS_MAX]
            take = np.concatenate([take, ours[:POINTS_MAX - take.size]])
            shown = np.sort(np.concatenate([take, tissue[:POINTS_MAX - take.size]]))
        out["points"] = {"x": np.round(px[shown], 2).tolist(), "y": np.round(py[shown], 2).tolist(),
                         "z": np.round(records["z"][shown].astype(np.float64), 3).tolist(),
                         "gene": row[shown].tolist(), "owner": owner[shown].tolist(), "near": close_by[shown].tolist(),
                         "w": np.round(w[shown], 3).tolist()}
        return out, records[shown]
