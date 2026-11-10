"""Compare: the run's frame and up to three of the sample's other segmentations side by
side, each read by the sample's mixing model on the one area they all cover, each by its own count.

Each column's `count` says whose count it reads: the run's, proseg's own count of its frame (the frame
AnalysisManager.compare_frame gives: the nearest with proseg's count; "proseg"); a segmentation's, its own when a table
annotating it holds one (zarr_samples.count_source: a table carrying proseg's run record, "proseg"; Xenium's cell table,
"xenium"), else the transcripts inside its outlines under the run's own input rule ("outlines"). Proseg's count sets
background aside, so its columns read less mixed for that reason too: the page says so beside the counting.
A run has one comparison at a time: a newer request replaces it (its counting is stopped; a worker call it waits for is
let finish, and dropped). A comparison is keyed by everything that decides its numbers (`job_key`): the model and whether
it holds the details kinds need, the columns (each shapes element as it is on disk: its stamp; whose count it reads, and
that table's version), the run's frame (and whether its count holds proseg's own draw), the counting rule and the
constants below. Its steps (`step`):
  1. "outlines", "area": each column's cell centres (the run's from its frame's cell statistics: frame_centres; a shapes
     element's from its exported outlines: outline_centres), the TILE_UM tiles holding at least TILE_MIN_CELLS of each
     column's cells, their intersection (common_area: the area, its box said within the sample and the run: area_clip),
     and each column's cells whose centre lies in it; the run's frame's count is held beside the comparison from its
     start, before it waits for the run's previous comparison (one in the engine's spool can be cleared before the
     worker reads it: _hold);
  2. "counting": each shapes element's counts by the sopa helper (ZarrSamples.counts): its table's own (the outlines it
     names; one naming none of the area's fails its column: NOT_HELD), or its outlines' under the run's input rule; the
     whole element, or only the area's box when that covers less than WHOLE_SHARE of the element; cached per element
     stamp (and table version);
  3. "scoring": the analysis worker's stateless op "compare" (AnalysisManager.submit_compare; analysis/worker.py), which
     reads the run's column from proseg's count of its frame, scores each column's cells on the area (as named here)
     with the model and lays them all out in one UMAP. The columns ready first (the run's, those whose counts were
     cached) are scored at once, then all of them together once counted (one shared layout; `layout` counts the layouts
     so far, `layout_kind` says what the worker laid them out with: "umap" on the GPU and the CPU alike).
A column's state: waiting, counting, working (being scored, or counted and waiting for it), ready or failed.
A finished comparison stays beside the run (compare/<job>/: result.json and the worker's points files, the RESULTS_KEPT
newest), and the same request later is answered from it.
Each shapes element is compared at the version its key names (its stamp, pinned in the job's `versions`): its outlines
and its counts must both be of it. The stamp is read again around the outlines, after the counts, after a read or count
that failed (ZarrSamples refuses an export or counts of an element written over meanwhile) and before the comparison is
ready. An element written again meanwhile (a notebook, another engine) starts the comparison again at the versions then
on disk (RESTARTS times; it is never kept, its key naming the versions before); written again once more, that column
fails (WRITTEN_AGAIN). A counts row is scored only as the outline at its position (the same id, its centre within the
outline's bounds or at its own centre: misfits), so a position is never taken for a cell of other outlines: rows that
misfit are left out, up to MISFITS_LEFT_OUT of them; more fail their column at once (DISAGREE): of the version pinned,
the same outlines and counts would be read again.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from ..analysis.counts_file import LIVE_COUNT, count_of
from ..analysis.worker import KIND_PALETTE
from .analysis import AnalysisError
from .proseg_options import OPTIONS
from .runs import RunError
from .samples import SampleError
from .zarr_samples import STAMP_RETRY_S, UNREAD, HelperFailed

TILE_UM = 100.0              # the area's tiles (µm)
TILE_MIN_CELLS = 3           # a column covers a tile holding at least this many of its cells' centres
MAX_COLUMNS = 4
MAX_POINTS = 60_000          # UMAP points over all columns (the worker samples each column evenly)
WHOLE_SHARE = 0.5            # count a whole element when the area's box covers at least this share of its extent
RESULT_VERSION = 4           # a change to what a comparison computes: earlier results are not reused (2: draw_seed;
                             # 3: the area's box within the sample; 4: each column by its own count, the run's proseg's)
RESULTS_KEPT = 6             # comparisons kept per run (compare/<job>/)
RESTARTS = 1                 # a compared element written again while being compared: compared again from the start
WRITTEN_AGAIN = "It was written again while being compared: compare again"   # ... and once more: its column fails
DISAGREE = "Its counts and its outlines disagree (details in the engine log)"   # rows not of the outlines, same version
SAME_CELL_UM = 1.0           # a counts row's centre (the helper's: the centroid of the element's geometry) lies within this
                             # of its outline's bounds (a centroid of a valid outline does, holes or several parts: the
                             # export's float32 rings), or of the outline's own centre (outline_centres: the same signed-
                             # area centre where a ring crosses itself, which can lie far outside its bounds)
MISFITS_LEFT_OUT = 0.01      # up to this share of a counts file's rows may misfit their outlines: those rows are left out
                             # (never scored: a ring that crosses itself in a cell of several parts, or whose lobes all
                             # but cancel, has no centre the helper and the engine agree on); beyond it the file is not
                             # of the outlines (DISAGREE)
RUN_COLUMN = "run"
RUN_COUNT = "proseg"         # the run's column's count (a column's `count`): proseg's own count of its frame
RUN_FIELDS = ("form", "like_from")   # the run's column also says which form of proseg's count it reads ("averaged" |
                             # "sampled": the count proseg saves) and whose draw its like-for-like cut starts from
                             # ("proseg": its own f; "app": a draw the app makes from proseg's count)
GONE = "Proseg's count of this iteration is gone: compare again"   # the run's frame's count cleared before it was held
FRAME_TRIES = 3              # the run's frame asked for at most this many times while its counts line names another
                             # file than the one given (its seq counted again for good, with proseg's draw: _frame)
RECOUNTED = "Proseg's count of this iteration changed while it was read: compare again"   # ... each time
NOT_HELD = "Its table holds none of its cells in the area (details in the engine log)"   # a table's own count naming
                             # none of the area's outlines (its ids written in another form): never a column of none
_HOLDS = itertools.count(1)  # each comparison's own number for the copy of the run's frame's count it holds (_hold)
DEFAULT_EXCLUDED = next(o["default"] for o in OPTIONS if o["flag"] == "excluded-genes")   # proseg's own
DEFAULT_RULE = {"min_qv": float(next(o["default"] for o in OPTIONS if o["flag"] == "min-qv")), "exclude": None}
WORKER_WAIT_S = 900.0        # longest wait for the worker's answer (it gives up on a request after 300 s itself)
# Kind colours by kind index (at most 16 kinds), for a kind the worker sends without its own: the worker's KIND_PALETTE
# (kept apart from the mixing colours and from each other, and light enough for the view's dark plots), so one kind has
# one colour whoever names it.
KIND_COLOURS = KIND_PALETTE
# What the page may say of a failure (compare.mjs shows a column's `error` and the comparison's as they come): the app's
# own errors, written for the page, as they are; anything else (an engine exception, the GPU worker's technical text) as
# a plain line, its own text left to the engine's log.
PAGE_ERRORS = (RunError, SampleError, AnalysisError)
INCOMPLETE_KINDS = ("failed", "busy")   # the kinds could not be made this time (the worker's kinds_reason)
# Why the sample's mixing check has no model, for Compare's refusal ("wait until it has learned this sample"
# while waiting could not help): the mixing check's own words for its reasons (frontend/src/qc-copy.mjs FAILURE_REASONS,
# UNAVAILABLE_REASONS; engine/qc.py REASONS); a reason not named here gets the general words of its state.
# The page's own words for the same reasons (frontend/src/qc-copy.mjs FAILURE_REASONS, UNAVAILABLE_REASONS). Too few
# cells covers a table of under 200 cells, too few cells with enough transcripts, and a fit that saw no mixing anywhere
# (the trainer's collapse_check: a small table learned too long).
CHECK_FAILED = {"oom": "the graphics card ran out of memory", "not_settled": "the learning did not settle",
                "timeout": "it took longer than its time limit", "unreadable": "the cell table could not be read",
                "too_few_cells": "it has too few cells to learn from",
                "gpu": "it could not run on the graphics card set aside for it",
                "unexpected": "an unexpected error (details in the engine log)"}
CHECK_UNAVAILABLE = {"no_table": "this sample has no table of cells with transcript counts",
                     "not_counts": "the sample's cell table doesn't hold whole transcript counts",
                     "positions": "the sample's cell table doesn't say where its cells lie on the transcripts",
                     "too_few_cells": "it has too few cells to learn from"}
NUMBERS =("judged", "mixed", "mixed_per100", "like100_per100", "like100_sd", "like100_cells", "histogram", "median_tx",
           "under100_share", "umap_points", "sampled")      # each column's numbers, as the worker gives them
# Each column's own map ("Each its own" on the page): how many clusters it found alone, how many of them match one of
# the run's column by their profiles, and whether it has a map of its own (too few cells: none)
OWN = {"clusters": 0, "clusters_matched": 0, "own": False}
# How the mixing check is missing on this computer (_unlearned's reasons; the page's own words are qc-copy.mjs
# UNAVAILABLE_REASONS): clauses of one sentence with one colon, never naming the method.
CHECK_SETUP = {
    "software": "it needs scvi-tools and scviva-tools, in this Python environment or in one named as python under "
                "[qc] in the app's config.toml",
    "no_gpu": "it learns on an NVIDIA graphics card or a Mac's own GPU, and this computer has none it can use",
    "apple_gpu": "it learns on this Mac's GPU, which failed a short test with this computer's PyTorch (the Log says "
                 "why; on macOS 14, installing the app again brings PyTorch 2.14, which passes it)",
    "torch_cpu": "it learns on an NVIDIA graphics card, which this computer's PyTorch can't use until its CUDA build "
                 "is installed",
    "driver": "it learns on an NVIDIA graphics card, whose driver is too old for this computer's PyTorch and needs "
              "updating"}
_OFFSET, _SPAN = 1 << 20, 1 << 21          # tile keys: (i + _OFFSET) * _SPAN + (j + _OFFSET), |i|, |j| < 2**20 tiles


class CompareError(RunError):
    pass


class Cancelled(Exception):
    """A newer comparison of the run replaced this one (or the run went)."""


class Changed(Exception):
    """A compared element was written again while being compared: its stamp is not the one pinned, or its counts are not
    of the outlines read."""


# ---- cells and the area -------------------------------------------------------------------------------------------
def outline_centres(outlines: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each exported outline's centre (µm, float64; the arrays of ZarrSamples._outlines): the area-weighted centre of its
    parts' rings (shoelace, around each ring's first vertex: exact far from the origin too), the mean of its parts'
    vertex means where it has no area, NaN for an outline without vertices."""
    offsets = np.asarray(outlines["offsets"], np.int64)
    coords = np.asarray(outlines["coords"], np.float64).reshape(-1, 2)
    owner = np.asarray(outlines["owner"], np.int64)
    count = len(outlines["ids"])
    sizes = np.diff(offsets)
    full = np.flatnonzero(sizes > 0)             # parts with vertices; theirs are coords[offsets[0]:offsets[-1]] in order
    if not full.size:
        return np.full(count, np.nan), np.full(count, np.nan)
    n = sizes[full]
    origin = coords[offsets[:-1][full]]
    local = coords[offsets[0]:offsets[-1]] - np.repeat(origin, n, axis=0)
    edges = np.zeros(full.size + 1, np.int64)
    np.cumsum(n, out=edges[1:])
    following = np.arange(len(local)) + 1
    following[edges[1:] - 1] = edges[:-1]        # a ring's last vertex closes onto its first
    x, y = local[:, 0], local[:, 1]
    xn, yn = x[following], y[following]
    cross = x * yn - xn * y
    twice = np.add.reduceat(cross, edges[:-1])   # twice each part's signed area
    flat = np.abs(twice) <= 1e-12
    safe = np.where(flat, 1.0, twice)
    px = origin[:, 0] + np.where(flat, np.add.reduceat(x, edges[:-1]) / n, np.add.reduceat((x + xn) * cross, edges[:-1]) / (3 * safe))
    py = origin[:, 1] + np.where(flat, np.add.reduceat(y, edges[:-1]) / n, np.add.reduceat((y + yn) * cross, edges[:-1]) / (3 * safe))
    weight, own = np.where(flat, 0.0, np.abs(twice)), owner[full]
    total = np.bincount(own, weights=weight, minlength=count)
    parts = np.bincount(own, minlength=count)
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = np.where(total > 0, np.bincount(own, weights=weight * px, minlength=count) / total,
                      np.bincount(own, weights=px, minlength=count) / parts)
        cy = np.where(total > 0, np.bincount(own, weights=weight * py, minlength=count) / total,
                      np.bincount(own, weights=py, minlength=count) / parts)
    return np.where(parts > 0, cx, np.nan), np.where(parts > 0, cy, np.nan)


def outline_cells(outlines: dict, centres=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each exported outline's id (text), bounds [x0, y0, x1, y1] (µm, float64: its parts' together; NaN without any)
    and centre (n x 2: `centres` as outline_centres gave them, else worked out here): what a counts row at its position
    must match (same_cells)."""
    names = np.asarray(outlines["ids"], dtype=str).reshape(-1)
    count = len(names)
    cx, cy = outline_centres(outlines) if centres is None else centres
    centre = np.c_[np.asarray(cx, np.float64).reshape(-1), np.asarray(cy, np.float64).reshape(-1)]
    parts = np.asarray(outlines["bounds"], np.float64).reshape(-1, 4)
    owner = np.asarray(outlines["owner"], np.int64)
    if owner.size == count and np.array_equal(owner, np.arange(count)):     # one part each, the usual case
        return names, parts, centre
    low, high = np.full((count, 2), np.inf), np.full((count, 2), -np.inf)
    np.fmin.at(low, owner, parts[:, :2])
    np.fmax.at(high, owner, parts[:, 2:])
    bounds = np.hstack([low, high])
    bounds[~np.isfinite(bounds).all(axis=1)] = np.nan
    return names, bounds, centre


def misfits(cells: tuple, position, ids, x, y) -> np.ndarray:
    """The counts rows (their order in the file) that are not the outlines `cells` (outline_cells) at their positions
    (zarr_bridge/helper.py count: each row's position in the element, id and centre as the helper read it): a row's id
    must be its outline's, and its centre within SAME_CELL_UM of the outline's bounds or of the outline's own centre (ids
    alone match for any element as long: a RangeIndex); no place on either side is alike. Every row when they are not
    one per position, or a position lies beyond the outlines."""
    names, bounds, centre = cells
    position = np.asarray(position, np.int64).reshape(-1)
    ids = np.asarray(ids).astype(str).reshape(-1)
    x, y = np.asarray(x, np.float64).reshape(-1), np.asarray(y, np.float64).reshape(-1)
    if not ids.size == x.size == y.size == position.size \
            or position.size and (position.min() < 0 or position.max() >= len(names)):
        return np.arange(max(ids.size, x.size, y.size, position.size))
    box, (cx, cy), near = bounds[position], centre[position].T, SAME_CELL_UM
    with np.errstate(invalid="ignore"):
        inside = (x >= box[:, 0] - near) & (x <= box[:, 2] + near) & (y >= box[:, 1] - near) & (y <= box[:, 3] + near)
        close = np.hypot(x - cx, y - cy) <= near
    nowhere = ~np.isfinite(x) & ~np.isfinite(y) & ~np.isfinite(box).any(axis=1)
    return np.flatnonzero((names[position] != ids) | ~(inside | close | nowhere))


def same_cells(cells: tuple, position, ids, x, y) -> bool:
    """Whether counts rows are the outlines `cells` at their positions: none of them misfits."""
    return not misfits(cells, position, ids, x, y).size


def frame_centres(path, ncells: int) -> tuple[np.ndarray, np.ndarray]:
    """A frame's cell centres (µm; NaN for a cell without a column in the frame) from its cell statistics
    (frames/<seq>.cells.zst: 7 x ncells little-endian: area, voxels, assigned, foreground, changed as u32, then the
    centre x, y as f32; native/proseg/src/live.rs)."""
    import zstandard
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(Path(path).read_bytes())
    if len(raw) != 28 * ncells:
        raise ValueError(f"the frame's cell statistics hold {len(raw) // 28} cells, not {ncells}")
    stats = np.frombuffer(raw, "<u4").reshape(7, ncells)
    present = stats[0] > 0
    x, y = stats[5].view("<f4").astype(np.float64), stats[6].view("<f4").astype(np.float64)
    return np.where(present, x, np.nan), np.where(present, y, np.nan)


def tile_keys(x, y, tile: float = TILE_UM, origin=(0.0, 0.0)) -> np.ndarray:
    """Each centre's tile (floor((x - ox) / tile), floor((y - oy) / tile)) as one int64 key; -1 for a missing centre
    (NaN)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    i = np.clip(np.floor((np.where(ok, x, origin[0]) - origin[0]) / tile), 1 - _OFFSET, _OFFSET - 1).astype(np.int64)
    j = np.clip(np.floor((np.where(ok, y, origin[1]) - origin[1]) / tile), 1 - _OFFSET, _OFFSET - 1).astype(np.int64)
    return np.where(ok, (i + _OFFSET) * _SPAN + (j + _OFFSET), -1)


def tile_of(keys) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) of tile keys."""
    keys = np.asarray(keys, np.int64)
    return keys // _SPAN - _OFFSET, keys % _SPAN - _OFFSET


def covered_tiles(x, y, tile: float = TILE_UM, least: int = TILE_MIN_CELLS, origin=(0.0, 0.0)) -> np.ndarray:
    """The sorted keys of the tiles holding at least `least` of these centres."""
    keys = tile_keys(x, y, tile, origin)
    found, counts = np.unique(keys[keys >= 0], return_counts=True)
    return found[counts >= least]


def within(x, y, limit, polygon=None) -> tuple[np.ndarray, np.ndarray]:
    """The centres with those outside the box `limit` [x0, y0, x1, y1) or the polygon (n x 2) as missing (NaN); all of
    them without either."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if limit is None and polygon is None:
        return x, y
    out = np.zeros(x.shape, bool)
    if limit is not None:
        with np.errstate(invalid="ignore"):
            out |= ~((x >= limit[0]) & (x < limit[2]) & (y >= limit[1]) & (y < limit[3]))
    if polygon is not None:                       # only the centres in the box are tested against the polygon's edges
        test = np.flatnonzero(~out & np.isfinite(x) & np.isfinite(y))
        out[test[~in_polygon(x[test], y[test], polygon)]] = True
        out |= ~(np.isfinite(x) & np.isfinite(y))
    return np.where(out, np.nan, x), np.where(out, np.nan, y)


def common_area(centres: dict, tile: float = TILE_UM, least: int = TILE_MIN_CELLS, limit=None, polygon=None,
                clip=None) -> dict:
    """The area every column covers: the tiles holding at least `least` cells of each column (`centres`:
    {name: (x, y)}), on a grid from (0, 0) of the frame; with `limit` (a region run's box [x0, y0, x1, y1]) only cells
    inside it count, the grid starts at its corner and the area stays inside it. {tiles: their sorted keys, box: [x0,
    y0, x1, y1] around them (µm; None when they share none), size_um: [w, h] (zeros without), count: how many, own:
    {name: the tiles that column covers by itself}, and the grid: tile, origin, limit, polygon (in_area takes them)}.
    `polygon` (a lasso run's, n x 2 µm): only cells whose centre lies inside it count. `clip` (area_clip: the sample
    and the run's bounds): the box and its size never go past it (whole tiles stated an area larger than the
    sample itself); which cells are in the area stays the tiles' business."""
    origin = (float(limit[0]), float(limit[1])) if limit is not None else (0.0, 0.0)
    own, common = {}, None
    for name, (x, y) in centres.items():
        tiles = covered_tiles(*within(x, y, limit, polygon), tile, least, origin)
        own[name] = int(tiles.size)
        common = tiles if common is None else np.intersect1d(common, tiles, assume_unique=True)
    common = np.zeros(0, np.int64) if common is None else common
    box = None
    if common.size:
        i, j = tile_of(common)
        box = [origin[0] + i.min() * tile, origin[1] + j.min() * tile,
               origin[0] + (i.max() + 1) * tile, origin[1] + (j.max() + 1) * tile]
        for bounds in (limit, clip):
            cut = overlap(box, bounds)
            box = cut if cut is not None else box      # a clip the tiles miss altogether says nothing of them
        box = [round(float(v), 2) for v in box]
    size = [round(box[2] - box[0], 2), round(box[3] - box[1], 2)] if box else [0.0, 0.0]
    return {"tiles": common, "box": box, "size_um": size, "count": int(common.size), "own": own, "tile": tile,
            "origin": origin, "limit": limit, "polygon": polygon}


def overlap(box, bounds) -> list[float] | None:
    """The part of `box` [x0, y0, x1, y1] inside `bounds` (`box` itself without bounds); None when they do not meet."""
    if bounds is None:
        return list(box)
    cut = [max(box[0], bounds[0]), max(box[1], bounds[1]), min(box[2], bounds[2]), min(box[3], bounds[3])]
    return cut if cut[2] > cut[0] and cut[3] > cut[1] else None


def _box(value) -> list[float] | None:
    """[x0, y0, x1, y1] as four finite floats of a box with some area, else None."""
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 and all(map(math.isfinite, box)) and box[2] > box[0] and box[3] > box[1] else None


def area_clip(run, extent=None, with_run: bool = True) -> list[float] | None:
    """What the area's box is said within (common_area `clip`): the sample's extent (its transcripts' bounding
    box, `extent`) and, when the run's column is compared, the run's own bounds (its region's bounding box; a full-sample
    run's is the sample's extent when it ran), where both are known; None when neither is."""
    bounds = _box(extent) if extent is not None else None
    if with_run:
        region = run.meta.get("region") if isinstance(run.meta.get("region"), dict) else {}
        own = _box(region.get("bbox") or region.get("box") or run.meta.get("box") or ())
        if own is not None:
            bounds = own if bounds is None else (overlap(own, bounds) or own)
    return bounds


def in_area(x, y, area: dict) -> np.ndarray:
    """Which centres lie in the area's tiles (common_area: its grid, limit and polygon)."""
    keys = tile_keys(*within(x, y, area.get("limit"), area.get("polygon")), area.get("tile", TILE_UM),
                     area.get("origin", (0.0, 0.0)))
    return (keys >= 0) & np.isin(keys, area["tiles"])


def count_box(box: list[float], extent) -> list[float] | None:
    """The box to count an element over: None (the whole element, which serves any later area too) when the area's
    box covers at least WHOLE_SHARE of the element's extent, else the area's box."""
    try:
        x0, y0, x1, y1 = (float(v) for v in extent)
        share = (box[2] - box[0]) * (box[3] - box[1]) / max((x1 - x0) * (y1 - y0), 1e-9)
    except (TypeError, ValueError):
        return box
    return None if share >= WHOLE_SHARE else box


POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("kind", "u1"), ("share", "u1")])   # 10 bytes, nothing between


def write_points(path: Path, points: dict):
    """A column's points file from {x, y, kind?, share?} lists of one length (kind 255: none; share: floor(share x 250),
    255 not judged; either missing: all 255), written whole or not at all."""
    values = np.zeros(len(points["x"]), POINT)
    values["x"], values["y"] = points["x"], points["y"]
    for name in ("kind", "share"):
        values[name] = points[name] if points.get(name) is not None else 255
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_bytes(values.tobytes())
    os.replace(temporary, path)


def job_key(model: dict, columns: list[str], stamps: dict, seq, rule: dict | None = None,
            thresholds: dict | None = None, clip: list[float] | None = None, counted: dict | None = None) -> dict:
    """Everything that decides a comparison's numbers (its digest names its folder): the model and whether it has its
    details, the columns with their element stamps and whose count each reads (`counted` {column: [count, its table,
    the table's version]}; the run's: [RUN_COUNT, whether its frame's count holds proseg's own draw], and "sampled"
    after them for a frame read on the count proseg saves, so an averaged frame's key is as it was), the frame, the
    form of proseg's count read live (LIVE_COUNT), the counting rule (the run's own input filters: `rule`), the mixing
    check's thresholds, the area's tiles, what its box is said within (`clip`) and the points' cap."""
    from ..zarr_bridge.helper import COUNT_RULE
    counted = counted or {}
    return {"v": RESULT_VERSION, "rule": COUNT_RULE, "input": rule or dict(DEFAULT_RULE), "model": model["id"],
            "deep": bool(model.get("deep")), "columns": [[c, stamps.get(c), counted.get(c)] for c in columns],
            "seq": seq if RUN_COLUMN in columns else None, "count": LIVE_COUNT, "thresholds": thresholds or {},
            "tile": TILE_UM, "least": TILE_MIN_CELLS, "points": MAX_POINTS, "clip": clip}


def input_rule(run) -> dict:
    """The run's own input filters, as proseg read its transcripts (its options, else proseg's defaults): every column
    with no count of its own (its outlines count it) is counted by them, {min_qv: quality at least, exclude: a gene-name
    pattern of the run's own, or None for proseg's default (which leaves out no gene of the model)}."""
    options = run.meta.get("options") if isinstance(run.meta.get("options"), dict) else {}
    try:
        min_qv = float(options.get("min-qv", DEFAULT_RULE["min_qv"]))
    except (TypeError, ValueError):
        min_qv = DEFAULT_RULE["min_qv"]
    pattern = options.get("excluded-genes")
    exclude = pattern if isinstance(pattern, str) and pattern and pattern != DEFAULT_EXCLUDED else None
    return {"min_qv": min_qv if np.isfinite(min_qv) else DEFAULT_RULE["min_qv"], "exclude": exclude}


def region_limit(run) -> tuple[list[float] | None, np.ndarray | None]:
    """A region run's limit for every column: its box ([x0, y0, x1, y1], the lasso's bounds) and, for a
    lasso, its polygon (n x 2, µm); (None, None) for a full-sample run. Runs that recorded no region: their box."""
    region = run.meta.get("region") if isinstance(run.meta.get("region"), dict) else None
    if region is None:
        box = run.meta.get("box")
        return ([float(v) for v in box] if box else None), None
    if region.get("kind") == "full":
        return None, None
    box = region.get("bbox") or region.get("box") or run.meta.get("box")
    box = [float(v) for v in box] if box else None
    if region.get("kind") == "polygon" and region.get("points"):
        return box, np.asarray(region["points"], np.float64).reshape(-1, 2)
    return box, None


def in_polygon(x, y, polygon: np.ndarray) -> np.ndarray:
    """Which points lie inside the polygon (even-odd rule; NaN points never), one edge at a time over all points."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    inside = np.zeros(x.shape, bool)
    xs, ys = polygon[:, 0], polygon[:, 1]
    for k in range(len(polygon)):
        x0, y0, x1, y1 = xs[k - 1], ys[k - 1], xs[k], ys[k]
        if y0 == y1:
            continue
        with np.errstate(invalid="ignore"):
            crosses = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
        inside ^= crosses
    return inside


def digest_of(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def draw_seed(run_id: str, model: dict) -> int:
    """The seed of a comparison's draws (like for like; the points picked): the run's and the model's, never the
    columns chosen, so a column keeps its like-for-like number when another column is added or taken away (only its
    own cells or the area change it)."""
    # The salt keeps the app's earlier name, so that draws (and results) stay those of earlier comparisons.
    return int(hashlib.sha1(f"proseg-live compare|{run_id}|{model['id']}".encode()).hexdigest()[:8], 16)


def said(exc: BaseException, plain: str) -> str:
    """The page's words for a failure (PAGE_ERRORS): an app error's own message, anything else `plain`. The sopa
    helper's failures (HelperFailed: its exception's own words, or its last lines) are technical: `plain` too."""
    return (str(exc) or plain) if isinstance(exc, PAGE_ERRORS) and not isinstance(exc, HelperFailed) else plain


def technical(exc: BaseException) -> bool:
    """Whether a failure's own words go to the engine log only (said() gives the page a plain line for it)."""
    return not isinstance(exc, PAGE_ERRORS) or isinstance(exc, HelperFailed)


def uncounted(job, index: int) -> str:
    """The page's words for a column whose counts could not be had: its table's own could not be read, or its
    transcripts could not be counted in its outlines."""
    return "Could not read its counts" if index in job.tables else "Could not count its transcripts"


def kind_colour(kind: dict) -> str:
    """A kind's colour: the worker's when it sends one (#rrggbb), else KIND_COLOURS by its index."""
    colour = kind.get("color")
    if isinstance(colour, str) and len(colour) == 7 and colour[0] == "#" \
            and all(c in "0123456789abcdefABCDEF" for c in colour[1:]):
        return colour.lower()
    return KIND_COLOURS[kind["index"] % len(KIND_COLOURS)]


# ---- jobs ---------------------------------------------------------------------------------------------------------
class CompareJob:
    """One comparison of one run: what was asked, its key, and where it stands (read by GET while its thread works)."""

    def __init__(self, run_id: str, digest: str, columns: list[dict], model: dict, seq, iteration, folder: Path,
                 rule: dict | None = None, clip: list[float] | None = None, versions: dict[int, str] | None = None,
                 frame: dict | None = None, tables: dict[int, dict] | None = None):
        self.run_id, self.digest, self.model, self.seq, self.iter, self.folder = run_id, digest, model, seq, iteration, folder
        self.rule = dict(rule or DEFAULT_RULE)   # the counting rule of the columns their outlines count: the run's own
        self.clip = clip                         # the area's box is said within it (area_clip)
        self.keyed = dict(versions or {})        # shapes column index -> its element's stamp, as the key names it
        self.versions = dict(self.keyed)         # ... as compared now (another after a restart: never kept then)
        self.final = True                        # this try fails a column written again, instead of starting again
        self.frame = frame                       # the run's column of the worker's op (compare_frame), None without it
        self.held: Path | None = None            # the copy of its frame's count this comparison holds (_hold)
        self.token = next(_HOLDS)
        self.tables = dict(tables or {})         # shapes column index -> {count_by, table}: a column its table counts
        self.seed = draw_seed(run_id, model)
        self.lock = threading.Lock()
        self.columns = [{"key": c["key"], "name": c["name"], "count": c.get("count"),
                         **{k: c[k] for k in RUN_FIELDS if k in c}, "state": "waiting",
                         "cells_in_area": None, "tiles": None, "error": None, "ready_s": None,
                         **dict.fromkeys(NUMBERS), **OWN}
                        for c in columns]
        self.state, self.step, self.error = "working", "waiting", None
        self.done, self.of = 0, 1
        self.area: dict | None = None
        self.kinds: list | None = None
        self.kinds_reason: str | None = None   # the worker's: needs_detail (no deep.npz: Learn the details), no_source, failed
        self.bounds: list | None = None        # the shared layout's view box (2nd-98th percentile), for every column
        self.layout_error: str | None = None   # the worker's: too_few_cells, or a message: numbers without a picture
        self.layout = 0
        self.layout_kind: str | None = None    # the worker's word for the newest layout ("umap"; "pca" for its numpy
                                               # stand-in), None while there is none
        self.files: dict[int, Path] = {}       # column index -> its newest points file
        # "Each its own" (the page's Map switch): column index -> {"points": its own map's points file (the same
        # records, its own x and y), "clusters": its own clusters (u16 per point)}; the worker's word for those maps,
        # their shared view box once lined up with the run's, and why there are none
        self.own: dict[int, dict] = {}
        self.own_layout: str | None = None
        self.own_bounds: dict | None = None
        self.own_error: str | None = None
        self.counts: dict[int, str] = {}       # shapes column index -> its counts file
        self.seconds: dict[str, float] = {}
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.after: CompareJob | None = None   # the run's previous comparison, finished before this one starts
        self.started = time.monotonic()

    def public(self) -> dict:
        with self.lock:
            return {"job": self.digest, "state": self.state, "step": self.step,
                    "progress": {"done": self.done, "of": self.of}, "model": self.model["id"], "seq": self.seq,
                    "iter": self.iter, "area": dict(self.area) if self.area else None,
                    "kinds": [dict(k) for k in self.kinds] if self.kinds is not None else None,
                    "kinds_reason": self.kinds_reason, "columns": [dict(c) for c in self.columns],
                    "layout": self.layout, "layout_kind": self.layout_kind, "bounds": self.bounds,
                    "layout_error": self.layout_error, "error": self.error,
                    "own_layout": self.own_layout, "own_bounds": self.own_bounds, "own_error": self.own_error,
                    "rule": dict(self.rule), "seconds": dict(self.seconds)}

    def whole(self) -> bool:
        """Nothing of it failed or had to wait: every column ready and the kinds made (or not to be made: no details
        learned yet), at the versions its key names. Only a whole comparison is kept and reused."""
        with self.lock:
            return self.state == "ready" and all(c["state"] == "ready" for c in self.columns) \
                and self.kinds_reason not in INCOMPLETE_KINDS and self.versions == self.keyed

    def set(self, **values):
        with self.lock:
            for name, value in values.items():
                setattr(self, name, value)

    def column(self, index: int, **values):
        with self.lock:
            self.columns[index].update(values)

    def advance(self, step: str | None = None, by: int = 1):
        with self.lock:
            self.done = min(self.of, self.done + by)
            if step is not None:
                self.step = step

    def note(self, name: str, seconds: float):
        with self.lock:
            self.seconds[name] = round(seconds, 2)

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()


class CompareManager:
    """POST/GET /runs/{id}/compare and its points (server/app.py). `analysis` is the AnalysisManager: `available`,
    `qc_thresholds()`, `compare_frame(run_id, seq)` (the run's column: the frame with proseg's count nearest `seq`) and
    `submit_compare(request)`, which queues one op "compare" for the GPU worker (the scheduler's thread is the only one
    that talks to it) and returns a Future of its reply {ok, compare: {...}} (AnalysisError when none came; a Future
    cancelled before it started is dropped); `qc` is the QcManager: `qc_model(run)` (the sample's model {id, path,
    deep}) and its `models` (QcModels: the table the model learned on, which kinds are made from)."""

    def __init__(self, runs, zarr, qc, analysis):
        self.runs, self.zarr, self.qc, self.analysis = runs, zarr, qc, analysis
        self.lock = threading.Lock()
        self.jobs: dict[str, CompareJob] = {}
        self.closed = False

    # ---- requests -------------------------------------------------------------------------------------------------
    def start(self, run_id: str, body) -> dict:
        """{columns: ["run" | a segmentation's key, ...] (1 to MAX_COLUMNS, each once), seq?: the run's frame (default:
        its newest; either way the nearest with proseg's count, whose seq and iteration the comparison says)} -> the
        comparison (202): the same one while it holds, one found beside the run, or a new one (replacing the run's
        comparison on its way). Each column says whose count it reads (`count`: RUN_COUNT for the run's; a
        segmentation's count_by, as the sample's listing says it). 404 unknown run, frame or segmentation, or not a
        proseg run on a SpatialData sample; 409 no model for its sample yet, or no frame to compare (none with proseg's
        count); 422 bad columns; 503 where Clusters & UMAP cannot run."""
        run = self.runs.get(run_id)
        if run.meta.get("kind") in ("stain", "patches") or run.meta.get("sample_kind") != "zarr":
            raise CompareError("Compare reads the cells of a proseg run on a SpatialData sample", 404)
        columns = body.get("columns") if isinstance(body, dict) else None
        if not isinstance(columns, list) or not 1 <= len(columns) <= MAX_COLUMNS \
                or not all(isinstance(c, str) and c for c in columns) or len(set(columns)) != len(columns):
            raise CompareError(f"columns: 1 to {MAX_COLUMNS} different columns, each \"run\" or a segmentation's key")
        if self.analysis is None or not getattr(self.analysis, "available", False) \
                or not hasattr(self.analysis, "submit_compare"):
            why = self.analysis.unavailable_why() if hasattr(self.analysis, "unavailable_why") else \
                "no analysis worker is attached"
            raise CompareError(f"Compare needs Clusters and the UMAP, which can't run here: {why}.", 503)
        model = self._model(run)
        if not model or not model.get("id") or not model.get("path"):
            raise CompareError(self._unlearned(run.meta["sample"]), 409)
        sid = run.meta["sample"]
        if not self.zarr.known(sid):                     # forgotten by the open dialog: opened again where the run found
            self.zarr.open(run.meta["sample_path"], recent=False)          # it, and not listed again for that
        listed = {s["key"]: s for s in self.zarr.segmentations(sid)}
        for key in columns:
            if key != RUN_COLUMN and key not in listed:
                raise CompareError(f"The sample has no segmentation '{key}'", 404)
            if key != RUN_COLUMN and listed[key].get("version") == UNREAD:   # its folder was briefly unreadable
                raise CompareError(f"'{key}' could not be read just now: try again in a moment", 503)
        seq, frame, drawn, count = self._frame(run, body.get("seq")) if RUN_COLUMN in columns else \
            (None, None, None, None)
        rule = input_rule(run)
        thresholds = self.analysis.qc_thresholds() if hasattr(self.analysis, "qc_thresholds") else {}
        clip = area_clip(run, self._extent(sid), RUN_COLUMN in columns)
        tables = {k: self._counted_by(sid, listed[k]) for k in columns if k != RUN_COLUMN}
        counted = {k: [t["count_by"], t["table"], t["version"]] for k, t in tables.items()}
        counted[RUN_COLUMN] = [RUN_COUNT, drawn, "sampled"] if count == "sampled" else [RUN_COUNT, drawn]
        digest = digest_of(job_key(model, columns, {k: listed[k]["version"] for k in columns if k in listed}, seq,
                                   rule, thresholds, clip, counted))
        names = {RUN_COLUMN: f"This run · iteration {frame.get('iter')}" if frame else "This run",
                 **{k: s["name"] for k, s in listed.items()}}
        whose = {RUN_COLUMN: RUN_COUNT, **{k: t["count_by"] for k, t in tables.items()}}
        # The run column says which form of proseg's count it reads (`form`: "sampled", the count proseg saves, or
        # "averaged") and whose draw its like-for-like cut starts from (`like_from`: "proseg", its own draw f, where the
        # frame's file holds it; "app", a draw the app makes from proseg's count).
        form = {"form": count, "like_from": "proseg" if drawn else "app"} if RUN_COLUMN in columns else {}
        with self.lock:
            if self.closed:
                raise CompareError("The engine is closing", 503)
            current = self.jobs.get(run_id)
            # The same comparison while it works, or once it is whole; one that ended with a part missing (a column
            # or the kinds failed, or the kinds had to wait for a save) is done again.
            if current is not None and current.digest == digest and (current.state == "working" or
                                                                     (current.state == "ready" and current.whole())):
                return current.public()
            job = self._stored(run, digest, model)
            if job is not None and form:                 # a comparison kept before these fields: its key holds both
                job.column(columns.index(RUN_COLUMN), **form)
            if job is None:
                job = CompareJob(run_id, digest, [{"key": c, "name": names[c], "count": whose[c],
                                                   **(form if c == RUN_COLUMN else {})} for c in columns],
                                 model, seq, frame.get("iter") if frame else None, run.dir / "compare" / digest, rule,
                                 clip, {i: listed[c]["version"] for i, c in enumerate(columns) if c != RUN_COLUMN},
                                 frame, {i: {"count_by": tables[c]["count_by"], "table": tables[c]["table"]}
                                         for i, c in enumerate(columns) if c in tables and tables[c]["table"]})
                job.thread = threading.Thread(target=self._work, args=(job, run, listed), daemon=True,
                                              name=f"compare-{run_id}")
                if current is not None and current.state == "working":
                    current.cancel.set()                 # its helper is stopped; a worker call it waits on finishes
                    job.after = current
            elif current is not None:
                current.cancel.set()
            self.jobs[run_id] = job
            if job.thread is not None:
                job.thread.start()
        return job.public()

    def describe(self, run_id: str) -> dict:
        """The run's comparison as it stands (GET; CompareJob.public); 404 without one."""
        self.runs.get(run_id)
        with self.lock:
            job = self.jobs.get(run_id)
        if job is None:
            raise CompareError("No comparison for this run yet", 404)
        return job.public()

    def points(self, run_id: str, index: int, job: str | None = None, layout: int | None = None, own: bool = False,
               what: str = "points") -> Path:
        """The newest points file of column `index` (written by the worker: per point f32 x, f32 y, u8 kind (255 none),
        u8 share (floor(share x 250), 255 not judged), little-endian, nothing between); 404 before it has one. `job`
        (the comparison's digest) and `layout` (its count of layouts), as the status the page draws from said them:
        409 once the run's comparison is another (replaced, from another window too) or has a newer layout, so a
        column never draws the points of another comparison, column order or layout. `own`: its own map's points
        instead (the same points in the same order); `what="clusters"`: its own clusters (u16 per point); 404 for a
        column with no map of its own."""
        self.runs.get(run_id)
        with self.lock:
            current = self.jobs.get(run_id)
        if job is not None and (current is None or current.digest != job):
            raise CompareError("This comparison was replaced", 409)
        path, shown = None, None
        if current is not None:
            with current.lock:                           # _score sets both together
                if own or what != "points":
                    path = current.own.get(index, {}).get(what)
                else:
                    path = current.files.get(index)
                shown = current.layout
        if layout is not None and current is not None and layout != shown:
            raise CompareError("These points were replaced by newer ones", 409)
        if path is None or not path.exists():
            raise CompareError("No points for this column yet", 404)
        return path

    def forget(self, run_id: str):
        """A deleted run: its comparison stops (its folder went with the run)."""
        with self.lock:
            job = self.jobs.pop(run_id, None)
        if job is not None:
            job.cancel.set()

    def close(self):
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            job.cancel.set()
        for job in jobs:
            if job.thread is not None and job.thread.is_alive():
                job.thread.join(5)

    def _unlearned(self, sid: str) -> str:
        """Compare's refusal without a model, in plain words: why the sample's mixing check has none when it could not
        learn the sample or cannot learn it (CHECK_FAILED, CHECK_UNAVAILABLE: waiting would never help),
        else to wait for it (turning it on first when it is off)."""
        try:
            check = self.qc.models.describe(sid)
        except Exception:                                # no state to tell: the general words
            check = None
        state = check.get("state") if isinstance(check, dict) else None
        reason = check.get("reason") if isinstance(check, dict) else None
        if state == "failed":
            return ("Compare needs this sample's mixing check, which could not learn this sample: "
                    f"{CHECK_FAILED.get(reason, 'it ran into a problem')}. Try again from the mixing check.")
        if state == "unavailable" and reason in CHECK_SETUP:
            return f"Compare needs the mixing check, which isn't set up on this computer: {CHECK_SETUP[reason]}."
        if state == "unavailable":
            return ("Compare needs this sample's mixing check, which cannot learn this sample: "
                    f"{CHECK_UNAVAILABLE.get(reason, 'it cannot learn from what the sample holds')}.")
        if state in ("waiting", "learning"):
            return "Compare needs this sample's mixing check: wait until it has learned this sample"
        return "Compare needs this sample's mixing check: turn it on and wait until it has learned this sample"

    def _extent(self, sid: str) -> list[float] | None:
        """The sample's extent (its detail: the transcripts' bounding box, µm), None when it cannot say."""
        try:
            return _box(self.zarr.detail(sid)["extent"])
        except Exception:                                # a sample read some other way: the run's own bounds only
            return None

    def _model(self, run) -> dict | None:
        """The sample's ready model {id, path (its folder), deep, table (the key of the table it learned on)}, or None."""
        model = self.qc.qc_model(run) if self.qc is not None else None
        models = getattr(self.qc, "models", None)
        if not model or models is None or not hasattr(models, "model"):
            return model
        try:
            full = models.model(run.meta["sample"]) or {}
        except Exception:                                # the table is only kinds' source: never a reason to refuse
            full = {}
        return {**model, "table": full.get("table")}

    def _frame(self, run, seq) -> tuple[int, dict, bool, str]:
        """The run's frame to compare (`seq`, or its newest), at the nearest frame with proseg's count, as the worker's
        op takes it (AnalysisManager.compare_frame: today's fields, its own seq and iter, `counts_file`), whether that
        count holds proseg's own draw (the like for like then reads it, else draws its own): `has_f` of the counts line
        (run.frames) that names the very file given, and which count the column reads (`count`, also the frame's
        "count" the worker reads: counts_file.count_of of that very line, "sampled" for a frame proseg counted for a
        pause, the hold, a finish or the end), so the key says what the worker reads. A seq can be counted
        again for good meanwhile (a hold, the finish, the end: the same count with proseg's draw, in the run's own
        folder), its line then naming that file: the frame is asked for again, FRAME_TRIES times at most (409
        RECOUNTED). Its refusals keep their words and status (404 an unknown frame; 409 none with proseg's count yet,
        or without exported transcripts)."""
        if seq is not None and (isinstance(seq, bool) or not isinstance(seq, int)):
            raise CompareError("seq must be a frame number")
        for _ in range(FRAME_TRIES):
            try:
                frame = dict(self.analysis.compare_frame(run.meta["id"], seq))
            except AnalysisError as exc:
                raise CompareError(str(exc), exc.status) from None
            found = frame.get("seq")
            if not isinstance(found, int) or not frame.get("cells_file") or not frame.get("counts_file"):
                raise CompareError("This iteration cannot be compared (no cell statistics or proseg's count)", 409)
            with run.lock:
                record = run.frames[found] if 0 <= found < len(run.frames) else {}
                named, line = record.get("counts_file"), record.get("counts") or {}
            if named and str(run.live / named) == frame["counts_file"]:
                count = count_of(line)
                frame["count"] = count
                return found, frame, bool(line.get("has_f")), count
        raise CompareError(RECOUNTED, 409)

    def _counted_by(self, sid: str, entry: dict) -> dict:
        """Whose count a segmentation's column reads, {count_by, table, version}: its own table's, as the sample's
        listing says (zarr_samples.count_source), with that table's version now (table_stamp: None where it cannot be
        read now; its counting reads it again), else its outlines' ("outlines", no table)."""
        table = entry.get("count_table") if entry.get("count_by") in ("proseg", "xenium") else None
        version = self.zarr.table_stamp(sid, table) if table and hasattr(self.zarr, "table_stamp") else None
        return {"count_by": entry["count_by"] if table else "outlines", "table": table, "version": version}

    def _stored(self, run, digest: str, model: dict) -> CompareJob | None:
        """A finished comparison found beside the run (result.json and its points files), else None."""
        folder = run.dir / "compare" / digest
        try:
            saved = json.loads((folder / "result.json").read_text())
            files = {int(i): folder / name for i, name in saved["files"].items()}
            own = {int(i): {k: folder / v for k, v in names.items()} for i, names in (saved.get("own_files") or {}).items()}
            if saved.get("job") != digest or saved.get("state") != "ready" or not all(p.exists() for p in files.values()):
                return None
            if "own_files" not in saved or not all(p.exists() for names in own.values() for p in names.values()):
                return None                              # kept before "each its own", or its maps gone: compared again
            job = CompareJob(run.meta["id"], digest, saved["columns"], model, saved.get("seq"), saved.get("iter"), folder,
                             saved.get("rule"))
            job.columns = [dict(c) for c in saved["columns"]]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        job.set(state="ready", step="done", done=1, of=1, area=saved.get("area"), kinds=saved.get("kinds"),
                kinds_reason=saved.get("kinds_reason"), bounds=saved.get("bounds"), layout_error=saved.get("layout_error"),
                layout=int(saved.get("layout") or 1), layout_kind=saved.get("layout_kind"),
                seconds=saved.get("seconds") or {}, files=files, own=own, own_layout=saved.get("own_layout"),
                own_bounds=saved.get("own_bounds"), own_error=saved.get("own_error"))
        return job

    # ---- the job ----------------------------------------------------------------------------------------------------
    def _work(self, job: CompareJob, run, listed: dict):
        try:
            if job.frame is not None:                    # held first: the wait below can be long (a worker call of
                try:                                     # the comparison it replaces), and the spool goes on clearing
                    (run.dir / "compare").mkdir(exist_ok=True)
                    job.folder.mkdir(exist_ok=True)
                    self._hold(job, run)
                except OSError:                          # gone already: _compare tries once more, and says so
                    pass
            if job.after is not None:
                job.after.thread.join()                  # one comparison of a run at a time (it stops at once)
                job.after = None
            for attempt in range(RESTARTS + 1):
                job.final = attempt == RESTARTS
                try:
                    self._compare(job, run, listed)
                    break
                except Changed as changed:               # the last try fails such a column instead (_written)
                    job.check()
                    if job.final:
                        raise CompareError(WRITTEN_AGAIN) from changed
                    print(f"SoPA companion: comparing {', '.join(c['key'] for c in job.columns)} of {job.run_id}: "
                          f"{changed}: compared again", flush=True)
                    self._again(job, run)
            print(f"SoPA companion: compared {', '.join(c['key'] for c in job.columns)} of {job.run_id}: "
                  f"{json.dumps(job.seconds)}", flush=True)
        except Cancelled:
            job.set(state="failed", error="replaced by a newer comparison")
        except Exception as exc:                         # the page says it; the engine carries on
            message = said(exc, "the engine ran into a problem (Log saves it)")
            with job.lock:
                job.state, job.error = "failed", message
                for column in job.columns:
                    if column["state"] not in ("ready", "failed"):
                        column.update(state="failed", error=message)
            print(f"SoPA companion: comparing {', '.join(c['key'] for c in job.columns)} of {job.run_id} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
        finally:
            if job.held is not None:                     # the worker has read it: the comparison keeps its numbers
                job.held.unlink(missing_ok=True)

    def _hold(self, job: CompareJob, run):
        """The run's frame's count held for the comparison: proseg's count of a frame the engine asked for lives in the
        engine's spool, which keeps only the newest frames' counts and clears the others as the run goes on, and a
        comparison can take a while to reach the worker (the run's previous comparison finished first, an outline
        export). It is copied beside the comparison as its work starts (run-<token>.counts.zst, gone once the
        comparison ends: _work), and the worker reads the copy; one in the run's own folder (a frame's lasting count) is
        never cleared, and is read where it is. Once held, held for good (a restart keeps it). OSError: it is gone
        already."""
        source = Path(job.frame["counts_file"])
        if job.held is not None or source.is_relative_to(run.live):
            return
        held = job.folder / f"run-{job.token}.counts.zst"
        temporary = held.with_name(f".{held.name}.{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, held)
        finally:
            temporary.unlink(missing_ok=True)
        job.held, job.frame = held, {**job.frame, "counts_file": str(held)}

    def _again(self, job: CompareJob, run):
        """Start again, a compared element written meanwhile: each shapes column pinned to its element's version now
        (element_stamp, read once more after STAMP_RETRY_S when its folder does not answer, as the listing does; one
        still not answering, or gone, fails), every number, area and picture of the try before gone (`layout` goes on
        counting: the page asks for the next one). The sample is not listed again: that can run the sopa helper (what
        made each element: ZarrSamples._made_by), which nothing stops; the first listing's extents
        still say whether an element is counted whole (count_box)."""
        sid = run.meta["sample"]
        keys = {i: c["key"] for i, c in enumerate(job.columns) if c["key"] != RUN_COLUMN}
        stamps = {i: self.zarr.element_stamp(sid, key) for i, key in keys.items()}
        if None in stamps.values():
            job.cancel.wait(STAMP_RETRY_S)
            job.check()
            stamps.update({i: self.zarr.element_stamp(sid, keys[i]) for i, stamp in stamps.items() if stamp is None})
        with job.lock:
            for index, column in enumerate(job.columns):
                column.update(state="waiting", cells_in_area=None, tiles=None, error=None, ready_s=None,
                              **dict.fromkeys(NUMBERS), **OWN)
                if index not in stamps:
                    continue
                if stamps[index] is None:
                    column.update(state="failed",
                                  error=f"'{column['key']}' could not be read just now: try again in a moment")
                else:
                    job.versions[index] = stamps[index]
            job.step, job.done, job.of, job.area, job.files, job.counts = "waiting", 0, 1, None, {}, {}
            job.own, job.own_layout, job.own_bounds, job.own_error = {}, None, None, None

    def _compare(self, job: CompareJob, run, listed: dict):
        job.check()
        sid = run.meta["sample"]
        shapes = [i for i, c in enumerate(job.columns) if c["key"] != RUN_COLUMN and c["state"] != "failed"]
        mark = time.monotonic()
        (run.dir / "compare").mkdir(exist_ok=True)       # not parents: a run deleted meanwhile stays deleted
        job.folder.mkdir(exist_ok=True)
        job.set(of=2 + len(shapes) + 2, step="outlines")

        # 1. Every column's cell centres: the run's frame (its count held for the worker as the work started, else tried
        # once more now), each element's outlines (exported now when never shown), and each outline's id, bounds and
        # centre, which its counts must match.
        centres, cells, moved = {}, {}, None
        with ThreadPoolExecutor(max_workers=max(1, len(shapes))) as pool:
            loading = {i: pool.submit(self._read_outlines, job, sid, i) for i in shapes}
            if job.frame is not None:
                index = next(i for i, c in enumerate(job.columns) if c["key"] == RUN_COLUMN)
                try:
                    self._hold(job, run)
                except OSError as exc:                   # cleared from the spool before it could be held
                    self._failed(job, index, exc, GONE)
                else:
                    try:
                        centres[index] = frame_centres(job.frame["cells_file"], int(job.frame["ncells"]))
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        self._failed(job, index, exc, "Could not read this iteration's cells")
            for index, future in loading.items():
                try:
                    outlines = future.result()
                    centres[index] = outline_centres(outlines)
                    cells[index] = outline_cells(outlines, centres[index])
                except Changed as changed:               # the others are read still: the next try takes them
                    moved = moved or self._written(job, index, changed)
                except Exception as exc:                 # this column fails; the others go on
                    if not job.cancel.is_set():          # stopped for a newer comparison: no failure of its own
                        self._failed(job, index, exc, "Could not read its outlines")
        job.check()
        if moved is not None:
            raise moved
        if not centres:
            raise CompareError("None of these could be read")
        job.note("outlines", time.monotonic() - mark)
        job.advance("area")

        # 2. The area they all cover, and each column's cells in it: inside a region run's own region when it is
        # compared, a rectangle (detail or tiled) or a lasso (other columns' cells beyond it are no part of what it
        # covers).
        mark = time.monotonic()
        limit, polygon = region_limit(run) if job.frame is not None else (None, None)
        area = common_area(centres, limit=limit, polygon=polygon, clip=job.clip)
        if area["box"] is None:
            raise CompareError("These share no tissue: no 100 µm square holds 3 cells of each", 409)
        inside = {i: in_area(*centres[i], area) for i in centres}
        for index, mask in inside.items():
            job.column(index, cells_in_area=int(mask.sum()), tiles=area["own"][index])
        job.set(area={"box": area["box"], "size_um": area["size_um"], "tiles": area["count"], "tile_um": TILE_UM})
        for index in inside:
            if job.columns[index]["key"] == RUN_COLUMN:
                np.save(job.folder / f"cells-{index}.npy", np.flatnonzero(inside[index]).astype("<i4"))
        job.note("area", time.monotonic() - mark)
        job.advance("counting")

        # 3. Counts of each element, its table's own or its outlines' (cached, or counted now), scored as they come: the
        # ready ones at once.
        counting = [i for i in shapes if i in inside]
        boxes = {i: count_box(area["box"], listed[job.columns[i]["key"]].get("extent")) for i in counting}
        waiting = []
        for index in counting:
            try:
                found = self.zarr.counts(sid, job.columns[index]["key"], boxes[index], cached_only=True, rule=job.rule,
                                         **job.tables.get(index, {}))
                if found is not None:
                    self._rows(job, sid, index, found, inside[index], cells[index])
            except Changed as changed:
                moved = self._written(job, index, changed)
                if moved is not None:
                    raise moved
                continue
            except Exception as exc:                     # this column fails; the others go on
                self._failed(job, index, exc, uncounted(job, index))
                continue
            if found is None:
                waiting.append(index)
                job.column(index, state="counting")
        job.set(of=job.of - (len(shapes) - len(waiting)))   # nothing to count for those cached, or failed
        mark = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, len(waiting))) as pool:
            try:
                pending = {i: pool.submit(self._count, job, sid, i, boxes[i], inside[i], cells[i]) for i in waiting}
                ready = [i for i in inside if job.columns[i]["state"] != "failed" and i not in pending]
                if ready and pending:
                    self._score_or_fail(job, ready)   # a failure here fails these columns alone: counting goes on
                else:
                    job.set(of=job.of - 1)               # one call of the worker only
                for index, future in pending.items():
                    try:
                        future.result()
                    except Cancelled:
                        raise
                    except Changed as changed:           # the others are counted still: the next try takes them
                        moved = moved or self._written(job, index, changed)
                    except Exception as exc:             # this column fails; the others go on
                        if not job.cancel.is_set():
                            self._failed(job, index, exc, uncounted(job, index))
                    job.advance()
            except BaseException:
                job.cancel.set()                         # stop the counting still going (its helpers are killed)
                raise
        job.check()
        if moved is not None:
            raise moved
        if waiting:
            job.note("counting", time.monotonic() - mark)
        left = [i for i, c in enumerate(job.columns) if c["state"] not in ("ready", "failed")]
        failure = self._score_or_fail(job, [i for i, c in enumerate(job.columns) if c["state"] != "failed"]) \
            if left else None
        for index in shapes:                             # still the versions compared, as the comparison is ready
            if job.columns[index]["state"] != "failed":
                try:
                    self._unmoved(job, sid, index)
                except Changed as changed:
                    moved = self._written(job, index, changed)
                    if moved is not None:
                        raise moved
        if not any(c["state"] == "ready" for c in job.columns):
            if failure is not None:
                raise failure                            # the comparison's own words: the worker's failure
            raise CompareError(next((c["error"] for c in job.columns if c["error"]), None) or "Nothing could be compared")
        job.note("total", time.monotonic() - job.started)
        job.set(state="ready", step="done", done=job.of)
        self._save(job, run)

    def _failed(self, job: CompareJob, index: int, exc: BaseException, plain: str):
        """Column `index` failed alone: the page's words for it (`said`), the exception's own in the log."""
        if technical(exc):
            print(f"SoPA companion: comparing {job.columns[index]['key']} of {job.run_id}: {type(exc).__name__}: {exc}",
                  flush=True)
        job.column(index, state="failed", error=said(exc, plain))

    def _read_outlines(self, job: CompareJob, sid: str, index: int) -> dict:
        """Column `index`'s exported outlines (ZarrSamples._outlines), of the version pinned (its stamp read before and
        after: an export of another version is never taken for it, nor is its refusal of one written over twice while
        exported taken for the column's failure: _or_moved); its export is stopped with the comparison (a newer one, the
        run gone, the engine closing)."""
        self._unmoved(job, sid, index)
        try:
            outlines = self.zarr._outlines(sid, job.columns[index]["key"], should_stop=job.cancel.is_set)
        except Exception:
            self._or_moved(job, sid, index)
            raise
        self._unmoved(job, sid, index)
        return outlines

    def _unmoved(self, job: CompareJob, sid: str, index: int):
        """Column `index`'s element is still the version pinned (job.versions), else Changed. A folder that does not
        answer just now says nothing (the counts' rows are matched with the outlines all the same)."""
        key, pinned = job.columns[index]["key"], job.versions.get(index)
        stamp = self.zarr.element_stamp(sid, key)
        if pinned is not None and stamp is not None and stamp != pinned:
            raise Changed(f"'{key}' was written again ({pinned} -> {stamp})")

    def _or_moved(self, job: CompareJob, sid: str, index: int):
        """After column `index`'s outline read or count failed: Cancelled when the comparison was stopped, Changed when
        its element is no longer the version pinned (ZarrSamples refuses an export or counts of an element written over
        while it worked, 409; a helper reading one may fail on it): compared again, never that column's failure. Else
        nothing: the failure is the column's own."""
        job.check()
        self._unmoved(job, sid, index)

    def _written(self, job: CompareJob, index: int, changed: Changed) -> Changed | None:
        """Column `index`'s element was written again while being compared (`changed`): the Changed that starts the
        comparison again, or on its last try (job.final) None, that column failed with WRITTEN_AGAIN (its numbers and
        points, if any, gone)."""
        if not job.final:
            return changed
        print(f"SoPA companion: comparing {job.columns[index]['key']} of {job.run_id}: {changed}", flush=True)
        with job.lock:
            job.columns[index].update(state="failed", error=WRITTEN_AGAIN, ready_s=None, **dict.fromkeys(NUMBERS),
                                      **OWN)
            job.files.pop(index, None)
            job.own.pop(index, None)
        return None

    def _count(self, job: CompareJob, sid: str, index: int, box, inside: np.ndarray, cells: tuple):
        try:
            path = self.zarr.counts(sid, job.columns[index]["key"], box, should_stop=job.cancel.is_set, rule=job.rule,
                                    **job.tables.get(index, {}))
        except Exception:
            self._or_moved(job, sid, index)
            raise
        job.check()
        self._rows(job, sid, index, path, inside, cells)

    def _rows(self, job: CompareJob, sid: str, index: int, path: Path, inside: np.ndarray, cells: tuple):
        """The rows of a counts file whose outline (by its position in the element) lies in the area, saved for the
        worker (cells-<index>.npy); the column then waits to be scored. Only counts of the outlines read: the element
        still the version pinned (else Changed: compared again), and each row scored the outline at its position (a
        position is never taken for a cell of other outlines): rows that misfit are left out, up to MISFITS_LEFT_OUT of
        them, else DISAGREE (of the version pinned: comparing again would read the same outlines and counts). A table's
        own count holds a row only for the outlines it names (by instance id): the share of the area's outlines it
        holds goes to the log when under all of them, and a table holding none of them fails its column (NOT_HELD)."""
        with np.load(path, allow_pickle=False) as counted:
            position = np.asarray(counted["index"], np.int64)
            ids, x, y = counted["ids"], counted["x"], counted["y"]
        self._unmoved(job, sid, index)
        bad = misfits(cells, position, ids, x, y)
        refused = bad.size > MISFITS_LEFT_OUT * position.size
        if bad.size:
            shown = position[bad[bad < position.size][:5]].tolist()
            print(f"SoPA companion: comparing {job.columns[index]['key']} of {job.run_id}: {bad.size} of {len(ids)} rows of "
                  f"its counts ({Path(path).name}) are not its outlines at version {job.versions.get(index)} (at "
                  f"positions {shown}{' …' if bad.size > len(shown) else ''}){'' if refused else ': left out'}",
                  flush=True)
        if refused:
            raise CompareError(DISAGREE)
        rows = np.flatnonzero(inside[position]) if position.size else np.zeros(0, np.int64)
        rows = np.setdiff1d(rows, bad, assume_unique=True) if bad.size else rows
        outlines = int(np.count_nonzero(inside))
        if index in job.tables and rows.size < outlines:
            print(f"SoPA companion: comparing {job.columns[index]['key']} of {job.run_id}: its table "
                  f"'{job.tables[index].get('table')}' holds {rows.size} of the {outlines} cells in the area "
                  f"({Path(path).name}){'' if rows.size else ': its column fails'}", flush=True)
            if not rows.size:
                raise CompareError(NOT_HELD)
        np.save(job.folder / f"cells-{index}.npy", rows.astype("<i4"))
        with job.lock:
            job.counts[index] = str(path)
            job.columns[index].update(state="working", cells_in_area=int(rows.size))

    def _score_or_fail(self, job: CompareJob, indices: list[int]) -> BaseException | None:
        """_score, where a failure is these columns' alone: a column scored by an earlier call keeps its numbers (and
        that call's picture), the others fail; the comparison goes on with what is left. The failure, or None."""
        try:
            self._score(job, indices)
            return None
        except Cancelled:
            raise
        except Exception as exc:                         # noqa: BLE001 - the worker's words are in the log already
            for index in indices:
                if job.columns[index]["ready_s"] is not None:
                    job.column(index, state="ready")
                else:
                    self._failed(job, index, exc, "The GPU worker could not compare it")
            return exc

    def _score(self, job: CompareJob, indices: list[int]):
        """One call of the worker's op "compare" on these columns: their numbers, the kinds and the points it wrote."""
        job.check()
        job.set(step="scoring")
        for index in indices:
            job.column(index, state="working")
        request = self._request(job, indices, job.layout + 1)
        mark = time.monotonic()
        reply = self._await(job, self.analysis.submit_compare(request))
        job.note(f"scoring{request['stage']}", time.monotonic() - mark)
        job.check()
        result = reply.get("compare") if isinstance(reply, dict) and reply.get("ok") else None
        if not isinstance(result, dict):                # the worker's own words are technical: in the log only
            print(f"SoPA companion: the GPU worker could not compare {', '.join(c['key'] for c in request['columns'])} of "
                  f"{job.run_id}: {reply.get('error') if isinstance(reply, dict) else reply!r}", flush=True)
            raise CompareError("The GPU worker could not compare these")
        given = {item.get("key"): item for item in result.get("columns") or [] if isinstance(item, dict)}
        kinds = result.get("kinds")
        now = round(time.monotonic() - job.started, 2)
        with job.lock:
            if isinstance(kinds, list):
                job.kinds = [{"index": k["index"], "name": str(k.get("name") or ""), "color": kind_colour(k)}
                             for k in kinds if isinstance(k, dict) and isinstance(k.get("index"), int) and k["index"] >= 0]
            else:
                job.kinds = None
            job.kinds_reason = result.get("kinds_reason")
            job.bounds, job.layout_error = result.get("bounds"), result.get("layout_error")
            job.own_bounds, job.own_error = result.get("own_bounds"), result.get("own_error")
            own_kind = result.get("own_layout")
            job.own_layout = own_kind if isinstance(own_kind, str) and own_kind else None
            kind = result.get("layout")
            job.layout_kind = kind if isinstance(kind, str) and kind else None
            for column in request["columns"]:
                index, item = column["index"], given.get(column["key"])
                target = job.columns[index]
                if item is None or item.get("error"):
                    if item is not None:
                        print(f"SoPA companion: the GPU worker could not read {column['key']} of {job.run_id}: "
                              f"{item.get('error')}", flush=True)
                    target.update(state="failed", error="The GPU worker could not read it" if item is not None
                                  else "The GPU worker sent nothing for it")
                    continue
                target.update({name: item.get(name) for name in NUMBERS}, state="ready")
                target["ready_s"] = target["ready_s"] if target["ready_s"] is not None else now
                path = Path(item.get("points_file") or column["out_points"])
                if not path.exists() and isinstance(item.get("points"), dict):
                    try:                                # a worker that answers its points inline
                        write_points(path, item["points"])
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"SoPA companion: the points of {column['key']} could not be kept: {exc}", flush=True)
                if path.exists():
                    job.files[index] = path
                else:                                   # no picture this time (too few cells): none from before either
                    job.files.pop(index, None)
                own = {name: Path(item[field]) for name, field in (("points", "own_points_file"), ("clusters", "clusters_file"))
                       if isinstance(item.get(field), str)}
                if path.exists() and len(own) == 2 and all(p.exists() for p in own.values()):
                    job.own[index] = own
                else:                                   # a column with no map of its own (too few cells)
                    job.own.pop(index, None)
                target.update(clusters=int(item.get("clusters") or 0),
                              clusters_matched=int(item.get("clusters_matched") or 0), own=index in job.own)
            job.layout += 1
            job.done = min(job.of, job.done + 1)
            job.seconds.setdefault("first", now)

    def _await(self, job: CompareJob, future):
        """The worker's reply to a submitted op (its Future). A comparison replaced while its op still waits in the queue
        takes it back (one already running is let finish, its answer dropped)."""
        from concurrent.futures import CancelledError
        deadline = time.monotonic() + WORKER_WAIT_S
        while True:
            try:
                return future.result(timeout=0.25)
            except TimeoutError:
                if job.cancel.is_set() and future.cancel():
                    raise Cancelled() from None
                if time.monotonic() > deadline:
                    future.cancel()
                    raise CompareError("The GPU worker did not answer in time") from None
            except CancelledError:                       # the engine closing: nobody will run it
                raise Cancelled() from None

    def _request(self, job: CompareJob, indices: list[int], stage: int) -> dict:
        """The worker's op "compare" (analysis/worker.py, its request and reply):
        every column names its cells on the area, so the worker scores them on this area. The run's column is its
        frame as AnalysisManager.compare_frame gave it (the worker reads proseg's count of it, `counts_file`: the copy
        the comparison holds when it was in the engine's spool, _hold)."""
        run = self.runs.get(job.run_id)
        thresholds = self.analysis.qc_thresholds() if hasattr(self.analysis, "qc_thresholds") else {}
        columns = []
        for index in indices:
            column = job.columns[index]
            item = {"index": index, "key": column["key"], "name": column["name"],
                    "cells": str(job.folder / f"cells-{index}.npy"),
                    "out_points": str(job.folder / f"points-{stage}-{index}.bin")}
            if column["key"] == RUN_COLUMN:
                item["frame"] = dict(job.frame)
            else:
                item["counts"] = job.counts[index]
            columns.append(item)
        folder = Path(job.model["path"])
        folder = folder if folder.suffix != ".npz" else folder.parent
        kinds = {"path": str(folder / "kinds.npz")}      # made there once, from the cells the model learned on
        if job.model.get("table") and run.meta.get("sample_path"):
            kinds["table"] = {"zarr": run.meta["sample_path"], "key": job.model["table"]}
        # Each column's own map ("Each its own") only in a call with every column: the run's column alone comes first
        # without waiting for one, and every own map is matched to the run's column in the same call.
        return {"run": job.run_id, "job": job.digest, "stage": stage, "model": str(folder), "model_id": job.model["id"],
                "kinds": kinds, "thresholds": thresholds, "area": dict(job.area), "max_points": MAX_POINTS,
                "seed": job.seed, "columns": columns, "own": len(indices) == len(job.columns)}

    def _save(self, job: CompareJob, run):
        """result.json beside the run (the same request later is answered from it), for a whole comparison only (a
        failure of the moment is never kept as its answer); the oldest comparisons beyond RESULTS_KEPT go."""
        if not job.whole():
            return
        try:
            if self.runs.get(job.run_id) is not run:
                return
        except RunError:                                 # deleted meanwhile
            return
        saved = {**job.public(), "files": {str(i): path.name for i, path in job.files.items()},
                 "own_files": {str(i): {k: p.name for k, p in names.items()} for i, names in job.own.items()}}
        temporary = job.folder / f".result.json.{os.getpid()}"
        temporary.write_text(json.dumps(saved))
        os.replace(temporary, job.folder / "result.json")
        try:
            kept = {path.name for path in job.files.values()} | {p.name for names in job.own.values() for p in names.values()}
            for path in job.folder.glob("points-*.bin"):      # the first layout's, replaced by the shared one
                if path.name not in kept:
                    path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            older = sorted((p for p in (run.dir / "compare").iterdir() if p.is_dir() and p != job.folder),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            for folder in older[RESULTS_KEPT - 1:]:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass
