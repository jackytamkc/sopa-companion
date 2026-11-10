"""A cell's brightness in a stain channel, for the brightness rule (remove the cells whose nucleus stain is below a
threshold, measured on the brightest part of each cell).

A cell's brightness is the mean of its brightest 5 µm² of the channel's full-resolution pixels, among the pixels whose
centres lie inside its outline (the exterior ring the outline arrays hold): the k brightest, k = 5 µm² in pixels (a cell
smaller than that: all its pixels). A fixed area, not a share of the cell: a cell with a nucleus scores high whatever
its size, one with none like the background around it. (The brightest 10% of each cell, the first choice, scored big
cells whose nucleus covers under a tenth of the outline, often large cells, like cells with no nucleus: in a test those
scored about a third of what they score now, close to cells with no nucleus.) A ring over no pixel centre has none
(NaN), and no rule removes it.

Numpy and zarr-python only: the engine and the zarr helper (sopa environment) import this same module, so the outlines,
the counts, the training patches and Save as shapes all leave out the same cells. Pixels are found by an even-odd
scanline over the rings themselves (a pixel belongs to every ring around its centre, so overlapping cells each keep
theirs); the image is read a block at a time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

AREA_UM2 = 5.0                     # the brightest part measured: its area, µm²
CACHE_TAG = "top5um2"              # in the names of stored values: another measure is never read as this one
BLOCK = 2048                       # image px: cells are measured a block of the image at a time


def default_channel(job: dict) -> str | None:
    """The channel a run's rule starts from: its nucleus channel; for InstanSeg the first of its channels whose name
    contains "dapi" (any case), else its first; with neither, the image's first channel."""
    settings, names = job.get("settings") or {}, (job.get("image") or {}).get("channels") or []
    if settings.get("nucleus") in names:
        return settings["nucleus"]
    chosen = [c for c in settings.get("channels") or [] if c in names]
    return next((c for c in chosen if "dapi" in c.lower()), chosen[0] if chosen else names[0] if names else None)


def geometry(image: dict, level: dict) -> tuple[float, float, float, float]:
    """(x0, y0, sx, sy): frame position of the level's pixel (0, 0) corner and frame units per pixel (as
    engine/zarr_samples.py ZarrSamples._geometry, repeated here so the sopa environment needs no engine import)."""
    matrix = np.asarray(image["to_frame"], float)
    ox, oy = level["origin"]
    px, py = level["pixel"]
    return (float(matrix[0, 0] * ox + matrix[0, 2]), float(matrix[1, 1] * oy + matrix[1, 2]),
            float(matrix[0, 0] * px), float(matrix[1, 1] * py))


def job_reader(job: dict, channel: str, array=None):
    """(read, geometry, size) for `channel` of a stain run's image at full resolution: read(c0, r0, c1, r1) returns that
    window as a 2-D array, size is (height, width). `array`: the level's zarr array when the caller holds it open."""
    image = job["image"]
    level = image["levels"][0]
    index = image["channels"].index(channel)
    if array is None:
        import zarr
        array = zarr.open_array(store=str(Path(job["path"]) / level["path"]), mode="r")
    dims = level["dims"]

    def read(c0, r0, c1, r1):
        key = tuple(index if d == "c" else slice(r0, r1) if d == "y" else slice(c0, c1) if d == "x" else 0 for d in dims)
        return np.asarray(array[key])

    return read, geometry(image, level), tuple(level["shape"][-2:])


def rings(shapes) -> tuple[np.ndarray, np.ndarray]:
    """Outline arrays (offsets, coords) of shapely polygons' exterior rings, as zarr_bridge/stain.py write_shapes keeps
    them (float32 coordinates), so the helper measures exactly what the engine measures."""
    coords, offsets = [], [0]
    for shape in shapes:
        ring = np.asarray(shape.exterior.coords, dtype=np.float32)
        coords.append(ring)
        offsets.append(offsets[-1] + len(ring))
    return (np.asarray(offsets, np.int64),
            np.concatenate(coords).astype(np.float64) if coords else np.zeros((0, 2)))


def _crossings(px, py, lo_row, hi_row):
    """Each edge's crossings of the pixel-centre rows it spans: (edge index, row, x). An edge spans the rows whose centre
    y (row + 0.5) lies in [its lower end, its upper end): every vertex counts once, horizontal edges never."""
    x1, y1, x2, y2 = px[0], py[0], px[1], py[1]
    low, high = np.minimum(y1, y2), np.maximum(y1, y2)
    first = np.maximum(np.ceil(low - 0.5), lo_row).astype(np.int64)
    last = np.minimum(np.ceil(high - 0.5) - 1, hi_row).astype(np.int64)
    counts = np.where(y1 != y2, np.maximum(last - first + 1, 0), 0)
    edge = np.repeat(np.arange(len(counts)), counts)
    if not len(edge):
        return edge, edge, np.zeros(0)
    step = np.arange(len(edge)) - np.repeat(np.cumsum(counts) - counts, counts)
    row = first[edge] + step
    t = (row + 0.5 - y1[edge]) / (y2[edge] - y1[edge])
    return edge, row, x1[edge] + t * (x2[edge] - x1[edge])


def _block(read, px, py, starts, ends, cells, size, k):
    """The brightness of `cells` (ring indices) whose pixel rings lie in one block: their window read once."""
    height, width = size
    lengths = ends[cells] - starts[cells]
    vertex = np.repeat(starts[cells], lengths) + np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    owner = np.repeat(np.arange(len(cells)), lengths)                  # each vertex's cell, 0..len(cells)-1
    following = vertex + 1
    last = np.cumsum(lengths) - 1
    following[last] = vertex[last - lengths + 1]                       # each ring's last vertex closes to its first
    xs, ys = px[vertex], py[vertex]
    c0 = max(0, int(math.floor(xs.min())))
    r0 = max(0, int(math.floor(ys.min())))
    c1 = min(width, int(math.ceil(xs.max())) + 1)
    r1 = min(height, int(math.ceil(ys.max())) + 1)
    out = np.full(len(cells), np.nan, np.float32)
    if c1 <= c0 or r1 <= r0:
        return out
    edge, row, x = _crossings((xs, px[following]), (ys, py[following]), r0, r1 - 1)
    if not len(edge):
        return out
    cell = owner[edge]
    order = np.lexsort((x, row, cell))
    cell, row, x = cell[order], row[order], x[order]
    # Even-odd: the crossings of one cell on one row pair up in order; a pair spans the pixels whose centres lie in
    # [x of the first, x of the second).
    group = np.flatnonzero(np.r_[True, (cell[1:] != cell[:-1]) | (row[1:] != row[:-1])])
    position = np.arange(len(cell)) - np.repeat(group, np.diff(np.r_[group, len(cell)]))
    pair = np.flatnonzero((position % 2 == 0)[:-1] & (cell[1:] == cell[:-1]) & (row[1:] == row[:-1]))
    if not len(pair):
        return out
    first = np.maximum(np.ceil(x[pair] - 0.5), c0).astype(np.int64)
    stop = np.minimum(np.ceil(x[pair + 1] - 0.5), c1).astype(np.int64)   # exclusive
    span = np.maximum(stop - first, 0)
    keep = span > 0
    first, span, cell_of, row_of = first[keep], span[keep], cell[pair][keep], row[pair][keep]
    if not len(span):
        return out
    plane = read(c0, r0, c1, r1)
    column = np.repeat(first, span) + np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span)
    values = plane[np.repeat(row_of, span) - r0, column - c0]
    which = np.repeat(cell_of, span)
    if np.issubdtype(values.dtype, np.integer) and values.dtype.itemsize <= 2 and values.min(initial=0) >= 0:
        key = np.sort((which.astype(np.int64) << 16) | values.astype(np.int64))
        which, ordered = key >> 16, (key & 0xFFFF).astype(np.float32)
    else:
        order = np.lexsort((values, which))
        which, ordered = which[order], values[order].astype(np.float32)
    count = np.bincount(which, minlength=len(cells))
    end = np.cumsum(count)
    top = np.minimum(count, k)                                          # the brightest k of each cell (all, if fewer)
    finite = np.isfinite(ordered)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, ordered, 0), dtype=np.float64)])
    # Unknown pixels make only their own cell unknown: one NaN must not poison the shared cumulative sum.
    has = (count > 0) & (np.bincount(which[~finite], minlength=len(cells)) == 0)
    out[has] = ((total[end[has]] - total[end[has] - top[has]]) / top[has]).astype(np.float32)
    return out


def brightness(read, frame, offsets, coords, size, *, block=BLOCK, area=AREA_UM2, progress=None) -> np.ndarray:
    """Per ring of the outline arrays (offsets, coords in frame units, µm; a ring may repeat its first point): the mean
    of the channel's brightest `area` µm² of pixels whose centres lie inside it, NaN for a ring over none.
    read(c0, r0, c1, r1): a window of the channel at full resolution; frame: (x0, y0, sx, sy) of that level; size: its
    (height, width). The rings are taken a block of `block` px at a time, by where their bounding box starts;
    progress(fraction) after each block."""
    offsets = np.asarray(offsets, np.int64)
    coords = np.asarray(coords, np.float64).reshape(-1, 2)
    n = max(len(offsets) - 1, 0)
    out = np.full(n, np.nan, np.float32)
    if not n or not len(coords):
        return out
    x0, y0, sx, sy = frame
    k = max(1, int(round(area / abs(sx * sy))))                        # pixels in the brightest part
    px, py = (coords[:, 0] - x0) / sx, (coords[:, 1] - y0) / sy       # pixel (c, r) covers [c, c+1) x [r, r+1)
    starts, ends = offsets[:-1], offsets[1:]
    lengths = ends - starts
    real = np.flatnonzero(lengths >= 3)
    if not len(real):
        return out
    owner = np.repeat(np.arange(n), lengths)
    left, top = np.full(n, np.inf), np.full(n, np.inf)
    np.minimum.at(left, owner, px[starts[0]:ends[-1]])
    np.minimum.at(top, owner, py[starts[0]:ends[-1]])
    left, top = left[real], top[real]
    tile = np.floor(top / block).astype(np.int64) * (1 << 32) + np.floor(left / block).astype(np.int64)
    order = np.argsort(tile, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tile[order][1:] != tile[order][:-1], True])
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cells = real[order[a:b]]
        out[cells] = _block(read, px, py, starts, ends, cells, size, k)
        if progress:
            progress((i + 1) / (len(bounds) - 1))
    return out
