"""Joining overlapping tiles without seams: each tile keeps the cells it owns, then sopa's `solve_conflicts`.

A tile's own part of the plan is the tile minus half of each overlap strip it shares with a planned neighbour; the own
parts of neighbouring tiles meet at the middle of their strip and never overlap. A tile keeps the cells whose centre
lies in its own part. A cell cut by an edge that has a neighbour across it has its centre within one radius of that
edge, inside the neighbour's own part, and the neighbour sees it whole because the overlap is wider than the cells
(the first tiles measure their longest sides: `needed_overlap`). Each cell is therefore kept once, whole, as soon as its tile is
finished, so the live view already shows the final cells.

The cells a tile does not own are its spare cells. The owning tile normally has them; when it does not (the model
disagrees across the overlap) the spare comes back once the owning tile is finished — as sopa would keep it — so no cell
either tile found is lost. sopa's `solve_conflicts` runs over all kept cells at the end, merging the rare duplicates
left.
"""
import math

import numpy as np


def own_part(boxes, index):
    """[left, top, right, bottom] of tile `index`'s own part (image pixels; ±inf where no neighbour shares an edge)."""
    boxes = np.asarray(boxes, float)
    x0, y0, x1, y1 = boxes[index]
    a0, b0, a1, b1 = boxes.T
    other = np.arange(len(boxes)) != index
    rows = other & (b0 < y1) & (b1 > y0)                 # beside it, sharing rows
    cols = other & (a0 < x1) & (a1 > x0)                 # above or below it, sharing columns
    left = rows & (a0 < x0) & (x0 < a1)
    right = rows & (a0 < x1) & (x1 < a1)
    top = cols & (b0 < y0) & (y0 < b1)
    bottom = cols & (b0 < y1) & (y1 < b1)
    return [float(np.max((x0 + a1[left]) / 2)) if left.any() else -math.inf,
            float(np.max((y0 + b1[top]) / 2)) if top.any() else -math.inf,
            float(np.min((x1 + a0[right]) / 2)) if right.any() else math.inf,
            float(np.min((y1 + b0[bottom]) / 2)) if bottom.any() else math.inf]


def owners(points, boxes, candidates):
    """For pixel points, the index among `candidates` (tile indices) whose own part holds each point, or -1."""
    result = np.full(len(points), -1)
    points = np.asarray(points, float).reshape(-1, 2)
    for k in candidates:
        left, top, right, bottom = own_part(boxes, k)
        inside = ((points[:, 0] >= left) & (points[:, 0] < right) & (points[:, 1] >= top) & (points[:, 1] < bottom) &
                  (result < 0))
        result[inside] = k
    return result


def neighbours(boxes, index):
    """Planned tiles overlapping tile `index` (itself excluded)."""
    boxes = np.asarray(boxes, float)
    x0, y0, x1, y1 = boxes[index]
    near = (boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0)
    near[index] = False
    return np.flatnonzero(near).tolist()


def label_centres(masks, origin):
    """(labels present, their centres in image pixels) for a label image whose (0, 0) pixel is at `origin`."""
    rows, cols = np.nonzero(masks)
    labels = masks[rows, cols]
    if not len(labels):
        return np.zeros(0, int), np.zeros((0, 2))
    count = np.bincount(labels).astype(float)
    present = np.flatnonzero(count[1:]) + 1
    # Pixel centres: column c covers [x0 + c, x0 + c + 1).
    cx = np.bincount(labels, weights=cols + origin[0] + .5)[present] / count[present]
    cy = np.bincount(labels, weights=rows + origin[1] + .5)[present] / count[present]
    return present, np.stack([cx, cy], axis=1)


def own_masks(masks, boxes, index):
    """(owned, spare): the tile's mask split into the cells it owns and the others (labels unchanged, the rest 0)."""
    masks = np.asarray(masks)
    labels, centres = label_centres(masks, boxes[index][:2])
    if not len(labels):
        return masks, np.zeros_like(masks)
    owner = owners(centres, boxes, [index, *neighbours(boxes, index)])
    keep = np.zeros(int(masks.max()) + 1, bool)
    keep[labels[owner == index]] = True
    spare = np.zeros_like(keep)
    spare[labels[owner != index]] = True
    return np.where(keep[masks], masks, 0).astype(masks.dtype), np.where(spare[masks], masks, 0).astype(masks.dtype)


def whole_cell_sizes(masks):
    """The longest side (pixels) of each cell in a tile that does not touch its border: the overlap must hold a cell
    whole along any axis, which an equivalent diameter understates for long, thin cells."""
    masks = np.asarray(masks)
    if not masks.size or not masks.max():
        return np.zeros(0)
    rows, cols = np.nonzero(masks)
    labels = masks[rows, cols]
    n = int(masks.max()) + 1
    low, high = np.full((n, 2), np.iinfo(np.int64).max), np.full((n, 2), -1)
    np.minimum.at(low, labels, np.stack([rows, cols], axis=1))
    np.maximum.at(high, labels, np.stack([rows, cols], axis=1))
    border = np.unique(np.concatenate([masks[0], masks[-1], masks[:, 0], masks[:, -1]]))
    present = high[:, 0] >= 0
    present[0] = False
    present[border] = False
    return (high[present] - low[present] + 1).max(axis=1).astype(float)


def cell_diameter(sizes):
    """The 99th percentile of the whole cells' longest sides (pixels), or None with fewer than 5 of them."""
    sizes = np.asarray(sizes, float)
    if len(sizes) < 5:
        return None
    return float(np.percentile(sizes, 99))


def needed_overlap(diameter, overlap, size):
    """The overlap (pixels) that keeps every measured cell whole in the tile owning it: 1.25 × the cell diameter."""
    if diameter is None:
        return overlap
    return int(min(size // 2, max(overlap, math.ceil(1.25 * diameter))))


def uncovered(kept, candidates):
    """The candidate cells no kept cell already covers (by half of the candidate's area)."""
    import shapely
    if not len(candidates):
        return []
    if not len(kept):
        return list(candidates)
    kept, candidates = np.asarray(kept, dtype=object), np.asarray(candidates, dtype=object)
    covered = np.zeros(len(candidates))
    pairs = shapely.STRtree(kept).query(candidates, predicate="intersects")
    if pairs.size:
        np.add.at(covered, pairs[0], shapely.area(shapely.intersection(candidates[pairs[0]], kept[pairs[1]])))
    return [c for c, area in zip(candidates, covered) if area < .5 * c.area]


def solve_conflicts(shapes, tiles):
    """sopa's `solve_conflicts` over the kept cells of all tiles (cells from one tile are never merged together)."""
    if not len(shapes):
        return []
    from sopa.segmentation import solve_conflicts as sopa_solve
    from sopa.shapes import ensure_polygon
    resolved = sopa_solve(list(shapes), threshold=.5, patch_indices=np.asarray(tiles))
    return [ensure_polygon(p) for p in resolved.geometry]
