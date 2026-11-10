"""Pure, shared stain-removal rules. Geometry and stored cell values are identical in the engine and helpers.

Areas and brightness cross the wire as float32. Comparisons explicitly promote those stored values to float64,
matching JavaScript numbers even on numpy 2 (NEP 50). Unknown values never remove a cell.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

MAX_SIZE = 1e6
MAX_BRIGHTNESS = 1e9


def centres_and_areas(offsets, coords):
    """Centroid (x, y) and area of each ring, from the outline arrays (rings may repeat their first point)."""
    offsets = np.asarray(offsets, np.int64)
    coords = np.asarray(coords, float)
    n = len(offsets) - 1
    if n <= 0 or not len(coords):
        return np.zeros((0, 2)), np.zeros(0)
    following = np.arange(1, len(coords) + 1)
    following[offsets[1:] - 1] = offsets[:-1]           # each ring's last vertex closes back to its first
    x, y = coords[:, 0], coords[:, 1]
    xn, yn = x[following], y[following]
    cross = x * yn - xn * y
    starts = offsets[:-1]
    signed = np.add.reduceat(cross, starts) / 2
    cx = np.add.reduceat((x + xn) * cross, starts)
    cy = np.add.reduceat((y + yn) * cross, starts)
    counts = np.diff(offsets)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.stack([cx / (6 * signed), cy / (6 * signed)], axis=1)
        mean = np.stack([np.add.reduceat(x, starts), np.add.reduceat(y, starts)], axis=1) / counts[:, None]
    flat = ~np.isfinite(centre).all(axis=1) | (np.abs(signed) < 1e-12)
    centre[flat] = mean[flat]                           # degenerate rings: the mean of their vertices
    return centre, np.abs(signed)


def inside(points, ring, eps=1e-6):
    """Even-odd point-in-polygon for many points and one lasso ring; a point on the ring (within eps µm) is inside, as
    shapely's intersects_xy has it in the save and the helper."""
    points, ring = np.asarray(points, float), np.asarray(ring, float)
    result = np.zeros(len(points), bool)
    if not len(points) or len(ring) < 3:
        return result
    lo, hi = ring.min(axis=0) - eps, ring.max(axis=0) + eps
    candidates = np.flatnonzero((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) &
                                (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]))
    if not len(candidates):
        return result
    px, py = points[candidates, 0], points[candidates, 1]
    hit = np.zeros(len(candidates), bool)
    edge = np.zeros(len(candidates), bool)
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        dx, dy = x2 - x1, y2 - y1
        edge |= ((np.abs((px - x1) * dy - (py - y1) * dx) <= eps * max(np.hypot(dx, dy), 1e-12)) &
                 (px >= min(x1, x2) - eps) & (px <= max(x1, x2) + eps) &
                 (py >= min(y1, y2) - eps) & (py <= max(y1, y2) + eps))
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        hit ^= crosses & (px < dx * (py - y1) / dy + x1)
    result[candidates] = hit | edge
    return result


def in_area(centres, areas):
    """A mask of cells whose ring centre lies in any removed lasso (including its edge)."""
    result = np.zeros(len(centres), bool)
    for area in areas:
        result |= inside(centres, area["points"] if isinstance(area, dict) else area)
    return result


def _range(value):
    if not isinstance(value, dict):
        return None
    limits = {end: float(value[end]) if value.get(end) is not None else None for end in ("min", "max")}
    return limits if any(v is not None for v in limits.values()) else None


def normalise(state):
    """Read the current state or an older minimum-only file, without changing its versions or the input."""
    state = state or {}
    size = state.get("size") if "size" in state else {"min": state.get("min_area_um2") or None, "max": None}
    bright = state.get("brightness") or {}
    if "brightness" not in state and state.get("min_brightness"):
        old = state["min_brightness"]
        bright = {old["channel"]: {"min": old["value"], "max": None}} if old["value"] else {}
    areas = deepcopy(state.get("areas") or [])
    return {"version": state.get("version", 0),
            "areas_version": state.get("areas_version", state.get("version", 0) if areas else 0),
            "next": state.get("next", max((a["id"] for a in areas), default=0) + 1), "areas": areas,
            "size": _range(size), "brightness": {c: r for c, v in bright.items() if (r := _range(v)) is not None}}


def check(body, kind, channels=()):
    """A size/brightness POST's range (or None to clear it); ValueError becomes the route's plain 422.

    `version` is checked against the current state under the run lock by the caller. Old minimum-only bodies are
    refused rather than silently clearing a rule. Brightness rules always name a channel, including when cleared.
    """
    if kind not in ("size", "brightness"):
        raise ValueError("Choose a cell size or brightness range")
    if not isinstance(body, dict) or not {"min", "max"} <= body.keys() or any(k in body for k in ("min_area_um2", "value")):
        raise ValueError("Choose a minimum and maximum, or leave them empty for no limit")
    if kind == "brightness" and body.get("channel") not in channels:
        raise ValueError(f"This image has no channel {body.get('channel')!r}")
    high = MAX_SIZE if kind == "size" else MAX_BRIGHTNESS
    # Negative float brightness is retained with no lower limit; an explicit bound follows the chooser's 0-based axis.
    for value in (body["min"], body["max"]):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or not 0 <= value <= high):
            name = "cell size" if kind == "size" else "brightness"
            raise ValueError(f"Choose a {name} from 0 to {high:,.0f}, or leave it empty for no limit")
    if body["min"] is not None and body["max"] is not None and body["min"] > body["max"]:
        raise ValueError("Make the minimum no more than the maximum")
    return _range(body)


def outside(values, limits):
    """Out-of-range mask with inclusive bounds, compared as doubles. NaN passes either bound."""
    values = np.asarray(values, np.float32).astype(np.float64)
    result = np.zeros(values.shape, bool)
    if limits:
        with np.errstate(invalid="ignore"):
            if limits.get("min") is not None:
                result |= values < float(limits["min"])
            if limits.get("max") is not None:
                result |= values > float(limits["max"])
    return result


def keep(area32, bright32_by_channel, in_area, rules):
    """Cells kept by every committed range and removed-area flag; absent brightness is unknown and kept."""
    state = normalise(rules)
    removed = np.asarray(in_area, bool).copy()
    if removed.shape != np.asarray(area32).shape or removed.ndim != 1:
        raise ValueError("The cell values do not match the outlines")
    removed |= outside(area32, state["size"])
    for channel, limits in state["brightness"].items():
        if channel in bright32_by_channel:
            values = np.asarray(bright32_by_channel[channel])
            if values.shape != removed.shape:
                raise ValueError("The cell values do not match the outlines")
            removed |= outside(values, limits)
    return ~removed


def digest(state):
    """Content identity for Undo and saved results; edit versions and lasso ids do not affect it."""
    state = normalise(state)
    rings = sorted(json.dumps(np.asarray(a["points"], float).tolist(), separators=(",", ":")) for a in state["areas"])
    content = {"areas": rings, "size": state["size"], "brightness": state["brightness"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
