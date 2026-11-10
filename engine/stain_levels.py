"""Image-pyramid choices and pixel units shared by the engine and stain helpers (numpy only).

A choice is one of the image's own pyramid levels, named as SpatialData names it: level k is the image's DataTree
child "scale{k}" (level_name), scale0 the image as stored. Stored Cellpose settings always use scale0 pixels. A
level's factor is the geometric mean of its x and y factors; its exact x/y geometry still places outlines and patches
in the sample.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import numpy as np

from .stain_intensity import geometry

OLD_LABEL = re.compile(r"1/(\d+(?:\.\d+)?)")     # an earlier version's label of a smaller copy ("1/2"; "Full": scale0)
MIN_ACROSS = 32           # the fewest pixels a run's region may span at its level, both ways (engine/stain.py)


def level_name(k):
    """Pyramid level k by its SpatialData name, the image's DataTree child: scale0 (as stored), scale1, …"""
    return f"scale{k}"


def resolution_name(record):
    """The level name of a stored resolution record (a run's, a saved layer's, a patch set's): from its level; a record
    of an earlier version with only a factor (or only a label such as "1/2") by the nearest power of two of it. ""
    when it says none of these."""
    record = record if isinstance(record, dict) else {}
    k = record.get("level")
    if type(k) is int and k >= 0:
        return level_name(k)
    f = record.get("factor")
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f <= 0:
        label = record.get("label") if isinstance(record.get("label"), str) else ""
        found = OLD_LABEL.fullmatch(label)
        f = 1. if label == "Full" else float(found[1]) if found and float(found[1]) > 0 else None
    # Halves round up, as the page's Math.round (stain.mjs resolutionName) does.
    return level_name(max(0, math.floor(math.log2(f) + .5))) if f else ""


def _usable(level):
    dims, shape = level.get("dims", []), level.get("shape", [])
    return (len(dims) == len(shape) and dims[-2:] == ["y", "x"] and
            all(n > 0 and (d in ("c", "x", "y") or n == 1) for d, n in zip(dims, shape)))


def factor(image, k):
    """Linear pixel-size factor relative to scale0, allowing the pyramid's rounding along each axis."""
    _, _, x0, y0 = geometry(image, image["levels"][0])
    _, _, x, y = geometry(image, image["levels"][k])
    return math.sqrt((x / x0) * (y / y0))


def offered(image):
    """Every level of this image's pyramid a stain run can read, scale0 first, each named by level_name.

    Accept x/y factors within 2% (real pyramids round their dimensions), each level coarser than the one offered before
    it (by more than that rounding) and at least MIN_ACROSS pixels both ways (no region could be run at a smaller one).
    A public sample image already carries these records as `scales`; accepting that form lets settings validation use
    the sample detail too.
    """
    levels = image.get("levels", [])
    if not isinstance(levels, list):
        return deepcopy(image.get("scales", []))
    if not levels or not _usable(levels[0]):
        return []
    try:
        matrix = np.asarray(image["to_frame"], float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[[0, 1], [1, 0]], 0):
            return []
        _, _, x0, y0 = geometry(image, levels[0])
        if not all(math.isfinite(v) and v > 0 for v in (x0, y0)):
            return []
        choices = [dict(level=0, factor=1., um_per_px=x0, um_per_px_y=y0, label=level_name(0))]
        for k, level in enumerate(levels[1:], 1):
            if not _usable(level) or min(level["shape"][-2:]) < MIN_ACROSS:
                continue
            _, _, x, y = geometry(image, level)
            fx, fy = x / x0, y / y0
            if not all(math.isfinite(v) and v > 0 for v in (fx, fy)) or not math.isclose(fx, fy, rel_tol=.02):
                continue
            f = math.sqrt(fx * fy)
            if f > choices[-1]["factor"] * 1.02:
                choices.append(dict(level=k, factor=f, um_per_px=x, um_per_px_y=y, label=level_name(k)))
        return choices
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return []


def job_level(job):
    """(index, level metadata, factor) for a job; jobs made before Resolution use scale0."""
    k = job.get("settings", {}).get("level", 0)
    if type(k) is not int or not any(s["level"] == k for s in offered(job["image"])):
        raise ValueError("Choose a resolution this image has")
    return k, job["image"]["levels"][k], factor(job["image"], k)


def at_level(settings, f):
    """Copy scale0 settings into level pixels, without revalidating a possibly very small level diameter."""
    result = deepcopy(settings)
    if isinstance(result.get("diameter"), (int, float)):
        result["diameter"] /= f
    if result.get("min_size") is not None and result["min_size"] != -1:
        result["min_size"] = max(0, round(result["min_size"] / (f * f)))
    params = result.get("normalize")
    if isinstance(params, dict):
        for key in ("sharpen_radius", "smooth_radius"):
            if params.get(key) is not None:
                params[key] /= f
        if params.get("tile_norm_blocksize") is not None:
            params["tile_norm_blocksize"] = max(0, round(params["tile_norm_blocksize"] / f))
    return result


def full_from_gui(values, set_um_per_px, image_um_per_px):
    """GUI patch pixels → this image's scale0 pixels, rounded and clamped to the picker's accepted settings."""
    from .cellpose_settings import NUMBERS, NORMALIZE_NUMBERS
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
               for v in (set_um_per_px, image_um_per_px)):
        raise ValueError("The patches' pixel size could not be read")
    result = at_level(values, image_um_per_px / set_um_per_px)
    for key in ("diameter", "min_size"):
        if result.get(key) is not None and not (key == "min_size" and result[key] == -1):
            _, low, high = NUMBERS[key]
            result[key] = min(high, max(low, result[key]))
    for key, (_, low, high) in NORMALIZE_NUMBERS.items():
        if (result.get("normalize") or {}).get(key) is not None:
            result["normalize"][key] = min(high, max(low, result["normalize"][key]))
    return result

