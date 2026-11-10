"""Run in the sopa environment by test_stain_zarr: real tiling/vectorization/conflicts, known masks."""
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys

import numpy as np
import shapely
from sopa_companion.zarr_bridge import stain as S

job = json.loads(Path(sys.argv[1]).read_text())
root = Path(job["out"]) / "stain"
clip = shapely.Polygon(job["region"]["points"])
size, overlap = S.tile_layout(job)
boxes = [b for b in S.tile_boxes([0, 0, 200, 200], size, overlap)
         if clip.intersects(shapely.box(*(v*.5 for v in b)))]
assert list(S.tile_boxes([0, 0, 128, 128], size, overlap)) == [[0, 0, 128, 128]]
events, calls = [], []
original_emit = S.emit


def segment(pixels):
    i = len(calls)
    calls.append(i)
    x0, y0, x1, y1 = boxes[i]
    assert pixels.shape == (y1-y0, x1-x0, 3)
    assert pixels[:, :, 0].max() > 0 and pixels[:, :, 1].max() > 0 and not pixels[:, :, 2].any()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = np.zeros(xx.shape, np.uint32)
    mask[(xx-110)**2+(yy-80)**2 < 14**2] = 1  # crosses the x tile seam
    mask[(xx-50)**2+(yy-50)**2 < 10**2] = 2
    return mask


def emit(ev, **fields):
    events.append(dict(ev=ev, **copy.deepcopy(fields)))
    if ev == "tile" and fields["progress"]["done"] == 1:
        S.atomic_json(root / "control.json", {"stop": True})


with redirect_stdout(sys.stderr):
    S.emit = emit
    S.run(job, segment)
    assert events[-1]["progress"]["done"] == 1 and not events[-1]["progress"]["complete"]
    before = (root / "tile-000000.masks.npz").stat().st_mtime_ns
    S.atomic_json(root / "control.json", {})
    S.emit = lambda ev, **fields: events.append(dict(ev=ev, **fields))
    S.run(job, segment)
    shapes = S.read_shapes(Path(job["out"]) / "output/boundaries.geojson")
    assert (root / "tile-000000.masks.npz").stat().st_mtime_ns == before
    assert len(calls) == len(boxes)
    assert events[-1]["ev"] == "done" and events[-1]["progress"]["complete"]
    assert len(shapes) == 2, len(shapes)
    clip = shapely.Polygon(job["region"]["points"])
    assert all(p.is_valid and clip.covers(p) for p in shapes)
    small = min(shapes, key=lambda p: p.area)
    assert np.allclose(small.centroid.coords[0], [25.25, 25.25], atol=.2), small.centroid  # pixel centres + scale
    # Geometric holes survive clipping and the saved GeoJSON; only the drawing path is exterior-only.
    hole = shapely.Polygon([(0,0),(10,0),(10,10),(0,10)], [[(2,2),(4,2),(4,4),(2,4)]])
    S.write_shapes(root / "holes", S.polygons_in_region([hole], shapely.box(0,0,8,8)), [0,0,8,8])
    assert len(S.read_shapes(root / "holes.geojson")[0].interiors) == 1
print(json.dumps({"cells": len(shapes), "tiles": len(calls), "area": S.summary(shapes)}))
