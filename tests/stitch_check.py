"""Tile ownership and sopa's solve_conflicts through the real run loop, with a known answer; never opens a zarr.

    python stitch_check.py <empty folder> grid|big|disagree|tissue|smaller|long   (sopa environment)

The image's first channel is the true label image and the second each pixel's position, so the stand-in model returns
exactly the cells inside the tile it is given — or, in "disagree", hides one cell from the tile that owns it.
"""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import shapely
from shapely.affinity import affine_transform
from sopa.shapes import vectorize
import zarr

from sopa_companion.zarr_bridge import stain as S

out, case = Path(sys.argv[1]), sys.argv[2]
(out / "stain").mkdir()
(out / "output").mkdir()
(out / "stain/control.json").write_text("{}")
size = 1200
radius = (36, 44) if case == "big" else (7, 16)
rng = np.random.default_rng(3)
yy, xx = np.ogrid[:size, :size]
truth = np.zeros((size, size), np.uint16)
centres = []
hidden = shrunk = None
if case in ("disagree", "smaller"):
    # Owned by the second tile (centre right of the overlap's middle) yet wholly inside the first: the first tile has
    # it as a spare; a reviewer misses it ("disagree") or sees a smaller cell inside it ("smaller").
    centres.append((490.0, 300.0, 10))
    hidden, shrunk = (1, None) if case == "disagree" else (None, 1)
if case == "long":
    # Long thin cells: 150 × 6 px bars. Their equivalent diameter (34 px) would leave the 64 px overlap unchanged.
    bars = []
    while len(bars) < 40:
        x, y = int(rng.integers(2, size - 152)), int(rng.integers(2, size - 8))
        if all(not (x < b[0] + 155 and b[0] < x + 155 and y < b[1] + 11 and b[1] < y + 11) for b in bars):
            bars.append((x, y))
    for label, (x, y) in enumerate(bars, 1):
        truth[y:y + 6, x:x + 150] = label
else:
    while len(centres) < (40 if case == "big" else 160):
        r = int(rng.integers(*radius))
        x, y = rng.uniform(r + 2, size - r - 2, 2)
        if all(np.hypot(x - a, y - b) > r + c + 3 for a, b, c in centres):
            centres.append((x, y, r))
    for label, (x, y, r) in enumerate(centres, 1):
        truth[(xx - x) ** 2 + (yy - y) ** 2 < r ** 2] = label
# Each pixel's position, x + 2000 y (exact in float32), so the stand-in model knows where its tile is.
where = (np.arange(size, dtype=np.float32)[None, :] + 2000 * np.arange(size, dtype=np.float32)[:, None])
planes = np.stack([truth.astype(np.float32), where])
image = dict(key="im", channels=["nucleus", "x"], to_frame=np.eye(3).tolist(),
             levels=[dict(path="unused", shape=list(planes.shape), dims=["c", "y", "x"], origin=[0, 0], pixel=[1, 1])])
job = dict(path="never-opened", image=image, out=str(out), tile_pixels=512, overlap_pixels=64, stitch="own",
           settings=dict(nucleus="nucleus", cytoplasm="x"), region=dict(kind="full", bbox=[0, 0, size, size]))
if case == "tissue":
    job["tissue"] = [[[0, 0], [600, 0], [600, size], [0, size], [0, 0]]]
events = []


def segment(pixels):
    labels = pixels[..., 0].astype(np.uint32)
    x0, y0 = int(pixels[0, 0, 1]) % 2000, int(pixels[0, 0, 1]) // 2000
    if hidden is not None and x0 != 0:                        # every tile but those starting at column 0
        labels[labels == hidden] = 0
    if shrunk is not None and x0 != 0 and (labels == shrunk).any():
        rows, cols = np.ogrid[:labels.shape[0], :labels.shape[1]]
        labels[labels == shrunk] = 0
        labels[(cols + x0 - 490) ** 2 + (rows + y0 - 300) ** 2 < 4 ** 2] = shrunk
    return labels


def emit(ev, **values):
    events.append((ev, values))


with redirect_stdout(sys.stderr), patch.object(zarr, "open_array", return_value=planes), \
        patch.object(S, "emit", side_effect=emit):
    S.run(job, segment)

tiling = json.loads((out / "stain/tiling.json").read_text())
records = json.loads((out / "stain/tiles.json").read_text())
final = S.read_shapes(out / "output/boundaries.geojson")
live = [p for i in range(len(records)) for p in S.read_shapes(out / f"stain/tile-{i:06d}.geojson")]
expected = [affine_transform(p, [1, 0, 0, 1, .5, .5]) for p in vectorize(truth).geometry]
if case == "tissue":
    expected = [p for p in expected if p.centroid.x < 600]
    assert all(p.centroid.x < 600 for p in final), "a cell centred outside the tissue was kept"
    plan = next(v for e, v in events if e == "plan")
    assert plan["tiles_skipped"] > 0 and plan["tiling"]["tiles_skipped"] == plan["tiles_skipped"], plan

# sopa's vectorize smooths relative to the cells it is given, so a tile's outline differs slightly from the full image's.
ious = []
tree = shapely.STRtree(final)
for cell in expected:
    found = [final[k] for k in tree.query(cell, predicate="intersects")]
    ious.append(max((cell.intersection(f).area / cell.union(f).area for f in found), default=0))
matched = int(sum(i > (.9 if case == "long" else .95) for i in ious))
pairs = tree.query(final, predicate="intersects")
pairs = pairs[:, pairs[0] < pairs[1]]
overlaps = int(sum(final[i].intersection(final[j]).area > 1e-6 for i, j in pairs.T))
result = dict(case=case, cells=len(final), expected=len(expected), matched=matched, overlaps=overlaps,
              live=len(live), tiles=len(records), overlap_px=tiling["overlap_px"], min_iou=round(min(ious), 3),
              cell_diameter_px=tiling["cell_diameter_px"])
assert len(final) == len(expected) and matched == len(expected), result         # every cell once, whole
assert overlaps == 0, result                                                     # no duplicates or cut pieces
# The live view is the final set; "smaller" shows the small cell beside the restored whole one until the join.
assert len(live) == len(final) + (case == "smaller"), result
if case in ("big", "long"):
    assert tiling["overlap_px"] > 64 and tiling["overlap_px"] >= 1.25 * tiling["cell_diameter_px"] - 1, result
else:
    assert tiling["overlap_px"] == 64, result
print(json.dumps(result))
