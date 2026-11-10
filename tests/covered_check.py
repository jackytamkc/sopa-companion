"""Coverage and joining on synthetic geometry; no sample, weights, display or GPU."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from sopa_companion.engine.runs import region_spec
from sopa_companion.zarr_bridge import stain as S


def shape(spec):
    return Polygon(spec["points"]) if spec["kind"] == "polygon" else box(*spec["box"])


def geometry(case):
    region = dict(kind="full", bbox=[0, 0, 300, 300])
    clip = box(*region["bbox"])
    boxes = list(S.tile_boxes([0, 0, 300, 300], 120, 20))
    done = 4
    if case == "first":
        done = 1
    elif case == "row":
        done = 3
    elif case == "complete":
        done = len(boxes)
    elif case == "empty":
        done = 0
    elif case == "image":
        clip = box(20, 30, 280, 260)
        done = len(boxes)
    elif case == "lasso":
        region = dict(kind="polygon", bbox=[0, 0, 300, 300], points=[[0, 0], [300, 0], [300, 300]])
        clip = Polygon(region["points"])
    elif case == "split":
        # One connected lasso whose top has two separate arms; cutting below their join gives two pieces.
        clip = Polygon([(0, 0), (80, 0), (80, 150), (200, 150), (200, 0), (300, 0), (300, 300), (0, 300)])
        region = dict(kind="polygon", bbox=[0, 0, 300, 300], points=list(clip.exterior.coords))
        done = 3
    elif case == "sparse":
        region = dict(kind="full", bbox=[0, 0, 10000, 10000])
        clip = box(*region["bbox"])
        boxes = [[5000, 4000, 6000, 5000], [8000, 4000, 9000, 5000], [5000, 4900, 6000, 5900]]
        done = 1
    elif case == "tiny":
        boxes = [[0, 0, 10, 10], [8, 0, 18, 10]]
        done = 1
    records = [dict(box=b) for b in boxes[:done]]
    covered = S.covered_area(region, clip, boxes, records)
    if case == "complete":
        assert covered is None
        return
    assert covered["done"] == done and covered["of"] == len(boxes)
    if case in ("empty", "tiny"):
        assert covered["region"] is None
        return
    actual = shape(covered["region"])
    assert actual.is_valid and actual.geom_type == "Polygon" and clip.covers(actual)
    assert covered["region"] == region_spec(covered["region"], region["bbox"])
    if case == "sparse":
        assert actual.bounds == (5000, 4000, 6000, 5000) and actual.area == 1000000
    elif case == "first":
        assert actual.bounds == (0, 0, 100, 120) and actual.area == 12000
    elif case == "row":
        assert actual.bounds == (0, 0, 300, 100) and actual.area == 30000
    elif case == "staircase":
        assert actual.area == 42000 and len(actual.exterior.coords) == 7
    elif case == "image":
        assert actual.equals(clip) and covered["why"] == "image"
    elif case == "split":
        assert covered["pieces"] == 2 and actual.bounds == (200, 0, 300, 100)
    print(json.dumps(covered))


def joining(root, case):
    (root / "stain").mkdir()
    (root / "output").mkdir()
    boxes = [[0, 0, 120, 120], [100, 0, 220, 120], [200, 0, 300, 120],
             [0, 100, 120, 220], [100, 100, 220, 220], [200, 100, 300, 220]]
    records = [dict(tile=i, box=b, cells=1) for i, b in enumerate(boxes[:4])]
    job = dict(path="missing-sample", out=str(root), settings=dict(method="instanseg", level=0),
               image=dict(channels=["DAPI"], to_frame=np.eye(3).tolist(), levels=[
                   dict(path="missing-array", shape=[1, 220, 300], dims=["c", "y", "x"], pixel=[1, 1], origin=[0, 0])]),
               tile_pixels=120, overlap_pixels=20, stitch="own", region=dict(kind="full", bbox=[0, 0, 300, 220]))
    S.atomic_json(root / "stain/job.json", job)
    S.atomic_json(root / "stain/plan.json", boxes)
    S.atomic_json(root / "stain/tiles.json", records)
    S.atomic_json(root / "stain/control.json", dict(paused=True))
    for i, b in enumerate(boxes[:4]):
        S.write_shapes(root / f"stain/tile-{i:06d}", [box(b[0]+10, b[1]+10, b[0]+30, b[1]+30)], b)
    # A late spare crosses the staircase's inner edge. Join must cut it at x=100.
    S.atomic_json(root / "stain/spare-000003.json", dict(spares=[
        dict(owner=4, geometry=shapely.geometry.mapping(box(90, 150, 115, 180)))]))
    before = {p: p.read_bytes() for p in (root / "stain").glob("tile-*")}
    events = []
    import zarr
    from sopa_companion.zarr_bridge import stain_instanseg as I
    with patch.object(zarr, "open_array", side_effect=AssertionError("Join read an image")), \
            patch.object(S, "Cellpose", side_effect=AssertionError("Join loaded Cellpose")), \
            patch.object(I, "InstanSegProcess", side_effect=AssertionError("Join started InstanSeg")), \
            patch.object(S, "checkpoint", side_effect=AssertionError("Join waited for control")), \
            patch.object(S, "emit", side_effect=lambda ev, **data: events.append(dict(ev=ev, **data))):
        if case == "join-cli":
            with patch.object(sys, "argv", ["stain", "--join-only", str(root / "stain/job.json")]):
                assert S.main() == 0
        else:
            S.run({**job, "join_only": True})
    assert events[-1]["ev"] == "done" and events[-1]["progress"]["resolved"]
    assert not events[-1]["progress"]["complete"]
    covered = shape(events[-1]["covered"]["region"])
    cells = S.read_shapes(root / "output/boundaries.geojson")
    assert len(cells) == 5 and all(covered.covers(c) for c in cells)
    spare = next(c for c in cells if c.bounds[1] == 150)
    assert spare.bounds == (90, 150, 100, 180)
    assert before == {p: p.read_bytes() for p in before}
    result = json.loads((root / "stain/result.json").read_text())
    assert result["covered"] == events[-1]["covered"]
    print(json.dumps(dict(cells=len(cells), covered=result["covered"])))


if __name__ == "__main__":
    root, case = Path(sys.argv[1]), sys.argv[2]
    with redirect_stdout(sys.stderr):
        if case.startswith("join"):
            joining(root, case)
        else:
            geometry(case)
