"""Real vectorization and joining with labelled in-memory images; never creates a zarr store."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
from shapely.ops import unary_union
import zarr

from sopa_companion.zarr_bridge import stain as S

out, case = Path(sys.argv[1]), sys.argv[2]
(out / "stain").mkdir()
(out / "output").mkdir()
(out / "stain/control.json").write_text("{}")
if case in ("level", "level-budget", "level-instanseg"):
    yy, xx = np.ogrid[:200, :240]
    mask = ((xx - 110)**2 + (yy - 80)**2 < 25**2).astype(np.uint16)
    planes = np.stack([mask, 2 * mask])
    image = dict(key="im", channels=["nucleus", "body"],
                 to_frame=[[2, 0, 17], [0, 2, 23], [0, 0, 1]], levels=[
                     dict(path=f"level{k}", shape=[2, 1600 // f, 1920 // f], dims=["c", "y", "x"],
                          origin=[.75, 1.25], pixel=[.25*f, .25*f]) for k, f in enumerate((1, 2, 4, 8))])
    job = dict(path="never-opened", image=image, out=str(out), tile_pixels=1024, overlap_pixels=128,
               settings=dict(nucleus="nucleus", cytoplasm="body", diameter=24 if case == "level-budget" else None,
                             level=3), stitch="own", region=dict(kind="full", bbox=[18.5, 25.5, 978.5, 825.5]))
    events, calls = [], []
    if case == "level-instanseg":
        job["settings"].update(method="instanseg", channels=["nucleus", "body"], model="fluorescence_nuclei_and_cells",
                               output="cells", pixel_size=.5)
        job.update(instanseg_python=sys.executable, instanseg_module="sopa_companion.tests.fake_stain_patches")
        S.atomic_json(out / "stain/job.json", job)

    def segment(pixels):
        calls.append(pixels.shape)
        assert np.array_equal(pixels[..., 1], 2*pixels[..., 0])
        return pixels[..., 0].astype(np.uint32)

    with redirect_stdout(sys.stderr), patch.object(zarr, "open_array", return_value=planes) as opened, \
            patch.object(S, "emit", side_effect=lambda kind, **data: events.append((kind, data))):
        S.run(job, None if case == "level-instanseg" else segment)
    opened.assert_called_once_with("never-opened/level3", mode="r")
    shapes = S.read_shapes(out / "output/boundaries.geojson")
    assert len(shapes) == 1 and shapes[0].is_valid
    assert np.allclose(shapes[0].centroid.coords[0], [18.5 + 110.5*4, 25.5 + 80.5*4], atol=2)
    with np.load(out / "stain/tile-000000.masks.npz") as data:
        assert data["level"] == 3 and np.array_equal(data["geometry"], [18.5, 25.5, 4, 4])
    layout = json.loads((out / "stain/tiling.json").read_text())
    if case == "level-budget":
        assert all(h*10*w*10 <= 1024**2 for h, w, _ in calls)
        assert layout["tile_px"] == 102
    elif case == "level":
        assert layout["cell_px_median"] is not None
    else:
        assert layout["tile_px"] == 128
        hello = next(data for kind, data in events if kind == "hello")
        assert [hello[k] for k in ("level", "factor", "um_per_px", "model_pixel_size")] == [3, 8, 4, .5]
    assert events[-1][1]["covered"] is None
    print(json.dumps(dict(case=case, cells=len(shapes), tiles=len(calls))))
    sys.exit(0)
if case == "lasso":
    planes = np.ones((2, 200, 200), np.uint16)
    image = dict(key="im", channels=["DAPI", "Boundary"], to_frame=np.eye(3).tolist(),
                 levels=[dict(path="unused", shape=list(planes.shape), dims=["c", "y", "x"], origin=[0, 0], pixel=[.5, .5])])
    job = dict(path="never-opened", image=image, out=str(out), tile_pixels=128, overlap_pixels=32,
               settings=dict(nucleus="DAPI", cytoplasm="Boundary"),
               region=dict(kind="polygon", bbox=[0, 0, 100, 100], points=[[0, 0], [100, 0], [0, 100]]))
    path = out / "job.json"
    path.write_text(json.dumps(job))
    sys.argv = [str(Path(__file__).with_name("stain_geometry_check.py")), str(path)]
    with patch.object(zarr, "open_array", return_value=planes):
        runpy.run_path(sys.argv[0], run_name="__main__")
    sys.exit(0)
width, height = (512, 512) if case == "budget" else (1920, 1920 if case == "corner" else 1024)
yy, xx = np.ogrid[:height, :width]
mask = (((xx-960)**2 + (yy-(960 if case == "corner" else 512))**2) < 350**2).astype(np.uint16)
if case == "neighbours":
    mask[:] = 0
    for label, cx in enumerate([800, 1120], 1):
        mask[(xx-cx)**2 + (yy-512)**2 < 150**2] = label
planes = np.stack([mask, mask * 2])
image = dict(key="im", channels=["nucleus", "body"], to_frame=np.eye(3).tolist(),
             levels=[dict(path="unused", shape=list(planes.shape), dims=["c", "y", "x"], origin=[0, 0], pixel=[1, 1])])
diameter = 700 if case in ("diameter", "resume", "legacy") else 10 if case == "budget" else None
job = dict(path="never-opened", image=image, out=str(out), tile_pixels=1024, overlap_pixels=128,
           settings=dict(nucleus="nucleus", cytoplasm="body", diameter=diameter),
           region=dict(kind="full", bbox=[0, 0, width, height]))
calls, events = [], []


def segment(pixels):
    calls.append(pixels.shape)
    assert np.array_equal(pixels[..., 0] * 2, pixels[..., 1]) and not pixels[..., 2].any()
    if case == "budget":
        assert (pixels.shape[0]*3)*(pixels.shape[1]*3) <= 1024**2, pixels.shape
    return pixels[..., 0].astype(np.uint32)


def emit(ev, **values):
    events.append((ev, values))
    if case in ("resume", "legacy") and ev == "tile" and values["progress"]["done"] == 1:
        S.atomic_json(out / "stain/control.json", {"stop": True})


with redirect_stdout(sys.stderr), patch.object(zarr, "open_array", return_value=planes) as opened, \
        patch.object(S, "emit", side_effect=emit):
    S.run(job, segment)
    if case in ("resume", "legacy"):
        assert not events[-1][1]["progress"]["complete"]
        mask_path = out / "stain/tile-000000.masks.npz"
        before = mask_path.stat().st_mtime_ns
        plan = (out / "stain/plan.json").read_bytes()
        if case == "legacy":
            (out / "stain/plan.json").unlink()
        S.atomic_json(out / "stain/control.json", {})
        S.run(job, segment)
        assert mask_path.stat().st_mtime_ns == before
        if case == "resume":
            assert (out / "stain/plan.json").read_bytes() == plan
        else:
            assert len(json.loads((out / "stain/plan.json").read_text())) == 2
assert all(call.args == ("never-opened/unused",) and call.kwargs == {"mode": "r"} for call in opened.call_args_list)
records = json.loads((out / "stain/tiles.json").read_text())
shapes = S.read_shapes(out / "output/boundaries.geojson")
expected = 0 if case == "budget" else 2 if case == "neighbours" else 1
assert len(shapes) == expected, (case, len(shapes), expected)
if shapes:
    fragments = [p for i in range(len(records)) for p in S.read_shapes(out / f"stain/tile-{i:06d}.geojson")]
    combined = unary_union(fragments)
    assert abs(sum(p.area for p in shapes)-combined.area) < 1e-6, "Joining lost area or kept duplicate area"
    assert all(p.is_valid for p in shapes)
if case == "diameter":
    assert records[0]["box"][2] - records[1]["box"][0] == 512, records
assert events[-1][0] == "done" and events[-1][1]["progress"]["complete"]
print(json.dumps(dict(case=case, cells=len(shapes), tiles=len(calls), shapes=calls)))
