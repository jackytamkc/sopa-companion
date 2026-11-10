import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import harness_crop, harness_path, needs_sopa, sopa_python


def run_check(tmp_path, *args):
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"} | {
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    done = subprocess.run([sopa_python(), str(Path(__file__).with_name("tissue_check.py")), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_sopa
def test_whole_cells_kept_by_their_centre_and_tiles_away_from_tissue_skipped(tmp_path):
    """Cells are never cut at the tissue edge; a tile that does not touch the tissue is not segmented."""
    assert run_check(tmp_path, "unit") == {"unit": "ok"}


# A disc of tissue (60 px radius, at 0.5 µm per pixel: 30 µm around (50, 50) µm) on glass: an H&E picture (8-bit RGB:
# pink on near-white) and a 3-channel fluorescence image (16-bit: bright on black).
DISCS = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, zarr
from sopa_companion.zarr_bridge.tissue import detect
root = sys.argv[1]
yy, xx = np.mgrid[0:200, 0:200]
disc = (xx - 100) ** 2 + (yy - 100) ** 2 <= 60 ** 2
found = {}
for name, channels, dtype, glass, tissue in (("he", ["r", "g", "b"], "uint8", (245, 242, 246), (200, 90, 160)),
                                             ("if3", ["DAPI", "CD45", "PanCK"], "uint16", (0, 0, 0), (900, 300, 600))):
    data = np.stack([np.where(disc, tissue[c], glass[c]) for c in range(3)]).astype(dtype)
    zarr.create_array(store=f"{root}/{name}", shape=data.shape, dtype=dtype, chunks=data.shape)[:] = data
    image = {"channels": channels, "to_frame": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
             "levels": [{"path": name, "shape": list(data.shape), "dims": ["c", "y", "x"], "origin": [0, 0], "pixel": [1, 1]}]}
    found[name] = [{"bounds": list(p.bounds), "area": p.area, "centre": p.contains(p.centroid) and p.centroid.coords[0]}
                   for p in detect(root, image)]
print(json.dumps(found))
'''


@needs_sopa
def test_tissue_on_an_h_and_e_picture_is_the_tissue_not_the_glass(tmp_path):
    """Detection took the brightest channel everywhere ("staining"), and on H&E the glass is the brightest: no
    tissue was found, and "Tissue only" refused every run. An 8-bit RGB picture uses sopa's own "saturation" mode;
    other images (a 3-channel 16-bit one too) keep "staining"."""
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"} | {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    done = subprocess.run([sopa_python(), "-c", DISCS, str(tmp_path)], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    found = json.loads(done.stdout.strip().splitlines()[-1])
    for name in ("he", "if3"):
        [tissue] = found[name]                          # one polygon: the disc, a little wider (sopa's 5 % expansion)
        x0, y0, x1, y1 = tissue["bounds"]
        assert 15 < x0 < 22 and 15 < y0 < 22 and 78 < x1 < 85 and 78 < y1 < 85, (name, tissue)
        assert np.pi * 30 ** 2 < tissue["area"] < np.pi * 35 ** 2 and tissue["centre"] == pytest.approx([50, 50], abs=1)


@needs_sopa
def test_detection_in_memory_matches_the_samples_region_of_interest(tmp_path):
    # The harness's crop of a real sample and its engine's cache (conftest: skipped where there is no harness).
    crop, cache = harness_crop(), harness_path("var", "cache", "zarr")
    inspect = next((p for p in cache.glob("*/inspect.json")
                    if json.loads(p.read_text()).get("path") == str(crop)), None)
    if inspect is None:
        pytest.skip("the crop has not been inspected by the harness engine yet")
    before = sorted(str(p.relative_to(crop)) + str(p.stat().st_mtime_ns) for p in crop.rglob("*"))
    result = run_check(tmp_path, crop, inspect)
    # The stored region was found on the whole slide's lowest level; the crop's own lowest level is finer.
    assert result["iou"] > .85 and result["seconds"] < 10
    assert sorted(str(p.relative_to(crop)) + str(p.stat().st_mtime_ns) for p in crop.rglob("*")) == before   # read-only
