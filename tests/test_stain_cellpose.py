"""Opt-in real cpsam smoke on a small, read-only part of the harness crop.

    cd <folder holding sopa_companion> && SOPA_COMPANION_STAIN_SMOKE=1 env -u DISPLAY PYTHONPATH=$PWD \\
        sopa_companion/.venv/bin/python -m pytest -q -s sopa_companion/tests/test_stain_cellpose.py

Never collected by the ordinary gate; skips without the sopa environment or the harness's crop (conftest.py).
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from conftest import harness_crop, sopa_python, wait_for
from sopa_companion.engine.runs import Runs
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples


def production_idle():
    # Same listening-port test as ss -ltn, also works where netlink sockets cannot be opened.
    for file in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(file).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == 8777:
                return False
    return True


if os.environ.get("SOPA_COMPANION_STAIN_SMOKE") == "1":
    def test_real_cellpose_crop(config):
        sopa, crop = sopa_python(), harness_crop()           # skipped without either
        assert production_idle(), "Production is open: real Cellpose must wait"
        before = {str(p.relative_to(crop)): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in crop.rglob("*") if p.is_file()}
        config["stain"].update(python=sopa, gpu=1, tile_pixels=256, overlap_pixels=32)
        config["zarr"]["python"] = sopa
        zarr = ZarrSamples(config)
        detail = zarr.open(str(crop))
        image = zarr._image_info(detail["id"], detail["image_key"])
        nucleus = next((c for c in image["channels"] if "dapi" in c.lower()), image["channels"][0])
        x, y, _, _ = detail["extent"]
        region = [x+200, y+200, x+300, y+300]   # 100 µm, about 4 overlapping 256-pixel tiles
        runs = Runs(config, Samples(config), zarr)
        try:
            t0 = time.monotonic()
            run = runs.get(runs.create(detail["id"], kind="stain", region={"kind": "box", "box": region},
                           stain={"image": image["key"], "nucleus": nucleus})["id"])
            runs.control(run.meta["id"], "resume")
            first = None
            deadline = time.monotonic()+900
            while run.worker.is_alive():
                assert production_idle(), "Production opened during the smoke"
                if run.meta["progress"]["done"] and first is None:
                    first = time.monotonic()-t0
                if time.monotonic() > deadline:
                    pytest.fail("Cellpose exceeded 900 s")
                time.sleep(.2)
            assert run.meta["state"] == "done", (run.meta, (run.dir / "stain.log").read_text()[-6000:])
            assert run.meta["summary"]["cells"] > 0
            assert run.meta["stain_runtime"]["model"] == "cpsam"
            assert run.meta["stain_runtime"]["cuda_visible_devices"] == "1"
            assert run.meta["stain_runtime"]["nice"] == 19
            assert run.meta["progress"]["done"] == run.meta["progress"]["total"] >= 2
            for path in (run.dir / "stain").glob("*.masks.npz"):
                with np.load(path) as data:
                    assert data["masks"].dtype == np.uint32 and max(data["masks"].shape) <= 256
            print(json.dumps({"run_dir": str(run.dir), "first_tile_s": first, "seconds": time.monotonic()-t0,
                              "summary": run.meta["summary"], "runtime": run.meta["stain_runtime"]}))
        finally:
            runs.close(save=False)
        after = {str(p.relative_to(crop)): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in crop.rglob("*") if p.is_file()}
        assert after == before
