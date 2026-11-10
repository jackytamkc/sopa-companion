"""The measuring harness's folder in small, for a computer without the harness (CI's tests): a crop made with sopa,
registered and opened as the harness registers and opens its own, so that the tests reading the harness's crop run.
Run it with the sopa environment's Python, then point SOPA_COMPANION_HARNESS_DIR at DIR (empty or not there yet):

    python tests/make_mini_harness.py DIR

DIR/fixtures/mini-crop.zarr                  the crop (make_spatialdata.py crop)
DIR/fixtures/fixtures.json                   its record, with the keys conftest.harness_crop reads (the harness's own)
DIR/var/cache/zarr/<sample id>/inspect.json  the engine's inspection of it, from one open by the engine's sample code
                                             (its data folder DIR/var, its state folder DIR/state), as the harness's
                                             engine leaves it; the crop's path in it is DIR's with its links resolved
The other folders of the harness (its calibrations, models and runs) are not made: the tests that read them still skip.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))        # the folder holding the sopa_companion package
sys.path.insert(0, str(HERE))                   # make_spatialdata, beside this file


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    root = Path(argv[0]).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        print(f"make_mini_harness.py: {root} must be an empty folder, or not there yet", file=sys.stderr)
        return 1
    started = time.monotonic()
    crop = root / "fixtures" / "mini-crop.zarr"
    crop.parent.mkdir(parents=True, exist_ok=True)
    import make_spatialdata
    make_spatialdata.crop(str(crop))

    from sopa_companion.engine.config import load_config
    from sopa_companion.engine.zarr_samples import ZarrSamples
    samples = ZarrSamples(load_config(overrides={"paths": {"data_dir": str(root / "var"), "state_dir": str(root / "state")},
                                                 "zarr": {"python": sys.executable}}))
    try:
        detail = samples.open(str(crop))
    finally:
        samples.close()
    record = {"crop": {"path": str(crop), "points_key": detail["points_key"], "box_um": [0.0, 0.0, 512.0, 512.0],
                       "extent": detail["extent"], "transcripts": detail["transcripts"],
                       "seconds": round(time.monotonic() - started, 1)}}
    (root / "fixtures" / "fixtures.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"make_mini_harness.py: {crop} made and opened in {record['crop']['seconds']} s; "
          f"SOPA_COMPANION_HARNESS_DIR={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
