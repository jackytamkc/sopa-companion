"""Real sopa tile/seam code with a separate fake inference process, entirely in temporary files."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import zarr

from sopa_companion.zarr_bridge import stain as S

out = Path(sys.argv[1])
(out / "stain").mkdir()
(out / "output").mkdir()
(out / "stain/control.json").write_text("{}")
yy, xx = np.ogrid[:200, :240]
mask = np.zeros((200, 240), np.uint16)
mask[(xx-110)**2+(yy-80)**2 < 25**2] = 1
mask[(xx-40)**2+(yy-40)**2 < 12**2] = 2
planes = np.stack([mask, mask+10, mask+20, mask*2])
image = dict(key="im", channels=["a", "b", "c", "d"], to_frame=np.eye(3).tolist(),
    levels=[dict(path="unused", shape=list(planes.shape), dims=["c", "y", "x"], origin=[0, 0], pixel=[.5, .5])])
job = dict(path="never-opened", image=image, out=str(out), tile_pixels=128, overlap_pixels=32,
    instanseg_python=sys.executable, instanseg_module="sopa_companion.tests.fake_instanseg",
    settings=dict(method="instanseg", model="fluorescence_nuclei_and_cells", channels=image["channels"],
                  pixel_size=.5, output="cells", resolve_cells=False),
    region=dict(kind="full", bbox=[0, 0, 120, 100]))
(out / "stain/job.json").write_text(json.dumps(job))
events, original = [], S.emit
paused = threading.Event()


def emit(ev, **data):
    events.append(dict(ev=ev, **data))
    if ev == "tile" and data["progress"]["done"] == 1:
        S.atomic_json(out / "stain/control.json", {"paused": True})
    if ev == "state" and data["state"] == "paused":
        paused.set()


def stop_paused():
    assert paused.wait(60)
    time.sleep(.2)
    assert len(json.loads((out / "stain/tiles.json").read_text())) == 1
    S.atomic_json(out / "stain/control.json", {"stop": True})


thread = threading.Thread(target=stop_paused, daemon=True)
thread.start()
with patch.object(zarr, "open_array", return_value=planes) as opened, patch.object(S, "emit", side_effect=emit):
    S.run(job)
    thread.join(2)
    assert not thread.is_alive()
    assert events[-1]["ev"] == "done" and not events[-1]["progress"]["complete"]
    before = (out / "stain/tile-000000.masks.npz").read_bytes()
    S.atomic_json(out / "stain/control.json", {})
    S.run(job)
    assert events[-1]["progress"]["complete"]
    assert events[-1]["summary"]["cells"] == 2
    assert (out / "stain/tile-000000.masks.npz").read_bytes() == before
    assert all(call.kwargs["mode"] == "r" for call in opened.call_args_list)
shapes = S.read_shapes(out / "output/boundaries.geojson")
assert len(shapes) == 2 and all(p.is_valid for p in shapes)
assert np.allclose(max(shapes, key=lambda p: p.area).centroid.coords[0], [55.25, 40.25], atol=.3)
assert not list((out / "stain").glob("instanseg-*.npy"))
print("InstanSeg process, four channels, pause/stop/resume, seams and pixel scale passed")
