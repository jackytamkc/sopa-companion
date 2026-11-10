"""The image view a page draws (ZarrSamples.read_view): one image pixel per screen pixel, from the coarsest pyramid
level still at least that fine, area-averaged down to it. The finest level within the requested size could be a level
step (2x) coarser than the screen, and a whole-tissue view looked as if it had not loaded."""
from __future__ import annotations

import numpy as np

from sopa_companion.engine.zarr_samples import _shrink
from sopa_companion.tests.test_samples_alpha import inspection, samples, stand_in, store, write_image


def opened(tmp_path, levels=3):
    """A 400 x 400 pixel image over 200 x 200 um of the sample (0.5 um per pixel at level 0, halved per level: 1, 2)
    and its sample id."""
    path = store(tmp_path)
    planes = (np.arange(2 * 400 * 400, dtype=np.uint32) % 6000).astype(np.uint16).reshape(2, 400, 400)
    zarr = samples(tmp_path)
    stand_in(zarr, inspection(path, images=[write_image(path, "morphology_focus", planes, ["DAPI", "Boundary"],
                                                        levels=levels)]))
    return zarr, zarr.open(str(path))["id"]


def test_a_view_is_never_coarser_than_the_screen(tmp_path):
    zarr, sid = opened(tmp_path)
    # 200 um across on 300 screen pixels: 0.67 um per pixel. Level 1 (1 um) would be coarser than the screen, so
    # level 0 (0.5 um) is read and averaged down to the 300 pixels asked for.
    view, where = zarr.read_view(sid, None, 0, [0, 0, 200, 200], max_px=300)
    assert where["level"] == 0 and view.shape == (300, 300) and (where["width"], where["height"]) == (300, 300)
    assert (where["x0"], where["y0"], where["x1"], where["y1"]) == (0.0, 0.0, 200.0, 200.0)   # the same ground
    # exactly a level's size: that level, not shrunk
    view, where = zarr.read_view(sid, None, 0, [0, 0, 200, 200], max_px=200)
    assert where["level"] == 1 and view.shape == (200, 200)
    # zoomed in past the finest level: the finest level as it is
    view, where = zarr.read_view(sid, None, 0, [50, 50, 75, 75], max_px=400)
    assert where["level"] == 0 and view.shape == (50, 50)
    # far out: the coarsest level that is still at least as fine as the screen, averaged down
    view, where = zarr.read_view(sid, None, 0, [0, 0, 200, 200], max_px=64)
    assert where["level"] == 2 and view.shape == (64, 64)


def test_shrinking_averages_the_pixels_each_one_covers():
    plane = np.array([[0, 100, 200, 300], [0, 100, 200, 300]], np.uint16)
    small = _shrink(plane, 2)
    assert small.dtype == np.dtype("<u2") and small.shape == (1, 2) and small.tolist() == [[50, 250]]
    assert _shrink(plane, 4) is plane                                      # already small enough: as it is
    tall = _shrink(np.full((1000, 10), 65535, np.uint16), 100)
    assert tall.shape == (100, 1) and int(tall.max()) == 65535             # the aspect kept, the range too
