"""A cell's brightness in a stain channel (engine/stain_intensity.py): the 90th percentile of its pixels whose centres lie
inside the outline, checked against a brute-force count and on the cases that break scanlines."""
import json

import numpy as np
import pytest

from sopa_companion.engine import stain_intensity as I


def arrays(polygons):
    coords = [np.asarray(p, float) for p in polygons]
    offsets = np.cumsum([0] + [len(c) for c in coords])
    return offsets, np.concatenate(coords) if coords else np.zeros((0, 2))


def reader(image):
    return lambda c0, r0, c1, r1: image[r0:r1, c0:c1]


def brute(image, frame, polygon, area=5.0):
    """Every pixel centre tested on its own (even-odd), then the mean of the brightest `area` µm² by sorting."""
    x0, y0, sx, sy = frame
    ring = (np.asarray(polygon, float) - [x0, y0]) / [sx, sy]
    rows, cols = np.mgrid[:image.shape[0], :image.shape[1]]
    cx, cy = cols.ravel() + 0.5, rows.ravel() + 0.5
    inside = np.zeros(len(cx), bool)
    for (xa, ya), (xb, yb) in zip(ring, np.roll(ring, -1, axis=0)):
        if ya == yb:
            continue
        crosses = (np.minimum(ya, yb) <= cy) & (cy < np.maximum(ya, yb))
        inside ^= crosses & (cx < xa + (cy - ya) * (xb - xa) / (yb - ya))
    values = np.sort(image.ravel()[inside]).astype(np.float64)
    k = max(1, int(round(area / abs(sx * sy))))
    return np.nan if not len(values) else float(np.float32(values[-k:].mean()))


def test_a_square_takes_the_pixels_whose_centres_it_holds_and_the_mean_of_its_brightest_5_um2():
    image = (np.arange(6)[:, None] * 10 + np.arange(6)[None, :]).astype(np.uint16)
    offsets, coords = arrays([[[1, 1], [3, 1], [3, 3], [1, 3]]])
    # Centres (1.5, 1.5) ... (2.5, 2.5): values 11, 12, 21, 22, all within 5 µm² of 1 µm pixels: their mean.
    assert I.brightness(reader(image), (0, 0, 1, 1), offsets, coords, image.shape).tolist() == [16.5]
    # The frame's origin and pixel size: the same square 0.5 µm per pixel from (100, 50).
    offsets, coords = arrays([[[100.5, 50.5], [101.5, 50.5], [101.5, 51.5], [100.5, 51.5], [100.5, 50.5]]])
    assert I.brightness(reader(image), (100, 50, 0.5, 0.5), offsets, coords, image.shape).tolist() == [16.5]
    # A larger cell: only its brightest 5 µm² count (here 5 pixels of 1 µm²: 51..55 of a 6 x 6 square, mean 53).
    offsets, coords = arrays([[[0, 0], [6, 0], [6, 6], [0, 6]]])
    assert I.brightness(reader(image), (0, 0, 1, 1), offsets, coords, image.shape).tolist() == [53]


def test_a_big_cell_with_a_small_nucleus_scores_like_its_nucleus_not_like_its_cytoplasm():
    # 0.2125 µm pixels, as Xenium's: a 30 x 30 µm cell (~20,000 pixels) with a 4 µm-wide nucleus (~350 pixels, 1.8 % of
    # it) at 900 over cytoplasm at 40, and a cell of the same size with no nucleus.
    frame = (0, 0, 0.2125, 0.2125)
    image = np.full((300, 300), 40, np.uint16)
    image[60:79, 60:79] = 900
    offsets, coords = arrays([[[0, 0], [30, 0], [30, 30], [0, 30]], [[31, 31], [61, 31], [61, 61], [31, 61]]])
    got = I.brightness(reader(image), frame, offsets, coords, image.shape)
    assert got.tolist() == [900, 40]


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8, np.float32, np.int16])
def test_random_outlines_match_a_brute_force_count(dtype):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 250, (90, 120)).astype(dtype)
    frame = (3.25, -2.5, 0.4, 0.4)
    polygons = []
    for _ in range(60):                                  # star-shaped rings, some leaving the image, some overlapping
        cx, cy = rng.uniform(-2, 52), rng.uniform(-6, 36)
        angles = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(3, 14)))
        radius = rng.uniform(0.2, 6, len(angles))
        polygons.append(np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1))
    offsets, coords = arrays(polygons)
    expected = [brute(image, frame, p) for p in polygons]
    for block in (7, 2048):                               # many blocks or one: the same numbers
        got = I.brightness(reader(image), frame, offsets, coords, image.shape, block=block)
        np.testing.assert_array_equal(got, np.asarray(expected, np.float32))
    assert np.isfinite(expected).sum() > 40 and np.isnan(expected).sum() >= 1


def test_overlapping_cells_each_keep_their_pixels_and_a_ring_over_no_centre_has_none():
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    offsets, coords = arrays([[[0, 0], [4, 0], [4, 4], [0, 4]], [[2, 2], [6, 2], [6, 6], [2, 6]],
                              [[5.1, 5.1], [5.4, 5.1], [5.4, 5.3]],            # inside one pixel, off its centre
                              [[1, 1], [2, 1]]])                               # two points: no ring
    got = I.brightness(reader(image), (0, 0, 1, 1), offsets, coords, image.shape)
    assert got[0] == brute(image, (0, 0, 1, 1), [[0, 0], [4, 0], [4, 4], [0, 4]])
    assert got[1] == brute(image, (0, 0, 1, 1), [[2, 2], [6, 2], [6, 6], [2, 6]])
    assert np.isnan(got[2]) and np.isnan(got[3])


def test_the_window_is_read_once_per_block_and_only_where_the_cells_are():
    image = np.ones((40, 40), np.uint16)
    reads = []
    def read(c0, r0, c1, r1):
        reads.append((c0, r0, c1, r1))
        return image[r0:r1, c0:c1]
    offsets, coords = arrays([[[1, 1], [3, 1], [3, 3]], [[2, 2], [5, 2], [5, 5]], [[30, 30], [33, 30], [33, 33]]])
    fractions = []
    I.brightness(read, (0, 0, 1, 1), offsets, coords, image.shape, block=16, progress=fractions.append)
    assert sorted(reads) == [(1, 1, 6, 6), (30, 30, 34, 34)] and fractions == [0.5, 1.0]


def test_default_channel_and_the_reader_of_a_run_image(tmp_path):
    job = {"settings": {"method": "cellpose", "nucleus": "DAPI", "cytoplasm": "Boundary"},
           "image": {"channels": ["Boundary", "DAPI"]}}
    assert I.default_channel(job) == "DAPI"
    job = {"settings": {"method": "instanseg", "channels": ["Boundary", "dapi_stain"]},
           "image": {"channels": ["Boundary", "dapi_stain"]}}
    assert I.default_channel(job) == "dapi_stain"
    job["settings"] = {"method": "instanseg", "channels": ["Boundary"]}
    assert I.default_channel(job) == "Boundary"
    job["settings"] = {}
    assert I.default_channel(job) == "Boundary"
    zarr = pytest.importorskip("zarr")
    planes = np.stack([np.zeros((20, 30), np.uint16), np.arange(600, dtype=np.uint16).reshape(20, 30)])
    store = tmp_path / "sample.zarr"
    array = zarr.open_array(store=str(store / "images/he/0"), mode="w", shape=planes.shape, dtype=planes.dtype,
                            chunks=(1, 8, 8))
    array[:] = planes
    job = {"path": str(store), "settings": {"nucleus": "DAPI"},
           "image": {"channels": ["Boundary", "DAPI"], "to_frame": np.eye(3).tolist(),
                     "levels": [{"path": "images/he/0", "shape": [2, 20, 30], "dims": ["c", "y", "x"],
                                 "origin": [0, 0], "pixel": [0.5, 0.5]}]}}
    read, frame, size = I.job_reader(json.loads(json.dumps(job)), "DAPI")
    assert frame == (0.0, 0.0, 0.5, 0.5) and size == (20, 30)
    assert read(2, 1, 5, 3).tolist() == planes[1, 1:3, 2:5].tolist()
    offsets, coords = arrays([[[1, 1], [3, 1], [3, 3], [1, 3]]])               # pixels 2..5 x 2..5
    assert I.brightness(read, frame, offsets, coords, size).tolist() == [brute(planes[1], frame, coords)]
