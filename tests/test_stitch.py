"""Seamless tile joining: ownership, spares, the measured overlap and sopa's solve_conflicts (zarr_bridge/stitch.py)."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from conftest import needs_sopa, sopa_python
from sopa_companion.zarr_bridge.stitch import cell_diameter, needed_overlap, own_masks, own_part, whole_cell_sizes

GRID = [[0, 0, 512, 512], [448, 0, 960, 512], [0, 448, 512, 960], [448, 448, 960, 960]]


def test_own_parts_meet_in_the_middle_of_each_overlap_and_never_overlap():
    assert own_part(GRID, 0) == [-np.inf, -np.inf, 480, 480]
    assert own_part(GRID, 3) == [480, 480, np.inf, np.inf]
    assert own_part([[0, 0, 512, 512]], 0) == [-np.inf, -np.inf, np.inf, np.inf]
    # A tile skipped outside the tissue is no neighbour: the edge facing it belongs to this tile.
    assert own_part([GRID[0], GRID[1]], 0) == [-np.inf, -np.inf, 480, np.inf]


def test_a_tile_keeps_whole_cells_centred_in_its_part_and_sets_aside_the_rest():
    masks = np.zeros((512, 512), np.uint32)
    masks[100:120, 100:120] = 1           # centre (110, 110): its own
    masks[200:220, 470:490] = 2           # centre (480, 210): past the middle of the strip, the right tile's
    masks[300:320, 500:512] = 3           # cut by the right edge: the right tile sees it whole
    owned, spare = own_masks(masks, GRID, 0)
    assert set(np.unique(owned)) == {0, 1} and set(np.unique(spare)) == {0, 2, 3}
    lone, extra = own_masks(masks, [GRID[0]], 0)          # no neighbour: every cell is its own, cut or not
    assert set(np.unique(lone)) == {0, 1, 2, 3} and not extra.any()


def test_the_overlap_grows_with_the_longest_side_of_the_measured_cells():
    masks = np.zeros((300, 300), np.uint32)
    for k, (y, x) in enumerate([(30, 30), (30, 150), (150, 30), (150, 150)], 1):
        masks[y:y+40, x:x+40] = k
    masks[230:236, 20:220] = 5                               # long and thin: 200 px long, 6 px wide
    masks[0:10, 280:300] = 6                                 # touches the border: not measured
    sizes = whole_cell_sizes(masks)
    assert sorted(sizes.tolist()) == [40., 40., 40., 40., 200.]
    assert cell_diameter(sizes) == pytest.approx(np.percentile(sizes, 99))       # the long cell sets it, not its area
    assert needed_overlap(cell_diameter(sizes), 64, 512) == int(np.ceil(1.25 * cell_diameter(sizes)))
    assert needed_overlap(40, 64, 512) == 64 and needed_overlap(None, 64, 512) == 64       # never narrower
    assert needed_overlap(1000, 64, 512) == 256 and cell_diameter(sizes[:4]) is None


@needs_sopa
@pytest.mark.parametrize("case", ["grid", "big", "disagree", "tissue", "smaller", "long"])
def test_tiles_join_into_every_cell_once_and_whole(tmp_path, case):
    """Through the real run loop with a known answer: no duplicates, no cut cells, nothing lost, live = final."""
    folder = tmp_path / "run"
    folder.mkdir()
    result = subprocess.run([sopa_python(), str(Path(__file__).with_name("stitch_check.py")), str(folder), case],
                            capture_output=True, text=True, timeout=300,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                                 "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                                 "NUMBA_CACHE_DIR": str(tmp_path / "numba"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
                                 "XDG_CACHE_HOME": str(tmp_path / "cache"), "OPENBLAS_NUM_THREADS": "1",
                                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["cells"] == summary["expected"] == summary["matched"]
    # \"smaller\": the small cell shows beside the restored whole one until the final join merges them.
    assert summary["live"] == summary["cells"] + (case == "smaller")
    assert summary["overlaps"] == 0
