"""Saved stain shapes seed proseg by themselves; no nucleus selection or pairing step."""
import json
import os
import subprocess

import numpy as np

from conftest import needs_sopa, sopa_python
from test_zarr import synthetic, sample_path
from test_stain_zarr import helper_save


@needs_sopa
def test_saved_kept_shapes_alone_seed_proseg(sample_path, tmp_path):
    result, saved, _ = helper_save(sample_path, tmp_path, np.array([True, False, True]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert saved["cells"] == 2 and "pairing" not in saved
    output = tmp_path / "seed.parquet"
    args = dict(path=str(sample_path), points_key="transcripts", out=str(output),
                region=dict(kind="box", bbox=[0, 0, 100, 50], box=[0, 0, 100, 50]),
                prior=dict(cells=dict(kind="shapes", key=saved["shapes_key"]), nuclei=None))
    result = subprocess.run([sopa_python(), "-m", "sopa_companion.zarr_bridge.helper", "prepare", json.dumps(args)],
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr
    ready = json.loads(result.stdout.strip().splitlines()[-1])
    assert ready["assigned"] > 0 and ready["seeded_from"] == "cells"
    import pyarrow.parquet as pq
    table = pq.read_table(output)
    assigned = table["cell_id"].to_pylist()
    assert set(assigned) - {"UNASSIGNED"} <= {"cellpose-7", "cellpose-42"}
    assert "cellpose-10" not in assigned
    assert all(flag for cell, flag in zip(assigned, table["overlaps_nucleus"].to_pylist()) if cell != "UNASSIGNED")
