"""Early-stop regions and model-free joins, using real geometry in the sopa environment."""
import os
from pathlib import Path
import subprocess

import pytest

from conftest import needs_sopa, sopa_python


@needs_sopa
@pytest.mark.parametrize("case", ["first", "row", "staircase", "complete", "empty", "image", "lasso", "split",
                                 "sparse", "tiny", "join", "join-cli"])
def test_stain_covered(tmp_path, case):
    result = subprocess.run([sopa_python(), str(Path(__file__).with_name("covered_check.py")), str(tmp_path), case],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
             "NUMBA_CACHE_DIR": str(tmp_path / "numba"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "XDG_CACHE_HOME": str(tmp_path / "cache"), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
