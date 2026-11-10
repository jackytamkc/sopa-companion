"""Opt-in cached-weight smoke; never downloads or modifies the source cache.

SOPA_COMPANION_INSTANSEG_SMOKE=1 collects this test. SOPA_COMPANION_INSTANSEG_MODELS can name
an existing cache; otherwise use the harness engine's instanseg-models directory
(var/instanseg-models in SOPA_COMPANION_HARNESS_DIR, conftest.py).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pytest

from conftest import harness_path, instanseg_python
from test_stain_cellpose import production_idle

MODEL = "fluorescence_nuclei_and_cells"


if os.environ.get("SOPA_COMPANION_INSTANSEG_SMOKE") == "1":
    def test_real_instanseg_cached_weights(tmp_path):
        python = Path(instanseg_python())               # [stain] instanseg_python: skipped without it
        # The cached weights: SOPA_COMPANION_INSTANSEG_MODELS, else the harness engine's (skipped without a harness).
        cache = Path(os.environ.get("SOPA_COMPANION_INSTANSEG_MODELS") or harness_path("var", "instanseg-models"))
        # Its model index, under whichever python its environment has (lib/python3.x/site-packages).
        found = python.parent.parent.glob("lib/python3*/site-packages/instanseg/bioimageio_models/model-index.json")
        index = max(found, default=None)
        if index is None:
            pytest.skip("InstanSeg is not installed in [stain] instanseg_python's environment")
        entry = next(m for m in json.loads(index.read_text()) if m["name"] == MODEL)
        weights = cache / MODEL / entry["version"] / "instanseg.pt"
        if not weights.is_file():
            pytest.skip("InstanSeg model is not cached; this smoke never downloads")
        assert production_idle(), "Production is open: real InstanSeg must wait"
        root = tmp_path / "instanseg-models"
        target = root / MODEL / entry["version"] / "instanseg.pt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(weights, target)
        out = tmp_path / "run"
        (out / "stain").mkdir(parents=True)
        job = dict(out=str(out), settings=dict(method="instanseg", model=MODEL, output="cells",
                                              pixel_size=.5, resolve_cells=False))
        (out / "stain/job.json").write_text(json.dumps(job))
        yy, xx = np.mgrid[:128, :128]
        pixels = np.stack([np.exp(-((xx-64)**2+(yy-64)**2)/100),
                           np.exp(-((xx-64)**2+(yy-64)**2)/400)], axis=-1).astype(np.float32)
        np.save(out / "stain/instanseg-input.npy", pixels, allow_pickle=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            XDG_CACHE_HOME=str(tmp_path / "cache"), MPLCONFIGDIR=str(tmp_path / "mpl"),
            NUMBA_CACHE_DIR=str(tmp_path / "numba"), TORCH_HOME=str(tmp_path / "torch"))
        for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
            env[key] = str(root)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        with (tmp_path / "stdout").open("w") as stdout, (tmp_path / "stderr").open("w") as stderr:
            process = subprocess.Popen([str(python), "-m", "sopa_companion.zarr_bridge.stain_instanseg",
                str(out / "stain/job.json")], env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True)
            try:
                process.stdin.write("tile\n")
                process.stdin.close()
                deadline = time.monotonic()+180
                while process.poll() is None:
                    assert production_idle(), "Production opened during the smoke"
                    assert time.monotonic() < deadline, "InstanSeg exceeded 180 seconds"
                    time.sleep(.2)
                assert process.returncode == 0, (tmp_path / "stderr").read_text()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        events = [json.loads(line) for line in (tmp_path / "stdout").read_text().splitlines()]
        hello = next(e for e in events if e["ev"] == "hello")
        assert hello["cuda_visible_devices"] == "1" and hello["nice"] == 19
        assert not any(e["ev"] == "progress" for e in events), "Cached smoke must never download"
        masks = np.load(out / "stain/instanseg-output.npy", allow_pickle=False)
        assert masks.shape == (128, 128) and masks.dtype == np.uint32
