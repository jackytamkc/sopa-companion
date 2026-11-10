"""Opt-in smoke of the real trainer on GPU 0: a 2-epoch fit in the sopa
environment, started exactly as the engine starts it (CUDA_VISIBLE_DEVICES=0 in PCI bus order), whose deep.npz loads and
passes both self-tests (the trainer's float64 and the worker's float32 tolerance); then a detail job of the same model as
one learned before Go deeper (the parity rule, and the details against the installed calculator).

Never part of the gate: the tests exist only with SOPA_COMPANION_TRAINER_SMOKE=1 (so nothing is ever skipped). They refuse
to start while the app is open (a listener on its default port, 8777) or another trainer runs, wait for GPU 0 to be
quiet for 10 s (under 30 % busy, 6,000 MB free: this smoke's own courtesy on a shared computer; the engine itself waits
only for the free memory) and hold the computer-wide fit lock (engine/qc.py FIT_LOCK, the one every engine of this user
honours) for both fits. About half a minute, most of it starting the trainer twice; GPU 1 is never visible.

    cd <folder holding sopa_companion> && SOPA_COMPANION_TRAINER_SMOKE=1 env -u DISPLAY PYTHONPATH=$PWD \\
        sopa_companion/.venv/bin/python -m pytest -q -p no:cacheprovider sopa_companion/tests/test_qc_trainer_smoke.py
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time

import numpy as np
import pytest

from sopa_companion.analysis import qc_trainer as T

GPU_ENV = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}     # exactly the engine's (_trainer_env)

if os.environ.get("SOPA_COMPANION_TRAINER_SMOKE") == "1":
    from conftest import sopa_python
    from sopa_companion.engine import qc as Q
    from sopa_companion.tests.test_qc_trainer import (MAKER, cpu_job, finish_trainer, installed_copy, snapshot,
                                                   start_trainer)

    def app_open() -> bool:
        listening = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=10).stdout
        return any(line.split()[3].endswith(":8777") for line in listening.splitlines()[1:] if len(line.split()) > 3)

    def trainers_running() -> list[str]:
        rows = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout.splitlines()
        return [row.strip() for row in rows if "sopa_companion.analysis.qc_trainer" in row]

    def wait_for_quiet_gpu0(limit_s: float = 180.0) -> dict:
        """GPU 0 under 30 % busy with >= 6,000 MB free for 10 s in a row: this smoke's own courtesy wait (the engine's
        until 2026-10-21; the engine now waits only for the free memory)."""
        calm, deadline = None, time.monotonic() + limit_s
        while time.monotonic() < deadline:
            probe = Q.gpu_probe(0)
            if probe is None:
                pytest.fail("nvidia-smi could not describe GPU 0")
            if probe["util"] < 30 and probe["free_mb"] >= 6000:
                calm = calm or time.monotonic()
                if time.monotonic() - calm >= 10:
                    return probe
            else:
                calm = None
            time.sleep(1)
        pytest.fail(f"GPU 0 did not become quiet within {limit_s:.0f} s")

    @pytest.fixture(scope="module")
    def gpu_runs(tmp_path_factory):
        if app_open():
            pytest.fail("the app is open (its default port 8777 listens): close it before the GPU smoke")
        if trainers_running():
            pytest.fail(f"another mixing-check trainer is running: {trainers_running()}")
        base = tmp_path_factory.mktemp("gpu-smoke")
        zarr = base / "s.zarr"
        subprocess.run([sopa_python(), "-I", str(MAKER), "--sample", str(zarr)], check=True, capture_output=True,
                       timeout=300, env={**{k: v for k, v in os.environ.items() if k != "DISPLAY"},
                                         "CUDA_VISIBLE_DEVICES": ""})
        Q.FIT_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(Q.FIT_LOCK, "a") as lock:                       # the real computer-wide lock, not the tests' own
            deadline = time.monotonic() + 600
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        pytest.fail("the computer-wide fit lock stayed taken for 10 minutes")
                    time.sleep(5)
            try:
                probe = wait_for_quiet_gpu0()
                job = {k: v for k, v in cpu_job(zarr, base / "fit").items() if k != "cpu_for_tests"}
                job["recipe"] = T.recipe(epochs=2, min_steps=1)
                fit = finish_trainer(start_trainer(job, base / "locks", GPU_ENV))
                assert fit[-1]["ev"] == "done", fit[-1]
                if app_open():
                    pytest.fail("the app opened during the smoke: the detail job was not started")
                older = installed_copy(base / "fit", base / "older")
                before = snapshot(older)
                detail = finish_trainer(start_trainer({**job, "out_dir": str(base / "staging"),
                                                       "detail_for": str(older)}, base / "locks", GPU_ENV))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return {"probe": probe, "fit": (base / "fit", fit), "detail": (older, before, base / "staging", detail)}

    def test_a_gpu_fit_exports_details_that_pass_both_self_tests(gpu_runs):
        out, lines = gpu_runs["fit"]
        hello = lines[0]
        assert hello["ev"] == "hello" and hello["cuda_visible_devices"] == "0" and hello["nice"] == 19
        device = next(m["device"] for m in lines if m.get("device"))
        assert device["pci_bus_id"] == Q.bus_number(gpu_runs["probe"]["bus"]) and device["name"] != "cpu"   # GPU 0
        enc = T.load_encoder(out / "encoder.npz")
        deep = T.load_deep(out / "deep.npz", enc["genes"])
        assert T.deep_self_test(deep) <= T.DEEP_SELFTEST_TOLERANCE                       # the trainer's
        assert T.deep_self_test(deep, dtype=np.float32) <= T.DEEP_WORKER_TOLERANCE       # the worker's
        model = json.loads((out / "model.json").read_text())
        assert model["epochs"] == 2 and model["device"]["pci_bus_id"] == device["pci_bus_id"]
        assert model["deep"]["from"] == "fit" and model["deep"]["bytes"] == (out / "deep.npz").stat().st_size
        assert model["provenance"]["epochs_run"] == 2 and "other_epochs" in model["provenance"]["differences"]

    def new_id(staging):
        return json.loads((staging / "model.json").read_text())["id"]

    def test_a_gpu_detail_job_adds_details_the_installed_model_accepts(gpu_runs):
        installed, before, staging, lines = gpu_runs["detail"]
        assert lines[-1]["ev"] == "done", lines[-1]
        # The new calculator stays beside the details in staging: the engine compares it with the installed one once more
        # before it installs anything, and removes it with staging.
        assert sorted(p.name for p in staging.iterdir()) == [T.DETAIL_ENCODER, "deep.npz", "model.json"]
        assert Q.calculators_agree(T.load_encoder(installed / "encoder.npz"), T.load_encoder(staging / T.DETAIL_ENCODER))
        Q.QcModels._check_detail(staging, new_id(staging), installed)            # the engine's own check passes
        assert snapshot(installed) == before
        new = json.loads((staging / "model.json").read_text())
        assert new["deep"]["from"] == "detail" and new["detail"]["accepted"]
        assert new["detail"]["identical"] or new["detail"]["reference_outputs_max_abs_diff"] <= T.DETAIL_TOLERANCE
        deep = T.load_deep(staging / "deep.npz", T.load_encoder(installed / "encoder.npz"))
        assert T.deep_self_test(deep) <= T.DEEP_SELFTEST_TOLERANCE
        assert T.deep_self_test(deep, dtype=np.float32) <= T.DEEP_WORKER_TOLERANCE
