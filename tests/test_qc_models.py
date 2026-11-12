"""The mixing check without a GPU: [qc] settings, table fingerprints and model keys, learning a sample with a fake
trainer (queue, the wait for room on GPU 0, progress, failures, the out-of-memory retry after its pause, time limit,
kill on close, the export check, GPU 0 only), the trainer's numpy calculator and recipe maths, the shared zarr lock,
and QcManager's headline over worker results (smoothing, rounding hysteresis, state.json, series, start line, picking
up the worker's files)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import types

import numpy as np
import pytest

from conftest import sopa_python, wait_for
from sopa_companion.analysis import qc_trainer as T
from sopa_companion.engine import qc as Q
from sopa_companion.engine.config import ROOT
from sopa_companion.zarr_bridge import helper

FAKE = "sopa_companion.tests.fake_qc_trainer"
FAKE_PROBE = "sopa_companion.tests.fake_qc_probe"        # the probe's stand-in: no torch, no NVIDIA driver library
IDLE = {"bus": "00000000:11:00.0", "total_mb": 97887.0, "used_mb": 800.0, "free_mb": 97087.0, "util": 0.0}


# ---- fixtures ----------------------------------------------------------------------------------------------------
def write_table(root: Path, key: str, n_obs=4000, n_vars=7, nnz=9000, dense=False, spatial=True, v2=False,
                proseg=False):
    """The zarr metadata of tables/<key> (what QcModels reads; no data: the fake trainer does not read it). `proseg`:
    a table of proseg's own count, as a save of a proseg run leaves it (uns/proseg_run, and the app's own record)."""
    folder = root / "tables" / key
    (folder / "X").mkdir(parents=True, exist_ok=True)
    (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    if proseg:
        for name in ("proseg_run", "sopa_companion"):
            (folder / "uns" / name).mkdir(parents=True, exist_ok=True)
            (folder / "uns" / name / "zarr.json").write_text(json.dumps({"node_type": "group", "attributes": {}}))
    if v2:
        (folder / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (folder / ".zattrs").write_text(json.dumps({"encoding-type": "anndata"}))
        (folder / "X" / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (folder / "X" / ".zattrs").write_text(json.dumps({"encoding-type": "csr_matrix", "shape": [n_obs, n_vars]}))
        (folder / "X" / "data").mkdir(exist_ok=True)
        (folder / "X" / "data" / ".zarray").write_text(json.dumps({"shape": [nnz], "dtype": "<f4"}))
        if spatial:
            (folder / "obsm" / "spatial").mkdir(parents=True, exist_ok=True)
            (folder / "obsm" / "spatial" / ".zarray").write_text(json.dumps({"shape": [n_obs, 2], "dtype": "<f8"}))
        return folder
    (folder / "zarr.json").write_text(json.dumps({"node_type": "group", "attributes": {"encoding-type": "anndata"}}))
    if dense:
        (folder / "X" / "zarr.json").write_text(json.dumps({"node_type": "array", "shape": [n_obs, n_vars],
                                                            "data_type": "float32", "attributes": {}}))
    else:
        (folder / "X" / "zarr.json").write_text(json.dumps({"node_type": "group", "attributes": {
            "encoding-type": "csr_matrix", "shape": [n_obs, n_vars]}}))
        (folder / "X" / "data").mkdir(exist_ok=True)
        (folder / "X" / "data" / "zarr.json").write_text(json.dumps({"node_type": "array", "shape": [nnz],
                                                                     "data_type": "float32", "attributes": {}}))
    if spatial:
        (folder / "obsm" / "spatial").mkdir(parents=True, exist_ok=True)
        (folder / "obsm" / "spatial" / "zarr.json").write_text(json.dumps({"node_type": "array", "shape": [n_obs, 2],
                                                                           "data_type": "float64", "attributes": {}}))
    return folder


class Registry:
    """ZarrSamples' registry as QcModels uses it (entry: a known sample's {id, name, path}, listed or not)."""

    def __init__(self, *entries):
        self.entries = list(entries)

    def list(self):
        return [dict(e) for e in self.entries]

    def entry(self, sid):
        return next((dict(e) for e in self.entries if e.get("id") == sid), None)


def sample(tmp_path, name="sample-a-crop", tables=("xenium_table", "table"), proseg=("table",)):
    """A sample's tables; those in `proseg` hold proseg's own count (the earlier proseg cells)."""
    root = tmp_path / "samples" / f"{name}.zarr"
    for key in tables:
        write_table(root, key, proseg=key in proseg)
    from sopa_companion.engine.zarr_samples import sample_id
    return {"id": sample_id(str(root.resolve())), "name": name, "path": str(root.resolve())}


def qc_config(tmp_path, **qc):
    return {"paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state")},
            "qc": {"python": sys.executable, "trainer_module": FAKE, "probe_module": FAKE_PROBE, "gpu_quiet_s": 0,
                   **qc}}


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """The fake trainer imports sopa_companion from the app tree (the engine puts it on PYTHONPATH) and logs launches."""
    log = tmp_path / "launches.jsonl"
    monkeypatch.setenv("FAKE_QC_LOG", str(log))
    monkeypatch.delenv("FAKE_QC_MODE", raising=False)

    def launches():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return launches


def models(tmp_path, *samples, probe=lambda: dict(IDLE), **qc):
    m = Q.QcModels(qc_config(tmp_path, **qc), Registry(*samples), probe=probe)
    return m


def settle(m, sid, timeout=30):
    wait_for(lambda: m.describe(sid)["state"] not in ("waiting", "learning"), timeout=timeout)
    return m.describe(sid)


# ---- settings, fingerprints, keys --------------------------------------------------------------------------------
def test_settings_are_checked(tmp_path):
    s = Q.QcSettings({"qc": {"python": "/x/python"}})
    assert (s.gpu, s.mixed_at, s.judge_min, s.fit_min, s.learn, s.tables) == (0, 0.3, 100, 20, "start", ["xenium_table", "table"])
    # [qc] gpu names the GPU the check learns on (0 here), any GPU's number; models_dir is a
    # full folder path (API mode's model_cache) or empty.
    assert Q.QcSettings({"qc": {"gpu": 1}}).gpu == 1 and s.models_dir == ""
    assert Q.QcSettings({"qc": {"models_dir": str(tmp_path / "m")}}).models_dir == str(tmp_path / "m")
    assert s.recipe == T.recipe(50, 3000, 20) and s.time_limit_s == 1800 and s.gpu_min_free_mb is None   # auto
    assert Q.QcSettings({"qc": {"gpu_min_free_mb": 3000}}).gpu_min_free_mb == 3000
    # The busy wait's settings and the notebook folder are retired: a config that still has them reads as before, and
    # they change nothing (the notebook's number was never shown).
    old = Q.QcSettings({"qc": {"gpu_busy_util": 30, "gpu_quiet_s": 10, "notebook_dir": "/somewhere/else"}})
    assert not hasattr(old, "gpu_busy_util") and not hasattr(old, "gpu_quiet_s") and not hasattr(old, "notebook_dir")
    assert set(Q.RETIRED) == {"gpu_busy_util", "gpu_quiet_s", "notebook_dir"} and "notebook_dir" not in Q.DEFAULTS
    assert s.priors == "pooled" and s.recipe["priors"] == "pooled" and s.recipe["version"] == 2
    assert Q.QcSettings({"qc": {"priors": "table"}}).recipe == T.recipe(priors="table")
    for bad in ({"gpu": -1}, {"gpu": "0"}, {"gpu": True}, {"models_dir": "relative/models"}, {"models_dir": 3},
                {"learn": "always"}, {"tables": []}, {"tables": ["../x"]}, {"mixed_at": 0},
                {"mixed_at": 1.5}, {"judge_min_counts": 10, "fit_min_counts": 20}, {"epochs": 0}, {"epochs": 2.5},
                {"gpu_min_free_mb": -1}, {"bin_um_tiled": -1}, {"time_limit_s": True}, {"priors": "both"}):
        with pytest.raises(ValueError, match=r"\[qc\]"):
            Q.QcSettings({"qc": bad})
    with pytest.raises(ValueError):
        T.recipe(priors="notebook")
    # The real config.toml: GPU 0, the decisions' thresholds (as calibrated).
    from sopa_companion.engine.config import load_config
    real_config = load_config()
    real = Q.QcSettings(real_config)
    assert real.gpu == 0 and real.mixed_at == 0.3 and real.judge_min == 100
    assert real.fit_min == 20 and real.bin_um_tiled == 100 and real.bin_um_detail == 50 and real.priors == "pooled"
    assert not real.enabled_by_default and real.learn == "start"     # opt-in: nothing is learned until switched on
    # The trainer runs in the sopa environment (the one conftest finds), where this computer's settings name one.
    if real.python and (sopa := sopa_python(skip=False)):
        assert real.python == sopa
    # The room a fit waits for: "auto" (the shipped default: from the panel) unless this computer's settings set it.
    configured = (real_config.get("qc") or {}).get("gpu_min_free_mb", "auto")
    assert real.gpu_min_free_mb == (None if configured in ("auto", "", None) else float(configured))
    assert real.module == "sopa_companion.analysis.qc_trainer"


def test_table_fingerprints_from_zarr_metadata(tmp_path):
    root = tmp_path / "s.zarr"
    write_table(root, "xenium_table", n_obs=41190, n_vars=5101, nnz=123)
    write_table(root, "dense", n_obs=10, n_vars=3, dense=True)
    write_table(root, "flat", spatial=False)
    write_table(root, "old", n_obs=5, n_vars=4, nnz=9, v2=True)
    info = Q.table_info(root, "xenium_table")
    assert {k: info[k] for k in ("key", "n_obs", "n_vars", "nnz", "spatial")} == \
        {"key": "xenium_table", "n_obs": 41190, "n_vars": 5101, "nnz": 123, "spatial": True}
    assert Q.table_info(root, "dense")["nnz"] is None and Q.table_info(root, "dense")["n_obs"] == 10
    assert Q.table_info(root, "flat")["spatial"] is False
    assert {k: Q.table_info(root, "old")[k] for k in ("n_obs", "n_vars", "nnz", "spatial")} == \
        {"n_obs": 5, "n_vars": 4, "nnz": 9, "spatial": True}
    assert Q.table_info(root, "missing") is None


def test_the_model_key_follows_table_recipe_and_software(tmp_path):
    root = tmp_path / "s.zarr"
    write_table(root, "xenium_table")
    info = Q.table_info(root, "xenium_table")
    versions = {"scviva-tools": "0.1.7", "scvi-tools": "1.5.1", "torch": "2.13.0"}
    key = Q.model_key(root, info, T.recipe(), versions)
    assert len(key) == 32 and key == Q.model_key(root, dict(info), T.recipe(), dict(versions))
    assert key != Q.model_key(root, info, T.recipe(), {**versions, "torch": "2.14.0"})       # software changed
    assert key != Q.model_key(root, info, T.recipe(epochs=60), versions)                       # recipe changed
    assert key != Q.model_key(root, {**info, "nnz": info["nnz"] + 1}, T.recipe(), versions)  # table changed
    assert key != Q.model_key(tmp_path / "other.zarr", info, T.recipe(), versions)             # another sample
    assert key != Q.model_key(root, info, T.recipe(priors="table"), versions)                  # priors from one table
    # The other table (pooled priors, the earlier proseg cells) is part of the key too.
    other = {"n_obs": 900, "n_vars": 7, "nnz": 50, "mtime": 12.5}
    pooled = Q.model_key(root, info, T.recipe(), versions, {"table": other})
    assert pooled != key and pooled == Q.model_key(root, info, T.recipe(), versions, {"table": dict(other, key="table")})
    assert pooled != Q.model_key(root, info, T.recipe(), versions, {"table": {**other, "mtime": 13.0}})
    assert key == Q.model_key(root, info, T.recipe(), versions, {})
    # Versions come from the environment's dist-info folders: no interpreter is started.
    site = tmp_path / "env" / "lib" / "python3.12" / "site-packages"
    for dist in ("scviva_tools-0.1.7", "scvi_tools-1.5.1", "torch-2.13.0", "pyro_ppl-1.9.2", "numpy-2.4.6", "other-1.0"):
        (site / f"{dist}.dist-info").mkdir(parents=True)
    (tmp_path / "env" / "bin").mkdir()
    (tmp_path / "env" / "bin" / "python").write_text("")
    found = Q.env_versions(str(tmp_path / "env" / "bin" / "python"))
    assert found == {"scviva-tools": "0.1.7", "scvi-tools": "1.5.1", "torch": "2.13.0", "pyro-ppl": "1.9.2",
                     "lightning": None, "anndata": None, "numpy": "2.4.6"}
    assert Q.estimate_seconds(41190, T.recipe()) == pytest.approx(146, abs=2)       # a whole sample: ~2 min
    assert Q.bus_number("00000000:11:00.0") == 17 and Q.bus_number("garbage") is None


def test_a_model_learned_on_apples_gpu_has_its_own_key_room_and_time(tmp_path):
    """On Apple's GPU (device "mps") a model's key holds the device (as good as an NVIDIA card's, not the same cells'
    numbers: never taken for each other), while a CUDA model's key is what it always was; learning there takes
    MPS_STEP_S a step and the room of the dense counts too (MPS holds no sparse tensor), in the computer's own memory."""
    root = tmp_path / "s.zarr"
    write_table(root, "xenium_table")
    info = Q.table_info(root, "xenium_table")
    versions = {"scviva-tools": "0.1.8", "scvi-tools": "1.5.2", "torch": "2.14.1"}
    key = Q.model_key(root, info, T.recipe(), versions)
    assert key == Q.model_key(root, info, T.recipe(), versions, None, "cuda")
    apple = Q.model_key(root, info, T.recipe(), versions, None, "mps")
    assert len(apple) == 32 and apple != key and apple == Q.model_key(root, dict(info), T.recipe(), versions, {}, "mps")
    assert (Q.step_seconds("cuda"), Q.step_seconds("mps")) == (Q.STEP_S, Q.MPS_STEP_S) and Q.MPS_STEP_S > Q.STEP_S
    cuda_s, mps_s = Q.estimate_seconds(41190, T.recipe()), Q.estimate_seconds(41190, T.recipe(), "mps")
    assert (mps_s - Q.FIXED_S) == pytest.approx((cuda_s - Q.FIXED_S) * Q.MPS_STEP_S / Q.STEP_S, abs=2)
    # The dense counts: cells x genes float32 more, with the same margin; nothing more for a CUDA fit.
    assert Q.room_needed_mb(5101, 41190, "cuda") == Q.room_needed_mb(5101, 41190)
    dense_mb = 41190 * 5101 * 4 / 2 ** 20
    extra = Q.room_needed_mb(5101, 41190, "mps") - Q.room_needed_mb(5101, 41190)
    assert dense_mb * Q.ROOM_MARGIN - 100 <= extra <= dense_mb * Q.ROOM_MARGIN + 100


def test_apples_gpu_room_is_the_computers_own_memory(monkeypatch):
    """apple_memory: the gpu_probe shape from the computer's memory, of which Metal lets one process use about
    APPLE_GPU_SHARE; what is free is what the system calls available (never more than that share); no bus. None when it
    cannot be read."""
    gib = 2 ** 30
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=128 * gib, available=40 * gib))
    room = Q.apple_memory()
    assert room == {"bus": None, "total_mb": 96 * 1024.0, "used_mb": 56 * 1024.0, "free_mb": 40 * 1024.0, "util": None}
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=16 * gib, available=15 * gib))
    assert Q.apple_memory()["free_mb"] == Q.apple_memory()["total_mb"] == 12 * 1024.0
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: (_ for _ in ()).throw(OSError("no")))
    assert Q.apple_memory() is None and Q.bus_number(None) is None


def mac_sample(tmp_path, name="sample-a-crop"):
    """A sample whose learned table is big enough for the dense counts to weigh in the room (40,000 cells x 500 genes:
    76 MB more on Apple's GPU)."""
    s = sample(tmp_path, name)
    write_table(Path(s["path"]), "xenium_table", n_obs=40000, n_vars=500, nnz=900000)
    return s


def test_on_a_mac_the_check_learns_on_apples_gpu(tmp_path, fake_env, monkeypatch):
    """A Mac whose Apple GPU passed the probe's test: the check is offered and learns there at once (the Mac's memory
    is never waited for: what macOS calls available stays low beside other programs), its fit told the device ("mps")
    its key was made for, MPS_TIME_FACTOR times time_limit_s, PyTorch moving an operation MPS lacks to the CPU; its model
    has a key of its own, and the export names the device."""
    gib = 2 ** 30
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=64 * gib, available=2 ** 20))
    monkeypatch.setenv("FAKE_QC_PROBE", "mac_mps")
    s = mac_sample(tmp_path)
    m = models(tmp_path, s, probe=lambda: pytest.fail("nvidia-smi is never asked on a Mac"))
    try:
        cap = answered(m)
        assert (cap["state"], cap["reason"]) == ("ready", None) and m.available and m.device() == "mps"
        assert cap["why"] == "torch 2.14.1 learns on Apple's GPU (MPS): it passed the probe's test"
        assert m._time_limit("mps") == m.s.time_limit_s * Q.MPS_TIME_FACTOR and m._time_limit("cuda") == m.s.time_limit_s
        m.ensure(s["id"])
        view = settle(m, s["id"])
        assert view["state"] == "ready" and view["room"] is None
        [launch] = fake_env()
        assert launch["job"]["device"] == "mps" and launch["job"]["deadline_s"] == m.s.time_limit_s * Q.MPS_TIME_FACTOR
        assert launch["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        record = m._record(s["id"])
        info = Q.table_info(record["path"], record["table"]["key"])
        others = {k: m._fingerprint(Q.table_info(record["path"], k)) for k in m.s.tables if k != info["key"]}
        assert record["device"] == "mps"
        assert record["key"] == Q.model_key(record["path"], info, m.s.recipe, m._versions(), others, "mps")
        assert record["key"] != Q.model_key(record["path"], info, m.s.recipe, m._versions(), others)
        stored = json.loads((m._folder(record) / "model.json").read_text())
        assert stored["device"]["kind"] == "mps"
        assert Q.room_needed_mb(500, 40000, "mps") - Q.room_needed_mb(500, 40000) >= 100   # the dense counts
    finally:
        m.close()


def test_a_fit_on_apples_gpu_never_says_its_gpu_is_shared(tmp_path):
    """The "shared" flag (steps slower than twice STEP_S) is calibrated on an NVIDIA card: on Apple's GPU, whose chips
    differ too much to tell, it is never raised; its time left takes MPS_STEP_S."""
    m = Q.QcModels(qc_config(tmp_path), Registry(), start=False, probe=lambda: dict(IDLE))
    try:
        for device, shared in (("cuda", True), ("mps", False)):
            record = m._blank("zsample")
            record.update(device=device, learning={})
            m._progress(record, {"stage": "preparing", "epochs": 10, "cells": 5000, "steps": 100})
            assert record["eta_s"] == int(Q.FIXED_S * 0.6 + Q.step_seconds(device) * 100)
            m._progress(record, {"stage": "learning", "step": 2, "of": 10, "seconds": 20.0})     # 1 s a step: slow
            assert record["shared"] is shared, device
    finally:
        m.close()


def test_a_mac_too_small_for_the_table_is_said_at_once(tmp_path, fake_env, monkeypatch):
    """A Mac whose GPU may use less than the fit needs in all (Metal's share of its memory) can never learn the table:
    failed, gpu_small, at once and with nothing read, its room saying it is a Mac's (the page's own words)."""
    gib = 2 ** 30
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=1 * gib, available=gib // 2))
    monkeypatch.setenv("FAKE_QC_PROBE", "mac_mps")
    s = mac_sample(tmp_path)
    m = models(tmp_path, s, probe=None)
    try:
        answered(m)
        m.ensure(s["id"])
        view = settle(m, s["id"])
        need = Q.room_needed_mb(500, 40000, "mps")
        assert (view["state"], view["reason"]) == ("failed", "gpu_small") and not fake_env()
        assert view["room"] == {"need_mb": need, "free_mb": 500, "total_mb": 768, "genes": 500, "mac": True}
        assert view["message"] == (f"learning it needs about {need} MB on Apple's GPU (500 genes, its counts included) "
                                   "and this Mac lets it use 768 MB in all")
    finally:
        m.close()


def test_the_key_and_the_job_name_the_same_device_whenever_the_probe_answers(tmp_path, fake_env, monkeypatch):
    """The device is kept with the key it was made for: a probe that answers "Apple's GPU" between a look and the fit
    (here at the moment the first look made its key) never leaves a CUDA key on a fit told "mps" (the look after the
    probe's answer makes both again), and an export naming another device than the key's is refused."""
    gib = 2 ** 30
    monkeypatch.setattr(Q.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=64 * gib, available=32 * gib))
    s = mac_sample(tmp_path)
    m = Q.QcModels(qc_config(tmp_path), Registry(s), start=False, probe=lambda: dict(IDLE),
                   prober=lambda: {"ev": "probe", "missing": [], "torch": "2.14.1", "cuda_build": None, "devices": 0,
                                   "nvidia": None, "driver_cuda": None, "platform": "darwin", "machine": "arm64",
                                   "mps": True, "mps_problem": None, "error": None})
    flipped = []
    relearn = m._relearn

    def flip_once(*args):
        if not flipped:                           # the probe answers right after the first look made its key
            flipped.append(True)
            m._probe()
        return relearn(*args)
    monkeypatch.setattr(m, "_relearn", flip_once)
    fit = m._fit

    def answer_moves_before_the_job(record_, info_, key_):
        # The probe's answer changes after the last look and before the job is written (no look in between): the job
        # must still name the device its key was made for.
        m.capability_state = {**m.capability_state, "probe": {"devices": 1, "mps": False}}
        assert m.device() == "cuda"
        return fit(record_, info_, key_)
    monkeypatch.setattr(m, "_fit", answer_moves_before_the_job)
    try:
        record = m._record(s["id"])
        m._refresh(record, force=True)
        # the answer came while the look made its key: a newer look made both again, for Apple's GPU
        assert flipped and record["device"] == "mps"
        m._learn(record)
        [launch] = fake_env()
        assert launch["job"]["device"] == record["device"] == "mps" and record["state"] == "ready"
        info = Q.table_info(record["path"], record["table"]["key"])
        others = {k: m._fingerprint(Q.table_info(record["path"], k)) for k in m.s.tables if k != info["key"]}
        assert record["key"] == launch["job"]["model_id"] == Q.model_key(record["path"], info, m.s.recipe,
                                                                         m._versions(), others, "mps")
    finally:
        m.close()
    # An export naming the other device: refused, nothing installed.
    monkeypatch.setenv("FAKE_QC_MODE", "wrongdevice")
    other = mac_sample(tmp_path, "sample-b-crop")
    monkeypatch.setenv("FAKE_QC_PROBE", "mac_mps")
    m = models(tmp_path, other, probe=None)
    try:
        answered(m)
        m.ensure(other["id"])
        view = settle(m, other["id"])
        assert (view["state"], view["reason"]) == ("failed", "unexpected") and view["model"] is None
        assert "learned on cuda, and its key is for mps" in view["message"]
    finally:
        m.close()


def test_a_venvs_versions_are_its_own_packages(tmp_path):
    """A venv's bin/python is a symlink to its base interpreter, whose folder holds none of the venv's packages: the
    versions are read in the venv itself (its pyvenv.cfg says it is one), so upgrading them gives its models a new key.
    Before, every one read as None there, and a model learned with older packages was used on. A venv that shares the
    base's packages takes those it lacks from there; a plain link to another environment's interpreter (no pyvenv.cfg)
    still leads to that environment."""
    def environment(root, *dists):
        site = root / "lib" / "python3.12" / "site-packages"
        for dist in dists:
            (site / f"{dist}.dist-info").mkdir(parents=True)
        (root / "bin").mkdir(parents=True)
        return root / "bin"
    base = environment(tmp_path / "base", "numpy-1.0.0", "anndata-0.12.0") / "python3.12"
    base.write_text("")
    venv = environment(tmp_path / "venv", "torch-2.13.0", "scvi_tools-1.5.1", "numpy-2.4.6")
    (venv.parent / "pyvenv.cfg").write_text(f"home = {base.parent}\ninclude-system-site-packages = false\n")
    (venv / "python").symlink_to(base)
    found = Q.env_versions(str(venv / "python"))
    assert (found["torch"], found["scvi-tools"], found["numpy"], found["anndata"]) == ("2.13.0", "1.5.1", "2.4.6", None)
    (venv.parent / "pyvenv.cfg").write_text(f"home = {base.parent}\ninclude-system-site-packages = true\n")
    found = Q.env_versions(str(venv / "python"))
    assert (found["torch"], found["numpy"], found["anndata"]) == ("2.13.0", "2.4.6", "0.12.0")    # its own first
    conda = environment(tmp_path / "conda", "torch-2.12.0") / "python"
    conda.write_text("")
    linked = tmp_path / "links" / "python"
    linked.parent.mkdir()
    linked.symlink_to(conda)
    assert Q.env_versions(str(linked))["torch"] == "2.12.0"


def test_the_trainer_sees_only_the_qc_gpu_counted_in_the_engines_gpus(tmp_path, monkeypatch):
    """The trainer's environment: [qc] gpu is an index within the engine's own CUDA_VISIBLE_DEVICES
    (engine/children.py): unset, GPU 0 itself; an engine shown only GPU 1 (a notebook's choice) gives it that one; an
    engine shown none gives it none. Whether it may learn there: the next test."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("DISPLAY", ":77")
    m = models(tmp_path)
    env = m._trainer_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "4"
    assert "DISPLAY" not in env and str(ROOT.parent) in env["PYTHONPATH"].split(os.pathsep)
    for visible, expected in (("1", "1"), ("GPU-5d2c,GPU-77aa", "GPU-5d2c"), ("", "")):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        assert m._trainer_env()["CUDA_VISIBLE_DEVICES"] == expected
    m.close()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    second = models(tmp_path, gpu=1)
    assert second._trainer_env()["CUDA_VISIBLE_DEVICES"] == "3"
    second.close()


def test_a_qc_gpu_the_trainer_cannot_learn_on_fails_before_anything_is_read(tmp_path, fake_env, monkeypatch):
    """The trainer learns with the CUDA_VISIBLE_DEVICES "0" it is given only (analysis/qc_trainer.py run, which refuses
    any other after reading the tables): a [qc] gpu that maps to another GPU (a kernel that pinned GPU 1, [qc] gpu 1),
    or to none (an engine that sees no GPU), fails at once with reason "gpu": no trainer started, nothing read, and
    Try again says the same. A detail job too, while the sample's model stays ready. The GPU it may learn on is asked
    of nvidia-smi and given to the trainer, its job naming it."""
    probed = []
    monkeypatch.setattr(Q, "gpu_probe", lambda index: probed.append(index) or dict(IDLE))     # what nvidia-smi says
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    s = sample(tmp_path)
    for visible, gpu in (("1", 0), ("", 0), (None, 1), ("2,3", 1), ("0", 1)):
        if visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        m = Q.QcModels(qc_config(tmp_path, gpu=gpu), Registry(s))                 # the engine's own probe
        m.ensure(s["id"])
        view = settle(m, s["id"])
        assert (view["state"], view["reason"]) == ("failed", "gpu"), (visible, gpu, view)
        m.ensure(s["id"], retry=True)
        assert settle(m, s["id"])["reason"] == "gpu" and fake_env() == [] and probed == []
        m.close()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    m = Q.QcModels(qc_config(tmp_path), Registry(s))
    m.ensure(s["id"])
    assert settle(m, s["id"])["state"] == "ready" and set(probed) == {"0"}
    (launch,) = fake_env()
    assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "0" and launch["job"]["gpu"] == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")                             # the same engine, shown no GPU now
    assert m.ensure(s["id"], detail=True)["detail"]["state"] in ("queued", "failed")
    wait_for(lambda: m.describe(s["id"])["detail"]["state"] == "failed", timeout=10)
    view = m.describe(s["id"])
    assert view["state"] == "ready" and view["detail"]["reason"] == "gpu" and len(fake_env()) == 1
    m.close()


# ---- learning with the fake trainer --------------------------------------------------------------------------------
def test_a_table_that_is_not_prosegs_own_count_is_never_scored(tmp_path, fake_env, monkeypatch):
    """A sample whose other table is no proseg save (a Xenium sample's "table": Xenium Ranger's outline re-count): the
    model learns as before (the priors pooled with it) but has no batch.parquet and no "batch" in model.json, as the
    trainer does it."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    s = sample(tmp_path, proseg=())
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    folder = Path(m.model(s["id"])["dir"])
    assert view["state"] == "ready" and not (folder / "batch.parquet").exists()
    assert json.loads((folder / "model.json").read_text())["batch"] is None
    assert fake_env()[0]["job"]["other_tables"] == ["table"]
    m.close()


def test_learning_a_sample_installs_a_checked_model_once(tmp_path, fake_env, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)          # GPU 0 itself (the fake trainer uses none)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    assert m.describe(s["id"])["state"] == "off" and m.describe(s["id"])["estimate_s"] > 0
    view = m.ensure(s["id"])
    assert view["state"] in ("waiting", "learning")
    view = settle(m, s["id"])
    assert view["state"] == "ready" and view["model"]["cells"] == 50 and view["model"]["table"] == "xenium_table"
    model = m.model(s["id"])
    folder = Path(model["dir"])
    assert folder.parent == tmp_path / "var" / "qc-models" / s["id"] and folder.name == view["model"]["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["batch.parquet", "cells.parquet", "deep.npz", "encoder.npz",
                                                        "model.json", "used"]
    assert not list(folder.parent.glob(".*.part-*"))                     # the staging folder was renamed into place
    T.load_encoder(folder / "encoder.npz")
    assert view["model"]["deep"] is True and view["detail"]["state"] == "ready"     # Go deeper's details came with it
    assert model["deep"] == str(folder / "deep.npz") and view["model"]["provenance"]["table"] == "xenium_table"
    (launch,) = fake_env()
    assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "0" and launch["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert launch["env"]["OMP_NUM_THREADS"] == "4" and launch["nice"] == 19 and launch["env"]["DISPLAY"] is None
    job = launch["job"]
    assert job["table"] == "xenium_table" and job["other_tables"] == ["table"] and job["model_id"] == view["model"]["id"]
    assert job["recipe"] == T.recipe() and job["deadline_s"] == 1800 and job["zarr"] == s["path"]
    assert job["mixed_at"] == 0.3 and job["judge_min"] == 100
    assert m.ensure(s["id"])["state"] == "ready" and len(fake_env()) == 1          # cached: never learned twice
    m.close()
    again = models(tmp_path, s)                                                    # a new engine finds it on disk
    assert again.describe(s["id"])["state"] == "ready" and again.model(s["id"])["id"] == view["model"]["id"]
    again.close()


def test_one_fit_at_a_time_in_order(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_EPOCH_S", "0.15")
    a, b = sample(tmp_path, "A"), sample(tmp_path, "B")
    m = models(tmp_path, a, b)
    m.ensure(a["id"])
    m.ensure(b["id"])
    assert settle(m, a["id"])["state"] == "ready" and settle(m, b["id"])["state"] == "ready"
    first, second = fake_env()
    assert first["job"]["sample"]["id"] == a["id"] and second["job"]["sample"]["id"] == b["id"]
    assert second["start"] >= first["end"]                                       # never two trainers at once
    m.close()


def test_engines_with_their_own_data_dirs_still_fit_one_at_a_time(tmp_path, fake_env, monkeypatch):
    """The computer-wide fit lock (two fits on GPU 0 at once each ran 2.2x slower): an engine with another
    data dir, e.g. a harness engine next to the app, waits while another engine fits ("lock"), then fits."""
    monkeypatch.setenv("FAKE_QC_EPOCH_S", "0.4")
    monkeypatch.setenv("FAKE_QC_EPOCHS", "4")
    a, b = sample(tmp_path, "A"), sample(tmp_path, "B")
    one = Q.QcModels(qc_config(tmp_path / "one"), Registry(a), probe=lambda: dict(IDLE))
    two = Q.QcModels(qc_config(tmp_path / "two"), Registry(b), probe=lambda: dict(IDLE))
    one.ensure(a["id"])
    wait_for(lambda: one.describe(a["id"])["state"] == "learning", timeout=10)
    two.ensure(b["id"])
    wait_for(lambda: two.describe(b["id"])["reason"] == "lock", timeout=10)
    assert two.describe(b["id"])["state"] == "waiting"
    assert settle(one, a["id"])["state"] == "ready" and settle(two, b["id"])["state"] == "ready"
    first, second = sorted(fake_env(), key=lambda launch: launch["start"])
    assert first["job"]["sample"]["id"] == a["id"] and second["start"] >= first["end"]
    one.close()
    two.close()


def test_the_fit_locks(tmp_path, monkeypatch):
    # Without their threads: a subprocess started meanwhile (the probe's) holds a copy of a lock's file until it runs
    # its program, so a lock released at that moment would still read as taken.
    m = Q.QcModels(qc_config(tmp_path), Registry(), start=False, probe=lambda: dict(IDLE))
    held = m._fit_lock()
    assert len(held) == 2 and [Path(held[1].name)] == Q.fit_lock_paths()           # the tests' own file alone
    other = Q.QcModels(qc_config(tmp_path / "other"), Registry(), start=False, probe=lambda: dict(IDLE))
    assert other._fit_lock() is None                                              # the computer's lock is taken
    for handle in held:
        handle.close()
    again = other._fit_lock()
    assert again is not None and len(again) == 2
    for handle in again:
        handle.close()
    # A computer-wide lock that cannot be created: said once, and the data dir's lock still keeps its engines apart.
    (tmp_path / "blocked").write_text("")
    monkeypatch.setenv("SOPA_COMPANION_QC_FIT_LOCK", str(tmp_path / "blocked" / "qc-fit.lock"))
    only = m._fit_lock()
    assert len(only) == 1 and m.lock_warned and Path(only[0].name) == m.root / ".fit.lock"
    assert m._fit_lock() is None                                                  # the data dir's lock is taken
    only[0].close()
    assert "cannot be used" in (tmp_path / "state" / "logs" / "qc-trainer.log").read_text()
    monkeypatch.delenv("SOPA_COMPANION_QC_FIT_LOCK")                     # the user's: the previous version's, then this one's
    assert Q.fit_lock_paths() == [Path.home() / ".cache" / "proseg_live" / "qc-fit.lock",
                                  Path.home() / ".cache" / "sopa-companion" / "qc-fit.lock"]
    m.close()
    other.close()


def test_a_changed_other_table_means_learning_again(tmp_path, fake_env, monkeypatch):
    """The priors are pooled with the sample's other table (and its cells are scored once): a new save of that table
    is a new model, never the old one."""
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    first = settle(m, s["id"])["model"]["id"]
    write_table(Path(s["path"]), "table", nnz=9001)                               # the earlier proseg cells, saved again
    monkeypatch.setattr(Q, "KEY_TTL_S", 0.0)
    view = m.describe(s["id"])
    assert view["state"] == "off" and view["model"] is None and view["relearn"]
    m.ensure(s["id"])
    assert settle(m, s["id"])["model"]["id"] != first and len(fake_env()) == 2
    m.close()


def test_learns_as_soon_as_gpu_0_has_room_however_busy_it_is(tmp_path, fake_env):
    # It learns once GPU 0 has the memory it needs, however busy the card is: it never waits for the card to be idle.
    s = sample(tmp_path)
    probe = [dict(IDLE, free_mb=200.0, util=5.0)]                                # idle but nearly full: it waits
    m = models(tmp_path, s, probe=lambda: dict(probe[0]), gpu_quiet_s=30)       # a retired setting changes nothing
    m.ensure(s["id"])
    wait_for(lambda: m.describe(s["id"])["reason"] == "gpu_memory", timeout=5)
    time.sleep(0.3)
    view = m.describe(s["id"])
    assert not fake_env() and view["state"] == "waiting"
    # It says how much the sample's panel needs and how much is free (to the nearest 100 MB), and the card's size.
    table = m._record(s["id"])["table"]
    assert view["room"] == {"need_mb": Q.room_needed_mb(table["n_vars"], table["n_obs"]), "free_mb": 200,
                            "total_mb": int(IDLE["total_mb"]), "genes": table["n_vars"]}
    asked = time.monotonic()
    probe[0] = dict(IDLE, util=100.0)                                             # room, but fully busy: it goes now
    assert settle(m, s["id"])["state"] == "ready" and m.describe(s["id"])["room"] is None
    assert time.monotonic() - asked < 10 and len(fake_env()) == 1
    m.close()


def test_the_room_a_fit_needs_grows_with_the_panel_and_covers_what_was_measured():
    """The free memory a fit waits for (room_needed_mb) is above every peak the trainer was measured at on GPU 0 (a
    41k-cell table of a 5,101-gene panel cut to each panel, and a 3.5k-cell crop of it), by a margin of 10 to 60 %, and grows with the genes far more than
    with the cells."""
    measured = [(50, 41190, 860), (300, 41190, 1030), (1000, 41190, 1432), (2500, 41190, 2682), (5101, 41190, 4082),
                (10202, 41190, 6792), (5101, 3455, 4024)]
    for genes, cells, peak in measured:
        assert 1.1 * peak <= Q.room_needed_mb(genes, cells) <= 1.6 * peak, (genes, cells, peak, Q.room_needed_mb(genes, cells))
    needs = [Q.room_needed_mb(genes, 40000) for genes in (50, 300, 1000, 2500, 5000, 10000, 20000)]
    assert needs == sorted(needs) and all(need % 100 == 0 for need in needs)
    # The README's table, and the chart beside it (docs/media/mixing-memory.png), say this rule for 100,000 cells.
    root = Path(Q.__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    rows = re.findall(r"^  \| ([\d,]+) genes[^|]*\| about ([\d.]+) GB \|$", readme, re.M)
    assert [int(genes.replace(",", "")) for genes, _ in rows] == [300, 1000, 5000, 10000, 20000]
    for genes, said in rows:
        assert float(said) == Q.room_needed_mb(int(genes.replace(",", "")), 100_000) / 1000, (genes, said)
    assert "every 100,000 more cells add about 0.2 GB" in readme
    assert Q.room_needed_mb(5000, 200_000) - Q.room_needed_mb(5000, 100_000) == 200
    assert "/docs/media/mixing-memory.png)" in readme and (root / "docs/media/mixing-memory.png").is_file()
    assert Q.room_needed_mb(5000, 500000) - Q.room_needed_mb(5000, 40000) < Q.room_needed_mb(10000, 40000) - needs[4]


def test_a_card_too_small_for_the_panel_is_said_at_once_and_nothing_is_read(tmp_path, fake_env):
    """A GPU with less memory in all than the sample's panel needs can never learn it: failed at once, reason
    gpu_small, with the numbers (what it needs, the card's size), and no trainer starts."""
    s = sample(tmp_path)
    small = dict(IDLE, total_mb=1000.0, used_mb=100.0, free_mb=900.0)
    m = models(tmp_path, s, probe=lambda: dict(small))
    m.ensure(s["id"])
    view = settle(m, s["id"])
    table = m._record(s["id"])["table"]
    assert view["state"] == "failed" and view["reason"] == "gpu_small" and not fake_env()
    assert view["room"] == {"need_mb": Q.room_needed_mb(table["n_vars"], table["n_obs"]), "free_mb": 900, "total_mb": 1000,
                            "genes": table["n_vars"]}
    m.close()


def test_a_fixed_room_in_the_settings_is_waited_for_whatever_the_panel(tmp_path, fake_env):
    """[qc] gpu_min_free_mb as a number: that much free memory is what a fit waits for, whatever the panel."""
    s = sample(tmp_path)
    probe = [dict(IDLE, free_mb=2500.0)]
    m = models(tmp_path, s, probe=lambda: dict(probe[0]), gpu_min_free_mb=3000)
    m.ensure(s["id"])
    wait_for(lambda: m.describe(s["id"])["reason"] == "gpu_memory", timeout=5)
    assert m.describe(s["id"])["room"]["need_mb"] == 3000 and not fake_env()
    probe[0] = dict(IDLE, free_mb=3100.0)
    assert settle(m, s["id"])["state"] == "ready" and len(fake_env()) == 1
    m.close()


def test_progress_and_estimates_while_learning(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_EPOCH_S", "0.25")
    monkeypatch.setenv("FAKE_QC_EPOCHS", "6")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    wait_for(lambda: (m.describe(s["id"])["progress"] or {}).get("step", 0) >= 2, timeout=10)
    view = m.describe(s["id"])
    assert view["state"] == "learning" and view["progress"]["of"] == 6 and view["stage"] == "learning"
    assert 0 < view["eta_s"] < 60
    assert settle(m, s["id"])["state"] == "ready"
    m.close()


@pytest.mark.parametrize("mode, reason", [("error:not_settled", "not_settled"), ("error:unreadable", "unreadable"),
                                          ("error:bogus", "unexpected"), ("crash", "unexpected"),
                                          ("badexport", "unexpected"), ("wronggpu", "gpu")])
def test_failures_say_why_and_leave_nothing_behind(tmp_path, fake_env, monkeypatch, mode, reason):
    monkeypatch.setenv("FAKE_QC_MODE", mode)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "failed" and view["reason"] == reason and view["model"] is None
    assert not list((tmp_path / "var" / "qc-models" / s["id"]).glob("*/model.json"))
    assert not list((tmp_path / "var" / "qc-models" / s["id"]).glob(".*.part-*"))
    assert m.ensure(s["id"])["state"] == "failed" and len(fake_env()) == 1          # no relearning by itself ...
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    assert m.ensure(s["id"], retry=True)["state"] in ("waiting", "learning")      # ... only when asked (Try again)
    assert settle(m, s["id"])["state"] == "ready"
    m.close()


def test_a_device_on_another_bus_is_refused(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_BUS", "241")                                      # a card on another bus than GPU 0
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "failed" and view["reason"] == "gpu"
    m.close()


def test_out_of_memory_is_retried_once_after_a_pause(tmp_path, fake_env, monkeypatch):
    """Out of GPU 0's memory: the failed trainer's own memory is free again at once, so the one retry first waits
    OOM_RETRY_S, saying it waits for room on GPU 0 (it used to start ~0.1 s later, into whatever filled the card).
    Another sample queued during the pause waits too (queuing wakes the thread: that must not end the pause), and
    close() ends the pause at once, with no retry."""
    monkeypatch.setenv("FAKE_QC_MODE", "error:oom")
    monkeypatch.setattr(Q, "OOM_RETRY_S", 60)
    a, b = sample(tmp_path, "A"), sample(tmp_path, "B")
    m = models(tmp_path, a, b)
    m.ensure(a["id"])
    wait_for(lambda: m.describe(a["id"])["reason"] == "gpu_memory", timeout=10)     # the pause after the first try
    assert m.describe(a["id"])["state"] == "waiting" and len(fake_env()) == 1
    m.ensure(b["id"])
    time.sleep(0.5)
    assert m.describe(b["id"])["reason"] == "queue" and len(fake_env()) == 1         # B waits for the pause too
    m.close()
    m.thread.join(5)
    assert not m.thread.is_alive() and m.describe(a["id"])["reason"] == "gpu_memory" and len(fake_env()) == 1
    # A short pause: the retry starts once it has passed, never before, and the second failure is final.
    monkeypatch.setattr(Q, "OOM_RETRY_S", 0.5)
    m = models(tmp_path, a)
    m.ensure(a["id"])
    view = settle(m, a["id"])
    assert view["state"] == "failed" and view["reason"] == "oom"
    _, first, second = fake_env()
    assert second["start"] >= first["end"] + Q.OOM_RETRY_S
    m.close()


def test_time_limit_and_close_kill_the_trainer(tmp_path, fake_env, monkeypatch):
    import psutil
    monkeypatch.setenv("FAKE_QC_MODE", "hang")
    monkeypatch.setattr(Q, "KILL_GRACE_S", 0.2)
    s = sample(tmp_path)
    m = models(tmp_path, s, time_limit_s=1.0)
    m.ensure(s["id"])
    wait_for(lambda: fake_env(), timeout=10)
    pid = fake_env()[0]["pid"]
    view = settle(m, s["id"])
    assert view["state"] == "failed" and view["reason"] == "timeout"
    wait_for(lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, timeout=10)
    m.close()
    # close(): a trainer still learning is stopped, never left running.
    m = models(tmp_path, s)
    m.ensure(s["id"], retry=True)
    wait_for(lambda: len(fake_env()) == 2, timeout=10)
    pid = fake_env()[1]["pid"]
    m.close()
    wait_for(lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, timeout=10)
    m.thread.join(5)
    assert not m.thread.is_alive()


def test_a_trainer_that_never_says_hello_is_given_up(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_MODE", "nohello")
    monkeypatch.setattr(Q, "HELLO_TIMEOUT_S", 0.5)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "failed" and view["reason"] == "unexpected" and "did not start" in view["message"]
    m.close()


def test_tables_without_whole_counts_fall_back_to_the_next(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_NOT_COUNTS", "xenium_table")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "ready" and view["model"]["table"] == "table"
    assert [l["job"]["table"] for l in fake_env()] == ["xenium_table", "table"]
    assert json.loads((tmp_path / "var" / "qc-models" / s["id"] / "tables.json").read_text())["xenium_table"]["reason"] == "not_counts"
    m.close()
    only = sample(tmp_path, "only", tables=("xenium_table",))
    m = models(tmp_path, only)
    m.ensure(only["id"])
    view = settle(m, only["id"])
    assert view["state"] == "unavailable" and view["reason"] == "not_counts"
    m.close()


def test_unavailable_without_a_table_or_the_environment(tmp_path, fake_env):
    bare = sample(tmp_path, "bare", tables=())
    (Path(bare["path"])).mkdir(parents=True, exist_ok=True)
    flat = sample(tmp_path, "flat", tables=())
    write_table(Path(flat["path"]), "xenium_table", spatial=False)
    m = models(tmp_path, bare, flat)
    assert m.ensure(bare["id"])["state"] == "unavailable" and m.describe(bare["id"])["reason"] == "no_table"
    # A table of counts without cell centroids, annotating no outlines: said so, never "no table".
    assert m.ensure(flat["id"])["state"] == "unavailable" and m.describe(flat["id"])["reason"] == "positions"
    assert "obsm['spatial']" in m.describe(flat["id"])["message"]
    m.close()
    s = sample(tmp_path)
    m = models(tmp_path, s, python=str(tmp_path / "missing" / "python"))
    assert not m.available and m.ensure(s["id"])["reason"] == "software"
    m.close()
    with pytest.raises(Q.QcError) as unknown:
        models(tmp_path).describe("z000000000000")
    assert unknown.value.status == 404
    assert not fake_env()


# ---- whether the check can learn here (its probe: tests/fake_qc_probe.py) ---------------------------------------------
def probe_runs(monkeypatch, tmp_path):
    """The fake probe's runs (FAKE_QC_PROBE_LOG), read when called."""
    log = tmp_path / "probe-runs.jsonl"
    monkeypatch.setenv("FAKE_QC_PROBE_LOG", str(log))
    return lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def answered(m, timeout=15) -> dict:
    """The check's capability once its probe has answered."""
    wait_for(lambda: m.capability()["state"] != "checking", timeout=timeout)
    return m.capability()


def test_the_mixing_check_is_offered_where_it_can_learn(tmp_path, fake_env, monkeypatch, capsys):
    """The probe says why the check cannot learn on this computer, in the order that helps: then nothing is offered (a
    model already learned is not served either: Go deeper and Compare need the check too), nothing is learned, and
    each sample says the reason. Where it can learn, everything goes on as before; a fit asked while the probe has not
    answered yet waits for its answer, then learns. Each answer is one line in the check's own log, never in the
    engine's output (a background line at any moment of it)."""
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    assert settle(m, s["id"])["state"] == "ready" and len(fake_env()) == 1     # learned once, here
    assert m.capability()["state"] == "ready" and m.capability()["why"] == "torch 2.14.1 sees 1 CUDA GPU(s)"
    m.close()
    log = tmp_path / "state" / "logs" / "qc-trainer.log"
    assert "qc: the mixing check ready: torch 2.14.1 sees 1 CUDA GPU(s)\n" in log.read_text()
    assert "the mixing check" not in capsys.readouterr().out
    for mode, reason in (("missing:scvi-tools,scviva-tools", "software"), ("torch_cpu", "torch_cpu"),
                         ("no_nvidia", "no_gpu"), ("driver", "driver"), ("hidden", "no_gpu"),
                         ("mac_mps_fails", "apple_gpu")):
        monkeypatch.setenv("FAKE_QC_PROBE", mode)
        m = models(tmp_path, s)
        try:
            cap = answered(m)
            assert (cap["state"], cap["reason"]) == ("unavailable", reason) and not m.available, mode
            view = m.describe(s["id"])
            assert (view["state"], view["reason"], view["message"]) == ("unavailable", reason, cap["why"]), mode
            assert m.ensure(s["id"], retry=True)["state"] == "unavailable" and m.model(s["id"]) is None
            assert len(fake_env()) == 1, mode                                   # nothing learned
            assert f"qc: the mixing check unavailable: {cap['why']}\n" in log.read_text(), mode
        finally:
            m.close()
    # Before the probe answers the check is offered; a fit asked meanwhile waits for the answer, then learns.
    release, asked = threading.Event(), []
    monkeypatch.delenv("FAKE_QC_PROBE")

    def slow():
        asked.append(time.monotonic())
        release.wait(10)
        return {"ev": "probe", "missing": [], "torch": "2.14.1", "cuda_build": "13.0", "devices": 1, "nvidia": 1,
                "driver_cuda": "13.0", "error": None}
    other = sample(tmp_path, "sample-b-crop")
    m = Q.QcModels(qc_config(tmp_path), Registry(other), probe=lambda: dict(IDLE), prober=slow)
    try:
        wait_for(lambda: asked)
        assert m.capability()["state"] == "checking" and m.available
        assert m.ensure(other["id"])["state"] == "waiting"
        time.sleep(0.5)
        assert m.describe(other["id"])["state"] == "waiting" and len(fake_env()) == 1   # it waits for the answer
        release.set()
        assert settle(m, other["id"])["state"] == "ready" and len(fake_env()) == 2
    finally:
        m.close()


def test_an_answer_reaches_the_samples_already_looked_at(tmp_path, fake_env, monkeypatch):
    """A sample described before the probe answered (its model stored: ready) is looked at again when the answer says
    the check cannot learn here: its new state reaches the pages (snapshot, the WebSocket's) without anyone asking."""
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    assert settle(m, s["id"])["state"] == "ready"
    m.close()
    release = threading.Event()

    def slow():
        release.wait(10)
        return {"ev": "probe", "missing": [], "nvidia": None, "devices": 0, "cuda_build": None, "error": None}
    m = Q.QcModels(qc_config(tmp_path), Registry(s), probe=lambda: dict(IDLE), prober=slow)
    try:
        assert m.describe(s["id"])["state"] == "ready" and m.capability()["state"] == "checking"
        version = m.version
        release.set()
        wait_for(lambda: m.snapshot().get(s["id"], {}).get("state") == "unavailable", timeout=10)
        assert m.snapshot()[s["id"]]["reason"] == "no_gpu" and m.version > version and m.model(s["id"]) is None
    finally:
        m.close()


def test_a_probe_that_cannot_answer_leaves_the_check_offered(tmp_path, fake_env, monkeypatch):
    """A probe that crashes, prints nothing it can read, or hangs past PROBE_TIMEOUT_S is no answer ("unknown"): the
    check stays offered as it was before there was a probe (a stored model is served, a new sample learns, the
    trainer's own checks say what fails), and the probe is asked again after PROBE_RETRY_S, until it answers."""
    runs = probe_runs(monkeypatch, tmp_path)
    a, b = sample(tmp_path, "sample-a-crop"), sample(tmp_path, "sample-b-crop")
    m = models(tmp_path, a, b)
    m.ensure(a["id"])
    assert settle(m, a["id"])["state"] == "ready"
    m.close()
    monkeypatch.setattr(Q, "PROBE_TIMEOUT_S", 1.0)
    for mode in ("crash", "garbage", "hang"):
        monkeypatch.setenv("FAKE_QC_PROBE", mode)
        m = models(tmp_path, a, b)
        try:
            cap = answered(m)
            assert (cap["state"], cap["reason"]) == ("unknown", None) and m.available, mode
            assert m.describe(a["id"])["state"] == "ready" and m.model(a["id"]) is not None, mode
        finally:
            m.close()
    monkeypatch.setenv("FAKE_QC_PROBE", "crash")
    monkeypatch.setattr(Q, "PROBE_RETRY_S", (0.2,))
    m = models(tmp_path, a, b)
    try:
        assert answered(m)["state"] == "unknown"
        m.ensure(b["id"])
        assert settle(m, b["id"])["state"] == "ready" and len(fake_env()) == 2    # learned all the same
        before = len(runs())
        wait_for(lambda: len(runs()) >= before + 2, timeout=10)                   # asked again, and again
        monkeypatch.setenv("FAKE_QC_PROBE", "ready")
        wait_for(lambda: m.capability()["state"] == "ready", timeout=10)
        assert m.available and m.describe(b["id"])["state"] == "ready"
    finally:
        m.close()


def test_a_probe_answer_it_cannot_read_is_no_answer(tmp_path, fake_env, monkeypatch):
    """A probe line that reads as JSON but holds a value of another kind (a count as text, from a broken or foreign
    probe module) is no answer, as a line that does not read: the check stays offered ("unknown", the error in its
    words), and the probe's thread, which the error does not end, asks again later."""
    monkeypatch.setattr(Q, "PROBE_RETRY_S", (0.2,))
    asked = []

    def odd():
        asked.append(time.monotonic())
        return {"ev": "probe", "missing": [], "torch": "2.14.1", "cuda_build": "13.0", "devices": "1", "nvidia": 1,
                "driver_cuda": "13.0", "error": None}
    m = Q.QcModels(qc_config(tmp_path), Registry(), probe=lambda: dict(IDLE), prober=odd)
    try:
        cap = answered(m)
        assert (cap["state"], cap["reason"]) == ("unknown", None) and m.available and not m.probing
        assert cap["why"].startswith("its Python could not be asked (TypeError "), cap["why"]
        wait_for(lambda: len(asked) >= 2, timeout=10)                 # asked again: its thread lives on
        assert m.probe_thread.is_alive()
    finally:
        m.close()


def test_a_press_asks_the_probe_again(tmp_path, fake_env, monkeypatch):
    """Where the check is not offered, a press of its switch or Try again (ensure) asks the probe again, at most every
    PROBE_AGAIN_S: a computer set up meanwhile is answered without a restart, and the sample then learns."""
    runs = probe_runs(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_QC_PROBE", "missing:scvi-tools,scviva-tools")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    try:
        assert answered(m)["reason"] == "software" and len(runs()) == 1
        m.ensure(s["id"])                                     # just answered: not asked again yet
        time.sleep(0.5)
        assert len(runs()) == 1
        m.probed_at -= Q.PROBE_AGAIN_S                        # as if it answered PROBE_AGAIN_S ago
        m.ensure(s["id"])
        m.ensure(s["id"], retry=True)                         # two presses: one more probe
        wait_for(lambda: len(runs()) == 2 and not m.probing)
        time.sleep(0.5)
        assert len(runs()) == 2 and m.capability()["reason"] == "software"
        monkeypatch.setenv("FAKE_QC_PROBE", "ready")          # scvi-tools and scviva-tools installed meanwhile
        m.probed_at -= Q.PROBE_AGAIN_S
        m.ensure(s["id"])
        wait_for(lambda: m.capability()["state"] == "ready", timeout=10)
        # Once it has answered, the probe's thread looks at the sample again (its table, its cache): a press meanwhile
        # may find that look under way and the sample still unavailable, so it is pressed until the sample is queued.
        wait_for(lambda: m.ensure(s["id"])["state"] in ("waiting", "learning", "ready"), timeout=10)
        assert settle(m, s["id"])["state"] == "ready" and len(fake_env()) == 1
    finally:
        m.close()


def test_without_a_probe_thread_the_first_fit_probes_itself(tmp_path, fake_env):
    """An instance made without its threads (a caller runs its fits) asks the probe in the fit's own thread first."""
    s = sample(tmp_path)
    m = Q.QcModels(qc_config(tmp_path), Registry(s), start=False, probe=lambda: dict(IDLE))
    try:
        assert m.probe_thread is None and m.capability()["state"] == "checking"
        assert m.ensure(s["id"])["state"] == "waiting"
        started = time.monotonic()
        m._learn(m.records[s["id"]])
        assert time.monotonic() - started < 5 and m.capability()["state"] == "ready"
        assert m.describe(s["id"])["state"] == "ready" and len(fake_env()) == 1
    finally:
        m.close()


def test_with_auto_lacking_packages_the_card_comes_first(tmp_path, fake_env, monkeypatch):
    """[qc] python "auto" found this Python without the check's packages (the engine's settings say so:
    config["locate"]): the check is not offered, and the probe still runs in this Python so that a computer without an
    NVIDIA card is told that first, not to install packages that could not learn there; with a card the packages are
    the reason. Without that report (a python set off), no probe runs."""
    runs = probe_runs(monkeypatch, tmp_path)
    s = sample(tmp_path)
    located = {"path": "", "how": "auto", "missing": ["scvi", "scviva"], "problem": "", "fix": "", "passed": []}
    for mode, reason in (("no_nvidia", "no_gpu"), ("ready", "software"), ("hidden", "no_gpu")):
        monkeypatch.setenv("FAKE_QC_PROBE", mode)
        m = Q.QcModels({**qc_config(tmp_path, python=""), "locate": {"qc": located}}, Registry(s),
                       probe=lambda: dict(IDLE))
        try:
            wait_for(lambda: not m.probing and m.probed_at > -math.inf)
            cap = m.capability()
            assert (cap["state"], cap["reason"]) == ("unavailable", reason) and not m.available, mode
            if reason == "software":
                assert cap["why"] == "this Python environment lacks scvi-tools and scviva-tools"
            view = m.describe(s["id"])
            assert (view["state"], view["reason"]) == ("unavailable", reason), mode
        finally:
            m.close()
    assert len(runs()) == 3 and runs()[0]["mode"] == "no_nvidia"
    m = models(tmp_path, s, python="")
    try:
        time.sleep(0.5)
        assert m.capability() == {"state": "unavailable", "reason": "software", "why": "python under [qc] is off",
                                  "probe": None}
        assert m.probe_thread is None and len(runs()) == 3 and m.describe(s["id"])["reason"] == "software"
    finally:
        m.close()
    assert not fake_env()


def test_the_probe_runs_in_the_trainers_environment(tmp_path, fake_env, monkeypatch):
    """The probe sees exactly the GPU the trainer would ([qc] gpu as its only one, numbered as nvidia-smi numbers them),
    and torch's weaker NVML check is never asked for."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    m = models(tmp_path)
    try:
        cap = answered(m)
        assert cap["probe"]["env"] == {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                       "PYTORCH_NVML_BASED_CUDA_CHECK": True}
    finally:
        m.close()
    bare = Q.QcModels(qc_config(tmp_path), Registry(), start=False)
    assert bare._trainer_env(sys.executable)["CUDA_VISIBLE_DEVICES"] == "0"


def write_counts(root: Path, key: str, values):
    """tables/<key>/X/data holding these values (a real zarr array: what the engine reads of a table's counts)."""
    import shutil
    import zarr
    values = np.asarray(values, np.float32)
    target = root / "tables" / key / "X" / "data"
    shutil.rmtree(target, ignore_errors=True)
    zarr.create_array(store=str(target), shape=values.shape, dtype=values.dtype, chunks=values.shape)[:] = values


def annotate(root: Path, key: str, region: str):
    """tables/<key> annotating the shapes element `region` (as spatialdata writes it in the group's attributes), and
    that element, so the trainer can place the table's cells by their outlines."""
    meta = root / "tables" / key / "zarr.json"
    data = json.loads(meta.read_text())
    data["attributes"].update(region=region, region_key="region", instance_key="cell_id")
    meta.write_text(json.dumps(data))
    (root / "shapes" / region).mkdir(parents=True, exist_ok=True)
    (root / "shapes" / region / "zarr.json").write_text(json.dumps({"node_type": "group", "attributes": {}}))


def test_any_counts_table_is_found_and_unlearnable_ones_are_said_before_the_wait(tmp_path, fake_env, monkeypatch):
    """A counts table is learned whatever it is called (after the configured names, any other by
    name); one the trainer would refuse is passed over before any wait for GPU 0 when its metadata already says so:
    no cell centroids and no outlines to place its cells by ("positions"), fewer rows than the trainer needs
    ("too_few_cells", failed as the trainer would say it: Try again looks again, at once), counts that are not whole
    numbers in their first chunk ("not_counts": a log-normalised table). A table without centroids that annotates
    outlines is learned (the trainer places its cells by them)."""
    monkeypatch.setattr(Q, "KEY_TTL_S", 0.0)
    cells = sample(tmp_path, "cells", tables=("cells",))
    m = models(tmp_path, cells)
    assert m.describe(cells["id"])["table"] == "cells"
    m.ensure(cells["id"])
    view = settle(m, cells["id"])
    assert view["state"] == "ready" and view["model"]["table"] == "cells"
    (job,) = [launch["job"] for launch in fake_env()]
    assert job["table"] == "cells" and job["other_tables"] == ["xenium_table", "table"]
    m.close()
    lognorm = sample(tmp_path, "lognorm", tables=("xenium_table",))
    write_counts(Path(lognorm["path"]), "xenium_table", np.log1p(np.arange(1, 9000) / 7.0))
    tiny = sample(tmp_path, "tiny", tables=())
    write_table(Path(tiny["path"]), "table", n_obs=12)
    both = sample(tmp_path, "both")                                      # a log-normalised table, then a good one
    write_counts(Path(both["path"]), "xenium_table", np.full(9000, 0.25))
    write_counts(Path(both["path"]), "table", np.arange(9000) % 7 + 1)
    outlined = sample(tmp_path, "outlined", tables=())
    write_table(Path(outlined["path"]), "xenium_table", spatial=False)
    annotate(Path(outlined["path"]), "xenium_table", "cell_boundaries")
    m = models(tmp_path, lognorm, tiny, both, outlined)
    view = m.ensure(lognorm["id"])
    assert (view["state"], view["reason"]) == ("unavailable", "not_counts") and "whole" in view["message"]
    view = m.ensure(tiny["id"])
    assert (view["state"], view["reason"]) == ("failed", "too_few_cells") and "12 cells" in view["message"]
    assert m.ensure(tiny["id"], retry=True)["state"] in ("waiting", "failed")
    assert settle(m, tiny["id"])["reason"] == "too_few_cells"
    m.ensure(both["id"])
    m.ensure(outlined["id"])
    assert settle(m, both["id"])["model"]["table"] == "table"
    assert settle(m, outlined["id"])["model"]["table"] == "xenium_table"
    assert [launch["job"]["table"] for launch in fake_env()[1:]] == ["table", "xenium_table"]   # nothing else started
    assert Q.table_info(outlined["path"], "xenium_table")["outlines"] is True
    plain_table = Q.table_info(both["path"], "table")
    assert plain_table["region"] == [] and plain_table["outlines"] is False
    m.close()


def test_the_first_counts_of_a_table_say_whether_they_are_whole(tmp_path):
    """What the engine reads of a table's counts before any fit: one chunk of X/data (sparse X) or
    the first rows of a dense X (zarr v3 or v2), at most `limit` values; None when they cannot be read (a table of
    metadata only: the trainer reads them all anyway)."""
    import zarr
    root = tmp_path / "s.zarr"
    write_table(root, "sparse")
    write_counts(root, "sparse", np.arange(5000) % 9)
    assert np.array_equal(Q.first_counts(root, "sparse", limit=100), np.arange(100) % 9)
    write_table(root, "dense", n_obs=300, n_vars=40, dense=True)
    (root / "tables" / "dense" / "X" / "zarr.json").unlink()
    zarr.create_array(store=str(root / "tables" / "dense" / "X"), shape=(300, 40), dtype="int64", chunks=(100, 40),
                      zarr_format=2)[:] = np.arange(12000).reshape(300, 40) % 5
    first = Q.first_counts(root, "dense", limit=400)
    assert first.shape == (10, 40) and Q.whole_counts(first)
    write_table(root, "metadata_only")
    assert Q.first_counts(root, "metadata_only") is None and Q.first_counts(root, "missing") is None
    assert Q.whole_counts(np.array([0, 3, 12], np.float32)) and Q.whole_counts(np.array([], np.float32))
    assert not Q.whole_counts(np.array([1.0, 0.5])) and not Q.whole_counts(np.array([-1.0, 2.0]))
    assert not Q.whole_counts(np.array([1.0, np.nan])) and not Q.whole_counts(np.array([-1], np.int32))


def test_a_table_whose_cells_cannot_be_placed_gives_way_to_the_next(tmp_path, fake_env, monkeypatch):
    """The trainer's "positions" (a table whose cells it cannot place on the transcripts) belongs to
    the table, as "not_counts" does: remembered for its fingerprint, the next table tried, and the sample unavailable,
    saying why, once none is left; nothing is learned again until a table changes."""
    monkeypatch.setenv("FAKE_QC_MODE", "error:positions")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert (view["state"], view["reason"]) == ("unavailable", "positions")
    assert [launch["job"]["table"] for launch in fake_env()] == ["xenium_table", "table"]
    rejected = json.loads((tmp_path / "var" / "qc-models" / s["id"] / "tables.json").read_text())
    assert {k: v["reason"] for k, v in rejected.items()} == {"xenium_table": "positions", "table": "positions"}
    assert m.ensure(s["id"])["reason"] == "positions" and len(fake_env()) == 2
    m.close()
    # A refusal remembered under the trainer's earlier rules (a table placed by its outlines refused for "positions")
    # is not taken: the tables are tried again.
    for note in rejected.values():
        note.pop("rules")
    (tmp_path / "var" / "qc-models" / s["id"] / "tables.json").write_text(json.dumps(rejected))
    again = models(tmp_path, s)
    again.ensure(s["id"])
    settle(again, s["id"])
    assert len(fake_env()) == 4
    again.close()


def test_every_sample_the_engine_knows_has_its_mixing_check(tmp_path):
    """The check found a sample by the open dialog's list (its 30 shown), so a sample the engine still knew
    (opened again for a run and kept off the list, or past the 30) said "Unknown sample" there while its detail
    answered."""
    from sopa_companion.engine.zarr_samples import RECENT_SHOWN, ZarrSamples
    entries = [{"id": f"z{k:02d}", "name": f"s{k}", "path": str(tmp_path / f"s{k}.zarr"), "kind": "zarr"}
               for k in range(RECENT_SHOWN + 2)]
    entries[1]["hidden"] = True
    (tmp_path / "var").mkdir(parents=True, exist_ok=True)
    (tmp_path / "var" / "samples.json").write_text(json.dumps(entries))
    zarr = ZarrSamples({"zarr": {"python": ""}, "proseg": {"max_region_um": 1000.0},
                        "paths": {"cache_dir": str(tmp_path / "cache"), "data_dir": str(tmp_path / "var")}})
    shown = {e["id"] for e in zarr.list()}
    m = Q.QcModels(qc_config(tmp_path), zarr, start=False, probe=lambda: dict(IDLE))
    try:
        for sid in ("z01", f"z{RECENT_SHOWN + 1:02d}"):       # kept off the list; past its 30
            assert sid not in shown and m.describe(sid)["sample"] == sid
        with pytest.raises(Q.QcError, match="Unknown sample"):
            m.describe("znobody")
    finally:
        m.close()
        zarr.close()


def test_the_job_carries_the_open_samples_frame(tmp_path, fake_env):
    """The trainer places a table's cells in the transcripts' frame: the job names the sample's
    points element and the transcripts' extent as the engine's open sample has them, and nothing when the engine has
    no such sample at hand (the trainer then reads them itself; the helper is never started for it)."""
    s = sample(tmp_path)

    class Open(Registry):
        details = {s["id"]: {"points_key": "transcripts", "extent": [0, 0.5, 400, 300.0]}}
    m = Q.QcModels(qc_config(tmp_path), Open(s), probe=lambda: dict(IDLE))
    m.ensure(s["id"])
    settle(m, s["id"])
    m.close()
    job = fake_env()[-1]["job"]
    assert job["points_key"] == "transcripts" and job["extent"] == [0.0, 0.5, 400.0, 300.0]
    Open.details = {s["id"]: {"points_key": "transcripts", "extent": [0, None, 400, 300]}}
    m = Q.QcModels(qc_config(tmp_path / "other"), Open(s), probe=lambda: dict(IDLE))
    assert m._frame(s["id"]) == {} and m._frame("z000000000000") == {}
    m.close()
    plain = Q.QcModels(qc_config(tmp_path / "plain"), Registry(s), probe=lambda: dict(IDLE))
    assert plain._frame(s["id"]) == {}
    plain.close()


def test_software_change_relearns_and_old_models_are_pruned(tmp_path, fake_env, monkeypatch):
    s = sample(tmp_path)
    m = models(tmp_path, s, keep_models=1)
    m.ensure(s["id"])
    first = settle(m, s["id"])["model"]["id"]
    m.versions_cache = (time.monotonic(), {**Q.env_versions(sys.executable), "torch": "9.9.9"})   # an upgraded env
    m.records[s["id"]]["checked"] = -1e9
    view = m.describe(s["id"])
    assert view["state"] == "off" and view["relearn"] is True                      # "learning it again"
    m.ensure(s["id"])
    second = settle(m, s["id"])["model"]["id"]
    assert second != first
    folders = lambda: [p.parent.name for p in (tmp_path / "var" / "qc-models" / s["id"]).glob("*/model.json")]
    # The old model goes just after the new one reads as ready (_install: ready, then _prune): waited for, not raced.
    wait_for(lambda: folders() == [second])                                        # keep_models = 1
    m.close()


def test_a_shared_models_folder_never_prunes_what_another_session_uses(tmp_path, fake_env, monkeypatch):
    """[qc] models_dir is shared by API sessions, each knowing only its own samples: a session's prune leaves a model
    any session used within SHARED_KEEP_S. One unused for longer goes, and the session that had it says so at its next
    look (off, nothing to score with) instead of handing out files that are gone."""
    a, b = sample(tmp_path, "A"), sample(tmp_path, "B")

    def session(name, s):
        config = qc_config(tmp_path / name, keep_models=1, models_dir=str(tmp_path / "models"))
        return Q.QcModels(config, Registry(s), probe=lambda: dict(IDLE))
    first, second = session("first", a), session("second", b)
    try:
        first.ensure(a["id"])
        assert settle(first, a["id"])["state"] == "ready"
        folder = Path(first.model(a["id"])["dir"])
        second.ensure(b["id"])
        assert settle(second, b["id"])["state"] == "ready"
        assert folder.is_dir() and first.model(a["id"])["dir"] == str(folder)    # the second's prune left it
        long_ago = time.time() - Q.SHARED_KEEP_S - 60
        os.utime(folder / "used", (long_ago, long_ago))
        second._prune()
        assert not folder.exists() and Path(second.model(b["id"])["dir"]).is_dir()
        assert first.model(a["id"]) is None and first.describe(a["id"])["state"] == "off"
    finally:
        first.close()
        second.close()


def slow_tables(monkeypatch, seconds=0.3):
    """table_info as a cold NTFS read (slower here than there): a look at a sample takes `seconds`."""
    real = Q.table_info

    def slow(path, key):
        time.sleep(seconds)
        return real(path, key)
    monkeypatch.setattr(Q, "table_info", slow)


def test_a_look_that_ends_late_never_undoes_a_queued_job(tmp_path, fake_env, monkeypatch):
    """The analysis thread looks at a sample (model(), every few seconds) while a request queues it (ensure(), e.g. a
    run created): the look, begun before, must neither turn "waiting" back into "off" nor leave the queued job without
    a key (it was dropped silently), and the queued job learns."""
    s = sample(tmp_path)
    m = Q.QcModels(qc_config(tmp_path), Registry(s), start=False, probe=lambda: dict(IDLE))
    slow_tables(monkeypatch)
    looking = threading.Thread(target=m.model, args=(s["id"],))
    looking.start()
    time.sleep(0.05)
    assert m.ensure(s["id"])["state"] == "waiting" and m.queue == [s["id"]]   # its first look has not ended yet
    looking.join()
    record = m.records[s["id"]]
    assert m.describe(s["id"])["state"] == "waiting" and m.queue == [s["id"]] and record["reason"] == "queue"
    m._learn(record)                                     # the qc-models thread's turn: it looks again, then learns
    assert record["state"] == "ready" and record["key"] and len(fake_env()) == 1
    m.close()


def test_a_queued_job_learns_what_is_true_when_its_turn_comes(tmp_path, fake_env, monkeypatch):
    """A job waits (no room on GPU 0, another fit): a model of its key stored meanwhile by another engine (or the
    pre-learning command) is used, never learned twice; a table saved again meanwhile is learned under its new key."""
    s = sample(tmp_path)
    waiting = Q.QcModels(qc_config(tmp_path), Registry(s), start=False, probe=lambda: dict(IDLE))
    assert waiting.ensure(s["id"])["state"] == "waiting"
    other = models(tmp_path, s)                          # another engine on the same data dir learns it first
    other.ensure(s["id"])
    stored = settle(other, s["id"])["model"]["id"]
    other.close()
    record = waiting.records[s["id"]]
    waiting._learn(record)
    assert record["state"] == "ready" and record["key"] == stored and len(fake_env()) == 1    # no second fit
    # A key that went stale while the job waited: the table was saved again.
    monkeypatch.setattr(Q, "KEY_TTL_S", 0.0)
    write_table(Path(s["path"]), "xenium_table", nnz=9001)
    assert waiting.ensure(s["id"])["state"] == "waiting"
    queued = record["key"]
    write_table(Path(s["path"]), "xenium_table", nnz=9002)
    waiting._learn(record)
    assert record["state"] == "ready" and record["key"] != queued and len(fake_env()) == 2
    assert (tmp_path / "var" / "qc-models" / s["id"] / record["key"] / "model.json").is_file()
    assert fake_env()[-1]["job"]["model_id"] == record["key"]
    waiting.close()


def test_a_broken_model_folder_does_not_block_its_key_forever(tmp_path, fake_env):
    """What an interrupted prune or a crash left of a model (its folder without model.json) is moved aside when the
    same model is stored again; before, every relearn trained for minutes and then failed to store it."""
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.describe(s["id"])
    broken = tmp_path / "var" / "qc-models" / s["id"] / m.records[s["id"]]["key"]
    broken.mkdir(parents=True)
    (broken / "cells.parquet").write_bytes(b"half a model")
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "ready" and (broken / "model.json").is_file()
    assert (broken / "cells.parquet").read_bytes() != b"half a model"
    assert not list(broken.parent.glob(".*"))            # nothing moved aside is left behind
    m.close()


@pytest.mark.parametrize("mode, words", [("truncated", "failed its check: BadZipFile"),
                                         ("badprogress", "following the learning process failed: ValueError")])
def test_a_trainer_is_never_left_running_whatever_goes_wrong(tmp_path, fake_env, monkeypatch, mode, words):
    """A damaged export (BadZipFile, not an OSError) or a malformed progress line: the fit fails with a reason, the
    trainer is stopped (before, it went on with nobody following it and the fit locks free), the staging folder goes."""
    import psutil
    monkeypatch.setenv("FAKE_QC_MODE", mode)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "failed" and view["reason"] == "unexpected" and words in view["message"], view
    pid = fake_env()[0]["pid"]
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert m.process is None and not list((tmp_path / "var" / "qc-models" / s["id"]).glob(".*.part-*"))
    held = m._fit_lock()                                  # the locks were released with the trainer gone
    assert held is not None
    for handle in held:
        handle.close()
    m.close()


class StuckProcess:
    """A trainer that outlives SIGKILL (stuck in the GPU driver) until `exited` is set."""

    def __init__(self):
        import types
        self.pid, self.exited, self.signals = 424242, False, []
        self.stdin = types.SimpleNamespace(close=lambda: None)

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout=None):
        if self.exited:
            return 0
        import subprocess
        raise subprocess.TimeoutExpired("trainer", timeout)

    def terminate(self):
        self.signals.append("TERM")

    def kill(self):
        self.signals.append("KILL")


def test_a_trainer_that_outlives_sigkill_keeps_its_fit_locks(tmp_path):
    """_kill never raises (it aborted the engine's shutdown, or freed the locks of a trainer still on GPU 0): a stuck
    trainer is reported, and no other fit starts until it is gone. The models run no thread of their own here: the
    probe's thread starts a process, which for a moment holds a copy of every open file, the fit locks included, so a
    lock just let go could read as held."""
    m = Q.QcModels(qc_config(tmp_path), Registry(), probe=lambda: dict(IDLE), start=False)
    stuck = StuckProcess()
    m.process = stuck
    assert m._kill() is False and stuck.signals == ["TERM", "KILL"]
    assert "did not exit after SIGKILL" in (tmp_path / "state" / "logs" / "qc-trainer.log").read_text()
    m.close()                                             # never raises either
    held = Q.QcModels(qc_config(tmp_path / "x"), Registry(), start=False)._fit_lock()
    m.stuck.append((stuck, held))                         # what _learn keeps after such a fit
    assert m._fit_lock() is None                          # no fit while it may still hold GPU 0
    stuck.exited = True
    again = m._fit_lock()
    assert again is not None and all(handle.closed for handle in held)
    for handle in again:
        handle.close()


def test_the_prelearning_command_keeps_to_its_data_dir_and_waits_for_the_app(tmp_path, fake_env, monkeypatch):
    """Decision 2's pass over every sample: models go into the configured data dir (where its engine looks), the log
    next to them (never into the app's state dir), and it refuses while the app runs or an engine owns the data dir."""
    import socket
    from sopa_companion.engine.config import load_config
    from sopa_companion.server.runtime_file import acquire_engine_lock
    monkeypatch.setattr(Q, "PRELEARN_POLL_S", 0.05)
    s = sample(tmp_path)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    toml = tmp_path / "prelearn.toml"
    toml.write_text(f"""[server]\nport = {port}\n[paths]\ndata_dir = "{tmp_path / 'data'}"\nstate_dir = "{tmp_path / 'state'}"
[qc]\npython = "{sys.executable}"\ntrainer_module = "{FAKE}"\nprobe_module = "{FAKE_PROBE}"\ngpu_quiet_s = 0\n""")
    args = ["learn", s["path"], "--config", str(toml)]
    assert Q.main(args, probe=lambda: dict(IDLE)) == 0
    root = tmp_path / "data" / "qc-models"
    assert list(root.glob(f"{s['id']}/*/model.json")) and "learned" in (root / "prelearn.log").read_text()
    assert not (tmp_path / "state").exists()              # nothing in the state dir
    config = load_config(str(toml))
    lock = acquire_engine_lock(config)                    # an engine owns the data dir
    try:
        assert Q.main(args, probe=lambda: dict(IDLE)) == 2
    finally:
        os.close(lock)
    runtime = Path(config["paths"]["state_dir"]) / "run" / "server.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(json.dumps({"schema": 1, "pid": os.getpid(), "port": port}))   # the app runs
    assert Q.app_running(config) and Q.main(args, probe=lambda: dict(IDLE)) == 2
    runtime.unlink()
    with socket.socket() as listening:                    # something holds the app's port
        listening.bind(("127.0.0.1", port))
        listening.listen()
        assert "listens on port" in Q.app_running(config)
    assert Q.app_running(config) is None and len(fake_env()) == 1


# ---- Go deeper's details ---------------------------------------------------------------------------------------------
def detail_settle(m, sid, timeout=30):
    wait_for(lambda: m.describe(sid)["detail"]["state"] not in ("queued", "learning"), timeout=timeout)
    return m.describe(sid)


def folder_files(folder: Path) -> dict:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file() and p.name != "used"}


def test_a_fit_installs_its_details_checked_and_says_where(tmp_path, fake_env):
    """New fits write deep.npz after encoder.npz: the engine loads it strictly and self-tests it against the calculator
    before installing; describe says the model has its details (model.deep, detail "ready") and the model's
    provenance; model() names the file."""
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    folder = Path(m.model(s["id"])["dir"])
    encoder = T.load_encoder(folder / "encoder.npz")
    assert T.deep_self_test(T.load_deep(folder / "deep.npz", encoder), encoder) <= Q.DEEP_TOLERANCE
    assert view["model"]["deep"] is True and m.model(s["id"])["deep"] == str(folder / "deep.npz")
    assert view["detail"] == {"state": "ready", "progress": None, "eta_s": None, "reason": None}
    provenance = view["model"]["provenance"]
    assert provenance["table"] == "xenium_table" and provenance["other_table"] == "table"
    assert provenance["priors"] == "pooled" and provenance["epochs_run"] == 3 and provenance["epochs_notebook"] == 50
    assert isinstance(provenance["differences"], list) and isinstance(provenance["notebook_equivalent"], bool)
    m.close()


def test_a_details_file_failing_its_check_is_left_out_and_the_model_still_scores(tmp_path, fake_env, monkeypatch):
    """A deep.npz that loads but fails its self-test: the engine leaves it out (model.json says why, as the trainer
    does when its own export fails); the calculator is installed and scores; only Go deeper says the details could
    not be added (detail "failed" with the reason, not "none": the window must not say the sample predates them), and
    Learn the details (its Try again) can still be asked for."""
    monkeypatch.setenv("FAKE_QC_MODE", "baddeep")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    folder = Path(m.model(s["id"])["dir"])
    data = json.loads((folder / "model.json").read_text())
    assert view["state"] == "ready" and view["model"]["deep"] is False
    assert view["detail"] == {"state": "failed", "progress": None, "eta_s": None, "reason": "unexpected"}
    assert not (folder / "deep.npz").exists() and data["deep"] is None and "engine's check" in data["deep_error"]["message"]
    assert m.model(s["id"])["deep"] is None and T.load_encoder(folder / "encoder.npz")
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    asked = m.ensure(s["id"], detail=True)
    assert asked["state"] == "ready" and asked["detail"]["state"] in ("queued", "learning")
    view = detail_settle(m, s["id"])
    assert view["detail"]["state"] == "ready" and view["model"]["deep"] is True
    assert "deep_error" not in json.loads((folder / "model.json").read_text())
    m.close()


def test_a_model_learned_before_go_deeper_gets_its_details_from_a_detail_job(tmp_path, fake_env, monkeypatch):
    """Learn the details: the trainer learns the same recipe again with "detail_for" the installed folder,
    on GPU 0 with the fit's rules, while the sample stays ready (the check goes on scoring); its deep.npz is moved into
    the installed folder and model.json rewritten with "deep", "detail" and "provenance". The key, the calculator and
    the parquet files stay byte for byte, so every run's results stay. Asked again: nothing to do."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)          # GPU 0 itself (the fake trainer uses none)
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    monkeypatch.setenv("FAKE_QC_EPOCH_S", "0.15")
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    view = settle(m, s["id"])
    key, folder = view["model"]["id"], Path(m.model(s["id"])["dir"])
    assert view["model"]["deep"] is False and view["detail"]["state"] == "none" and m.model(s["id"])["deep"] is None
    # An older model.json has no provenance of its own: the trainer's rule works it out from the rest of the file.
    assert view["model"]["provenance"]["table"] == "xenium_table" and view["model"]["provenance"]["priors"] == "pooled"
    before = folder_files(folder)
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    asked = m.ensure(s["id"], detail=True)
    assert asked["state"] == "ready" and asked["detail"]["state"] in ("queued", "learning")
    during = {}                         # the view in which a step was seen, not a later one (the job may end meanwhile)
    wait_for(lambda: (during.update(m.describe(s["id"])) or (during["detail"]["progress"] or {}).get("step", 0) >= 1),
             timeout=10)
    assert during["state"] == "ready" and during["reason"] is None and m.model(s["id"]) is not None   # still scores
    assert during["detail"]["state"] == "learning" and during["detail"]["progress"]["of"] == 3
    assert during["progress"] is None and during["stage"] is None        # the sample's own fields stay as they were
    view = detail_settle(m, s["id"])
    assert view["detail"] == {"state": "ready", "progress": None, "eta_s": None, "reason": None}
    assert view["model"]["deep"] is True and view["model"]["id"] == key and m.model(s["id"])["deep"] == str(folder / "deep.npz")
    after = folder_files(folder)
    assert {k: after[k] for k in ("encoder.npz", "cells.parquet", "batch.parquet")} == \
        {k: before[k] for k in ("encoder.npz", "cells.parquet", "batch.parquet")}
    old, new = json.loads(before["model.json"]), json.loads(after["model.json"])
    assert {k: v for k, v in new.items() if k not in ("deep", "detail", "provenance")} == old
    assert new["deep"]["from"] == "detail" and new["detail"]["accepted"] and new["provenance"]["table"] == "xenium_table"
    encoder = T.load_encoder(folder / "encoder.npz")
    assert T.deep_self_test(T.load_deep(folder / "deep.npz", encoder), encoder) <= Q.DEEP_TOLERANCE
    fit, detail = fake_env()
    assert detail["job"]["detail_for"] == str(folder) and detail["job"]["model_id"] == key
    assert detail["env"]["CUDA_VISIBLE_DEVICES"] == "0" and detail["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert detail["nice"] == 19 and "detail_for" not in fit["job"]
    assert not list(folder.parent.glob(".*.detail-*"))                   # its staging folder went
    assert m.ensure(s["id"], detail=True)["detail"]["state"] == "ready" and len(fake_env()) == 2   # nothing to do
    m.close()
    again = models(tmp_path, s)                                          # another engine sees them too
    assert again.describe(s["id"])["model"]["deep"] is True and again.model(s["id"])["id"] == key
    again.close()


@pytest.mark.parametrize("mode, reason", [("detail_mismatch", "detail_mismatch"), ("baddeep", "unexpected"),
                                          ("crash", "unexpected"), ("error:oom", "oom")])
def test_a_detail_job_that_fails_changes_nothing_and_can_be_tried_again(tmp_path, fake_env, monkeypatch, mode, reason):
    """The parity rule refused (detail_mismatch: the new learning is another model), a details file failing the
    engine's check against the installed calculator, a crash, running out of GPU memory (retried once, as a fit): the
    installed model is left exactly as it was and keeps scoring; the detail job says why (Try again: ensure(detail))."""
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    monkeypatch.setattr(Q, "OOM_RETRY_S", 0.1)                         # the retry's pause (its own test below)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    key = settle(m, s["id"])["model"]["id"]
    folder = Path(m.model(s["id"])["dir"])
    before = folder_files(folder)
    monkeypatch.setenv("FAKE_QC_MODE", mode)
    m.ensure(s["id"], detail=True)
    view = detail_settle(m, s["id"])
    assert view["detail"]["state"] == "failed" and view["detail"]["reason"] == reason, view["detail"]
    assert view["state"] == "ready" and view["model"]["id"] == key and view["model"]["deep"] is False
    assert folder_files(folder) == before and not list(folder.parent.glob(".*.detail-*"))
    assert len(fake_env()) == (3 if mode == "error:oom" else 2)          # one retry after running out of memory
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    assert m.ensure(s["id"], detail=True)["detail"]["state"] in ("queued", "learning")
    assert detail_settle(m, s["id"])["detail"]["state"] == "ready" and m.describe(s["id"])["model"]["deep"] is True
    m.close()


def test_a_detail_job_out_of_memory_waits_out_the_same_pause(tmp_path, fake_env, monkeypatch):
    """As a fit: a detail job that ran out of GPU 0's memory is tried once more only after OOM_RETRY_S, saying
    meanwhile that it waits for room on GPU 0 while the sample stays ready and scores; close() during the pause ends it
    at once, with no retry."""
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    monkeypatch.setattr(Q, "OOM_RETRY_S", 60)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    settle(m, s["id"])
    monkeypatch.setenv("FAKE_QC_MODE", "error:oom")
    m.ensure(s["id"], detail=True)
    wait_for(lambda: m.describe(s["id"])["detail"]["reason"] == "gpu_memory", timeout=10)    # the pause
    view = m.describe(s["id"])
    assert view["detail"]["state"] == "queued" and view["state"] == "ready" and m.model(s["id"]) is not None
    m.close()
    m.thread.join(5)
    assert not m.thread.is_alive() and m.describe(s["id"])["detail"]["reason"] == "gpu_memory"
    assert len(fake_env()) == 2                                          # the fit and the first try, no retry
    monkeypatch.setattr(Q, "OOM_RETRY_S", 0.5)                           # a short pause: the retry after it
    m = models(tmp_path, s)
    m.ensure(s["id"], detail=True)
    view = detail_settle(m, s["id"])
    assert view["detail"]["state"] == "failed" and view["detail"]["reason"] == "oom"
    _, _, first, second = fake_env()
    assert second["start"] >= first["end"] + Q.OOM_RETRY_S
    m.close()


def test_a_detail_job_has_the_fits_time_limit_and_close_stops_it(tmp_path, fake_env, monkeypatch):
    """A detail job that never ends is stopped at the time limit (the trainer killed, its staging folder removed, the
    model as it was); close() stops one in progress, and the job is not left "learning"."""
    import psutil
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    monkeypatch.setattr(Q, "KILL_GRACE_S", 0.2)
    s = sample(tmp_path)
    m = models(tmp_path, s)
    m.ensure(s["id"])
    settle(m, s["id"])
    m.s.time_limit_s = 1.0                       # the detail job's limit: on a busy machine the fit itself takes longer
    folder = Path(m.model(s["id"])["dir"])
    before = folder_files(folder)
    monkeypatch.setenv("FAKE_QC_MODE", "hang")
    m.ensure(s["id"], detail=True)
    view = detail_settle(m, s["id"])
    assert view["detail"]["state"] == "failed" and view["detail"]["reason"] == "timeout" and view["state"] == "ready"
    pid = fake_env()[1]["pid"]
    wait_for(lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, timeout=10)
    assert folder_files(folder) == before and not list(folder.parent.glob(".*.detail-*"))
    m.ensure(s["id"], detail=True)                                       # Try again, then the engine closes
    wait_for(lambda: len(fake_env()) == 3, timeout=10)
    pid = fake_env()[2]["pid"]
    m.close()
    wait_for(lambda: not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, timeout=10)
    m.thread.join(5)
    assert not m.thread.is_alive() and m.describe(s["id"])["detail"]["state"] != "learning"
    assert folder_files(folder) == before


def test_a_detail_job_waits_for_gpu_0_and_the_sample_stays_ready(tmp_path, fake_env, monkeypatch):
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    s = sample(tmp_path)
    probe = [dict(IDLE)]
    m = models(tmp_path, s, probe=lambda: dict(probe[0]))
    m.ensure(s["id"])
    settle(m, s["id"])
    probe[0] = dict(IDLE, free_mb=200.0)
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    m.ensure(s["id"], detail=True)
    wait_for(lambda: m.describe(s["id"])["detail"]["reason"] == "gpu_memory", timeout=5)
    view = m.describe(s["id"])
    assert view["detail"]["state"] == "queued" and view["state"] == "ready" and view["reason"] is None
    assert len(fake_env()) == 1
    probe[0] = dict(IDLE)
    assert detail_settle(m, s["id"])["detail"]["state"] == "ready" and len(fake_env()) == 2
    m.close()


def test_the_engine_checks_a_detail_export_against_the_installed_calculator(tmp_path):
    """_check_detail: this model's model.json with its "deep" block, deep.npz self-tested against the INSTALLED
    calculator, a refused parity record or a different new calculator beside it (the trainer's .detail-encoder.npz)
    is a mismatch; and a staging folder of a detail job that never finished is swept after an hour."""
    from sopa_companion.tests import fake_qc_trainer as F
    installed, staging = tmp_path / "model", tmp_path / "staging"
    job = {"zarr": "/x.zarr", "table": "xenium_table", "other_tables": ["table"], "model_id": "k" * 32,
           "recipe": T.recipe(), "sample": {"id": "s"}}
    F.write_model(installed, job, 3, deep=False)
    F.write_detail(staging, {**job, "detail_for": str(installed)})
    Q.QcModels._check_detail(staging, "k" * 32, installed)              # as the trainer wrote it: fine
    with pytest.raises(ValueError, match="does not describe"):
        Q.QcModels._check_detail(staging, "other", installed)
    np.savez(staging / ".detail-encoder.npz", **F.calculator(seed=1))   # another learning left beside it
    with pytest.raises(Q.DetailMismatch):
        Q.QcModels._check_detail(staging, "k" * 32, installed)
    np.savez(staging / ".detail-encoder.npz", **F.calculator(seed=0))   # the same calculator: fine
    Q.QcModels._check_detail(staging, "k" * 32, installed)
    data = json.loads((staging / "model.json").read_text())
    (staging / "model.json").write_text(json.dumps({**data, "detail": {**data["detail"], "accepted": False}}))
    with pytest.raises(Q.DetailMismatch):
        Q.QcModels._check_detail(staging, "k" * 32, installed)
    F.write_detail(staging, {**job, "detail_for": str(installed)}, bad_deep=True)
    (staging / ".detail-encoder.npz").unlink()
    with pytest.raises(ValueError, match="differ"):
        Q.QcModels._check_detail(staging, "k" * 32, installed)
    assert Q.calculators_agree(F.calculator(), F.calculator()) and not Q.calculators_agree(F.calculator(), F.calculator(1))
    # The sweep: an old detail staging folder goes, a fresh one (a job running now) stays.
    root = tmp_path / "var" / "qc-models" / "sid"
    old, fresh = root / f".{'k' * 32}.detail-1-2", root / f".{'k' * 32}.detail-3-4"
    for folder in (old, fresh):
        folder.mkdir(parents=True)
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    Q.QcModels(qc_config(tmp_path), Registry(), start=False)
    assert not old.exists() and fresh.exists()


def test_prelearning_with_detail_adds_the_details(tmp_path, fake_env, monkeypatch):
    """python -m sopa_companion.engine.qc learn --detail: a model learned before Go deeper gets its details (a detail
    job); a sample without a model is learned (new fits come with them); one line per sample names the details."""
    import socket
    monkeypatch.setattr(Q, "PRELEARN_POLL_S", 0.05)
    a, b = sample(tmp_path, "A"), sample(tmp_path, "B")
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    toml = tmp_path / "prelearn.toml"
    toml.write_text(f"""[server]\nport = {port}\n[paths]\ndata_dir = "{tmp_path / 'data'}"\nstate_dir = "{tmp_path / 'state'}"
[qc]\npython = "{sys.executable}"\ntrainer_module = "{FAKE}"\nprobe_module = "{FAKE_PROBE}"\ngpu_quiet_s = 0\n""")
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    assert Q.main(["learn", a["path"], "--config", str(toml)], probe=lambda: dict(IDLE)) == 0
    assert not list((tmp_path / "data" / "qc-models").glob(f"{a['id']}/*/deep.npz"))
    monkeypatch.setenv("FAKE_QC_MODE", "ok")
    assert Q.main(["learn", "--detail", a["path"], b["path"], "--config", str(toml)], probe=lambda: dict(IDLE)) == 0
    for s in (a, b):
        assert list((tmp_path / "data" / "qc-models").glob(f"{s['id']}/*/deep.npz")), s["name"]
    jobs = [launch["job"] for launch in fake_env()]
    assert [bool(j.get("detail_for")) for j in jobs] == [False, True, False]   # A's details, B learned with its own
    c = sample(tmp_path, "C")                                            # the details refused: exit 1
    monkeypatch.setenv("FAKE_QC_MODE", "nodeep")
    assert Q.main(["learn", c["path"], "--config", str(toml)], probe=lambda: dict(IDLE)) == 0
    monkeypatch.setenv("FAKE_QC_MODE", "detail_mismatch")
    assert Q.main(["learn", "--detail", c["path"], "--config", str(toml)], probe=lambda: dict(IDLE)) == 1


def test_no_notebook_number_is_read_or_served(tmp_path, fake_env, monkeypatch):
    """The sample's view once carried a "notebook number" read from a folder of notebook results
    (config notebook_dir) that the page never showed. It is gone: a config that still names such a folder learns as
    before, nothing in it is opened, and the view has no such field."""
    import pyarrow.parquet as pq
    notebook = tmp_path / "notebook" / "sample-a"
    notebook.mkdir(parents=True)
    (notebook / "Xenium_cells.parquet").write_bytes(b"not read")
    opened, real_read = [], pq.read_table
    monkeypatch.setattr(pq, "read_table", lambda f, *a, **k: (opened.append(str(f)), real_read(f, *a, **k))[1])
    s = sample(tmp_path, "sample-a-crop")
    m = models(tmp_path, s, notebook_dir=str(tmp_path / "notebook"))
    m.ensure(s["id"])
    view = settle(m, s["id"])
    assert view["state"] == "ready" and "notebook_start_per100" not in view
    assert not hasattr(m, "notebook_per100") and not hasattr(m, "notebook_for_model")
    assert not [f for f in opened if f.startswith(str(tmp_path / "notebook"))]
    m.close()


# ---- the trainer's numpy parts -------------------------------------------------------------------------------------
def tiny_encoder(rng, genes=6, hidden=5):
    enc = {"format": np.array(T.FORMAT), "genes": np.array([f"G{i}" for i in range(genes)]),
           "gene_ids": np.array([""] * genes), "W0": rng.normal(size=(hidden, genes)).astype(np.float32),
           "b0": rng.normal(size=hidden).astype(np.float32), "W1": rng.normal(size=(hidden, hidden)).astype(np.float32),
           "b1": rng.normal(size=hidden).astype(np.float32), "Wv": rng.normal(size=(3, hidden)).astype(np.float32),
           "bv": rng.normal(size=3).astype(np.float32), "ln_eps": np.float32(1e-5), "var_eps": np.float32(1e-3),
           "diffusion_eps": np.float32(0.01)}
    counts = rng.poisson(2.0, size=(9, genes)).astype(np.float32)
    rows, cols = np.nonzero(counts)
    enc.update(selftest_indptr=np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=9))]).astype(np.int64),
               selftest_indices=cols.astype(np.int32), selftest_data=counts[rows, cols])
    return enc, counts


def reference_weights(enc, counts):
    """ResolVI's guide written out plainly (torch.nn.LayerNorm without affine, biased variance, eps 1e-5)."""
    def layer(h, w, b):
        h = h @ w.T.astype(np.float64) + b
        h = (h - h.mean(1, keepdims=True)) / np.sqrt(h.var(1, keepdims=True) + 1e-5)
        return np.maximum(h, 0)
    h = layer(np.log1p(counts.astype(np.float64)), enc["W0"], enc["b0"])
    h = layer(h, enc["W1"], enc["b1"])
    z = h @ enc["Wv"].T.astype(np.float64) + enc["bv"]
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True) + 1e-3
    p[:, 1] += 0.01
    return p


def test_the_calculator_export_round_trip_and_self_test(tmp_path):
    rng = np.random.default_rng(3)
    enc, counts = tiny_encoder(rng)
    want = reference_weights(enc, counts)
    enc["selftest_expected"] = want.astype(np.float32)
    np.savez(tmp_path / "encoder.npz", **enc)
    loaded = T.load_encoder(tmp_path / "encoder.npz")
    assert loaded["genes"] == [f"G{i}" for i in range(6)] and T.self_test(loaded) < 1e-6
    np.testing.assert_allclose(T.selftest_counts(loaded), counts)
    weights = T.mixing_weights(loaded, counts)                                    # float32, dense
    assert weights.dtype == np.float32 and np.abs(weights - want).max() < 1e-5
    share = T.neighbour_share(want)
    np.testing.assert_allclose(share, want[:, 1] / want.sum(1))
    assert np.all(share > 0) and np.all(share < 1)
    bytes_ = T.quantise(np.array([0.0, 0.3, 0.299, 1.0, 0.5, np.nan]), np.array([True, True, True, True, False, True]))
    assert bytes_.tolist() == [0, 75, 74, 250, 255, 0] and bytes_.dtype == np.uint8   # floor: >= 75 is >= 0.30
    # The same codes as the GPU worker writes (one coding everywhere: colours and headline count the same cells).
    from sopa_companion.analysis import worker as W
    shares = np.random.default_rng(8).uniform(0, 1, 100_000).astype(np.float32)
    judged = np.random.default_rng(9).random(100_000) < 0.8
    assert np.array_equal(T.quantise(shares, judged), W.share_codes(np, shares, judged))
    # Anything unexpected is refused: other arrays, other shapes, another format, non-finite weights.
    for change in ({"extra": np.zeros(1)}, {"W1": enc["W1"][:, :4]}, {"format": np.array("other/1")},
                   {"b0": np.full(5, np.nan, np.float32)}, {"genes": np.array(["G0"] * 6)},
                   {"selftest_indices": np.full(len(enc["selftest_indices"]), 99, np.int32)}, {"W0": enc["W0"].astype(np.float64)}):
        np.savez(tmp_path / "bad.npz", **{**enc, **change})
        with pytest.raises(ValueError):
            T.load_encoder(tmp_path / "bad.npz")
    missing = dict(enc)
    missing.pop("Wv")
    np.savez(tmp_path / "bad.npz", **missing)
    with pytest.raises(ValueError, match="missing"):
        T.load_encoder(tmp_path / "bad.npz")


class Csr:
    """Just enough of a CSR matrix (scipy's in the trainer, cupyx's in the worker): stored values and a product."""

    def __init__(self, dense):
        rows, cols = np.nonzero(dense)
        self.shape, self.data, self.indices = dense.shape, dense[rows, cols], cols
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=dense.shape[0]))])

    def astype(self, dtype):                         # cupyx's astype takes no copy argument
        other = Csr.__new__(Csr)
        other.shape, other.indices, other.indptr, other.data = self.shape, self.indices, self.indptr, self.data.astype(dtype)
        return other

    def __matmul__(self, right):
        dense = np.zeros(self.shape, self.data.dtype)
        dense[np.repeat(np.arange(self.shape[0]), np.diff(self.indptr)), self.indices] = self.data
        return dense @ right


def test_the_calculator_reads_sparse_counts():
    rng = np.random.default_rng(4)
    enc, counts = tiny_encoder(rng)
    dense = T.mixing_weights(enc, counts, np, np.float64)
    csr = T.mixing_weights(enc, Csr(counts), np, np.float64)                      # log1p of stored values only
    np.testing.assert_allclose(csr, dense, atol=1e-12)
    np.testing.assert_allclose(dense, reference_weights(enc, counts), atol=1e-12)


def test_recipe_maths_as_the_notebook():
    rec = T.recipe()
    assert T.epochs_for(41190, rec) == 50 and T.epochs_for(31000, rec) == 50      # >= 3,000 steps at 50 passes
    assert T.epochs_for(3401, rec) == 429 and T.epochs_for(3401, rec) * 7 >= 3000  # small tables: more passes
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 500, (300, 2))
    xy[7] = xy[3]                                                                  # two cells on one centroid
    indices, d2 = T.knn_graph(xy, 10)                                              # scipy, or the numpy fallback
    assert indices.shape == (300, 10) and d2.dtype == np.float32 and indices.dtype == np.int64
    assert not (indices == np.arange(300)[:, None]).any()                          # never itself
    full = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(full, np.inf)
    np.testing.assert_allclose(np.sort(d2, 1), np.sort(full, 1)[:, :10], rtol=1e-5)
    assert (np.diff(d2, axis=1) >= 0).all() and 3 in indices[7] and d2[7, 0] == 0
    counts = rng.poisson(1.0, (300, 100)).astype(np.float32)
    priors = T.dataset_priors(counts, d2, rec)
    low = np.argsort(counts.sum(0))[:2]
    assert priors["background_ratio"] == pytest.approx(np.mean(counts[:, low].mean(1) / counts.mean(1)))
    assert priors["median_distance"] == pytest.approx(np.median(np.sort(d2, 1)[:, 5]))
    with pytest.raises(T.TrainerError) as few:
        T.knn_graph(xy[:5], 10)
    assert few.value.reason == "too_few_cells"


def test_the_neighbour_graph_does_not_depend_on_the_block_size():
    rng = np.random.default_rng(6)
    xy = rng.uniform(0, 300, (500, 2))
    whole, blocked = T.knn_graph(xy, 10, block=4096), T.knn_graph(xy, 10, block=64)
    assert np.array_equal(whole[0], blocked[0]) and np.array_equal(whole[1], blocked[1])


def test_a_fit_must_settle():
    assert T.loss_settled([10, 8, 7, 6, 5.5, 5.2, 5.1, 5.0, 4.95, 4.9, 4.9, 4.88])[0]
    settled_noise = list(np.linspace(30e6, 17e6, 540)) + list(17e6 * (1 + 0.0008 * np.sin(np.arange(10))))
    assert T.loss_settled(settled_noise)[0]                                       # the crop's 550-pass curve: +0.08 %
    assert not T.loss_settled([10, 8, 7, float("nan"), 5])[0]
    assert not T.loss_settled([10, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5.2, 5.3, 5.3, 5.3, 5.3])[0]   # rising at the end
    assert not T.loss_settled([5, 6])[0] and not T.loss_settled([])[0]
    assert T.loss_settled([5, 4])[0]


def test_reference_cells_cover_depth_and_mixing():
    rng = np.random.default_rng(7)
    totals = rng.integers(20, 2000, 5000)
    share = rng.uniform(0, 0.8, 5000)
    chosen = T._reference_cells(totals, share)
    assert len(chosen) == 256 == len(np.unique(chosen))
    assert totals[chosen].min() == totals.min() and totals[chosen].max() == totals.max()
    assert share[chosen].max() == share.max()
    assert T._reference_cells(totals[:100], share[:100]).tolist() == list(range(100))


def test_reading_waits_for_a_save_and_leaves_no_trace(tmp_path, monkeypatch):
    locks = tmp_path / "locks"
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(locks))
    from sopa_companion.zarr_bridge.helper import _lock_paths, _store_lock
    store = tmp_path / "s.zarr"
    store.mkdir()
    # Nothing was ever saved into it: one read, and no lock file is created (the harness guard sees none).
    reads = []
    assert T.read_consistently(store, lambda: reads.append(1) or "table") == "table" and reads == [1]
    assert not locks.exists() or not list(locks.iterdir())
    # A save that begins during a lock-free read: the read is done again, after the save.
    events, saving, finished = [], threading.Event(), threading.Event()

    def save(hold=0.4):
        with _store_lock(store):
            events.append("save start")
            saving.set()
            time.sleep(hold)
            events.append("save end")
        finished.set()

    def read():
        events.append("read")
        if not saving.is_set():
            threading.Thread(target=save).start()
            saving.wait(5)
        return len(events)
    T.read_consistently(store, read)
    assert events == ["read", "save start", "save end", "read"]       # never a torn read
    finished.wait(5)
    [target] = _lock_paths(store.resolve(), create=False)
    # The sample has a lock file now: reads wait for a save holding it, and readers do not wait for each other.
    events.clear()
    saving.clear()
    thread = threading.Thread(target=save)
    thread.start()
    saving.wait(5)
    T.read_consistently(store, lambda: events.append("read"))
    thread.join()
    assert events == ["save start", "save end", "read"]
    stamp = target.stat().st_mtime_ns
    T.read_consistently(store, lambda: None)
    assert target.stat().st_mtime_ns == stamp                         # reading never touches the lock file
    done = []

    def reader():
        with T.shared_zarr_lock(store):
            done.append(1)
    with T.shared_zarr_lock(store):
        other = threading.Thread(target=reader)
        other.start()
        other.join(2)
        assert done == [1]


def test_reading_waits_for_a_save_of_the_users_app_in_either_lock_folder(tmp_path, monkeypatch):
    """The app reads under the locks its saves take (zarr_bridge/helper.py _lock_paths): the user's two shared folders,
    the previous version's first (its saves take that one alone), and creates nothing there. With a private folder
    (SOPA_COMPANION_ZARR_LOCK_DIR: a test's or the harness's engine) that folder alone: the user's lock files are then
    neither waited on nor opened."""
    import fcntl
    import hashlib
    home, private = tmp_path / "home", tmp_path / "private"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(private))
    store = tmp_path / "s.zarr"
    store.mkdir()
    name = hashlib.sha1(str(store.resolve()).encode()).hexdigest() + ".lock"
    assert T.store_lock_paths(store) == [private / name]
    monkeypatch.delenv("SOPA_COMPANION_ZARR_LOCK_DIR")
    old, new = home / ".cache/proseg_live/zarr-locks" / name, home / ".cache/sopa-companion/zarr-locks" / name
    assert T.store_lock_paths(store) == [old, new] and not (home / ".cache").exists()
    old.parent.mkdir(parents=True)
    target = old                                         # an engine of the previous version: its folder's lock only
    events, saving = [], threading.Event()

    def users_save(hold=0.4):                            # the user's engine: its save lock, held exclusively
        with open(target, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            events.append("save start")
            saving.set()
            time.sleep(hold)
            events.append("save end")
    thread = threading.Thread(target=users_save)
    thread.start()
    saving.wait(5)
    T.read_consistently(store, lambda: events.append("read"))
    thread.join()
    assert events == ["save start", "save end", "read"] and not private.exists() and not new.exists()
    # A first save that begins in the shared folder during a read: the read is done again.
    target.unlink()
    events.clear()
    saving.clear()

    def read():
        events.append("read")
        if not saving.is_set():
            threading.Thread(target=users_save).start()
            saving.wait(5)
        return len(events)
    T.read_consistently(store, read)
    assert events == ["read", "save start", "save end", "read"]
    # A private folder: that folder alone, so a held lock of the user's is not waited for.
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(private))
    events.clear()
    saving.clear()
    thread = threading.Thread(target=users_save)
    thread.start()
    saving.wait(5)
    T.read_consistently(store, lambda: events.append("read"))
    thread.join()
    assert events == ["save start", "read", "save end"] and not private.exists()


# ---- QcManager: the headline over worker results --------------------------------------------------------------------
def test_the_shown_number_moves_only_past_a_boundary():
    shown = None
    seen = []
    for value in (8.4, 8.6, 8.79, 8.81, 8.3, 8.21, 8.19, 7.6, 12.3, None):
        shown = Q.shown_integer(value, shown)
        seen.append(shown)
    assert seen == [8, 8, 8, 9, 9, 9, 8, 8, 12, 12]
    assert Q.shown_integer(8.79, 8, margin=0.2) == 9 and Q.shown_integer(8.79, 8) == 8     # the mode's own margin


def test_the_headline_weighs_results_by_the_run_clock(monkeypatch):
    """The mode's weight w per second of the run clock (the smoothing for the shown integer), w per result without
    times: each mode's own (engine/qc_constants.py headline_weight, headline_ref_s), a region's by default. On proseg's
    count a region's w is 0.2 and a tiled region's 0.1 (measured: its results move more from one to the next than the
    outline count's did at 0.5); a whole sample keeps 0.5 until it is measured."""
    assert Q.headline_weight(None, 3.0) == Q.headline_weight(3.0, None) == 0.2
    assert Q.headline_weight(2.0, 1.0) == pytest.approx(0.2)
    assert Q.headline_weight(1.1, 1.0) == pytest.approx(1 - 0.8 ** 0.1)            # a crop's results, 0.1 s apart
    assert Q.headline_weight(1.0, 1.0) == Q.headline_weight(0.5, 1.0) == 0.2       # no time passed, or a clock gone back
    assert Q.headline_weight(61.0, 1.0) > 0.99999                                   # after a pause: almost all new
    assert Q.headline_weight(3.0, 1.0, "tiled") == pytest.approx(1 - 0.9 ** 2)
    assert Q.headline_weight(3.0, 1.0, "whole") == pytest.approx(1 - 0.5 ** 2)
    assert Q._seconds(True) is None and Q._seconds(float("nan")) is None and Q._seconds("1") is None
    assert Q._seconds(2) == 2.0
    from sopa_companion.engine import qc_constants as K
    monkeypatch.setitem(K.TABLE["whole"], "headline_weight", 0.25)
    monkeypatch.setitem(K.TABLE["whole"], "headline_ref_s", 2.0)
    assert Q.headline_weight(None, 3.0, "whole") == 0.25
    assert Q.headline_weight(3.0, 1.0, "whole") == pytest.approx(0.25) and Q.headline_weight(3.0, 1.0) == pytest.approx(0.36)


def test_thinning_keeps_the_last_point():
    items = [{"seq": i, "per100": float(i)} for i in range(100)]
    thin = Q._thin(items, 10)
    assert len(thin) == 10 and thin[0]["seq"] == 0 and thin[-1]["seq"] == 99
    assert Q._thin(items[:3], 10) == [{**{k: None for k in ("iter", "shown", "raw_per100", "judged", "mixed",
                                                             "median_tx", "single", "matched_change", "count")}, **i}
                                      for i in items[:3]]


def test_point_in_polygon_for_lasso_regions():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    x, y = np.array([5.0, 15, -1, 9.9, 5]), np.array([5.0, 5, 5, 9.9, 11])
    assert Q._point_in_polygon(x, y, square).tolist() == [True, False, False, True, False]


# ---- the real trainer process (sopa environment, CPU only: everything before the GPU) -----------------------------
WRITE_TABLES = """
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, zarr
data, root = np.load(sys.argv[1]), sys.argv[2]
zarr.open_group(root, mode="w")
genes = [f"G{i}" for i in range(data["X"].shape[1])]
ids = [f"ENSG{i:05d}" for i in range(data["X"].shape[1])]
# xenium_table; "halves" (not whole counts); "table": other cells, genes in reverse order; "saved": the same cells as a
# save of a proseg run leaves them (its uns holds proseg's run record and the app's): the earlier proseg cells.
for key, X, xy, order in (("xenium_table", data["X"], data["xy"], slice(None)), ("halves", data["X"] * 0.5, data["xy"],
                          slice(None)), ("table", data["X2"], data["xy2"], slice(None, None, -1)),
                          ("saved", data["X2"], data["xy2"], slice(None, None, -1))):
    obs = pd.DataFrame({"cell_id": [f"{key}{i}" for i in range(len(X))], "cell_area": np.full(len(X), 80.0)},
                       index=[str(i) for i in range(len(X))])
    var = pd.DataFrame({"gene_ids": ids[order]}, index=genes[order])
    table = ad.AnnData(sp.csr_matrix(X[:, order].astype(np.float32)), obs=obs, var=var)
    table.obsm["spatial"] = xy
    if key == "saved":
        table.uns["proseg_run"] = {"iterations": 3}
        table.uns["sopa_companion"] = {"run": "r1"}
    table.write_zarr(f"{root}/tables/{key}")
"""


@pytest.fixture(scope="module")
def synthetic_table(tmp_path_factory):
    import subprocess
    root = tmp_path_factory.mktemp("qc-table")
    rng = np.random.default_rng(11)
    n, genes = 400, 60
    X = rng.poisson(rng.uniform(0.2, 3.0, genes), size=(n, genes)).astype(np.float32)
    X[:15] = 0
    X[:15, 0] = 5                                                # 15 cells under 20 counts: not learned from
    X[:, 59] = 0                                                 # a gene the Xenium cells never hold ...
    xy = rng.uniform(0, 300, (n, 2))
    X2 = rng.poisson(rng.uniform(0.2, 3.0, genes), size=(300, genes)).astype(np.float32)
    X2[:9] = 0
    X2[:9, 3] = 4                                                # 9 earlier proseg cells under 20 counts
    X2[:, 58] = 0                                                # ... and one only the earlier proseg cells lack
    xy2 = rng.uniform(0, 300, (300, 2))
    np.savez(root / "data.npz", X=X, xy=xy, area=rng.uniform(20, 200, n), X2=X2, xy2=xy2)
    subprocess.run([sopa_python(), "-c", WRITE_TABLES, str(root / "data.npz"), str(root / "s.zarr")], check=True,
                   capture_output=True, timeout=300)
    return root / "s.zarr", X, xy, X2, xy2


def notebook_priors(tables, rec):
    """The pairing recipe (resolvi_prepare_pair), line by line on dense arrays: tables = [(counts of the kept
    cells over all genes in one shared order, their squared neighbour distances)] -> (priors, genes kept)."""
    pooled_totals = sum(np.asarray(X.sum(axis=0)).ravel() for X, _ in tables)
    pair = [(X[:, pooled_totals > 0], d2) for X, d2 in tables]
    n_genes = pair[0][0].shape[1]
    low_genes = np.argsort(pooled_totals[pooled_totals > 0])[:n_genes // rec["low_genes_per"]]
    background_ratio = np.mean([np.mean(np.asarray(X[:, low_genes].mean(axis=1)).ravel() /
                                        np.asarray(X.mean(axis=1)).ravel()) for X, _ in pair])
    median_distance = np.median(np.concatenate([np.partition(d2, 5, axis=1)[:, 5] for _, d2 in pair]))
    return {"background_ratio": float(background_ratio), "median_distance": float(median_distance)}, n_genes


def installed_version(package: str) -> str:
    """`package`'s version in the trainer's environment, as that environment itself reads it."""
    import subprocess
    done = subprocess.run([sopa_python(), "-I", "-c", f"from importlib.metadata import version; print(version({package!r}))"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def run_trainer(job: dict, env: dict, on_line=None, timeout: float = 300):
    """The real trainer as the engine starts it; -> its JSON lines."""
    import subprocess
    process = subprocess.Popen([sopa_python(), "-m", "sopa_companion.analysis.qc_trainer"], cwd=str(ROOT.parent),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               env={**os.environ, "PYTHONPATH": str(ROOT.parent), "OMP_NUM_THREADS": "4", **env})
    process.stdin.write(json.dumps(job) + "\n")
    process.stdin.flush()
    lines = []
    deadline = time.monotonic() + timeout
    for line in process.stdout:
        lines.append(json.loads(line))
        if on_line:
            on_line(lines[-1])
        if lines[-1]["ev"] in ("done", "error") or time.monotonic() > deadline:
            break
    process.stdin.close()
    process.wait(30)
    return lines


def test_the_trainer_reads_under_the_lock_and_prepares_as_the_notebook(synthetic_table, tmp_path, monkeypatch):
    from sopa_companion.zarr_bridge.helper import _store_lock
    path, X, xy, _, _ = synthetic_table
    locks = tmp_path / "locks"
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(locks))
    job = {"zarr": str(path), "table": "xenium_table", "other_tables": ["halves"], "out_dir": str(tmp_path / "out"),
           "recipe": T.recipe(), "prepare_only": True}
    released = []
    saving = _store_lock(path.resolve())
    saving.__enter__()                                            # a save into the sample is under way

    def on_line(message):
        if message.get("stage") == "lock" and message.get("note") and not released:
            released.append(time.monotonic())
            saving.__exit__(None, None, None)                     # the save ends: now it may read
    lines = run_trainer(job, {"CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(locks)}, on_line)
    if not released:
        saving.__exit__(None, None, None)
    hello, *progress, done = lines
    assert hello["ev"] == "hello" and hello["nice"] == 19
    assert hello["versions"]["scviva-tools"] == installed_version("scviva-tools")
    stages = [p["stage"] for p in progress]
    assert released and stages[:3] == ["lock", "lock", "reading"] and "waiting for a save" in progress[1]["note"]
    assert done["ev"] == "done", done
    model = done["model"]
    keep = X.sum(1) >= 20
    assert model["cells"] == int(keep.sum()) == 385 and model["genes"] == 59 and model["input_cells"] == 400
    assert model["epochs"] == T.epochs_for(385, T.recipe()) and model["steps"] == model["epochs"]
    indices, d2 = T.knn_graph(xy[keep], 10)                        # the same graph as the trainer's cKDTree
    assert model["knn"]["first"] == indices[0].tolist()
    np.testing.assert_allclose(model["knn"]["d2"], d2[0], rtol=1e-6)
    # The other table ("halves") holds no whole counts: the priors come from the learned table alone, and say why.
    expected = T.dataset_priors(X[keep][:, :59], d2, T.recipe())
    assert model["priors"]["background_ratio"] == pytest.approx(expected["background_ratio"], rel=1e-6)
    assert model["priors"]["median_distance"] == pytest.approx(expected["median_distance"], rel=1e-6)
    assert model["priors_from"]["rule"] == "table" and model["priors_from"]["tables"] == ["xenium_table"]
    assert model["priors_from"]["cells"] == [385] and "could not be used" in model["priors_from"]["why"]


def test_the_trainer_pools_genes_and_priors_with_the_other_table_as_the_notebook(synthetic_table, tmp_path):
    """The priors (and kept genes) are pooled over the Xenium cells and the earlier proseg
    cells; the other table's genes are matched by name (here in reverse order) and its graph is its own."""
    path, X, xy, X2, xy2 = synthetic_table
    job = {"zarr": str(path), "table": "xenium_table", "other_tables": ["table"], "out_dir": str(tmp_path / "out"),
           "recipe": T.recipe(), "prepare_only": True}
    lines = run_trainer(job, {"CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks")})
    assert lines[-1]["ev"] == "done", lines[-1]
    model = lines[-1]["model"]
    keep, keep2 = X.sum(1) >= 20, X2.sum(1) >= 20
    _, d2 = T.knn_graph(xy[keep], 10)
    _, d2b = T.knn_graph(xy2[keep2], 10)
    expected, genes = notebook_priors([(X[keep], d2), (X2[keep2], d2b)], T.recipe())
    assert genes == 60 and model["genes"] == 60                     # gene 59 kept: only the earlier cells hold it
    assert model["cells"] == 385 and model["priors_from"] == {"rule": "pooled", "tables": ["xenium_table", "table"],
                                                              "cells": [385, int(keep2.sum())]}
    assert model["priors"]["background_ratio"] == pytest.approx(expected["background_ratio"], rel=1e-6)
    assert model["priors"]["median_distance"] == pytest.approx(expected["median_distance"], rel=1e-6)
    alone = T.dataset_priors(X[keep][:, :59], d2, T.recipe())
    assert model["priors"]["median_distance"] != pytest.approx(alone["median_distance"], rel=1e-6)   # really pooled
    # The recipe's "table" priors keep the learned table alone.
    lines = run_trainer({**job, "recipe": T.recipe(priors="table")},
                        {"CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks")})
    model = lines[-1]["model"]
    assert model["genes"] == 59 and model["priors_from"]["rule"] == "table"
    assert model["priors"]["median_distance"] == pytest.approx(alone["median_distance"], rel=1e-6)


def test_only_a_table_of_prosegs_own_count_is_scored_as_the_earlier_proseg_cells(synthetic_table, tmp_path):
    """The other table is pooled into the priors whatever it is, but scored into batch.parquet (the "earlier proseg
    cells" line) only when it holds proseg's own count: a save of a proseg run (its uns: proseg's run record). A
    Xenium sample's "table", Xenium Ranger's outline re-count, never is."""
    path, _, _, _, _ = synthetic_table
    env = {"CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks")}
    assert T.proseg_table(path, "saved") and not T.proseg_table(path, "table")
    for other, scored in (("table", None), ("saved", "saved")):
        job = {"zarr": str(path), "table": "xenium_table", "other_tables": [other], "out_dir": str(tmp_path / other),
               "recipe": T.recipe(), "prepare_only": True}
        lines = run_trainer(job, env)
        assert lines[-1]["ev"] == "done", lines[-1]
        model = lines[-1]["model"]
        assert model["scored"] == scored and model["priors_from"]["tables"] == ["xenium_table", other], other


def test_the_proseg_table_rule_reads_folders_only(tmp_path):
    """qc_trainer.proseg_table: tables/<key>/uns holds proseg_run, or the app's own save record (this version's or the
    previous one's); nothing else, and no key that is not a table's name."""
    root = tmp_path / "s.zarr"
    for key, record in (("a", "proseg_run"), ("b", "sopa_companion"), ("c", "proseg_live"), ("d", "spatialdata_attrs")):
        (root / "tables" / key / "uns" / record).mkdir(parents=True)
    assert [T.proseg_table(root, k) for k in "abcd"] == [True, True, True, False]
    assert T.PROSEG_TABLE_KEYS is helper.PROSEG_TABLE_KEYS == ("proseg_run", "sopa_companion", "proseg_live")
    (root / "tables" / "e" / "uns").mkdir(parents=True)
    (root / "tables" / "e" / "uns" / "proseg_run").write_text("not a folder")
    assert not T.proseg_table(root, "e") and not T.proseg_table(root, "missing")
    assert not any(T.proseg_table(root, k) for k in (None, "", "../a", "a/../a", 3))
    assert not T.proseg_table(None, "a") and not T.proseg_table("", "a")


def test_pooled_priors_are_the_notebooks():
    """pooled_priors and the gene order matching against cell 16 of the Part 3 notebook (dense, two tables)."""
    rng = np.random.default_rng(12)
    rec = T.recipe()
    tables = []
    for n in (500, 350):
        counts = rng.poisson(rng.uniform(0.05, 4.0, 200), size=(n, 200)).astype(np.float32)
        counts[:, rng.choice(200, 6, replace=False)] = 0
        _, d2 = T.knn_graph(rng.uniform(0, 400, (n, 2)), 10)
        tables.append((counts, d2))
    tables[0][0][:, 7] = tables[1][0][:, 7] = 0                     # a gene neither table holds: dropped
    expected, genes = notebook_priors(tables, rec)
    pooled = sum(c.sum(axis=0) for c, _ in tables)
    kept = pooled > 0
    got = T.pooled_priors([(c[:, kept], d2) for c, d2 in tables], pooled[kept], rec)
    assert got == expected and genes == int(kept.sum()) < 200
    # One table alone is the same rule with one part.
    single, _ = notebook_priors(tables[:1], rec)
    alone = tables[0][0][:, tables[0][0].sum(axis=0) > 0]
    assert T.dataset_priors(alone, tables[0][1], rec) == single
    # The other table's genes are matched by name, else by id; other panels are not pooled.
    names, ids = np.array(["A", "B", "C"]), np.array(["i1", "i2", "i3"])
    assert T.gene_order(names, ids, names[::-1], ids[::-1]).tolist() == [2, 1, 0]
    assert T.gene_order(names, ids, np.array(["x", "y", "z"]), ids[[1, 2, 0]]).tolist() == [2, 0, 1]
    assert T.gene_order(names, ids, np.array(["A", "B", "D"]), np.array(["i1", "i2", "i9"])) is None
    assert T.gene_order(names, ids, names[:2], ids[:2]) is None
    assert T.priors_partner(None, names, ids, rec) == (None, {"table": None, "why": "the sample has no other cell table"})


@pytest.mark.parametrize("table, env, reason", [("halves", {}, "not_counts"), ("missing", {}, "no_table"),
                                                ("xenium_table", {"CUDA_VISIBLE_DEVICES": "1"}, "gpu"),
                                                ("xenium_table", {"CUDA_VISIBLE_DEVICES": "0,1"}, "gpu"),
                                                # "0" in CUDA's own order may be GPU 1 of nvidia-smi's (and the worker's)
                                                ("xenium_table", {"CUDA_VISIBLE_DEVICES": "0",
                                                                  "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, "gpu")])
def test_the_trainer_refuses_what_it_cannot_learn_from(synthetic_table, tmp_path, table, env, reason):
    path, _, _, _, _ = synthetic_table
    job = {"zarr": str(path), "table": table, "out_dir": str(tmp_path / "out"), "recipe": T.recipe()}
    lines = run_trainer(job, {"CUDA_VISIBLE_DEVICES": "", "SOPA_COMPANION_ZARR_LOCK_DIR": str(tmp_path / "locks"), **env})
    assert lines[-1]["ev"] == "error" and lines[-1]["reason"] == reason, lines[-1]
    assert not (tmp_path / "locks").exists() or not list((tmp_path / "locks").iterdir())   # reading left no lock file
