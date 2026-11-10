"""The mixing check's trainer for Go deeper: deep.npz, its strict
loader and self-test, the numpy recipe of one cell, the detail job and its parity rule, and model.json's "deep" and
"provenance" blocks.

numpy only (the app's .venv): the stored tiny model (tests/data/qc_deep_model, torch's own self-test outputs), the fake
trainer's synthetic export, an independent re-implementation of the recipe. The real trainer process (the sopa
environment, CPU only: the tests-only "cpu_for_tests" mode, where no GPU is visible at all): a fit, one whose other
table is not proseg's own count (pooled, never scored), detail jobs and the torch-side refusals. The opt-in GPU smoke
is tests/test_qc_trainer_smoke.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from conftest import sopa_python
from sopa_companion.analysis import qc_trainer as T
from sopa_companion.tests import fake_qc_trainer as F

HERE = Path(__file__).resolve().parent
APP_PARENT = HERE.parents[1]
STORED = HERE / "data" / "qc_deep_model"                # tests/data/make_qc_deep_model.py
MAKER = HERE / "data" / "make_qc_deep_model.py"
PHASE_A_KEYS = ["format", "genes", "Z0", "zb0", "Z1", "zb1", "Zm", "zbm", "Zv", "zbv", "D0", "db0", "D1", "db1", "Ds",
                "dbs", "ln_eps", "z_var_eps", "background", "typical", "median_distance", "clamp", "diffuse_scale",
                "train_logit", "train_index", "rank_logit", "train_ids_sha256", "st_rows", "st_indptr", "st_indices",
                "st_data", "st_d2", "st_z", "st_pi", "st_own_rows", "st_own", "st_genes", "st_nb_split"]


# ---- helpers -----------------------------------------------------------------------------------------------------------
def synthetic(tmp_path, n=50, seed=0, **kwargs):
    """The fake trainer's calculator and deep.npz arrays for n learning cells -> (enc as load_encoder gives it, deep
    arrays, ids, xy, the encoder.npz path)."""
    enc = F.calculator(seed)
    np.savez(tmp_path / "encoder.npz", **enc)
    columns = F.cell_columns(n)
    xy = np.stack([columns["x"], columns["y"]], 1)
    arrays = F.synthetic_deep(enc, columns["cell_id"], xy, seed=seed, **kwargs)
    return T.load_encoder(tmp_path / "encoder.npz"), arrays, columns["cell_id"], xy, tmp_path / "encoder.npz"


def save(path, arrays) -> Path:
    np.savez(path, **arrays)
    return Path(path)


# ---- names, formats, the stored real model -----------------------------------------------------------------------------
def test_the_export_names_are_phase_as():
    """The worker reads deep.npz by these names (built in parallel with it): they never change."""
    assert T.DEEP_FORMAT == "proseg-live-mixing-deep/1" and T.DEEP_SELFTEST_TOLERANCE == 1e-5 == T.DEEP_TOLERANCE
    assert T.DEEP_WORKER_TOLERANCE == 1e-4 and T.DETAIL_TOLERANCE == 1e-6
    assert T.DEEP_KEYS == set(PHASE_A_KEYS) and len(PHASE_A_KEYS) == 38
    assert set(T.Z_LAYERS.values()) | set(T.D_LAYERS.values()) <= T.DEEP_KEYS
    assert T.FORMAT == "proseg-live-mixing-encoder/1" and T.RECIPE_VERSION == 2     # the calculator and the key: unchanged


def test_the_stored_model_reproduces_torch():
    """A real fit (the app's trainer, CPU) whose deep.npz holds torch's outputs with the noise off: the numpy recipe
    matches them within the trainer's tolerance in float64 and the worker's in float32, the details belong to the
    calculator beside them, and model.json describes the file."""
    enc = T.load_encoder(STORED / "encoder.npz")
    deep = T.load_deep(STORED / "deep.npz", enc["genes"])
    assert deep["enc"]["genes"] == enc["genes"] == deep["genes"] and len(deep["genes"]) == 60   # the calculator beside it
    parts = T.deep_self_test_parts(deep)
    assert parts["max"] <= T.DEEP_SELFTEST_TOLERANCE and parts["nb_split"] is not None
    assert set(parts) == {"z", "pi", "own", "nb_split", "max"} and parts["max"] == max(parts["z"], parts["pi"],
                                                                                         parts["own"], parts["nb_split"])
    assert T.deep_self_test(deep, dtype=np.float32) <= T.DEEP_WORKER_TOLERANCE
    assert T.deep_self_test(deep) == parts["max"]
    import pyarrow.parquet as pq
    ids = pq.read_table(STORED / "cells.parquet").column("cell_id").to_pylist()
    assert str(deep["train_ids_sha256"]) == T.ids_sha256(ids) and deep["train_logit"].shape == (len(ids), 10)
    model = json.loads((STORED / "model.json").read_text())
    assert model["deep"]["bytes"] == (STORED / "deep.npz").stat().st_size and model["deep"]["from"] == "fit"
    assert model["deep"]["selftest"]["max_abs_diff"] == pytest.approx(parts["max"], abs=1e-9)
    assert model["deep"]["selftest"]["cells"] == len(deep["st_rows"]) == 16
    assert model["provenance"] == T.provenance(model) and model["provenance"]["differences"] == ["other_epochs"]
    assert model["cells"]["mixed"] > 0 and (np.abs(deep["train_logit"]) > 1).any()     # a model with something to say


def test_ids_sha256_is_the_ids_joined_by_newlines():
    assert T.ids_sha256(["a", "b-1", 3]) == hashlib.sha256(b"a\nb-1\n3").hexdigest()
    assert T.ids_sha256([]) == hashlib.sha256(b"").hexdigest()


# ---- the strict loader ---------------------------------------------------------------------------------------------------
def test_a_synthetic_export_round_trips(tmp_path):
    enc, arrays, ids, _, enc_path = synthetic(tmp_path)
    path = save(tmp_path / "deep.npz", arrays)
    attached = T.load_deep(path, enc["genes"])                      # encoder.npz beside it: attached
    assert attached["genes"] == enc["genes"] and attached["enc"]["genes"] == enc["genes"]
    assert T.deep_self_test(attached) <= T.DEEP_SELFTEST_TOLERANCE
    assert T.deep_self_test(attached, dtype=np.float32) <= T.DEEP_WORKER_TOLERANCE
    given = T.load_deep(path, enc)                                  # the calculator itself as the second argument
    assert given["enc"] is enc and T.deep_self_test(given) <= T.DEEP_SELFTEST_TOLERANCE
    enc_path.unlink()                                               # alone (a detail job's folder): no calculator
    alone = T.load_deep(path)
    assert "enc" not in alone
    parts = T.deep_self_test_parts(alone)
    assert parts["nb_split"] is None and parts["max"] == max(parts["z"], parts["pi"], parts["own"]) <= 1e-5
    assert T.deep_self_test(alone, enc) == T.deep_self_test(given)  # the calculator passed in later
    assert alone["train_ids_sha256"].item() == T.ids_sha256(ids)


def _covariate(arrays):
    arrays["Z0"] = np.concatenate([arrays["Z0"], np.zeros((arrays["Z0"].shape[0], 1), np.float32)], axis=1)


def _decoder_covariate(arrays):
    arrays["D0"] = np.concatenate([arrays["D0"], np.zeros((arrays["D0"].shape[0], 1), np.float32)], axis=1)


@pytest.mark.parametrize("damage, words", [
    (lambda a: a.update(extra=np.zeros(3, np.float32)), "unknown \\['extra'\\]"),
    (lambda a: a.pop("Zv"), "missing \\['Zv'\\]"),
    (lambda a: a.update(format=np.array("proseg-live-mixing-deep/2")), "unknown deep format"),
    (lambda a: a.update(genes=a["genes"][::-1].copy()), "differ from encoder.npz genes"),
    (_covariate, "deep array Z0"),
    (_decoder_covariate, "deep array D0"),
    (lambda a: a.update(Ds=a["Ds"].astype(np.float64)), "deep array Ds: float64"),
    (lambda a: a.update(zb1=np.where(np.arange(len(a["zb1"])) == 0, np.nan, a["zb1"]).astype(np.float32)),
     "deep array zb1"),
    (lambda a: a.update(train_logit=np.where(np.arange(a["train_logit"].size).reshape(a["train_logit"].shape) == 3,
                                             10.5, a["train_logit"]).astype(np.float32)), "outside \\(-10, 10\\)"),
    (lambda a: a.update(rank_logit=(a["rank_logit"] + 1e-3).astype(np.float32)), "column mean"),
    (lambda a: a.update(train_index=np.where(a["train_index"] == a["train_index"].max(), 10_000,
                                             a["train_index"]).astype(np.int32)), "out of range"),
    (lambda a: a.update(train_index=np.concatenate([[[0] * a["train_index"].shape[1]], a["train_index"][1:]]).astype(
        np.int32)), "its own neighbour"),
    (lambda a: a.update(train_index=a["train_index"].astype(np.int64)), "train_index: int64"),
    (lambda a: a.update(train_ids_sha256=np.array("abc")), "not a sha256"),
    (lambda a: a.update(st_indptr=a["st_indptr"][:-1]), "malformed deep self-test counts"),
    (lambda a: a.update(st_indices=np.where(a["st_indices"] == 0, 99, a["st_indices"]).astype(np.int32)),
     "malformed deep self-test counts"),
    (lambda a: a.update(background=(a["background"] * 1.1).astype(np.float32)), "background is not a profile"),
    (lambda a: a.update(st_own_rows=np.array([0, 99], np.int32), st_own=np.zeros((2, a["st_own"].shape[1]),
                                                                                 np.float32)), "out of range"),
    (lambda a: a.update(median_distance=np.float32(0.0)), "scalars out of range"),
])
def test_the_loader_refuses_what_it_cannot_trust(tmp_path, damage, words):
    enc, arrays, *_ = synthetic(tmp_path)
    damage(arrays)
    with pytest.raises(ValueError, match=words):
        T.load_deep(save(tmp_path / "deep.npz", arrays), enc["genes"])


def test_the_loader_refuses_a_calculator_of_other_genes(tmp_path):
    enc, arrays, *_ = synthetic(tmp_path)
    path = save(tmp_path / "deep.npz", arrays)
    other = F.calculator(1, genes=[f"H{i}" for i in range(7)])
    np.savez(tmp_path / "encoder.npz", **other)                    # the encoder.npz beside it is another model's
    with pytest.raises(ValueError, match="differ from encoder.npz genes"):
        T.load_deep(path)
    with pytest.raises(ValueError, match="differ from encoder.npz genes"):
        T.load_deep(path, enc["genes"][::-1])


def test_rank_logit_must_be_averaged_in_float64(tmp_path):
    """A float32 axis-0 mean over tens of thousands of learned values is off by more than the loader
    accepts; the export averages them in float64."""
    rng = np.random.default_rng(3)
    logit = (rng.uniform(4.0, 10.0, (40_000, 10)) * rng.choice([-1, 1], 10)).astype(np.float32)   # whole-sample sizes
    exact = logit.mean(axis=0, dtype=np.float64)
    sloppy = logit.mean(axis=0)                                     # float32 accumulation
    assert np.abs(sloppy - exact).max() > 1e-5 >= np.abs(exact.astype(np.float32) - exact).max()
    enc, arrays, *_ = synthetic(tmp_path)
    ring = ((np.arange(len(logit))[:, None] + np.arange(1, 11)[None]) % len(logit)).astype(np.int32)
    big = dict(arrays, train_logit=logit, train_index=ring)          # 40,000 learning cells (the loader's checks only)
    with pytest.raises(ValueError, match="column mean"):
        T.load_deep(save(tmp_path / "sloppy.npz", dict(big, rank_logit=sloppy)), enc["genes"])
    T.load_deep(save(tmp_path / "exact.npz", dict(big, rank_logit=exact.astype(np.float32))), enc["genes"])


# ---- the self-test ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key, part", [("st_z", "z"), ("st_pi", "pi"), ("st_own", "own"), ("st_nb_split", "nb_split"),
                                       ("Zm", "z"), ("D1", "own"), ("background", "nb_split")])
def test_the_self_test_catches_a_damaged_export(tmp_path, key, part):
    enc, arrays, *_ = synthetic(tmp_path)
    good = T.deep_self_test_parts(T.load_deep(save(tmp_path / "good.npz", arrays), enc))
    if key == "background":                                         # still a profile: shifted between two genes
        bg = arrays["background"].astype(np.float64)
        moved = bg[0] * 0.9
        bg[0] -= moved
        bg[1] += moved
        arrays[key] = bg.astype(np.float32)
    else:
        arrays[key] = (arrays[key] + np.float32(0.05) * np.sign(arrays[key] + 1e-6)).astype(np.float32)
    parts = T.deep_self_test_parts(T.load_deep(save(tmp_path / "bad.npz", arrays), enc))
    assert good["max"] <= T.DEEP_SELFTEST_TOLERANCE < parts[part] and parts["max"] > T.DEEP_SELFTEST_TOLERANCE


def test_deep_selftest_rows_spread_over_mixing_and_depth():
    rng = np.random.default_rng(0)
    totals = rng.integers(20, 3000, 1000).astype(np.float64)
    share = rng.uniform(0, 0.6, 1000)
    chosen = T._reference_cells(totals, share)
    rows = T.deep_selftest_rows(chosen, totals, share, 100, 0.3)
    assert len(rows) == 16 == len(set(rows.tolist())) and set(rows.tolist()) <= set(chosen.tolist())
    assert (share[rows[:8]] >= 0.3).all() and ((share[rows[8:12]] >= 0.1) & (share[rows[8:12]] < 0.3)).all()
    assert (share[rows[12:]] < 0.1).all() and (totals[rows] >= 100).all()
    assert np.all(np.diff(totals[rows[:8]]) >= 0)                   # each group from shallow to deep
    # Nothing looks mixed: the other judged cells fill in; nothing judged at all: any reference cell.
    calm = np.minimum(share, 0.05)
    assert len(T.deep_selftest_rows(chosen, totals, calm, 100, 0.3)) == 16
    shallow = np.minimum(totals, 99)
    rows = T.deep_selftest_rows(chosen, shallow, share, 100, 0.3)
    assert len(rows) == 16 and set(rows.tolist()) <= set(chosen.tolist())


# ---- the recipe of one cell (3.2), against an independent re-implementation ---------------------------------------------
def _ln_relu(h, eps):
    mean = h.mean(-1, keepdims=True)
    var = ((h - mean) ** 2).mean(-1, keepdims=True)
    return np.maximum((h - mean) / np.sqrt(var + eps), 0.0)


def _softmax(a):
    e = np.exp(a - a.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def reference_explain(enc, deep, x, xn, d2, logit=None, rounds=100, tol=1e-3, cap=50):
    """The model's equations written out again, float64, loops where they say "for each": the calculator's
    raw parts; own kind from the cell's own mean; neighbour kinds from ONE mean over all K x G neighbour counts, each
    at the cell's depth; shares from the learned values minus the clamped distance (or the MAP rounds); the observed
    split, then alternating column/row scaling; tags against the giving neighbour; own genes."""
    p = {k: np.asarray(v, np.float64) for k, v in deep.items() if isinstance(v, np.ndarray) and v.dtype.kind == "f"}
    e = {k: np.asarray(enc[k], np.float64) for k in ("W0", "b0", "W1", "b1", "Wv", "bv")}
    eps = float(enc["ln_eps"])
    h = _ln_relu(np.log1p(x) @ e["W0"].T + e["b0"], eps)
    h = _ln_relu(h @ e["W1"].T + e["b1"], eps)
    rho = _softmax(h @ e["Wv"].T + e["bv"]) + float(enc["var_eps"])
    rho[1] += float(enc["diffusion_eps"])

    def z_mean(v):
        g = _ln_relu(np.log1p(v) @ p["Z0"].T + p["zb0"], float(deep["ln_eps"]))
        g = _ln_relu(g @ p["Z1"].T + p["zb1"], float(deep["ln_eps"]))
        return g @ p["Zm"].T + p["zbm"]

    def profile(z):
        g = _ln_relu(z @ p["D0"].T + p["db0"], float(deep["ln_eps"]))
        g = _ln_relu(g @ p["D1"].T + p["db1"], float(deep["ln_eps"]))
        return _softmax(g @ p["Ds"].T + p["dbs"])
    L = x.sum()
    s = profile(z_mean(x / x.mean()))
    joint = xn.sum() / xn.size                                       # ONE mean over all K x G neighbour counts
    sk = np.array([profile(z_mean(xn[k] / joint)) for k in range(len(xn))])
    distance = np.minimum(np.sqrt(d2 / float(deep["median_distance"])), float(deep["clamp"]))
    K = len(xn)
    if logit is not None:
        pi = _softmax(np.asarray(logit, np.float64) - distance)
    else:
        alpha = 30.0 * (np.exp(-np.minimum(float(deep["diffuse_scale"]) * d2 / float(deep["median_distance"]), 20.0))
                        + 1e-3)
        pi = _softmax(-distance)
        for _ in range(rounds):
            n_k = np.zeros(K)
            for g in np.nonzero(x)[0]:
                mu_k = rho[1] * L * pi * sk[:, g]
                mu = rho[0] * L * s[g] + mu_k.sum() + rho[2] * L * p["background"][g]
                n_k += x[g] * mu_k / mu
            pi = np.maximum(n_k + alpha - 1.0, 1e-6)
            pi = pi / pi.sum()
    genes = np.nonzero(x)[0]
    A = np.zeros((len(genes), 3))
    per_k = np.zeros((K, len(genes)))
    for j, g in enumerate(genes):
        own, nbs, loose = rho[0] * L * s[g], rho[1] * L * pi * sk[:, g], rho[2] * L * p["background"][g]
        A[j] = x[g] * np.array([own, nbs.sum(), loose]) / (own + nbs.sum() + loose)
        per_k[:, j] = nbs / nbs.sum()
    raw = A.copy()
    target = L * rho / rho.sum()
    for rounds in range(1, cap + 1):
        A *= target / A.sum(0)
        A *= (x[genes] / A.sum(1))[:, None]
        if np.abs(A.sum(0) - target).max() <= tol:
            break
    Ak = A[:, 1] * per_k
    similar = sk @ s / (np.linalg.norm(sk, axis=1) * np.linalg.norm(s))
    giver = Ak.argmax(0)
    tag = np.array([x[g] >= 2 and A[j, 1] / x[g] >= 0.5 and similar[giver[j]] < 0.9 and sk[giver[j], g] >= 4 * s[g]
                    for j, g in enumerate(genes)])
    own_ok = (x[genes] >= 2) & (A[:, 0] / x[genes] >= 0.5)
    return {"rho": rho, "parts": rho / rho.sum(), "pi": pi, "raw": raw, "A": A, "Ak": Ak, "tag": tag, "own_ok": own_ok,
            "similar": similar, "genes": genes, "s": s, "sk": sk, "rounds": rounds}


@pytest.mark.parametrize("learned", [True, False])
def test_deep_forward_is_the_designs_recipe(learned):
    """deep_forward against the independent re-implementation on the stored real model's self-test cells (learned
    values carried over, and the MAP rounds for a cell the model never learned), float64 and float32 networks."""
    enc = T.load_encoder(STORED / "encoder.npz")
    deep = T.load_deep(STORED / "deep.npz", enc)
    counts = T.deep_selftest_counts(deep).astype(np.float64)
    for i in range(0, counts.shape[0], 3):
        x, xn, d2 = counts[i, 0], counts[i, 1:], deep["st_d2"][i].astype(np.float64)
        logit = deep["train_logit"][int(deep["st_rows"][i])].astype(np.float64) if learned else None
        ref = reference_explain(enc, deep, x, xn, d2, logit)
        got = T.deep_forward(np, T._as_dtype(enc, np.float64), T._as_dtype(deep, np.float64), x, xn, d2, logit)
        np.testing.assert_allclose(got["parts"], ref["parts"], atol=1e-10)
        np.testing.assert_allclose(got["pi"], ref["pi"], atol=1e-9)
        assert np.array_equal(got["genes_nz"], ref["genes"])
        np.testing.assert_allclose(got["A_raw"], ref["raw"], atol=1e-8)
        assert got["scale_rounds"] == ref["rounds"]
        np.testing.assert_allclose(got["A"], ref["A"], atol=1e-8)
        np.testing.assert_allclose(got["Ak"], ref["Ak"], atol=1e-8)
        np.testing.assert_allclose(got["similar"], ref["similar"], atol=1e-10)
        assert np.array_equal(got["tag"], ref["tag"]) and np.array_equal(got["own_ok"], ref["own_ok"])
        # the sums the window relies on: each gene's parts add up to its transcripts, each part to the card's share
        np.testing.assert_allclose(got["A"].sum(axis=1), x[got["genes_nz"]], atol=1e-9)
        assert got["scale_col_err"] <= 1e-3 and got["scale_rounds"] <= 50
        np.testing.assert_allclose(got["Ak"].sum(axis=0), got["A"][:, 1], atol=1e-9)
        assert got["tx_k"].sum() == pytest.approx(got["A"][:, 1].sum(), abs=1e-9)
        # float32 networks (the worker's): the same tags, the split within a fraction of a transcript
        f32 = T.deep_forward(np, enc, {k: v for k, v in deep.items() if k != "enc"}, x.astype(np.float32),
                             xn.astype(np.float32), d2.astype(np.float32),
                             None if logit is None else logit.astype(np.float32))
        assert f32["A"].dtype == np.float64 and np.abs(f32["A"] - got["A"]).max() < 0.01
        assert np.array_equal(f32["tag"], got["tag"]) and f32["scale_col_err"] <= 1e-3


def test_neighbour_kinds_use_one_joint_mean_and_the_cells_depth():
    """3.1's quirks: the neighbours' counts are normalised by ONE mean over all K x G counts (not each neighbour's own),
    and each neighbour's profile enters at the CENTRE cell's depth L."""
    enc = T.load_encoder(STORED / "encoder.npz")
    deep = T._as_dtype({k: v for k, v in T.load_deep(STORED / "deep.npz").items()}, np.float64)
    counts = T.deep_selftest_counts(deep).astype(np.float64)
    x, xn = counts[0, 0], counts[0, 1:]
    got = T.deep_forward(np, T._as_dtype(enc, np.float64), deep, x, xn, deep["st_d2"][0], deep["train_logit"][0])
    joint = T.deep_z_mean(np, deep, xn / xn.mean())
    each = T.deep_z_mean(np, deep, xn / xn.mean(axis=1, keepdims=True))
    np.testing.assert_allclose(got["mk"], joint, atol=1e-12)
    assert np.abs(each - joint).max() > 1e-3                         # the other reading would differ
    doubled = T.deep_forward(np, T._as_dtype(enc, np.float64), deep, x, xn * 2, deep["st_d2"][0], deep["train_logit"][0])
    np.testing.assert_allclose(doubled["A_raw"], got["A_raw"], atol=1e-9)   # neighbours' depth does not matter


def test_scaling_reaches_the_card_in_float64_for_deep_cells():
    """The split and its scaling run in float64 even with float32 networks (float32 sums stall at ~1e-3
    transcripts in cells of a few thousand)."""
    enc = T.load_encoder(STORED / "encoder.npz")
    deep = {k: v for k, v in T.load_deep(STORED / "deep.npz").items()}
    counts = T.deep_selftest_counts(deep)
    deepest = int(np.argmax(counts[:, 0].sum(axis=1)))
    x = (counts[deepest, 0] * 12).astype(np.float32)                  # a few thousand transcripts
    assert x.sum() > 2000
    r = T.deep_forward(np, enc, deep, x, counts[deepest, 1:], deep["st_d2"][deepest],
                       deep["train_logit"][int(deep["st_rows"][deepest])])
    assert r["A"].dtype == np.float64 and r["scale_col_err"] <= 1e-3
    np.testing.assert_allclose(r["A"].sum(axis=0), x.sum() * r["parts"].astype(np.float64), atol=1e-3)


# ---- the detail job's parity rule (5.4) ---------------------------------------------------------------------------------
def test_the_detail_parity_rule(tmp_path):
    installed = F.calculator(0)
    path = save(tmp_path / "installed.npz", installed)
    old = T.load_encoder(path)
    same = T.detail_parity(path, T.load_encoder(path))               # learned again bit for bit
    assert same["identical"] and same["accepted"] and same["reference_outputs_max_abs_diff"] is None
    assert set(same["max_abs_diff"]) == {"W0", "b0", "W1", "b1", "Wv", "bv", "ln_eps", "var_eps", "diffusion_eps"}

    def changed(**arrays):
        new = dict(old, **arrays)
        return {**new, "genes": list(new["genes"])}
    tiny = changed(bv=np.nextafter(old["bv"], np.float32(np.inf)))  # one float32 step: within 1e-6 on the reference cells
    record = T.detail_parity(old, tiny)
    assert record["accepted"] and not record["identical"] and record["reference_outputs_max_abs_diff"] <= 1e-6
    for new, words in ((changed(W0=(old["W0"] * 1.01).astype(np.float32)), "reference cells differ"),
                       (changed(genes=[f"H{i}" for i in range(7)]), "other genes"),
                       (changed(W1=old["W1"][:, :5]), "other layer sizes")):
        with pytest.raises(T.TrainerError, match=words) as raised:
            T.detail_parity(path, new)
        assert raised.value.reason == "detail_mismatch" and "nothing was changed" in str(raised.value)


# ---- provenance ----------------------------------------------------------------------------------------------------------
def model_json(**changes):
    """model.json as the trainer writes it for a whole sample (40,513 cells, 50 passes, priors pooled with the proseg
    table)."""
    model = {"table": {"key": "xenium_table"}, "recipe": T.recipe(), "epochs": 50,
             "priors_from": {"rule": "pooled", "tables": ["xenium_table", "table"], "cells": [40513, 40070]},
             "cells": {"fitted": 40513}}
    for key, value in changes.items():
        if value is None:
            model.pop(key)
        else:
            model[key] = value
    return model


def test_provenance_says_whether_the_notebook_learns_the_same_model():
    whole = T.provenance(model_json())
    assert whole == {"table": "xenium_table", "other_table": "table", "priors": "pooled", "priors_why": None,
                     "epochs_run": 50, "epochs_notebook": 50, "cells": 40513, "notebook_equivalent": True,
                     "differences": []}
    crop = T.provenance(model_json(epochs=429, cells={"fitted": 3455}))     # a small table gets more passes
    assert crop["differences"] == ["more_epochs_small_table"] and not crop["notebook_equivalent"]
    assert crop["epochs_run"] == 429 and crop["cells"] == 3455
    alone = T.provenance(model_json(priors_from={"rule": "table", "tables": ["xenium_table"], "cells": [40513],
                                                 "why": "the sample has no other cell table"}))
    assert alone["priors"] == "single" and alone["other_table"] is None and alone["differences"] == ["single_table_priors"]
    assert alone["priors_why"] == "the sample has no other cell table"
    other = T.provenance(model_json(table={"key": "table"}, priors_from={
        "rule": "table", "tables": ["table"], "cells": [40070], "why": "its genes differ from the learned table's"}))
    assert other["table"] == "table" and other["differences"] == ["other_table_used", "single_table_priors"]
    longer = T.provenance(model_json(recipe=T.recipe(epochs=80), epochs=80))
    assert longer["differences"] == ["other_epochs"]
    seed = T.provenance(model_json(recipe={**T.recipe(), "seed": 1}))
    assert seed["differences"] == ["other_settings"]
    by_recipe = T.provenance(model_json(recipe=T.recipe(priors="table"), priors_from={
        "rule": "table", "tables": ["xenium_table"], "cells": [40513], "why": "the recipe uses the learned table alone"}))
    assert by_recipe["differences"] == ["single_table_priors"]
    unknown = T.provenance(model_json(priors_from=None, epochs=None))
    assert unknown["priors"] is None and unknown["differences"] == ["not_recorded"] and not unknown["notebook_equivalent"]
    assert T.provenance({})["differences"] == ["not_recorded"] and T.provenance({"table": "x"})["table"] is None
    everything = T.provenance(model_json(table={"key": "table"}, recipe={**T.recipe(epochs=40), "lr": 0.01}, epochs=429,
                                         priors_from={"rule": "table", "tables": ["table"], "cells": [10]}))
    assert everything["differences"] == list(T.PROVENANCE_DIFFERENCES)[:5]   # in the documented order
    assert all(isinstance(text, str) and text for text in T.PROVENANCE_DIFFERENCES.values())


# ---- the recipe on small panels and small tables, and the cells' centres -----------------------------------------------
def test_the_background_prior_takes_at_least_one_gene():
    """The n_genes // 50 lowest-expressed genes are none below 50 genes, and the mean over none divided by zero
    (every toy panel of 5-44 genes failed at once). At least one is taken; at 50 genes or more nothing changes."""
    assert [T.low_genes(n) for n in (1, 5, 44, 49, 50, 99, 100, 5101)] == [1, 1, 1, 1, 1, 1, 2, 102]
    rng = np.random.default_rng(21)
    rec = T.recipe()
    for genes in (44, 5):
        counts = rng.poisson(rng.uniform(0.2, 4.0, genes), size=(300, genes)).astype(np.float32)
        counts[:, 0] += 1                                        # every cell holds something
        _, d2 = T.knn_graph(rng.uniform(0, 300, (300, 2)), 10)
        priors = T.dataset_priors(counts, d2, rec)
        lowest = np.argsort(counts.sum(axis=0))[:1]
        assert np.isfinite(priors["background_ratio"]) and priors["background_ratio"] == pytest.approx(
            np.mean(counts[:, lowest].mean(axis=1) / counts.mean(axis=1)))
    counts = rng.poisson(rng.uniform(0.05, 4.0, 100), size=(300, 100)).astype(np.float32)   # 100 genes: as before
    _, d2 = T.knn_graph(rng.uniform(0, 300, (300, 2)), 10)
    low = np.argsort(counts.sum(axis=0))[:100 // 50]
    assert T.dataset_priors(counts, d2, rec)["background_ratio"] == pytest.approx(
        np.mean(counts[:, low].mean(axis=1) / counts.mean(axis=1)), rel=1e-12)


def test_the_settle_rule_averages_over_more_passes_on_long_curves():
    """A table of 512 cells or fewer is one minibatch per pass, so every pass's loss is one noisy estimate (about
    3 % apart on a settled curve). Its 3,000 passes were judged on the last 5 against the 5 before: chance alone failed
    about one fit in four. The windows now grow with the passes (passes // 20, at least 5): a plateau with 3 % noise is
    settled, a curve that really rises over its last tenth is not, and 50 passes keep the 5-pass windows."""
    rng = np.random.default_rng(7)
    plateau = list(np.linspace(1.2e5, 7.7e4, 600)) + list(7.7e4 * (1 + 0.03 * rng.standard_normal(2400)))
    plateau[-5:] = [7.7e4 * 1.03] * 5                           # the last five happen to sit high, as noise does
    values = np.asarray(plateau)
    assert values[-5:].mean() > 1.01 * values[-10:-5].mean()   # what the 5-pass windows judged: "rose"
    assert T.loss_settled(plateau) == (True, "")
    rising = list(np.linspace(1.2e5, 7.7e4, 600)) + [7.7e4] * 2100 + list(np.linspace(7.7e4, 8.3e4, 300))
    settled, why = T.loss_settled(rising)
    assert not settled and "the loss rose over the last 300 passes" in why
    assert T.loss_settled(rising[:2700])[0]                     # the same curve before it rose
    # 50 passes (the notebook's, on large tables): 5-pass windows, as before.
    fifty = list(np.linspace(30e6, 17e6, 45)) + [17e6] * 5
    assert T.loss_settled(fifty)[0]
    late = list(np.linspace(30e6, 17e6, 40)) + [17e6] * 5 + [17.4e6] * 5    # +2.4 % over its last 5 passes
    assert not T.loss_settled(late)[0] and "last 10 passes" in T.loss_settled(late)[1]


def test_a_calculator_that_sees_no_mixing_anywhere_is_refused():
    """After its 3,000 passes over the same few hundred cells the calculator gives 92 % of the toy table's cells
    exactly its lowest share (a softmax of (1, 0, 0) plus its offsets): a model whose check says "no mixing" whatever
    the data. With 80 % or more of the learning cells there, the fit fails in plain words; the recipe stays as it is."""
    floor = T.floor_share(1e-3, 1e-2)
    assert floor == pytest.approx(0.0108588, abs=1e-6)          # the [0.9882, 0.0109, 0.0010] the toy model gave
    rng = np.random.default_rng(3)
    collapsed = np.where(np.arange(1000) < 900, floor + rng.uniform(0, 5e-5, 1000), rng.uniform(0.02, 0.5, 1000))
    assert T.at_floor(collapsed, 1e-3, 1e-2) == pytest.approx(0.9)
    with pytest.raises(T.TrainerError) as refused:
        T.collapse_check(collapsed.astype(np.float32), 1e-3, 1e-2)
    assert refused.value.reason == "too_few_cells"
    assert "sees no mixing in 90 % of the cells" in str(refused.value) and "too few cells" in str(refused.value)
    healthy = np.where(np.arange(1000) < 100, floor, rng.uniform(0.02, 0.5, 1000))
    assert T.collapse_check(healthy, 1e-3, 1e-2) == pytest.approx(0.1)
    assert T.at_floor(np.full(10, floor + 2 * T.FLOOR_TOLERANCE), 1e-3, 1e-2) == 0.0   # just above it: not at it
    assert T.at_floor([], 1e-3, 1e-2) == 0.0 and T.FLOOR_AT == 0.8


def test_cell_centres_are_placed_on_the_transcripts():
    """A table is learned with its cells' centres in the transcripts' frame. Its own centroids when they agree
    with its outlines' centres in that frame; the outlines' when they do not (CosMx keeps raw slide pixels); its own
    when it has no outlines and they lie on the transcripts; refused in plain words otherwise."""
    rng = np.random.default_rng(9)
    um = rng.uniform(0, 246, (400, 2))                           # the cells' outline centres, in µm
    extent = [0.0, 0.0, 246.0, 246.0]                           # the transcripts' frame
    near = um + rng.normal(0, 0.2, um.shape)                    # a vendor's centroids: within a fraction of a µm
    xy, found = T.choose_centres(near, um, extent, "table", "cell_boundaries")
    assert xy is near or np.array_equal(xy, near)
    assert found["from"] == "table" and found["offset"] < 0.5 * found["spacing"] and found["checked"]
    pixels = 8.31 * um + [12020.0, 20567.0]                     # CosMx's obsm['spatial']
    xy, found = T.choose_centres(pixels, um, extent, "table", "cells_polygons")
    assert np.array_equal(xy, um) and found["from"] == "outlines" and found["outlines"] == "cells_polygons"
    xy, found = T.choose_centres(None, um, extent, "table", "cells_polygons")   # no obsm['spatial'] at all
    assert np.array_equal(xy, um) and found["from"] == "outlines"
    xy, found = T.choose_centres(near, None, extent, "table")  # no outlines: its own, on the transcripts
    assert found["from"] == "table" and found["outlines"] is None
    edge = um.copy()
    edge[:10] += 10.0                                           # a few cells a little past the outermost transcript
    assert T.choose_centres(edge, None, extent, "table")[1]["from"] == "table"
    with pytest.raises(T.TrainerError) as off:
        T.choose_centres(pixels, None, extent, "table")
    assert off.value.reason == "positions" and "not in the transcripts' units" in str(off.value)
    assert "x 12" in str(off.value) and "x 0-246" in str(off.value)
    with pytest.raises(T.TrainerError) as none:
        T.choose_centres(None, None, extent, "cells")
    assert none.value.reason == "positions" and "no cell centroids" in str(none.value)
    with pytest.raises(T.TrainerError):                         # outlines that are off the transcripts too
        T.choose_centres(None, 8.31 * um + [12020.0, 20567.0], extent, "table")
    # Transcripts in two elements, the one in use covering x 0-123 only (half the cells beyond it): cells the outlines
    # placed are learned, whether the table's centroids agree with them or not; centroids alone must lie on them.
    half = [0.0, 0.0, 123.0, 246.0]
    assert T.choose_centres(near, um, half, "table", "cell_boundaries")[1]["from"] == "table"
    assert T.choose_centres(pixels, um, half, "table", "cell_boundaries")[1]["from"] == "outlines"
    assert T.choose_centres(None, um, half, "table", "cell_boundaries")[1]["from"] == "outlines"
    with pytest.raises(T.TrainerError, match="not in the transcripts' units"):
        T.choose_centres(near, None, half, "table")
    # A sample without transcripts to compare with (the tests' bare tables): its own centroids, unchecked.
    xy, found = T.choose_centres(pixels, None, None, "table")
    assert np.array_equal(xy, pixels) and found == {"table": "table", "from": "table", "outlines": None,
                                                     "checked": False}


# ---- the fake trainer's export (the engine's tests read it) --------------------------------------------------------------
def test_the_fake_trainers_export_passes_the_real_checks(tmp_path):
    job = {"zarr": "/x.zarr", "table": "xenium_table", "other_tables": ["table"], "model_id": "k", "recipe": T.recipe()}
    model = F.write_model(tmp_path / "fit", job, epochs=3)
    enc = T.load_encoder(tmp_path / "fit" / "encoder.npz")
    deep = T.load_deep(tmp_path / "fit" / "deep.npz", enc["genes"])
    assert T.deep_self_test(deep) <= T.DEEP_SELFTEST_TOLERANCE and model["deep"]["from"] == "fit"
    assert model["provenance"] == T.provenance(model) and model["provenance"]["notebook_equivalent"]
    import pyarrow.parquet as pq
    ids = pq.read_table(tmp_path / "fit" / "cells.parquet").column("cell_id").to_pylist()
    assert str(deep["train_ids_sha256"]) == T.ids_sha256(ids)
    old = F.write_model(tmp_path / "old", job, epochs=3, deep=False)      # learned before Go deeper
    assert "deep" not in old and "provenance" not in old and not (tmp_path / "old" / "deep.npz").exists()
    detail = F.write_detail(tmp_path / "staging", {**job, "detail_for": str(tmp_path / "old")})
    assert sorted(p.name for p in (tmp_path / "staging").iterdir()) == ["deep.npz", "model.json"]
    assert detail["deep"]["from"] == "detail" and detail["detail"]["accepted"] and detail["id"] == "k"
    assert T.deep_self_test(T.load_deep(tmp_path / "staging" / "deep.npz", enc)) <= T.DEEP_SELFTEST_TOLERANCE
    bad = F.write_model(tmp_path / "bad", job, epochs=3, bad_deep=True)
    assert bad["deep"]["selftest"]["max_abs_diff"] <= 1e-5                  # model.json claims it passed ...
    assert T.deep_self_test(T.load_deep(tmp_path / "bad" / "deep.npz")) > T.DEEP_SELFTEST_TOLERANCE   # ... it does not


# ---- the real trainer process: sopa environment, CPU only ----------------------------------------------------------------
def trainer_env(locks: Path, env: dict | None = None) -> dict:
    """The trainer's environment as the engine sets it, but with no GPU visible (CUDA_VISIBLE_DEVICES set and empty)
    unless `env` says otherwise (a value of None removes the variable)."""
    full = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "CUDA_VISIBLE_DEVICES")}
    full.update(PYTHONPATH=str(APP_PARENT), OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="",
                SOPA_COMPANION_ZARR_LOCK_DIR=str(locks))
    for key, value in (env or {}).items():
        if value is None:
            full.pop(key, None)
        else:
            full[key] = value
    return full


def start_trainer(job: dict, locks: Path, env: dict | None = None) -> subprocess.Popen:
    """The real trainer as the engine starts it (trainer_env), the job sent; stdin stays open until finish_trainer."""
    process = subprocess.Popen([sopa_python(), "-m", "sopa_companion.analysis.qc_trainer"], cwd=str(APP_PARENT),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               env=trainer_env(locks, env))
    process.stdin.write(json.dumps(job) + "\n")
    process.stdin.flush()
    return process


def finish_trainer(process: subprocess.Popen, timeout: float = 300) -> list[dict]:
    """Its JSON lines up to "done" or "error"; then stdin is closed (which ends a trainer that is still running)."""
    lines, deadline = [], time.monotonic() + timeout
    for line in process.stdout:
        lines.append(json.loads(line))
        if lines[-1]["ev"] in ("done", "error") or time.monotonic() > deadline:
            break
    process.stdin.close()
    process.wait(60)
    return lines


def run_trainer(job: dict, locks: Path, env: dict | None = None, timeout: float = 300) -> list[dict]:
    return finish_trainer(start_trainer(job, locks, env), timeout)


def cpu_job(zarr, out, **extra) -> dict:
    """The engine's job for the synthetic sample (10 passes) in the tests-only CPU mode."""
    return {"zarr": str(zarr), "table": "xenium_table", "other_tables": ["table"], "out_dir": str(out),
            "model_id": "m-test", "sample": {"id": "zsynthetic", "name": "synthetic"},
            "recipe": T.recipe(epochs=10, min_steps=1), "fingerprint": {"n_obs": 420, "n_vars": 60},
            "mixed_at": 0.3, "judge_min": 100, "deadline_s": 600, "cpu_for_tests": True, **extra}


@pytest.fixture(scope="module")
def sample_zarr(tmp_path_factory):
    root = tmp_path_factory.mktemp("deep-sample") / "s.zarr"
    subprocess.run([sopa_python(), "-I", str(MAKER), "--sample", str(root)], check=True, capture_output=True,
                   timeout=300, env={**{k: v for k, v in os.environ.items() if k != "DISPLAY"}, "CUDA_VISIBLE_DEVICES": ""})
    return root


def installed_copy(out: Path, folder: Path, older: bool = True) -> Path:
    """The fit's model as an installed model folder; `older`: as learned before Go deeper (no deep.npz, and no "deep"
    or "provenance" in model.json)."""
    shutil.copytree(out, folder)
    if older:
        (folder / "deep.npz").unlink()
        meta = json.loads((folder / "model.json").read_text())
        for key in ("deep", "provenance"):
            meta.pop(key)
        (folder / "model.json").write_text(json.dumps(meta, indent=1))
    return folder


def other_calculator(folder: Path):
    """Replace the folder's calculator with another one that passes its own self-test. (A uniform shift of W0 would
    not do: the layer norm cancels it, and the parity rule rightly accepts such a calculator as the same one.)"""
    arrays = dict(np.load(folder / "encoder.npz"))
    arrays["W0"] = (arrays["W0"] + np.random.default_rng(1).normal(0, 0.01, arrays["W0"].shape)).astype(np.float32)
    np.savez(folder / "encoder.npz", **arrays)
    changed = T.load_encoder(folder / "encoder.npz")
    arrays["selftest_expected"] = T.mixing_weights(changed, T.selftest_counts(changed).astype(np.float64), np,
                                                   np.float64).astype(np.float32)
    np.savez(folder / "encoder.npz", **arrays)


def snapshot(folder: Path) -> dict:
    return {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.fixture(scope="module")
def runs(sample_zarr, tmp_path_factory):
    """The real trainer's CPU runs, each once: the fit; then, side by side, a detail job of it as a model learned before
    Go deeper, a detail job against a calculator that differs, and the torch-side script (TORCH_SIDE)."""
    base = tmp_path_factory.mktemp("deep-runs")
    locks = base / "locks"
    fit = base / "fit"
    lines = run_trainer(cpu_job(sample_zarr, fit), locks)
    assert lines[-1]["ev"] == "done", lines[-1]
    older = installed_copy(fit, base / "older")
    differs = installed_copy(fit, base / "differs")
    other_calculator(differs)
    found = {"fit": (fit, lines), "older": (older, snapshot(older)), "differs": (differs, snapshot(differs))}
    started = {name: start_trainer(cpu_job(sample_zarr, base / f"{name}-staging", detail_for=str(folder)), locks)
               for name, folder in (("older", older), ("differs", differs))}
    with open(base / "torch-side.stderr", "w") as errors:
        degraded = json.dumps(cpu_job(sample_zarr, base / "degraded"))
        script = subprocess.Popen([sopa_python(), "-c", TORCH_SIDE, degraded],
                                  stdout=subprocess.PIPE, stderr=errors, text=True, cwd=str(APP_PARENT),
                                  env=trainer_env(locks))
        for name, process in started.items():
            found[name] += (base / f"{name}-staging", finish_trainer(process))
        output, _ = script.communicate(timeout=300)
    assert script.returncode == 0, (base / "torch-side.stderr").read_text()[-3000:]
    found["torch_side"] = (json.loads(output.strip().splitlines()[-1]), base / "degraded")
    return found


def test_a_fit_writes_the_details_after_the_calculator(runs):
    out, lines = runs["fit"]
    assert sorted(p.name for p in out.iterdir()) == ["batch.parquet", "cells.parquet", "deep.npz", "encoder.npz",
                                                     "model.json", "train.log"]
    stages = [(m.get("stage"), m.get("file")) for m in lines if m["ev"] == "progress"]
    at = stages.index(("exporting", None))
    assert ("checking", None) in stages[:at] and stages[at + 1] == ("exporting", "deep.npz")
    assert lines[-2]["stage"] == "scoring" and lines[-1]["model"] == json.loads((out / "model.json").read_text())
    enc = T.load_encoder(out / "encoder.npz")
    deep = T.load_deep(out / "deep.npz", enc["genes"])
    parts = T.deep_self_test_parts(deep)                             # numpy here against torch's stored outputs
    assert parts["max"] <= T.DEEP_SELFTEST_TOLERANCE and parts["nb_split"] is not None
    assert T.deep_self_test(deep, dtype=np.float32) <= T.DEEP_WORKER_TOLERANCE
    import pyarrow.parquet as pq
    ids = pq.read_table(out / "cells.parquet").column("cell_id").to_pylist()
    assert deep["train_logit"].shape == deep["train_index"].shape == (len(ids), 10) == (409, 10)
    assert str(deep["train_ids_sha256"]) == T.ids_sha256(ids) and deep["genes"] == enc["genes"]
    assert deep["Z0"].shape == (128, 60) and deep["Zm"].shape == (10, 128) and deep["D0"].shape == (32, 10)
    assert deep["Ds"].shape == (60, 32) and float(deep["clamp"]) == 10.0
    assert float(deep["ln_eps"]) == pytest.approx(1e-5) and float(deep["z_var_eps"]) == pytest.approx(1e-4)
    model = json.loads((out / "model.json").read_text())
    assert float(deep["median_distance"]) == pytest.approx(model["priors"]["median_distance"], rel=1e-6)
    block = model["deep"]
    assert {k: block[k] for k in ("format", "file", "from")} == {"format": T.DEEP_FORMAT, "file": "deep.npz",
                                                                 "from": "fit"}
    assert block["bytes"] == (out / "deep.npz").stat().st_size and block["created"]
    assert block["selftest"]["cells"] == 16 and block["selftest"]["tolerance"] == T.DEEP_SELFTEST_TOLERANCE
    assert block["selftest"]["max_abs_diff"] == pytest.approx(parts["max"], abs=1e-9)
    assert block["selftest"]["float32"]["max"] <= T.DEEP_WORKER_TOLERANCE and "deep_error" not in model
    assert model["provenance"] == {"table": "xenium_table", "other_table": "table", "priors": "pooled",
                                   "priors_why": None, "epochs_run": 10, "epochs_notebook": 50, "cells": 409,
                                   "notebook_equivalent": False, "differences": ["other_epochs"]}
    assert model["device"] == {"name": "cpu", "pci_bus_id": None, "memory_mb": None}


def test_an_other_table_that_is_not_prosegs_own_count_is_pooled_and_never_scored(sample_zarr, tmp_path):
    """The real fit of the synthetic sample whose other table holds no proseg run record (as a Xenium sample's own
    "table": Xenium Ranger's outline re-count): pooled into the priors as the notebook does, but none of its cells is
    scored (no batch.parquet, model.json's batch null, no scoring stage), so the earlier proseg cells' line never
    shows an outline re-count as proseg's cells."""
    zarr = tmp_path / "outline.zarr"
    subprocess.run([sopa_python(), "-I", str(MAKER), "--sample", str(zarr), "--not-proseg"], check=True,
                   capture_output=True, timeout=300,
                   env={**{k: v for k, v in os.environ.items() if k != "DISPLAY"}, "CUDA_VISIBLE_DEVICES": ""})
    assert T.proseg_table(sample_zarr, "table") and not T.proseg_table(zarr, "table")
    lines = run_trainer(cpu_job(zarr, tmp_path / "out"), tmp_path / "locks")
    assert lines[-1]["ev"] == "done", lines[-1]
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["cells.parquet", "deep.npz", "encoder.npz", "model.json",
                                                     "train.log"]
    model = json.loads((out / "model.json").read_text())
    assert model["batch"] is None and lines[-1]["model"] == model
    assert model["priors_from"]["rule"] == "pooled" and model["priors_from"]["tables"] == ["xenium_table", "table"]
    assert "scoring" not in [m.get("stage") for m in lines if m["ev"] == "progress"]


def test_a_detail_job_adds_only_the_details(runs):
    fit, _ = runs["fit"]
    installed, before, staging, lines = runs["older"]
    assert lines[-1]["ev"] == "done", lines[-1]
    assert [m["stage"] for m in lines if m["ev"] == "progress"][-2:] == ["checking", "exporting"]
    # deep.npz, model.json and the new calculator beside them (no log): the engine compares that calculator with the
    # installed one once more before it installs anything, and removes it with staging (it was deleted here once, so
    # the engine's own parity check never ran).
    assert sorted(p.name for p in staging.iterdir()) == [T.DETAIL_ENCODER, "deep.npz", "model.json"]
    from sopa_companion.engine.qc import calculators_agree
    assert calculators_agree(T.load_encoder(installed / "encoder.npz"), T.load_encoder(staging / T.DETAIL_ENCODER))
    assert snapshot(installed) == before                                              # the installed model untouched
    old = json.loads(before["model.json"])
    new = json.loads((staging / "model.json").read_text())
    assert lines[-1]["model"] == new
    assert {k: v for k, v in new.items() if k not in ("deep", "detail", "provenance")} == old
    assert new["deep"]["from"] == "detail" and new["deep"]["format"] == T.DEEP_FORMAT
    assert new["deep"]["bytes"] == (staging / "deep.npz").stat().st_size
    assert new["deep"]["selftest"]["installed_max_abs_diff"] <= T.DEEP_SELFTEST_TOLERANCE
    detail = new["detail"]
    assert detail["accepted"] and detail["installed"] == str(installed) and detail["epochs"] == 10
    assert detail["identical"] or detail["reference_outputs_max_abs_diff"] <= T.DETAIL_TOLERANCE
    assert detail["device"]["name"] == "cpu" and detail["seconds"]["total"] > 0
    assert new["provenance"] == T.provenance(old) == T.provenance(new)
    # What the engine checks before installing: the details against the INSTALLED calculator.
    enc = T.load_encoder(installed / "encoder.npz")
    deep = T.load_deep(staging / "deep.npz", enc)
    assert T.deep_self_test(deep) <= T.DEEP_SELFTEST_TOLERANCE
    assert T.deep_self_test(T.load_deep(staging / "deep.npz")) <= T.DEEP_SELFTEST_TOLERANCE   # nothing beside it
    # The same model learned again gives the same details: compared with tolerances, never byte for byte.
    first = T.load_deep(fit / "deep.npz")
    for key in T.DEEP_KEYS - {"format", "genes", "train_ids_sha256"}:
        if deep[key].dtype.kind == "f":
            np.testing.assert_allclose(deep[key], first[key], rtol=0, atol=1e-5, err_msg=key)
        else:
            assert np.array_equal(deep[key], first[key]), key
    assert str(deep["train_ids_sha256"]) == str(first["train_ids_sha256"]) and deep["genes"] == first["genes"]


def test_a_detail_job_whose_learning_differs_changes_nothing(runs):
    installed, before, staging, lines = runs["differs"]
    assert lines[-1]["ev"] == "error" and lines[-1]["reason"] == "detail_mismatch", lines[-1]
    assert "reference cells differ" in lines[-1]["message"] and "nothing was changed" in lines[-1]["message"]
    assert "learning" in [m.get("stage") for m in lines]             # it learned, then compared
    assert not staging.exists() or not list(staging.iterdir())       # no details, no model.json, no calculator left
    assert snapshot(installed) == before


@pytest.mark.parametrize("change, reason, words", [
    ("none", None, None),
    ("model_id", "detail_mismatch", "not other"),
    ("table", "detail_mismatch", "learned from table 'xenium_table'"),
    ("recipe", "detail_mismatch", "another recipe"),
    ("cells", "detail_mismatch", "cells are not the ones"),
    ("genes", "detail_mismatch", "genes are not the ones"),
    ("priors", "detail_mismatch", "priors differ"),
    ("missing", "unexpected", "could not be read"),
    ("damaged", "unexpected", "could not be read"),
])
def test_a_detail_job_refuses_another_model_before_it_learns(runs, sample_zarr, tmp_path, change, reason, words):
    """prepare_only: everything before learning, so the refusals cost no GPU time (and these tests no torch)."""
    out, _ = runs["fit"]
    installed = installed_copy(out, tmp_path / "installed")
    job = cpu_job(sample_zarr, tmp_path / "staging", detail_for=str(installed), prepare_only=True)
    meta = json.loads((installed / "model.json").read_text())
    if change == "model_id":
        job["model_id"] = "other"
    elif change == "table":
        job["table"], job["other_tables"] = "table", ["xenium_table"]
    elif change == "recipe":
        job["recipe"] = T.recipe(epochs=11, min_steps=1)
    elif change == "cells":
        import pyarrow as pa
        import pyarrow.parquet as pq
        cells = pq.read_table(installed / "cells.parquet")
        ids = cells.column("cell_id").to_pylist()
        ids[7] = "someone-else"
        pq.write_table(cells.set_column(0, "cell_id", pa.array(ids)), installed / "cells.parquet")
    elif change == "genes":
        arrays = dict(np.load(installed / "encoder.npz"))
        arrays["genes"] = np.array([f"X{i}" for i in range(len(arrays["genes"]))])
        np.savez(installed / "encoder.npz", **arrays)
    elif change == "priors":
        meta["priors"]["median_distance"] *= 1.5
        (installed / "model.json").write_text(json.dumps(meta))
    elif change == "missing":
        job["detail_for"] = str(tmp_path / "nowhere")
    elif change == "damaged":
        (installed / "encoder.npz").write_bytes((installed / "encoder.npz").read_bytes()[:1000])
    lines = run_trainer(job, tmp_path / "locks")
    if reason is None:
        assert lines[-1]["ev"] == "done" and lines[-1]["model"]["prepared"]
        assert lines[-1]["model"]["detail_for"] == str(installed)
        return
    assert lines[-1]["ev"] == "error" and lines[-1]["reason"] == reason and words in lines[-1]["message"], lines[-1]
    assert "learning" not in [m.get("stage") for m in lines]


@pytest.mark.parametrize("visible", [None, "-1"])
def test_a_cpu_test_fit_runs_only_where_no_gpu_can_be_seen(sample_zarr, tmp_path, visible):
    """The tests-only CPU mode needs CUDA_VISIBLE_DEVICES set and empty, and stops before reading anything otherwise."""
    lines = run_trainer(cpu_job(sample_zarr, tmp_path / "out"), tmp_path / "locks", env={"CUDA_VISIBLE_DEVICES": visible})
    assert lines[-1]["ev"] == "error" and lines[-1]["reason"] == "gpu" and "set and empty" in lines[-1]["message"]
    assert [m.get("stage") for m in lines if m["ev"] == "progress"] == []
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["train.log"]       # the refusal, in words; nothing else


def test_a_fit_on_apples_gpu_runs_only_after_its_test(sample_zarr, tmp_path):
    """A job for Apple's GPU (device "mps") learns there only after zarr_bridge/devices.py MPS_LEARNING_TEST passed in a
    process of its own (PyTorch 2.15 calls MPS available on macOS 14, where every linear layer then aborts): where it
    fails (here: no Apple GPU) the fit stops with reason "gpu" in the test's words before it learns, its CUDA guards not
    asked (a Mac has no CUDA_VISIBLE_DEVICES of its own). On a Mac whose GPU passes it learns there, and says so."""
    job = {k: v for k, v in cpu_job(sample_zarr, tmp_path / "out", device="mps").items() if k != "cpu_for_tests"}
    lines = run_trainer(job, tmp_path / "locks", env={"CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": ""})
    stages = [m.get("stage") for m in lines if m["ev"] == "progress"]
    if lines[-1]["ev"] == "error":
        assert lines[-1]["reason"] == "gpu" and lines[-1]["message"].startswith("Apple's GPU failed a short test with "
                                                                                "PyTorch "), lines[-1]
        assert "learning" not in stages
    else:
        assert sys.platform == "darwin" and lines[-1]["ev"] == "done", lines[-1]
        devices = [m["device"] for m in lines if m["ev"] == "progress" and m.get("device")]
        assert devices and all(d["kind"] == "mps" and d["pci_bus_id"] is None for d in devices)


def test_the_trainer_asks_apples_gpu_the_learnings_own_test(sample_zarr, tmp_path):
    """The trainer, told "mps", asks Apple's GPU MPS_LEARNING_TEST (the learning's own operations, lgamma against the
    CPU), not Cellpose's MPS_TEST, and stops with its words before it learns: the real trainer, its devices module's
    answer replaced by one that records what was asked."""
    asked = tmp_path / "asked.txt"
    driver = ("from sopa_companion.zarr_bridge import devices as D\n"
              "def answer(torch, test=D.MPS_TEST):\n"
              f"    open({str(asked)!r}, 'w').write('learning' if test == D.MPS_LEARNING_TEST else 'another')\n"
              "    return 'the stand-in says no'\n"
              "D.apple_gpu_problem = answer\n"
              "from sopa_companion.analysis import qc_trainer as T\n"
              "raise SystemExit(T.main())\n")
    job = {k: v for k, v in cpu_job(sample_zarr, tmp_path / "out", device="mps").items() if k != "cpu_for_tests"}
    process = subprocess.Popen([sopa_python(), "-c", driver], cwd=str(APP_PARENT), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               env=trainer_env(tmp_path / "locks", {"CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": ""}))
    process.stdin.write(json.dumps(job) + "\n")
    process.stdin.flush()
    lines = finish_trainer(process)
    assert lines[-1] == {"ev": "error", "reason": "gpu", "message": "the stand-in says no"}
    assert asked.read_text() == "learning"
    assert "learning" not in [m.get("stage") for m in lines if m["ev"] == "progress"]


def test_apples_gpu_is_recorded_by_its_chip_and_memory(monkeypatch):
    """A fit on Apple's GPU records it as a CUDA fit records its card: the chip's name (sysctl's
    machdep.cpu.brand_string), its kind, no bus, and the memory Metal recommends one process uses at most; plain words
    where either cannot be read."""
    torch = types.SimpleNamespace(mps=types.SimpleNamespace(recommended_max_memory=lambda: 96 * 2 ** 30))
    monkeypatch.setattr(T.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0,
                                                                                                "Apple M2 Ultra\n", ""))
    assert T.apple_gpu(torch) == {"name": "Apple M2 Ultra GPU (MPS)", "kind": "mps", "pci_bus_id": None,
                                  "memory_mb": 96 * 1024}

    def missing(*args, **kwargs):
        raise OSError("no sysctl")
    monkeypatch.setattr(T.subprocess, "run", missing)
    assert T.apple_gpu(types.SimpleNamespace(mps=types.SimpleNamespace())) == {
        "name": "Apple silicon GPU (MPS)", "kind": "mps", "pci_bus_id": None, "memory_mb": None}


TORCH_SIDE = r'''
import json, sys
import numpy as np
import anndata, torch
from scviva.model import ResolVI
from sopa_companion.analysis import qc_trainer as T

rng = np.random.default_rng(0)
G, N = 30, 120
xy = rng.uniform(0, 100, (N, 2))
index, d2 = T.knn_graph(xy, 10)


def build(batches=1):
    adata = anndata.AnnData(X=rng.poisson(3.0, (N, G)).astype(np.float32) + 1)
    adata.obs_names = [f"c{i}" for i in range(N)]
    adata.var_names = [f"g{i}" for i in range(G)]
    adata.obsm["X_spatial"] = xy.astype(np.float32)
    adata.obsm["index_neighbor"] = index
    adata.obsm["distance_neighbor"] = d2
    keys = {}
    if batches > 1:
        adata.obs["batch"] = [f"b{i % batches}" for i in range(N)]
        keys = {"batch_key": "batch"}
    ResolVI.setup_anndata(adata, prepare_data=False, **keys)
    return ResolVI(adata, n_latent=10, n_hidden_encoder=128)


results = {}


def check(name, model, genes=G):
    try:
        T.deep_structure(model, genes)
        results[name] = "ok"
    except T.TrainerError as exc:
        results[name] = exc.reason + ": " + str(exc)


plain = build()
check("plain", plain)
check("other_gene_count", plain, G + 1)
z_layers = plain.module.z_encoder.encoder.fc_layers
d_layers = plain.module.model.decoder.px_decoder.fc_layers
model = build()
model.module.z_encoder.encoder.fc_layers[0][1] = torch.nn.BatchNorm1d(128, affine=False, track_running_stats=False)
check("batch_norm", model)
model = build()
model.module.model.decoder.px_decoder.fc_layers[1][2] = torch.nn.LayerNorm(32, elementwise_affine=True)
check("affine_layer_norm", model)
model = build()
model.module.z_encoder.encoder.fc_layers[1][2] = None
check("missing_layer_norm", model)
model = build()
model.module.z_encoder.encoder.fc_layers[0][2].eps = 1e-3
check("other_eps", model)
check("covariates", build(batches=2))
model = build()
model.module.model.dispersion = "gene-batch"
check("dispersion", model)
model = build()
model.module.z_encoder.var_activation = torch.nn.functional.softplus
check("variance", model)
results["layers"] = {"z": [type(m).__name__ for m in z_layers[0] if m is not None],
                     "decoder": [type(m).__name__ for m in d_layers[0] if m is not None]}

# A fit whose deep export fails keeps its calculator alone and says why (model.json "deep": null, "deep_error").
class Report:
    log_path = None
    def progress(self, stage, step=None, of=None, **extra):
        notes.append({"stage": stage, **extra})
notes = []
def broken(*args, **kwargs):
    (args[1] / "deep.npz").write_bytes(b"half")
    raise T.TrainerError("unexpected", "a simulated export failure")
T.export_deep = broken
job = json.loads(sys.argv[1])
model_json = T.fit(job, Report())
results["degraded"] = {"deep": model_json["deep"], "deep_error": model_json.get("deep_error"),
                       "provenance": model_json["provenance"], "notes": [n for n in notes if n.get("file")]}
print(json.dumps(results))
'''


def test_the_export_refuses_other_networks_and_never_keeps_a_failed_export(runs):
    """Structural checks on the library's own modules (BatchNorm, an affine or missing LayerNorm, another
    eps, covariate columns, other dispersion or variance, another gene count), and a fit whose deep export fails:
    the calculator stays, deep.npz goes, model.json says why."""
    results, out = runs["torch_side"]
    assert results["plain"] == "ok", results
    assert results["layers"] == {"z": ["Linear", "LayerNorm", "ReLU", "Dropout"], "decoder": ["Linear", "LayerNorm", "ReLU"]}
    expected = {"other_gene_count": "another gene count", "batch_norm": "normalisation layers",
                "affine_layer_norm": "decoder's layers are not the expected ones",
                "missing_layer_norm": "normalisation layers", "other_eps": "other LayerNorm eps",
                "covariates": "covariates", "dispersion": "one-batch, per-gene-dispersion",
                "variance": "variance is not exp"}
    for name, words in expected.items():
        assert results[name].startswith("unexpected: ") and words in results[name], (name, results[name])
    degraded = results["degraded"]
    assert degraded["deep"] is None and degraded["deep_error"] == {"reason": "unexpected",
                                                                   "message": "TrainerError: a simulated export failure"}
    assert degraded["provenance"]["epochs_run"] == 10
    assert [n["file"] for n in degraded["notes"]] == ["deep.npz", "deep.npz"] and "were not kept" in degraded["notes"][1]["note"]
    assert sorted(p.name for p in out.iterdir()) == ["batch.parquet", "cells.parquet", "encoder.npz", "model.json"]
    stored = json.loads((out / "model.json").read_text())
    assert stored["deep"] is None and stored["deep_error"]["reason"] == "unexpected"
    T.self_test(T.load_encoder(out / "encoder.npz"))                  # the check itself still has its calculator


# ---- small panels, collapsed calculators and cell centres in the real trainer process (sopa environment, CPU only) -----
WRITE_FEW_GENES = r'''
import sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, zarr
root, n_genes = sys.argv[1], int(sys.argv[2])
# make_qc_deep_model.py's two tables (cells of 4 kinds mixed with their nearest neighbours) with a panel of n_genes.
rng = np.random.default_rng(5)
profiles = rng.gamma(0.3, 1.0, (4, n_genes))
profiles /= profiles.sum(axis=1, keepdims=True)
genes = np.array([f"G{i}" for i in range(n_genes)])
zarr.open_group(root, mode="w")
for key, n, seed in (("xenium_table", 420, 1), ("table", 380, 2)):
    r = np.random.default_rng(seed)
    xy = r.uniform(0, 320, (n, 2))
    kind = ((xy[:, 0] // 80).astype(int) + (xy[:, 1] // 80).astype(int)) % 4
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    mix = r.uniform(0, 0.5, n)[:, None]
    p = (1 - mix) * profiles[kind] + mix * profiles[kind[np.argsort(d2, axis=1)[:, :3]]].mean(axis=1)
    counts = r.poisson(p * r.integers(60, 600, n)[:, None]).astype(np.float32)
    obs = pd.DataFrame({"cell_id": [f"{key[:3]}{i}" for i in range(n)]}, index=[str(i) for i in range(n)])
    table = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes))
    table.obsm["spatial"] = xy
    table.write_zarr(f"{root}/tables/{key}")
'''

WRITE_PIXEL_SAMPLE = r'''
import sys, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, geopandas as gpd, shapely
import spatialdata as sd
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale
# A sample laid out as sopa.io.cosmx writes one: the transcripts in µm (their own frame, the working frame) with
# global = Scale(8.31) slide pixels; the cells' outlines in slide pixels (Identity); the table annotating them by global
# id with obsm['spatial'] in raw slide pixels. "agrees": the same cells, obsm['spatial'] within 0.2 µm of the outlines'
# centres (as Xenium's); "unplaced": no annotation, pixel centroids.
root, PX = sys.argv[1], 8.31
rng = np.random.default_rng(5)
n, n_genes = 420, 60
profiles = rng.gamma(0.3, 1.0, (4, n_genes))
profiles /= profiles.sum(axis=1, keepdims=True)
r = np.random.default_rng(1)
xy = r.uniform(5, 315, (n, 2))
kind = ((xy[:, 0] // 80).astype(int) + (xy[:, 1] // 80).astype(int)) % 4
counts = r.poisson(profiles[kind] * r.integers(60, 600, n)[:, None]).astype(np.float32)
ids = np.arange(100, 100 + n)
genes = [f"G{i}" for i in range(n_genes)]
at = xy[np.repeat(np.arange(n), 30)] + r.normal(0, 2.0, (30 * n, 2))
points = PointsModel.parse(pd.DataFrame({"x": at[:, 0], "y": at[:, 1], "z": np.zeros(len(at)),
                                         "target": pd.Categorical(r.choice(genes, len(at)))}),
                           feature_key="target", transformations={"global": Scale([PX, PX], axes=("x", "y"))})
outlines = gpd.GeoDataFrame(geometry=[shapely.box(*(PX * (c - 4.0)), *(PX * (c + 4.0))) for c in xy], index=pd.Index(ids))
shapes = ShapesModel.parse(outlines, transformations={"global": Identity()})


def table(spatial):
    obs = pd.DataFrame({"global_cell_id": ids, "region": pd.Categorical(["cells_polygons"] * n), "cell_ID": np.arange(n)},
                       index=[str(i) for i in range(n)])
    found = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes))
    found.obsm["spatial"] = spatial
    return TableModel.parse(found, region="cells_polygons", region_key="region", instance_key="global_cell_id")
plain = ad.AnnData(sp.csr_matrix(counts), obs=pd.DataFrame(index=[str(i) for i in range(n)]),
                   var=pd.DataFrame(index=genes))
plain.obsm["spatial"] = PX * xy + [12000.0, 20000.0]
tables = {"table": table(PX * xy + [12000.0, 20000.0]), "agrees": table(xy + r.normal(0, 0.2, xy.shape)),
          "unplaced": TableModel.parse(plain)}
sd.SpatialData(points={"transcripts": points}, shapes={"cells_polygons": shapes}, tables=tables).write(root)
np.save(root + ".um.npy", xy)
'''

GUARDS_SIDE = r'''
import json, sys
import numpy as np
from sopa_companion.analysis import qc_trainer as T
from scviva.model import ResolVI
import anndata, scipy.sparse as sp
from scvi import REGISTRY_KEYS
results = {}


def build(cls, genes):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (120, 2))
    index, d2 = T.knn_graph(xy, 10)
    adata = anndata.AnnData(X=sp.csr_matrix(rng.poisson(3.0, (120, genes)).astype(np.float32) + 1))
    adata.obs_names = [f"c{i}" for i in range(120)]
    adata.var_names = [f"g{i}" for i in range(genes)]
    adata.obsm["index_neighbor"], adata.obsm["distance_neighbor"] = index, d2
    cls.setup_anndata(adata, prepare_data=False)
    return cls(adata, n_latent=10, n_hidden_encoder=128)
# ResolVI's own dataset priors (computed in its __init__ before the app's are used): none below 50 genes in the
# library, at least one with the app's class; at 120 genes the library's own number (120 // 50).
for genes in (5, 44):
    try:
        build(ResolVI, genes)
        results[f"library_{genes}"] = "ok"
    except Exception as exc:
        results[f"library_{genes}"] = type(exc).__name__
    model = build(T.resolvi_class(), genes)
    results[f"app_{genes}"] = model.compute_dataset_dependent_priors()
model = build(T.resolvi_class(), 120)
results["app_120"] = model.compute_dataset_dependent_priors()
results["library_120"] = ResolVI.compute_dataset_dependent_priors(model, 120 // 50)


# A fit whose learned calculator gives every cell its floor share: refused, nothing exported.
class Report:
    log_path = None

    def progress(self, stage, step=None, of=None, **extra):
        pass
T.neighbour_share = lambda weights, xp=np: np.full(len(weights), T.floor_share(1e-3, 1e-2))
job = json.loads(sys.argv[1])
try:
    T.fit(job, Report())
    results["collapsed"] = "done"
except T.TrainerError as exc:
    results["collapsed"] = [exc.reason, str(exc)]
print(json.dumps(results))
'''


def write_with_sopa(script: str, *args, timeout: float = 300):
    subprocess.run([sopa_python(), "-c", script, *map(str, args)], check=True, capture_output=True, timeout=timeout,
                   env={**{k: v for k, v in os.environ.items() if k != "DISPLAY"}, "CUDA_VISIBLE_DEVICES": ""})


@pytest.fixture(scope="module")
def few_genes_zarr(tmp_path_factory):
    root = tmp_path_factory.mktemp("few-genes") / "s.zarr"
    write_with_sopa(WRITE_FEW_GENES, root, 40)
    return root


@pytest.fixture(scope="module")
def pixel_zarr(tmp_path_factory):
    root = tmp_path_factory.mktemp("pixel-sample") / "s.zarr"
    write_with_sopa(WRITE_PIXEL_SAMPLE, root)
    return root


def test_a_panel_of_fewer_than_50_genes_is_learned(few_genes_zarr, tmp_path):
    """The app's own trainer on a 40-gene table (every toy panel had 5-44 genes): it failed within a second, first
    in the app's priors, then (with those fixed) in ResolVI's own; now it learns, exports and self-tests."""
    lines = run_trainer(cpu_job(few_genes_zarr, tmp_path / "out", fingerprint={"n_obs": 420, "n_vars": 40}),
                        tmp_path / "locks")
    assert lines[-1]["ev"] == "done", lines[-1]
    model = lines[-1]["model"]
    assert model["cells"]["genes"] == 40 and model["priors_from"]["rule"] == "pooled"
    assert 0 < model["priors"]["background_ratio"] < 1 and model["cells"]["at_floor"] < T.FLOOR_AT
    enc = T.load_encoder(tmp_path / "out" / "encoder.npz")
    assert len(enc["genes"]) == 40 and T.self_test(enc) <= T.SELFTEST_TOLERANCE
    assert model["centres"]["table"] == {"table": "xenium_table", "from": "table", "outlines": None, "checked": False,
                                         "frame": None}       # tables alone: no transcripts to place them on


@pytest.fixture(scope="module")
def guards(few_genes_zarr, tmp_path_factory):
    base = tmp_path_factory.mktemp("guards")
    job = cpu_job(few_genes_zarr, base / "collapsed")
    result = subprocess.run([sopa_python(), "-c", GUARDS_SIDE, json.dumps(job)], capture_output=True, text=True,
                            timeout=300, cwd=str(APP_PARENT), env=trainer_env(base / "locks"))
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1]), base / "collapsed"


def test_resolvis_own_priors_take_at_least_one_low_gene(guards):
    """ResolVI's own dataset priors divide by zero below 50 genes; the app's class asks for at least one low gene
    and, at 50 genes or more, for exactly the library's own number (the same model as before)."""
    results, _ = guards
    assert results["library_5"] == results["library_44"] == "ZeroDivisionError"
    for genes in (5, 44):
        assert all(np.isfinite(v) for v in results[f"app_{genes}"].values()), results[f"app_{genes}"]
    assert results["app_120"] == results["library_120"]


def test_a_fit_whose_calculator_sees_no_mixing_is_refused(guards):
    """A fit whose learned calculator gives its cells the floor share is refused in plain words, before anything
    is exported (no calculator, no details, no cells)."""
    results, out = guards
    reason, message = results["collapsed"]
    assert reason == "too_few_cells" and "sees no mixing in 100 % of the cells" in message
    assert not (out / "encoder.npz").exists() and not (out / "model.json").exists()


@pytest.mark.parametrize("table, frame", [("table", False), ("table", True), ("agrees", False), ("unplaced", False)])
def test_cell_centres_from_the_outlines_in_the_transcripts_frame(pixel_zarr, tmp_path, table, frame):
    """A table whose obsm['spatial'] is in raw slide pixels (CosMx) is learned with its outlines' centres in the
    transcripts' µm: the priors' median distance is the µm one, not 69 times it; one whose centroids agree with its
    outlines keeps them (the notebook's); one with neither is refused in plain words, before learning. The engine's
    frame (its open sample's points element and extent) gives the same as the trainer's own look."""
    um = np.load(str(pixel_zarr) + ".um.npy")
    extra = {"points_key": "transcripts", "extent": [0.0, 0.0, 320.0, 320.0]} if frame else {}
    job = {**cpu_job(pixel_zarr, tmp_path / "out", table=table, other_tables=[], prepare_only=True), **extra}
    lines = run_trainer(job, tmp_path / "locks")
    if table == "unplaced":
        assert lines[-1]["ev"] == "error" and lines[-1]["reason"] == "positions", lines[-1]
        assert "not in the transcripts' units" in lines[-1]["message"]
        assert "learning" not in [m.get("stage") for m in lines]
        return
    assert lines[-1]["ev"] == "done", lines[-1]
    model = lines[-1]["model"]
    centres = model["centres"]["table"]
    assert centres["from"] == ("outlines" if table == "table" else "table") and centres["frame"] == "transcripts"
    assert centres["outlines"] == "cells_polygons" and centres["checked"]
    _, d2 = T.knn_graph(um, 10)                                 # the cells' centres in µm
    expected = T.dataset_priors(np.ones((len(um), 1)), d2, T.recipe())["median_distance"]
    assert model["priors"]["median_distance"] == pytest.approx(expected, rel=0.05)
    assert model["priors"]["median_distance"] < 2 * expected     # not 69 x it (squared pixels)
