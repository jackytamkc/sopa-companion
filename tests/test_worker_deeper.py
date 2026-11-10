"""The GPU worker's Go deeper op ("deeper") and the mixing check's honesty additions (coverage counts and the
like-for-like change against the frame's start block), all on proseg's own count of each frame (its counts file) and
without a GPU: the worker's own code with numpy, its GPU code (cupyx matrices and all) on numpy dressed as cupy with a
minimal CSR on numpy, the D0 switch, and the fake worker the engine tests use. Real data (measured models) is used when
it is on this computer and skipped otherwise."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import harness_path
from sopa_companion.analysis import counts_file
from sopa_companion.analysis import worker as W
from test_analysis import (JUDGE, KEEP, MODEL_GENES, NCELLS, RUN_GENES, X0, Y0, NX, NY, VOXEL, GpuLike, cell_stats,
                           counted, counts_files, expected_scores, fake_cupyx, frame_plane, like_for_like_flags,
                           like_for_like_reference, make_model, mixing_model, model_rows, prior_plane,
                           prior_upper_layer, proseg_count, write_run)

HERE = Path(__file__).resolve().parent
RUN_IDS = [f"c{i:02d}" for i in range(NCELLS)]                 # the run's cells.txt: line k names label k + 1
UNLEARNED = {"c03", "c11"}                                     # run cells the model never learned from (MAP)
EXTRA = ["x0", "x1", "x2"]                                     # learned cells the run does not hold
K = 10


# ---- a small model with its details, self-tested against an independent float64 reference -------------------------
def softmax(a, axis=-1):
    a = np.exp(a - a.max(axis=axis, keepdims=True))
    return a / a.sum(axis=axis, keepdims=True)


def ln_relu(h, eps):
    centred = h - h.mean(axis=-1, keepdims=True)
    return np.maximum(centred / np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + eps), 0)


def reference(enc, deep, x, xn, d2, logit):
    """3.2 steps 3-8 written out again in float64 from the model's formulas (not the worker's code): z means, the own
    profile, the neighbour shares and the observed neighbour split x * mu_nb / mu of every gene."""
    f = {k: np.asarray(v, np.float64) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v
         for k, v in deep.items()}
    e = {k: np.asarray(enc[k], np.float64) for k in ("W0", "b0", "W1", "b1", "Wv", "bv")}
    h = ln_relu(np.log1p(x) @ e["W0"].T + e["b0"], 1e-5)
    h = ln_relu(h @ e["W1"].T + e["b1"], 1e-5)
    rho = softmax(h @ e["Wv"].T + e["bv"]) + 1e-3
    rho[1] += 1e-2
    eps = float(f["ln_eps"])

    def zmean(v):
        g = ln_relu(np.log1p(v) @ f["Z0"].T + f["zb0"], eps)
        g = ln_relu(g @ f["Z1"].T + f["zb1"], eps)
        return g @ f["Zm"].T + f["zbm"]

    def profile(z):
        g = ln_relu(z @ f["D0"].T + f["db0"], eps)
        g = ln_relu(g @ f["D1"].T + f["db1"], eps)
        return softmax(g @ f["Ds"].T + f["dbs"])
    m, mk = zmean(x / x.mean()), zmean(xn / xn.mean())
    s, sk = profile(m), profile(mk)
    pi = softmax(logit - np.minimum(np.sqrt(d2 / float(f["median_distance"])), float(f["clamp"])))
    L = x.sum()
    mu_own, mu_nb, mu_bg = rho[0] * L * s, rho[1] * L * (pi[:, None] * sk).sum(0), rho[2] * L * f["background"]
    split = np.where(x > 0, mu_nb / (mu_own + mu_nb + mu_bg), 0.0)
    return {"z": np.concatenate([m[None], mk]), "s": s, "pi": pi, "split": split, "rho": rho}


def model_counts(folder):
    """The run's cells at the start (the prior's labels) over the model's genes: the learning table stand-in."""
    raw = counted(folder, np.stack([prior_plane(), prior_upper_layer()]))       # write_run's dense prior
    index = W.gene_index({"genes": {"genes": MODEL_GENES}}, RUN_GENES)[0]
    out = np.zeros((NCELLS, len(MODEL_GENES)), np.float32)
    out[:, index[index >= 0]] = raw[:, index >= 0]
    return out


def make_deep(folder: Path, run_folder: Path, seed=0, hidden=16, hidden_d=8, n_latent=4, st_cells=4, **changes):
    """A model folder like analysis/qc_trainer.py's: encoder.npz (make_model), deep.npz (DEEP_KEYS) and cells.parquet.
    Learned cells: the run's cells but UNLEARNED, plus EXTRA; training neighbours: the 10 nearest learned cells by
    their starting centres; the self-test outputs from `reference`. `changes` replace arrays (for refusals)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder.mkdir(parents=True, exist_ok=True)
    make_model(folder / "encoder.npz", MODEL_GENES, seed=seed)
    enc = W.load_encoder(folder / "encoder.npz")
    rng = np.random.default_rng(seed + 100)
    G = len(MODEL_GENES)
    learned = [c for c in RUN_IDS if c not in UNLEARNED] + EXTRA
    j, i = np.mgrid[0:NY, 0:NX]
    plane = prior_plane()
    centre = {RUN_IDS[c]: (X0 + VOXEL * (i[plane == c + 1].mean() + 0.5), Y0 + VOXEL * (j[plane == c + 1].mean() + 0.5))
              for c in range(NCELLS)}
    centre.update({x: (X0 + rng.uniform(0, NX * VOXEL), Y0 + rng.uniform(0, NY * VOXEL)) for x in EXTRA})
    xy = np.array([centre[c] for c in learned])
    run_counts = model_counts(run_folder)
    counts = np.array([run_counts[RUN_IDS.index(c)] if c in RUN_IDS else rng.poisson(30, G) for c in learned],
                      np.float32)
    d2_all = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2_all, np.inf)
    train_index = np.argsort(d2_all, axis=1, kind="stable")[:, :K].astype(np.int32)
    train_d2 = np.take_along_axis(d2_all, train_index, axis=1).astype(np.float32)
    N = len(learned)
    f32 = lambda *shape: rng.normal(0, 0.5, shape).astype(np.float32)                       # noqa: E731
    deep = {"format": np.array("proseg-live-mixing-deep/1"), "genes": np.array(MODEL_GENES),
            "Z0": f32(hidden, G), "zb0": f32(hidden), "Z1": f32(hidden, hidden), "zb1": f32(hidden),
            "Zm": f32(n_latent, hidden), "zbm": f32(n_latent), "Zv": f32(n_latent, hidden), "zbv": f32(n_latent),
            "D0": f32(hidden_d, n_latent), "db0": f32(hidden_d), "D1": f32(hidden_d, hidden_d), "db1": f32(hidden_d),
            "Ds": (rng.normal(0, 2.0, (G, hidden_d))).astype(np.float32), "dbs": f32(G),
            "ln_eps": np.float32(1e-5), "z_var_eps": np.float32(1e-4),
            "background": softmax(rng.normal(0, 0.1, G)).astype(np.float32),
            "typical": (counts / counts.sum(1, keepdims=True)).mean(0).astype(np.float32),
            "median_distance": np.float32(60.0), "clamp": np.float32(10.0), "diffuse_scale": np.float32(6.0),
            "train_logit": rng.uniform(-6, 6, (N, K)).astype(np.float32), "train_index": train_index}
    deep["background"] /= deep["background"].sum(dtype=np.float64)
    deep["typical"] /= deep["typical"].sum(dtype=np.float64)
    deep["rank_logit"] = deep["train_logit"].mean(axis=0, dtype=np.float64).astype(np.float32)
    deep["train_ids_sha256"] = np.array(hashlib.sha256("\n".join(learned).encode()).hexdigest())
    rows = np.arange(min(st_cells, N), dtype=np.int32)
    st = np.concatenate([np.concatenate([[r], train_index[r]]) for r in rows])
    block = counts[st]
    nz_r, nz_c = np.nonzero(block)
    T = min(5, G)
    outs = [reference(enc, deep, counts[r], counts[train_index[r]], train_d2[r].astype(np.float64),
                      deep["train_logit"][r].astype(np.float64)) for r in rows]
    genes = np.stack([np.lexsort((np.arange(G), -(counts[r] * o["split"])))[:T] for r, o in zip(rows, outs)])
    deep.update(st_rows=rows, st_indptr=np.concatenate([[0], np.cumsum(np.bincount(nz_r, minlength=len(st)))]).astype(
                    np.int64),
                st_indices=nz_c.astype(np.int32), st_data=block[nz_r, nz_c].astype(np.float32),
                st_d2=train_d2[rows], st_z=np.stack([o["z"] for o in outs]).astype(np.float32),
                st_pi=np.stack([o["pi"] for o in outs]).astype(np.float32), st_own_rows=np.array([0, 1], np.int32),
                st_own=np.stack([outs[0]["s"], outs[1]["s"]]).astype(np.float32), st_genes=genes.astype(np.int32),
                st_nb_split=np.stack([o["split"][g] for o, g in zip(outs, genes)]).astype(np.float32))
    deep.update(changes)
    np.savez(folder / "deep.npz", **deep)
    pq.write_table(pa.table({"cell_id": learned, "x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32),
                             "counts": counts.sum(1).astype(np.int32)}), folder / "cells.parquet")
    return folder


@pytest.fixture
def setup(tmp_path):
    """A run (write_run, detail) with its cells.txt, a model folder with details, a numpy worker and the requests."""
    run = write_run(tmp_path / "run")
    live = run / "live"
    (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    model = make_deep(tmp_path / "model", run)
    info = json.loads((live / "info.json").read_text())
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    out = tmp_path / "qc"
    out.mkdir()
    common = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    qc = {"model": str(model), "model_id": "feedface", "mixed_at": 0.3, "judge_min": JUDGE, "fit_min": 5,
          "grid": {"x0": X0, "y0": Y0, "bin": 10.0, "nx": 4, "ny": 3}}
    start = {"op": "qc_start", **common, "seq": None, **counts_files(live, 0, start=False),
             "prior": {**info["prior"], "file": str(live / info["prior"]["file"])},
             "qc": {**qc, "out_cells": str(out / "start.bin"), "out_start": str(out / "start.npz"),
                    "out_bins": str(out / "start.bins.npz")}}
    frame_req = {"op": "analyze", "ops": ["qc"], **common, "seq": 2, "iter": 2, **counts_files(live, 2),
                 "frame": str(live / frame["file"]), "format": "dense",
                 **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                 "cells_file": str(live / frame["cells_file"]),
                 "qc": {**qc, "start": str(out / "start.npz"), "out_cells": str(out / "000002.qc.bin"),
                        "out_bins": str(out / "000002.bins.npz")}}
    deeper = {"model": str(model), "model_id": "feedface", "start": False, "k": K, "fit_min": 5, "judge_min": JUDGE,
              "codes": str(out / "000002.qc.bin"), "cell_ids": str(live / "cells.txt"), "v": 1}
    return {"run": run, "live": live, "model": model, "out": out, "start": start, "frame": frame_req, "deeper": deeper,
            "common": common, "worker": W.Worker(xp=np)}


def codes_of(path):
    from numcodecs import Zstd
    return np.frombuffer(Zstd().decode(Path(path).read_bytes()), np.uint8)


# ---- deep.npz: strict reading and the self-test --------------------------------------------------------------------
def test_deep_exports_are_read_strictly(tmp_path):
    run = write_run(tmp_path / "run")
    good = dict(np.load(make_deep(tmp_path / "m", run) / "deep.npz"))
    deep = W.load_deep(tmp_path / "m", MODEL_GENES)
    assert deep["genes"] == MODEL_GENES and deep["Z0"].dtype == np.float32 and len(deep["train_ids_sha256"]) == 64

    def refused(problem, names=MODEL_GENES, **arrays):
        np.savez(tmp_path / "bad.npz", **arrays)
        with pytest.raises(ValueError, match=problem):
            W.load_deep(tmp_path / "bad.npz", names)
    without = {k: v for k, v in good.items() if k != "Zv"}
    refused("missing \\['Zv'\\]", **without)
    refused("unknown \\['bn_running_mean'\\]", **good, bn_running_mean=np.zeros(16, np.float32))   # BatchNorm
    refused("unknown \\['ln_weight'\\]", **good, ln_weight=np.ones(16, np.float32))                # affine LayerNorm
    refused("unknown format", **{**good, "format": np.array("proseg-live-mixing-deep/2")})
    refused("genes differ", names=MODEL_GENES[::-1], **good)                                        # another gene order
    refused("D0 is", **{**good, "D0": np.zeros((8, 5), np.float32)})                               # a covariate column
    refused("Z1 is float64", **{**good, "Z1": good["Z1"].astype(np.float64)})
    refused("Ds is", **{**good, "Ds": np.full_like(good["Ds"], np.nan)})
    refused("outside \\(-10, 10\\)", **{**good, "train_logit": good["train_logit"] * 3})
    refused("rank_logit", **{**good, "rank_logit": good["rank_logit"] + 0.01})
    refused("out of range", **{**good, "train_index": good["train_index"] + 100})
    refused("not a profile", **{**good, "background": good["background"] * 2})
    refused("malformed self-test counts", **{**good, "st_indices": good["st_indices"] + 100})
    refused("not a sha256", **{**good, "train_ids_sha256": np.array("abc")})


def test_the_trainers_details_format_is_the_workers(tmp_path):
    """One format, two readers: analysis/qc_trainer.py writes deep.npz (DEEP_FORMAT, DEEP_KEYS) and reads it strictly
    for its own self-test; the worker reads the same file with its own loader."""
    from sopa_companion.analysis import qc_trainer as T
    assert T.DEEP_FORMAT in W.DEEP_FORMATS and set(T.DEEP_KEYS) == set(W.DEEP_KEYS)
    folder = make_deep(tmp_path / "m", write_run(tmp_path / "run"))
    theirs = T.load_deep(folder / "deep.npz", MODEL_GENES,          # the calculator as the trainer's reader names it
                         enc={**W.load_encoder(folder / "encoder.npz"), "genes": MODEL_GENES})
    ours = W.load_deep(folder / "deep.npz", MODEL_GENES)
    assert all(np.array_equal(np.asarray(theirs[k]), np.asarray(ours[k])) for k in ("Z0", "Ds", "train_logit",
                                                                                    "train_index", "st_nb_split"))
    assert str(theirs["train_ids_sha256"]) == ours["train_ids_sha256"]


def test_the_worker_checks_the_details_against_their_own_reference_cells(tmp_path):
    run = write_run(tmp_path / "run")
    folder = make_deep(tmp_path / "m", run)
    worker = W.Worker(xp=np)
    model = worker.deep_model(folder)
    assert model["selftest"]["max"] <= W.DEEP_SELFTEST_TOLERANCE and model["selftest"]["max"] < 1e-5   # float32 here
    assert worker.deep_model(folder / "encoder.npz") is model                     # cached; a folder or its encoder
    assert model["rows"]["x2"] == len(RUN_IDS) - len(UNLEARNED) + 2 and "c03" not in model["rows"]
    # Weights that are not the ones the outputs came from fail the self-test (a damaged export).
    worker.stat_ttl_s = 0
    stored = dict(np.load(folder / "deep.npz"))
    np.savez(folder / "deep.npz", **{**stored, "Z1": stored["Z1"] * 1.01})
    with pytest.raises(ValueError, match="failed their self-test"):
        worker.deep_model(folder)
    np.savez(folder / "deep.npz", **stored)
    # cells.parquet must be the learned cells deep.npz names (their ids' sha256).
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = pq.read_table(folder / "cells.parquet").column("cell_id").to_pylist()
    pq.write_table(pa.table({"cell_id": ids[::-1]}), folder / "cells.parquet")
    with pytest.raises(ValueError, match="cells.parquet does not match"):
        worker.deep_model(folder)
    pq.write_table(pa.table({"cell_id": ids}), folder / "cells.parquet")
    assert worker.deep_model(folder)["rows"] == model["rows"]
    # A model learned before Go deeper (encoder.npz alone) says what is missing.
    (folder / "deep.npz").unlink()
    with pytest.raises(ValueError, match=r"no details for Go deeper \(deep.npz is missing\): learn the details"):
        worker.deep_model(folder)
    # At most MODELS_KEPT models' details stay in memory.
    for n in range(W.MODELS_KEPT + 1):
        worker.deep_model(make_deep(tmp_path / f"m{n}", run, seed=n))
    assert len(worker.deeps) == W.MODELS_KEPT and str(tmp_path / "m0" / "deep.npz") not in worker.deeps


def test_the_recipe_takes_one_blas_thread_where_it_can():
    """threadpoolctl (the RAPIDS env's) limits numpy's BLAS to one thread around the recipe; without it, a no-op."""
    with W.one_blas_thread():
        assert float((np.ones((11, 64), np.float32) @ np.ones((64, 8), np.float32)).sum()) == 11 * 64 * 8


def test_a_file_replaced_in_place_is_seen_within_the_stat_interval(tmp_path):
    path = tmp_path / "f"
    path.write_text("a")
    worker = W.Worker(xp=np)
    assert worker.stat_pool is None, "the numpy worker asks the disk at once"
    first = worker.stat_ns(path)
    os.utime(path, ns=(first + 10 ** 9, first + 10 ** 9))
    assert worker.stat_ns(path) == first                       # within stat_ttl_s: the disk is not asked again
    worker.stat_ttl_s = 0
    assert worker.stat_ns(path) == first + 10 ** 9


def test_an_old_lookup_is_renewed_on_the_stat_thread(tmp_path):
    """With a stat thread (the GPU worker) an answer never waits on the disk for a file it looked up before: an old
    lookup is renewed there while the answer goes on with the value it has; the next answer sees the new one, and a file
    gone raises FileNotFoundError as the disk does. A file never looked up is asked at once."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "f"
    path.write_text("a")
    worker = W.Worker(xp=np)
    worker.stat_pool = ThreadPoolExecutor(1)
    settle = lambda: worker.stat_pool.submit(lambda: None).result(timeout=5)   # noqa: E731 (the one thread, in order)
    try:
        first = worker.stat_ns(path)
        os.utime(path, ns=(first + 10 ** 9, first + 10 ** 9))
        worker.stat_ttl_s = 0
        asked, gate = [], threading.Event()
        real = worker._restat
        worker._restat = lambda key: (gate.wait(5), asked.append(key), real(key))
        assert worker.stat_ns(path) == first, "renewed in the background: the old value now"
        assert worker.stat_ns(path) == first and len(worker.stat_pending) == 1, "asked once"
        gate.set()
        settle()
        assert asked == [str(path)] and not worker.stat_pending
        # The next answer renews the lookup again: held until the file is gone, or it could find the file still there.
        unlinked = threading.Event()
        worker._restat = lambda key: (unlinked.wait(5), real(key))
        assert worker.stat_ns(path) == first + 10 ** 9, "the next answer sees the new one"   # (and renews it again)
        path.unlink()
        unlinked.set()
        settle()                                               # that renewal finds the file gone
        with pytest.raises(FileNotFoundError):
            worker.stat_ns(path)
        with pytest.raises(FileNotFoundError):
            worker.stat_ns(tmp_path / "never")                 # never looked up: asked at once
    finally:
        worker.stat_pool.shutdown(wait=True)


# ---- the recipe -----------------------------------------------------------------------------------------------------
def test_the_recipe_matches_the_reference_and_scales_the_split_to_the_cards_parts(setup):
    model = setup["worker"].deep_model(setup["model"])
    deep, enc = model["deep"], model["encoder"]
    counts = model_counts(setup["run"])
    rng = np.random.default_rng(1)
    for c in range(8):                                         # the judged cells of the start line
        others = np.array([o for o in range(NCELLS) if o != c and counts[o].sum() >= 5])[:K]
        x, xn = counts[c], counts[others]
        d2 = rng.uniform(5, 400, len(others)).astype(np.float32)
        logit = rng.uniform(-5, 5, len(others)).astype(np.float32)
        r = W.deep_forward(np, deep, x, xn, d2, logit, W.deep_calculator(enc, x))
        ref = reference(enc, deep, x.astype(np.float64), xn.astype(np.float64), d2.astype(np.float64),
                        logit.astype(np.float64))
        assert np.abs(np.concatenate([r["m"][None], r["mk"]]) - ref["z"]).max() < 1e-4
        assert np.abs(r["s"] - ref["s"]).max() < 1e-5 and np.abs(r["pi"] - ref["pi"]).max() < 1e-5
        assert np.abs(r["A_raw"][:, 1] / r["x_nz"] - ref["split"][r["genes_nz"]]).max() < 1e-4
        assert np.abs(r["parts"] - ref["rho"] / ref["rho"].sum()).max() < 1e-5
        # Step 9: every gene's row adds up to its transcripts, every part to the card's share of the cell.
        np.testing.assert_allclose(r["A"].sum(axis=1), r["x_nz"], rtol=1e-9)
        assert r["scale_col_err"] <= 1e-3 or r["scale_rounds"] == W.DEEP_RULES["scale_cap"]
        np.testing.assert_allclose(r["A"].sum(axis=0), x.sum() * r["parts"].astype(np.float64), atol=1e-3)
        np.testing.assert_allclose(r["Ak"].sum(axis=0), r["A"][:, 1], rtol=1e-6)      # step 10 (float32 shares)
        assert abs(r["tx_k"].sum() - r["A"][:, 1].sum()) < 1e-5 and r["rough"] == (r["raw_gap"] > 0.10)
        # Step 12 and 13 exactly as stated.
        xs, A = r["x_nz"].astype(np.float64), r["A"]
        giver = r["Ak"].argmax(axis=0)
        expect_tag = (xs >= 2) & (A[:, 1] / xs >= 0.5) & ~(r["similar"][giver] >= 0.9) & \
            (r["sk"][giver, r["genes_nz"]] >= 4 * r["s"][r["genes_nz"]])
        assert np.array_equal(r["tag"], expect_tag)
        assert np.array_equal(r["own_ok"], (xs >= 2) & (A[:, 0] / xs >= 0.5))


def test_neighbours_are_the_nearest_candidates_with_ties_by_index():
    rng = np.random.default_rng(4)
    n = 400
    centres = rng.integers(0, 30, (n, 2)).astype(np.float64)      # many exact ties
    centres[rng.random(n) < 0.05] = np.nan                         # cells without a position
    totals = rng.integers(0, 60, n).astype(np.float64)              # candidates hold >= 20
    frame = W.DeepFrame(totals, centres, 20, K)
    for c in range(0, n, 7):
        cells, d2 = frame.neighbours(c)
        ok = [j for j in range(n) if j != c and totals[j] >= 20 and np.isfinite(centres[j]).all()]
        dist = {j: float(((centres[j] - centres[c]) ** 2).sum()) for j in ok}
        expect = sorted((j for j in ok if np.isfinite(dist[j])), key=lambda j: (dist[j], j))[:K]
        assert cells.tolist() == expect, c
        assert np.allclose(d2, [dist[j] for j in expect])
    assert W.DeepFrame(totals[:3], centres[:3], 1e9).neighbours(0)[0].size == 0     # nobody to compare with


def test_big_frames_look_near_the_cell_first_with_the_same_answer():
    """Past DEEP_NEAR_MIN candidates the search looks within DEEP_NEAR_UM first (whole samples): the same k cells and
    distances as over all candidates, ties by index included, and the full search where too few lie that close."""
    rng = np.random.default_rng(11)
    n = W.DEEP_NEAR_MIN * 2
    dense = rng.integers(0, 900, (n, 2)).astype(np.float64)           # ~5 um apart: many cells near, exact ties
    sparse = rng.uniform(0, 40000, (n, 2))                             # ~900 um apart: nobody within 60 um
    for centres in (dense, sparse):
        centres = centres.copy()
        centres[rng.random(n) < 0.03] = np.nan
        totals = rng.integers(0, 60, n).astype(np.float64)
        frame = W.DeepFrame(totals, centres, 20, K)
        ok = np.flatnonzero((totals >= 20) & np.isfinite(centres).all(axis=1))
        for c in rng.choice(n, 25, replace=False):
            if not np.isfinite(centres[c]).all():
                continue
            cells, d2 = frame.neighbours(int(c))
            dist = ((centres[ok] - centres[c]) ** 2).sum(axis=1)
            keep = ok != c
            order = np.lexsort((ok[keep], dist[keep]))[:K]
            assert cells.tolist() == ok[keep][order].tolist(), int(c)
            assert np.array_equal(d2, dist[keep][order])


def test_kind_words_of_many_profiles_at_once_are_each_profiles_own():
    """deep_kind_words_rows gives each row the words the gene-by-gene rule gives it: top genes by s ln(s / typical)
    among genes with s >= 2 typical, ties by gene index, fewer when fewer qualify."""
    rng = np.random.default_rng(5)
    G = 300
    typical = rng.dirichlet(np.ones(G)).astype(np.float32)
    rows = rng.dirichlet(np.full(G, 0.3), size=6).astype(np.float32)
    rows[1] = typical                                                  # nothing qualifies
    rows[2, :] = 0.0
    rows[2, [7, 3, 9]] = typical[[7, 3, 9]] * 2.0                      # exactly at the bar: these three qualify
    deep = {"typical": typical}

    def reference(s, n):                                               # the rule, written out over all genes
        score = np.where(s >= 2.0 * typical, s * np.log(np.maximum(s, 1e-30) / np.maximum(typical, 1e-30)), -np.inf)
        score = score.astype(np.float64)
        good = [g for g in range(G) if np.isfinite(score[g])]
        return sorted(good, key=lambda g: (-score[g], g))[:n]
    ns = (4, 3, 3, 3, 4, 2)
    got = W.deep_kind_words_rows(np, deep, rows, ns)
    assert got == [reference(rows[r], ns[r]) for r in range(len(ns))]
    assert got[1] == [] and sorted(got[2]) == [3, 7, 9]
    assert W.deep_kind_words(np, deep, rows[0], 4) == got[0]


def test_gene_names_reads_a_models_list_and_other_sequences():
    names = ["APOB", "AFM", "C4B", "HPD"]
    assert W.gene_names(names, np.array([2, 0, 3])) == ["C4B", "APOB", "HPD"]
    assert W.gene_names(names, [1]) == ["AFM"]                         # cached as all-str: the same answer again
    assert W.gene_names(np.array(names), np.array([3, 1])) == ["HPD", "AFM"]
    assert W.gene_names(["A", 5, "C"], np.array([1, 2])) == ["5", "C"]  # not all str: each name as str


def test_learned_values_carry_over_by_id_and_new_pairs_get_their_ranks_mean():
    deep = {"train_index": np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], np.int32),
            "train_logit": np.array([[5, -1, 2], [1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)}
    deep["rank_logit"] = deep["train_logit"].mean(axis=0).astype(np.float32)
    run_row = np.array([0, 2, -1, 3, 1])                         # run cell -> training row (-1: not learned)
    logits, known = W.learned_logits(run_row, deep, 0, np.array([3, 1, 2]))
    # run cell 3 = training row 3 (slot 2 of row 0: 2.0), run cell 1 = row 2 (slot 1: -1.0), run cell 2 never learned:
    # the mean of rank 2 (the third neighbour).
    assert logits.tolist() == [2.0, -1.0, pytest.approx(float(deep["rank_logit"][2]))]
    assert known.tolist() == [True, True, False]
    assert W.learned_logits(run_row, deep, 2, np.array([0, 1]))[0] is None              # MAP for an unlearned cell


def test_cells_the_model_never_learned_get_map_shares(setup):
    model = setup["worker"].deep_model(setup["model"])
    deep, enc = model["deep"], model["encoder"]
    counts = model_counts(setup["run"])
    x, xn = counts[3], counts[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]
    d2 = np.linspace(30, 900, 10).astype(np.float32)
    rho = W.deep_calculator(enc, x)
    r = W.deep_forward(np, deep, x, xn, d2, None, rho)
    assert r["pi"].sum() == pytest.approx(1.0, abs=1e-5) and (r["pi"] > 0).all()
    more = W.deep_map(np, deep, x, float(x.sum()), rho, r["s"], r["sk"], d2, rounds=400)
    assert np.abs(more - r["pi"]).max() < 1e-3                   # 100 rounds are at the fixed point
    near = W.deep_map(np, deep, x, float(x.sum()), rho, r["s"], r["sk"], d2, rounds=0)
    assert near.argmax() == 0                                    # it starts from distance alone (nearest first)


def test_the_result_stays_under_its_size_cap():
    genes = [[f"GENE{i:05d}", 1 if i > 50 else 9, 0.4, 0.5, 0] for i in range(3000)]
    result = {"v": 1, "state": "ready", "genes": genes, "genes_total": 3000}
    fitted = W.fit_reply(json.loads(json.dumps(result)))
    size = len(json.dumps(fitted, separators=(",", ":")))
    assert size <= W.DEEP_REPLY_MAX - W.DEEP_REPLY_ROOM and fitted["genes"] == genes[:len(fitted["genes"])]
    rest = genes[len(fitted["genes"]):]
    assert fitted["genes_rest"] == [len(rest), len(rest), round(0.4 * len(rest), 1), round(0.5 * len(rest), 1)]
    small = {"v": 1, "genes": genes[:20]}
    assert W.fit_reply(json.loads(json.dumps(small))) == small and "genes_rest" not in small


def test_a_rows_size_is_worked_out_as_json_writes_it():
    """fit_reply measures only the rows it drops, by arithmetic for the usual plain row (a big cell's reply ~2 ms
    faster): always the length json.dumps gives, also for names JSON escapes and rows of other shapes."""
    rows = [["APOB", 39, 7.9, 31.1, 1], ["G0", 1, 0.0, 0.0, 0], ["X", 1234, 1033.4, 199.9, 0], ["NEG", 2, -0.0, 2.0, 0],
            ['A"B', 3, 1.0, 2.0, 0], ["C\\D", 3, 1.0, 2.0, 0], ["ÄÖ", 3, 1.0, 2.0, 1], ["T\tX", 1, 0.5, 0.5, 0],
            ["F", 1, float("nan"), 0.0, 0], ["S", 1, 0.1, 0.2], ["B", True, 0.1, 0.2, 0], ("T", 1, 0.1, 0.2, 0)]
    rng = np.random.default_rng(2)
    rows += [[f"G{i}", int(t), float(np.round(o, 1)), float(np.round(b, 1)), int(f)]
             for i, (t, o, b, f) in enumerate(zip(rng.integers(1, 5000, 500), rng.random(500) * 900,
                                                 rng.random(500) * 90, rng.integers(0, 2, 500)))]
    for row in rows:
        assert W._row_bytes(row) == len(json.dumps(row, separators=(",", ":"))), row


def test_rows_built_where_they_fit_and_partial_sorts_give_the_full_routes_results():
    """explain_cell builds only the gene rows that fit (lengths from arrays) and finds a neighbour's top genes with a
    partial sort: always what building every row and fit_reply, or a full lexsort, give (odd names, ties, tiny caps)."""
    rng = np.random.default_rng(11)
    names = np.array([f"G{i}" for i in range(3000)] + ['A"B', "C\\D", "ÄÖ", "T\tX"])
    for trial in range(400):
        n = int(rng.choice([0, 1, 2, 40, 600, 2500]))
        g_idx = rng.choice(len(names), n, replace=False) if n else np.zeros(0, np.int64)
        tx = rng.integers(1, int(rng.choice([3, 60, 4000])), n).astype(np.int64)
        own = np.round(tx * rng.random(n), 1)
        nb = np.round((tx - own) * rng.random(n), 1)
        tag = rng.integers(0, 2, n).astype(np.int64)
        base = {"v": 1, "state": "ready", "pad": "x" * int(rng.choice([0, 31000])), "genes": [], "genes_total": n}
        cap = int(rng.choice([W.DEEP_REPLY_MAX - W.DEEP_REPLY_ROOM, 900, 120]))
        full = {**base, "genes": [list(t) for t in zip([str(names[g]) for g in g_idx], tx.tolist(), own.tolist(),
                                                        nb.tolist(), tag.tolist())]}
        expected = W.fit_reply(json.loads(json.dumps(full)), cap)
        assert json.dumps(W.fitted_gene_rows(dict(base), names, g_idx, tx, own, nb, tag, cap)) == json.dumps(expected)
        values = np.round(rng.random(n) * 10, int(rng.choice([0, 2])))
        keys = rng.permutation(max(n, 1))[:n]
        for k in (1, 4):
            assert np.array_equal(W.top_order(values, keys, k), np.lexsort((keys, -values))[:k])


def test_the_neighbours_top_genes_in_one_pass_are_top_orders_own():
    """top_orders (one partition for a cell's K neighbours) gives each row exactly what top_order gives it: rows with
    ties inside the top, at its edge, fewer values than asked for, and a value that is not finite."""
    rng = np.random.default_rng(23)
    for trial in range(300):
        R, n = int(rng.integers(0, 12)), int(rng.choice([1, 3, 4, 5, 40, 900]))
        values = np.round(rng.random((R, n)) * 10, int(rng.choice([0, 1, 3])))   # 0 decimals: many ties
        if trial % 37 == 0 and R and n:
            values[int(rng.integers(R)), int(rng.integers(n))] = np.nan
        keys = rng.permutation(4 * n)[:n].astype(np.int64)
        for k in (1, 4):
            got = W.top_orders(values, keys, k)
            assert len(got) == R
            for r in range(R):
                assert np.array_equal(got[r], W.top_order(values[r], keys, k)), (trial, r, k)


# ---- the op in the worker -------------------------------------------------------------------------------------------
READY_KEYS = {"v", "run", "seq", "iter", "cell", "name", "model", "count", "state", "transcripts", "outside", "code",
              "parts", "rough", "genes", "genes_total", "kind", "own_genes", "same_kind_share", "neighbours", "learned",
              "gap", "ms", "worker_ms"}


def check_ready(result, codes, ncells=NCELLS):
    # load_ms: the model's details loaded (or the start line counted) for this answer: once, apart from worker_ms.
    assert set(result) - {"load_ms", "start_change"} == READY_KEYS, set(result) ^ READY_KEYS
    assert "load_ms" not in result or result["load_ms"] >= 0
    c = result["cell"]
    assert result["state"] == "ready" and result["name"] == RUN_IDS[c] and result["code"] == int(codes[c])
    assert sum(result["parts"]) == pytest.approx(1.0, abs=2e-4)
    L = result["transcripts"]                                         # proseg's count: fractional, exact (32nds)
    n = len(result["genes"])
    assert isinstance(L, float) and isinstance(result["outside"], float)
    # Exact, in the counts file's 32nds (to 0.1, a cell at x.46875 read x + 1 in the window and x on the stage).
    assert (L * 32).is_integer() and (result["outside"] * 32).is_integer(), (L, result["outside"])
    assert abs(sum(g[1] for g in result["genes"]) + result.get("genes_rest", [0, 0])[1] - L) <= 0.05 * (n + 2)
    for row in result["genes"]:                                       # each gene's split adds up (0.1 rounding)
        gene, tx, own, nb, flag = row[:5]
        assert len(row) in (5, 6) and gene in MODEL_GENES and tx > 0 and own >= 0 and nb >= 0 and own + nb <= tx + 0.15
        assert flag in (0, 1) and isinstance(tx, float)
        if len(row) == 6:                                             # its main giver's row, from 0.5 on
            assert 0 <= row[5] < len(result["neighbours"]) and nb >= W.DEEP_GIVER_MIN - 0.1
    tagged = sum(1 for g in result["genes"] if g[4] == 1)             # a neighbour's gene: rounded down (D3)
    assert abs(sum(g[2] for g in result["genes"]) - L * result["parts"][0]) <= 0.05 * n + 0.01 + L * 1e-4
    assert abs(sum(g[3] for g in result["genes"]) - L * result["parts"][1]) <= 0.05 * n + 0.05 * tagged + 0.01 + L * 1e-4
    assert [g[1] for g in result["genes"]] == sorted((g[1] for g in result["genes"]), reverse=True)
    rows = result["neighbours"]
    assert 1 <= len(rows) <= K and [r["tx"] for r in rows] == sorted((r["tx"] for r in rows), reverse=True)
    assert all(0 <= r["cell"] < ncells and r["cell"] != c and r["name"] == RUN_IDS[r["cell"]] for r in rows)
    assert sum(r["tx"] for r in rows) == pytest.approx(L * result["parts"][1], abs=0.05 * len(rows) + 0.01 + L * 1e-4)
    assert all(len(r["genes"]) <= W.DEEP_SOURCE_GENES and -1 <= r["similar"] <= 1 for r in rows)
    learned = "no" if RUN_IDS[c] in UNLEARNED else ("yes" if all(r["known"] for r in rows) else "partly")
    assert result["learned"] == learned and 0 <= result["same_kind_share"] <= 1
    assert len(json.dumps(result, separators=(",", ":"))) <= W.DEEP_REPLY_MAX


def test_go_deeper_rides_the_check_and_shows_the_cards_byte(setup):
    worker, out = setup["worker"], setup["out"]
    assert worker.handle(setup["start"])["ok"]
    judged = None
    for c in range(NCELLS):
        reply = worker.handle({**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], "cell": c}})
        assert reply["ok"], reply
        codes = codes_of(out / "000002.qc.bin")                  # written in the same pass
        result = reply["deeper"]
        assert result["v"] == 1 and result["run"] == "r1" and result["seq"] == 2 and result["iter"] == 2
        assert result["model"] == "feedface" and result["cell"] == c and result["worker_ms"] >= result["ms"] >= 0
        if codes[c] != W.NOT_JUDGED:
            check_ready(result, codes)
            judged = judged if judged is not None else c
            # Alone, from the frame's stored byte: the same answer (only the timings differ).
            alone = worker.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c}})
            assert alone["ok"] and "qc" not in alone
            assert {k: v for k, v in alone["deeper"].items() if k not in ("ms", "worker_ms", "load_ms")} == \
                {k: v for k, v in result.items() if k not in ("ms", "worker_ms", "load_ms")}
        else:
            assert result["state"] in ("too_few", "empty")
    learned = {worker.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c}})[
        "deeper"].get("learned") for c in range(8)}
    assert {"no", "partly"} <= learned or {"no", "yes"} <= learned          # the unlearned cell c03 is re-estimated
    # The states: cell 12 vanished in frame 2 (no transcripts), a bottom-row cell holds too few to split.
    empty = worker.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": 11}})["deeper"]
    assert empty["state"] == "empty" and empty["transcripts"] == 0 and empty["name"] == "c11"
    few = next(r for r in (worker.handle({**setup["frame"], "ops": ["deeper"],
                                          "deeper": {**setup["deeper"], "cell": c}})["deeper"] for c in range(8, 11))
               if r["state"] == "too_few")
    assert 0 < few["transcripts"] < JUDGE and 1 <= len(few["genes"]) <= W.DEEP_TOO_FEW_GENES
    assert sum(g[1] for g in few["genes"]) <= few["transcripts"] + 0.05 * len(few["genes"])
    assert set(few) >= {"genes", "outside", "name"} and all(isinstance(g[1], float) for g in few["genes"])
    # A cell that is not the run's, or a model without details, fails Go deeper alone: the check still answers.
    for block, problem in (({"cell": NCELLS}, "no cell 12"), ({"cell": True}, "no cell True"),
                           ({"cell": 0, "model": str(setup["out"])}, "No such file|no details")):
        reply = worker.handle({**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], **block}})
        assert not reply["ok"] and "qc" in reply and set(reply["errors"]) == {"deeper"}
        assert pytest.importorskip("re").search(problem, reply["errors"]["deeper"]), reply["errors"]
    assert judged is not None


def test_go_deeper_on_the_start_line_reads_the_init_frames_count_as_the_start_line_does(setup):
    """Go deeper on the start line reads proseg's count of the init frame (frame 0's counts file), as the start line
    does: the same transcripts per cell; its centres and bytes from start.npz, else the prior's footprints."""
    worker, out = setup["worker"], setup["out"]
    assert worker.handle(setup["start"])["ok"]
    stored = np.load(out / "start.npz")
    request = {"op": "deeper", **setup["common"], "seq": "start", **counts_files(setup["live"], 0, start=False),
               "deeper": {**setup["deeper"], "start": True, "codes": str(out / "start.bin")}}
    named = {**request, "op": "analyze", "ops": ["deeper"], "seq": None,       # as the scheduler sends it
             "prior": setup["start"]["prior"], "deeper": {**request["deeper"], "start_file": str(out / "start.npz")}}
    for c in range(NCELLS):
        reply = W.Worker(xp=np).handle({**request, "deeper": {**request["deeper"], "cell": c}})   # a fresh worker
        assert reply["ok"], reply
        result = reply["deeper"]
        assert result["seq"] == "start" and result["iter"] is None
        again = worker.handle({**named, "deeper": {**named["deeper"], "cell": c}})
        assert again["ok"] and "qc" not in again and again["seq"] is None
        assert {k: v for k, v in again["deeper"].items() if "ms" not in k} == \
            {k: v for k, v in result.items() if "ms" not in k}
        assert result.get("transcripts", 0) == round(float(stored["totals"][c]), 5)     # exactly qc_start's count
        if stored["code"][c] != W.NOT_JUDGED:
            check_ready(result, stored["code"])
    # Without start.npz the prior footprint gives the centres and there is no stored byte: the same split.
    c = int(np.nonzero(stored["code"] != W.NOT_JUDGED)[0][0])
    first = worker.handle({**request, "deeper": {**request["deeper"], "cell": c}})["deeper"]
    # The start line's frame is worked out once per run and model (load_ms), then kept: the same answer, no load.
    kept = worker.handle({**request, "deeper": {**request["deeper"], "cell": c}})["deeper"]
    assert "load_ms" not in kept and {k: v for k, v in kept.items() if "ms" not in k} == \
        {k: v for k, v in first.items() if "ms" not in k}
    fresh = W.Worker(xp=np)
    loaded = fresh.handle({**request, "deeper": {**request["deeper"], "cell": c}})["deeper"]
    assert loaded["load_ms"] > 0 and loaded["worker_ms"] >= 0
    (out / "start.npz").unlink()
    again = W.Worker(xp=np).handle({**request, "deeper": {**request["deeper"], "cell": c}})["deeper"]
    assert {k: v for k, v in again.items() if k not in ("ms", "worker_ms", "load_ms", "code")} == \
        {k: v for k, v in first.items() if k not in ("ms", "worker_ms", "load_ms", "code")}
    # Riding on a frame request is no different: it reads the init frame's count, the frame's file is the check's.
    both = worker.handle({**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**request["deeper"], "cell": c}})
    assert both["ok"] and both["deeper"]["seq"] == "start" and both["qc"]["seq"] == 2
    assert {k: v for k, v in both["deeper"].items() if "ms" not in k} == {k: v for k, v in again.items() if "ms" not in k}
    # ... and riding without the init frame's file, the reason: the frame's own count is never answered as the start.
    bare = {k: v for k, v in setup["frame"].items() if not k.endswith("_counts_file")}
    riding = worker.handle({**bare, "ops": ["qc", "deeper"], "deeper": {**request["deeper"], "cell": c}})
    assert not riding["ok"] and riding["reasons"] == {"deeper": "no_counts"} and riding["qc"]["seq"] == 2
    # Without the init frame's count: the reason, never a count of the prior's outlines.
    gone = worker.handle({**request, "counts_file": None, "deeper": {**request["deeper"], "cell": c}})
    assert not gone["ok"] and gone["reasons"] == {"deeper": "no_counts"}


def test_the_fake_worker_answers_go_deeper_with_the_workers_own_code(setup, monkeypatch):
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    assert worker.handle(setup["frame"])["ok"]
    env = {**os.environ, "PYTHONPATH": str(HERE) + os.pathsep + str(HERE.parents[1])}
    process = subprocess.Popen([sys.executable, "-u", str(HERE / "fake_analysis_worker.py")], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert json.loads(process.stdout.readline())["ready"]
        request = {**setup["frame"], "ops": ["deeper", "qc"], "deeper": {**setup["deeper"], "cell": 1}}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.write(json.dumps({**setup["frame"], "op": "deeper", "ops": None,
                                        "deeper": {**setup["deeper"], "cell": 1}}) + "\n")
        process.stdin.write(json.dumps({"op": "bogus", "run": "r1", "seq": 9}) + "\n")
        process.stdin.write(json.dumps({**setup["frame"], "ops": ["qc", "explain"]}) + "\n")
        process.stdin.flush()
        replies = [json.loads(process.stdout.readline()) for _ in range(4)]
    finally:
        process.stdin.close()
        process.wait(10)
    mine = worker.handle(request)["deeper"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("ms", "worker_ms", "load_ms")}    # noqa: E731
    assert replies[0]["ok"] and strip(replies[0]["deeper"]) == strip(mine) and replies[0]["qc"]["seq"] == 2
    assert replies[1]["ok"] and strip(replies[1]["deeper"]) == strip(mine) and "qc" not in replies[1]
    assert replies[2] == {"ok": False, "run": "r1", "seq": 9, "error": "ValueError: unknown op 'bogus'",
                          "errors": {"bogus": "ValueError: unknown op 'bogus'"}, "reasons": {"bogus": "unknown_op"}}
    assert not replies[3]["ok"] and replies[3]["reasons"] == {"explain": "unknown_op"} and replies[3]["qc"]["seq"] == 2


def test_an_unknown_op_never_leaves_the_engine_waiting(setup):
    """The engine reads one reply per request (engine/analysis.py _call, 300 s): the worker answers every request but
    "forget", an op it does not know and a line it cannot read included."""
    worker = setup["worker"]
    r, w = os.pipe()
    replies_r, replies_w = os.pipe()
    out = os.fdopen(replies_w, "w")
    stream = os.fdopen(r, "rb", buffering=0)
    thread = threading.Thread(target=W.serve, args=(worker, stream, out, 60.0))
    thread.start()
    lines = [{**setup["frame"], "op": "deeper", "ops": None, "seq": 4, "deeper": {**setup["deeper"], "cell": 99}},
             {"op": "explain", "run": "r1", "seq": 5}, {"op": "forget", "run": "r1"}, [1, 2],
             {**setup["frame"], "seq": 6, "ops": ["umapx"]}]
    try:
        for line in lines:
            os.write(w, json.dumps(line).encode() + b"\n")
        os.write(w, b"{oops\n\n")
    finally:
        os.close(w)
        thread.join(30)
    assert not thread.is_alive()
    out.close()
    with os.fdopen(replies_r) as stream_out:
        replies = [json.loads(line) for line in stream_out]
    stream.close()
    assert [r.get("seq") for r in replies] == [4, 5, None, 6, None]        # no reply to forget or a blank line
    assert replies[0]["errors"] == {"deeper": "ValueError: no cell 99 in this run (12 cells)"} and not replies[0]["ok"]
    assert replies[1]["reasons"] == {"explain": "unknown_op"}
    assert replies[2]["reasons"] == {"request": "unreadable"} and replies[4]["reasons"] == {"request": "unreadable"}
    assert replies[3]["reasons"] == {"umapx": "unknown_op"} and not replies[3]["ok"]
    assert W.request_ops({"op": "deeper"}) == ["deeper"] and W.request_ops({"ops": ["umap", "deeper", "qc"]}) == \
        ["qc", "deeper", "umap"]


# ---- coverage and the depth-matched change ---------------------------------------------------------------------------
def test_every_summary_counts_all_of_the_runs_cells_by_depth(setup):
    worker = setup["worker"]
    start = worker.handle(setup["start"])["qc"]
    frame = worker.handle(setup["frame"])["qc"]
    for summary, f in ((start, 0), (frame, 2)):                 # proseg's count: the init frame's, frame 2's
        totals, share = expected_scores(W.load_encoder(setup["model"] / "encoder.npz"), proseg_count(setup["run"], f))
        assert summary["empty"] == (totals == 0).sum() and summary["under20"] == ((totals > 0) & (totals < 5)).sum()
        assert summary["under100"] == ((totals >= 5) & (totals < JUDGE)).sum()
        assert summary["judged"] + summary["empty"] + summary["under20"] + summary["under100"] == NCELLS
        assert summary["tx_sum"] == summary["transcripts"] == round(float(totals.astype(np.float64).sum()), 1)
    assert frame["empty"] >= 1                                  # cell 12 vanished in frame 2
    # A reproduction: totals [0, 0, 200] -> one judged cell of three, two empty.
    codes, table, summary = W.qc_tables(np, np.array([np.nan, np.nan, 0.1], np.float32),
                                        np.array([0, 0, 200], np.float32),
                                        {"mixed_at": 0.3, "judge_min": 100, "fit_min": 20},
                                        {"x0": 0, "y0": 0, "bin": 10, "nx": 1, "ny": 1})
    assert (summary["judged"], summary["empty"], summary["under20"], summary["under100"]) == (1, 2, 0, 0)
    assert summary["median_tx"] == 200.0                         # the existing key keeps its meaning


class NumpyCSR:
    """Just enough of cupyx's CSR matrix on numpy to run the GPU's CSR code paths here."""

    def __init__(self, arg, shape=None):
        if isinstance(arg, tuple):
            self.data, self.indices, self.indptr = (np.asarray(a) for a in arg)
            self.shape = tuple(shape)
        else:
            dense = np.asarray(arg, np.float32)
            rows, cols = np.nonzero(dense)
            self.data, self.indices = dense[rows, cols], cols.astype(np.int32)
            self.indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=dense.shape[0]))]).astype(np.int32)
            self.shape = dense.shape

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        take = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)] + [np.zeros(0, np.int64)])
        return NumpyCSR((self.data[take], self.indices[take], np.concatenate([[0], np.cumsum(ends - starts)])),
                        shape=(len(rows), self.shape[1]))

    def toarray(self):
        out = np.zeros(self.shape, np.float32)
        out[np.repeat(np.arange(self.shape[0]), np.diff(self.indptr)), self.indices] = self.data
        return out

    def __matmul__(self, other):
        return self.toarray() @ other

    def copy(self):
        return NumpyCSR((self.data.copy(), self.indices.copy(), self.indptr.copy()), shape=self.shape)


def synthetic(n=6000, genes=40, seed=0):
    """Cells mixed from 8 kinds at depths 50-900, and a calculator whose shares spread around 0.30 on them."""
    rng = np.random.default_rng(seed)
    kinds = rng.gamma(0.4, 1.0, (8, genes))
    kinds /= kinds.sum(1, keepdims=True)
    start = rng.poisson(rng.dirichlet(np.full(8, 0.3), n) @ kinds * rng.integers(50, 900, n)[:, None]).astype(np.float32)
    enc = {"W0": rng.normal(0, 0.6, (16, genes)), "b0": rng.normal(0, .1, 16), "W1": rng.normal(0, .3, (16, 16)),
           "b1": rng.normal(0, .1, 16), "Wv": rng.normal(0, 3.0, (3, 16))}
    enc = {k: v.astype(np.float32) for k, v in enc.items()}
    enc.update(var_eps=1e-3, diffusion_eps=1e-2, ln_eps=1e-5)
    h = W.layer_norm_relu(np, W.layer_norm_relu(np, np.log1p(start) @ enc["W0"].T + enc["b0"], 1e-5) @ enc["W1"].T
                          + enc["b1"], 1e-5)
    enc["bv"] = (-np.median(h @ enc["Wv"].T, axis=0) + np.array([0.0, np.log(0.3), -4.0])).astype(np.float32)
    return rng, start, enc


def matched_of(counts_now, counts_start, enc, csr=False):
    """The like-for-like change of `counts_now` against `counts_start` (each cell's start-block row), as a frame's check
    makes it (matched_change, on dense rows or the GPU's CSR path): (per-cell sums, summary, the raw change of the two
    headlines)."""
    genes = enc["W0"].shape[1]
    weights, in_model = enc["W0"].T.copy(), np.ones(genes, np.float32)
    sides = []
    for counts in (counts_now, counts_start):
        totals = (counts @ in_model).astype(np.float32)
        share = np.full(len(totals), np.nan, np.float32)
        rows = totals >= 20
        share[rows] = W.neighbour_share(W.mixing_weights(np, np.log1p(counts[rows]) @ weights, enc))
        sides.append((totals, share))
    (tn, sn), (ts, ss) = sides
    now, then = (NumpyCSR(counts_now), NumpyCSR(counts_start)) if csr else (counts_now, counts_start)
    sums, summary = W.matched_change(np, now, tn, sn, then, ts, ss, {"judge_min": 100, "mixed_at": 0.3}, weights,
                                     in_model, enc)
    raw = lambda t, s: 100 * ((t >= 100) & (s >= np.float32(0.3))).sum() / (t >= 100).sum()     # noqa: E731
    return sums, summary, raw(tn, sn) - raw(ts, ss)


def matched_flags(counts_now, counts_start, enc):
    """matched_of's comparison written out on dense rows: per cell (compared: judged now and in the start block, looks
    mixed now, looked mixed at the start), each compared cell's two rows scaled to the shallower of its two totals and
    all of them scored in one batch, now rows first."""
    ones = np.ones(counts_now.shape[1], np.float32)
    tn, ts = (counts_now @ ones).astype(np.float32), (counts_start @ ones).astype(np.float32)
    compared = (tn >= 100) & (ts >= 100)
    cells = np.nonzero(compared)[0]
    depth = np.minimum(tn[cells], ts[cells])
    rows = np.concatenate([counts_now[cells] * (depth / tn[cells])[:, None],
                           counts_start[cells] * (depth / ts[cells])[:, None]]).astype(np.float32)
    mixed = np.floor(W.neighbour_share(W.mixing_weights(np, np.log1p(rows) @ enc["W0"].T.copy(), enc)) * 250) >= 75
    now, then = np.zeros(len(tn), bool), np.zeros(len(tn), bool)
    now[cells], then[cells] = mixed[:cells.size], mixed[cells.size:]
    return compared, now, then


def test_the_like_for_like_change_reads_transcript_loss_and_an_unchanged_assignment_as_exactly_0():
    """No draws: every gene of a cell losing the same share scales its averaged profile onto its start-block row, and an
    unchanged assignment (the start block is taken at the frame's keep) is that row itself, so either reads exactly 0,
    on dense rows and on the GPU's CSR path alike, while the raw headline drifts with the loss; a loss stored in the
    counts file's steps of 1/32 reads within a few hundredths per 100."""
    rng, start, enc = synthetic()
    start = start * np.float32(KEEP)                  # proseg's averaged count of the start: 0.75 of every transcript
    raws = []
    for keep in (1.0, 0.75, 0.5, 0.25):              # 1.0: the frame is its start block (nothing changed)
        now = start * np.float32(keep)
        for csr in (False, True):
            sums, summary, raw = matched_of(now, start, enc, csr)
            assert summary["paired"] > 1000 and summary["change"] == 0.0, (keep, csr, summary)
            assert summary == {"paired": summary["paired"], "used": summary["paired"], "per100_now": summary["per100_now"],
                               "per100_start": summary["per100_now"], "change": 0.0, "sd": None, "seeds": 0}
            assert np.array_equal(sums["matched_mixed_now"], sums["matched_mixed_start"])
            assert sums["matched_paired"].sum() == summary["paired"] and set(np.unique(sums["matched_paired"])) <= {0, 1}
        raws.append(raw)
    assert raws[0] == 0 and abs(raws[3]) > abs(raws[2]) > 0                 # the raw headline is fooled by the loss
    # As a counts file keeps them (steps of 1/32 of a transcript), a loss of any other share is proportional only to
    # that step: a cell at the cut-off may flip either way, a few hundredths per 100 in all.
    kept = (np.floor(start * 32 + 0.5) / 32).astype(np.float32)
    for keep in (0.9, 0.7, 0.6, 1 / 3):
        lost = (np.floor(kept * np.float32(keep) * 32 + 0.5) / 32).astype(np.float32)
        assert abs(matched_of(lost, kept, enc)[1]["change"]) < 0.1
    # A change of what cells hold at the same depth (a fifth of them now read like cells that look mixed) still shows.
    totals = start.sum(1)
    share = W.neighbour_share(W.mixing_weights(np, np.log1p(start) @ enc["W0"].T, enc))
    donors = np.nonzero((share >= 0.5) & (totals >= 300))[0]
    changed = start.copy()
    for i in np.nonzero((rng.random(len(start)) < 0.2) & (totals < 260) & (share < 0.3))[0]:
        j = donors[rng.integers(len(donors))]
        changed[i] = start[j] * np.float32(totals[i] / totals[j])
    assert matched_of(changed, start, enc)[1]["change"] > 2.0


def test_an_unchanged_assignment_reads_exactly_0_whatever_the_backends_last_bits(monkeypatch):
    """A sparse product on the GPU (cuSPARSE) may score two copies of one row a few bits apart, and differently from one
    pass to the next: a cell whose start-block row is its row now, bit for bit, takes its now row's share, so an
    unchanged assignment reads exactly 0 and its per-cell flags agree on every backend. Pinned on the CPU by nudging the
    second half of the batch's shares (the start rows), as such a backend could; a changed row keeps its own share."""
    rng, start, enc = synthetic(n=1500, seed=5)
    start = start * np.float32(KEEP)
    now = (rng.binomial((start / np.float32(KEEP)).astype(np.int64), 0.8) * KEEP).astype(np.float32)
    plain = matched_of(now, start, enc)[1]
    score_rows = W.score_rows

    def nudged(xp, rows, weights, in_model, device, inplace=False):
        totals, shares = score_rows(xp, rows, weights, in_model, device, inplace)
        shares = shares.copy()
        shares[shares.shape[0] // 2:] += np.float32(0.05)
        return totals, shares
    monkeypatch.setattr(W, "score_rows", nudged)
    for csr in (False, True):
        sums, summary, raw = matched_of(start, start, enc, csr=csr)
        assert summary["paired"] > 100 and summary["change"] == 0.0 and raw == 0, summary
        assert np.array_equal(sums["matched_mixed_now"], sums["matched_mixed_start"])
    moved = matched_of(now, start, enc)[1]                          # the nudge does move a changed row's share
    assert moved["per100_start"] > plain["per100_start"] and moved["per100_now"] == plain["per100_now"]


def test_the_like_for_like_change_is_the_same_every_time_and_on_csr():
    rng, start, enc = synthetic(n=1500, seed=3)
    start = start * np.float32(KEEP)
    now = (rng.binomial((start / np.float32(KEEP)).astype(np.int64), 0.8) * KEEP).astype(np.float32)
    a_sums, a, _ = matched_of(now, start, enc)
    b_sums, b, _ = matched_of(now, start, enc, csr=True)            # the GPU's CSR path: the same answer, bit for bit
    assert a == b and all(np.array_equal(a_sums[k], b_sums[k]) for k in W.MATCHED_SUMS)
    assert matched_of(now, start, enc)[1] == a                     # the same frame again: the same answer
    assert set(a) == {"paired", "used", "per100_now", "per100_start", "change", "sd", "seeds"} and a["change"] != 0
    assert a["change"] == round(a["per100_now"] - a["per100_start"], 4) and a["used"] == a["paired"]
    for name in W.MATCHED_SUMS:                                     # per cell: compared or not, mixed or not
        assert set(np.unique(a_sums[name])) <= {0.0, 1.0}
    assert (a_sums["matched_mixed_now"] <= a_sums["matched_paired"]).all()
    # Each cell's sums are its own, on either path: compared, looks mixed now, looked mixed at the start, never the two
    # sides swapped (the bins add them up per area, and engine/qc_areas.local_change reads 100 x (now - start) /
    # compared of them); over the run they are the summary's two sides.
    flags = matched_flags(now, start, enc)
    assert (flags[1] != flags[2]).any()
    for sums, summary in ((a_sums, a), (b_sums, b)):
        for name, expected in zip(W.MATCHED_SUMS, flags):
            assert np.array_equal(sums[name], expected), name
        paired = summary["paired"]
        assert sums["matched_paired"].sum() == paired
        assert round(100 * sums["matched_mixed_now"].sum() / paired, 4) == summary["per100_now"]
        assert round(100 * sums["matched_mixed_start"].sum() / paired, 4) == summary["per100_start"]
    # A cell judged now but under the judging floor in its start block (it grew past it since) is not compared.
    grown = (start.sum(1) >= 160) & (start.sum(1) < 220)
    then = np.where(grown[:, None], start * np.float32(0.45), start).astype(np.float32)
    sums, summary, _ = matched_of(start, then, enc)
    both = (start.sum(1) >= 100) & (then.sum(1) >= 100)
    assert grown.sum() > 50 and summary["paired"] == both.sum() and not sums["matched_paired"][grown].any()
    assert summary == matched_of(start[both], then[both], enc)[1]
    # Nobody judged at both ends: nothing to compare.
    empty = W.matched_change(np, now[:5] * 0, np.zeros(5, np.float32), np.full(5, np.nan, np.float32), start[:5],
                             start[:5].sum(1), np.full(5, np.nan, np.float32), {"judge_min": 100, "mixed_at": 0.3},
                             enc["W0"].T, np.ones(40, np.float32), enc)[1]
    assert empty == {"paired": 0, "used": 0, "per100_now": None, "per100_start": None, "change": None, "sd": None,
                     "seeds": 0}


def test_binomial_draws_keep_their_share_of_whole_counts():
    """thin_values (Compare's run column without a sampled count): a count n becomes Binomial(n, p), never more than n;
    p = 1 keeps every transcript; the same seed, the same draws."""
    values = np.array([0, 1, 5, 40, 300], np.float32)
    for p in (0.0, 0.3, 1.0):
        drawn = W.thin_values(np, values, np.full(5, p, np.float32), 7)
        assert (drawn <= values).all() and drawn.dtype == np.float32
        assert (p > 0 or drawn.sum() == 0) and (p < 1 or np.array_equal(drawn, values))
    many = W.thin_values(np, np.full(20000, 3, np.float32), np.full(20000, 0.25, np.float32), 11)
    assert many.mean() == pytest.approx(0.75, abs=0.03)
    assert np.array_equal(many, W.thin_values(np, np.full(20000, 3, np.float32), np.full(20000, 0.25, np.float32), 11))


def test_frames_carry_the_like_for_like_change_and_their_bins_its_sums(setup, tmp_path):
    """A frame read against its start block carries the like-for-like change, and its bins each compared cell's 0 or 1
    in the bin of the cell: compared, looks mixed now, looked mixed at the start (engine/qc_areas.local_change reads
    each area's change as 100 x (now - start) / compared of them, so the two sides swapped would turn every area's
    arrow round). Under a model by which about half of frame 2's cells look mixed, fewer of them look mixed now than
    in its start block (write_run: proseg's iteration-0 assignment at the frame's keep), like for like."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    e2 = proseg_count(setup["run"], 2)
    rows = model_rows(W.load_encoder(setup["model"] / "encoder.npz"), e2)
    model = str(mixing_model(tmp_path / "mixing" / "encoder.npz", rows[rows.sum(axis=1) >= JUDGE]))
    qc = {**setup["frame"]["qc"], "model": model}
    frame = {**setup["frame"], "qc": qc}
    alone = {k: v for k, v in frame.items() if not k.endswith("_counts_file")}     # no start line
    plain = worker.handle({**alone, "qc": {**qc, "start": None}})["qc"]
    assert plain["matched"] is None and plain["start_block"] is None and "paired" not in plain
    assert worker.handle({**setup["start"], "qc": {**setup["start"]["qc"], "model": model}})["ok"]
    summary = worker.handle(frame)["qc"]
    matched = summary["matched"]
    assert set(matched) == {"paired", "used", "per100_now", "per100_start", "change", "sd", "seeds", "ms"}
    assert matched["paired"] == summary["paired"]["n"] == matched["used"] > 0   # judged now and in the start block
    assert (matched["sd"], matched["seeds"]) == (None, 0)
    # Worked out cell by cell: the frame's count against its start block.
    compared, now_mixed, start_mixed = like_for_like_flags(W.load_encoder(model), e2, proseg_count(setup["run"], 0), qc)
    n, now_m, start_m = int(compared.sum()), int(now_mixed.sum()), int(start_mixed.sum())
    assert 0 < now_m < start_m and {k: v for k, v in matched.items() if k != "ms"} == {
        "paired": n, "used": n, "per100_now": round(100 * now_m / n, 4), "per100_start": round(100 * start_m / n, 4),
        "change": round(100 * now_m / n - 100 * start_m / n, 4), "sd": None, "seeds": 0}
    bins = np.load(out / "000002.bins.npz")
    for name in W.MATCHED_SUMS:
        assert bins[name].dtype == np.float32
    # Each compared cell sits in the bin of its centroid in the frame (its cell statistics; cell 12, gone from frame 2,
    # is not compared).
    stats = cell_stats(frame_plane(2), 2)
    x, y = stats[5].view(np.float32)[compared], stats[6].view(np.float32)[compared]
    grid = qc["grid"]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    where = (np.clip(np.floor((y - grid["y0"]) / grid["bin"]), 0, grid["ny"] - 1) * grid["nx"]
             + np.clip(np.floor((x - grid["x0"]) / grid["bin"]), 0, grid["nx"] - 1)).astype(np.int64)
    for name, flags in zip(W.MATCHED_SUMS, (compared, now_mixed, start_mixed)):
        per_bin = np.bincount(where, weights=flags[compared], minlength=grid["nx"] * grid["ny"])
        assert np.array_equal(bins[name], per_bin[bins["bin"]]), name
    assert (bins["matched_paired"].sum(), bins["matched_mixed_now"].sum(), bins["matched_mixed_start"].sum()) == \
        (n, now_m, start_m)
    over_all_bins = 100 * (bins["matched_mixed_now"].sum() - bins["matched_mixed_start"].sum()) / n
    assert over_all_bins == pytest.approx(matched["change"], abs=1e-4) and over_all_bins < 0
    assert "tx" in bins and "tx_start" in bins                   # retention per area: tx / tx_start
    assert summary["ms"]["bins"] >= matched["ms"] >= 0           # its time is part of the per-bin sums'
    # A fresh worker reads the init frame's count again: the same answer.
    again = W.Worker(xp=np).handle(frame)["qc"]
    assert {k: v for k, v in again["matched"].items() if k != "ms"} == {k: v for k, v in matched.items() if k != "ms"}
    # A start.npz that is not the init frame's count gives no like-for-like change, never a wrong one; the frame keeps
    # its check and its start block (proseg's own).
    stored = dict(np.load(out / "start.npz"))
    np.savez(out / "start.npz", **{**stored, "totals": stored["totals"] + 1})
    reply = W.Worker(xp=np).handle(frame)
    assert reply["ok"] and reply["qc"]["matched"]["error"].startswith("ValueError: the start line's counts differ")
    assert reply["qc"]["judged"] == summary["judged"] and reply["qc"]["start_block"] == summary["start_block"]
    np.savez(out / "start.npz", **stored)
    # The counts file named for the frame's start block holds none: no comparison with the start; the check stands.
    read = counts_file.read(live / "frames" / "000002.counts.zst")
    counts_file.write(tmp_path / "plain.counts.zst", ncells=read.ncells, ngenes=read.ngenes, iteration=2,
                      indptr=read.indptr, genes=read.genes, n=read.n, e=read.e)
    reply = W.Worker(xp=np).handle({**frame, "start_counts_file": str(tmp_path / "plain.counts.zst")})
    assert reply["ok"] and reply["qc"]["judged"] == summary["judged"] and reply["qc"]["start_block"] is None
    assert "start block is not in the counts file" in reply["qc"]["matched"]["error"]


def test_a_frame_whose_assignment_is_its_start_reads_no_change(setup, tmp_path):
    """Through the worker: a frame whose cells hold what they started with (proseg's iteration-0 assignment), at any
    keep, and one whose cells lost the same share of every gene, read exactly 0 like for like, in the tissue and in
    each cell's Go deeper answer; the start block's row, not the start line's, is each cell's start."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    assert worker.handle(setup["start"])["ok"]
    init = counts_file.read(live / "frames" / "000000.counts.zst")
    rows = {"ncells": init.ncells, "ngenes": init.ngenes, "indptr": init.indptr, "genes": init.genes, "n": init.n}
    counts_file.write(tmp_path / "kept.counts.zst", iteration=7, **rows, e=init.n * 0.5, start_e=init.n * 0.5)
    counts_file.write(tmp_path / "lost.counts.zst", iteration=8, **rows, e=init.n * 0.375, start_e=init.n * 0.75)
    seen = 0
    for name in ("kept", "lost"):
        path = str(tmp_path / f"{name}.counts.zst")
        frame = {**setup["frame"], "counts_file": path, "start_counts_file": path}
        summary = worker.handle(frame)["qc"]
        matched = summary["matched"]
        assert matched["paired"] > 0 and matched["change"] == 0.0 and matched["per100_now"] == matched["per100_start"]
        for c in range(NCELLS):
            result = worker.handle({**frame, "ops": ["qc", "deeper"],
                                    "deeper": {**setup["deeper"], "cell": c, "start_line": True}})["deeper"]
            if "start_change" in result:
                seen += 1
                assert result["start_change"]["points"] == 0.0 and result["start_change"]["direction"] == "same"
                block = model_counts_of(setup["model"], (init.n * (0.5 if name == "kept" else 0.75)), init, c)
                assert result["start_change"]["start_tx"] == round(float(block.sum()), 1)
                assert result["start_change"]["now_tx"] == round(result["transcripts"], 1)
    assert seen > 0
    # The start block's own row: here half of the start line's (start.npz: the init frame at keep 0.75).
    stored = np.load(out / "start.npz")
    path = str(tmp_path / "kept.counts.zst")
    c = int(np.argmax(stored["totals"]))
    result = worker.handle({**setup["frame"], "counts_file": path, "start_counts_file": path, "ops": ["deeper"],
                            "deeper": {**setup["deeper"], "cell": c, "start_line": True}})["deeper"]
    assert result["start_change"]["start_tx"] == pytest.approx(stored["totals"][c] * 0.5 / KEEP, abs=0.051)


def test_a_start_block_whose_file_went_costs_the_frame_nothing_and_says_so_in_one_plain_line(setup, tmp_path, capsys):
    """A frame's start comparison whose start-block file (or the init frame's) is gone or not this run's (no_counts:
    the spool pruned it, --live-counts-start K > 1 keeps it in another frame's file): the frame keeps its check,
    matched says why, and the log has one plain line for it, as every other no_counts path, never a traceback."""
    worker, live = setup["worker"], setup["live"]
    assert worker.handle(setup["start"])["ok"]
    capsys.readouterr()
    alien = tmp_path / "alien.counts.zst"                            # another run's (3 cells): not this run's count
    counts_file.write(alien, ncells=3, ngenes=len(RUN_GENES), iteration=0, indptr=[0, 0, 0, 0], genes=[], n=[], e=[])
    for field, path, error in (("start_counts_file", tmp_path / "gone-block.counts.zst", "NoCounts: "),
                               ("init_counts_file", tmp_path / "gone-init.counts.zst", "NoCounts: "),
                               ("start_counts_file", alien, "CountsError: ")):
        reply = worker.handle({**setup["frame"], field: str(path)})
        assert reply["ok"], reply
        matched = reply["qc"]["matched"]
        assert matched["error"].startswith(error), matched
        assert reply["qc"]["start_block"] is None
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    assert err.count("no comparison with the start for run r1: NoCounts: ") == 2, err
    assert err.count("no comparison with the start for run r1: CountsError: ") == 1, err


def test_each_frame_is_read_against_its_own_start_block(setup, tmp_path):
    """D1 (a'): a frame's start side is its start block (proseg's iteration-0 assignment kept as proseg keeps RNA at
    this frame: the counts file named for it, on the init frame's rows), read at its own depth: start_block, the pairs,
    the bins' judged_start, mixed_start and tx_start, and the like-for-like change; never the init frame's own count
    (the start line, start.npz) nor the frame's. Here the block keeps a fifth of every other cell's RNA and less of some
    genes than others, under a model by which some cells look mixed: it judges and flags other cells than the start
    line, and cells that grew past the judging floor since are not compared. A block kept in an earlier counted frame's
    file (the frame's own holds none) reads the same."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    init = counts_file.read(live / "frames" / "000000.counts.zst")
    now = counts_file.read(live / "frames" / "000002.counts.zst")
    row = np.repeat(np.arange(NCELLS), np.diff(init.indptr))
    cell_keep = np.where(np.arange(NCELLS) % 2 == 0, 0.2, 0.6)
    gene_keep = np.linspace(0.4, 1.0, len(RUN_GENES))
    frame_rows = {"ncells": now.ncells, "ngenes": now.ngenes, "indptr": now.indptr, "genes": now.genes, "n": now.n,
                  "e": now.e, "f": now.f}
    counts_file.write(tmp_path / "block.counts.zst", iteration=9, **frame_rows,
                      start_e=init.n * cell_keep[row] * gene_keep[init.genes])
    block = np.zeros((NCELLS, len(RUN_GENES)), np.float32)
    block[row, init.genes] = counts_file.read(tmp_path / "block.counts.zst").start_e    # as stored: steps of 1/32
    e2 = proseg_count(setup["run"], 2)                                  # the frame's own count, as before
    rows = model_rows(W.load_encoder(setup["model"] / "encoder.npz"), np.concatenate([e2, block]))
    model = mixing_model(tmp_path / "mixing" / "encoder.npz", rows[rows.sum(axis=1) >= JUDGE])
    encoder = W.load_encoder(model)
    assert worker.handle({**setup["start"], "qc": {**setup["start"]["qc"], "model": str(model)}})["ok"]
    frame = {**setup["frame"], "counts_file": str(tmp_path / "block.counts.zst"),
             "start_counts_file": str(tmp_path / "block.counts.zst"), "qc": {**setup["frame"]["qc"], "model": str(model)}}
    reply = worker.handle(frame)
    assert reply["ok"], reply
    summary = reply["qc"]
    # Worked out here: the frame now, and the block at its own depth.
    totals, share = expected_scores(encoder, e2)
    start_totals, start_share = expected_scores(encoder, block)
    judged, start_judged = (totals >= JUDGE) & np.isfinite(share), (start_totals >= JUDGE) & np.isfinite(start_share)
    mixed = judged & (W.share_codes(np, share, judged) >= 75)
    start_mixed = start_judged & (W.share_codes(np, start_share, start_judged) >= 75)
    j, m = int(start_judged.sum()), int(start_mixed.sum())
    scored = (start_totals >= 5) & np.isfinite(start_share)
    assert summary["start_block"] == {"per100": round(100 * m / j, 2), "judged": j, "mixed": m,
                                      "tx_sum": round(float(start_totals.sum()), 1),
                                      "median_tx": round(float(np.median(start_totals[scored])), 1), "iteration": 9}
    paired = judged & start_judged
    assert summary["paired"] == {"n": paired.sum(), "mixed_now": (paired & mixed).sum(),
                                 "mixed_start": (paired & start_mixed).sum(), "n10": (paired & mixed & ~start_mixed).sum(),
                                 "n01": (paired & ~mixed & start_mixed).sum()}
    bins = np.load(out / "000002.bins.npz")
    assert bins["cells"].sum() == NCELLS and bins["judged_start"].sum() == j and bins["mixed_start"].sum() == m
    assert bins["tx_start"].sum() == pytest.approx(float(start_totals.sum()), abs=0.01)
    n, now_m, start_m = like_for_like_reference(encoder, e2, block, {"judge_min": JUDGE, "mixed_at": 0.3})
    assert {k: v for k, v in summary["matched"].items() if k != "ms"} == {
        "paired": n, "used": n, "per100_now": round(100 * now_m / n, 4), "per100_start": round(100 * start_m / n, 4),
        "change": round(100 * now_m / n - 100 * start_m / n, 4), "sd": None, "seeds": 0}
    assert bins["matched_paired"].sum() == n == paired.sum()
    assert (bins["matched_mixed_now"].sum(), bins["matched_mixed_start"].sum()) == (now_m, start_m)
    # The block is not the start line (start.npz: the init frame's count, 0.75 of each assigned transcript): other
    # cells judged and flagged, other transcripts; and some cells grew past the judging floor since.
    line = np.load(out / "start.npz")
    line_judged = line["code"] != W.NOT_JUDGED
    assert m > 0 and (j, m) != (int(line_judged.sum()), int((line_judged & (line["code"] >= 75)).sum()))
    assert abs(float(start_totals.sum()) - float(line["totals"].sum())) > 100
    assert (judged & ~start_judged & (start_totals >= 5)).any()
    # The block in an earlier counted frame's file, the frame's own without one: the same reading.
    counts_file.write(tmp_path / "plain.counts.zst", iteration=2, **frame_rows)
    later = W.Worker(xp=np).handle({**frame, "counts_file": str(tmp_path / "plain.counts.zst")})["qc"]
    assert later["start_block"] == summary["start_block"] and later["paired"] == summary["paired"]
    assert {k: v for k, v in later["matched"].items() if k != "ms"} == \
        {k: v for k, v in summary["matched"].items() if k != "ms"}


def test_d0_is_one_switch_and_like_for_like_stays_on_the_averaged_count(setup, monkeypatch):
    """D0 is one switch (counts_file.LIVE_COUNT). Under its sampled count every live number is proseg's sampled count f
    (the check, the start line, Go deeper's rows and Compare's run column; transcripts_in_cells: test_worker_backend)
    and the summaries say "sampled", while the like-for-like comparisons stay on the averaged count e, as the start
    block is: the same answers as under the averaged count. A frame kept without f has no live count then (reason
    no_counts): proseg would have to keep f with every counted frame."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    assert worker.handle(setup["start"])["ok"]
    cell = int(np.argmax(np.load(out / "start.npz")["totals"]))
    asked = {**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], "cell": cell, "start_line": True}}
    averaged = worker.handle(asked)
    assert averaged["ok"] and averaged["qc"]["count"] == "averaged" and "start_change" in averaged["deeper"]
    monkeypatch.setattr(counts_file, "LIVE_COUNT", "sampled")
    monkeypatch.setattr(counts_file, "LIVE_ARRAY", "f")
    encoder = W.load_encoder(setup["model"] / "encoder.npz")
    f0, f2 = (W.Worker(xp=np).matrix(counts_file.read(live / "frames" / f"{seq:06d}.counts.zst"), "f") for seq in (0, 2))
    sampled = W.Worker(xp=np)
    start = sampled.handle(setup["start"])
    totals0, _ = expected_scores(encoder, f0)
    assert start["ok"] and start["qc"]["count"] == "sampled" and start["qc"]["tx_sum"] == round(float(totals0.sum()), 1)
    assert np.array_equal(np.load(out / "start.npz")["totals"], totals0)
    reply = sampled.handle(asked)
    assert reply["ok"], reply
    totals, share = expected_scores(encoder, f2)
    qc, deep = reply["qc"], reply["deeper"]
    assert (qc["count"], qc["basis"]) == ("sampled", "proseg") and qc["tx_sum"] == round(float(totals.sum()), 1)
    assert qc["judged"] == ((totals >= JUDGE) & np.isfinite(share)).sum() and deep["transcripts"] == round(
        float(totals[cell]), 1)
    strip = lambda d: {k: v for k, v in d.items() if k != "ms"}          # noqa: E731
    assert strip(qc["matched"]) == strip(averaged["qc"]["matched"]) and qc["start_block"] == averaged["qc"]["start_block"]
    assert deep["start_change"] == averaged["deeper"]["start_change"]
    column = sampled.compare_frame({k: setup["frame"][k] for k in ("run", "genes", "counts_file", "ncells", "cells_file")},
                                   encoder, 0)["csr"]
    got = np.zeros(column.shape, np.float32)
    got[np.repeat(np.arange(column.shape[0]), np.diff(column.indptr)), column.indices] = column.data
    assert np.array_equal(got, model_rows(encoder, f2))
    without = sampled.handle({**setup["frame"], "seq": 1, **counts_files(live, 1),
                              "qc": {**setup["frame"]["qc"], "out_cells": str(out / "000001.qc.bin"), "out_bins": None}})
    assert without["reasons"] == {"qc": "no_counts"} and "no sampled count" in without["errors"]["qc"]


def saved_f(live, low=None, seed=7):
    """Frame 2 counted again for the end, its line now holding f, the count proseg saves: one draw of Binomial(n, q)
    per stored pair, q proseg's keep pair by pair (so f is not e scaled: e stays the fixture's 0.75 n), and cell `low`'s
    pairs kept at 0.1 (judged on e and at the start, not on f). f over the run's genes (cells x genes)."""
    from counts_fixtures import write_frame_counts
    read = counts_file.read(live / "frames" / "000002.counts.zst")
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(NCELLS), np.diff(read.indptr))
    q = rng.uniform(0.35, 0.95, read.nnz)
    if low is not None:
        q[rows == low] = 0.1
    f = rng.binomial(read.n.astype(np.int64), q)
    write_frame_counts(live, 2, 2, ncells=NCELLS, ngenes=len(RUN_GENES), indptr=read.indptr, genes=read.genes,
                       n=read.n, e=read.e, f=f, start_e=read.start_e, reason="end")
    dense = np.zeros((NCELLS, len(RUN_GENES)), np.float32)
    dense[rows, read.genes] = f
    return dense


def test_a_frame_read_on_the_count_proseg_saves_is_scored_on_f_beside_e(setup):
    """A request for a frame proseg counted for a pause, the hold, a finish or the end names count "sampled" (no
    switch): the check scores f, the count proseg saves, for its numbers and its bytes, and e beside it for the
    summary's `averaged` block (an averaged request's numbers), its bytes on e (out_cells_e), the bins file (an averaged
    request's of the same frame, and no second one) and every comparison with the start: the pairs with the start block
    and the like-for-like change are the averaged request's, never f's. Here f is a draw of its own (proseg's keep
    differs from pair to pair: f is not e scaled) and one cell keeps a tenth of its RNA in it, judged on e and at the
    start but not on f, so f's pairs and like for like differ from e's. Go deeper in the same pass reads f's totals and
    codes (`kept`), its change since the start e's."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    assert worker.handle(setup["start"])["ok"]
    order = np.argsort(-np.load(out / "start.npz")["totals"], kind="stable")
    cell, low = int(order[0]), int(order[1])
    f2 = saved_f(live, low=low)
    path = live / "frames" / "000002.counts.zst"
    read = counts_file.read(path)
    encoder = W.load_encoder(setup["model"] / "encoder.npz")
    e2 = worker.matrix(read, "e")
    assert np.array_equal(worker.matrix(read, "f"), f2) and not np.allclose(f2 * (e2.sum() / f2.sum()), e2)
    asked = {**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], "cell": cell, "start_line": True}}
    averaged = W.Worker(xp=np).handle(asked)
    assert averaged["ok"] and averaged["qc"]["count"] == "averaged" and "start_change" in averaged["deeper"]
    e_codes, e_bins = codes_of(out / "000002.qc.bin"), dict(np.load(out / "000002.bins.npz"))
    for name in ("000002.qc.bin", "000002.bins.npz"):
        (out / name).unlink()
    reply = W.Worker(xp=np).handle({**asked, "count": "sampled",
                                    "qc": {**asked["qc"], "out_cells_e": str(out / "000002.qc.averaged.bin")}})
    assert reply["ok"], reply
    qc, deep, plain = reply["qc"], reply["deeper"], averaged["qc"]
    totals, share = expected_scores(encoder, f2)
    judged = (totals >= JUDGE) & np.isfinite(share)
    codes = W.share_codes(np, share, judged)
    assert (qc["count"], qc["basis"]) == ("sampled", "proseg") and qc["tx_sum"] == round(float(totals.sum()), 1)
    assert (qc["judged"], qc["mixed"]) == (judged.sum(), (judged & (codes >= 75)).sum())
    assert (qc["judged"], qc["tx_sum"]) != (plain["judged"], plain["tx_sum"]), "f is not e here"
    assert np.array_equal(codes_of(out / "000002.qc.bin"), codes)
    assert np.array_equal(codes_of(out / "000002.qc.averaged.bin"), e_codes)
    assert qc["averaged"] == {k: plain[k] for k in ("judged", "mixed", "tx_sum", "median_tx")}
    # Every comparison with the start reads e: the pairs and like for like are the averaged request's, worked out here
    # on e against the frame's start block, and differ from what f would give.
    init = counts_file.read(live / "frames" / "000000.counts.zst")
    block = np.zeros_like(e2)
    block[np.repeat(np.arange(NCELLS), np.diff(init.indptr)), init.genes] = read.start_e
    e_totals, e_share = expected_scores(encoder, e2)
    e_judged = (e_totals >= JUDGE) & np.isfinite(e_share)
    start_totals, start_share = expected_scores(encoder, block)
    start_judged = (start_totals >= JUDGE) & np.isfinite(start_share)
    assert e_judged[low] and start_judged[low] and not judged[low], "judged on e and at the start, not on f"
    assert qc["paired"] == plain["paired"] and qc["paired"]["n"] == (e_judged & start_judged).sum() != \
        (judged & start_judged).sum()
    strip = lambda d: {k: v for k, v in d.items() if k != "ms"}          # noqa: E731
    thresholds = {"judge_min": JUDGE, "mixed_at": 0.3}
    on_e, on_f = (like_for_like_reference(encoder, now, block, thresholds) for now in (e2, f2))
    n, now_m, start_m = on_e
    assert strip(qc["matched"]) == strip(plain["matched"]) == {
        "paired": n, "used": n, "per100_now": round(100 * now_m / n, 4), "per100_start": round(100 * start_m / n, 4),
        "change": round(100 * now_m / n - 100 * start_m / n, 4), "sd": None, "seeds": 0}
    assert on_f != on_e and qc["start_block"] == plain["start_block"]
    bins = dict(np.load(out / "000002.bins.npz"))
    assert bins.keys() == e_bins.keys()
    assert all(bins[k].dtype == e_bins[k].dtype and np.array_equal(bins[k], e_bins[k]) for k in bins), "the areas read e"
    assert qc["files"] == {"cells": str(out / "000002.qc.bin"), "bins": str(out / "000002.bins.npz"),
                           "cells_e": str(out / "000002.qc.averaged.bin")}
    assert sorted(p.name for p in out.glob("000002*")) == ["000002.bins.npz", "000002.qc.averaged.bin", "000002.qc.bin"]
    # Go deeper in the same pass: f's totals and the cell's byte on f; its change since the start on e.
    assert deep["count"] == "sampled" and deep["code"] == int(codes[cell])
    assert deep["transcripts"] == pytest.approx(float(totals[cell]), abs=1e-4)
    assert abs(deep["transcripts"] - averaged["deeper"]["transcripts"]) > 1
    assert deep["start_change"] == averaged["deeper"]["start_change"]


def test_a_frame_read_on_f_compares_its_averaged_profiles_with_its_start(setup, tmp_path):
    """Like for like on a frame read on f compares the frame's averaged count with its start block, profiles and
    depths alike, never f's draw: under a model by which some cells look mixed (and the same cells judged on e and on
    f), the change and the per-bin like-for-like sums are an averaged request's, while f's profiles at the same depths
    would flag other cells."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    f2 = saved_f(live)
    read = counts_file.read(live / "frames" / "000002.counts.zst")
    e2 = worker.matrix(read, "e")
    init = counts_file.read(live / "frames" / "000000.counts.zst")
    block = np.zeros_like(e2)
    block[np.repeat(np.arange(NCELLS), np.diff(init.indptr)), init.genes] = read.start_e
    rows = model_rows(W.load_encoder(setup["model"] / "encoder.npz"), np.concatenate([e2, f2, block]))
    model = mixing_model(tmp_path / "mixing" / "encoder.npz", rows[rows.sum(axis=1) >= JUDGE])
    encoder = W.load_encoder(model)
    thresholds = {"judge_min": JUDGE, "mixed_at": 0.3}
    e_flags, f_flags = (like_for_like_flags(encoder, now, block, thresholds) for now in (e2, f2))
    assert np.array_equal(e_flags[0], f_flags[0]) and e_flags[0].any(), "the same cells compared"
    assert not np.array_equal(e_flags[1], f_flags[1]), "f's profiles flag other cells"
    assert worker.handle({**setup["start"], "qc": {**setup["start"]["qc"], "model": str(model)}})["ok"]
    frame = {**setup["frame"], "qc": {**setup["frame"]["qc"], "model": str(model)}}
    plain = W.Worker(xp=np).handle(frame)
    e_bins = dict(np.load(out / "000002.bins.npz"))
    on_f = W.Worker(xp=np).handle({**frame, "count": "sampled",
                                   "qc": {**frame["qc"], "out_cells_e": str(out / "000002.qc.averaged.bin")}})
    assert plain["ok"] and on_f["ok"], (plain, on_f)
    strip = lambda d: {k: v for k, v in d.items() if k != "ms"}          # noqa: E731
    n, now_m, start_m = (int(flags.sum()) for flags in e_flags)
    assert strip(on_f["qc"]["matched"]) == strip(plain["qc"]["matched"])
    assert (on_f["qc"]["matched"]["paired"], on_f["qc"]["matched"]["per100_now"]) == (n, round(100 * now_m / n, 4))
    bins = dict(np.load(out / "000002.bins.npz"))
    for name in ("matched_paired", "matched_mixed_now", "matched_mixed_start"):
        assert np.array_equal(bins[name], e_bins[name]), name
    assert (int(bins["matched_paired"].sum()), int(bins["matched_mixed_now"].sum())) == (n, now_m)


def test_one_worker_answers_go_deeper_on_each_count_its_frame_is_read_on(setup):
    """The worker keeps a frame's totals for Go deeper on another of its cells later, by the count they were read on:
    after an answer on the frame's averaged count, an answer on the count proseg saves (the hold counted the frame
    again: the engine asks for it explicitly) reads f's totals, never the e totals kept from the first; and back on e,
    e's again."""
    worker, out, live = setup["worker"], setup["out"], setup["live"]
    assert worker.handle(setup["start"])["ok"]
    cell = int(np.argmax(np.load(out / "start.npz")["totals"]))
    f2 = saved_f(live)
    encoder = W.load_encoder(setup["model"] / "encoder.npz")
    e_totals = expected_scores(encoder, worker.matrix(counts_file.read(live / "frames" / "000002.counts.zst"), "e"))[0]
    f_totals = expected_scores(encoder, f2)[0]
    assert abs(float(f_totals[cell]) - float(e_totals[cell])) > 1
    assert worker.handle({**setup["frame"], "ops": ["qc"]})["ok"]       # the frame's bytes, for the cards' byte
    ask = {**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": cell}}
    answers = [worker.handle(request) for request in (ask, {**ask, "count": "sampled"}, ask)]
    assert all(answer["ok"] for answer in answers), answers
    assert [answer["deeper"]["count"] for answer in answers] == ["averaged", "sampled", "averaged"]
    for answer, totals in zip(answers, (e_totals, f_totals, e_totals)):
        assert answer["deeper"]["transcripts"] == pytest.approx(float(totals[cell]), abs=1e-4)


def test_a_genes_amounts_are_sent_at_3_decimals_on_the_whole_number_they_round_to(setup):
    """H4: a gene's transcripts are one entry of the counts file (e, a multiple of 1/32; or f) and the page rounds each
    once, to whole (Math.round): sent at DEEP_DIGITS (3) decimals every multiple of 1/32 up to 200 rounds to the whole
    number it rounds to itself (at 1 decimal 15/32 = 0.469 went as 0.5: one too high once rounded); a cell's gene of
    e = 2.46875 is sent as 2.469, never 2.5; each such number's length as json writes it is _amount_bytes'."""
    whole = lambda v: np.floor(np.asarray(v, np.float64) + 0.5)          # noqa: E731  (Math.round's rule)
    exact = np.arange(200 * 32 + 1) / 32
    sent = np.round(exact, W.DEEP_DIGITS)
    assert W.DEEP_DIGITS == 3 and np.array_equal(whole(sent), whole(exact))
    assert whole(round(15 / 32, 1)) != whole(15 / 32)                  # what 1 decimal did
    assert W._amount_bytes(sent).tolist() == [len(json.dumps(float(v))) for v in sent]
    model = setup["worker"].deep_model(setup["model"])
    deep, enc = model["deep"], model["encoder"]
    counts = model_counts(setup["run"])
    c = int(np.argmax(counts.sum(1)))
    g = int(np.nonzero(counts[c])[0][0])
    counts[c, g] = 2.46875
    frame = W.DeepFrame(counts.sum(1), np.random.default_rng(5).uniform(0, 40, (NCELLS, 2)), 5.0)
    result = W.explain_cell(c, frame, lambda rows: (counts[np.asarray(rows)], np.zeros(len(rows))), deep, enc,
                            RUN_IDS, np.full(NCELLS, -1), None, JUDGE)
    (row,) = [row for row in result["genes"] if row[0] == deep["genes"][g]]
    assert row[1] == 2.469 and json.dumps(row).count("2.469") == 1


def model_counts_of(model, values, init, c):
    """Cell c's row of `values` (on the init frame's rows) over the model's genes."""
    encoder = W.load_encoder(Path(model) / "encoder.npz")
    index = W.gene_index(encoder, RUN_GENES)[0]
    a, b = int(init.indptr[c]), int(init.indptr[c + 1])
    return W.model_row(init.genes[a:b], np.asarray(values, np.float32)[a:b], index, encoder["ngenes"])


def test_the_gpus_csr_paths_give_the_dense_answers(setup, monkeypatch):
    """The GPU worker keeps a frame's count as a CSR (cupyx): its own code (each count made by Worker.wrap's cupyx
    branch: the frame's, the start block's, the init frame's and the start line's; the check's rows, the like-for-like
    change's scaled rows in one batch, Go deeper's rows read back packed, and the start's), on numpy dressed as cupy
    with a minimal cupyx CSR over numpy, gives exactly the numpy worker's dense answers and files."""
    dense, out = setup["worker"], setup["out"]
    fake_cupyx(monkeypatch, NumpyCSR)
    sparse = W.Worker(xp=GpuLike(), backend="gpu")
    matrix, read = sparse.count(sparse.state(setup["frame"]), setup["frame"])
    kept = read.e != 0                                                  # the frame's matrix: e, int32 rows, no zeros
    assert isinstance(matrix, NumpyCSR) and matrix.shape == (NCELLS, len(RUN_GENES)) and matrix.data.dtype == np.float32
    assert matrix.indices.dtype == matrix.indptr.dtype == np.int32 and np.array_equal(matrix.data, read.e[kept])
    assert np.array_equal(matrix.indices, read.genes[kept]) and np.array_equal(matrix.toarray(), dense.matrix(read))
    requests = [setup["start"]] + [{**setup["frame"], "ops": ["qc", "deeper"],
                                    "deeper": {**setup["deeper"], "cell": c, "start_line": True}} for c in range(NCELLS)]
    requests.append({"op": "deeper", **setup["common"], "seq": "start", **counts_files(setup["live"], 0, start=False),
                     "deeper": {**setup["deeper"], "start": True, "cell": 1, "codes": str(out / "start.bin")}})
    strip = lambda value: {k: strip(v) for k, v in value.items() if "ms" not in k and k != "gpu"} \
        if isinstance(value, dict) else value                                                          # noqa: E731
    files = {}
    for worker in (dense, sparse):
        replies = [strip(worker.handle(r)) for r in requests]
        assert all(r["ok"] for r in replies), replies
        files[worker] = {p.name: p.read_bytes() for p in out.iterdir() if p.suffix == ".bin"}
        if worker is dense:
            first = replies
    assert replies == first and files[dense] == files[sparse]
    assert sum("start_change" in r.get("deeper", {}) for r in replies) > 0
    assert sparse.rows_kernel is False and sum(r["qc"]["matched"]["paired"] for r in replies if "qc" in r
                                               and r["qc"].get("matched")) > 0


# ---- real data (measured models and runs), when they are on this computer -------------------------------------------
def phase_a(*parts: str) -> Path:
    """The measured models' folder in the measuring harness's (conftest: skipped where it is not)."""
    return harness_path("resolvi-prep", "go-deeper", "phaseA", *parts)


def phase_a_model(which: str) -> Path:
    """The measured model of the crop, or of the whole sample its cells were read with (the first word of
    out/A5_cells.json's "model": the model folder's name)."""
    if which == "crop":
        return phase_a("models", "crop")
    return phase_a("models", json.loads(phase_a("out", "A5_cells.json").read_text())["model"].split()[0])


@pytest.mark.parametrize("which", ["crop", "whole"])
def test_phase_a_models_pass_the_workers_self_test(which):
    folder = phase_a_model(which)
    if not (folder / "deep.npz").exists():
        pytest.skip("the measured models are not on this computer")
    model = W.Worker(xp=np).deep_model(folder)
    assert model["selftest"]["max"] <= W.DEEP_SELFTEST_TOLERANCE and model["selftest"]["max"] < 1e-5   # torch's own


# ---- a neighbour's kind at the cut-off -----------------------------------------------------------------------------
def test_a_neighbour_near_the_same_kind_cut_off_keeps_its_class():
    """A neighbour sitting at the 0.9 cut-off switched class from one pass to the next, and the genes it gives came
    and went (17 flips in 153 frames). A neighbour classed before keeps its class until its likeness clearly crosses it
    (same from 0.92, another kind below 0.88); at first sight the cut-off decides, as the rule says."""
    similar = np.array([0.95, 0.91, 0.89, 0.87, 0.905, 0.895], np.float32)
    assert W.same_kind_classes(similar).tolist() == [True, True, False, False, True, False]
    prev = np.array([0, 0, 1, 1, -1, -1], np.int8)
    assert W.same_kind_classes(similar, prev).tolist() == [True, False, True, False, True, False]
    assert W.same_kind_classes(similar, np.ones(6, np.int8)).tolist() == [True, True, True, False, True, True]
    assert (W.DEEP_RULES["same_enter"], W.DEEP_RULES["same_at"], W.DEEP_RULES["same_leave"]) == (0.92, 0.9, 0.88)


def test_the_op_uses_the_classes_last_shown_and_says_which_it_used(setup):
    worker, out = setup["worker"], setup["out"]
    assert worker.handle(setup["start"])["ok"]
    ask = lambda c, prev=None: worker.handle({**setup["frame"], "ops": ["qc", "deeper"],                   # noqa: E731
                                              "deeper": {**setup["deeper"], "cell": c,
                                                         **({"same_prev": prev} if prev is not None else {})}})["deeper"]
    c = next(c for c in range(NCELLS) if ask(c)["state"] == "ready")
    first = ask(c)
    assert all(n["same"] == (n["similar"] >= 0.9) for n in first["neighbours"]), "first sight: the cut-off"
    # Every neighbour classed the other way before, with its likeness inside 0.88-0.92: the old class is kept.
    flipped = {str(n["cell"]): not n["same"] for n in first["neighbours"]}
    again = ask(c, flipped)
    for n in again["neighbours"]:
        inside = 0.88 <= n["similar"] < 0.92
        assert n["same"] == (flipped[str(n["cell"])] if inside else n["similar"] >= 0.92), n


def test_a_neighbours_gene_is_sent_rounded_down_and_names_the_neighbour_that_gives_most_of_it(setup):
    """D3: the chip's first number is rounded down, so a tagged gene's amount from neighbours must never be sent rounded
    up (2.96 of 3 as 3.0 read "3 of 3"); C4-2: the row names the neighbour giving most of the gene (from the whole split,
    not a neighbour's top-four list), from half a transcript on."""
    model = setup["worker"].deep_model(setup["model"])
    deep, enc = model["deep"], model["encoder"]
    counts = model_counts(setup["run"])
    rng = np.random.default_rng(5)
    seen_tag = seen_giver = 0
    for c in range(8):
        others = np.array([o for o in range(NCELLS) if o != c and counts[o].sum() >= 5])[:K]
        totals = counts.sum(1)
        centres = rng.uniform(0, 40, (NCELLS, 2))
        frame = W.DeepFrame(totals, centres, 5.0)
        fetch = lambda rows: (counts[np.asarray(rows)], np.zeros(len(rows)))            # noqa: E731
        result = W.explain_cell(c, frame, fetch, deep, enc, RUN_IDS, np.full(NCELLS, -1), None, JUDGE)
        if result["state"] != "ready":
            continue
        nbrs, d2 = frame.neighbours(c)
        r = W.deep_forward(np, deep, counts[c], counts[nbrs], d2.astype(np.float32), None, W.deep_calculator(enc, counts[c]))
        order = np.lexsort((r["genes_nz"], -r["x_nz"]))
        ranked = np.lexsort((np.arange(len(nbrs)), -r["tx_k"]))
        for row, i in zip(result["genes"], order):
            true_nb, tag = float(r["A"][i, 1]), bool(r["tag"][i])
            if tag:
                seen_tag += 1
                assert row[3] <= true_nb + 1e-9 and true_nb - row[3] < 0.001 + 1e-9, (row, true_nb)
                assert int(row[3] + 1e-6) <= true_nb                     # the chip never claims more
            else:
                assert row[3] == round(true_nb, 3)
            if true_nb >= W.DEEP_GIVER_MIN:
                seen_giver += 1
                giver = int(r["Ak"][:, i].argmax())
                assert len(row) == 6 and result["neighbours"][row[5]]["cell"] == int(nbrs[giver]), row
                assert row[5] == int(np.nonzero(ranked == giver)[0][0])
            else:
                assert len(row) == 5, row
    assert seen_giver > 0


def test_the_cell_against_itself_at_the_start_like_for_like():
    """D2: the window's "up / down from q % at the start" compared raw bytes at different depths (26 % of its "up" calls
    were depth alone). like_for_like scales the cell's averaged count now and its start-block row to the shallower of
    the two totals and scores both with the frozen calculator in one batch: no draws, so pure loss (and an unchanged
    cell) reads exactly 0 points, a real change shows, and the same rows give the same answer every time. It says the
    two numbers it compares (now, start: the cell's share from neighbours in percent, 2 decimals, both at the matched
    depth), points within 0.01 of their difference."""
    rng, start, enc = synthetic(n=400, seed=7)
    start = start * np.float32(KEEP)
    w0t = np.ascontiguousarray(enc["W0"].T)
    share = lambda x: float(W.neighbour_share(W.mixing_weights(np, np.log1p(x[None]) @ enc["W0"].T, enc))[0])  # noqa: E731
    deep_cells = [i for i in range(len(start)) if start[i].sum() >= 450][:40]
    assert len(deep_cells) >= 20
    for i in deep_cells:                                          # only lost transcripts: like for like, no change
        for keep in (1.0, 0.5, 0.25):
            now = start[i] * np.float32(keep)
            found = W.like_for_like(enc, w0t, now, start[i])
            assert {k: v for k, v in found.items() if k not in ("now", "start")} == {
                "points": 0.0, "band": W.DEEP_START_BAND, "draws": 0, "now_tx": round(float(now.sum()), 1),
                "start_tx": round(float(start[i].sum()), 1)}, found
            assert found["now"] == found["start"] == pytest.approx(100 * share(now), abs=0.006)   # the same row twice
            raw = 100 * (share(now) - share(start[i]))
            assert keep == 1.0 or raw != 0.0                    # the raw reading moves with the loss alone
    # A real change at the same depth (a cell now made of another kind's RNA) reads as one, the same every time.
    donors = [i for i in range(len(start)) if share(start[i]) >= 0.5 and start[i].sum() >= 300]
    clean = [i for i in deep_cells if share(start[i]) < 0.2]
    assert donors and clean
    j, i = donors[0], clean[0]
    now = start[j] * np.float32(min(1.0, start[i].sum() / start[j].sum()))
    changed = W.like_for_like(enc, w0t, now, start[i])
    assert changed["points"] > changed["band"] and changed == W.like_for_like(enc, w0t, now, start[i])
    depth = min(float(now.sum()), float(start[i].sum()))
    assert changed["now"] == pytest.approx(100 * share(now * np.float32(depth / float(now.sum()))), abs=0.006)
    assert changed["start"] == pytest.approx(100 * share(start[i] * np.float32(depth / float(start[i].sum()))),
                                             abs=0.006)
    assert changed["now"] == round(changed["now"], 2) and changed["start"] == round(changed["start"], 2)
    assert changed["now"] - changed["start"] > changed["band"]
    assert abs(changed["points"] - (changed["now"] - changed["start"])) <= 0.01 + 1e-9
    # Fewer than 100 on either side: nothing compared.
    assert W.like_for_like(enc, w0t, start[i] * 0 + (start[i] > 0), start[i]) is None
    assert W.like_for_like(enc, w0t, start[i], start[i] * np.float32(0.01)) is None


def test_go_deeper_runs_without_the_cycle_collector(setup, monkeypatch):
    """A collection inside the op paused it for milliseconds: the op runs with the collector off and turns it back on
    after (and leaves it off when it was off)."""
    import gc
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    seen = []
    real = W.explain_cell
    monkeypatch.setattr(W, "explain_cell", lambda *a, **k: seen.append(gc.isenabled()) or real(*a, **k))
    assert gc.isenabled()
    assert worker.handle({**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], "cell": 1}})["ok"]
    assert seen == [False] and gc.isenabled()
    gc.disable()
    try:
        with W.no_collection():
            assert not gc.isenabled()
        assert not gc.isenabled(), "left as it was"
    finally:
        gc.enable()


def test_a_cell_without_a_centre_in_the_frame_sits_at_its_starting_footprint(setup, tmp_path):
    """Go deeper finds a cell's neighbours by the centres of the frame's cell statistics; a cell without one there (gone
    from the frame, or a frame without statistics) sits at its starting footprint (start.npz, read by the check's pass):
    positions only. A frame without statistics answers as one whose statistics put every cell there."""
    from numcodecs import Zstd
    worker, out = setup["worker"], setup["out"]
    assert worker.handle(setup["start"])["ok"]
    begun = np.load(out / "start.npz")
    stats = np.zeros((7, NCELLS), np.uint32)                             # every cell at its starting footprint
    stats[5], stats[6] = begun["cx"].astype(np.float32).view(np.uint32), begun["cy"].astype(np.float32).view(np.uint32)
    (tmp_path / "start.cells.zst").write_bytes(Zstd(level=1).encode(stats.astype("<u4").tobytes()))
    strip = lambda r: {k: v for k, v in r["deeper"].items() if "ms" not in k}   # noqa: E731
    ready = 0
    for c in range(NCELLS):
        asked = {**setup["frame"], "ops": ["qc", "deeper"], "deeper": {**setup["deeper"], "cell": c}}
        without = worker.handle({**asked, "cells_file": None})
        placed = worker.handle({**asked, "cells_file": str(tmp_path / "start.cells.zst")})
        assert without["ok"] and placed["ok"] and strip(without) == strip(placed)
        if without["deeper"]["state"] == "ready":
            ready += 1
            x, y = begun["cx"].astype(np.float64), begun["cy"].astype(np.float64)
            for row in without["deeper"]["neighbours"]:
                d = row["cell"]
                assert row["um"] == round(float(np.sqrt((x[d] - x[c]) ** 2 + (y[d] - y[c]) ** 2)), 1)
    assert ready > 0


def test_a_frames_centres_are_read_once_for_go_deeper_on_it(setup, monkeypatch):
    """Go deeper on a paused (or stepped) frame asks for one cell after another without the check: the frame's centres
    and bytes come from the check's own pass of it (frame_centres, frame_code), never read from the disk again for
    each cell; a frame the check never scored here has its centres read once and its bytes from its file."""
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    reads, codes = [], []
    real, real_code = W.read_centres, W.stored_code
    monkeypatch.setattr(W, "read_centres", lambda path, n: reads.append(path) or real(path, n))
    monkeypatch.setattr(W, "stored_code", lambda path, c, n: codes.append(path) or real_code(path, c, n))
    assert worker.handle({**setup["frame"], "ops": ["qc"]})["ok"]
    answers = [worker.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c}})
               for c in range(4)]
    assert all(a["ok"] for a in answers) and reads == [] and codes == []
    assert list(worker.totals) == [("r1", 2, str(setup["model"]), NCELLS, "averaged")], \
        "the frame's totals, worked out once"
    fresh = W.Worker(xp=np)
    again = [fresh.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c}})
             for c in range(4)]
    assert reads == [setup["frame"]["cells_file"]] and codes == [setup["deeper"]["codes"]] * 4
    strip = lambda r: {k: v for k, v in r["deeper"].items() if "ms" not in k}   # noqa: E731
    assert [strip(a) for a in again] == [strip(a) for a in answers]


def test_a_cells_word_since_the_start_is_kept_near_its_band_as_last_shown(setup):
    """start_word: up or down past the band, else about the same; a word shown before is kept while the points stay on
    its side at half the band or more (a followed cell at its band flipped its word from one answer to the next). The
    op takes the word last shown from the block (start_prev) and says it in start_change.direction (held when kept)."""
    assert W.start_word(6.0, 5.0) == ("more", False) and W.start_word(-5.0, 5.0) == ("fewer", False)
    assert W.start_word(4.9, 5.0) == ("same", False) and W.start_word(4.9, 5.0, "same") == ("same", False)
    assert W.start_word(2.5, 5.0, "more") == ("more", True) and W.start_word(2.4, 5.0, "more") == ("same", False)
    assert W.start_word(-2.5, 5.0, "fewer") == ("fewer", True) and W.start_word(2.6, 5.0, "fewer") == ("same", False)
    assert W.start_word(-6.0, 5.0, "more") == ("fewer", False), "a clear change the other way is said at once"
    assert W.DEEP_START_KEEP == 0.5
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    stored = np.load(setup["out"] / "start.npz")
    seen = 0
    for c in np.argsort(-stored["totals"])[:6]:
        request = {**setup["frame"], "ops": ["qc", "deeper"],
                   "deeper": {**setup["deeper"], "cell": int(c), "start_line": True}}
        plain = worker.handle(request)["deeper"].get("start_change")
        if plain is None:
            continue
        seen += 1
        word, kept = W.start_word(plain["points"], plain["band"])
        assert plain["direction"] == word and "held" not in plain
        for prev in ("more", "fewer", "same"):
            again = worker.handle({**request, "deeper": {**request["deeper"], "start_prev": prev}})["deeper"]
            want = W.start_word(plain["points"], plain["band"], prev)
            assert again["start_change"]["direction"] == want[0] and again["start_change"].get("held", False) == want[1]
            assert again["start_change"]["points"] == plain["points"], "the word never moves the numbers"
    assert seen > 0


def test_the_init_frames_count_is_read_once_per_run(setup, monkeypatch):
    """Following a cell: the init frame's count (the start block's rows) is read once per run and kept while its file
    stays the same, so each answer reads only the frame's own file; the answer is the same as with it read again."""
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    stored = np.load(setup["out"] / "start.npz")
    c = int(np.argmax(stored["totals"]))
    read, real = [], counts_file.read
    monkeypatch.setattr(counts_file, "read", lambda path, *a, **k: read.append(Path(path).name) or real(path, *a, **k))
    request = {**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c, "start_line": True}}
    first, again = (worker.handle(request)["deeper"] for _ in range(2))
    assert "start_change" in first and read.count("000000.counts.zst") == 1 and read.count("000002.counts.zst") == 2
    assert {k: v for k, v in again.items() if "ms" not in k} == {k: v for k, v in first.items() if "ms" not in k}
    fresh = W.Worker(xp=np)
    assert fresh.handle(request)["deeper"]["start_change"] == first["start_change"]
    assert read.count("000000.counts.zst") == 2


def test_a_frame_answer_carries_its_like_for_like_change_since_the_start(setup):
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    stored = np.load(setup["out"] / "start.npz")
    seen = 0
    for c in range(NCELLS):
        reply = worker.handle({**setup["frame"], "ops": ["qc", "deeper"],
                               "deeper": {**setup["deeper"], "cell": c, "start_line": True}})
        assert reply["ok"], reply
        result = reply["deeper"]
        if result["state"] != "ready":
            continue
        if stored["totals"][c] >= JUDGE:
            seen += 1
            sc = result["start_change"]                        # write_run: the start block is the init frame's count
            assert set(sc) == {"points", "band", "draws", "now_tx", "start_tx", "now", "start", "direction"}
            assert sc["start_tx"] == round(float(stored["totals"][c]), 1) and sc["now_tx"] == round(result["transcripts"], 1)
            assert (sc["band"], sc["draws"]) == (W.DEEP_START_BAND, 0)
            # The two numbers it compares: the cell's share from neighbours now and at the start, in percent.
            assert all(0.0 <= sc[k] <= 100.0 and sc[k] == round(sc[k], 2) for k in ("now", "start"))
            assert abs(sc["points"] - (sc["now"] - sc["start"])) <= 0.01 + 1e-9
        else:
            assert "start_change" not in result, "too few at the start: nothing compared"
        plain = worker.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": c}})["deeper"]
        assert "start_change" not in plain and {k: v for k, v in plain.items() if "ms" not in k} == \
            {k: v for k, v in result.items() if "ms" not in k and k != "start_change"}
    assert seen > 0


def test_the_row_kernel_failing_falls_back_to_cupyx_rows(setup, monkeypatch):
    """When the GPU row kernel cannot compile or launch, every Go deeper answer failed (and the window retried
    every 30 s); now the cupyx row path answers, said once in the worker's log."""
    class Xp:                                         # just enough of cupy for deep_fetch's paths
        int32, float32 = np.int32, np.float32
        asarray = staticmethod(np.asarray)
        concatenate = staticmethod(np.concatenate)
        zeros = staticmethod(np.zeros)
    worker = W.Worker(xp=np)
    worker.xp = Xp()
    counts = NumpyCSR(np.arange(24, dtype=np.float32).reshape(4, 6))
    counts.indptr, counts.indices = counts.indptr.astype(np.int32), counts.indices.astype(np.int32)
    counts.data = counts.data.astype(np.float32)
    def broken(xp):
        raise RuntimeError("NVRTC could not compile")
    monkeypatch.setattr(W, "deep_rows_kernel", broken)
    logged = []
    monkeypatch.setattr(W, "log", lambda text: logged.append(text))
    index = np.array([0, 1, -1, 2, 3, -1])
    rows, outside = worker.deep_fetch(counts, index, 4)(np.array([1, 3]))
    dense = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert np.array_equal(rows, dense[[1, 3]][:, [0, 1, 3, 4]]) and np.array_equal(outside, dense[[1, 3]][:, [2, 5]].sum(1))
    assert worker.rows_kernel is False and len(logged) == 1 and "row kernel failed" in logged[0]
    worker.deep_fetch(counts, index, 4)(np.array([0]))           # the fallback from then on, said once
    assert len(logged) == 1


def test_loading_the_models_details_is_timed_apart_from_the_ops_share_of_its_pass(setup):
    """worker_ms is the op's own share of its pass; loading the model's details (once per model
    version) and reading the init frame's count (once per run) are load_ms, so the first answer is not counted as a
    pass."""
    worker = setup["worker"]
    assert worker.handle(setup["start"])["ok"]
    fresh = W.Worker(xp=np)                                      # nothing loaded, the init frame's count not read
    first = fresh.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": 0,
                                                                           "start_line": True}})["deeper"]
    again = fresh.handle({**setup["frame"], "ops": ["deeper"], "deeper": {**setup["deeper"], "cell": 0,
                                                                           "start_line": True}})["deeper"]
    assert first["load_ms"] > 0 and "load_ms" not in again
    assert first["worker_ms"] >= first["ms"] >= 0 and again["worker_ms"] >= again["ms"]
