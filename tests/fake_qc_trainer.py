"""A CPU stand-in for analysis/qc_trainer.py speaking the same JSON-lines protocol (engine tests; no GPU, no ResolVI).
On success it writes a real-format export, so the engine's checks run for real: a tiny random calculator whose reference
cells it computes with the trainer's own numpy helpers (encoder.npz), Go deeper's details with a matching self-test
(deep.npz), model.json (with "deep" and "provenance"), cells.parquet and, as the trainer does only for an other table
of proseg's own count (qc_trainer.proseg_table: the first of the job's other tables the sample has, whose uns holds
proseg's run record), batch.parquet. A job with "detail_for" (an installed model folder) writes only deep.npz and
model.json (the installed one plus "deep" from "detail", "detail" and "provenance") into out_dir, as the real trainer's
detail job does.

FAKE_QC_MODE: ok (default) | error:<reason> | crash | hang | nohello | badexport | wronggpu | wrongdevice (model.json
names the other device than the job's: Apple's GPU for a CUDA job and the other way round) | truncated (a damaged
encoder.npz) | badprogress (a malformed progress line, then it hangs) | nodeep (the calculator alone and no "deep" or
"provenance" in model.json, as models learned before Go deeper) | baddeep (a deep.npz whose self-test fails) |
detail_mismatch (a detail job refuses with reason detail_mismatch: the new learning does not match the installed model)
FAKE_QC_EPOCHS: progress lines (default 3); FAKE_QC_EPOCH_S: seconds per epoch (default 0.01)
FAKE_QC_BUS: the PCI bus the "device" reports (default 17, GPU 0 here)
FAKE_QC_LOG: a file that gets one JSON line per launch {pid, job, env, start, end}
FAKE_QC_NOT_COUNTS: table names to report as "not_counts"

The builders (calculator, synthetic_deep, deep_block, write_cells, write_model) are importable for other tests:
`from sopa_companion.tests import fake_qc_trainer as F`; the protocol runs only as `python -m ...fake_qc_trainer`.
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from sopa_companion.analysis import qc_trainer as T

GENES = [f"G{i}" for i in range(7)]
N_CELLS = 50


def calculator(seed: int = 0, genes=GENES) -> dict:
    """encoder.npz's arrays: a tiny random calculator (8 hidden) with 16 reference cells and the weights the trainer's
    numpy gives them."""
    rng = np.random.default_rng(seed)
    n = len(genes)
    enc = {"format": np.array(T.FORMAT), "genes": np.array(genes), "gene_ids": np.array([f"ID{i}" for i in range(n)]),
           "W0": rng.normal(size=(8, n)).astype(np.float32), "b0": rng.normal(size=8).astype(np.float32),
           "W1": rng.normal(size=(8, 8)).astype(np.float32), "b1": rng.normal(size=8).astype(np.float32),
           "Wv": rng.normal(size=(3, 8)).astype(np.float32), "bv": rng.normal(size=3).astype(np.float32),
           "ln_eps": np.float32(1e-5), "var_eps": np.float32(1e-3), "diffusion_eps": np.float32(0.01)}
    counts = rng.poisson(3.0, size=(16, n)).astype(np.float32)
    rows, cols = np.nonzero(counts)
    enc.update(selftest_indptr=np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=16))]).astype(np.int64),
               selftest_indices=cols.astype(np.int32), selftest_data=counts[rows, cols].astype(np.float32))
    enc["selftest_expected"] = T.mixing_weights(enc, counts.astype(np.float64), np, np.float64).astype(np.float32)
    return enc


def cell_columns(n: int = N_CELLS) -> dict:
    """cells.parquet's columns for n learning cells (ids c0..c<n-1> along a diagonal)."""
    return {"cell_id": [f"c{i}" for i in range(n)], "x": np.linspace(0, 400, n, dtype=np.float32),
            "y": np.linspace(0, 300, n, dtype=np.float32), "area": np.full(n, 50.0, np.float32),
            "counts": np.where(np.arange(n) % 5 == 0, 50, 150).astype(np.int32),
            "share": (np.arange(n) % 4 / 4.0 + 0.05).astype(np.float32)}


def write_cells(path, n: int = N_CELLS):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({k: pa.array(v) for k, v in cell_columns(n).items()}), str(path))


def synthetic_deep(enc: dict, ids, xy, seed: int = 0, hidden: int = 8, latent: int = 3, dec_hidden: int = 6,
                   k: int = 10) -> dict:
    """deep.npz's arrays (exactly qc_trainer.DEEP_KEYS) for the calculator `enc` (as written or as load_encoder gives
    it) and its learning cells (ids and centres xy, cells.parquet order): random networks, training counts and learned
    neighbour values, and a self-test block whose expected outputs are the trainer's own numpy recipe (float64, stored
    float32), so qc_trainer.deep_self_test passes and a damaged file fails."""
    rng = np.random.default_rng(seed)
    genes = [str(g) for g in (enc["genes"].tolist() if hasattr(enc["genes"], "tolist") else enc["genes"])]
    G, N = len(genes), len(ids)
    f32 = lambda a: np.asarray(a, np.float32)                     # noqa: E731
    deep = {"format": np.array(T.DEEP_FORMAT), "genes": np.array(genes),
            "Z0": f32(rng.normal(0, 0.5, (hidden, G))), "zb0": f32(rng.normal(0, 0.1, hidden)),
            "Z1": f32(rng.normal(0, 0.4, (hidden, hidden))), "zb1": f32(rng.normal(0, 0.1, hidden)),
            "Zm": f32(rng.normal(0, 0.6, (latent, hidden))), "zbm": f32(rng.normal(0, 0.1, latent)),
            "Zv": f32(rng.normal(0, 0.2, (latent, hidden))), "zbv": f32(rng.normal(0, 0.1, latent)),
            "D0": f32(rng.normal(0, 0.6, (dec_hidden, latent))), "db0": f32(rng.normal(0, 0.1, dec_hidden)),
            "D1": f32(rng.normal(0, 0.5, (dec_hidden, dec_hidden))), "db1": f32(rng.normal(0, 0.1, dec_hidden)),
            "Ds": f32(rng.normal(0, 1.0, (G, dec_hidden))), "dbs": f32(rng.normal(0, 0.3, G)),
            "ln_eps": np.float32(1e-5), "z_var_eps": np.float32(1e-4), "clamp": np.float32(T.DEEP_CLAMP),
            "diffuse_scale": np.float32(6.0), "train_ids_sha256": np.array(T.ids_sha256(ids))}
    background = np.exp(rng.normal(0, 0.3, G))
    deep["background"] = f32(background / background.sum())
    counts = rng.poisson(rng.gamma(1.0, 6.0, (N, G))).astype(np.float32)
    counts[:, 0] += 20                                             # every learning cell holds >= 20 transcripts
    deep["typical"] = f32((counts / counts.sum(axis=1, keepdims=True)).mean(axis=0))
    index, d2 = T.knn_graph(np.asarray(xy, np.float64), k)
    deep["median_distance"] = np.float32(max(float(np.median(d2[:, min(5, k - 1)])), 1.0))
    deep["train_index"] = index.astype(np.int32)
    deep["train_logit"] = f32(rng.uniform(-9.0, 9.0, (N, k)))
    deep["rank_logit"] = deep["train_logit"].mean(axis=0, dtype=np.float64).astype(np.float32)
    rows = np.unique(np.linspace(0, N - 1, min(16, N)).round().astype(int))
    block = np.concatenate([counts[[r, *index[r]]] for r in rows])
    nz_rows, nz_cols = np.nonzero(block)
    deep.update(st_rows=rows.astype(np.int32), st_d2=f32(d2[rows]),
                st_indptr=np.concatenate([[0], np.cumsum(np.bincount(nz_rows, minlength=len(block)))]).astype(np.int64),
                st_indices=nz_cols.astype(np.int32), st_data=block[nz_rows, nz_cols].astype(np.float32),
                st_own_rows=np.array([r for r in (0, 1, 8, 12) if r < len(rows)], np.int32))
    d = T._as_dtype(deep, np.float64)
    e = T._as_dtype({key: enc[key] for key in ("W0", "b0", "W1", "b1", "Wv", "bv", "ln_eps", "var_eps",
                                               "diffusion_eps")}, np.float64)
    top = min(T.DEEP_SELFTEST_GENES, G)
    z, pi, own, genes_top, split_top = [], [], [], [], []
    for i, r in enumerate(rows):
        x, xn = counts[r].astype(np.float64), counts[index[r]].astype(np.float64)
        m = T.deep_z_mean(np, d, (x / x.mean())[None])[0]
        z.append(np.concatenate([m[None], T.deep_z_mean(np, d, xn / xn.mean())]))
        pi.append(T.deep_pi(np, d, d["st_d2"][i], d["train_logit"][r]))
        if i in deep["st_own_rows"]:
            own.append(T.deep_profile(np, d, m))
        out = T.deep_forward(np, e, d, x, xn, d["st_d2"][i], d["train_logit"][r], profiles=False)
        split = np.zeros(G)
        split[out["genes_nz"]] = out["A_raw"][:, 1] / out["x_nz"]
        chosen = np.lexsort((np.arange(G), -(x * split)))[:top]
        genes_top.append(chosen)
        split_top.append(split[chosen])
    deep.update(st_z=f32(z), st_pi=f32(pi), st_own=f32(own), st_genes=np.asarray(genes_top, np.int32),
                st_nb_split=f32(split_top))
    return deep


def deep_block(path, enc: dict, source: str = "fit") -> dict:
    """model.json's "deep" block for a deep.npz, as the trainer writes it (self-tests run for real)."""
    loaded = T.load_deep(path, enc)
    parts, worker = T.deep_self_test_parts(loaded), T.deep_self_test_parts(loaded, dtype=np.float32)
    return {"format": T.DEEP_FORMAT, "file": "deep.npz", "bytes": os.path.getsize(path),
            "selftest": {"cells": int(len(loaded["st_rows"])), "max_abs_diff": parts["max"],
                         "tolerance": T.DEEP_SELFTEST_TOLERANCE, "parts": parts, "float32": worker,
                         "worker_tolerance": T.DEEP_WORKER_TOLERANCE},
            "from": source, "created": "2026-10-19T12:00:00+0100", "seconds": 0.01,
            "cells": int(loaded["train_logit"].shape[0]), "genes": len(loaded["genes"]),
            "neighbours": int(loaded["train_logit"].shape[1])}


def write_deep(out: Path, enc: dict, ids, xy, source: str = "fit", bad: bool = False) -> dict:
    """deep.npz into `out` for the calculator `enc` (in memory) and its "deep" block (bad: the self-test's stored torch
    outputs damaged afterwards, so the file loads but fails its self-test, while model.json still claims it passed)."""
    arrays = synthetic_deep(enc, ids, xy)
    np.savez(out / "deep.npz", **arrays)
    block = deep_block(out / "deep.npz", enc, source)
    if bad:
        arrays["st_z"] = arrays["st_z"] + np.float32(0.5)
        np.savez(out / "deep.npz", **arrays)
    return block


def write_model(out: Path, job: dict, epochs: int, deep: bool = True, bad_deep: bool = False,
                damage=None) -> dict:
    """A whole fit's export into `out` (damage: called with the encoder's arrays before they are saved, and with the
    path after; deep.npz is made for the undamaged calculator, as a fit would) -> model.json's content."""
    out.mkdir(parents=True, exist_ok=True)
    clean = calculator()
    enc = dict(clean)
    if damage:
        damage(enc)
    np.savez(out / "encoder.npz", **enc)
    if damage:
        damage(out / "encoder.npz")
    n = N_CELLS
    write_cells(out / "cells.parquet", n)
    others = [t for t in job.get("other_tables", []) if t != job["table"]]
    other = next((t for t in others if (Path(job["zarr"]) / "tables" / t).is_dir()), None)
    scored = other if T.proseg_table(job["zarr"], other) else None
    if scored:
        write_cells(out / "batch.parquet", n)
    model = {"format": T.FORMAT, "id": job.get("model_id"), "sample": job.get("sample"), "zarr": job["zarr"],
             "table": {"key": job["table"], **(job.get("fingerprint") or {})}, "recipe": job["recipe"], "versions": {},
             "created": "2026-10-17T12:00:00+0100", "seconds": {"total": 1.5},
             "cells": {"input": n, "fitted": n, "genes": len(GENES), "judged": 40, "mixed": 10, "per100": 25.0},
             "epochs": epochs, "steps": epochs * 8, "priors": {},
             "priors_from": ({"rule": "pooled", "tables": [job["table"], others[0]], "cells": [n, n]} if others else
                             {"rule": "table", "tables": [job["table"]], "cells": [n],
                              "why": "the sample has no other cell table"}),
             "loss": [100.0 - e for e in range(epochs)], "selftest": {"cells": 16, "max_abs_diff": 0.0},
             # the device it learned on, as the trainer records it: Apple's GPU with its kind (the engine checks it)
             "device": {"name": "fake", "pci_bus_id": None if job.get("device") == "mps" else 17,
                        **({"kind": "mps"} if job.get("device") == "mps" else {})},
             "batch": {"table": scored, "cells": n, "judged": 40, "mixed": 10, "per100": 25.0} if scored else None}
    if deep:
        cells = cell_columns(n)
        model["deep"] = write_deep(out, clean, cells["cell_id"], np.stack([cells["x"], cells["y"]], 1), "fit", bad_deep)
        model["provenance"] = T.provenance(model)
    with open(out / "model.json", "w") as stream:
        json.dump(model, stream)
    return model


def write_detail(out: Path, job: dict, bad_deep: bool = False) -> dict:
    """A detail job's output into `out`: deep.npz for the installed model's calculator and cells, and its model.json with
    "deep" (from "detail"), "detail" (identical) and "provenance" -> model.json's content."""
    import pyarrow.parquet as pq
    installed = Path(job["detail_for"])
    enc = T.load_encoder(installed / "encoder.npz")
    meta = json.loads((installed / "model.json").read_text())
    cells = pq.read_table(installed / "cells.parquet").to_pydict()
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in meta.items() if k != "deep_error"}
    meta["deep"] = write_deep(out, enc, cells["cell_id"], np.stack([cells["x"], cells["y"]], 1), "detail", bad_deep)
    meta["detail"] = {"identical": True, "same_genes": True, "max_abs_diff": {}, "reference_outputs_max_abs_diff": None,
                      "tolerance": T.DETAIL_TOLERANCE, "accepted": True, "installed": str(installed),
                      "created": "2026-10-19T12:00:00+0100", "device": {"name": "fake", "pci_bus_id": 17},
                      "epochs": meta.get("epochs"), "versions": {}, "seconds": {"total": 1.5}}
    meta["provenance"] = T.provenance(meta)
    with open(out / "model.json", "w") as stream:
        json.dump(meta, stream)
    return meta


def main():
    mode = os.environ.get("FAKE_QC_MODE", "ok")
    started = time.time()
    if mode == "nohello":
        time.sleep(60)
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    visible = "1" if mode == "wronggpu" else os.environ.get("CUDA_VISIBLE_DEVICES")

    def send(**message):
        print(json.dumps(message), flush=True)

    send(ev="hello", pid=os.getpid(), python=sys.executable, versions={}, cuda_visible_devices=visible, nice=os.nice(0),
         threads=os.environ.get("OMP_NUM_THREADS"))
    job = json.loads(sys.stdin.readline())
    log = os.environ.get("FAKE_QC_LOG")

    def record(end=None):
        if log:
            with open(log, "a") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "job": job, "start": started, "end": end,
                                         "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES",
                                                                                "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS",
                                                                                "PYTHONPATH", "DISPLAY",
                                                                                "PYTORCH_ENABLE_MPS_FALLBACK")},
                                         "nice": os.nice(0)}) + "\n")

    if job["table"] in os.environ.get("FAKE_QC_NOT_COUNTS", "").split(","):
        record(time.time())
        send(ev="error", reason="not_counts", message=f"table '{job['table']}' does not hold whole transcript counts")
        sys.exit(1)
    if mode in ("hang", "badprogress"):
        record()
        send(ev="progress", stage="learning", step=0, of=3, device={"name": "fake", "pci_bus_id": 17})
        if mode == "badprogress":
            send(ev="progress", stage="learning", step="two", of=3, seconds=1.0)
        time.sleep(600)
    epochs = int(os.environ.get("FAKE_QC_EPOCHS", "3"))
    send(ev="progress", stage="preparing", cells=4000, genes=7, epochs=epochs, steps=epochs * 8)
    send(ev="progress", stage="learning", step=0, of=epochs,
         device={"name": "fake", "pci_bus_id": int(os.environ.get("FAKE_QC_BUS", "17"))})
    for epoch in range(1, epochs + 1):
        time.sleep(float(os.environ.get("FAKE_QC_EPOCH_S", "0.01")))
        send(ev="progress", stage="learning", step=epoch, of=epochs, loss=100.0 - epoch, seconds=0.01 * epoch)
    if mode.startswith("error:"):
        record(time.time())
        send(ev="error", reason=mode.split(":", 1)[1], message="the fake trainer was told to fail")
        sys.exit(1)
    if mode == "crash":
        record(time.time())
        sys.exit(3)
    out = Path(os.path.abspath(job["out_dir"]))
    if job.get("detail_for"):
        if mode == "detail_mismatch":
            record(time.time())
            send(ev="error", reason="detail_mismatch", message="the new learning does not match the installed calculator "
                                                               "(the fake trainer was told so); nothing was changed")
            sys.exit(1)
        send(ev="progress", stage="checking")
        send(ev="progress", stage="exporting", file="deep.npz")
        model = write_detail(out, job, bad_deep=mode == "baddeep")
    else:
        send(ev="progress", stage="checking")
        send(ev="progress", stage="exporting")

        def damage(target):
            if mode == "badexport" and isinstance(target, dict):
                target["W1"] = target["W1"][:, :5]
            if mode == "truncated" and isinstance(target, Path):
                with open(target, "r+b") as stream:
                    stream.truncate(os.path.getsize(target) // 2)
        written = {**job, "device": "cuda" if job.get("device") == "mps" else "mps"} if mode == "wrongdevice" else job
        model = write_model(out, written, epochs, deep=mode != "nodeep", bad_deep=mode == "baddeep",
                            damage=damage if mode in ("badexport", "truncated") else None)
    record(time.time())
    send(ev="done", model=model)


if __name__ == "__main__":
    main()
