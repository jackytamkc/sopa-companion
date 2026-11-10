"""Writes tests/data/qc_deep_model/: a real, tiny mixing-check model with Go deeper's details (encoder.npz, deep.npz,
cells.parquet, model.json), learned by the app's own trainer (analysis/qc_trainer.py in its tests-only CPU mode: no GPU
visible) from the synthetic two-table sample below. Its deep.npz holds torch's own self-test outputs (noise off), so
tests/test_qc_trainer.py checks the numpy recipe against torch without the sopa environment, and other tests can read a
real export (60 genes, about 400 learning cells, 300 passes over them).

    cd <folder holding sopa_companion> && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice -n 19 \\
        <sopa python> sopa_companion/tests/data/make_qc_deep_model.py

`--sample <folder>.zarr` writes only the synthetic sample (tests/test_qc_trainer.py uses it for its CPU fits); with
`--not-proseg` its "table" holds no proseg run record (as a Xenium sample's own "table", Xenium Ranger's outline
re-count). Nothing is installed; nothing but the sample and the model folder is written; no GPU is ever visible to the
trainer.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
APP_PARENT = HERE.parents[2]                     # the folder holding the sopa_companion package
OUT = HERE / "qc_deep_model"
KEEP = ("encoder.npz", "deep.npz", "cells.parquet", "model.json")


def write_sample(root, proseg: bool = True) -> None:
    """A synthetic sample: xenium_table (420 cells of 4 kinds on a grid, each mixed with up to half of its 3 nearest
    neighbours' kinds, 12 of them under 20 transcripts) and table (380 other cells, the genes in reverse order: the
    earlier proseg cells the notebook pools its priors with, saved as a proseg run's save leaves them: its uns holds
    proseg's run record, so the trainer also scores them into batch.parquet; `proseg` False: no such record, so the
    trainer pools them but scores none). 60 genes."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    import zarr
    rng = np.random.default_rng(5)
    n_genes, kinds = 60, 4
    profiles = rng.gamma(0.3, 1.0, (kinds, n_genes))
    profiles /= profiles.sum(axis=1, keepdims=True)
    genes = np.array([f"G{i}" for i in range(n_genes)])
    gene_ids = np.array([f"ENSG{i:05d}" for i in range(n_genes)])
    zarr.open_group(str(root), mode="w")
    for key, n, seed, order in (("xenium_table", 420, 1, slice(None)), ("table", 380, 2, slice(None, None, -1))):
        r = np.random.default_rng(seed)
        xy = r.uniform(0, 320, (n, 2))
        kind = ((xy[:, 0] // 80).astype(int) + (xy[:, 1] // 80).astype(int)) % kinds
        d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nearest = np.argsort(d2, axis=1)[:, :3]
        mix = r.uniform(0, 0.5, n)[:, None]
        p = (1 - mix) * profiles[kind] + mix * profiles[kind[nearest]].mean(axis=1)
        depth = r.integers(60, 600, n)
        depth[:12] = r.integers(1, 19, 12)
        counts = r.poisson(p * depth[:, None]).astype(np.float32)
        obs = pd.DataFrame({"cell_id": [f"{key[:3]}{i}" for i in range(n)], "cell_area": np.full(n, 80.0)},
                           index=[str(i) for i in range(n)])
        var = pd.DataFrame({"gene_ids": gene_ids[order]}, index=genes[order])
        table = ad.AnnData(sp.csr_matrix(counts[:, order]), obs=obs, var=var)
        table.obsm["spatial"] = xy
        if key == "table" and proseg:
            table.uns["proseg_run"] = {"iterations": 3}
        table.write_zarr(f"{root}/tables/{key}")


def job(zarr, out, epochs: int) -> dict:
    """The engine's job for the sample's Xenium cells, with a short recipe and the tests-only CPU mode."""
    recipe = {"version": 2, "method": "scviva.model.ResolVI", "semisupervised": False, "n_latent": 10,
              "n_hidden_encoder": 128, "epochs": epochs, "min_steps": 1, "batch_size": 512, "lr": 0.003, "seed": 0,
              "n_neighbors": 10, "fit_min_counts": 20, "distance_rank": 6, "low_genes_per": 50, "priors": "pooled"}
    return {"zarr": str(zarr), "table": "xenium_table", "other_tables": ["table"], "out_dir": str(out),
            "model_id": "qc-deep-model-fixture", "sample": {"id": "zsynthetic", "name": "synthetic"},
            "recipe": recipe, "fingerprint": {"n_obs": 420, "n_vars": 60}, "mixed_at": 0.3, "judge_min": 100,
            "deadline_s": 900, "cpu_for_tests": True}


def main():
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--sample" and sys.argv[3:] in ([], ["--not-proseg"]):
        write_sample(Path(sys.argv[2]), proseg=sys.argv[3:] != ["--not-proseg"])
        return
    with tempfile.TemporaryDirectory(prefix="qc-deep-model-") as tmp:
        tmp = Path(tmp)
        write_sample(tmp / "s.zarr")
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        env.update(PYTHONPATH=str(APP_PARENT), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                   SOPA_COMPANION_ZARR_LOCK_DIR=str(tmp / "locks"))
        # stdin stays open until the reply: its end tells the trainer to stop (the engine's cancel)
        process = subprocess.Popen([sys.executable, "-m", "sopa_companion.analysis.qc_trainer"], cwd=str(APP_PARENT),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   env=env)
        process.stdin.write(json.dumps(job(tmp / "s.zarr", tmp / "model", 300)) + "\n")
        process.stdin.flush()
        last = {}
        for line in process.stdout:
            last = json.loads(line)
            if last.get("ev") in ("done", "error"):
                break
        process.stdin.close()
        process.wait(60)
        if last.get("ev") != "done":
            raise SystemExit(f"the trainer failed: {last}")
        shutil.rmtree(OUT, ignore_errors=True)
        OUT.mkdir()
        for name in KEEP:
            shutil.copy2(tmp / "model" / name, OUT / name)
        model = json.loads((OUT / "model.json").read_text())
        model["zarr"] = "synthetic (tests/data/make_qc_deep_model.py)"
        model["python"] = Path(model["python"]).name
        (OUT / "model.json").write_text(json.dumps(model, indent=1))
    deep = model["deep"]
    print(f"{OUT}: {sum((OUT / n).stat().st_size for n in KEEP)} bytes; cells {model['cells']}; deep self-test "
          f"{deep['selftest']['max_abs_diff']:.2e} (float32 {deep['selftest']['float32']['max']:.2e}); "
          f"learned values |mean| {deep['train_logit_abs_mean']:.2f}")


if __name__ == "__main__":
    main()
