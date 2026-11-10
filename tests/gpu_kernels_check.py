"""Opt-in check of the GPU worker's two raw CUDA kernels against their numpy twins: the unit tests run in the app's
venv, which has no cupy, so these kernels were only checked by hand.

  * THIN_SOURCE (analysis/worker.py thin_values): Compare's binomial draw of the run's column (where its frame has no
    sampled count) must give the numpy path's draws bit for bit (float32 counts), for random counts, keep
    probabilities and seeds;
  * DEEP_ROWS_SOURCE (Worker.deep_fetch): Go deeper's rows over the model's genes and each row's sum of genes the model
    lacks must equal the numpy path's for random CSR matrices, gene maps (a fifth of the genes unknown to the model) and
    cell sets.

Not a pytest file (the gate never collects it, so nothing is ever skipped). Run it with the GPU worker's own python (the
RAPIDS env of config.toml [analysis] python) on GPU 1, with production idle:

    cd <folder holding sopa_companion> && CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID env -u DISPLAY \\
        PYTHONPATH=$PWD <[analysis] python> -m sopa_companion.tests.gpu_kernels_check

It prints one line per check and exits 1 on any difference.
"""
from __future__ import annotations

import sys

import numpy as np


def check_thinning(cp, W, rng) -> list[str]:
    problems = []
    for trial in range(40):
        n = int(rng.integers(1, 20000))
        values = rng.integers(0, 60, n).astype(np.float32)
        p = rng.uniform(0, 1, n).astype(np.float32)
        seed = int(rng.integers(0, 2 ** 63))
        on_cpu = W.thin_values(np, values, p, seed)
        on_gpu = cp.asnumpy(W.thin_values(cp, cp.asarray(values), cp.asarray(p), seed))
        if on_cpu.dtype != on_gpu.dtype or not np.array_equal(on_cpu, on_gpu):
            problems.append(f"thin_values trial {trial}: {int((on_cpu != on_gpu).sum())} of {n} draws differ")
    return problems


def check_rows(cp, cpx, W, rng) -> list[str]:
    problems = []
    for trial in range(60):
        cells, run_genes, model_genes = int(rng.integers(20, 3000)), int(rng.integers(10, 600)), None
        dense = rng.poisson(0.4, (cells, run_genes)).astype(np.float32) * (rng.random((cells, run_genes)) < 0.3)
        index = rng.permutation(run_genes).astype(np.int64)
        model_genes = run_genes + 5
        index[rng.random(run_genes) < 0.2] = -1                     # genes the model lacks
        counts_gpu = cpx.csr_matrix(cp.asarray(dense))
        counts_gpu.sum_duplicates()
        counts_gpu = cpx.csr_matrix((counts_gpu.data.astype(cp.float32), counts_gpu.indices.astype(cp.int32),
                                     counts_gpu.indptr.astype(cp.int32)), shape=counts_gpu.shape)
        rows = rng.choice(cells, size=int(rng.integers(1, 12)), replace=False)
        cpu_worker, gpu_worker = W.Worker(xp=np), W.Worker(xp=cp)
        want, want_out = cpu_worker.deep_fetch(dense, index, model_genes)(rows)
        got, got_out = gpu_worker.deep_fetch(counts_gpu, index, model_genes)(rows)
        if gpu_worker.rows_kernel is False:
            problems.append(f"deep_rows trial {trial}: the kernel failed and the fallback answered")
        if not (np.array_equal(want, got) and np.allclose(want_out, got_out, rtol=0, atol=0)):
            problems.append(f"deep_rows trial {trial}: rows differ by {float(np.abs(want - got).max()):.3g}, "
                            f"outside sums by {float(np.abs(want_out - got_out).max()):.3g}")
    return problems


def main() -> int:
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    from sopa_companion.analysis import worker as W
    rng = np.random.default_rng(20261019)
    found = {"thin_values (THIN_SOURCE) vs numpy": check_thinning(cp, W, rng),
             "deep_fetch rows (DEEP_ROWS_SOURCE) vs numpy": check_rows(cp, cpx, W, rng)}
    bad = 0
    for name, problems in found.items():
        print(f"{'ok  ' if not problems else 'FAIL'} {name}" + ("" if not problems else f": {len(problems)} problem(s)"))
        for line in problems[:5]:
            print(f"     {line}")
        bad += len(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
