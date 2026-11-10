"""The CPU worker's compiled kernels (analysis/cpu.py enable imports this module on the CPU path only: numba compiles
them here, or loads them from its cache). Binomial thinning: thin_values' draws (Compare's run column where its frame
holds no sampled count), one thread per stored count. numpy's way expands every count into its transcripts (a whole
sample's frame: 20.7 M of them, 11 s); this loop draws them in place (~20 ms)."""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(parallel=True, cache=True, nogil=True)
def _thin(values, p, seed):
    n = values.shape[0]
    out = np.zeros(n, np.float32)
    mix = np.uint64(0x9E3779B97F4A7C15)
    for i in numba.prange(n):
        count = np.int64(values[i])
        kept = 0
        base = np.uint64(seed) ^ (mix * (np.uint64(i) + np.uint64(1)))
        limit = p[i]
        for t in range(count):
            key = base + mix * (np.uint64(t) + np.uint64(1))
            z = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
            if np.float32(z >> np.uint64(40)) * np.float32(1.0 / 16777216.0) < limit:
                kept += 1
        out[i] = kept
    return out


def thin(values, p, seed: int) -> np.ndarray:
    """thin_values' draws (analysis/worker.py THIN_SOURCE, SplitMix64 per (seed, entry, transcript)), bit for bit:
    float32 thinned values of whole counts `values`, each kept with its probability `p`."""
    return _thin(np.ascontiguousarray(values, np.float32).ravel(), np.ascontiguousarray(p, np.float32).ravel(),
                 np.uint64(seed))
