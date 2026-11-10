"""Writes tests/data/qc_selftest.npz: torch reference vectors for the mixing check's calculator (tests/test_analysis.py).

ResolVI's per-cell own/neighbour/background split is the guide's `diffusion_encoder` (scviva/module/_resolvae.py, built
at 876-888, applied at 1003-1009): scvi's Encoder with two 128-wide hidden layers (Linear, LayerNorm without scale or
shift, ReLU), a 3-way softmax head, + 1e-3 on every part (var_eps), then + diffusion_eps (0.01) on the neighbour part.
This script builds exactly that module with the classes the sopa environment ships (scvi-tools 1.5.1, scviva-tools
0.1.7), gives it seeded random weights (the head scaled and shifted so the neighbour shares spread across the 0.30
threshold), runs 256 cells of counts through it on the CPU, and saves weights, counts and outputs in the QC model
export's format (analysis/worker.py `load_encoder`). 60 genes keep the file small; the calculator does not depend on the
panel size.

CPU only, nothing installed or written but the .npz:

    cd <folder holding sopa_companion> && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice -n 19 \
        <sopa python> -I sopa_companion/tests/data/make_qc_selftest.py
"""
from pathlib import Path

import numpy as np
import torch
from scvi.nn import Encoder

GENES, CELLS, SEED = 60, 256, 0
DIFFUSION_EPS = 0.01                          # RESOLVAEGuide(diffusion_eps=0.01), the default ResolVI uses


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    encoder = Encoder(GENES, 3, n_cat_list=None, n_layers=2, n_hidden=128, dropout_rate=0.0, inject_covariates=True,
                      use_batch_norm=False, use_layer_norm=True, var_activation=torch.nn.Softmax(dim=-1), var_eps=1e-3)
    with torch.no_grad():
        encoder.var_encoder.weight.mul_(3.0)
        encoder.var_encoder.bias.copy_(torch.tensor([1.5, -1.5, -3.0]))
    encoder.eval()
    rng = np.random.default_rng(SEED)
    profiles = rng.gamma(0.4, 1.0, (4, GENES))            # four "cell types", mixed per cell
    profiles /= profiles.sum(axis=1, keepdims=True)
    mix = rng.dirichlet(np.full(4, 0.3), CELLS)
    depth = np.concatenate([[0, 1, 5, 19, 20, 99, 100], rng.integers(20, 3000, CELLS - 7)])
    counts = rng.poisson(mix @ profiles * depth[:, None]).astype(np.int32)
    with torch.no_grad():
        x = torch.from_numpy(counts.astype(np.float32))
        _, weights, _ = encoder(torch.log1p(x), torch.zeros(CELLS, 1, dtype=torch.long))
        weights = weights.clone()
        weights[..., 1] += DIFFUSION_EPS                  # exactly as the guide does before sampling the site
    state = {k: v.detach().numpy().astype(np.float32) for k, v in encoder.state_dict().items()}
    names = {"W0": "encoder.fc_layers.Layer 0.0.weight", "b0": "encoder.fc_layers.Layer 0.0.bias",
             "W1": "encoder.fc_layers.Layer 1.0.weight", "b1": "encoder.fc_layers.Layer 1.0.bias",
             "Wv": "var_encoder.weight", "bv": "var_encoder.bias"}
    unexpected = sorted(set(state) - set(names.values()) - {"mean_encoder.weight", "mean_encoder.bias"})
    assert not unexpected, f"the encoder has parameters the export does not know: {unexpected}"
    share = (weights[:, 1] / weights.sum(dim=1)).numpy()
    out = Path(__file__).with_name("qc_selftest.npz")
    np.savez_compressed(out, **{k: state[v] for k, v in names.items()},
                        genes=np.array([f"G{i}" for i in range(GENES)]),
                        var_eps=np.float32(1e-3), diffusion_eps=np.float32(DIFFUSION_EPS), ln_eps=np.float32(1e-5),
                        selftest_counts=counts, selftest_expected=weights.numpy().astype(np.float32))
    print(f"{out}: {out.stat().st_size} bytes; torch {torch.__version__}; neighbour share "
          f"min {share.min():.3f} median {np.median(share):.3f} max {share.max():.3f}; "
          f">= 0.30 in {(share >= 0.3).mean():.0%} of cells")


if __name__ == "__main__":
    main()
