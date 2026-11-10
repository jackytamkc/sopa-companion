"""Can the mixing check learn on a Mac's own GPU (Apple's MPS)? Run it on a Mac with Apple silicon, in the environment
where the app is installed with the mixing check's packages (the README's Install), from any folder:

    python mps_probe.py [--cells 20000] [--genes 500] [--epochs 5]

1. The operations ResolVI uses at every learning step, on Apple's GPU beside the CPU: whether each runs there at all
   (else PyTorch would have to move it to the CPU, PYTORCH_ENABLE_MPS_FALLBACK=1) and whether its values agree with the
   CPU's (lgamma has given wrong values there for some shapes).
2. The app's own recipe (sopa_companion.analysis.qc_trainer: recipe, knn_graph, pooled_priors, ResolVI as the trainer
   builds it) on a synthetic sample, learned for a few passes on the CPU and on Apple's GPU, each in a process of its
   own: seconds per pass, NaNs, the loss reached, and the cells' neighbour shares learned on each (from different
   random draws, so their spread is compared, not each cell). On Apple's GPU a tiny fit (1,000 cells, 100 genes, one
   pass) comes first, so a failure at both sizes points at an operation and one at the full size only at a limit.
   When a fit crashes, it says why (Metal's reason, an assertion, a Python error) and where (the Python stack when the
   process stopped).

Prints what it finds and, last, one JSON line. Nothing is written anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import textwrap

OPS = r"""
import json, math, sys
import torch
results = {}
dev = sys.argv[1]
torch.manual_seed(0)

def compare(name, fn, tolerance):
    # Relative for large values, absolute near zero (lgamma crosses zero at 1 and 2, where a relative error means
    # nothing): |a - b| / max(|b|, 1).
    try:
        on_dev = fn(dev)
        on_cpu = fn("cpu")
        a, b = on_dev.detach().float().cpu(), on_cpu.detach().float()
        finite = bool(torch.isfinite(a).all())
        error = float(((a - b).abs() / b.abs().clamp(min=1.0)).max())
        results[name] = {"runs": True, "finite": finite, "max_abs_error": float((a - b).abs().max()),
                         "max_scaled_error": error, "agrees": finite and error < tolerance}
    except Exception as exc:
        results[name] = {"runs": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}

def runs(name, fn):
    try:
        out = fn(dev)
        results[name] = {"runs": True, "finite": bool(torch.isfinite(out.detach().float().cpu()).all())}
    except Exception as exc:
        results[name] = {"runs": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}

g = torch.Generator().manual_seed(1)
conc = torch.rand(512, 500, generator=g) * 5 + 0.1
rate = torch.rand(512, 500, generator=g) * 2 + 0.1
counts = torch.poisson(torch.rand(512, 500, generator=g) * 4)
probs = torch.softmax(torch.randn(512, 11, generator=g), dim=-1)
a, b = torch.rand(512, generator=g) * 10 + 0.5, torch.rand(500, generator=g) * 10 + 0.5

def gamma_step(d):
    c = conc.to(d).requires_grad_()
    sample = torch.distributions.Gamma(c, rate.to(d)).rsample()
    sample.sum().backward()
    return c.grad
runs("gamma rsample and its gradient (the NB likelihood, every step)", gamma_step)
compare("lgamma, broadcast shapes", lambda d: torch.lgamma(a.to(d)[:, None] + b.to(d)[None, :]), 1e-3)
compare("lgamma, plain", lambda d: torch.lgamma(conc.to(d)), 1e-3)
compare("Poisson log_prob", lambda d: torch.distributions.Poisson(rate.to(d)).log_prob(counts.to(d)), 1e-3)
compare("Gamma log_prob", lambda d: torch.distributions.Gamma(conc.to(d), rate.to(d)).log_prob(rate.to(d)), 1e-3)
compare("Dirichlet log_prob", lambda d: torch.distributions.Dirichlet(conc[:, :11].to(d)).log_prob(probs.to(d)), 1e-3)
runs("Multinomial sample (downsampling)",
     lambda d: torch.distributions.Multinomial(total_count=50, probs=probs.to(d)).sample())
print(json.dumps(results))
"""

FIT = r"""
import json, math, sys, time, warnings
import numpy as np, scipy.sparse as sp
import anndata, torch
from sopa_companion.analysis import qc_trainer as T
cells, genes, epochs, device = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fallbacks = []
warnings.simplefilter("always")
def note(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if "fall back" in text or "fallback" in text.lower():
        fallbacks.append(text.split("'")[1] if "'" in text else text[:120])
warnings.showwarning = note

# A synthetic sample: cell types with their own genes, each cell also holding some of a neighbour's RNA.
rng = np.random.default_rng(0)
side = math.sqrt(cells) * 12.0
xy = rng.uniform(0, side, (cells, 2)).astype(np.float64)
types = rng.integers(0, 8, cells)
programs = rng.lognormal(0, 1.2, (8, genes)) * (rng.random((8, genes)) < 0.3)
library = rng.lognormal(math.log(80), 0.4, cells)
own = programs[types] / programs[types].sum(axis=1, keepdims=True)
indices, distances2 = T.knn_graph(xy, k=10)
mix = rng.beta(1, 6, cells)[:, None]
expected = library[:, None] * ((1 - mix) * own + mix * own[indices[:, 0]])
counts = sp.csr_matrix(rng.poisson(expected).astype(np.float32))
rec = T.recipe(epochs=epochs, min_steps=1)
totals = np.asarray(counts.sum(axis=0)).ravel()
keep = totals > 0
counts = counts[:, keep]
priors = T.pooled_priors([(counts, distances2)], totals[keep], rec)

import scvi
ResolVI = T.resolvi_class()
scvi.settings.dl_num_workers = 0
# Dense counts, on both devices alike: ResolVI fetches each cell's neighbours' counts as a sparse tensor when its X is
# sparse (load_sparse_tensor=True), and Apple's GPU holds no sparse tensor (NotImplementedError: new_compressed_tensor).
adata = anndata.AnnData(X=counts.toarray())
adata.obs_names = [str(i) for i in range(cells)]
adata.var_names = [f"g{i}" for i in range(counts.shape[1])]
adata.obsm["X_spatial"] = xy.astype(np.float32)
adata.obsm["index_neighbor"] = indices
adata.obsm["distance_neighbor"] = distances2
scvi.settings.seed = 0
torch.manual_seed(0)
ResolVI.setup_anndata(adata, prepare_data=False)
model = ResolVI(adata, semisupervised=False, n_latent=int(rec["n_latent"]), n_hidden_encoder=int(rec["n_hidden_encoder"]),
                **priors)
started = time.monotonic()
model.train(max_epochs=epochs, batch_size=int(rec["batch_size"]), lr=float(rec["lr"]), accelerator=device,
            device="auto", enable_checkpointing=False, enable_progress_bar=False)
if device == "mps":
    torch.mps.synchronize()
seconds = time.monotonic() - started
history = model.history.get("elbo_train")
curve = [float(v) for v in history.iloc[:, 0].to_numpy()] if history is not None else []
encoder = model.module.guide.diffusion_encoder
dev = next(encoder.parameters()).device
shares = []
with torch.no_grad():
    for start in range(0, cells, 4096):
        xb = torch.as_tensor(counts[start:start + 4096].toarray(), device=dev)
        _, w, _ = encoder(torch.log1p(xb), torch.zeros((xb.shape[0], 1), device=dev))
        shares.append(w.float().cpu().numpy())
share = T.neighbour_share(np.concatenate(shares).astype(np.float64))
print(json.dumps({"device": device, "cells": cells, "genes": int(counts.shape[1]), "epochs": epochs,
                  "seconds_per_pass": round(seconds / epochs, 3), "loss_first": curve[0] if curve else None,
                  "loss_last": curve[-1] if curve else None, "nan": bool(not np.isfinite(curve).all() or
                                                                        not np.isfinite(share).all()),
                  "share_mean": float(np.mean(share)), "share_q90": float(np.quantile(share, 0.9)),
                  "mixed_at_0.3": float(np.mean(share >= 0.3)), "fallbacks": sorted(set(fallbacks))}))
"""


WHY = ("reason", "Error", "error:", "Exception", "ssertion", "MPS", "Metal", "MTL", "mps")


def crash_report(stderr: str) -> dict:
    """What a failed child left on stderr: why (a Python error's last line, or a crash's own words: the "reason:" of an
    Objective-C exception, an assertion, the Metal and MPS frames of its throw stack) and where (the Python stack
    faulthandler prints when the process aborts, most recent call first)."""
    errors = [line.strip()[:400] for line in stderr.splitlines() if line.strip()]
    why = [line for line in errors if not line.startswith('File "') and any(word in line for word in WHY)]
    where, inside = [], False
    for line in errors:
        if line.startswith("Current thread"):
            where, inside = [], True
        elif inside and line.startswith('File "'):
            where.append(line)
        elif inside:
            inside = False
    told = ([line[line.find("***"):] if "***" in line else line for line in errors if "reason:" in line] +
            [line for line in errors if "failed assertion" in line] +
            [line for line in errors[::-1] if re.match(r"[\w.]+(Error|Exception|Interrupt)\b", line)])
    return {"error": (told or errors or ["no output"])[0], "why": why[-10:], "where": where[:15],
            "stderr_tail": errors[-25:]}


def run(code: str, *args, fallback: bool = False) -> dict:
    environment = {key: value for key, value in os.environ.items() if key != "PYTORCH_ENABLE_MPS_FALLBACK"}
    if fallback:
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # An uncaught Objective-C exception (Metal's) prints its reason through CFLog: to stderr, also through a pipe.
    environment["CFLOG_FORCE_STDERR"] = "1"
    done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code, *map(str, args)], capture_output=True,
                          text=True, env=environment, timeout=3600)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        return {"failed": True, "exit": done.returncode, **crash_report(done.stderr)}
    return json.loads(lines[-1])


def explain(label: str, result: dict) -> None:
    if not result.get("failed"):
        return
    print(f"  why the {label} failed:")
    for line in result.get("why") or result.get("stderr_tail", [])[-8:]:
        print("    " + line)
    if result.get("where"):
        print("  where (the Python stack when it stopped, most recent call first):")
        for line in result["where"]:
            print("    " + line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=20000)
    parser.add_argument("--genes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    import torch
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    found = {"python": sys.version.split()[0], "torch": torch.__version__, "machine": platform.machine(),
             "macos": platform.mac_ver()[0] or None, "mps_available": mps}
    print(f"PyTorch {torch.__version__} on {platform.platform()}; Apple's GPU (MPS) available: {mps}")
    if mps:
        found["ops"] = run(OPS, "mps")
        for name, result in found["ops"].items() if isinstance(found["ops"], dict) else ():
            if not isinstance(result, dict):
                continue
            state = ("runs" if result.get("runs") else "MISSING: " + result.get("error", "")) + \
                ("" if "agrees" not in result else
                 f", agrees with the CPU (largest difference {result['max_abs_error']:.2g})" if result["agrees"] else
                 f", DIFFERS from the CPU (largest difference {result['max_abs_error']:.2g}, "
                 f"{result['max_scaled_error']:.2g} of the value)")
            print(f"  {name}: {state}")
    found["cpu"] = run(FIT, args.cells, args.genes, args.epochs, "cpu")
    print("CPU:", json.dumps(found["cpu"]))
    if mps:
        # A tiny fit first: whether it fails there too (an operation) or only at the full size (a limit).
        found["mps_tiny"] = run(FIT, 1000, 100, 1, "mps", fallback=True)
        print("Apple's GPU, a tiny fit (1,000 cells, 100 genes, 1 pass):",
              "learned" if not found["mps_tiny"].get("failed") else "FAILED")
        found["mps"] = run(FIT, args.cells, args.genes, args.epochs, "mps", fallback=True)
        print("Apple's GPU:", json.dumps({k: v for k, v in found["mps"].items() if k not in ("why", "where", "stderr_tail")}))
        explain("tiny fit", found["mps_tiny"])
        explain("full fit", found["mps"])
        cpu, gpu = found["cpu"], found["mps"]
        if not cpu.get("failed") and not gpu.get("failed"):
            print(textwrap.dedent(f"""\
                Apple's GPU learned {cpu['seconds_per_pass'] / gpu['seconds_per_pass']:.1f}x as fast as the CPU per pass;
                operations moved to the CPU: {', '.join(gpu['fallbacks']) or 'none'}; NaN: {gpu['nan']};
                share of cells looking mixed (>= 0.3): {gpu['mixed_at_0.3']:.3f} here, {cpu['mixed_at_0.3']:.3f} on the CPU."""))
    else:
        print("No Apple GPU here: only the CPU's numbers (run this on a Mac with Apple silicon).")
    print(json.dumps(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
