"""Which device a stain helper runs torch on (zarr_bridge/stain.py Cellpose; zarr_bridge/stain_gui.py, Cellpose's
window; the dormant zarr_bridge/stain_instanseg.py): CUDA's one GPU, else Apple's GPU (MPS) where it passes a short
test, else the CPU. The mixing check's learning (analysis/qc_probe.py, analysis/qc_trainer.py) asks the same of Apple's
GPU with a test of its own operations. The standard library only: torch is the caller's."""
from __future__ import annotations

from contextlib import suppress
import os
import platform
import subprocess
import sys

# A few of the operations Cellpose's networks use (a linear layer, layer norm, attention, a convolution), on Apple's GPU,
# finished on the CPU's side. PyTorch can call MPS available where these abort the process instead (PyTorch 2.15 on
# macOS 14 calls a macOS 15 function in every linear layer), which no fallback inside the helper could catch.
MPS_TEST = """
import torch
d = torch.device("mps")
x = torch.nn.functional.layer_norm(torch.nn.Linear(32, 32).to(d)(torch.randn(4, 16, 32, device=d)), (32,))
a = torch.nn.functional.scaled_dot_product_attention(x, x, x)
c = torch.nn.functional.conv2d(torch.randn(1, 3, 16, 16, device=d), torch.randn(4, 3, 3, 3, device=d), padding=1)
print(bool(torch.isfinite(a).all().item() and torch.isfinite(c).all().item()))
"""
# The mixing check's learning (ResolVI): a linear layer and layer norm as above, and what its likelihood does at every
# step, a Gamma draw with its gradient and lgamma, whose values on broadcast shapes must also agree with the CPU's (MPS
# has given wrong ones there): "False: <why>" when they do not.
MPS_LEARNING_TEST = """
import torch
d = torch.device("mps")
x = torch.nn.functional.layer_norm(torch.nn.Linear(32, 32).to(d)(torch.randn(4, 16, 32, device=d)), (32,))
c = (torch.rand(64, 32, device=d) * 5 + 0.1).requires_grad_()
g = torch.distributions.Gamma(c, torch.rand(64, 32, device=d) + 0.1).rsample()
(x.sum() + g.sum() + torch.lgamma(c).sum()).backward()
a, b = torch.rand(256) * 10 + 0.5, torch.rand(300) * 10 + 0.5
cpu = torch.lgamma(a[:, None] + b[None, :])
gpu = torch.lgamma(a.to(d)[:, None] + b.to(d)[None, :]).cpu()
off = float(((gpu - cpu).abs() / cpu.abs().clamp(min=1.0)).max())
if not (torch.isfinite(x).all().item() and torch.isfinite(c.grad).all().item()):
    print("False: its values are not finite")
elif not off < 1e-3:
    print(f"False: its lgamma differs from the CPU's by {off:.2g}")
else:
    print(True)
"""
MPS_TEST_S = 120                 # Cellpose's helpers: no other wait around them
MPS_LEARNING_TEST_S = 60         # within the engine's wait for the mixing check's probe (PROBE_TIMEOUT_S, 120)
_tested: dict[str, str | None] = {}         # this process's answers, by test: each test runs once


def apple_gpu_problem(torch, test: str = MPS_TEST) -> str | None:
    """Whether Apple's GPU runs `test` (MPS_TEST, or MPS_LEARNING_TEST for the mixing check), in a process of its own
    (about half a second), as the helpers run (PYTORCH_ENABLE_MPS_FALLBACK=1: an operation MPS lacks moves to the CPU):
    None when it does, else why not, in words. The process's own words when it aborts are Metal's ("reason: ...")."""
    key = "mps" if test == MPS_TEST else f"mps:{hash(test)}"
    if key in _tested:
        return _tested[key]
    limit = MPS_LEARNING_TEST_S if test == MPS_LEARNING_TEST else MPS_TEST_S
    where = f"PyTorch {getattr(torch, '__version__', '?')} on macOS {platform.mac_ver()[0] or '?'}"
    try:
        done = subprocess.run([sys.executable, "-P", "-B", "-c", test], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=limit,
                              env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    except subprocess.TimeoutExpired:
        why = f"Apple's GPU did not finish a short test within {limit} s ({where})"
    except OSError as exc:
        why = f"Apple's GPU could not be tested ({where}): {exc.strerror or exc}"
    else:
        last = (done.stdout.strip().splitlines() or [""])[-1].strip()
        if done.returncode == 0 and last == "True":
            why = None
        elif done.returncode == 0 and last.startswith("False"):
            why = f"Apple's GPU failed a short test with {where}: {last[5:].strip(': ') or 'its values are not finite'}"
        else:
            lines = [line.strip() for line in done.stderr.splitlines() if line.strip()]
            told = next((line.split("reason:", 1)[1].strip(" '") for line in lines if "reason:" in line), None)
            told = told or (lines[-1] if lines else f"exit {done.returncode}")
            why = f"Apple's GPU failed a short test with {where}: {told[:300]}"
    _tested[key] = why
    return why


def choose_device(torch) -> tuple[str, bool, str | None]:
    """The device a stain helper runs torch on: "cuda:0" (the one GPU it sees: [stain] gpu), else "mps" (Apple's GPU)
    where it passes the test, else "cpu"; whether it is a GPU; and why Apple's GPU, there, is not used (else None). A
    torch without either (or a fake one in a test) gives the CPU."""
    with suppress(Exception):
        if torch.cuda.is_available():
            return "cuda:0", True, None
    available = False
    with suppress(Exception):
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        available = mps is not None and bool(mps.is_available())
    if available:
        why = apple_gpu_problem(torch)
        return ("cpu", False, why) if why else ("mps", True, None)
    return "cpu", False, None


def pick_device(torch) -> tuple[str, bool]:
    """choose_device without its reason."""
    return choose_device(torch)[:2]


def empty_cache(torch, device: str) -> None:
    """Free the device's cached blocks after a fallback (cuda, or mps); nothing on the CPU; never raises."""
    with suppress(Exception):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(device).startswith("mps"):
            torch.mps.empty_cache()
