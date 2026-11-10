"""What the mixing check's learning needs, looked at without learning (engine/qc.py QcModels): one JSON line on
stdout. The packages are found, not imported (scvi-tools' import takes seconds). The NVIDIA driver's own library (NVML,
through ctypes) says whether this computer has an NVIDIA card at all and which CUDA its driver supports, without torch.
torch, when installed, is imported and asked as the trainer asks (analysis/qc_trainer.py: torch.cuda.is_available(),
device_count()): the CUDA runtime starts the driver (cuInit) and makes no context, and this process has no fork to
spoil. PYTORCH_NVML_BASED_CUDA_CHECK is never set (one inherited is dropped first): it counts a card whose driver is too
old for torch's CUDA build. Without a CUDA GPU, Apple's GPU (MPS) where torch calls it available is tested as the
trainer would learn on it, in a process of its own (zarr_bridge/devices.py MPS_LEARNING_TEST: PyTorch 2.15 calls it
available on macOS 14, where every linear layer then aborts the process).

    <python> -m sopa_companion.analysis.qc_probe
    {"ev": "probe", "python": sys.executable, "platform": sys.platform, "machine": platform.machine(),
     "missing": [packages not found],
     "torch": version | null, "cuda_build": torch.version.cuda | null, "devices": torch's count | null,
     "torch_warning": text | null, "nvidia": the driver's count of NVIDIA cards | null (no NVIDIA driver library),
     "driver_cuda": "13.0" | null, "visible": CUDA_VISIBLE_DEVICES as this process got it | null,
     "mps": Apple's GPU passed the test, "mps_problem": why it did not (torch calls it available) | null,
     "error": text | null}

It imports the standard library only at its top: the engine reads PACKAGES from it.
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import sys
import warnings

# (module, package): what the trainer imports, and the package that brings it, as pip and the page name it.
PACKAGES = (("torch", "torch"), ("scvi", "scvi-tools"), ("scviva", "scviva-tools"))
# NVML's library by platform: the first name that loads (macOS has none: no NVIDIA driver there). Recent Windows drivers
# put nvml.dll where the bare name finds it; older ones in NVSMI under the Program Files folder.
NVML = {"linux": ("libnvidia-ml.so.1",),
        "win32": ("nvml.dll", "<%ProgramFiles%>/NVIDIA Corporation/NVSMI/nvml.dll")}
TEXT_LIMIT = 300                       # a warning or an error is cut to this many characters


def load_nvml():
    """The NVIDIA driver's NVML library (ctypes.CDLL of the first NVML[sys.platform] name that loads), None without
    one."""
    folder = os.environ.get("ProgramFiles", r"C:\Program Files")
    for name in NVML.get(sys.platform, ()):
        try:
            return ctypes.CDLL(name.replace("<%ProgramFiles%>", folder))
        except OSError:
            continue
    return None


def nvidia() -> tuple[int | None, str | None]:
    """(NVIDIA cards, the driver's CUDA version as "13.0") from NVML: nvmlInit_v2, nvmlDeviceGetCount_v2,
    nvmlSystemGetCudaDriverVersion_v2 (an int, 13000 -> "13.0"), nvmlShutdown whenever the start worked. (None, None)
    when no library loads, it does not start (no driver) or cannot count; a count with a None version when only the
    version call fails. Never raises. NVML ignores CUDA_VISIBLE_DEVICES: it counts the computer's cards."""
    try:
        library = load_nvml()
    except Exception:
        return None, None
    if library is None:
        return None, None
    try:
        if library.nvmlInit_v2() != 0:
            return None, None
    except Exception:
        return None, None
    cards = version = None
    try:
        count = ctypes.c_uint(0)
        if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0:
            cards = int(count.value)
            driver = ctypes.c_int(0)
            if library.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(driver)) == 0 and driver.value > 0:
                version = f"{driver.value // 1000}.{driver.value % 1000 // 10}"
    except Exception:
        pass
    finally:
        try:
            library.nvmlShutdown()
        except Exception:
            pass
    return cards, version


def _found(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def blank() -> dict:
    """The line before anything is known."""
    return {"ev": "probe", "python": sys.executable, "platform": sys.platform, "machine": platform.machine(),
            "missing": [], "torch": None,
            "cuda_build": None, "devices": None, "torch_warning": None, "nvidia": None, "driver_cuda": None,
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "mps": False, "mps_problem": None, "error": None}


def probe() -> dict:
    """The line (module docstring). torch is imported only when found; its answer is read inside
    warnings.catch_warnings(record=True), and the first warning's text (torch's "CUDA initialization: The NVIDIA driver
    on your system is too old ...") becomes torch_warning."""
    line = blank()
    line["missing"] = [package for module, package in PACKAGES if not _found(module)]
    line["nvidia"], line["driver_cuda"] = nvidia()
    if "torch" in line["missing"]:
        return line
    os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    import torch
    line["torch"] = str(torch.__version__)
    line["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line["devices"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if caught:
        line["torch_warning"] = str(caught[0].message)[:TEXT_LIMIT]
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(mps is not None and mps.is_available())
    except Exception:
        available = False
    if available and not line["devices"]:                # a CUDA GPU learns first: Apple's is asked only without one
        from ..zarr_bridge.devices import MPS_LEARNING_TEST, apple_gpu_problem
        line["mps_problem"] = apple_gpu_problem(torch, MPS_LEARNING_TEST)
        line["mps"] = line["mps_problem"] is None
    return line


def main() -> int:
    """Print the line: exit 0, also when something is missing; 1 only when looking raised (the line's "error")."""
    try:
        line, code = probe(), 0
    except Exception as exc:
        line, code = {**blank(), "error": f"{type(exc).__name__}: {exc}"[:TEXT_LIMIT]}, 1
    print(json.dumps(line), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
