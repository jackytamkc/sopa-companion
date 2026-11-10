"""A stand-in for analysis/qc_probe.py (engine tests: neither torch nor the NVIDIA driver's library is touched): one
probe line with every key of the real one, as FAKE_QC_PROBE chooses:
  ready (the default)   an NVIDIA card that a CUDA 13.0 torch sees, nothing missing
  missing:<a,b>         those packages missing (with torch among them, no torch fields)
  no_nvidia             a Mac's answer: no NVIDIA library, a torch without CUDA that sees nothing
  mac_mps               a Mac whose Apple GPU passed the probe's test (PyTorch 2.14 on macOS 14)
  mac_mps_fails         a Mac whose Apple GPU failed it (PyTorch 2.15 on macOS 14: Metal's words)
  torch_cpu             an NVIDIA card, a torch built without CUDA
  driver                an NVIDIA card whose driver supports CUDA 12.8, a CUDA 13.0 torch that sees none (its warning)
  hidden                two NVIDIA cards, none that torch sees (CUDA_VISIBLE_DEVICES hides them)
  error                 a line with "error" set, exit 1
  crash                 exit 3, no line
  garbage               prints "not json"
  hang                  sleeps 600 s
For tests the line also holds "env": the CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER it was given and whether
PYTORCH_NVML_BASED_CUDA_CHECK is set; FAKE_QC_PROBE_LOG, when set, gets one JSON line per run ({pid, mode, env}). It
never writes FAKE_QC_LOG (the fake trainer's launches, which tests count)."""
import json
import os
import sys
import time

mode = os.environ.get("FAKE_QC_PROBE") or "ready"
env = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
       "PYTORCH_NVML_BASED_CUDA_CHECK": "PYTORCH_NVML_BASED_CUDA_CHECK" in os.environ}
if os.environ.get("FAKE_QC_PROBE_LOG"):
    with open(os.environ["FAKE_QC_PROBE_LOG"], "a") as log:
        log.write(json.dumps({"pid": os.getpid(), "mode": mode, "env": env}) + "\n")
line = {"ev": "probe", "python": sys.executable, "platform": "linux", "machine": "x86_64", "missing": [],
        "torch": "2.14.1", "cuda_build": "13.0", "devices": 1, "torch_warning": None, "nvidia": 1, "driver_cuda": "13.0",
        "visible": env["CUDA_VISIBLE_DEVICES"], "mps": False, "mps_problem": None, "error": None, "env": env}
if mode.startswith("missing:"):
    line["missing"] = [name for name in mode.split(":", 1)[1].split(",") if name]
    if "torch" in line["missing"]:
        line.update(torch=None, cuda_build=None, devices=None)
elif mode == "no_nvidia":
    line.update(nvidia=None, driver_cuda=None, cuda_build=None, devices=0)
elif mode == "mac_mps":
    line.update(platform="darwin", machine="arm64", nvidia=None, driver_cuda=None, cuda_build=None, devices=0, mps=True)
elif mode == "mac_mps_fails":
    line.update(platform="darwin", machine="arm64", nvidia=None, driver_cuda=None, cuda_build=None, devices=0,
                torch="2.15.0",
                mps_problem="Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6: "
                            "-[MPSNDArray arrayViewWithShape:strides:]: unrecognized selector sent to instance 0x1")
elif mode == "torch_cpu":
    line.update(cuda_build=None, devices=0)
elif mode == "driver":
    line.update(driver_cuda="12.8", devices=0, torch_warning="CUDA initialization: The NVIDIA driver on your system is "
                                                              "too old (found version 12080)")
elif mode == "hidden":
    line.update(nvidia=2, devices=0)
elif mode == "error":
    line["error"] = "ImportError: torch could not be imported"
    print(json.dumps(line), flush=True)
    sys.exit(1)
elif mode == "crash":
    sys.exit(3)
elif mode == "garbage":
    print("not json", flush=True)
    sys.exit(0)
elif mode == "hang":
    time.sleep(600)
    sys.exit(0)
print(json.dumps(line), flush=True)
