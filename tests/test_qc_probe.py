"""The mixing check's probe (analysis/qc_probe.py) and what its answer means (engine/qc.py capability_of, device_of),
without a GPU: the reasons in the order that helps (no NVIDIA card first, a Mac's Apple GPU then, the packages last; no
answer keeps the check offered), the
NVIDIA driver's library read through a fake one, torch asked as the trainer asks through a fake torch, and the one line
it prints, in the app's venv and, where it is, in the sopa environment with no GPU visible."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types
import warnings

from conftest import needs_sopa, sopa_python
from sopa_companion.analysis import qc_probe as P
from sopa_companion.engine.qc import capability_of, device_of

APP_PARENT = str(Path(__file__).resolve().parents[2])      # the folder holding the sopa_companion package


def answer(**fields) -> dict:
    """A probe line: a CUDA 13.0 torch that sees one NVIDIA card, nothing missing, with `fields` over it."""
    return {"ev": "probe", "python": "/envs/sopa/bin/python", "platform": "linux", "machine": "x86_64", "missing": [],
            "torch": "2.14.1", "cuda_build": "13.0", "devices": 1, "torch_warning": None, "nvidia": 1,
            "driver_cuda": "13.0", "visible": "0", "mps": False, "mps_problem": None, "error": None, **fields}


def mac(**fields) -> dict:
    """A Mac's probe line: no NVIDIA library, a torch without CUDA that sees nothing, with `fields` over it."""
    return answer(**{"platform": "darwin", "machine": "arm64", "nvidia": None, "driver_cuda": None, "cuda_build": None,
                     "devices": 0, **fields})


def test_the_reasons_come_in_the_order_that_helps():
    """First match wins: no answer keeps the check offered ("unknown"); then no NVIDIA card (every Mac, and Windows with
    PyPI's CPU-only torch and no card: never sent to install a CUDA build or the packages), torch missing, a torch built
    without CUDA, a driver older than torch's CUDA, a card torch cannot see (its warning said), the packages last."""
    rows = [
        (None, ("unknown", None)), (answer(error="ImportError: torch"), ("unknown", None)),
        # a Mac: no NVIDIA library, a torch without CUDA that sees nothing, and no scvi-tools either
        (answer(nvidia=None, driver_cuda=None, cuda_build=None, devices=0, missing=["scvi-tools", "scviva-tools"]),
         ("unavailable", "no_gpu")),
        # Windows with PyPI's CPU-only torch and no card
        (answer(nvidia=None, driver_cuda=None, cuda_build=None, devices=0), ("unavailable", "no_gpu")),
        (answer(nvidia=0, driver_cuda="13.0", cuda_build=None, devices=0), ("unavailable", "no_gpu")),
        # an NVIDIA card with a CPU-only torch; with no torch at all
        (answer(cuda_build=None, devices=0), ("unavailable", "torch_cpu")),
        (answer(missing=["torch", "scvi-tools", "scviva-tools"], torch=None, cuda_build=None, devices=None),
         ("unavailable", "software")),
        # torch's CUDA 13.0 on a driver that supports 12.8; on a 13.0 driver with no card torch can see
        (answer(driver_cuda="12.8", devices=0, torch_warning="CUDA initialization: The NVIDIA driver on your system is "
                                                              "too old (found version 12080)"), ("unavailable", "driver")),
        (answer(devices=0, nvidia=2, visible="", torch_warning="CUDA initialization: no device"),
         ("unavailable", "no_gpu")),
        # a CUDA torch that sees one card: the packages decide
        (answer(missing=["scvi-tools"]), ("unavailable", "software")),
        (answer(), ("ready", None)),
        # NVML missing (no library found) but torch sees a card: never refused for it
        (answer(nvidia=None, driver_cuda=None), ("ready", None)),
        # A Mac whose Apple GPU passed the probe's test: ready, unless the packages are missing; one whose test failed
        # (PyTorch 2.15 on macOS 14): "apple_gpu"; without torch there: the packages first (it could learn there)
        (mac(mps=True), ("ready", None)),
        (mac(mps=True, missing=["scvi-tools", "scviva-tools"]), ("unavailable", "software")),
        (mac(mps_problem="Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6: -[MPSNDArray ...]"),
         ("unavailable", "apple_gpu")),
        (mac(missing=["torch", "scvi-tools", "scviva-tools"], torch=None, devices=None), ("unavailable", "software")),
        # an Intel Mac without torch: no GPU the check could ever learn on there, said first
        (mac(machine="x86_64", missing=["torch", "scvi-tools", "scviva-tools"], torch=None, devices=None),
         ("unavailable", "no_gpu")),
        (mac(), ("unavailable", "no_gpu")),          # no GPU that torch calls available (an Intel Mac)
    ]
    for line, expected in rows:
        state, reason, why = capability_of(line)
        assert (state, reason) == expected, (line, state, reason, why)
        assert isinstance(why, str) and why
    assert capability_of(None)[2] == "its Python could not be asked (no answer): offered, asked again later"
    assert capability_of(answer(error="ImportError: torch"))[2].startswith("its Python could not be asked (ImportError")
    assert capability_of(answer(nvidia=None, devices=0, cuda_build=None))[2] == "this computer has no NVIDIA graphics card"
    assert capability_of(answer(cuda_build=None))[2] == "torch 2.14.1 was built without CUDA"
    assert capability_of(answer(driver_cuda="12.8"))[2] == \
        "the NVIDIA driver supports CUDA 12.8, older than torch 2.14.1's CUDA 13.0"
    hidden = capability_of(answer(devices=0, visible=""))[2]
    assert hidden == "torch sees no CUDA GPU (CUDA_VISIBLE_DEVICES='')"
    assert "too old" in capability_of(answer(devices=0, torch_warning="CUDA initialization: too old"))[2]
    assert capability_of(answer(missing=["scvi-tools", "scviva-tools"]))[2] == \
        "scvi-tools and scviva-tools are not installed in /envs/sopa/bin/python"
    assert capability_of(answer(missing=["torch"], torch=None, cuda_build=None, devices=None))[2] == \
        "torch is not installed in /envs/sopa/bin/python"
    assert capability_of(answer(devices=2))[2] == "torch 2.14.1 sees 2 CUDA GPU(s)"
    # A driver version it cannot read is no reason to refuse.
    assert capability_of(answer(driver_cuda="unknown"))[:2] == ("ready", None)
    assert capability_of(mac(mps=True))[2] == "torch 2.14.1 learns on Apple's GPU (MPS): it passed the probe's test"
    assert capability_of(mac(mps_problem="Apple's GPU failed a short test"))[2] == "Apple's GPU failed a short test"


def test_the_device_is_apples_gpu_only_without_a_cuda_gpu_and_after_its_test():
    """device_of: "mps" when no CUDA GPU is seen and Apple's GPU passed the probe's test; "cuda" otherwise, also with no
    answer (as without the probe) and for a CUDA computer whose torch would call MPS available too."""
    assert device_of(mac(mps=True)) == "mps"
    assert device_of(mac(mps_problem="failed")) == device_of(mac()) == "cuda"
    assert device_of(answer()) == device_of(answer(mps=True)) == device_of(None) == device_of({}) == "cuda"


class FakeNvml:
    """NVML's four calls as a library answers them, through their ctypes arguments: `fail` names the one that fails."""

    def __init__(self, cards=2, version=13000, fail=None):
        self.cards, self.version, self.fail, self.calls = cards, version, fail, []

    def _call(self, name, arg=None, value=None):
        self.calls.append(name)
        if self.fail == name:
            return 999
        if arg is not None:                          # ctypes.byref(x): the call fills x
            arg._obj.value = value
        return 0

    def nvmlInit_v2(self):
        return self._call("nvmlInit_v2")

    def nvmlDeviceGetCount_v2(self, count):
        return self._call("nvmlDeviceGetCount_v2", count, self.cards)

    def nvmlSystemGetCudaDriverVersion_v2(self, version):
        return self._call("nvmlSystemGetCudaDriverVersion_v2", version, self.version)

    def nvmlShutdown(self):
        return self._call("nvmlShutdown")


def test_nvml_is_read_without_torch(monkeypatch):
    """The cards and the driver's CUDA come from the driver's own library: (None, None) without one or when it does not
    start, a count without a version when only that call fails; it is always shut down once started, and never raises."""
    real = P.load_nvml
    library = FakeNvml()
    monkeypatch.setattr(P, "load_nvml", lambda: library)
    assert P.nvidia() == (2, "13.0") and library.calls[-1] == "nvmlShutdown"
    library = FakeNvml(version=12080)
    monkeypatch.setattr(P, "load_nvml", lambda: library)
    assert P.nvidia() == (2, "12.8")
    library = FakeNvml(fail="nvmlInit_v2")
    monkeypatch.setattr(P, "load_nvml", lambda: library)
    assert P.nvidia() == (None, None) and "nvmlShutdown" not in library.calls
    monkeypatch.setattr(P, "load_nvml", lambda: None)
    assert P.nvidia() == (None, None)
    library = FakeNvml(fail="nvmlSystemGetCudaDriverVersion_v2")
    monkeypatch.setattr(P, "load_nvml", lambda: library)
    assert P.nvidia() == (2, None) and library.calls[-1] == "nvmlShutdown"
    library = FakeNvml(fail="nvmlDeviceGetCount_v2")
    monkeypatch.setattr(P, "load_nvml", lambda: library)
    assert P.nvidia() == (None, None) and library.calls[-1] == "nvmlShutdown"

    def broken():
        raise OSError("no such library")
    monkeypatch.setattr(P, "load_nvml", broken)
    assert P.nvidia() == (None, None)
    # Its library by platform: Linux's name; none on a Mac (no NVIDIA driver there).
    assert P.NVML["linux"] == ("libnvidia-ml.so.1",) and "darwin" not in P.NVML
    monkeypatch.setattr(P.sys, "platform", "darwin")
    assert real() is None                                    # nothing is loaded, nothing called


def test_the_probe_asks_torch_as_the_trainer_does(monkeypatch):
    """torch's own CUDA check, as the trainer asks it (is_available, device_count), its warning kept: a driver too old
    for torch's CUDA build sees no card. PYTORCH_NVML_BASED_CUDA_CHECK is never set (one inherited is dropped first)."""
    asked = []

    def available():
        asked.append("is_available")
        warnings.warn("CUDA initialization: The NVIDIA driver on your system is too old (found version 12080)")
        return False

    def count():
        asked.append("device_count")
        return 0
    torch = types.SimpleNamespace(__version__="2.14.1", version=types.SimpleNamespace(cuda="13.0"),
                                  cuda=types.SimpleNamespace(is_available=available, device_count=count),
                                  backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(P, "_found", lambda module: True)
    monkeypatch.setattr(P, "nvidia", lambda: (1, "12.8"))
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    line = P.probe()
    assert asked == ["is_available"] and "PYTORCH_NVML_BASED_CUDA_CHECK" not in os.environ
    assert line == {"ev": "probe", "python": sys.executable, "platform": sys.platform, "machine": platform.machine(),
                    "missing": [], "torch": "2.14.1",
                    "cuda_build": "13.0", "devices": 0,
                    "torch_warning": "CUDA initialization: The NVIDIA driver on your system is too old "
                                     "(found version 12080)",
                    "nvidia": 1, "driver_cuda": "12.8", "visible": "0", "mps": False, "mps_problem": None, "error": None}
    assert capability_of(line)[:2] == ("unavailable", "driver")
    # A torch that sees its card: counted as the trainer counts it.
    torch.cuda.is_available, torch.cuda.device_count = (lambda: True), (lambda: 1)
    assert P.probe()["devices"] == 1 and P.probe()["torch_warning"] is None
    # Apple's GPU, where torch calls it available, is tested only without a CUDA GPU: as the trainer would learn on it,
    # in a process of its own (zarr_bridge/devices.py MPS_LEARNING_TEST); its words when it fails.
    from sopa_companion.zarr_bridge import devices
    tested = []
    torch.backends.mps.is_available = lambda: True
    monkeypatch.setattr(devices, "apple_gpu_problem", lambda torch_, test: tested.append(test) or None)
    assert P.probe()["mps"] is False and tested == []                    # the CUDA GPU learns
    torch.cuda.is_available, torch.cuda.device_count = (lambda: False), (lambda: 0)
    line = P.probe()
    assert (line["mps"], line["mps_problem"], tested) == (True, None, [devices.MPS_LEARNING_TEST])
    monkeypatch.setattr(devices, "apple_gpu_problem", lambda torch_, test: "Apple's GPU failed a short test")
    line = P.probe()
    assert (line["mps"], line["mps_problem"]) == (False, "Apple's GPU failed a short test")
    # Without torch nothing is imported: the packages are found, not loaded.
    monkeypatch.setattr(P, "_found", lambda module: module != "torch")
    monkeypatch.delitem(sys.modules, "torch")
    assert P.probe()["missing"] == ["torch"] and P.probe()["torch"] is None and "torch" not in sys.modules


def run_probe(python, env, before: str = "") -> subprocess.CompletedProcess:
    """The probe's own process, NVML's read left out (no call reaches the NVIDIA driver from a test); `before` runs
    first."""
    code = ("import sopa_companion.analysis.qc_probe as q; q.nvidia = lambda: (None, None); " + before +
            "raise SystemExit(q.main())")
    return subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**env, "PYTHONPATH": APP_PARENT, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})


def test_the_probe_answers_in_one_line(tmp_path):
    """One JSON line on stdout, exit 0 also when packages are missing: those this Python lacks (the app's venv has none
    of the three, CI's one environment has them all) and one that no Python has."""
    done = run_probe(sys.executable, os.environ, 'q.PACKAGES += (("sopa_companion_absent", "absent-package"),); ')
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    lacking = [package for module, package in P.PACKAGES if importlib.util.find_spec(module) is None]
    assert line["ev"] == "probe" and line["missing"] == [*lacking, "absent-package"]
    if "torch" in lacking:
        assert line["torch"] is None and line["devices"] is None
    else:                                                       # asked as the trainer asks it, no GPU visible to it
        assert isinstance(line["torch"], str) and line["devices"] == 0
    assert line["error"] is None and line["visible"] == ""
    assert set(line) == {"ev", "python", "platform", "machine", "missing", "torch", "cuda_build", "devices",
                         "torch_warning", "nvidia", "driver_cuda", "visible", "mps", "mps_problem", "error"}
    # Looking that raises: the line says so, exit 1 (the engine keeps the check offered and asks again later).
    code = ("import sopa_companion.analysis.qc_probe as q\n"
            "def broken(): raise RuntimeError('broken')\n"
            "q.probe = broken\nraise SystemExit(q.main())\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": APP_PARENT})
    assert done.returncode == 1 and json.loads(done.stdout)["error"] == "RuntimeError: broken"


@needs_sopa
def test_the_probe_in_the_sopa_environment_sees_no_hidden_gpu():
    """In the trainer's environment with no GPU visible: every package found, torch and its CUDA build named as torch
    names them there (this computer's a CUDA build, CI's CPU-only torch none), no card seen."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = run_probe(sopa_python(), env)
    assert done.returncode == 0, done.stderr
    line = json.loads(done.stdout.strip().splitlines()[-1])
    own = subprocess.run([sopa_python(), "-I", "-c", "import json, torch; print(json.dumps([str(torch.__version__), "
                          "torch.version.cuda]))"], capture_output=True, text=True, timeout=120,
                         env={**env, "CUDA_VISIBLE_DEVICES": ""})
    assert own.returncode == 0, own.stderr
    assert line["missing"] == [] and line["devices"] == 0
    assert [line["torch"], line["cuda_build"]] == json.loads(own.stdout.strip().splitlines()[-1])
    assert capability_of(line)[:2] == ("unavailable", "no_gpu")
