"""Helper regressions without a store, model weights or GPU inference."""
import os
import math
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import needs_sopa, sopa_python
from sopa_companion.zarr_bridge.stain import Cellpose


@needs_sopa
@pytest.mark.parametrize("case", ["seam", "corner", "diameter", "neighbours", "budget", "resume", "legacy", "lasso",
                                 "level", "level-budget", "level-instanseg"])
def test_stain_helper_seams_and_tile_budget(tmp_path, case):
    result = subprocess.run([sopa_python(), str(Path(__file__).with_name("stain_seam_check.py")), str(tmp_path), case],
                            capture_output=True, text=True, timeout=90,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                                 "NUMBA_CACHE_DIR": str(tmp_path / "numba"),
                                 "MPLCONFIGDIR": str(tmp_path / "mpl"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
                                 "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("diameter,side,message", [
    (1, 1024, "between 10 and 1000"),
    (10, 1024, "tile is too large"),
    (None, 2048, "tile is too large"),
])
def test_stain_cellpose_refuses_unsafe_input_before_eval(diameter, side, message):
    wrapper = object.__new__(Cellpose)
    wrapper.settings = dict(diameter=diameter, flow_threshold=.4, cellprob_threshold=0)
    wrapper.model = types.SimpleNamespace(eval=Mock(return_value=(np.zeros((1, 1)),)))
    wrapper.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=RuntimeError))
    wrapper.gpu = False
    pixels = np.broadcast_to(np.float32(0), (side, side, 3))
    with pytest.raises(ValueError, match=message):
        wrapper(pixels)
    wrapper.model.eval.assert_not_called()


def test_level_cellpose_settings_and_network_budget(tmp_path, monkeypatch):
    from sopa_companion.zarr_bridge import stain as S
    # Levels as inspect lists them: the engine's resolution choices (engine/stain_levels.py) read their dims and shape.
    image = dict(to_frame=np.eye(3).tolist(), levels=[
        dict(path=f"images/i/{k}", shape=[1, 4096 // f, 4096 // f], dims=["c", "y", "x"], pixel=[.25 * f, .25 * f],
             origin=[0, 0]) for k, f in enumerate((1, 2, 4, 8))])
    settings = dict(diameter=24, min_size=64, niter=75,
                    normalize=dict(sharpen_radius=8, smooth_radius=4, tile_norm_blocksize=128))
    job = dict(image=image, settings={**settings, "level": 3}, tile_pixels=1024, overlap_pixels=128)
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(models=object()))
    monkeypatch.setattr(S, "model_file", lambda *args: {"model_path": "unused"})
    model = types.SimpleNamespace(eval=Mock(return_value=(np.zeros((64, 64), np.uint32),)))
    monkeypatch.setattr(S.Cellpose, "load", lambda self: model)
    monkeypatch.setattr(S.importlib.metadata, "version", lambda name: "test")
    events = []
    monkeypatch.setattr(S, "emit", lambda kind, **data: events.append((kind, data)))
    (tmp_path / "stain").mkdir()
    cellpose = S.Cellpose(job, tmp_path)
    pixels = np.zeros((64, 64, 3), np.float32)
    cellpose(pixels)
    args, kwargs = model.eval.call_args
    assert args[0] is pixels
    assert kwargs["diameter"] == 3 and kwargs["min_size"] == 1 and kwargs["niter"] == 75
    assert kwargs["normalize"] == dict(sharpen_radius=1, smooth_radius=.5, tile_norm_blocksize=16)
    assert settings["normalize"]["tile_norm_blocksize"] == 128
    assert events[-1][1]["level"] == 3 and events[-1][1]["um_per_px"] == 2
    costs = []
    for k, f in enumerate((1, 2, 4, 8)):
        current = {**job, "settings": {**settings, "level": k}}
        size, overlap = S.tile_layout(current)
        assert overlap == math.ceil(128 / f)
        boxes = list(S.tile_boxes([0, 0, 4096 // f, 4096 // f], size, overlap))
        costs.append(sum((b[2] - b[0]) * (b[3] - b[1]) * (30 * f / 24)**2 for b in boxes))
        for b in boxes:
            S.check_tile((b[3] - b[1], b[2] - b[0]), settings, f)
    assert max(costs) / min(costs) < 1.1, costs


@needs_sopa
@pytest.mark.parametrize("case", ["level", "level-model", "ranges"])
def test_stain_patch_level(tmp_path, case):
    result = subprocess.run([sopa_python(), str(Path(__file__).with_name("stain_patches_check.py")), str(tmp_path), case],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
             "NUMBA_CACHE_DIR": str(tmp_path / "numba"), "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "XDG_CACHE_HOME": str(tmp_path / "cache"), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


def device_torch(cuda: bool, mps: bool):
    """A torch with CUDA and Apple's GPU (MPS) as a computer has them, each cache's emptying recorded."""
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda, empty_cache=Mock(), OutOfMemoryError=MemoryError),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        mps=types.SimpleNamespace(empty_cache=Mock()), device=lambda name: f"device:{name}")


def cellpose_on(tmp_path, monkeypatch, torch, model):
    """A Cellpose helper made as a run makes it, on `torch`, every CellposeModel being `model`; its events."""
    from sopa_companion.zarr_bridge import stain as S
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    models = types.SimpleNamespace(CellposeModel=Mock(return_value=model))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(models=models))
    monkeypatch.setattr(S, "model_file", lambda *args: {"model_path": str(weights)})
    monkeypatch.setattr(S.importlib.metadata, "version", lambda name: "test")
    events = []
    monkeypatch.setattr(S, "emit", lambda kind, **data: events.append((kind, data)))
    image = {"key": "image", "channels": ["DAPI"], "to_frame": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "levels": [{"path": "images/image/scale0/image", "shape": [1, 64, 64], "dims": ["c", "y", "x"],
                         "origin": [0, 0], "pixel": [1, 1]}]}
    (tmp_path / "stain").mkdir(exist_ok=True)
    return S.Cellpose({"settings": {"diameter": 30}, "image": image}, tmp_path), models, events


@pytest.mark.parametrize("cuda,mps,device", [(True, False, "cuda:0"), (True, True, "cuda:0"), (False, True, "mps"),
                                             (False, False, "cpu")])
def test_cellpose_runs_on_cuda_then_apples_gpu_then_the_cpu(tmp_path, monkeypatch, cuda, mps, device):
    """CUDA's GPU first, then Apple's (MPS, here passing its short test), then the CPU: the helper's hello names the
    device, and Cellpose gets it (with its gpu flag on for either GPU)."""
    from sopa_companion.zarr_bridge import devices
    from sopa_companion.zarr_bridge.devices import pick_device
    monkeypatch.setattr(devices, "_tested", {"mps": None})
    model = types.SimpleNamespace(eval=Mock(return_value=(np.ones((64, 64), np.uint32),)))
    cellpose, models, events = cellpose_on(tmp_path, monkeypatch, device_torch(cuda, mps), model)
    assert (cellpose.device, cellpose.gpu) == (device, device != "cpu") == pick_device(device_torch(cuda, mps))
    models.CellposeModel.assert_called_once_with(gpu=device != "cpu", pretrained_model=str(tmp_path / "weights"),
                                                 device=f"device:{device}")
    assert [kind for kind, _ in events] == ["hello"] and events[0][1]["device"] == device
    assert cellpose(np.zeros((64, 64, 3), np.float32)).shape == (64, 64)
    # A torch without either (an old one, or a fake without backends): the CPU, never a failure.
    assert pick_device(types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))) == ("cpu", False)
    assert pick_device(object()) == ("cpu", False)


def test_an_operation_apples_gpu_lacks_moves_the_run_to_the_cpu(tmp_path, monkeypatch):
    """On MPS, an operation it lacks (NotImplementedError) ends the GPU's part: its cache is emptied, the helper says
    "fallback" with the CPU, and the tile is done again on the CPU; a failure on the CPU is raised as it is."""
    from sopa_companion.zarr_bridge import devices
    from sopa_companion.zarr_bridge.devices import empty_cache
    monkeypatch.setattr(devices, "_tested", {"mps": None})
    torch = device_torch(False, True)
    masks = np.full((64, 64), 3, np.uint32)
    model = types.SimpleNamespace(eval=Mock(side_effect=[NotImplementedError("aten::op is not implemented for MPS"),
                                                         (masks,)]))
    cellpose, models, events = cellpose_on(tmp_path, monkeypatch, torch, model)
    assert (cellpose(np.zeros((64, 64, 3), np.float32)) == 3).all() and model.eval.call_count == 2
    assert [kind for kind, _ in events] == ["hello", "fallback"] and events[1][1]["device"] == "cpu"
    assert "not implemented for MPS" in events[1][1]["reason"]
    torch.mps.empty_cache.assert_called_once_with()
    torch.cuda.empty_cache.assert_not_called()
    assert (cellpose.device, cellpose.gpu) == ("cpu", False)
    assert models.CellposeModel.call_args_list[-1].kwargs == {"gpu": False, "device": "device:cpu",
                                                             "pretrained_model": str(tmp_path / "weights")}
    model.eval.side_effect = RuntimeError("the CPU failed too")
    with pytest.raises(RuntimeError, match="the CPU failed too"):
        cellpose(np.zeros((64, 64, 3), np.float32))
    # Emptying a cache never raises (a torch without one); nothing on the CPU.
    empty_cache(types.SimpleNamespace(), "mps")
    empty_cache(torch, "cpu")
    torch.mps.empty_cache.assert_called_once_with()


ABORTED = ("2026-11-06 python[123:456] *** Terminating app due to uncaught exception 'NSInvalidArgumentException', "
           "reason: '-[MPSNDArray arrayViewWithShape:strides:]: unrecognized selector sent to instance 0x1'\n"
           "*** First throw call stack:\n(\n\t0   CoreFoundation 0x1 __exceptionPreprocess + 176\n)\n"
           "libc++abi: terminating due to uncaught exception of type NSException\n")


def test_apples_gpu_is_used_only_where_it_passes_a_short_test_in_a_process_of_its_own(monkeypatch):
    """PyTorch can call Apple's GPU available where its operations abort the process (PyTorch 2.15 on macOS 14: every
    linear layer calls a macOS 15 function), which no fallback inside the helper could catch. A short test in a
    process of its own (this interpreter, its working folder never on the path) decides, once per process: passed,
    Apple's GPU; aborted, the CPU and Metal's own words; too slow, the CPU. CUDA is never tested."""
    from sopa_companion.zarr_bridge import devices
    torch = device_torch(False, True)
    torch.__version__ = "2.15.0"
    monkeypatch.setattr(devices.platform, "mac_ver", lambda: ("14.7.6", ("", "", ""), "arm64"))
    for answer, expected in [(subprocess.CompletedProcess([], 0, "True\n", ""), ("mps", True, None)),
                             (subprocess.CompletedProcess([], -6, "", ABORTED), "aborted"),
                             (subprocess.TimeoutExpired([], devices.MPS_TEST_S), "slow"),
                             (subprocess.CompletedProcess([], 1, "", "Traceback ...\nRuntimeError: no MPS\n"), "error"),
                             (subprocess.CompletedProcess([], 0, "False: its lgamma differs from the CPU's by 0.021\n",
                                                          ""), "differs"),
                             (subprocess.CompletedProcess([], 0, "False\n", ""), "not finite")]:
        monkeypatch.setattr(devices, "_tested", {})
        run = Mock(side_effect=answer) if isinstance(answer, Exception) else Mock(return_value=answer)
        monkeypatch.setattr(devices.subprocess, "run", run)
        chosen = devices.choose_device(torch)
        assert devices.choose_device(torch) == chosen and devices.pick_device(torch) == chosen[:2]
        run.assert_called_once()
        command = run.call_args.args[0]
        assert command[0] == sys.executable and command[1:4] == ["-P", "-B", "-c"] and command[4] == devices.MPS_TEST
        assert run.call_args.kwargs["timeout"] == devices.MPS_TEST_S
        if expected == "aborted":
            assert chosen == ("cpu", False, "Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6: "
                              "-[MPSNDArray arrayViewWithShape:strides:]: unrecognized selector sent to instance 0x1")
        elif expected == "slow":
            assert chosen[:2] == ("cpu", False)
            assert f"did not finish a short test within {devices.MPS_TEST_S} s" in chosen[2]
            assert devices.MPS_LEARNING_TEST_S < 120 <= devices.MPS_TEST_S   # the learning's within the probe's wait
        elif expected == "differs":
            assert chosen == ("cpu", False, "Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6: its "
                                            "lgamma differs from the CPU's by 0.021")
        elif expected == "not finite":
            assert chosen[:2] == ("cpu", False) and chosen[2].endswith(": its values are not finite")
        elif expected == "error":
            assert chosen[:2] == ("cpu", False) and chosen[2].endswith(": RuntimeError: no MPS")
        else:
            assert chosen == expected
    # CUDA first, untested; and no GPU at all asks nothing.
    run = Mock()
    monkeypatch.setattr(devices.subprocess, "run", run)
    monkeypatch.setattr(devices, "_tested", {})
    assert devices.choose_device(device_torch(True, True)) == ("cuda:0", True, None)
    assert devices.choose_device(device_torch(False, False)) == ("cpu", False, None)
    run.assert_not_called()


def test_the_short_test_of_apples_gpu_runs_for_real_here(monkeypatch):
    """The test's own process, for real: without Apple's GPU (Linux, Windows, GitHub's Mac runners) it fails and says
    why in the child's last words; on a Mac whose GPU runs it, nothing."""
    from sopa_companion.zarr_bridge import devices
    monkeypatch.setattr(devices, "_tested", {})
    why = devices.apple_gpu_problem(types.SimpleNamespace(__version__="test"))
    if sys.platform != "darwin":
        assert why.startswith("Apple's GPU failed a short test with PyTorch test on macOS ?: ") and "Error" in why
    else:
        assert why is None or why.startswith("Apple's GPU failed a short test")


def test_cellpose_on_apples_gpu_that_fails_the_test_runs_on_the_cpu_and_says_why(tmp_path, monkeypatch):
    """The helper never loads Cellpose on a GPU that failed the test: it says "fallback" with the reason before its
    hello, which names the CPU (the engine keeps the reason in the run's Log and its saved provenance)."""
    from sopa_companion.zarr_bridge import devices
    why = "Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6: -[MPSNDArray ...]"
    monkeypatch.setattr(devices, "_tested", {"mps": why})
    model = types.SimpleNamespace(eval=Mock(return_value=(np.ones((64, 64), np.uint32),)))
    cellpose, models, events = cellpose_on(tmp_path, monkeypatch, device_torch(False, True), model)
    assert (cellpose.device, cellpose.gpu) == ("cpu", False)
    assert [kind for kind, _ in events] == ["fallback", "hello"]
    assert events[0][1] == {"device": "cpu", "reason": why} and events[1][1]["device"] == "cpu"
    models.CellposeModel.assert_called_once_with(gpu=False, pretrained_model=str(tmp_path / "weights"),
                                                 device="device:cpu")


def test_cellposes_window_keeps_off_apples_gpu_where_it_fails_the_test(monkeypatch, capsys):
    """Cellpose's own check only moves a tensor to Apple's GPU, which works even where its operations abort: on such a
    Mac its window starts with "use GPU" off (cellpose.core.use_gpu False), saying why in its log. Where the test
    passes, and on any other system (not even torch imported), Cellpose decides as always."""
    from sopa_companion.zarr_bridge import devices, stain_gui
    core = types.SimpleNamespace(use_gpu=Mock(return_value=True))
    monkeypatch.setitem(sys.modules, "cellpose", types.SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "cellpose.core", core)
    monkeypatch.setitem(sys.modules, "torch", device_torch(False, True))
    monkeypatch.setattr(stain_gui.sys, "platform", "darwin")
    monkeypatch.setattr(devices, "_tested", {"mps": None})
    stain_gui.keep_off_apples_gpu()
    assert core.use_gpu() is True and capsys.readouterr().out == ""
    monkeypatch.setattr(devices, "_tested", {"mps": "Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6"})
    stain_gui.keep_off_apples_gpu()
    assert core.use_gpu() is False and core.use_gpu(use_torch=True) is False
    assert capsys.readouterr().out == ("Apple's GPU failed a short test with PyTorch 2.15.0 on macOS 14.7.6. "
                                       "Cellpose's window runs on the CPU.\n")
    monkeypatch.setattr(stain_gui.sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "torch")
    stain_gui.keep_off_apples_gpu()                 # returns before importing torch
    assert "torch" not in sys.modules


def test_a_control_file_that_cannot_be_read_keeps_the_last_one(tmp_path, monkeypatch):
    """Replaced on NTFS through ntfs-3g, control.json is missing for a moment: a paused run neither ends nor goes on
    then, and a stop read before still stops it."""
    from sopa_companion.zarr_bridge import stain as S
    said, naps = [], []
    control = tmp_path / "control.json"

    def nap(seconds):                       # while paused: the file goes missing, then says run
        naps.append(seconds)
        if len(naps) == 1:
            control.unlink()
        elif len(naps) == 3:
            control.write_text('{"paused": false, "stop": false}')
    monkeypatch.setattr(S, "emit", lambda ev, **values: said.append(values.get("state")))
    monkeypatch.setattr(S, "time", types.SimpleNamespace(sleep=nap))
    control.write_text('{"paused": true, "stop": false}')
    assert S.checkpoint(tmp_path, 0) is True
    assert said == ["paused", "running"] and len(naps) == 3, "missing meanwhile: still paused, and no error"
    control.write_text('{"paused": false, "stop": true}')
    assert S.read_control(tmp_path)["stop"]
    control.unlink()
    assert S.checkpoint(tmp_path, 1) is False, "a stop read before the file went still stops"
