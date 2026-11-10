"""The analysis worker's backend, in the app's own venv (numpy alone: no RAPIDS, no CPU libraries): which backend it
picks and why (pick_backend, with fake packages and a fake cupy), the hello of every way it can start or fail
(main), the hello of this very Python (it lacks the CPU libraries), the module that runs when the engine starts the
worker, how many runs the CPU worker keeps, and the one writer of a clusters and UMAP pass that both backends share."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import sopa_companion.analysis as analysis_package
from sopa_companion.analysis import backend
from sopa_companion.analysis import worker as W
from test_analysis import write_run

APP = Path(__file__).resolve().parents[1]
CPU_PACKAGES = ["scipy", "scikit-learn", "igraph", "umap-learn", "numba"]


def fake_cupy(calls: list, devices: int = 1, count_fails=None, sum_fails=None):
    """Just enough of cupy for pick_backend: a device count and a tiny sum, each call noted in `calls`."""
    def count():
        calls.append("count")
        if count_fails is not None:
            raise count_fails
        return devices

    def arange(n, dtype=None):
        calls.append("sum")
        if sum_fails is not None:
            raise sum_fails
        return SimpleNamespace(sum=lambda: float(n * (n - 1) / 2))
    return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=count)), arange=arange,
                           float32="float32")


def test_the_worker_picks_its_backend(monkeypatch):
    """The GPU when cupy, cuML, cuGraph and cuDF are installed, CUDA shows a device and cupy runs code on it (the GPU's
    waits set to sleep before any of it); else the CPU when its libraries are installed, saying why not the GPU; a
    setting of "cpu" never looks at the GPU, one of "gpu" never falls back to the CPU. Missing packages last for the
    session, a GPU that cannot be reached does not."""
    calls: list = []
    present: set = set()

    def found(name):
        if name == "broken":
            raise ValueError("broken.__spec__ is None")
        return object() if name in present else None
    monkeypatch.setattr(backend, "find_spec", found)
    monkeypatch.setattr(W, "gpu_waits_block", lambda: calls.append("waits") or True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def pick(setting="auto", have=(*backend.GPU_MODULES, *backend.CPU_MODULES), **cupy):
        calls.clear()
        present.clear()
        present.update(have)
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy(calls, **cupy))
        return W.pick_backend(setting)

    assert pick() == {"backend": "gpu", "why": None, "missing": [], "lasting": False}                  # (a)
    assert calls == ["waits", "count", "sum"]                         # the waits sleep before CUDA starts
    choice = pick(count_fails=RuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected"))   # (b)
    assert choice == {"backend": "cpu", "why": "no CUDA device is visible (RuntimeError: cudaErrorNoDevice: no "
                                                "CUDA-capable device is detected)", "missing": [], "lasting": False}
    choice = pick(have=("cugraph", "cudf", *backend.CPU_MODULES))                                      # (c)
    assert choice == {"backend": "cpu", "why": "cupy and cuml are not installed", "missing": [], "lasting": False}
    assert calls == []                                                # nothing of CUDA's touched
    choice = pick(sum_fails=RuntimeError("CUDA driver version is insufficient for CUDA runtime version"))  # (d)
    assert choice["backend"] == "cpu"
    assert choice["why"] == ("the GPU could not run cupy's code (RuntimeError: CUDA driver version is insufficient "
                             "for CUDA runtime version)")
    choice = pick("cpu")                                                                               # (e)
    assert choice == {"backend": "cpu", "why": '[analysis] backend is "cpu"', "missing": [], "lasting": False}
    assert calls == []
    choice = pick("gpu", devices=0)                                                                    # (f)
    assert choice == {"backend": None, "why": "no CUDA device is visible (CUDA_VISIBLE_DEVICES='')", "missing": [],
                      "lasting": False}
    assert W.cannot(choice) == "no CUDA device is visible (CUDA_VISIBLE_DEVICES='')"
    choice = pick("gpu", have=("cuml", "cugraph", "cudf", *backend.CPU_MODULES))                       # (g)
    assert choice == {"backend": None, "why": "cupy is not installed", "missing": ["cupy"], "lasting": True}
    assert W.cannot(choice) == "it lacks cupy"
    choice = pick(have=("numcodecs", "scipy", "sklearn", "numba"))                                     # (h)
    assert choice == {"backend": None, "why": "cupy, cuml, cugraph and cudf are not installed",
                      "missing": ["igraph", "umap-learn"], "lasting": True}
    assert W.cannot(choice) == "it lacks igraph and umap-learn (and the GPU: cupy, cuml, cugraph and cudf are not " \
                               "installed)"
    # The GPU's stack installed but its device out of reach now (busy, or out of memory while another job holds it), in
    # a Python that lacks one of the CPU's packages (a RAPIDS environment without igraph): neither runs now, but the
    # device may come back, so that does not last (the engine asks again), and the GPU's reason comes first.
    rapids = (*backend.GPU_MODULES, "numcodecs", "scipy", "sklearn", "umap", "numba")
    for cupy in ({"count_fails": RuntimeError("cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or "
                                              "unavailable")},
                 {"sum_fails": MemoryError("cudaErrorMemoryAllocation: out of memory")}):
        choice = pick(have=rapids, **cupy)
        assert (choice["backend"], choice["missing"], choice["lasting"]) == (None, ["igraph"], False), choice
        assert W.cannot(choice) == f"{choice['why']}, and it lacks igraph to run on the CPU"
    assert pick("cpu", have=rapids)["lasting"] is True                # the CPU chosen: what it lacks lasts
    calls.clear()
    present.update(backend.GPU_MODULES + backend.CPU_MODULES)
    monkeypatch.setitem(sys.modules, "cupy", None)                    # found, but its import fails
    choice = W.pick_backend("auto")
    assert choice["backend"] == "cpu" and choice["why"].startswith("cupy could not be loaded (ModuleNotFoundError: ")
    assert backend.missing(["numba", "broken", "scipy.nothing"]) == ["broken", "scipy.nothing"]
    assert backend.distributions(backend.CPU_MODULES) == ["numcodecs", *CPU_PACKAGES]
    assert [W.and_list(words) for words in ([], ["a"], ["a", "b"], ["a", "b", "c"])] == \
        ["", "a", "a and b", "a, b and c"]


def test_every_way_the_worker_starts_or_fails_says_hello(monkeypatch, capsys):
    """main() always prints one hello: on the CPU or the GPU, which and why; a Python without either stack, its missing
    packages (lasting); CPU libraries that cannot load (lasting: numba refusing this NumPy); a GPU stack that fails or
    anything else that goes wrong (not lasting: the engine asks again)."""
    monkeypatch.setattr(os, "nice", lambda n: 0)                       # never this test process's own priority
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    served = []
    monkeypatch.setattr(W, "serve", lambda worker, stream: served.append(worker))
    fake = ModuleType("sopa_companion.analysis.cpu")
    fake.enable, fake.warm_steps = (lambda: None), (lambda worker: ["a step"])
    monkeypatch.setitem(sys.modules, "sopa_companion.analysis.cpu", fake)
    monkeypatch.setattr(analysis_package, "cpu", fake, raising=False)

    def hello(choice=None, fails=None):
        monkeypatch.setattr(W, "pick_backend", (lambda setting: choice) if fails is None else
                            (lambda setting: (_ for _ in ()).throw(fails)))
        code = W.main()
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1, lines
        return code, json.loads(lines[0])

    code, said = hello({"backend": "cpu", "why": "cupy is not installed", "missing": [], "lasting": False})
    assert code == 0 and said == {"ok": True, "ready": True, "pci_bus_id": None, "backend": "cpu",
                                  "why": "cupy is not installed", "threads": 4}
    assert served[-1].backend == "cpu" and served[-1].warm_steps == ["a step"]
    code, said = hello({"backend": None, "why": "cupy is not installed", "missing": ["igraph"], "lasting": True})
    assert code == 1 and said == {"ok": False, "ready": False, "missing": ["igraph"], "lasting": True,
                                  "error": "Clusters & UMAP cannot run in this Python environment: it lacks igraph "
                                           "(and the GPU: cupy is not installed)"}
    fake.enable = lambda: (_ for _ in ()).throw(ImportError("Numba needs NumPy 2.5 or less. Got NumPy 2.6."))
    code, said = hello({"backend": "cpu", "why": "cupy is not installed", "missing": [], "lasting": False})
    assert code == 1 and said == {"ok": False, "ready": False, "missing": [], "lasting": True,
                                  "error": "Clusters & UMAP cannot run in this Python environment: its CPU libraries "
                                           "could not be loaded (ImportError: Numba needs NumPy 2.5 or less. Got NumPy "
                                           "2.6.)"}
    # The CPU taken only because the GPU's stack, installed, could not reach its device now: that may come back, so the
    # CPU's libraries failing does not last, and the GPU's reason comes first; with [analysis] backend "cpu" it lasts.
    monkeypatch.setattr(backend, "find_spec", lambda name: object() if name in backend.GPU_MODULES else None)
    monkeypatch.delenv(backend.BACKEND_ENV, raising=False)            # "auto"
    busy = {"backend": "cpu", "why": "no CUDA device is visible (RuntimeError: busy)", "missing": [], "lasting": False}
    code, said = hello(busy)
    assert code == 1 and said["lasting"] is False and said["error"] == (
        "Clusters & UMAP cannot run in this Python environment: no CUDA device is visible (RuntimeError: busy), and "
        "its CPU libraries could not be loaded (ImportError: Numba needs NumPy 2.5 or less. Got NumPy 2.6.)")
    monkeypatch.setenv(backend.BACKEND_ENV, "cpu")
    code, said = hello({**busy, "why": '[analysis] backend is "cpu"'})
    assert code == 1 and said["lasting"] is True and "its CPU libraries could not be loaded" in said["error"]
    monkeypatch.delenv(backend.BACKEND_ENV)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy([]))
    monkeypatch.setattr(W, "Worker", lambda **kwargs: (_ for _ in ()).throw(MemoryError("out of memory")))
    code, said = hello({"backend": "gpu", "why": None, "missing": [], "lasting": False})
    assert code == 1 and not said["lasting"] and said["error"].endswith("its GPU libraries could not be loaded "
                                                                        "(MemoryError: out of memory)")
    code, said = hello(fails=OSError("no such file"))
    assert code == 1 and not said["lasting"] and said["missing"] == [] and \
        said["error"].endswith("its analysis libraries could not be loaded (OSError: no such file)")
    monkeypatch.setenv("OMP_NUM_THREADS", "four")
    assert W.threads() is None


def test_the_hello_says_what_this_python_lacks(tmp_path):
    """The app's own venv has neither RAPIDS nor the CPU libraries: the worker started in it says so in its first line,
    naming the packages by the names pip knows, and that this lasts (the engine stops asking this session)."""
    lacking = backend.missing(backend.CPU_MODULES)
    if not lacking or not backend.missing(backend.GPU_MODULES):
        pytest.skip("this Python can run Clusters & UMAP")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", backend.BACKEND_ENV)}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(APP.parent))
    done = subprocess.run([sys.executable, "-m", "sopa_companion.analysis.worker"], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert done.returncode == 1, done.stderr[-3000:]
    lines = done.stdout.splitlines()
    assert len(lines) == 1, done.stdout
    hello = json.loads(lines[0])
    assert {k: hello[k] for k in ("ok", "ready", "lasting")} == {"ok": False, "ready": False, "lasting": True}
    assert hello["missing"] == backend.distributions(lacking)
    assert hello["error"].startswith("Clusters & UMAP cannot run in this Python environment: it lacks ")
    if lacking == list(backend.CPU_MODULES[1:]):                      # the app's venv: numcodecs only
        assert hello["missing"] == CPU_PACKAGES


def test_run_as_a_module_the_worker_is_the_packages_own():
    """The engine runs `python -m sopa_companion.analysis.worker`: what runs is the package's module itself, not a
    second copy of it, so the CPU code (which imports the worker module) sets up the module this process runs."""
    code = ("import runpy\n"
            "import sopa_companion.analysis.worker as W\n"
            "W.main = lambda: print('the package module ran') or 7\n"
            "runpy.run_module('sopa_companion.analysis.worker', run_name='__main__', alter_sys=True)\n")
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(APP.parent))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 7 and done.stdout.strip() == "the package module ran", done.stderr[-3000:]


def test_cpu_workers_keep_fewer_runs(tmp_path):
    """The CPU worker keeps 2 runs (the run followed live and one looked at: their layouts and start rows sit in host
    memory), the GPU's 4; the least recently used goes first. Making a CPU-mode worker loads none of the CPU
    libraries."""
    heavy = ("scipy", "sklearn", "igraph", "umap", "numba")
    before = set(sys.modules)
    worker = W.Worker(xp=np, backend="cpu")
    loaded = {m for m in set(sys.modules) - before if m.split(".")[0] in heavy or m.endswith("analysis.cpu")}
    assert not loaded, loaded
    numpy_alone = W.Worker(xp=np)
    assert worker.backend == "cpu" and worker.runs_kept == W.CPU_RUNS_KEPT == 2
    assert numpy_alone.backend == "numpy" and numpy_alone.runs_kept == W.RUNS_KEPT
    assert worker.stat_pool is not None and numpy_alone.stat_pool is None    # the worker process's own thread
    paths = {}
    for name in ("live", "a", "b"):
        folder = write_run(tmp_path / name, frames=1)
        paths[name] = {"run": name, "transcripts": str(folder / "live" / "transcripts.bin.gz"),
                       "genes": str(folder / "live" / "genes.txt")}
    live = worker.state(paths["live"])
    worker.state(paths["a"])
    assert worker.state(paths["live"]) is live
    worker.state(paths["b"])                                          # "a" goes: the least recently used
    assert list(worker.runs) == ["live", "b"]


def test_one_writer_for_both_backends(tmp_path, monkeypatch):
    """Both backends end a pass with the same code: the clusters' summary entries (each a colour of its own, top genes
    with a finite score), the layout normalised and turned onto the last one, the cells file (int16 clusters with -1 for
    cells not kept, then float32 [cells, 3] with z = 0 and NaN rows) and a summary with SUMMARY_KEYS (proseg's count:
    basis, count, and its live count in cells, from the frame's counts file: the averaged count, or the sampled one
    under D0's other setting), "umap" and "write" the last steps timed."""
    run = SimpleNamespace(genes=["G0", "G1", "G2", "G3"], colours={}, freed={}, embedding=None, updates=0,
                          layout_key=None, transcripts=500)
    found = W.summary_clusters(run, np.array([3, 7]), np.array([[1, 2, 0], [3, 0, 1]]),
                               np.array([[0.9, 0.5, -np.inf], [0.7, 0.2, 0.1]]),
                               np.array([[0.5, 0.4, 0.1], [0.6, 0.3, 0.2]]), np.array([3.0, 2.0]))
    assert [(c["id"], c["size"], c["label"]) for c in found] == [(3, 3, "G1 · G2"), (7, 2, "G3 · G0 · G1")]
    assert found[0]["top"] == [{"gene": "G1", "score": 0.9, "pct": 0.5}, {"gene": "G2", "score": 0.5, "pct": 0.4}]
    assert found[0]["color"] != found[1]["color"] and run.colours == {3: found[0]["color"], 7: found[1]["color"]}
    keep = np.array([0, 1, 3, 4, 6])
    with pytest.raises(W.TooFewCells, match="only 0 cells have >= 20 transcripts"):
        W.umap_cells(np, np.full((40, 3), 5.0, np.float32), {**W.DEFAULTS, "min_counts": 20})
    assert W.umap_start(run, keep, np.zeros((5, 3), int), 7, W.DEFAULTS) == (None, False, None) and run.updates == 1
    clusters = np.full(7, -1, np.int32)
    clusters[keep] = [3, 3, 7, 3, 7]
    raw = (np.random.default_rng(0).normal(size=(5, 2)) * 3).astype(np.float32)
    previous = raw @ np.array([[0, -1], [1, 0]], np.float32)          # the last layout: this one turned by 90 degrees
    watch = W.Stopwatch(lambda: None, {"load": 1.0, "counts": 2.0})
    request = {"ncells": 7, "seq": 4, "iter": 9, "out_cells": str(tmp_path / "4.cells.bin"),
               "out_json": str(tmp_path / "4.json")}
    read = SimpleNamespace(e=np.array([0.75, 1.25, 0.5, 0.03125], np.float32))    # the frame's counts file: e
    summary = W.Worker(xp=np).finish_umap(run, request, dict(W.DEFAULTS), keep, clusters, found, raw, previous, True,
                                          watch, 5, read)
    data = (tmp_path / "4.cells.bin").read_bytes()
    assert len(data) == 2 * 7 + 12 * 7 and not list(tmp_path.glob("*.tmp"))
    assert np.frombuffer(data[:14], "<i2").tolist() == clusters.tolist()
    layout = np.frombuffer(data[14:], "<f4").reshape(7, 3)
    assert np.isnan(layout[[2, 5]]).all() and (layout[keep, 2] == 0).all()
    np.testing.assert_allclose(layout[keep, :2], W.normalise(previous), atol=1e-4)   # turned back onto the last one
    np.testing.assert_array_equal(run.embedding[keep], layout[keep, :2])
    assert set(summary) == W.SUMMARY_KEYS and set(json.loads((tmp_path / "4.json").read_text())) == W.SUMMARY_KEYS
    assert list(summary["ms"]) == ["load", "counts", "umap", "write"] and summary["gpu"] is None
    assert (summary["seq"], summary["iter"], summary["ncells"], summary["nclusters"], summary["warm"]) == \
        (4, 9, 5, 2, True)
    assert summary["transcripts"] == 500 and summary["transcripts_in_cells"] == 3 and summary["bounds"] is not None
    assert (summary["basis"], summary["count"]) == ("proseg", "averaged")
    assert all(len(c["centroid"]) == 3 for c in summary["clusters"])
    init, warm, last = W.umap_start(run, keep, np.tile(keep, (5, 1)), 7, W.DEFAULTS)   # the next pass starts from it
    assert warm and run.updates == 2
    np.testing.assert_allclose(init, layout[keep, :2], atol=0.01)
    assert W.umap_start(run, keep, np.tile(keep, (5, 1)), 7, {**W.DEFAULTS, "refresh_every": 3})[1] is False
    assert W.umap_start(run, keep, np.tile(keep, (5, 1)), 7, {**W.DEFAULTS, "n_neighbors": 30})[1] is False
    # A frame read on the count proseg saves (its request's count "sampled"): the pass counts f in cells and says so.
    read = SimpleNamespace(e=read.e, f=np.array([1, 2, 1, 1], np.int32))    # sum of e: 2.53 (3); of f: 5
    for count, total in (("sampled", 5), ("averaged", 3)):
        summary = W.Worker(xp=np).finish_umap(run, {**request, "count": count}, dict(W.DEFAULTS), keep, clusters, found,
                                              raw, previous, True, W.Stopwatch(lambda: None), 5, read)
        assert summary["transcripts_in_cells"] == total and (summary["basis"], summary["count"]) == ("proseg", count)
    # D0's sampled count: the pass clusters f, and its summary counts f in cells and says so.
    from sopa_companion.analysis import counts_file
    monkeypatch.setattr(counts_file, "LIVE_COUNT", "sampled")
    monkeypatch.setattr(counts_file, "LIVE_ARRAY", "f")
    read = SimpleNamespace(e=read.e, f=np.array([1, 2, 1, 1], np.int32))    # sum of e: 2.53 (3); of f: 5
    summary = W.Worker(xp=np).finish_umap(run, request, dict(W.DEFAULTS), keep, clusters, found, raw, previous, True,
                                          W.Stopwatch(lambda: None), 5, read)
    assert summary["transcripts_in_cells"] == 5 and (summary["basis"], summary["count"]) == ("proseg", "sampled")
