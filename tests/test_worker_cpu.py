"""Clusters and the UMAP on the CPU worker (analysis/cpu.py), the mixing check, Go deeper and Compare on it, its warm-up
and its hello. The CPU worker needs scipy, scikit-learn, igraph, umap-learn and numba, which the app's own venv lacks:
the worker process and the cases of tests/cpu_cases.py run in a Python that has them (cpu_python: this one, else the API
environment's, else the sopa environment's), with no GPU and four threads; the tests make the data and judge the
answers here. What needs none of those libraries (Leiden's warm start, the warm-up's size) runs here too."""
from __future__ import annotations

from contextlib import contextmanager, suppress
import gzip
import json
import math
import os
from pathlib import Path
import queue
import shutil
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import api_python, harness_path, sopa_python, wait_for
from sopa_companion.analysis import backend, cpu
from sopa_companion.analysis import worker as W
from sopa_companion.engine.analysis import CPU_MAX_GENES
from test_analysis import JUDGE, KEEP, MODEL_GENES, NCELLS, RECORD, X0, Y0, counts_files, write_run
from test_worker_compare import LEARNED, QC, write_counts
from test_worker_deeper import K, RUN_IDS, make_deep

APP = Path(__file__).resolve().parents[1]            # the sopa_companion folder; its parent goes on PYTHONPATH
FOUND: dict = {}                                     # cpu_python's answer, worked out once
VOXEL, SQUARE, GX0, GY0 = 2.0, 3, 100.0, 50.0        # a grouped run's cells: 3 x 3 squares of 2 um voxels from GX0, GY0


# ---- where the CPU code runs -----------------------------------------------------------------------------------------
def has_cpu_libraries(python: str) -> bool:
    probe = f"import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in {backend.CPU_MODULES!r}))"
    return subprocess.run([python, "-I", "-c", probe], capture_output=True, timeout=300).returncode == 0


def cpu_python() -> str:
    """The Python the CPU worker's tests run in: this one when it has the CPU libraries (a sopa environment's tests),
    else the API environment's (made as a user's: `pip install sopa`), else the sopa environment's; skipped where none
    has them."""
    if "python" not in FOUND:
        FOUND["python"] = sys.executable if not backend.missing(backend.CPU_MODULES) else next(
            (python for python in (api_python(skip=False), sopa_python(skip=False))
             if python and has_cpu_libraries(python)), None)
    if FOUND["python"] is None:
        pytest.skip("no Python with scipy, scikit-learn, igraph, umap-learn and numba here: set "
                    "SOPA_COMPANION_API_PYTHON or SOPA_COMPANION_SOPA_PYTHON")
    return FOUND["python"]


@pytest.fixture(scope="session")
def numba_cache(tmp_path_factory) -> Path:
    """numba's cache for every CPU process of the session (umap-learn's compiled functions: made once, not per test)."""
    return tmp_path_factory.mktemp("numba")


def cpu_env(cache: Path | None, **extra) -> dict:
    """The environment of a CPU process: no GPU, four threads, numba's cache in `cache` (None: as the process makes
    it), the app on PYTHONPATH; never the display, nor a backend or cache folder the engine would give."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY", "NUMBA_CACHE_DIR", backend.BACKEND_ENV, backend.CPU_CACHE_ENV)}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OMP_WAIT_POLICY="PASSIVE", PYTHONPATH=str(APP.parent), PYTHONDONTWRITEBYTECODE="1")
    if cache is not None:
        env["NUMBA_CACHE_DIR"] = str(cache)
    env.update(extra)
    return env


def run_case(case: str, cache: Path, timeout: float = 900.0, **args) -> tuple[dict, str]:
    """One case of tests/cpu_cases.py: (its answer, the worker's log)."""
    done = subprocess.run([cpu_python(), "-m", "sopa_companion.tests.cpu_cases", case, json.dumps(args)],
                          capture_output=True, text=True, env=cpu_env(cache), cwd=str(APP.parent), timeout=timeout)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


class WorkerProcess:
    """`python -m sopa_companion.analysis.worker` in the CPU Python, its replies and its log read as they come."""

    def __init__(self, env: dict):
        self.process = subprocess.Popen([cpu_python(), "-m", "sopa_companion.analysis.worker"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                        cwd=str(APP.parent))
        self.replies: queue.Queue = queue.Queue()
        self.lines: list[str] = []
        self.readers = [threading.Thread(target=read, daemon=True) for read in (self._out, self._err)]
        for reader in self.readers:
            reader.start()

    def _out(self):
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def _err(self):
        for line in self.process.stderr:
            self.lines.append(line)

    @property
    def log(self) -> str:
        return "".join(self.lines)

    def reply(self, timeout: float = 600.0) -> dict:
        line = self.replies.get(timeout=timeout)
        assert line is not None, f"the worker ended without a reply: {self.log[-3000:]}"
        return json.loads(line)

    def send(self, request: dict):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def close(self, timeout: float = 120.0) -> int:
        """Its stdin closed (the worker ends on its own), then everything it wrote read: its exit code."""
        with suppress(OSError):
            self.process.stdin.close()
        try:
            return self.process.wait(timeout)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            for reader in self.readers:
                reader.join(30)


@contextmanager
def worker_process(env: dict):
    worker = WorkerProcess(env)
    try:
        yield worker
    finally:
        worker.close()


# ---- data ------------------------------------------------------------------------------------------------------------
def write_grouped_run(folder: Path, groups: int = 3, cells: int = 60, genes: int = 40, seed: int = 0, frames: int = 3,
                      low: int = 0, gradient: bool = False) -> tuple[Path, np.ndarray]:
    """A run folder as proseg-live leaves it (live/info.json, genes.txt, transcripts.bin.gz, frames.jsonl, dense
    frames of one layer and proseg's count of each: e = KEEP x the transcripts in each cell's voxels, as
    test_analysis.write_run writes it), with planted expression groups: cell c belongs to
    group c % groups and draws its transcripts mostly (30 to 1) from its group's own block of genes, 40-199 of them
    (`low` cells only 5: too few to cluster); with `gradient`, each cell also sits somewhere along its group (the two
    halves of its block in proportions of its own), as real cells do, so its neighbours are its own. Cells are 3 x
    3-voxel squares on a grid; each frame after the first gives 5 % of the cells' voxels a neighbouring voxel's label.
    -> (the folder, each cell's group)."""
    rng = np.random.default_rng(seed)
    live = folder / "live"
    (live / "frames").mkdir(parents=True)
    side = math.ceil(math.sqrt(cells))
    nx = ny = side * SQUARE
    j, i = np.mgrid[0:ny, 0:nx]
    slot = (j // SQUARE) * side + i // SQUARE
    base = np.where(slot < cells, slot + 1, 0).astype(np.uint32)
    group = np.arange(cells) % groups
    profiles = np.ones((groups, genes))
    for g in range(groups):
        profiles[g, g * genes // groups:(g + 1) * genes // groups] = 30.0
    profiles /= profiles.sum(axis=1, keepdims=True)
    depth = rng.integers(40, 200, cells)
    depth[rng.choice(cells, low, replace=False)] = 5
    cell = np.repeat(np.arange(cells), depth)
    gene = np.zeros(len(cell), np.uint32)
    if gradient:
        block, along = genes // groups, rng.uniform(0, 1, cells)
        for c in range(cells):
            weights = np.ones(genes)
            first = group[c] * block
            weights[first:first + block // 2] = 30.0 * (0.1 + along[c])
            weights[first + block // 2:first + block] = 30.0 * (1.1 - along[c])
            gene[cell == c] = rng.choice(genes, int(depth[c]), p=weights / weights.sum())
    else:
        for g in range(groups):
            mine = group[cell] == g
            gene[mine] = rng.choice(genes, int(mine.sum()), p=profiles[g])
    records = np.zeros(len(cell), RECORD)
    records["x"] = GX0 + VOXEL * ((cell % side) * SQUARE + rng.uniform(0.01, SQUARE - 0.01, len(cell)))
    records["y"] = GY0 + VOXEL * ((cell // side) * SQUARE + rng.uniform(0.01, SQUARE - 0.01, len(cell)))
    records["z"], records["gene"] = rng.uniform(0, 1, len(cell)), gene
    (live / "transcripts.bin.gz").write_bytes(gzip.compress(records.tobytes()))
    (live / "genes.txt").write_text("".join(f"G{g}\n" for g in range(genes)))
    from counts_fixtures import counts_from_labels, write_frame_counts
    index = []
    inside = np.flatnonzero(base > 0)
    labels = base
    for f in range(frames):
        if f:
            moved = np.random.default_rng(seed * 100 + f).choice(inside, int(0.05 * len(inside)), replace=False)
            dj, di = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])[np.random.default_rng(f).integers(0, 4, len(moved))].T
            mj, mi = np.clip(moved // nx + dj, 0, ny - 1), np.clip(moved % nx + di, 0, nx - 1)
            labels = labels.copy()
            labels.flat[moved] = labels[mj, mi]
        made = counts_from_labels(records, labels.ravel(), {"x0": GX0, "y0": GY0, "voxel": VOXEL, "nx": nx, "ny": ny,
                                                            "nz": 1}, 0.0, 1.0, cells, genes, KEEP)
        write_frame_counts(live, f, f, ncells=cells, ngenes=genes, **{k: made[k] for k in ("indptr", "genes", "n", "e")},
                           reason="init" if f == 0 else "request")
        name = f"frames/{f:06d}.bin.gz"
        raw = np.concatenate([labels.ravel(), np.zeros(nx * ny), labels.ravel()]).astype("<u4")
        (live / name).write_bytes(gzip.compress(raw.tobytes()))
        index.append({"seq": f, "iter": f, "phase": "burnin", "file": name, "nz": 1, "nx": nx, "ny": ny, "x0": GX0,
                      "y0": GY0, "voxel": VOXEL, "cells": cells})
    (live / "info.json").write_text(json.dumps({"cells": cells, "genes": genes, "layers": 1, "transcripts": len(cell),
                                                "z": {"grid_origin": 0.0, "grid_spacing": 1.0}}))
    (live / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    return folder, group


def umap_request(folder: Path, seq: int, out: Path, run: str = "r1", tag: str = "", **params) -> dict:
    """An analyze request with op "umap" for frame `seq` of a grouped run, as the engine sends it (files named by run,
    tag and frame in `out`)."""
    live = folder / "live"
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[seq])
    name = f"{run}{tag}-{seq:06d}"
    return {"op": "analyze", "ops": ["umap"], "run": run, "seq": seq, "iter": seq, **counts_files(live, seq, start=False),
            "frame": str(live / frame["file"]),
            "format": "dense", **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")}, "z_origin": 0.0,
            "z_spacing": 1.0, "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
            "ncells": frame["cells"], "params": params, "out_json": str(out / f"{name}.json"),
            "out_cells": str(out / f"{name}.cells.bin")}


def layout_of(item: dict) -> np.ndarray:
    return np.array(item["layout"], dtype=float)


def moved(a: dict, b: dict) -> float:
    """The median per-cell displacement between two passes' layouts (layout units), over the cells both placed."""
    before, after = layout_of(a), layout_of(b)
    both = np.isfinite(before[:, 0]) & np.isfinite(after[:, 0])
    return float(np.median(np.linalg.norm(after[both, :2] - before[both, :2], axis=1)))


def untimed(value):
    """A summary without its timings and its files' paths (what two runs of the same work may differ in)."""
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if k not in ("ms", "worker_ms", "load_ms", "files")}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def closest_dots(item: dict, share: float = 0.1) -> float:
    """The `share` quantile of the distances from the dots of a pass's layout to their nearest other dot: its closest
    dots, where min_dist acts (at 0 dots may pile up, at 1 they keep apart), whatever the clusters' arrangement, which
    sways the median as much as min_dist does (the layout is scaled to its 95th-percentile radius)."""
    points = layout_of(item)
    points = points[np.isfinite(points[:, 0]), :2]
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.quantile(distances.min(axis=1), share))


# ---- the hello -------------------------------------------------------------------------------------------------------
def test_without_cuda_the_worker_runs_on_the_cpu(numba_cache):
    """Where no CUDA device is visible (or RAPIDS is not installed) the worker picks the CPU itself, loads its libraries
    and says so in its hello, with why the GPU is not used and the threads it was given."""
    with worker_process(cpu_env(numba_cache)) as worker:
        hello = worker.reply()
    assert hello["ok"] is True and hello["ready"] is True and hello["backend"] == "cpu"
    assert hello["pci_bus_id"] is None and hello["threads"] == 4
    assert isinstance(hello["why"], str) and hello["why"], hello
    assert worker.process.returncode == 0                             # its stdin closed: it ends on its own


def test_the_hello_says_why_the_cpu_libraries_cannot_load(numba_cache, tmp_path):
    """A CPU stack that is installed but cannot load (numba refusing this NumPy, as it does when NumPy is newer than it
    knows) is named in the hello, which says it lasts (the engine stops asking this session); a cache folder that cannot
    be made never stops the worker: numba's cache goes to a temporary folder, said in the worker's log."""
    fake = tmp_path / "fake" / "numba"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text('raise ImportError("Numba needs NumPy 2.5 or less. Got NumPy 2.6.")\n')
    with worker_process(cpu_env(numba_cache, PYTHONPATH=f"{fake.parent}{os.pathsep}{APP.parent}")) as worker:
        hello = worker.reply()
        assert worker.close() == 1
    assert {k: hello[k] for k in ("ok", "ready", "lasting", "missing")} == \
        {"ok": False, "ready": False, "lasting": True, "missing": []}
    assert hello["error"].startswith("Clusters & UMAP cannot run in this Python environment: ")
    assert hello["error"].endswith("its CPU libraries could not be loaded (ImportError: Numba needs NumPy 2.5 or less. "
                                   "Got NumPy 2.6.)")
    if getattr(os, "geteuid", lambda: 0)() == 0:
        return                                    # root writes anywhere (and Windows has no such folder modes)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("this file system ignores folder modes: no folder this user cannot write")
        with worker_process(cpu_env(None, **{backend.CPU_CACHE_ENV: str(locked / "cache")})) as worker:
            hello = worker.reply()
        assert hello["ready"] is True and hello["backend"] == "cpu", hello
        lines = [line for line in worker.lines if "numba's cache cannot be kept in" in line]
        assert len(lines) == 1 and str(locked / "cache") in lines[0], worker.log[-3000:]
        fallback = Path(lines[0].rsplit("): ", 1)[1].split(" instead", 1)[0])
        assert fallback.name.startswith("sopa-companion-numba-")
        assert any(fallback.iterdir())                                # numba's cache went there
        shutil.rmtree(fallback, ignore_errors=True)
    finally:
        locked.chmod(stat.S_IRWXU)


# ---- clusters and the UMAP -------------------------------------------------------------------------------------------
def test_cpu_clusters_write_the_gpus_files(numba_cache, tmp_path):
    """A CPU pass writes what a GPU pass writes: the cells file (int16 clusters, -1 for cells not kept, then float32
    [cells, 3] with z = 0 and NaN rows), a summary with the same keys and step names (no GPU memory: "gpu" null), and
    clusters that find the planted groups, each with its own colour."""
    folder, groups = write_grouped_run(tmp_path / "run", cells=60, low=3)
    out = tmp_path / "out"
    out.mkdir()
    request = umap_request(folder, 0, out)
    found, _ = run_case("passes", numba_cache, requests=[request], truth=groups.tolist())
    item = found["passes"][0]
    assert item["reply"]["ok"], item["reply"]
    summary = item["reply"]["summary"]
    raw = Path(request["out_cells"]).read_bytes()
    assert len(raw) == 2 * 60 + 12 * 60
    clusters = np.frombuffer(raw[:120], "<i2")
    layout = np.frombuffer(raw[120:], "<f4").reshape(60, 3)
    kept = clusters >= 0
    assert (~kept).sum() == 3 and np.isnan(layout[~kept]).all()
    assert np.isfinite(layout[kept]).all() and (layout[kept, 2] == 0).all()
    assert set(summary) == W.SUMMARY_KEYS and set(json.loads(Path(request["out_json"]).read_text())) == W.SUMMARY_KEYS
    assert list(summary["ms"]) == ["load", "counts", "pca", "knn", "leiden", "markers", "umap", "write"]
    assert summary["gpu"] is None and summary["ncells"] == 57 and summary["nclusters"] == len(summary["clusters"])
    assert summary["transcripts_in_cells"] > 0 and summary["warm"] is False and summary["bounds"] is not None
    assert item["ari_truth"] >= 0.9, item["ari_truth"]
    assert all({"id", "size", "color", "label", "top", "centroid"} <= set(c) for c in summary["clusters"])
    colours = [c["color"] for c in summary["clusters"]]
    assert len(set(colours)) == len(colours)
    assert sorted(c["id"] for c in summary["clusters"]) == sorted(set(clusters[kept].tolist()))


def test_the_cpu_layout_starts_from_the_last_and_holds_still(numba_cache, tmp_path):
    """Passes on frames that differ in 5 % of their labels start from the last layout (warm): with no iterations at all
    a warm pass gives the last layout back (umap-learn only rescales its start: a linear map of it); with the usual 60
    the dots move a little and the clusters keep their ids. Leiden starts from the last clusters (each kept cell's), and
    afresh at a run's first pass and with a fresh layout. Every refresh_every updates the map is laid out afresh, then
    rotated onto the last one: no other turn or mirror of it lies closer to the last layout."""
    folder, _ = write_grouped_run(tmp_path / "run", cells=1000, gradient=True)
    out = tmp_path / "out"
    out.mkdir()
    requests = [umap_request(folder, f, out, run="a") for f in range(3)] + \
        [umap_request(folder, f, out, run="still", warm_epochs=0) for f in range(3)] + \
        [umap_request(folder, f, out, run="b", refresh_every=3) for f in range(3)]
    found, _ = run_case("passes", numba_cache, requests=requests, watch_leiden=True)
    assert all(p["reply"]["ok"] for p in found["passes"]), [p["reply"] for p in found["passes"]]
    kept = [p["reply"]["summary"]["ncells"] for p in found["passes"]]
    assert found["leiden_starts"] == [None, kept[1], kept[2], None, kept[4], kept[5], None, kept[7], None], \
        (found["leiden_starts"], kept)
    a1, a2, a3, _, s2, s3, b1, b2, b3 = found["passes"]
    warm = lambda *passes: [p["reply"]["summary"]["warm"] for p in passes]    # noqa: E731
    assert warm(a1, a2, a3) == [False, True, True] and warm(s2, s3) == [True, True]
    last, now = layout_of(s2)[:, :2], layout_of(s3)[:, :2]
    basis = np.column_stack([last, np.ones(len(last))])
    fit = np.linalg.lstsq(basis, now, rcond=None)[0]
    assert np.median(np.linalg.norm(basis @ fit - now, axis=1)) < 0.05     # the last layout, linearly mapped
    assert moved(a2, a3) < 2.5, moved(a2, a3)                         # the layout's own iterations move dots a little
    ids = [{c["id"] for c in p["reply"]["summary"]["clusters"]} for p in (a2, a3)]
    assert ids[0] == ids[1] and a3["ari_prev"] >= 0.9, (ids, a3["ari_prev"])
    assert warm(b1, b2, b3) == [False, True, False]                   # the 3rd update: laid out afresh
    last, now = layout_of(b2), layout_of(b3)
    both = np.isfinite(last[:, 0]) & np.isfinite(now[:, 0])
    last, now = last[both, :2], now[both, :2]
    square = lambda turned: float(np.mean(((turned - last) ** 2).sum(axis=1)))    # noqa: E731
    turns = [np.array(m, float) for m in ([[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]], [[1, 0], [0, -1]],
                                          [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]])]
    assert all(square(now) < square(now @ turn) for turn in turns)


def test_cpu_passes_honour_the_five_settings(numba_cache, tmp_path):
    """Round 8b's five settings change the CPU's passes as they change the GPU's: the resolution the number of clusters,
    the neighbours and the fewest transcripts what is kept and laid out, the dot spacing the layout; "Genes used" is
    held to the CPU's maximum (5,000) whatever is asked, said once in the worker's log, and too many neighbours for the
    cells is the too-few-cells reason."""
    folder, _ = write_grouped_run(tmp_path / "run", cells=60, low=3)
    genes, _ = write_grouped_run(tmp_path / "genes", cells=400, genes=6000, seed=1)
    out = tmp_path / "out"
    out.mkdir()
    asks = {"coarse": {"resolution": 0.2}, "fine": {"resolution": 3.0}, "five": {"n_neighbors": 5},
            "fifty": {"n_neighbors": 50}, "fewest10": {"min_counts": 10}, "fewest60": {"min_counts": 60},
            "genes5": {"n_top_genes": 5}, "tight": {"min_dist": 0.0}, "spread": {"min_dist": 1.0},
            "crowded": {"n_neighbors": 59}}
    requests = [umap_request(folder, 0, out, run=run, **params) for run, params in asks.items()]   # each a run afresh
    requests += [umap_request(genes, f, out, run="genes", n_top_genes=10000) for f in range(2)]
    # The dot spacing on two more sets of cells: a layout's numbers depend on the computer (umap-learn's parallel
    # steps, numba compiled for its processor), so two of the three sets must show it, not each
    spacing = [write_grouped_run(tmp_path / f"spacing{seed}", cells=60, low=3, seed=seed)[0] for seed in (1, 2)]
    requests += [umap_request(other, 0, out, run=f"{kind}{i}", min_dist=d) for i, other in enumerate(spacing)
                 for kind, d in (("tight", 0.0), ("spread", 1.0))]
    found, log = run_case("passes", numba_cache, requests=requests)
    p = dict(zip(asks, found["passes"]))
    summary = {run: item["reply"].get("summary") for run, item in p.items()}
    assert summary["coarse"]["nclusters"] < summary["fine"]["nclusters"], (summary["coarse"]["nclusters"],
                                                                           summary["fine"]["nclusters"])
    assert summary["five"] is not None and summary["fifty"] is not None
    assert summary["fewest60"]["ncells"] < summary["fewest10"]["ncells"] == 57
    assert summary["genes5"] is not None and summary["genes5"]["params"]["n_top_genes"] == 5
    more = found["passes"][len(asks) + 2:]
    spaced = [(closest_dots(t), closest_dots(s)) for t, s in [(p["tight"], p["spread"]), *zip(more[0::2], more[1::2])]]
    assert sum(spread > tight for tight, spread in spaced) >= 2, spaced
    assert p["crowded"]["reply"]["ok"] is False and p["crowded"]["reply"]["reasons"] == {"umap": "too_few_cells"}
    held = [item["reply"] for item in found["passes"][len(asks):len(asks) + 2]]
    assert all(r["ok"] and r["summary"]["params"]["n_top_genes"] == cpu.MAX_GENES for r in held), held
    assert log.count("Genes used is at most 5000 on the CPU (asked: 10000)") == 1, log[-3000:]
    assert cpu.MAX_GENES == CPU_MAX_GENES == 5000                     # the engine's limit while its worker is the CPU


def test_leiden_starts_from_the_last_clusters_while_the_settings_stay():
    """The CPU's Leiden starts from the run's last clusters (cells not seen before with ids of their own) while the
    settings that find them stay, and afresh when one changes, when the map is laid out afresh, when fewer than half of
    the cells were seen before, or with no clusters yet."""
    p = dict(W.DEFAULTS)
    keep = np.arange(7)
    run = SimpleNamespace(clusters=np.array([4, 4, 1, 1, -1, 2, 2, -1], np.int32), leiden_key=cpu.leiden_key(p),
                          updates=4)
    assert cpu.leiden_start(run, keep, p) == [4, 4, 1, 1, 5, 2, 2]     # a cell not seen before: an id of its own
    assert cpu.leiden_start(run, np.array([0, 2, 4, 5, 7]), p) == [4, 1, 5, 2, 6]
    assert cpu.leiden_start(run, keep, {**p, "resolution": 2.0}) is None          # a setting changed: afresh
    assert cpu.leiden_start(run, np.array([0, 4, 7]), p) is None                  # under half of them seen before
    assert cpu.leiden_start(SimpleNamespace(**{**vars(run), "updates": int(p["refresh_every"]) - 1}), keep, p) is None
    assert cpu.leiden_start(SimpleNamespace(**{**vars(run), "clusters": None}), keep, p) is None
    assert cpu.leiden_key({**p, "min_dist": 0.9}) == cpu.leiden_key(p)            # the spacing only moves the dots


def test_numba_thinning_draws_as_numpy_does(numba_cache):
    """The CPU worker's numba thinning (Compare's binomial draws of a run's column) makes numpy's draws, bit for bit, on
    random counts and the edge cases (no transcripts, nothing kept, everything kept, no entries, the largest seed)."""
    found, _ = run_case("thinning", numba_cache)
    assert found["trials"] == 40 and len(found["equal"]) == 45 and all(found["equal"]), found["equal"]
    assert found["thinned_something"]


# ---- the mixing check, Go deeper and Compare on the CPU worker -------------------------------------------------------
def test_the_cpu_mixing_check_writes_numpys_bytes(numba_cache, tmp_path):
    """The mixing check's start line and frames (each against its start block, like for like), and Go deeper on a
    frame's cell (with its change since the start) and on the start line, give the same bytes, files and answers on the
    CPU worker (proseg's count as a scipy CSR) as numpy's dense counts do."""
    run = write_run(tmp_path / "run")
    live = run / "live"
    (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    model = make_deep(tmp_path / "model", run)
    info = json.loads((live / "info.json").read_text())
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    common = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    qc = {"model": str(model), "model_id": "feedface", "mixed_at": 0.3, "judge_min": JUDGE, "fit_min": 5,
          "grid": {"x0": 100.0, "y0": 50.0, "bin": 10.0, "nx": 4, "ny": 3}}
    prior = {**info["prior"], "file": str(live / info["prior"]["file"])}
    deeper = {"model": str(model), "model_id": "feedface", "start": False, "k": K, "fit_min": 5, "judge_min": JUDGE,
              "codes": "@OUT@/000002.qc.bin", "cell_ids": str(live / "cells.txt"), "v": 1}
    frame_request = {"op": "analyze", **common, "seq": 2, "iter": 2, **counts_files(live, 2),
                     "frame": str(live / frame["file"]),
                     "format": "dense", **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                     "cells_file": str(live / frame["cells_file"]),
                     "qc": {**qc, "start": "@OUT@/start.npz", "out_cells": "@OUT@/000002.qc.bin",
                            "out_bins": "@OUT@/000002.bins.npz", "out_json": "@OUT@/000002.qc.json"}}
    requests = [{"op": "qc_start", **common, "seq": None, **counts_files(live, 0, start=False), "prior": prior,
                 "qc": {**qc, "out_cells": "@OUT@/start.bin", "out_start": "@OUT@/start.npz",
                        "out_bins": "@OUT@/start.bins.npz", "out_json": "@OUT@/start.json"}}]
    requests += [{**frame_request, "ops": ["qc", "deeper"], "deeper": {**deeper, "cell": c, "start_line": True}}
                 for c in range(4)]
    requests += [{**frame_request, "ops": ["deeper"], "deeper": {**deeper, "cell": 5}},
                 {"op": "deeper", **common, "seq": "start", **counts_files(live, 0, start=False),
                  "deeper": {**deeper, "start": True, "cell": 1, "codes": "@OUT@/start.bin"}}]
    out_numpy, out_cpu = tmp_path / "numpy", tmp_path / "cpu"
    found, _ = run_case("mixing", numba_cache, requests=requests, out_numpy=str(out_numpy), out_cpu=str(out_cpu))
    assert found["thin"] and found["numba_draws"] == 0               # like for like: no draws on either worker
    replies = {name: json.loads(json.dumps(found[name]).replace(str(folder), "@OUT@"))
               for name, folder in (("numpy", out_numpy), ("cpu", out_cpu))}
    assert all(r["ok"] for r in replies["numpy"]), replies["numpy"]
    assert replies["cpu"] == replies["numpy"]
    assert sum(r.get("deeper", {}).get("state") == "ready" for r in replies["cpu"]) >= 2
    assert replies["cpu"][1]["qc"]["matched"]["paired"] > 0           # the frame against its start block
    assert any("start_change" in r.get("deeper", {}) for r in replies["cpu"])
    names = sorted(p.name for p in out_numpy.iterdir())
    assert names == sorted(p.name for p in out_cpu.iterdir()) == sorted(
        ["start.bin", "start.npz", "start.bins.npz", "start.json", "000002.qc.bin", "000002.bins.npz",
         "000002.qc.json"]), names
    # The bytes, counts and codes are the same; sums of shares agree to float32's last digits only (a dense product
    # and a sparse one add the same terms in other orders).
    for name in names:
        a, b = out_numpy / name, out_cpu / name
        if name.endswith(".npz"):
            with np.load(a) as x, np.load(b) as y:
                assert sorted(x.files) == sorted(y.files)
                for key in x.files:
                    if x[key].dtype.kind == "f":
                        np.testing.assert_allclose(y[key], x[key], rtol=1e-5, atol=1e-6, err_msg=f"{name}: {key}")
                    else:
                        assert np.array_equal(x[key], y[key]), (name, key)
        elif name.endswith(".json"):
            assert untimed(json.loads(a.read_text())) == untimed(json.loads(b.read_text())), name
        else:
            assert a.read_bytes() == b.read_bytes(), name


def test_compare_lays_out_with_umap_on_the_cpu(numba_cache, tmp_path):
    """Compare on the CPU worker lays its columns' points out in a real UMAP (umap-learn, laid out cold as umap() lays a
    run out afresh), with the view box and one points file per column; numpy alone keeps its stand-in (two principal
    components). The run's like-for-like cut comes from the same binomial draws on both (numba's and numpy's)."""
    run = write_run(tmp_path / "run")
    live = run / "live"
    (live / "cells.txt").write_text("".join(c + "\n" for c in RUN_IDS))
    model = make_deep(tmp_path / "model", run)
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[1])      # frame 1: no sampled count, so the
    rng = np.random.default_rng(7)                                                # run's cut is from binomial draws
    learning = write_counts(tmp_path / "learning.npz", rng.poisson(30, (len(LEARNED), len(MODEL_GENES))), MODEL_GENES,
                            np.zeros(len(LEARNED)), np.zeros(len(LEARNED)), ids=LEARNED)
    other = write_counts(tmp_path / "other.npz", rng.poisson(rng.gamma(0.6, 50, (300, len(MODEL_GENES)))),
                         MODEL_GENES[::-1], rng.uniform(X0, X0 + 30, 300), rng.uniform(Y0, Y0 + 20, 300))
    spec = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
            **counts_files(live, 1, start=False), "frame": str(live / frame["file"]), "format": "dense",
            **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")}, "z_origin": 0.0, "z_spacing": 0.5,
            "ncells": NCELLS, "cells_file": str(live / frame["cells_file"])}
    request = {"op": "compare", "run": "r1", "model": str(model), "model_id": "feedface",
               "kinds": {"path": "@OUT@/kinds.npz", "counts": learning},
               "columns": [{"key": "run", "name": "This run", "frame": spec}, {"key": "other", "name": "B",
                                                                               "counts": other}],
               "area": None, "max_points": W.COMPARE_MAX_POINTS, "seed": 0, **QC, "out_dir": "@OUT@/points"}
    found, _ = run_case("compare", numba_cache, request=request, out=str(tmp_path / "out"))
    for name, how in (("cpu", "umap"), ("numpy", "pca")):
        reply = found[name]
        assert reply["ok"], reply
        result = reply["compare"]
        assert result["layout"] == how and result["layout_error"] is None and result["bounds"] is not None, result
        assert result["points"] > 30 and result["points"] == sum(c["umap_points"] for c in result["columns"])
        # each its own: the columns with enough points laid out and clustered alone, on the CPU as with numpy
        assert result["own_layout"] == how and result["own_error"] is None and result["own_bounds"] is not None, result
        assert any(c["own_points_file"] for c in result["columns"]), result["columns"]
        for column in result["columns"]:
            assert column["points_size"] == 10 * column["umap_points"], column
    numbers = ("cells", "cells_in_area", "judged", "mixed", "mixed_per100", "like100_per100", "histogram", "median_tx")
    assert [{k: c[k] for k in numbers} for c in found["cpu"]["compare"]["columns"]] == \
        [{k: c[k] for k in numbers} for c in found["numpy"]["compare"]["columns"]]     # only the picture differs


# ---- the warm-up -----------------------------------------------------------------------------------------------------
def test_the_cpu_warm_up_runs_while_idle(numba_cache, tmp_path):
    """The CPU worker's libraries load before its hello; while it is idle its warm-up takes four steps, one at a time
    (a tiny PCA and its neighbours, Leiden, umap-learn's layouts, the thinning), so the first pass finds umap-learn's
    layout compiled. Four lines fit what the Log keeps of the newest warm-up."""
    from sopa_companion.engine.runlog import WARM_KEPT
    steps = cpu.warm_steps(W.Worker(xp=np, backend="cpu"))
    assert [s.__name__ for s in steps] == ["pca_neighbours", "leiden", "layouts", "thinning"]
    assert len(steps) <= WARM_KEPT
    folder, _ = write_grouped_run(tmp_path / "run", cells=60)
    out = tmp_path / "out"
    out.mkdir()
    with worker_process(cpu_env(numba_cache)) as worker:
        hello = worker.reply()
        assert hello["ready"] and hello["backend"] == "cpu", hello
        try:
            wait_for(lambda: "warm-up thinning:" in worker.log or "warm-up stopped" in worker.log, timeout=120)
        except AssertionError:
            raise AssertionError(f"no warm-up within 120 s: {worker.log[-3000:]}") from None
        names = [line.split("warm-up ", 1)[1].split(":", 1)[0] for line in worker.lines if "warm-up " in line]
        assert names == ["pca_neighbours", "leiden", "layouts", "thinning"], worker.log[-3000:]
        worker.send(umap_request(folder, 0, out))
        reply = worker.reply()
    assert reply["ok"], reply
    assert reply["summary"]["ms"]["umap"] < 3000, reply["summary"]["ms"]   # no compile left (a busy machine: generous)


# ---- against the GPU's own results -----------------------------------------------------------------------------------
def measured_run() -> Path:
    """The first crop run in the harness folder (oldest first) whose counted frames the GPU analysed on proseg's count
    (analysis/pc1) at the default settings, 25 stored results or more: read only. (A run made by the harness's
    play_umap with --keep: `plh run prepare_paused play_umap --sample crop --keep`.)"""
    from sopa_companion.analysis import counts_file
    runs = harness_path("var", "runs")
    for run in sorted(p for p in runs.iterdir() if "crop" in p.name and (p / "live" / "counts.jsonl").is_file()):
        stored = sorted((run / "analysis" / counts_file.BASIS).glob("[0-9]" * 6 + ".json"))
        if len(stored) >= 25:
            params = json.loads(stored[0].read_text()).get("params") or {}
            if all(params.get(k) == W.DEFAULTS[k] for k in ("resolution", "n_neighbors", "min_dist", "min_counts",
                                                             "n_top_genes", "n_pcs")):
                return run
    pytest.skip("no crop run in the harness folder has 25 GPU results on proseg's count at the default settings")


def test_cpu_results_agree_with_the_gpus_stored_ones(numba_cache, tmp_path):
    """Over a crop's frames that the GPU analysed: the CPU keeps the same cells, finds clusters that agree with the
    GPU's (as two Leiden runs agree), moves the dots about as much between frames, keeps as many of each cell's PCA
    neighbours in the layout, and its clusters hold steadier from frame to frame than the GPU's: Leiden started from
    the last clusters keeps a median agreement of 0.8 or more with the previous frame's (0.88 measured; 0.72 started
    afresh each time, still above the GPU's 0.65). Each frame is read on the form of proseg's count the GPU read it on
    (its stored result's `count`: the count proseg saves for a frame counted for a pause, the hold, a finish or the
    end). Since proseg's counts wait in the engine's spool (2026-10-31), a run keeps the count files of its lasting
    frames alone (its start, an upscale, a pause, the hold, a finish, the end), far apart: the frames read are those,
    and the floor of 0.8 holds between consecutive counted frames only (far apart, both backends' clusters change
    with the run; the CPU's must still hold at least as steady as the GPU's)."""
    from sopa_companion.analysis import counts_file
    run = measured_run()
    order = [json.loads(line)["seq"] for line in (run / "live" / "counts.jsonl").read_text().splitlines()]
    counted = sorted(set(order))
    seqs = sorted(int(p.name[:6]) for p in (run / "analysis" / counts_file.BASIS).glob("[0-9]" * 6 + ".cells.bin")
                  if int(p.name[:6]) in counted and (run / "live" / "frames" / f"{p.name[:6]}.counts.zst").is_file())
    assert len(seqs) >= 2, f"{run.name}: fewer than two counted frames kept with their count files ({seqs})"
    consecutive = all(counted.index(b) - counted.index(a) == 1 for a, b in zip(seqs, seqs[1:]))
    found, _ = run_case("agree", numba_cache, timeout=1800, run=str(run), seqs=seqs, out=str(tmp_path / "out"))
    frames = found.get("frames")
    assert frames and len(frames) == len(seqs), found
    assert all(f["same_kept"] for f in frames)
    assert min(f["ari_gpu"] for f in frames) >= 0.5, [round(f["ari_gpu"], 3) for f in frames]
    median = lambda key: float(np.median([f[key] for f in frames if key in f]))   # noqa: E731
    assert median("moved_cpu") <= 1.5 * median("moved_gpu"), (median("moved_cpu"), median("moved_gpu"))
    assert median("pca_kept_cpu") >= 0.9 * median("pca_kept_gpu"), (median("pca_kept_cpu"), median("pca_kept_gpu"))
    assert median("ari_prev_cpu") >= max(median("ari_prev_gpu"), 0.8 if consecutive else 0.0), \
        (median("ari_prev_cpu"), median("ari_prev_gpu"), seqs)
