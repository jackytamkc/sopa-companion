"""The parts of tests/test_worker_cpu.py that need the CPU worker's libraries (scipy, scikit-learn, igraph, umap-learn,
numba), which the app's own venv does not have: each case runs in a Python that has them, as

    <python> -m sopa_companion.tests.cpu_cases <case> <its arguments as JSON>

and prints one line of JSON with what the test checks. The tests make the data and judge the answers; a case only runs
the worker's code (Worker(xp=np, backend="cpu"), as the CPU worker process makes it) and measures what needs these
libraries (adjusted Rand indices, neighbour sets)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from sopa_companion.analysis import counts_file
from sopa_companion.analysis import worker as W
from sopa_companion.analysis import cpu

TIMINGS = ("ms", "worker_ms", "load_ms")      # what two equal answers may differ in


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def cells_file(path, ncells: int):
    """(clusters int16 -> int, layout [ncells, 3]) of a pass's .cells.bin."""
    raw = Path(path).read_bytes()
    return (np.frombuffer(raw[:2 * ncells], "<i2").astype(np.int64),
            np.frombuffer(raw[2 * ncells:], "<f4").reshape(-1, 3))


def passes(args: dict) -> dict:
    """Clusters and the UMAP on one CPU-mode worker, request after request (args["requests"]), as the worker process
    takes them. Per pass: the reply, the clusters and layout of its cells file, the adjusted Rand index of the kept
    cells' clusters against the planted groups (args["truth"]) and against the previous pass's. With
    args["watch_leiden"], igraph's own Leiden call is watched: for each pass, how many cells its start gave a cluster
    (the last clusters), None for a fresh start."""
    starts = []
    if args.get("watch_leiden"):
        import igraph
        leiden = igraph.Graph.community_leiden

        def watched(graph, *rest, **options):
            start = options.get("initial_membership")
            starts.append(None if start is None else len(start))
            return leiden(graph, *rest, **options)
        igraph.Graph.community_leiden = watched
    worker = W.Worker(xp=np, backend="cpu")
    truth = np.asarray(args["truth"]) if args.get("truth") is not None else None
    out, before = [], {}
    for request in args["requests"]:
        reply = worker.handle(request)
        item = {"reply": reply}
        if reply.get("ok"):
            clusters, layout = cells_file(request["out_cells"], int(request["ncells"]))
            kept = clusters >= 0
            item.update(clusters=clusters.tolist(), layout=np.where(np.isfinite(layout), layout, None).tolist(),
                        size=Path(request["out_cells"]).stat().st_size)
            if truth is not None:
                item["ari_truth"] = ari(truth[kept], clusters[kept])
            last = before.get(request["run"])                # the same run's previous pass
            if last is not None:
                both = kept & (last >= 0)
                item["ari_prev"] = ari(last[both], clusters[both])
            before[request["run"]] = clusters
        out.append(item)
    return {"passes": out, "leiden_starts": starts}


def thinning(args: dict) -> dict:
    """The numba thinning against numpy's (worker.thin_values: Compare's binomial draws of a run's column) on random
    trials and the edge cases."""
    from sopa_companion.analysis import cpu_kernels
    rng = np.random.default_rng(int(args.get("seed", 0)))
    W.CPU_THIN = None
    trials = []
    for t in range(40):
        n = int(rng.integers(0, 400))
        values = rng.poisson(rng.gamma(0.5, 8.0, n)).astype(np.float32)
        p = rng.uniform(0, 1, n).astype(np.float32)
        trials.append((values, p, int(rng.integers(0, 2 ** 63)) * 2 + int(rng.integers(0, 2))))
    edges = [(np.zeros(5, np.float32), np.full(5, 0.5, np.float32), 7),                    # no transcripts
             (np.full(6, 9, np.float32), np.zeros(6, np.float32), 8),                       # nothing kept
             (np.full(6, 9, np.float32), np.ones(6, np.float32), 9),                        # everything kept
             (np.zeros(0, np.float32), np.zeros(0, np.float32), 10),                        # no entries at all
             (np.array([3000.0, 1.0], np.float32), np.array([0.25, 0.75], np.float32), 2 ** 64 - 1)]
    equal = [bool(np.array_equal(cpu_kernels.thin(v, p, s), W.thin_values(np, v, p, s))) for v, p, s in trials + edges]
    values, keep, seed = trials[0]
    return {"trials": len(trials), "edges": len(edges), "equal": equal,
            "thinned_something": bool(cpu_kernels.thin(values, keep, seed).sum() < values.sum())}


def substitute(value, out: str):
    """A request with every "@OUT@" in its strings replaced by `out` (each worker's own output folder)."""
    if isinstance(value, str):
        return value.replace("@OUT@", out)
    if isinstance(value, list):
        return [substitute(v, out) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, out) for k, v in value.items()}
    return value


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if k not in TIMINGS}
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


def mixing(args: dict) -> dict:
    """The mixing check and Go deeper (args["requests"], their output paths under "@OUT@") on a numpy-mode worker
    (proseg's count as dense rows) and then, after cpu.enable(), on a CPU-mode worker (a scipy CSR), each into its own
    folder: their replies without timings, and every file each wrote, read back; and how many draws went through
    numba's thinning (none: the like-for-like change draws nothing)."""
    def run(worker, out: Path) -> list:
        out.mkdir(parents=True, exist_ok=True)
        return [without_timings(worker.handle(substitute(request, str(out)))) for request in args["requests"]]
    W.CPU_THIN = None
    numpy_replies = run(W.Worker(xp=np), Path(args["out_numpy"]))
    cpu.enable()
    thin, draws = W.CPU_THIN, []

    def counted(values, p, seed):
        draws.append(len(values))
        return thin(values, p, seed)
    if thin is not None:
        W.CPU_THIN = counted
    cpu_replies = run(W.Worker(xp=np, backend="cpu"), Path(args["out_cpu"]))
    return {"numpy": numpy_replies, "cpu": cpu_replies, "thin": thin is not None, "numba_draws": len(draws)}


def compare(args: dict) -> dict:
    """Op "compare" (args["request"], its out_dir under "@OUT@") on a CPU-mode worker and on a numpy-mode one: the
    results without timings and the size of each points file."""
    found = {}
    for name, worker in (("cpu", W.Worker(xp=np, backend="cpu")), ("numpy", W.Worker(xp=np))):
        out = Path(args["out"]) / name
        out.mkdir(parents=True, exist_ok=True)
        reply = worker.handle(substitute(args["request"], str(out)))
        if reply.get("ok"):
            for column in reply["compare"]["columns"]:
                column["points_size"] = Path(column["points_file"]).stat().st_size if column["points_file"] else None
        found[name] = without_timings(reply)
    return found


def agree(args: dict) -> dict:
    """The CPU's passes over a measured run's counted frames that the GPU analysed too (its stored results on proseg's
    count, read only), in order: per frame, whether the kept cells are the GPU's, the adjusted Rand index of the
    clusters against the GPU's, and both backends' movement (median per-cell displacement) and cluster agreement with
    their previous frame, and the share of each cell's 15 PCA neighbours kept in each layout."""
    from sklearn.neighbors import NearestNeighbors
    run = Path(args["run"])
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((run / "live" / "info.json").read_text())
    frames, counts = {}, {}
    for line in (run / "live" / "frames.jsonl").read_text().splitlines():
        record = json.loads(line)
        frames[record["seq"]] = record
    for line in (run / "live" / "counts.jsonl").read_text().splitlines():     # proseg's count of each counted frame
        record = json.loads(line)
        counts[record["seq"]] = run / "live" / record["file"]
    ncells = int(info["cells"])
    worker = W.Worker(xp=np, backend="cpu")

    def kept_share(pcs, layout, k=15):
        a = NearestNeighbors(n_neighbors=k + 1).fit(pcs).kneighbors(pcs, return_distance=False)[:, 1:]
        b = NearestNeighbors(n_neighbors=k + 1).fit(layout).kneighbors(layout, return_distance=False)[:, 1:]
        return sum(np.intersect1d(x, y).size for x, y in zip(a, b)) / (k * len(a))

    def moved(before, after):
        both = np.isfinite(before[:, 0]) & np.isfinite(after[:, 0])
        return float(np.median(np.linalg.norm(after[both, :2] - before[both, :2], axis=1)))

    found, before = [], None
    for seq in args["seqs"]:
        record = frames[seq]
        # The form of proseg's count the GPU read this frame on (its stored result's `count`): "sampled", the count
        # proseg saves, for a frame counted for a pause, the hold, a finish or the end; else the averaged count. The
        # CPU reads it the same way, as the engine asks both (engine/analysis.py: `count` only when "sampled").
        gpu_result = json.loads((run / "analysis" / counts_file.BASIS / f"{seq:06d}.json").read_text())
        form = {"count": "sampled"} if gpu_result.get("count") == "sampled" else {}
        request = {"op": "analyze", "ops": ["umap"], "run": run.name, "seq": seq, "iter": record.get("iter"), **form,
                   "counts_file": str(counts[seq]),
                   "transcripts": str(run / "live" / "transcripts.bin.gz"), "genes": str(run / "live" / "genes.txt"),
                   "frame": str(run / "live" / record["file"]), "format": record.get("format", "dense"),
                   **{k: record[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                   "z_origin": info["z"]["grid_origin"], "z_spacing": info["z"]["grid_spacing"], "ncells": ncells,
                   "out_json": str(out / f"{seq:06d}.json"), "out_cells": str(out / f"{seq:06d}.cells.bin")}
        reply = worker.handle(request)
        if not reply.get("ok"):
            return {"error": reply.get("error"), "seq": seq}
        clusters, layout = cells_file(request["out_cells"], ncells)
        gpu_clusters, gpu_layout = cells_file(run / "analysis" / counts_file.BASIS / f"{seq:06d}.cells.bin", ncells)
        kept = clusters >= 0
        # The pass's own PCA, made again as it makes it, for the neighbours each layout keeps.
        matrix, _ = worker.count(worker.state(request), request)
        p = dict(W.DEFAULTS)
        totals, keep = W.umap_cells(np, matrix, p)
        x = cpu.normalised(matrix, keep, totals)
        top = cpu.top_genes(x, int(p["n_top_genes"]))
        pcs = cpu.pca(*cpu.top_genes_csr(x, top), W.principal_components(p["n_pcs"], int(keep.size), len(top)))
        item = {"seq": seq, "same_kept": bool(np.array_equal(kept, gpu_clusters >= 0)),
                "ari_gpu": ari(gpu_clusters[kept], clusters[kept]),
                "pca_kept_cpu": kept_share(pcs, layout[keep, :2]),
                "pca_kept_gpu": kept_share(pcs, gpu_layout[keep, :2])}
        if before is not None:
            cpu_before, gpu_before = before
            item["moved_cpu"], item["moved_gpu"] = moved(cpu_before[1], layout), moved(gpu_before[1], gpu_layout)
            both, gboth = kept & (cpu_before[0] >= 0), (gpu_clusters >= 0) & (gpu_before[0] >= 0)
            item["ari_prev_cpu"] = ari(cpu_before[0][both], clusters[both])
            item["ari_prev_gpu"] = ari(gpu_before[0][gboth], gpu_clusters[gboth])
        before = ((clusters, layout), (gpu_clusters, gpu_layout))
        found.append(item)
    return {"frames": found}


CASES = {"passes": passes, "thinning": thinning, "mixing": mixing, "compare": compare, "agree": agree}


if __name__ == "__main__":
    print(json.dumps(CASES[sys.argv[1]](json.loads(sys.argv[2])), default=float), flush=True)
