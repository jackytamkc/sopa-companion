"""A CPU stand-in for analysis/worker.py speaking the same JSON-lines protocol (engine tests). Its UMAP part is a fake
with the real summary's fields (2-D layout with z = 0, cluster centroids, view bounds), computed with the worker's own
helpers; the mixing check (ops "qc" and "qc_start"), Go deeper (op "deeper", also on the start line) and Compare (op
"compare": its numbers, kinds and shared layout) are the worker's own code, run with numpy instead of cupy. Like the
worker it answers every request but "forget", also one it cannot take (an unknown op, a line that is not JSON:
worker.answer), so the engine never waits for a reply that would not come. FAKE_ANALYSIS_CRASH_AT=<seq> exits when
asked for that frame, to test worker restarts; FAKE_ANALYSIS_BUS names the PCI bus its hello reports (the real worker's
GPU), none without it; FAKE_ANALYSIS_BACKEND the backend it says it runs on ("gpu", "cpu"; none without it, as a worker
of the previous version). FAKE_ANALYSIS_LASTING=<a,b> makes it say, as the real worker does in a Python that cannot run
Clusters & UMAP at all, that it lacks those packages (a lasting not-ready hello), and exit 1. A UMAP request that names
the frame's counts file (proseg's own count, analysis/counts_file.py) gets the real summary's basis, count (the
request's, echoed: worker.request_count) and transcripts_in_cells = round(Σ) of that count from it (e, or f on a frame
read on the count proseg saves); without one, or with one it cannot read, today's summary (the fake never refuses a
UMAP)."""
import json, os, sys
import numpy as np
from sopa_companion.analysis import counts_file
from sopa_companion.analysis.worker import OPS, Worker, answer, cluster_centroids, layout_bounds, request_count, \
    request_ops


def proseg_count(request):
    """round(Σ) of the request's counts file in the count it names (request_count: e by default, f on a frame read on
    the count proseg saves), as the real worker's transcripts_in_cells; None without a readable one that fits the run or
    holds it."""
    if not request.get("counts_file"):
        return None
    try:
        counts = counts_file.read(request["counts_file"], ncells=request.get("ncells"))
        live = getattr(counts, counts_file.ARRAY[request_count(request)])
    except (OSError, ValueError):           # missing, malformed or another run's (CountsError is a ValueError)
        return None
    return None if live is None else round(float(np.sum(live, dtype=np.float64)))


def fake_umap(request):
    n = request["ncells"]
    clusters = (np.arange(n) % 2).astype("<i2")
    coords = np.zeros((n, 3), "<f4")
    coords[:, :2] = np.random.default_rng(request["seq"]).normal(size=(n, 2))
    open(request["out_cells"], "wb").write(clusters.tobytes() + coords.tobytes())
    centroids = cluster_centroids(clusters, coords[:, :2])
    summary = {"seq": request["seq"], "iter": request.get("iter"), "ncells": n, "nclusters": 2, "umap_dims": 2,
               "bounds": layout_bounds(coords),
               "clusters": [{"id": c, "size": int((clusters == c).sum()), "color": c, "label": f"G{c}", "top": [],
                             "centroid": None if centroids.get(c) is None else
                             [round(float(centroids[c][0]), 3), round(float(centroids[c][1]), 3), 0.0]} for c in (0, 1)],
               "ms": {}, "params": request.get("params", {})}
    total = proseg_count(request)
    if total is not None:
        summary.update(basis="proseg", count=request_count(request), transcripts_in_cells=total)
    open(request["out_json"], "w").write(json.dumps(summary))
    return summary


if os.environ.get("FAKE_ANALYSIS_LASTING"):
    missing = [name for name in os.environ["FAKE_ANALYSIS_LASTING"].split(",") if name]
    error = "Clusters & UMAP cannot run in this Python environment: it lacks " + ", ".join(missing)
    print(json.dumps({"ok": False, "ready": False, "missing": missing, "lasting": True, "error": error}), flush=True)
    sys.exit(1)
worker = Worker(xp=np)
hello = {"ok": True, "ready": True}
if os.environ.get("FAKE_ANALYSIS_BUS"):
    hello["pci_bus_id"] = os.environ["FAKE_ANALYSIS_BUS"]
if os.environ.get("FAKE_ANALYSIS_BACKEND"):
    hello["backend"] = os.environ["FAKE_ANALYSIS_BACKEND"]
    hello["why"] = "no CUDA device is visible (CUDA_VISIBLE_DEVICES='')" if hello["backend"] == "cpu" else None
print(json.dumps(hello), flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
    except ValueError:
        request = None
    op = request.get("op") if isinstance(request, dict) else None
    if op in OPS and str(request.get("seq")) == os.environ.get("FAKE_ANALYSIS_CRASH_AT"):
        sys.exit(3)
    if op not in OPS or op in ("qc_start", "compare"):       # forget, qc_start, compare, refusals: the worker's own
        reply = answer(worker, line)
    else:
        ops = request_ops(request)
        named = request.get("ops") if isinstance(request.get("ops"), list) else ops
        real = [o for o in named if o != "umap"]             # the check, Go deeper and unknown ops: the worker's code
        reply = worker.handle({**request, "ops": real}) if real else \
            {"ok": True, "run": request["run"], "seq": request.get("seq")}
        if "umap" in ops:
            reply["summary"] = fake_umap(request)
    if reply is not None:
        print(json.dumps(reply), flush=True)
