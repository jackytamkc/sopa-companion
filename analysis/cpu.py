"""Clusters and the UMAP on the CPU: the analysis worker's backend where RAPIDS or a CUDA GPU is missing
(analysis/worker.py pick_backend), with what sopa installs (scipy, scikit-learn, igraph, umap-learn, numba). A pass
writes exactly the GPU's files and summary (the shared parts and Worker.finish_umap), so the page cannot tell them
apart. Its recipe, beside the GPU's:

  the frame's counts (proseg's own count, a scipy CSR: Worker.count, of the count the request names: e, or f, the
    count proseg saves, on a frame counted for a pause, the hold, a finish or the end) -> cells with >= min_counts
    -> library-size normalisation, log1p -> the n_top_genes most variable genes
    (at most MAX_GENES: the covariance below grows with genes squared)
  -> PCA: the exact covariance of those genes, summed over PCA_BLOCK_ROWS rows made dense at a time (never the whole
     dense matrix), and its top n_pcs eigenvectors (ARPACK's eigsh; numpy's eigh for a small panel): the same
     neighbours as a full decomposition finds, where a randomised one changes them
  -> each cell's exact nearest neighbours (scikit-learn's brute force, as cuML's; the cell itself first)
  -> igraph's Leiden on the graph cuGraph gets (unweighted, each cell's own link kept), LEIDEN_ITERATIONS passes started
     from the run's last clusters while the settings stay (leiden_start: steadier colours than a fresh start; cuGraph
     cannot start from a partition), afresh otherwise
  -> clusters matched to the previous update, top genes per cluster, the layout's start: the GPU's own code
  -> umap-learn's layout on the same neighbours: from the last layout (warm_epochs at warm_learning_rate) or a
     spectral start (cold_epochs), as cuML's, then normalised and rotated onto the last one by the shared writer.

The libraries are imported inside the functions, so this module loads (and its pure helpers run) where they are
missing; enable() loads them all before the worker's hello. Threads: OMP_NUM_THREADS as the engine gives it, for
scikit-learn's and numba's OpenMP and for numpy's BLAS; both OpenMP runtimes wait PASSIVE (asleep, not spinning
against each other: the neighbours step took 111 ms instead of 47 ms on a 3.4k-cell region without it)."""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np

from . import backend
from . import worker as W

LEIDEN_ITERATIONS = 2      # igraph Leiden passes per update: from the last clusters, 2 settle (40k cells: modularity
                           # within 0.001 of running it to the end, 0.12-0.18 s instead of 0.7-2.1 s)
PCA_BLOCK_ROWS = 4096      # rows made dense at a time for the covariance (a 40k x 5,000 dense matrix would be 0.76 GB)
SMALL_GENES = 256          # at most this many genes: numpy's full eigh (cheap); more: ARPACK's top n_pcs
MAX_GENES = 5000           # "Genes used" on the CPU: the covariance grows with genes squared (40k cells x 5,000 genes:
                           # PCA 1.5-1.6 s; 10,000 genes: 24-57 s); engine/analysis.py sends it as the field's limit
RNG = random.Random(0)     # igraph's random numbers, seeded again before every Leiden: a pass is repeatable, as
                           # cuGraph's random_state=0 makes the GPU's
WARM_CELLS = 512           # the warm-up's tiny fits: points ...
WARM_GENES = SMALL_GENES + 64   # ... and genes (more than SMALL_GENES, so its PCA goes through eigsh as a pass's does)
_AB: dict[float, tuple[float, float]] = {}   # min_dist -> umap-learn's curve (a, b): ab_params


def enable() -> None:
    """Make this process the CPU worker (main() calls it once, before the hello): the threads' settings first, then
    every library loaded, so one that cannot load fails here and the hello names it (numba checks its NumPy and
    llvmlite as it loads; umap-learn compiles its functions or loads them from numba's cache), then the numba thinning
    for Compare's binomial draws (worker.CPU_THIN). numba keeps its compiled code in $SOPA_COMPANION_CPU_CACHE (the
    engine's cache folder); a folder that cannot be made or written gives a temporary one, said in the worker's log (a
    cold cache costs ~3 s at the start, never the worker)."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads:
        os.environ.setdefault("NUMBA_NUM_THREADS", threads)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", threads)      # macOS's Accelerate
    cache = os.environ.get(backend.CPU_CACHE_ENV)
    if cache and "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = numba_cache(cache)
    import numba  # noqa: F401
    import scipy.sparse.linalg  # noqa: F401
    import sklearn.neighbors  # noqa: F401
    import igraph  # noqa: F401
    import umap.umap_  # noqa: F401
    import umap.layouts  # noqa: F401
    from . import cpu_kernels
    W.CPU_THIN = cpu_kernels.thin


def numba_cache(folder: str) -> str:
    """The folder numba's cache goes in: `folder` (made when missing) when this user can write there, else a new
    temporary one, said once in the worker's log. numba refuses to load a module whose cached functions have nowhere to
    go, so an unwritable cache would otherwise stop umap-learn's import."""
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
        return folder
    except OSError as exc:
        fallback = tempfile.mkdtemp(prefix="sopa-companion-numba-")
        W.log(f"numba's cache cannot be kept in {folder} ({W.problem(exc)}): {fallback} instead, for this worker")
        return fallback


# ---- genes, PCA, neighbours ------------------------------------------------------------------------------------------
def normalised(counts, keep: np.ndarray, totals: np.ndarray):
    """The kept cells' counts normalised to 1e4 transcripts per cell and log1p'd (a new CSR), as the GPU's umap()."""
    x = counts[keep]
    x.data *= np.repeat((1e4 / np.maximum(totals[keep], 1)).astype(np.float32), np.diff(x.indptr))
    np.log1p(x.data, out=x.data)
    return x


def top_genes(x, n_top: int) -> np.ndarray:
    """The n_top most variable genes of x (all of a smaller panel), most variable first: plain variance over the rows,
    from sums kept in float64 (near-ties may order otherwise than the GPU's float32: nothing that matters changes)."""
    n, width = x.shape
    mean = np.bincount(x.indices, weights=x.data, minlength=width) / max(n, 1)
    squares = np.bincount(x.indices, weights=x.data.astype(np.float64) ** 2, minlength=width) / max(n, 1)
    return np.argsort(-(squares - mean * mean), kind="stable")[:min(int(n_top), width)]


def top_genes_csr(x, top: np.ndarray):
    """(x's columns `top`, in that order, as a CSR (the remap rule of worker.csr_columns); each column's mean over the
    rows, float32)."""
    import scipy.sparse as sparse
    n, width = x.shape
    index = np.full(width, -1, np.int64)
    index[top] = np.arange(len(top))
    mapped = index[x.indices]
    inside = mapped >= 0
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(x.indptr))
    indptr = np.concatenate([np.zeros(1, np.int64), np.cumsum(np.bincount(rows[inside], minlength=n))])
    xt = sparse.csr_matrix((x.data[inside], mapped[inside], indptr), shape=(n, len(top)))
    mu = (np.bincount(mapped[inside], weights=x.data[inside], minlength=len(top)) / max(n, 1)).astype(np.float32)
    return xt, mu


def top_vectors(cov: np.ndarray, k: int) -> np.ndarray:
    """The top k eigenvectors of the symmetric `cov`, largest eigenvalue first, float32 [genes, k]: numpy's eigh for
    a small matrix (or nearly all of them), else ARPACK's eigsh from a fixed start (eigh again if it does not
    converge)."""
    genes = cov.shape[0]
    if genes > SMALL_GENES and k < genes - 1:
        from scipy.sparse.linalg import ArpackNoConvergence, eigsh
        try:
            start = np.random.default_rng(0).standard_normal(genes).astype(cov.dtype)
            _, vectors = eigsh(cov, k=k, which="LA", v0=start)
            return np.ascontiguousarray(vectors[:, ::-1], dtype=np.float32)
        except ArpackNoConvergence:
            pass
    _, vectors = np.linalg.eigh(cov)
    return np.ascontiguousarray(vectors[:, ::-1][:, :k], dtype=np.float32)


def pca(xt, mu: np.ndarray, n_pcs: int) -> np.ndarray:
    """The rows' first n_pcs principal components (float32 [rows, n_pcs]) from their top genes (xt, a CSR, and the
    columns' means mu): the covariance sum over the centred blocks (float32 products), its top eigenvectors, then the
    centred projection xt @ V - mu @ V (the same, in algebra, as projecting the centred blocks). Never more than one
    block and the covariance in memory (5,000 genes: 0.1 GB each)."""
    n, genes = xt.shape
    cov = np.zeros((genes, genes), np.float32)
    for start in range(0, n, PCA_BLOCK_ROWS):
        block = xt[start:start + PCA_BLOCK_ROWS].toarray()
        block -= mu
        cov += block.T @ block
    vectors = top_vectors(cov, n_pcs)
    return (np.asarray(xt @ vectors) - mu @ vectors).astype(np.float32)


def nearest(pcs: np.ndarray, nn: int):
    """(distances float32 [n, nn], indices int64 [n, nn]) of each row's nn nearest rows, exact (scikit-learn's brute
    force, as cuML's on the GPU), the row itself first (at distance 0: the product form leaves it a rounding error,
    which umap-learn would take for the nearest neighbour's distance); its threads are OpenMP's (OMP_NUM_THREADS)."""
    from sklearn.neighbors import NearestNeighbors
    distances, indices = NearestNeighbors(n_neighbors=nn, algorithm="brute").fit(pcs).kneighbors(pcs)
    distances = distances.astype(np.float32)
    distances[indices == np.arange(len(indices))[:, None]] = 0.0
    return distances, indices.astype(np.int64)


# ---- clusters --------------------------------------------------------------------------------------------------------
def leiden_key(p: dict) -> tuple:
    """The settings the clusters are found with: while they stay, Leiden starts from the last clusters."""
    return p["resolution"], p["n_neighbors"], p["min_counts"], p["n_top_genes"], p["n_pcs"]


def leiden_start(run, keep: np.ndarray, p: dict) -> list[int] | None:
    """The run's last clusters as Leiden's start for the kept cells (cells not seen before get ids of their own;
    igraph renumbers them), or None for a fresh start: no clusters yet, a setting changed (leiden_key), the update
    about to happen lays the map out afresh (every refresh_every: the clusters start afresh too), or fewer than half of
    the kept cells have a cluster already."""
    if run.clusters is None or getattr(run, "leiden_key", None) != leiden_key(p) \
            or (run.updates + 1) % int(p["refresh_every"]) == 0:
        return None
    before = run.clusters[keep].astype(np.int64)
    seen = before >= 0
    if not seen.any() or seen.mean() < 0.5:
        return None
    new = ~seen
    before[new] = before.max() + 1 + np.arange(int(new.sum()))
    return before.tolist()


def leiden(n: int, indices: np.ndarray, resolution: float, start: list[int] | None) -> np.ndarray:
    """Clusters (int32 [n]) of the neighbour graph: each cell linked to its neighbours (`indices`, itself included),
    unweighted, as cuGraph gets it; igraph's Leiden (modularity at `resolution`, LEIDEN_ITERATIONS passes) from `start`
    (leiden_start) or afresh, with its random numbers from RNG seeded again: the same graph gives the same clusters."""
    import igraph
    nn = indices.shape[1]
    graph = igraph.Graph(n=n, edges=np.column_stack([np.repeat(np.arange(n), nn), indices.ravel()]), directed=False)
    graph.es["weight"] = 1.0
    graph.simplify(multiple=True, loops=False, combine_edges="max")
    igraph.set_random_number_generator(RNG)
    RNG.seed(0)
    found = graph.community_leiden(objective_function="modularity", weights="weight", resolution=resolution,
                                   n_iterations=LEIDEN_ITERATIONS,
                                   **({} if start is None else {"initial_membership": start}))
    return np.asarray(found.membership, np.int32)


# ---- the layout ------------------------------------------------------------------------------------------------------
def ab_params(min_dist: float) -> tuple[float, float]:
    """umap-learn's curve for `min_dist` (find_ab_params: a curve fit, so once per value)."""
    if min_dist not in _AB:
        from umap.umap_ import find_ab_params
        a, b = find_ab_params(1.0, min_dist)
        _AB[min_dist] = (float(a), float(b))
    return _AB[min_dist]


def layout(pcs: np.ndarray, indices: np.ndarray, distances: np.ndarray, nn: int, p: dict, init, warm: bool,
           note=None) -> np.ndarray:
    """umap-learn's 2-D layout (float32 [n, 2], neither normalised nor rotated: Worker.finish_umap does both) of the
    rows' neighbour graph (their nn nearest, given): from `init` (the last layout's start, umap_start) for warm_epochs
    at warm_learning_rate when `warm`, else from a spectral start for cold_epochs at rate 1, as cuML lays the GPU's out.
    umap-learn rescales any start to its own box first, so the first warm update after a cold layout moves a little
    more. Its warnings go to note(key, text) (the worker's note_once: once per message), else to the worker's log."""
    from umap.umap_ import fuzzy_simplicial_set, simplicial_set_embedding
    a, b = ab_params(float(p["min_dist"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph, _, _ = fuzzy_simplicial_set(pcs, nn, np.random.RandomState(0), "euclidean", knn_indices=indices,
                                           knn_dists=distances)
        embedding, _ = simplicial_set_embedding(
            data=pcs, graph=graph, n_components=2, initial_alpha=float(p["warm_learning_rate"]) if warm else 1.0,
            a=a, b=b, gamma=1.0, negative_sample_rate=5, n_epochs=int(p["warm_epochs"] if warm else p["cold_epochs"]),
            init=W.normalise(init) if warm else "spectral", random_state=np.random.RandomState(0), metric="euclidean",
            metric_kwds={}, densmap=False, densmap_kwds={}, output_dens=False, parallel=True, verbose=False)
    for warning in caught:
        text = f"umap-learn: {warning.category.__name__}: {warning.message}"
        if note is None:
            W.log(text)
        else:
            note(f"umap:{text}", text)
    return np.asarray(embedding, np.float32)


# ---- the pass, Compare's layout, the warm-up -------------------------------------------------------------------------
def umap(worker, run, request, counts, read, ms: dict) -> dict:
    """One pass of clusters and the UMAP on the CPU (Worker.umap on the CPU worker): the recipe above on the frame's
    counts (`read`: its counts file), with the steps timed under the GPU's names (normalising falls in "pca", the graph
    in "leiden"), then Worker.finish_umap's files and summary. "Genes used" is held to MAX_GENES whatever the request
    says (the summary's params give the value used)."""
    import scipy.sparse as sparse
    p = {**W.DEFAULTS, **(request.get("params") or {})}
    if int(p["n_top_genes"]) > MAX_GENES:        # the engine's limit says so too; held here whatever is sent
        worker.note_once("cpu_genes", f"Genes used is at most {MAX_GENES} on the CPU (asked: {p['n_top_genes']})")
        p["n_top_genes"] = MAX_GENES
    watch = W.Stopwatch(worker.sync, ms)
    ncells = int(request["ncells"])
    totals, keep = W.umap_cells(np, counts, p)
    x = normalised(counts, keep, totals)
    top = top_genes(x, int(p["n_top_genes"]))
    n = int(keep.size)
    xt, mu = top_genes_csr(x, top)
    pcs = pca(xt, mu, W.principal_components(p["n_pcs"], n, len(top)))
    del xt
    watch.mark("pca")
    nn = int(min(p["n_neighbors"], n - 1))
    distances, indices = nearest(pcs, nn)
    watch.mark("knn")
    local = leiden(n, indices, float(p["resolution"]), leiden_start(run, keep, p))
    clusters = W.Worker.match_clusters(run, keep, local, ncells)
    run.leiden_key = leiden_key(p)
    watch.mark("leiden")
    found = W.summary_clusters(run, *W.cluster_markers(np, sparse, x, clusters[keep], p))
    watch.mark("markers")
    init, warm, previous = W.umap_start(run, keep, indices, ncells, p)
    embedding = layout(pcs, indices, distances, nn, p, init, warm, worker.note_once)
    return worker.finish_umap(run, request, p, keep, clusters, found, embedding, previous, warm, watch, n, read)


def compare_layout(dense: np.ndarray) -> tuple[np.ndarray, str]:
    """Compare's shared layout on the CPU (Worker.compare_layout): the points (dense [n, genes]: their log-normalised
    top genes) centred, their principal components, nearest neighbours and a cold umap-learn layout with DEFAULTS, as
    umap() lays a run out afresh, normalised to RADIUS: (layout [n, 2], "umap")."""
    p = W.DEFAULTS
    n = int(dense.shape[0])
    if n < max(30, int(p["n_neighbors"]) + 2):
        raise W.TooFewCells(f"only {n} judged cells to lay out")
    dense -= dense.mean(axis=0)
    n_pcs = W.principal_components(p["n_pcs"], n, dense.shape[1])
    pcs = (dense @ top_vectors(dense.T @ dense, n_pcs)).astype(np.float32)
    nn = int(min(p["n_neighbors"], n - 1))
    distances, indices = nearest(pcs, nn)
    return W.normalise(layout(pcs, indices, distances, nn, p, None, False)), "umap"


def compare_own(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """One column's own map on the CPU (Worker.compare_own): compare_layout's recipe on its points alone and its own
    clusters, Leiden afresh on the same neighbour graph with the run's Clusters defaults: (layout, clusters, "umap")."""
    p = W.DEFAULTS
    n = int(dense.shape[0])
    if n < max(30, int(p["n_neighbors"]) + 2):
        raise W.TooFewCells(f"only {n} judged cells to lay out")
    dense -= dense.mean(axis=0)
    n_pcs = W.principal_components(p["n_pcs"], n, dense.shape[1])
    pcs = (dense @ top_vectors(dense.T @ dense, n_pcs)).astype(np.float32)
    nn = int(min(p["n_neighbors"], n - 1))
    distances, indices = nearest(pcs, nn)
    clusters = leiden(n, indices, float(p["resolution"]), None)
    return W.normalise(layout(pcs, indices, distances, nn, p, None, False)), clusters, "umap"


def warm_steps(worker) -> list:
    """The CPU worker's warm-up (Worker.warm_umap_steps), four steps that serve() takes one at a time while the worker
    is idle (the libraries were loaded before the hello: enable): a tiny PCA through eigsh and its neighbours, Leiden
    afresh and from its own clusters (and the clusters' matching: scipy's assignment), umap-learn's layout cold then
    warm with a pass's dtypes (float32 points and distances, int64 neighbours: other dtypes compile again; its parallel
    epochs are compiled at their first call in every process, ~3.5 s, never cached), and the thinning. Fewer than the
    Log keeps of one warm-up (engine/runlog.py WARM_KEPT)."""
    made = {}

    def pca_neighbours():
        dense = np.random.default_rng(0).normal(size=(WARM_CELLS, WARM_GENES)).astype(np.float32)
        dense -= dense.mean(axis=0)
        made["pcs"] = (dense @ top_vectors(dense.T @ dense, int(W.DEFAULTS["n_pcs"]))).astype(np.float32)
        made["knn"] = nearest(made["pcs"], int(W.DEFAULTS["n_neighbors"]))

    def clusters():
        _, indices = made["knn"]
        first = leiden(len(indices), indices, float(W.DEFAULTS["resolution"]), None)
        again = leiden(len(indices), indices, float(W.DEFAULTS["resolution"]), first.tolist())
        run = SimpleNamespace(clusters=None, next_cluster=0)
        W.Worker.match_clusters(run, np.arange(len(indices)), first, len(indices))
        W.Worker.match_clusters(run, np.arange(len(indices)), again, len(indices))
    clusters.__name__ = "leiden"                  # the worker log's name for the step, as the GPU's

    def layouts():
        distances, indices = made.pop("knn")
        pcs = made.pop("pcs")
        nn = indices.shape[1]
        first = layout(pcs, indices, distances, nn, {**W.DEFAULTS, "cold_epochs": 30}, None, False, worker.note_once)
        layout(pcs, indices, distances, nn, W.DEFAULTS, W.normalise(first), True, worker.note_once)

    def thinning():
        from . import cpu_kernels
        cpu_kernels.thin(np.ones(16, np.float32), np.full(16, 0.5, np.float32), 1)

    return [pca_neighbours, clusters, layouts, thinning]
