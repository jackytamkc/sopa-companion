"""Mixing check, learning step: fit ResolVI (scviva) once per sample and export its per-cell mixing calculator.

Runs in the sopa environment ([qc] python in config.toml; used read-only, nothing is ever installed), started by
engine/qc.py with PYTHONPATH set to the app's own tree:

    CUDA_VISIBLE_DEVICES=0 <sopa python> -m sopa_companion.analysis.qc_trainer

The job is one JSON object on the first line of stdin. stdin then stays open: its end (the engine stopped, or wants
the fit cancelled) ends the process. One fit per launch: ResolVI clears pyro's global parameter store and a CUDA
device assert poisons the process, so nothing is ever reused between fits. Replies are JSON lines on stdout (anything
the libraries print goes to stderr, which the engine keeps in logs/qc-trainer.log):

    {"ev": "hello", "pid", "python", "versions", "cuda_visible_devices", "nice"}            at once
    {"ev": "progress", "stage", "step", "of", ...}     stages: lock, reading, preparing, learning (once per pass over
                                                       the cells: epoch, loss, seconds), checking, exporting, scoring
    {"ev": "done", "model": {...}}                     out_dir holds the complete export
    {"ev": "error", "reason", "message"}               reason: gpu | no_table | not_counts | positions | too_few_cells
                                                       | unreadable | oom | not_settled | timeout | unexpected
                                                       | detail_mismatch

Guards: GPU 0 only (a fit refuses to start unless CUDA_VISIBLE_DEVICES is exactly "0" in PCI bus order,
CUDA_DEVICE_ORDER=PCI_BUS_ID, and the engine also checks the device's PCI bus), or with the job's "device" "mps" on a
Mac Apple's GPU, after it passed zarr_bridge/devices.py MPS_LEARNING_TEST in a process of its own (the counts dense
there: MPS holds no sparse tensor); nice 19, 4 CPU threads, no DataLoader workers, the job's deadline. The sample's cell table is read read-only (zarr mode "r") and never while a Save to the
sample writes it (read_consistently: the sample's save locks, zarr_bridge/helper.py, held shared: exactly the ones a
save takes, both shared folders with the previous version's first, or the private folder SOPA_COMPANION_ZARR_LOCK_DIR
names alone, so a test's or the harness's engine never opens the user's lock files; they are never created, so
reading leaves no trace). Tests only: a job with "cpu_for_tests" learns on the CPU, and only where no GPU can be seen
at all (CUDA_VISIBLE_DEVICES set and empty; refused otherwise, before anything is read); the engine never sends it and
always sets CUDA_VISIBLE_DEVICES=0, so a test job can never reach GPU 0 or the GPU worker's GPU.

The recipe is a fixed reference recipe: scviva-tools ResolVI, unsupervised, n_latent
10, batch 512, lr 3e-3, seed 0; cells with >= 20 integer counts; an exact 10-nearest-neighbour graph on the cell
centroids with squared distances in the transcripts' frame (µm for Xenium; prepare_data=False); background_ratio from
the n_genes // 50 lowest-expressed genes (at least one: a panel of fewer than 50 genes has none otherwise, and both the
app's priors and ResolVI's own then divide by zero; nothing changes at 50 genes or more);
median_distance = the median squared distance to the 6th neighbour; 50 passes over the cells, more for small
tables so that every model gets at least `min_steps` optimisation steps. With recipe priors "pooled" (the default)
the genes kept and both priors are pooled over the learned table and the sample's other cell table (the earlier proseg
cells), exactly as the reference recipe pools them: in tests, priors from the learned table alone changed the model as
much as another seed would, while pooled ones reproduced the reference model exactly.

The cell centres (choose_centres): a run's cells, regions and distances are in the transcripts'
frame (the points' intrinsic coordinates, as zarr_bridge/helper.py hands everything out), so a table is learned with
its cells' centres in that frame: its own centroids (obsm['spatial'], the reference recipe's) when they agree with its
cells' outlines mapped into that frame, else those outlines' centres (CosMx keeps raw slide pixels in obsm['spatial']);
a table whose cells cannot be placed on the transcripts is refused (reason "positions"). Two plain refusals after
learning, before anything is exported: a loss that did not settle (loss_settled, its windows scaled with the passes run)
and a calculator that sees no mixing anywhere (collapse_check: FLOOR_AT of the learning cells at the lowest share it can
give, which small tables reach after their many passes).

Export (out_dir, see ENCODER_KEYS):
    encoder.npz    the per-cell calculator: log1p(counts) -> 128 -> 128 (layer norm, ReLU) -> 3-way softmax,
                   + var_eps on every part and + diffusion_eps on the neighbour part; gene names; 256 reference
                   cells for self-tests (selftest_indptr / selftest_indices / selftest_data: their counts as CSR;
                   selftest_expected: the weights torch gave them; the names the GPU worker's loader reads)
    model.json     recipe, package versions, table fingerprint, priors, loss curve, cells, seconds, self-test
    cells.parquet  the learning table's cells: cell_id, x, y (centre in the transcripts' frame: µm for Xenium), area
                   (µm², NaN if unknown), counts, share
    batch.parquet  the sample's other cell table, scored once, same columns: only a table of proseg's own count (a
                   proseg result saved into the sample, `proseg_table`: the earlier proseg cells); another table (e.g.
                   Xenium Ranger's outline re-count, a Xenium sample's "table") is pooled into the priors, never
                   scored, and no batch.parquet is written
    deep.npz       Go deeper (DEEP_FORMAT, exactly DEEP_KEYS), written after
                   encoder.npz: the z encoder and the decoder (strict name maps Z_LAYERS / D_LAYERS after the structural
                   checks of deep_structure), the loose-RNA profile, the typical profile, the guide's distance scale, the
                   learned neighbour values of every learning-table cell (constrained, in training-row order = the order
                   of cells.parquet) with its training neighbours, the cell ids' sha256, and a 16-cell self-test against
                   torch with the noise off (numpy float64 within DEEP_SELFTEST_TOLERANCE, float32 within
                   DEEP_WORKER_TOLERANCE). If that export fails, the model keeps the calculator alone ("deep": null and
                   "deep_error" in model.json): the check still scores, only Go deeper needs the details.
    train.log      what was reported, in words

model.json also carries "deep" {format, file, bytes, selftest {cells, max_abs_diff, tolerance, ...}, from: "fit" |
"detail", created} and "provenance" (see provenance(): the table learned from, the other table pooled into
the priors, the priors' source, the passes run against the reference recipe's 50, the cells, and whether the model is
the one the reference recipe learns, with a plain key for each difference).

A detail job (the job's "detail_for": an installed model folder) learns the same recipe again for a model
learned before deep.npz existed. It refuses with reason detail_mismatch, changing nothing, unless the installed model is
of the job's table, recipe and cells and the new calculator equals the installed one (every array identical, or the
installed reference cells' outputs within DETAIL_TOLERANCE); then it writes only deep.npz and model.json (the installed
one plus "deep" from "detail", "detail" with the parity record, and "provenance") into out_dir. The engine moves
deep.npz into the installed folder and replaces its model.json; encoder.npz, the parquet files and the key stay.

The numpy helpers (load_encoder, mixing_weights, neighbour_share, self_test, quantise, knn_graph, dataset_priors,
loss_settled, epochs_for, floor_share, at_floor, collapse_check, choose_centres, and for Go deeper load_deep,
deep_forward, deep_self_test, detail_parity, provenance) need only
numpy: the engine (.venv) checks every export with load_encoder and self_test (deep.npz: load_deep and deep_self_test)
before it installs it, and the tests use them. The GPU worker (analysis/worker.py, RAPIDS) has its own cupy-ready copy of
the calculator and loader (load_encoder, mixing_weights, neighbour_share, share_codes), which reads only exports the
engine installed; the tests hold the two together (the trainer's exports read and self-tested by the worker, the same
byte codes).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

# A cell table of proseg's own count (a proseg result saved into the sample) by the sample code's own rule: its uns
# holds proseg's own run record ("proseg_run", which proseg-output.zarr carries) or this app's save record
# (names.PROVENANCE_KEYS). The helper's module top needs only the standard library and numpy: it imports in every
# environment this runs in.
from ..zarr_bridge.helper import PROSEG_TABLE_KEYS

# Format ids keep the app's earlier name: learned models are checked by exact match (here and in analysis/worker.py),
# so a new name would make every model learned so far, and any shared model cache, unusable.
FORMAT = "proseg-live-mixing-encoder/1"
RECIPE_VERSION = 2                 # 2: priors and kept genes pooled over the sample's two cell tables (the reference)
PRIORS = ("pooled", "table")       # recipe "priors": pooled over the learned and the other table, or the learned alone
SELFTEST_CELLS = 256
SELFTEST_TOLERANCE = 1e-5          # max |numpy - torch| over the reference cells' raw weights
THREADS = 4
QUANT_SCALE = 250                  # u8 per cell: floor(share x 250), as the GPU worker codes it; 255 = not judged
NOT_JUDGED = 255
MIN_CELLS = 200                    # cells with >= fit_min counts and a centre a table needs (the engine checks n_obs
                                   # against it before it waits for GPU 0)
SETTLE_WINDOW = 5                  # loss_settled compares the mean of the last w passes with the w before them: w is
SETTLE_PARTS = 20                  # max(SETTLE_WINDOW, passes // SETTLE_PARTS), so 5 for the reference 50 passes
FLOOR_TOLERANCE = 1e-4             # a cell "at the floor": its share within this of the lowest the calculator gives
FLOOR_AT = 0.80                    # collapse_check: a fit with this share of its learning cells at the floor is refused
CENTRES_AGREE = 0.5                # choose_centres: obsm['spatial'] agrees with the outlines' centres when their median
                                   # distance is at most this x the outlines' median nearest-neighbour spacing
CENTRES_INSIDE = 0.95              # ... and lies on the transcripts when this share of the centres falls within their
CENTRES_MARGIN = 0.05              # extent, widened by this x its size on each side
CENTRES_TOUCH = 0.01               # centres the outlines placed (the sample's transformations put them in the frame)
                                   # need only touch the transcripts: a sample's table may hold cells beyond the points
                                   # element in use (one of two halves, one of several fields of view)
PACKAGES = ("scviva-tools", "scvi-tools", "torch", "pyro-ppl", "lightning", "anndata", "numpy")
ENCODER_LAYERS = {                 # diffusion_encoder.state_dict() names -> export names (anything else is refused)
    "encoder.fc_layers.Layer 0.0.weight": "W0", "encoder.fc_layers.Layer 0.0.bias": "b0",
    "encoder.fc_layers.Layer 1.0.weight": "W1", "encoder.fc_layers.Layer 1.0.bias": "b1",
    "var_encoder.weight": "Wv", "var_encoder.bias": "bv",
}
ENCODER_UNUSED = {"mean_encoder.weight", "mean_encoder.bias"}
ENCODER_KEYS = {"format", "genes", "gene_ids", "W0", "b0", "W1", "b1", "Wv", "bv", "ln_eps", "var_eps", "diffusion_eps",
                "selftest_indptr", "selftest_indices", "selftest_data", "selftest_expected"}

# Go deeper: deep.npz
DEEP_FORMAT = "proseg-live-mixing-deep/1"
DEEP_SELFTEST_TOLERANCE = 1e-5     # numpy float64 vs torch with the noise off, every self-test output (trainer, engine)
DEEP_TOLERANCE = DEEP_SELFTEST_TOLERANCE   # the name the engine uses
DEEP_WORKER_TOLERANCE = 1e-4       # the GPU worker's float32 numpy at load
DETAIL_TOLERANCE = 1e-6            # a detail job's calculator vs the installed one: the reference cells' outputs
DETAIL_ENCODER = ".detail-encoder.npz"   # a detail job's new calculator, compared with the installed one (here and
                                         # again by the engine before it installs deep.npz), removed with staging
DEEP_SELFTEST = (("mixed", 8), ("some", 4), ("little", 4))   # of encoder.npz's reference cells, spread over depth
DEEP_SELFTEST_PROFILES = 4         # decoded own profiles kept (2 mixed, 1 some, 1 little)
DEEP_SELFTEST_GENES = 20           # per cell: the top genes by torch's observed neighbour split (before scaling)
DEEP_CLAMP = 10.0                  # the guide's clamp on sqrt(d2 / median_distance)
DEEP_SOME_AT = 0.10
Z_LAYERS = {                       # model.module.z_encoder.state_dict() -> export names (anything else is refused)
    "encoder.fc_layers.Layer 0.0.weight": "Z0", "encoder.fc_layers.Layer 0.0.bias": "zb0",
    "encoder.fc_layers.Layer 1.0.weight": "Z1", "encoder.fc_layers.Layer 1.0.bias": "zb1",
    "mean_encoder.weight": "Zm", "mean_encoder.bias": "zbm", "var_encoder.weight": "Zv", "var_encoder.bias": "zbv"}
D_LAYERS = {                       # model.module.model.decoder.state_dict() -> export names
    "px_decoder.fc_layers.Layer 0.0.weight": "D0", "px_decoder.fc_layers.Layer 0.0.bias": "db0",
    "px_decoder.fc_layers.Layer 1.0.weight": "D1", "px_decoder.fc_layers.Layer 1.0.bias": "db1",
    "px_scale_decoder.0.weight": "Ds", "px_scale_decoder.0.bias": "dbs"}
D_UNUSED = {"px_r_decoder.weight", "px_r_decoder.bias", "px_dropout_decoder.weight", "px_dropout_decoder.bias"}
DEEP_SELFTEST_KEYS = ("st_rows", "st_indptr", "st_indices", "st_data", "st_d2", "st_z", "st_pi", "st_own_rows", "st_own",
                      "st_genes", "st_nb_split")
DEEP_KEYS = {"format", "genes", *Z_LAYERS.values(), *D_LAYERS.values(), "ln_eps", "z_var_eps", "background", "typical",
             "median_distance", "clamp", "diffuse_scale", "train_logit", "train_index", "rank_logit", "train_ids_sha256",
             *DEEP_SELFTEST_KEYS}

TABLE_KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")

# Provenance: what the reference recipe learns, so model.json can say whether this model is that model.
NOTEBOOK_TABLE = "xenium_table"    # the reference recipe learns ResolVI from the sample's Xenium cells
NOTEBOOK_EPOCHS = 50
NOTEBOOK_SETTINGS = {"method": "scviva.model.ResolVI", "semisupervised": False, "n_latent": 10, "n_hidden_encoder": 128,
                     "batch_size": 512, "lr": 0.003, "seed": 0, "n_neighbors": 10, "fit_min_counts": 20,
                     "distance_rank": 6, "low_genes_per": 50}
PROVENANCE_DIFFERENCES = {         # provenance()["differences"]: plain keys (the page words them), in this order
    "other_table_used": "learned from a cell table other than the Xenium cells the notebook learns from",
    "single_table_priors": "the two dataset priors come from the learned table alone, not pooled with the sample's "
                           "other cell table as the notebook does",
    "more_epochs_small_table": "more passes over the cells than the notebook's 50, because the table is small",
    "other_epochs": "a number of passes other than the notebook's 50 was set",
    "other_settings": "other learning settings than the notebook's",
    "not_recorded": "the model's file does not record enough to tell",
}


class TrainerError(Exception):
    """A failure with a reason the engine turns into plain words."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# ---- the recipe ---------------------------------------------------------------------------------------------------
def recipe(epochs: int = 50, min_steps: int = 3000, fit_min_counts: int = 20, priors: str = "pooled") -> dict:
    """The learning recipe (part of every model's cache key: a change means relearning, never mixing yardsticks)."""
    if priors not in PRIORS:
        raise ValueError(f"priors must be one of {', '.join(PRIORS)}")
    return {"version": RECIPE_VERSION, "method": "scviva.model.ResolVI", "semisupervised": False, "n_latent": 10,
            "n_hidden_encoder": 128, "epochs": int(epochs), "min_steps": int(min_steps), "batch_size": 512,
            "lr": 0.003, "seed": 0, "n_neighbors": 10, "fit_min_counts": int(fit_min_counts), "distance_rank": 6,
            "low_genes_per": 50, "priors": priors}


def epochs_for(n_cells: int, rec: dict) -> int:
    """Passes over the cells: the recipe's epochs, more for small tables so the model gets >= min_steps steps (the
    score's scale drifts with training length until about 2-3 k steps)."""
    batches = max(1, math.ceil(int(n_cells) / int(rec["batch_size"])))
    return max(int(rec["epochs"]), math.ceil(int(rec["min_steps"]) / batches))


def knn_graph(xy: np.ndarray, k: int = 10, block: int = 2048):
    """The reference recipe's exact within-table graph: each cell's k nearest other cells by centroid (self excluded
    even when another cell shares its centroid), as (indices int64 [n, k], squared distances float32 [n, k]), nearest
    first. scipy's cKDTree when available (the sopa environment), else a blockwise brute force with the same result."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n <= k:
        raise TrainerError("too_few_cells", f"only {n} cells: at least {k + 1} are needed for the neighbour graph")
    try:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(xy).query(xy, k=k + 1)
    except ImportError:
        indices = np.empty((n, k + 1), np.int64)
        distances = np.empty((n, k + 1), np.float64)
        for start in range(0, n, block):
            part = xy[start:start + block]
            d2 = ((part[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argpartition(d2, k, axis=1)[:, :k + 1]
            rows = np.arange(len(part))[:, None]
            order = np.lexsort((nearest, d2[rows, nearest]), axis=1)
            indices[start:start + block] = nearest[rows, order]
            distances[start:start + block] = np.sqrt(d2[rows, indices[start:start + block]])
    order = np.argsort(indices == np.arange(n)[:, None], axis=1, kind="stable")[:, :k]
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int64)
    distances = np.take_along_axis(distances, order, axis=1)
    return indices, np.square(distances).astype(np.float32)


def low_genes(n_genes: int, per: int = 50) -> int:
    """How many of the lowest-expressed genes the background prior averages: n_genes // per, as the reference recipe
    (and ResolVI itself) takes them, but at least one: below 50 genes that is none, and the mean over no genes divides
    by zero. Nothing changes at 50 genes or more."""
    return max(1, int(n_genes) // int(per))


def pooled_priors(parts, totals: np.ndarray, rec: dict) -> dict:
    """ResolVI's two dataset priors shared by one or more cell tables, exactly as the reference recipe computes them
    (the same operations in the same order, so the same float32 results):
    background_ratio = the mean over tables of each table's mean, over its cells, of the mean count of the
    n_genes // 50 lowest-expressed genes (low_genes: at least one; by the pooled `totals`) over the cell's mean count;
    median_distance = the median, over all tables' cells, of the squared distance to the 6th neighbour within its own
    table. parts: [(counts: cells x genes, scipy sparse or dense, the genes with pooled counts only, in one shared
    order; squared neighbour distances [cells, k])]; totals: the pooled count of each of those genes."""
    low = np.argsort(totals)[: low_genes(len(totals), rec["low_genes_per"])]
    ratio = np.mean([np.mean(np.asarray(counts[:, low].mean(axis=1)).ravel() / np.asarray(counts.mean(axis=1)).ravel())
                     for counts, _ in parts])
    rank = int(rec["distance_rank"]) - 1
    median = np.median(np.concatenate([np.partition(distances2, rank, axis=1)[:, rank] for _, distances2 in parts]))
    return {"background_ratio": float(ratio), "median_distance": float(median)}


def dataset_priors(counts, distances2: np.ndarray, rec: dict) -> dict:
    """pooled_priors of one table alone. `counts`: cells x genes (scipy sparse or dense), genes with no counts
    already removed."""
    return pooled_priors([(counts, distances2)], np.asarray(counts.sum(axis=0)).ravel(), rec)


def gene_order(genes, gene_ids, other_genes, other_ids) -> np.ndarray | None:
    """Where each of `genes` sits in the other table (by name, else by gene id), or None when the two tables do not
    hold the same genes (the reference recipe refuses to pool different panels)."""
    for mine, theirs in ((genes, other_genes), (gene_ids, other_ids)):
        mine, theirs = [str(g) for g in mine], [str(g) for g in theirs]
        if len(mine) != len(theirs) or len(set(mine)) != len(mine) or set(mine) != set(theirs) or "" in set(mine):
            continue
        position = {g: i for i, g in enumerate(theirs)}
        return np.fromiter((position[g] for g in mine), np.int64, len(mine))
    return None


def priors_partner(other, genes, gene_ids, rec: dict):
    """The other cell table as the reference recipe pools it with the learned one: (its counts in the learned table's
    gene order, cells with >= fit_min counts and a centroid; its own neighbour distances, squared) and a note
    {table, cells} -- or (None, {table, why}) when it cannot be pooled (no table, other genes, too few cells).
    `other`: (key, counts, obs, genes, gene_ids, centroids) as read_table gives it, or None."""
    if other is None:
        return None, {"table": None, "why": "the sample has no other cell table"}
    key, counts, obs, other_genes, other_ids, xy = other
    order = gene_order(genes, gene_ids, other_genes, other_ids)
    if order is None:
        return None, {"table": key, "why": "its genes differ from the learned table's"}
    if not np.array_equal(order, np.arange(len(order))):
        counts = counts[:, order]                       # the learned table's gene order (the reference)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    keep = (totals >= int(rec["fit_min_counts"])) & np.isfinite(xy).all(axis=1)
    if int(keep.sum()) <= int(rec["n_neighbors"]):
        return None, {"table": key, "why": f"only {int(keep.sum())} of its cells have >= {rec['fit_min_counts']} counts"}
    _, distances2 = knn_graph(xy[keep], int(rec["n_neighbors"]))
    return (counts[keep].tocsr() if hasattr(counts, "tocsr") else counts[keep], distances2), \
        {"table": key, "cells": int(keep.sum())}


def _spacing(xy: np.ndarray) -> float:
    """The median distance from each centre to the nearest other one (0 with fewer than two)."""
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return 0.0
    _, distances2 = knn_graph(xy, 1)
    return float(np.sqrt(np.median(distances2[:, 0])))


def _inside(xy: np.ndarray, extent) -> float:
    """The share of the finite centres within `extent` [x0, y0, x1, y1], widened by CENTRES_MARGIN of its size on
    each side (a cell's centre may lie a little past the outermost transcript)."""
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in extent)
    mx, my = CENTRES_MARGIN * max(x1 - x0, 0.0), CENTRES_MARGIN * max(y1 - y0, 0.0)
    inside = (xy[:, 0] >= x0 - mx) & (xy[:, 0] <= x1 + mx) & (xy[:, 1] >= y0 - my) & (xy[:, 1] <= y1 + my)
    return float(inside.mean())


def _span(xy: np.ndarray) -> str:
    """Where the finite centres lie, in words for a message: "x 0-246, y 1-246"."""
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return "nowhere"
    return (f"x {xy[:, 0].min():.6g}-{xy[:, 0].max():.6g}, y {xy[:, 1].min():.6g}-{xy[:, 1].max():.6g}")


def choose_centres(spatial, outlines, extent, key: str, region: str | None = None) -> tuple[np.ndarray, dict]:
    """The cell centres a table is learned with, in the transcripts' frame (what a run's cells, regions and distances
    are in: the kNN graph, the priors' median distance, cells.parquet and batch.parquet use them), and how they were
    found (model.json "centres": {table, from, outlines, checked, offset, spacing}). `spatial`: the table's
    obsm['spatial'] (None without one); `outlines`: its cells' outline centres in that frame, one per row
    (table_centres), or None; `extent`: the transcripts' [x0, y0, x1, y1] in that frame, or None when the sample has
    no transcripts to compare with (the tests' bare tables). In order:
    - the table's own centroids ("table": the reference recipe's) when they agree with the outline centres: their median
      distance at most CENTRES_AGREE x the outlines' median nearest-neighbour spacing (a sample's vendor centroids sit
      within a fraction of a µm of their outlines' centres, the cells about 10 µm apart);
    - else the outline centres ("outlines": CosMx keeps raw slide pixels in obsm['spatial'], so the squared distances
      were many times their µm values and Go deeper weighed every neighbour almost alike);
    - without outlines, the table's own centroids.
    The table's own centroids with no outlines to check them against must lie on the transcripts (CENTRES_INSIDE of
    them within the widened extent, when one is known); centres the outlines placed, or that agree with them, need only
    touch them (CENTRES_TOUCH): the transformations put them in the frame, and the cells may reach past the points
    element in use (a table of both halves of a sample whose transcripts are two elements was refused). Else
    TrainerError("positions") in plain words, as for a table without centroids or outlines."""
    spatial = None if spatial is None else np.asarray(spatial, dtype=np.float64)[:, :2]
    found = {"table": key, "from": None, "outlines": region, "checked": extent is not None}
    if outlines is not None:
        outlines = np.asarray(outlines, dtype=np.float64)[:, :2]
        chosen, found["from"] = outlines, "outlines"
        if spatial is not None and len(spatial) == len(outlines):
            both = np.isfinite(spatial).all(axis=1) & np.isfinite(outlines).all(axis=1)
            if both.sum() >= 2:
                offset = float(np.median(np.hypot(*(spatial[both] - outlines[both]).T)))
                spacing = _spacing(outlines)
                found.update(offset=round(offset, 4), spacing=round(spacing, 4))
                if offset <= CENTRES_AGREE * spacing:
                    chosen, found["from"] = spatial, "table"
    elif spatial is not None:
        chosen, found["from"] = spatial, "table"
    else:
        raise TrainerError("positions", f"table '{key}' has no cell centroids (obsm['spatial']) and its cells' outlines "
                                        "could not be placed on the transcripts")
    if extent is not None and _inside(chosen, extent) < (CENTRES_INSIDE if outlines is None else CENTRES_TOUCH):
        corners = np.asarray([extent[:2], extent[2:]], dtype=np.float64)
        raise TrainerError("positions", f"table '{key}' places its cells at {_span(chosen)}, away from the "
                                        f"transcripts ({_span(corners)}): its cell centres are not in the "
                                        "transcripts' units")
    return chosen, found


def loss_settled(curve) -> tuple[bool, str]:
    """A fit is kept only if every epoch's loss is finite, the mean of the last w passes is not more than 1 % worse
    than the mean of the w before them, and the loss fell overall. w = max(SETTLE_WINDOW, passes // SETTLE_PARTS): the
    windows grow with the passes run. A table of 512 cells or fewer is one minibatch per pass, so each pass's loss is
    one noisy estimate, about 3 % apart from pass to pass on a settled curve, and its 3,000 passes with 5-pass windows
    failed about one fit in four by chance; over 150 passes that noise averages out. The reference recipe's
    50 passes keep w = 5 (many batches a pass: a settled curve's noise stays within ~0.1 %)."""
    values = np.asarray(curve, dtype=np.float64)
    if not len(values):
        return False, "no loss was recorded"
    if not np.isfinite(values).all():
        return False, "the loss became infinite or undefined"
    w = max(SETTLE_WINDOW, len(values) // SETTLE_PARTS)
    if len(values) >= 2 * w:
        last, before = values[-w:].mean(), values[-2 * w:-w].mean()
        if last - before > 0.01 * abs(before):
            return False, f"the loss rose over the last {2 * w} passes ({before:.4g} -> {last:.4g})"
    if len(values) >= 2 and values[-1] >= values[0]:
        return False, f"the loss did not fall ({values[0]:.4g} -> {values[-1]:.4g})"
    return True, ""


def floor_share(var_eps: float, diffusion_eps: float) -> float:
    """The lowest neighbour share the calculator can give: its softmax at exactly (1, 0, 0), with var_eps on every
    part and diffusion_eps on the neighbour part (mixing_weights): about 0.0109 for ResolVI's offsets."""
    return (float(var_eps) + float(diffusion_eps)) / (1.0 + 3.0 * float(var_eps) + float(diffusion_eps))


def at_floor(share, var_eps: float, diffusion_eps: float) -> float:
    """The fraction of cells whose neighbour share sits at the calculator's floor (within FLOOR_TOLERANCE of
    floor_share): cells it says took nothing from their neighbours, to the last digit."""
    share = np.asarray(share, dtype=np.float64)
    if not share.size:
        return 0.0
    return float(np.mean(share <= floor_share(var_eps, diffusion_eps) + FLOOR_TOLERANCE))


def collapse_check(share, var_eps: float, diffusion_eps: float) -> float:
    """Refuse a calculator that sees no mixing anywhere: when FLOOR_AT or more of the learning cells
    sit at its floor, TrainerError("too_few_cells"), instead of installing a model whose check says "no mixing" on any
    data. Small tables get this: a table of 512 cells or fewer is one minibatch per pass and is given min_steps
    passes over the same cells (epochs_for), after which almost every cell sits at the floor (the toy Xenium table:
    0 % after 300 passes, 27 % after 1,000, 92 % after 3,000). The recipe's training length stays as it is (changing it
    is a decision, not this fix); this only says so plainly. Returns the fraction at the floor otherwise (model.json
    keeps it)."""
    fraction = at_floor(share, var_eps, diffusion_eps)
    if fraction >= FLOOR_AT:
        raise TrainerError("too_few_cells", f"the learned model sees no mixing in {100 * fraction:.0f} % of the cells "
                                            "it learned from (they sit at the lowest share it can give): this sample "
                                            "has too few cells for the check to learn from")
    return fraction


# ---- the calculator (numpy; cupy in the GPU worker) ---------------------------------------------------------------
def load_encoder(path) -> dict:
    """An exported calculator, checked: exactly ENCODER_KEYS, the format, consistent shapes, finite float32 weights,
    unique gene names, a well-formed reference set. Raises ValueError otherwise (e.g. an export from a ResolVI with
    other layers)."""
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if names != ENCODER_KEYS:
            raise ValueError(f"unexpected calculator arrays: missing {sorted(ENCODER_KEYS - names)}, "
                             f"unknown {sorted(names - ENCODER_KEYS)}")
        enc = {k: data[k] for k in data.files}
    if str(enc["format"]) != FORMAT:
        raise ValueError(f"unknown calculator format {str(enc['format'])!r} (this app reads {FORMAT})")
    genes = [str(g) for g in enc["genes"].tolist()]
    if not genes or len(set(genes)) != len(genes) or any(not g for g in genes):
        raise ValueError("the calculator's gene names must be unique and non-empty")
    n_genes, hidden0, hidden1 = len(genes), enc["W0"].shape[0], enc["W1"].shape[0]
    shapes = {"W0": (hidden0, n_genes), "b0": (hidden0,), "W1": (hidden1, hidden0), "b1": (hidden1,),
              "Wv": (3, hidden1), "bv": (3,), "ln_eps": (), "var_eps": (), "diffusion_eps": ()}
    for key, shape in shapes.items():
        array = enc[key]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"calculator array {key}: {array.dtype}{list(array.shape)}, expected float32{list(shape)}")
    if len(enc["gene_ids"]) != n_genes:
        raise ValueError("gene_ids must match genes")
    if not (0 < float(enc["ln_eps"]) < 1 and 0 <= float(enc["var_eps"]) < 1 and 0 <= float(enc["diffusion_eps"]) < 1):
        raise ValueError("calculator offsets out of range")
    indptr, indices, counts, weights = (enc[k] for k in ("selftest_indptr", "selftest_indices", "selftest_data",
                                                         "selftest_expected"))
    cells = len(indptr) - 1
    if (indptr.dtype != np.int64 or indices.dtype != np.int32 or counts.dtype != np.float32
            or weights.dtype != np.float32 or cells < 1 or weights.shape != (cells, 3) or indptr[0] != 0
            or indptr[-1] != len(indices) or len(indices) != len(counts) or (np.diff(indptr) < 0).any()
            or (len(indices) and (indices.min() < 0 or indices.max() >= n_genes))
            or not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(weights).all()):
        raise ValueError("malformed reference cells")
    enc["genes"] = genes
    enc["gene_ids"] = [str(g) for g in enc["gene_ids"].tolist()]
    return enc


def _norm_relu(h, xp, eps):
    mean = h.mean(axis=1, keepdims=True)
    var = ((h - mean) ** 2).mean(axis=1, keepdims=True)
    return xp.maximum((h - mean) / xp.sqrt(var + eps), 0)


def mixing_weights(enc: dict, counts, xp=np, dtype=np.float32):
    """Raw (own, neighbour, background) weights of each row of `counts`: cells x the calculator's genes, raw counts,
    dense or CSR (numpy/scipy, or cupy/cupyx with xp=cupy). ResolVI's guide: log1p -> Linear -> LayerNorm (no affine)
    -> ReLU, twice -> Linear -> softmax, + var_eps on every part, + diffusion_eps on the neighbour part."""
    def param(key):
        return xp.asarray(enc[key], dtype=dtype)
    eps = float(enc["ln_eps"])
    if hasattr(counts, "indptr"):                     # CSR: log1p of the stored values only (log1p(0) = 0)
        x = counts.astype(dtype)                      # a new matrix (scipy and cupyx), so the caller's is untouched
        x.data = xp.log1p(x.data)
        h = xp.asarray(x @ param("W0").T)
    else:
        h = xp.log1p(xp.asarray(counts, dtype=dtype)) @ param("W0").T
    h = _norm_relu(h + param("b0"), xp, eps)
    h = _norm_relu(h @ param("W1").T + param("b1"), xp, eps)
    z = h @ param("Wv").T + param("bv")
    z = xp.exp(z - z.max(axis=1, keepdims=True))
    w = z / z.sum(axis=1, keepdims=True) + float(enc["var_eps"])
    w[:, 1] += float(enc["diffusion_eps"])
    return w


def neighbour_share(weights, xp=np):
    """The per-cell number: the neighbour part over the sum of the three parts (ResolVI's neighbour fraction)."""
    return weights[:, 1] / weights.sum(axis=1)


def quantise(share, judged, xp=np):
    """One byte per cell, as the GPU worker codes it (analysis/worker.py share_codes): floor(share x 250) for judged
    cells, so a code >= 75 means exactly a share >= 0.30; 255 = not judged."""
    value = xp.clip(xp.floor(xp.nan_to_num(xp.asarray(share)) * QUANT_SCALE), 0, QUANT_SCALE)
    return xp.where(xp.asarray(judged), value, NOT_JUDGED).astype(xp.uint8)


def selftest_counts(enc: dict):
    """The reference cells' counts as a dense float32 array [cells, genes]."""
    indptr, indices, values = enc["selftest_indptr"], enc["selftest_indices"], enc["selftest_data"]
    dense = np.zeros((len(indptr) - 1, len(enc["genes"])), np.float32)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    dense[rows, indices] = values
    return dense


def self_test(enc: dict) -> float:
    """Max |numpy - torch| over the reference cells' raw weights (numpy in float64)."""
    weights = mixing_weights(enc, selftest_counts(enc).astype(np.float64), np, np.float64)
    return float(np.abs(weights - enc["selftest_expected"].astype(np.float64)).max())


# ---- Go deeper: deep.npz and one cell explained, numpy -----------------------------------------------------------------
def ids_sha256(ids) -> str:
    """sha256 of the learning table's cell ids (cells.parquet order) joined by newlines: the worker checks the ids it reads
    from cells.parquet against deep.npz's train_ids_sha256 before it maps a run's cells to training rows."""
    import hashlib
    return hashlib.sha256("\n".join(str(i) for i in ids).encode()).hexdigest()


def load_deep(path, genes=None, enc: dict | None = None) -> dict:
    """An exported deep.npz, checked like load_encoder: exactly DEEP_KEYS, the format, consistent float32 shapes, finite
    values, the calculator's gene order (`genes`: its gene names, or the calculator itself as load_encoder gives it),
    profiles summing to 1, learned neighbour values within (-10, 10), neighbour rows and self-test indices in range,
    rank_logit the float64 column mean of train_logit, a well-formed self-test block. Raises ValueError otherwise (an
    export with other layers, a covariate column, or another gene order than encoder.npz's included).

    The calculator the file belongs to is attached as deep["enc"] (deep_self_test needs it for the gene split): `enc`,
    or `genes` when that is the calculator, else the encoder.npz beside `path` when there is one (loaded strictly)."""
    if isinstance(genes, dict):
        enc, genes = genes, None
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if names != DEEP_KEYS:
            raise ValueError(f"unexpected deep arrays: missing {sorted(DEEP_KEYS - names)}, "
                             f"unknown {sorted(names - DEEP_KEYS)}")
        deep = {k: data[k] for k in data.files}
    if deep["format"].shape != () or str(deep["format"]) != DEEP_FORMAT:
        raise ValueError(f"unknown deep format {str(deep['format'])!r} (this app reads {DEEP_FORMAT})")
    gene_names = [str(g) for g in deep["genes"].tolist()] if deep["genes"].ndim == 1 else []
    if not gene_names or len(set(gene_names)) != len(gene_names) or any(not g for g in gene_names):
        raise ValueError("the deep export's gene names must be unique and non-empty")
    if genes is not None and gene_names != [str(g) for g in genes]:
        raise ValueError("deep.npz genes differ from encoder.npz genes (names or order)")
    for key in ("train_logit", "train_index", "st_genes", "st_z", "Zm", "D0", "Z0"):
        if deep[key].ndim not in ((3,) if key == "st_z" else (2,)):
            raise ValueError(f"deep array {key} has {deep[key].ndim} dimensions")
    G, H, Hd = len(gene_names), deep["Z0"].shape[0], deep["D0"].shape[0]
    n_latent = deep["Zm"].shape[0]
    N, K = deep["train_logit"].shape
    S, T = deep["st_genes"].shape
    P = len(deep["st_own_rows"]) if deep["st_own_rows"].ndim == 1 else -1
    shapes = {"Z0": (H, G), "zb0": (H,), "Z1": (H, H), "zb1": (H,), "Zm": (n_latent, H), "zbm": (n_latent,),
              "Zv": (n_latent, H), "zbv": (n_latent,), "D0": (Hd, n_latent), "db0": (Hd,), "D1": (Hd, Hd), "db1": (Hd,),
              "Ds": (G, Hd), "dbs": (G,), "ln_eps": (), "z_var_eps": (), "background": (G,), "typical": (G,),
              "median_distance": (), "clamp": (), "diffuse_scale": (), "train_logit": (N, K), "rank_logit": (K,),
              "st_d2": (S, K), "st_z": (S, K + 1, n_latent), "st_pi": (S, K), "st_own": (P, G), "st_nb_split": (S, T)}
    for key, shape in shapes.items():
        array = deep[key]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"deep array {key}: {array.dtype}{list(array.shape)}, expected float32{list(shape)}")
    for key, shape in {"train_index": (N, K), "st_rows": (S,), "st_genes": (S, T), "st_own_rows": (P,)}.items():
        if deep[key].shape != shape or deep[key].dtype != np.int32:
            raise ValueError(f"deep array {key}: {deep[key].dtype}{list(deep[key].shape)}, expected int32{list(shape)}")
    if N <= K or K < 1 or S < 1 or not 1 <= T <= G or P < 1:
        raise ValueError("the deep export holds too few learning cells, neighbours or self-test cells")
    if not (0 < float(deep["ln_eps"]) < 1 and 0 <= float(deep["z_var_eps"]) < 1 and float(deep["median_distance"]) > 0
            and float(deep["clamp"]) > 0 and float(deep["diffuse_scale"]) > 0):
        raise ValueError("deep scalars out of range")
    for key in ("background", "typical"):
        if (deep[key] < 0).any() or abs(float(deep[key].sum(dtype=np.float64)) - 1.0) > 1e-3:
            raise ValueError(f"deep {key} is not a profile summing to 1")
    if (np.abs(deep["train_logit"]) > 10.0).any():
        raise ValueError("learned neighbour values outside (-10, 10)")
    if deep["train_index"].min() < 0 or deep["train_index"].max() >= N or deep["st_rows"].min() < 0 \
            or deep["st_rows"].max() >= N or deep["st_genes"].min() < 0 or deep["st_genes"].max() >= G \
            or deep["st_own_rows"].min() < 0 or deep["st_own_rows"].max() >= S:
        raise ValueError("deep neighbour rows or self-test indices out of range")
    if (deep["train_index"] == np.arange(N, dtype=np.int32)[:, None]).any():
        raise ValueError("a learning cell is listed as its own neighbour")
    if np.abs(deep["rank_logit"] - deep["train_logit"].mean(axis=0, dtype=np.float64)).max() > 1e-5:
        raise ValueError("rank_logit is not the column mean of train_logit")
    sha = str(deep["train_ids_sha256"])
    if deep["train_ids_sha256"].shape != () or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("train_ids_sha256 is not a sha256")
    indptr, indices, values = deep["st_indptr"], deep["st_indices"], deep["st_data"]
    if (indptr.dtype != np.int64 or indices.dtype != np.int32 or values.dtype != np.float32 or indptr.ndim != 1
            or len(indptr) != S * (K + 1) + 1 or indptr[0] != 0 or indptr[-1] != len(indices)
            or len(indices) != len(values) or (np.diff(indptr) < 0).any()
            or (len(indices) and (indices.min() < 0 or indices.max() >= G)) or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError("malformed deep self-test counts")
    if enc is None and path.with_name("encoder.npz").is_file():
        enc = load_encoder(path.with_name("encoder.npz"))
    if enc is not None and list(enc["genes"]) != gene_names:
        raise ValueError("deep.npz genes differ from encoder.npz genes (names or order)")
    deep["genes"] = gene_names
    if enc is not None:
        deep["enc"] = enc
    return deep


def _host(array) -> np.ndarray:
    """A host numpy copy of a numpy or cupy array."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _ln_relu_last(xp, h, eps):
    """torch LayerNorm (no affine, biased variance) over the last axis, then ReLU; any leading shape."""
    mean = h.mean(axis=-1, keepdims=True)
    centred = h - mean
    return xp.maximum(centred / xp.sqrt((centred * centred).mean(axis=-1, keepdims=True) + eps), 0)


def _softmax_last(xp, a):
    a = xp.exp(a - a.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


def deep_z_mean(xp, deep, normalised):
    """The z encoder's mean of log1p(normalised counts) (rows x model genes; any leading shape)."""
    eps = float(deep["ln_eps"])
    h = _ln_relu_last(xp, xp.log1p(normalised) @ deep["Z0"].T + deep["zb0"], eps)
    h = _ln_relu_last(xp, h @ deep["Z1"].T + deep["zb1"], eps)
    return h @ deep["Zm"].T + deep["zbm"]


def deep_profile(xp, deep, z):
    """The decoder's px_scale (a gene profile summing to 1) of latent rows z (any leading shape)."""
    eps = float(deep["ln_eps"])
    h = _ln_relu_last(xp, z @ deep["D0"].T + deep["db0"], eps)
    h = _ln_relu_last(xp, h @ deep["D1"].T + deep["db1"], eps)
    return _softmax_last(xp, h @ deep["Ds"].T + deep["dbs"])


def deep_pi(xp, deep, d2, logit):
    """The guide's neighbour shares: softmax_k(logit_k - min(sqrt(d2_k / median_distance), clamp))."""
    return _softmax_last(xp, logit - xp.minimum(xp.sqrt(d2 / float(deep["median_distance"])), float(deep["clamp"])))


def deep_map(xp, deep, x, L, rho, s, sk, d2, rounds: int = 100, floor: float = 1e-6):
    """Neighbour shares for a cell the model never learned (3.2 step 6, "learned no"): start from distance only, then
    `rounds` MAP-EM steps under the model's Dirichlet prior alpha_k = 30 (exp(-min(diffuse_scale d2_k / median, 20))
    + 0.001), with the profiles and parts fixed. Only genes with x_g > 0 enter the expected counts n_k (the others add 0)."""
    median = float(deep["median_distance"])
    alpha = 30.0 * (xp.exp(-xp.minimum(float(deep["diffuse_scale"]) * d2 / median, 20.0)) + 1e-3)
    pi = deep_pi(xp, deep, d2, xp.zeros_like(d2))
    nz = x > 0
    xs = x[nz]
    base = rho[0] * L * s[nz] + rho[2] * L * deep["background"][nz]
    a = (rho[1] * L) * sk[:, nz]
    for _ in range(int(rounds)):
        comp = pi[:, None] * a
        n_k = (comp * (xs / (base + comp.sum(axis=0)))[None, :]).sum(axis=1)
        num = xp.maximum(n_k + alpha - 1.0, floor)
        pi = num / num.sum()
    return pi


def deep_kind_words(xp, deep, s, n: int = 4) -> list[int]:
    """What a profile looks like: its top genes by s_g ln(s_g / typical_g) among genes with s_g >= 2 typical_g (gene
    indices, best first)."""
    typical = deep["typical"]
    score = _host(xp.where(s >= 2.0 * typical, s * xp.log(xp.maximum(s, 1e-30) / xp.maximum(typical, 1e-30)),
                           -xp.inf)).astype(np.float64)
    top = np.argpartition(-score, n)[:n] if len(score) > n else np.arange(len(score))
    top = top[np.lexsort((top, -score[top]))]
    return [int(g) for g in top if np.isfinite(score[g])]


def deep_forward(xp, enc, deep, x, xn, d2, logit=None, *, rho=None, map_rounds=100, same_at=0.9, tag_ratio=4.0,
                 tag_min_tx=2.0, tag_share=0.5, own_share=0.5, own_min_tx=2.0, scale_tol=1e-3, scale_cap=50,
                 rough_at=0.10, profiles=True) -> dict:
    """One cell explained (the defaults are calibrated rules). x [G]:
    its counts over the model's genes; xn [K, G]: its K neighbours' counts (the frame's nearest); d2 [K]: squared centre
    distances (um^2); logit [K]: learned neighbour values (carried over by id; the rank means for new pairs), or None
    (a cell the model never learned: MAP rounds). rho: the calculator's raw weights if already known. The networks run in
    x's dtype; the gene split and its scaling always in float64. Returns the parts, the per-gene split before and after
    scaling (genes with x_g > 0), the amounts per neighbour, the tags, its own genes, the kind words and diagnostics."""
    L = float(x.sum())
    if rho is None:
        rho = mixing_weights(enc, x[None], xp, x.dtype)[0]
    parts = rho / rho.sum()
    m = deep_z_mean(xp, deep, (x / x.mean())[None])[0]
    s = deep_profile(xp, deep, m)
    mk = deep_z_mean(xp, deep, xn / xn.mean())               # ONE mean over all K x G neighbour counts
    sk = deep_profile(xp, deep, mk)
    if logit is not None:
        pi = deep_pi(xp, deep, d2, logit)
    else:
        pi = deep_map(xp, deep, x, L, rho, s, sk, d2, map_rounds)
    nz = np.nonzero(_host(x) > 0)[0]
    xs = x[nz]
    mu_own = rho[0] * L * s[nz]
    mu_k = (rho[1] * L) * pi[:, None] * sk[:, nz]            # [K, n]; each neighbour profile at THIS cell's depth
    mu_nb = mu_k.sum(axis=0)
    mu_bg = rho[2] * L * deep["background"][nz]
    mu = mu_own + mu_nb + mu_bg
    # the split and its scaling in float64 whatever the network's dtype (float32 sums of a few thousand transcripts
    # stall the scaling at ~1e-3 transcripts)
    A = (xs[:, None] * xp.stack([mu_own, mu_nb, mu_bg], axis=1) / mu[:, None]).astype(np.float64)   # [n, 3] raw
    raw = A.copy()
    xs64 = xs.astype(np.float64)
    raw_gap = float(abs(A[:, 1].sum() / L - float(parts[1])))
    target = L * parts.astype(np.float64)
    rounds = 0
    for rounds in range(1, int(scale_cap) + 1):
        A = A * (target / A.sum(axis=0))[None, :]
        A = A * (xs64 / A.sum(axis=1))[:, None]
        if float(xp.abs(A.sum(axis=0) - target).max()) <= scale_tol:
            break
    col_err = float(xp.abs(A.sum(axis=0) - target).max())
    share_k = (mu_k / xp.maximum(mu_nb, 1e-30)[None, :]).astype(np.float64)
    Ak = A[:, 1][None, :] * share_k                          # [K, n]
    tx_k = Ak.sum(axis=1)
    similar = (sk @ s) / (xp.sqrt((sk * sk).sum(axis=1)) * xp.sqrt((s * s).sum()))
    same = similar >= same_at
    nb_total = float(tx_k.sum())
    same_kind_share = float(tx_k[same].sum() / nb_total) if nb_total > 0 else 0.0
    giver = Ak.argmax(axis=0)                                # the neighbour giving most of each gene
    ratio_ok = sk[giver, nz] >= tag_ratio * s[nz]
    tag = (xs64 >= tag_min_tx) & (A[:, 1] / xs64 >= tag_share) & ~same[giver] & ratio_ok
    own_ok = (xs64 >= own_min_tx) & (A[:, 0] / xs64 >= own_share)
    out = {"L": L, "rho": rho, "parts": parts, "pi": pi, "genes_nz": nz, "x_nz": xs, "A_raw": raw, "A": A, "Ak": Ak,
           "tx_k": tx_k, "similar": similar, "same": same, "same_kind_share": same_kind_share, "giver": giver,
           "tag": tag, "own_ok": own_ok, "raw_gap": raw_gap, "rough": raw_gap > rough_at, "scale_rounds": rounds,
           "scale_col_err": col_err, "m": m, "mk": mk}
    if profiles:
        out.update(s=s, sk=sk, kind=deep_kind_words(xp, deep, s, 4),
                   kind_k=[deep_kind_words(xp, deep, sk[k], 3) for k in range(sk.shape[0])])
    return out


def deep_selftest_counts(deep) -> np.ndarray:
    """The self-test cells and their training neighbours as a dense float32 array [S, K + 1, G] (row 0: the cell)."""
    S, K = deep["st_d2"].shape
    G = deep["Ds"].shape[0]
    indptr, indices, values = deep["st_indptr"], deep["st_indices"], deep["st_data"]
    dense = np.zeros((S * (K + 1), G), np.float32)
    dense[np.repeat(np.arange(S * (K + 1)), np.diff(indptr)), indices] = values
    return dense.reshape(S, K + 1, G)


def _as_dtype(arrays: dict, dtype) -> dict:
    return {k: (v.astype(dtype, copy=False) if isinstance(v, np.ndarray) and v.dtype == np.float32 else v)
            for k, v in arrays.items()}


def deep_self_test_parts(deep: dict, enc: dict | None = None, dtype=np.float64) -> dict:
    """Max |numpy - torch (noise off)| per self-test output, over the self-test cells: "z" (the z means of each cell and
    its K training neighbours [S, K+1, n_latent]), "pi" (the guide's neighbour shares [S, K]), "own" (decoded own
    profiles [P, G]), "nb_split" (the top genes' observed neighbour split before scaling, A_nb / x [S, T]: it needs the
    calculator, `enc` or deep["enc"], and is None without one) and "max". numpy in `dtype` (the trainer and the engine
    float64, the worker float32)."""
    enc = enc if enc is not None else deep.get("enc")
    d = _as_dtype(deep, dtype)
    e = _as_dtype(enc, dtype) if enc is not None else None
    counts = deep_selftest_counts(deep).astype(dtype)
    diff = {"z": 0.0, "pi": 0.0, "own": 0.0, "nb_split": 0.0 if e is not None else None}
    own_at = {int(r): p for p, r in enumerate(deep["st_own_rows"].tolist())}
    for i in range(counts.shape[0]):
        x, xn, d2 = counts[i, 0], counts[i, 1:], d["st_d2"][i]
        logit = d["train_logit"][int(deep["st_rows"][i])]
        m = deep_z_mean(np, d, (x / x.mean())[None])[0]
        mk = deep_z_mean(np, d, xn / xn.mean())
        diff["z"] = max(diff["z"], float(np.abs(np.concatenate([m[None], mk]) - d["st_z"][i]).max()))
        diff["pi"] = max(diff["pi"], float(np.abs(deep_pi(np, d, d2, logit) - d["st_pi"][i]).max()))
        if i in own_at:
            diff["own"] = max(diff["own"], float(np.abs(deep_profile(np, d, m) - d["st_own"][own_at[i]]).max()))
        if e is not None:
            r = deep_forward(np, e, d, x, xn, d2, logit, profiles=False)
            split = np.zeros(x.shape[0], np.float64)
            split[r["genes_nz"]] = r["A_raw"][:, 1] / r["x_nz"]
            genes = deep["st_genes"][i]
            diff["nb_split"] = max(diff["nb_split"], float(np.abs(split[genes] - d["st_nb_split"][i]).max()))
    diff["max"] = max(v for v in diff.values() if v is not None)
    return diff


def deep_self_test(deep: dict, enc: dict | None = None, dtype=np.float64) -> float:
    """The largest |numpy - torch (noise off)| over every self-test output (deep_self_test_parts), against
    DEEP_SELFTEST_TOLERANCE (float64) or DEEP_WORKER_TOLERANCE (float32). With the calculator (`enc`, or the one
    load_deep attached) the gene split is tested too; without one, the networks and the neighbour shares only."""
    return float(deep_self_test_parts(deep, enc, dtype)["max"])


def deep_selftest_rows(chosen: np.ndarray, totals: np.ndarray, share: np.ndarray, judge_min: float,
                       mixed_at: float) -> np.ndarray:
    """16 of the calculator's reference cells (training rows): 8 that look mixed, 4 with some mixing, 4 with little,
    each group spread over depth; short groups are filled from the other judged reference cells, then from the rest."""
    chosen = np.asarray(chosen)
    judged = chosen[totals[chosen] >= judge_min]
    groups = {"mixed": judged[share[judged] >= mixed_at],
              "some": judged[(share[judged] >= DEEP_SOME_AT) & (share[judged] < mixed_at)],
              "little": judged[share[judged] < DEEP_SOME_AT]}
    picked = []
    for name, n in DEEP_SELFTEST:
        pool = groups[name][np.argsort(totals[groups[name]], kind="stable")]
        if len(pool):
            picked.extend(pool[np.unique(np.linspace(0, len(pool) - 1, min(n, len(pool))).round().astype(int))].tolist())
    want = sum(n for _, n in DEEP_SELFTEST)
    for pool in (judged, chosen):
        rest = [int(r) for r in pool[np.argsort(totals[pool], kind="stable")] if int(r) not in set(picked)]
        while len(picked) < want and rest:
            picked.append(rest.pop(len(rest) // 2))
    return np.asarray(picked[:want], np.int64)


def detail_parity(installed, enc: dict) -> dict:
    """A detail job's parity rule: the new calculator `enc` against the installed one (`installed`: its
    encoder.npz, or the calculator as load_encoder gives it): accepted when the gene names and every exported array are
    identical, or when the installed reference cells' raw weights differ by at most DETAIL_TOLERANCE. Raises
    TrainerError("detail_mismatch") otherwise; returns the record model.json keeps under "detail"."""
    old = installed if isinstance(installed, dict) else load_encoder(installed)
    keys = ("W0", "b0", "W1", "b1", "Wv", "bv", "ln_eps", "var_eps", "diffusion_eps")
    same_genes = list(old["genes"]) == list(enc["genes"])
    identical = same_genes and all(np.array_equal(old[k], enc[k]) for k in keys)
    max_abs = {k: float(np.abs(old[k].astype(np.float64) - enc[k].astype(np.float64)).max())
               for k in keys if old[k].shape == enc[k].shape}
    outputs = None
    if not identical and same_genes and all(old[k].shape == enc[k].shape for k in keys):
        ref = selftest_counts(old).astype(np.float64)
        outputs = float(np.abs(mixing_weights(old, ref, np, np.float64) - mixing_weights(enc, ref, np, np.float64)).max())
    accepted = identical or (outputs is not None and outputs <= DETAIL_TOLERANCE)
    record = {"identical": bool(identical), "same_genes": bool(same_genes), "max_abs_diff": max_abs,
              "reference_outputs_max_abs_diff": outputs, "tolerance": DETAIL_TOLERANCE, "accepted": bool(accepted)}
    if not accepted:
        why = "other genes" if not same_genes else (f"the reference cells differ by {outputs:.2e}" if outputs is not None
                                                     else "other layer sizes")
        raise TrainerError("detail_mismatch", f"the new learning does not match the installed calculator ({why}); "
                                              "nothing was changed")
    return record


def provenance(model_json: dict) -> dict:
    """model.json's "provenance" block, from the rest of model.json (so an older model's file gives it as
    well): table (the cell table learned from), other_table (the sample's other cell table pooled into the priors, None
    when none was), priors ("pooled" over both tables as the reference recipe does, or "single": the learned table
    alone, with priors_why), epochs_run against epochs_notebook (50), cells (learned from), notebook_equivalent, and
    differences: the keys of PROVENANCE_DIFFERENCES that apply, in that order (none when the model is the one the
    reference recipe learns)."""
    rec = model_json.get("recipe") if isinstance(model_json.get("recipe"), dict) else {}
    table = (model_json.get("table") or {}).get("key") if isinstance(model_json.get("table"), dict) else None
    source = model_json.get("priors_from") if isinstance(model_json.get("priors_from"), dict) else {}
    tables = source.get("tables") if isinstance(source.get("tables"), list) else []
    rule = source.get("rule")
    priors = None if rule is None else ("pooled" if rule == "pooled" and len(tables) > 1 else "single")
    epochs = model_json.get("epochs") if isinstance(model_json.get("epochs"), int) else None
    planned = rec.get("epochs") if isinstance(rec.get("epochs"), int) else None
    found = set()
    if table is not None and table != NOTEBOOK_TABLE:
        found.add("other_table_used")
    if priors == "single":
        found.add("single_table_priors")
    if epochs is not None and planned is not None and epochs > planned:
        found.add("more_epochs_small_table")
    if planned is not None and planned != NOTEBOOK_EPOCHS:
        found.add("other_epochs")
    if rec and any(rec.get(k) != v for k, v in NOTEBOOK_SETTINGS.items()):
        found.add("other_settings")
    if table is None or priors is None or epochs is None or planned is None:
        found.add("not_recorded")
    differences = [key for key in PROVENANCE_DIFFERENCES if key in found]
    return {"table": table, "other_table": tables[1] if priors == "pooled" else None, "priors": priors,
            "priors_why": source.get("why") if priors == "single" else None, "epochs_run": epochs,
            "epochs_notebook": NOTEBOOK_EPOCHS, "cells": (model_json.get("cells") or {}).get("fitted")
            if isinstance(model_json.get("cells"), dict) else None,
            "notebook_equivalent": not differences, "differences": differences}


# ---- the process --------------------------------------------------------------------------------------------------
class Reporter:
    """JSON lines to the engine on the real stdout (fd 1 is pointed at stderr, so nothing a library prints can break
    the protocol), and the same in words to out_dir/train.log, written first: the engine ends this process (the end of
    its stdin, which leaves at once) as soon as it has read a last line, so a line sent first could miss the log."""

    def __init__(self):
        self.out = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self.log_path: Path | None = None
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def send(self, **message):
        line = json.dumps(message, default=float)
        with self.lock:
            if self.log_path is not None:
                try:
                    with open(self.log_path, "a") as log:
                        log.write(f"{time.monotonic() - self.started:8.1f} s  {line}\n")
                except OSError:
                    pass
            try:
                self.out.write(line + "\n")
                self.out.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def progress(self, stage: str, step=None, of=None, **extra):
        self.send(ev="progress", stage=stage, step=step, of=of, **extra)


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def _watch_stdin():
    """The end of stdin means the engine stopped or cancelled the fit: leave at once (os._exit: no teardown that
    could hang on the GPU)."""
    def watch():
        try:
            for _ in sys.stdin:
                pass
        except (OSError, ValueError):
            pass
        os._exit(75)
    threading.Thread(target=watch, name="stdin-watch", daemon=True).start()


def _die_with_parent():
    """Linux: SIGTERM when the engine dies (PR_SET_PDEATHSIG). macOS and Windows have no such signal: the engine's
    reaper ends the trainer there (portable.tie), and the end of stdin ends it as everywhere (_watch_stdin)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def store_lock_paths(zarr_path) -> list[Path]:
    """The sample's save lock files a read respects: exactly those a save takes (zarr_bridge/helper.py `_lock_paths`:
    the private folder SOPA_COMPANION_ZARR_LOCK_DIR names, alone; else the previous version's folder, then this
    version's), in that order. Nothing is created."""
    from sopa_companion.zarr_bridge.helper import _lock_paths
    return _lock_paths(Path(zarr_path).resolve(), create=False)


@contextmanager
def _held_shared(target: Path, report: Reporter | None, deadline: float | None = None):
    """Yields True while holding `target` shared (opened read-only: neither created nor touched), False when there is
    no such file, or no longer the one at the path (prune_locks removed it meanwhile). `deadline` (time.monotonic()):
    TimeoutError when a save still holds it then, instead of waiting for the save to finish."""
    from .. import portable
    try:
        fd = os.open(target, os.O_RDONLY | portable.O_NOFOLLOW)
    except FileNotFoundError:
        yield False
        return
    try:
        try:
            portable.lock(fd, shared=True, block=False)
        except BlockingIOError:
            if report:
                report.progress("lock", note="waiting for a save into this sample to finish")
            if deadline is None:
                portable.lock(fd, shared=True)
            else:                                     # polled until the deadline
                while True:
                    time.sleep(0.05)
                    try:
                        portable.lock(fd, shared=True, block=False)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("a save into the sample is under way") from None
        try:
            mine, current = os.fstat(fd), os.stat(target)
            held = (mine.st_dev, mine.st_ino) == (current.st_dev, current.st_ino)
        except FileNotFoundError:
            held = False
        yield held
    finally:
        os.close(fd)


def read_consistently(zarr_path, read, report: Reporter | None = None, attempts: int = 5, timeout: float | None = None):
    """read() while no save into the sample can be writing it. A save (zarr_bridge/helper.py) creates the sample's
    lock file, then holds it exclusively, before it writes anything. So: hold each of the sample's lock files that
    exists (store_lock_paths) shared, read, then check that no other one appeared meanwhile (a save that began where the
    sample had no lock file yet, e.g. the first save into it), else read again. The lock files are never created or
    touched here: reading leaves no trace (and the harness's guard on its input zarrs sees none). `timeout` (seconds, in
    all): TimeoutError when a save still holds the sample by then; None waits for the save to finish."""
    targets = store_lock_paths(zarr_path)
    deadline = None if timeout is None else time.monotonic() + timeout
    for _ in range(attempts):
        present = [target for target in targets if target.exists()]
        with ExitStack() as held:
            if all(held.enter_context(_held_shared(target, report, deadline)) for target in present):
                result = read()
                if all(target in present for target in targets if target.exists()):
                    return result
    raise TrainerError("unreadable", "the sample kept changing while it was read (a save into it is under way)")


@contextmanager
def shared_zarr_lock(zarr_path: Path, report: Reporter | None = None):
    """Wait for any save into the sample to finish and keep new ones out while the block runs, for each of its lock
    files that exists (see read_consistently, which also covers samples that never had one)."""
    with ExitStack() as held:
        for target in store_lock_paths(zarr_path):
            held.enter_context(_held_shared(target, report))
        yield


def proseg_table(zarr_path, key) -> bool:
    """Whether tables/<key> of the sample at `zarr_path` holds proseg's own count: its uns has one of
    PROSEG_TABLE_KEYS (a group or an array: a folder). Only folders are looked at (nothing is opened or created); False
    for a missing or unreadable one, or a key that names no table. Only such a table is scored into batch.parquet (the
    earlier proseg cells' line); the engine shows that line on the same rule (engine/qc.py `_batch`)."""
    if not zarr_path or not isinstance(key, str) or not TABLE_KEY.fullmatch(key):
        return False
    uns = Path(zarr_path) / "tables" / key / "uns"
    try:
        return any((uns / name).is_dir() for name in PROSEG_TABLE_KEYS)
    except OSError:
        return False


def read_table(zarr_path: Path, key: str):
    """tables/<key> of the sample, opened read-only. -> (CSR float32 counts, obs DataFrame, var names, gene ids,
    its own centroids obsm['spatial'] [n, 2] float64, or None without them: table_centres places its cells then).
    Refuses counts that are not whole, non-negative numbers (not_counts)."""
    import anndata
    from scipy import sparse
    folder = Path(zarr_path) / "tables" / key
    if not folder.is_dir():
        raise TrainerError("no_table", f"the sample has no table '{key}'")
    try:
        table = anndata.read_zarr(str(folder))
    except Exception as exc:
        raise TrainerError("unreadable", f"table '{key}' could not be read: {type(exc).__name__}: {exc}") from exc
    spatial = table.obsm.get("spatial") if hasattr(table.obsm, "get") else None
    if spatial is not None and (np.asarray(spatial).ndim != 2 or np.asarray(spatial).shape[1] < 2
                                or np.asarray(spatial).shape[0] != table.n_obs):
        spatial = None
    counts = table.X
    if counts is None:
        raise TrainerError("not_counts", f"table '{key}' holds no counts")
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    counts.eliminate_zeros()
    values = counts.data
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.rint(values)).all():
        raise TrainerError("not_counts", f"table '{key}' does not hold whole transcript counts")
    genes = np.asarray(table.var_names.astype(str))
    gene_ids = np.asarray(table.var["gene_ids"].astype(str)) if "gene_ids" in table.var else np.asarray([""] * len(genes))
    return counts, table.obs, genes, gene_ids, None if spatial is None else np.asarray(spatial, dtype=np.float64)[:, :2]


# ---- the cells' centres in the transcripts' frame (choose_centres decides) ----------------------------------------
def table_annotation(zarr_path, key: str) -> dict:
    """What tables/<key> annotates, as spatialdata writes it into the table group's attributes (zarr.json, or .zattrs
    in format 0.1), else its uns["spatialdata_attrs"]: {"region": [element names], "region_key", "instance_key"};
    {"region": [], ...None} when it says nothing."""
    folder = Path(zarr_path) / "tables" / key
    attrs = {}
    for name in ("zarr.json", ".zattrs"):
        try:
            data = json.loads((folder / name).read_text())
        except (OSError, ValueError):
            continue
        attrs = (data.get("attributes") or {}) if name == "zarr.json" else data
        break
    if "region" not in attrs:
        try:
            import zarr
            from anndata.io import read_elem
            attrs = dict(read_elem(zarr.open_group(str(folder / "uns" / "spatialdata_attrs"), mode="r")))
        except Exception:
            attrs = {}
    region = attrs.get("region")
    regions = [str(region)] if isinstance(region, str) else \
        [str(r) for r in region] if isinstance(region, (list, tuple, np.ndarray)) else []
    return {"region": regions, "region_key": attrs.get("region_key"), "instance_key": attrs.get("instance_key")}


def sample_frame(zarr_path, points_key: str | None = None):
    """The sample read with spatialdata, read-only (its points lazily: no transcript is loaded; its shapes), and the
    points element whose intrinsic frame is the working frame: the job's (the engine's open sample), else
    zarr_bridge/helper.py's rule (the attrs' transcripts element, else "transcripts", else the first). -> (sdata,
    points_key), or (None, None) when the sample has no points element or spatialdata cannot read it (nothing can be
    placed on the transcripts then: the table's own centroids are used unchecked)."""
    folder = Path(zarr_path) / "points"
    try:
        if not any(child.is_dir() for child in folder.iterdir()):
            return None, None
    except OSError:
        return None, None
    try:
        import warnings
        import spatialdata as sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdata = sd.read_zarr(str(zarr_path), selection=("points", "shapes"))
    except Exception as exc:
        print(f"qc_trainer: the sample's transcripts and outlines could not be read ({type(exc).__name__}: {exc}): "
              "the tables' own centroids are used as they are", file=sys.stderr, flush=True)
        return None, None
    keys = list(sdata.points)
    if not keys:
        return None, None
    if points_key not in keys:
        attrs = dict(sdata.attrs or {})
        points_key = next((k for k in (attrs.get("transcripts_dataframe"), attrs.get("transcripts"), "transcripts")
                           if k in keys), keys[0])
    return sdata, points_key


def points_extent(sdata, points_key: str, zarr_path=None) -> list[float] | None:
    """The transcripts' [x0, y0, x1, y1] in their own frame: from their parquet files' statistics (nothing read but
    the footers), else from their x and y columns; None when that fails."""
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    try:
        import pyarrow.parquet as pq
        base = Path(zarr_path) / "points" / points_key / "points.parquet"
        files = [base] if base.is_file() else sorted(base.glob("*.parquet"))
        if not files:
            raise LookupError("no parquet files")
        for path in files:
            meta = pq.ParquetFile(path).metadata
            names = [meta.schema.column(i).name for i in range(meta.num_columns)]
            for group in range(meta.num_row_groups):
                for axis, name in enumerate(("x", "y")):
                    stats = meta.row_group(group).column(names.index(name)).statistics
                    if stats is None or not stats.has_min_max:
                        raise LookupError(f"no statistics for {name}")
                    low[axis], high[axis] = min(low[axis], float(stats.min)), max(high[axis], float(stats.max))
    except Exception:
        try:
            import dask
            points = sdata.points[points_key]
            low[0], high[0], low[1], high[1] = (float(v) for v in dask.compute(
                points["x"].min(), points["x"].max(), points["y"].min(), points["y"].max()))
        except Exception as exc:
            print(f"qc_trainer: the transcripts' extent could not be read: {type(exc).__name__}: {exc}", file=sys.stderr,
                  flush=True)
            return None
    extent = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
    return extent if np.isfinite(extent).all() and extent[2] >= extent[0] and extent[3] >= extent[1] else None


def _ids(values) -> np.ndarray:
    """Cell ids as strings, whole numbers without a decimal point (an int index and a float column name one cell)."""
    array = np.asarray(values)
    if array.dtype.kind == "f" and np.isfinite(array).all() and np.equal(array, np.rint(array)).all():
        array = array.astype(np.int64)
    return array.astype(str)


def outline_centres(sdata, points_key: str, annotation: dict, obs) -> tuple[np.ndarray | None, str]:
    """Each table row's outline centre in the transcripts' frame: the shapes element(s) the table annotates (its
    "region"; rows told apart by its region_key when it names several), mapped with sopa's to_intrinsic as the helper
    maps every outline (a circle's centre is its point), each row matched to its outline by the table's instance_key.
    -> (centres [n, 2], the elements' names) when every row found its outline; else (None, why): a CosMx table of all
    fields beside one field's outlines (other ids), or a table that annotates labels, not outlines."""
    regions = [r for r in annotation.get("region") or [] if r in sdata.shapes]
    instance, region_key = annotation.get("instance_key"), annotation.get("region_key")
    if not regions or instance is None or instance not in obs:
        return None, "the table names no outlines of its cells"
    n = len(obs)
    if region_key in obs:
        which = np.asarray(obs[region_key].astype(str))
    elif len(annotation["region"]) == 1:
        which = np.full(n, regions[0])
    else:
        return None, "the table names several elements but not which row is whose"
    import pandas as pd
    from sopa.utils import to_intrinsic
    ids = _ids(obs[instance])
    centres = np.full((n, 2), np.nan)
    placed = np.zeros(n, bool)
    for region in regions:
        rows = np.nonzero(which == region)[0]
        if not len(rows):
            continue
        shapes = to_intrinsic(sdata, region, points_key)
        index = pd.Index(_ids(shapes.index))
        if not index.is_unique:
            return None, f"the outlines of '{region}' repeat their ids"
        at = index.get_indexer(ids[rows])
        found = at >= 0
        centroid = shapes.geometry.centroid
        xy = np.stack([centroid.x.to_numpy(dtype=np.float64), centroid.y.to_numpy(dtype=np.float64)], axis=1)
        centres[rows[found]] = xy[at[found]]
        placed[rows[found]] = True
    if not placed.all():
        return None, f"only {int(placed.sum())} of its {n} cells have an outline in {', '.join(regions)}"
    return centres, ", ".join(regions)


def table_centres(zarr_path, key: str, obs, spatial, sdata, points_key, extent) -> tuple[np.ndarray, dict]:
    """choose_centres for tables/<key>: its outlines' centres when it annotates shapes that hold every one of its
    cells (outline_centres), checked against the transcripts' `extent`; model.json keeps the record ("centres")."""
    outlines, region = None, None
    if sdata is not None:
        try:
            outlines, region = outline_centres(sdata, points_key, table_annotation(zarr_path, key), obs)
        except Exception as exc:                          # outlines that cannot be mapped: as if it had none
            outlines, region = None, f"its outlines could not be placed: {type(exc).__name__}: {exc}"[:300]
    xy, found = choose_centres(spatial, outlines, extent, key, region if outlines is not None else None)
    if outlines is None and region:
        found["why"] = region
    found["frame"] = points_key
    return xy, found


def _cell_columns(obs, counts):
    """cell id (the obs cell_id column if any, else the index), area (cell_area if any) and totals."""
    ids = np.asarray((obs["cell_id"] if "cell_id" in obs else obs.index).astype(str))
    area = np.asarray(obs["cell_area"], dtype=np.float64) if "cell_area" in obs else np.full(len(ids), np.nan)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    return ids, area, totals


def _write_cells(path: Path, ids, xy, area, totals, share):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"cell_id": pa.array(ids.astype(str)), "x": pa.array(xy[:, 0].astype(np.float32)),
                      "y": pa.array(xy[:, 1].astype(np.float32)), "area": pa.array(area.astype(np.float32)),
                      "counts": pa.array(totals.astype(np.int32)), "share": pa.array(share.astype(np.float32))})
    pq.write_table(table, path, compression="zstd")


def _per100(share, totals, mixed_at, judge_min) -> dict:
    judged = totals >= judge_min
    mixed = judged & (share >= mixed_at)
    count = int(judged.sum())
    return {"cells": int(len(totals)), "judged": count, "mixed": int(mixed.sum()),
            "per100": round(100.0 * int(mixed.sum()) / count, 2) if count else None}


def _reference_cells(totals: np.ndarray, share: np.ndarray, n: int = SELFTEST_CELLS) -> np.ndarray:
    """Up to n cells spread over depth (counts) and over the neighbour share, so the self-test covers shallow, deep,
    clean and mixed cells."""
    if len(totals) <= n:
        return np.arange(len(totals))
    by_depth = np.argsort(totals, kind="stable")[np.linspace(0, len(totals) - 1, n // 2).round().astype(int)]
    by_share = np.argsort(share, kind="stable")[np.linspace(0, len(totals) - 1, n // 2).round().astype(int)]
    chosen = np.unique(np.concatenate([by_depth, by_share]))
    if len(chosen) < n:
        rest = np.setdiff1d(np.arange(len(totals)), chosen)
        chosen = np.unique(np.concatenate([chosen, rest[np.linspace(0, len(rest) - 1, n - len(chosen)).round().astype(int)]]))
    return chosen[:n]


class _Epochs:
    """Lightning callback: one progress line per pass over the cells (mean loss of its batches, accumulated on the
    GPU and read once per pass), the deadline, and how many passes really ran."""

    def __init__(self, report: Reporter, epochs: int, deadline: float):
        self.report, self.epochs, self.deadline = report, epochs, deadline
        self.sum, self.batches, self.done, self.curve = None, 0, 0, []
        self.started = time.monotonic()

    def callback(self):
        from lightning.pytorch import Callback
        outer = self

        class EpochProgress(Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if loss is not None:
                    value = loss.detach()
                    outer.sum = value if outer.sum is None else outer.sum + value
                    outer.batches += 1

            def on_train_epoch_end(self, trainer, pl_module):
                loss = float(outer.sum) / outer.batches if outer.batches else None
                outer.sum, outer.batches = None, 0
                outer.done += 1
                outer.curve.append(loss)
                outer.report.progress("learning", outer.done, outer.epochs, loss=loss,
                                      seconds=round(time.monotonic() - outer.started, 2))
                if time.monotonic() > outer.deadline:
                    raise TrainerError("timeout", "the learning took longer than its time limit")
        return EpochProgress()


def apple_gpu(torch) -> dict:
    """The device record of a fit on Apple's GPU, as a CUDA fit's ({name, pci_bus_id, memory_mb}) and its kind: the
    chip's name (sysctl's machdep.cpu.brand_string, e.g. "Apple M2 Ultra"), no bus, and the memory Metal recommends one
    process uses at most (torch.mps.recommended_max_memory) when torch says it."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        chip = ""
    try:
        memory = int(torch.mps.recommended_max_memory() // 2 ** 20)
    except Exception:
        memory = None
    return {"name": f"{chip or 'Apple silicon'} GPU (MPS)", "kind": "mps", "pci_bus_id": None, "memory_mb": memory}


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "cublas_status_alloc_failed" in text


# ---- Go deeper: the export from the model in this process (torch) ---------------------
def deep_structure(model, n_genes: int):
    """The z encoder's and the decoder's weights by the strict name maps (Z_LAYERS; D_LAYERS, D_UNUSED not exported),
    after the structural checks: two LayerNorms without affine and with one eps per network, and no BatchNorm;
    an encoder taking exactly the model's `n_genes` and a first decoder layer of exactly [hidden, n_latent] (no
    covariate columns); per-gene dispersion, one batch and an exp(.) variance (what deep_forward re-implements).
    -> (z encoder arrays, decoder arrays, LayerNorm eps), float32 numpy. TrainerError("unexpected") otherwise."""
    import torch
    mod = model.module
    zenc, dec = mod.z_encoder, mod.model.decoder
    zs = {k: v.detach().float().cpu().numpy() for k, v in zenc.state_dict().items()}
    ds = {k: v.detach().float().cpu().numpy() for k, v in dec.state_dict().items()}
    if set(zs) != set(Z_LAYERS):
        raise TrainerError("unexpected", f"the z encoder's layers are not the expected ones: {sorted(zs)}")
    if set(ds) != set(D_LAYERS) | D_UNUSED:
        raise TrainerError("unexpected", f"the decoder's layers are not the expected ones: {sorted(ds)}")
    eps = set()
    for net in (zenc, dec):
        norms = [m for m in net.modules() if isinstance(m, torch.nn.LayerNorm)]
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules()) or len(norms) != 2 \
                or any(m.elementwise_affine for m in norms):
            raise TrainerError("unexpected", "the deep networks' normalisation layers are not the expected ones")
        eps |= {float(m.eps) for m in norms}
    if len(eps) != 1:
        raise TrainerError("unexpected", "the z encoder and the decoder use other LayerNorm eps")
    n_latent = zs["mean_encoder.weight"].shape[0]
    if ds["px_decoder.fc_layers.Layer 0.0.weight"].shape[1] != n_latent \
            or zs["encoder.fc_layers.Layer 0.0.weight"].shape[1] != n_genes or ds["px_scale_decoder.0.weight"].shape[0] != n_genes:
        raise TrainerError("unexpected", "the deep networks take covariates or another gene count")
    if getattr(mod.model, "dispersion", None) != "gene" or int(getattr(mod, "n_batch", 1)) != 1:
        raise TrainerError("unexpected", "the model is not the one-batch, per-gene-dispersion ResolVI")
    if getattr(zenc, "var_activation", None) is not torch.exp:
        raise TrainerError("unexpected", "the z encoder's variance is not exp(.)")
    return zs, ds, eps.pop()


def _pstore(name: str) -> np.ndarray:
    """A pyro parameter by name (exactly one store key equal to or ending with it), CONSTRAINED (pyro.param), float32
    numpy. Read in the fit's own process right after learning: model.pt keeps them unconstrained, and another
    ResolVI(...) would clear the store."""
    import pyro
    store = pyro.get_param_store()
    hits = [k for k in store.keys() if k == name or k.endswith(name)]
    if len(hits) != 1:
        raise TrainerError("unexpected", f"pyro parameter {name}: {hits}")
    return pyro.param(hits[0]).detach().float().cpu().numpy()


def deep_torch_reference(model, rows: np.ndarray, own_rows: np.ndarray, top: int = DEEP_SELFTEST_GENES) -> dict:
    """The library's own guide and model, traced on the self-test cells with the noise off (Normal draws at their mean,
    the Gamma at its mean): the z means of each cell and of its K training neighbours (the shared z encoder, hooked),
    the guide's neighbour shares, the model's px_scale, and the observed neighbour split x * mu_nb / mu of each cell's
    top genes (by that split's count)."""
    import pyro.distributions as pdist
    import torch
    from pyro import poutine
    from scvi import REGISTRY_KEYS
    mod = model.module
    mod.eval()
    codes = np.asarray(model.adata_manager.get_from_registry(REGISTRY_KEYS.INDICES_KEY)).ravel().astype(np.int64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    loader = model._make_data_loader(adata=model.adata, indices=sorted_rows, batch_size=len(sorted_rows), shuffle=False)
    tensors = next(iter(loader))
    args, kwargs = mod._get_fn_args_from_batch(tensors)
    if not np.array_equal(kwargs["ind_x"].cpu().numpy().astype(np.int64), codes[sorted_rows]):
        raise TrainerError("unexpected", "the self-test batch came back in another order")
    captured = []
    # float64 on the CPU only: Apple's GPU (MPS) has none (float32 -> float64 is exact wherever it is done).
    handle = mod.z_encoder.register_forward_hook(lambda m_, i_, o_: captured.append(o_[0].detach().cpu().double().numpy()))
    saved = (pdist.Normal.rsample, pdist.Normal.sample, pdist.Gamma.rsample)

    def normal_mean(self, sample_shape=torch.Size()):
        return self.loc.expand(self._extended_shape(sample_shape))

    def gamma_mean(self, sample_shape=torch.Size()):
        return (self.concentration / self.rate).expand(self._extended_shape(sample_shape))
    pdist.Normal.rsample, pdist.Normal.sample, pdist.Gamma.rsample = normal_mean, normal_mean, gamma_mean
    try:
        with torch.inference_mode():
            gtr = poutine.trace(mod.guide).get_trace(*args, **kwargs)
            mtr = poutine.trace(poutine.replay(mod.model, gtr)).get_trace(*args, **kwargs)
    finally:
        pdist.Normal.rsample, pdist.Normal.sample, pdist.Gamma.rsample = saved
        handle.remove()
    if len(captured) != 2:
        raise TrainerError("unexpected", f"the z encoder ran {len(captured)} times in one guide + model pass")

    def site(trace, name):
        return trace.nodes[name]["value"].detach().cpu().double().numpy()
    B = len(sorted_rows)
    K = int(mod.n_neighbors) if hasattr(mod, "n_neighbors") else int(model.adata.obsm["index_neighbor"].shape[1])
    z = np.concatenate([captured[0][:, None, :], captured[1].reshape(B, K, -1)], axis=1)
    pi = site(gtr, "per_neighbor_diffusion")
    px_scale = site(mtr, "px_scale")
    x = kwargs["x"].detach().cpu().double().numpy()
    mu_own = site(mtr, "true_mixture_proportion") * site(mtr, "px_rate")
    mu_nb = (site(mtr, "diffusion_proportion_per_neighbor") * site(mtr, "px_rate_n")).sum(axis=-2)
    mu_bg = site(mtr, "background")
    split = np.where(x > 0, mu_nb / (mu_own + mu_nb + mu_bg), 0.0)
    a_nb = x * split
    back = np.argsort(order, kind="stable")                  # sorted batch -> the requested order
    z, pi, px_scale, split, a_nb = z[back], pi[back], px_scale[back], split[back], a_nb[back]
    genes = np.stack([np.lexsort((np.arange(a_nb.shape[1]), -a_nb[i]))[:top] for i in range(B)]).astype(np.int32)
    return {"st_z": z.astype(np.float32), "st_pi": pi.astype(np.float32),
            "st_own": px_scale[np.asarray(own_rows)].astype(np.float32), "st_genes": genes,
            "st_nb_split": np.take_along_axis(split, genes.astype(np.int64), axis=1).astype(np.float32)}


def export_deep(model, out: Path, counts, ids, totals: np.ndarray, share: np.ndarray, chosen: np.ndarray, enc: dict,
                judge_min: float, mixed_at: float, source: str, also: tuple = ()) -> dict:
    """Write out/deep.npz (exactly DEEP_KEYS) from the model in this process: the z encoder and the decoder
    (deep_structure), the loose-RNA profile softmax(background_concentration), the typical profile (mean x / L of the
    learning cells), the guide's median distance, clamp and diffuse_scale (constrained), the learned neighbour values
    (constrained) re-ordered by the _indices category codes into training-row order with the training neighbour list,
    rank_logit (their column means, averaged in float64), the ids' sha256, and the self-test against torch with the noise
    off. Then reload it strictly and self-test it against the calculator `enc` (float64 within DEEP_SELFTEST_TOLERANCE,
    float32 within DEEP_WORKER_TOLERANCE) and each calculator in `also` (a detail job: the installed one; float64).
    TrainerError("unexpected") on any failure. Returns model.json's "deep" block (`source`: "fit" or "detail")."""
    import torch
    from scvi import REGISTRY_KEYS
    t0 = time.monotonic()
    zs, ds, eps = deep_structure(model, len(enc["genes"]))
    mod = model.module
    codes = np.asarray(model.adata_manager.get_from_registry(REGISTRY_KEYS.INDICES_KEY)).ravel().astype(np.int64)
    pmap = _pstore("per_neighbor_diffusion_map")
    if pmap.shape[0] != counts.shape[0] or len(codes) != counts.shape[0] \
            or not np.array_equal(np.sort(codes), np.arange(counts.shape[0])):
        raise TrainerError("unexpected", "the learned neighbour values do not match the learning table's cells")
    train_logit = np.ascontiguousarray(pmap[codes], dtype=np.float32)       # training-row order (cells.parquet)
    index = np.asarray(model.adata.obsm["index_neighbor"]).astype(np.int32)
    background = torch.softmax(torch.as_tensor(_pstore("background_concentration")), dim=-1).numpy()   # loose RNA
    if background.ndim != 2 or background.shape[0] != 1:
        raise TrainerError("unexpected", f"the loose-RNA profile has shape {list(background.shape)}")
    from scipy import sparse
    typical = np.asarray((sparse.diags(1.0 / np.maximum(totals.astype(np.float64), 1.0)) @ counts).mean(axis=0)).ravel()
    rows = deep_selftest_rows(chosen, totals, share, judge_min, mixed_at)
    if not len(rows):
        raise TrainerError("unexpected", "no reference cells for the deep self-test")
    own_rows = np.unique(np.concatenate([np.arange(min(2, len(rows))), [8, 12]]))
    own_rows = own_rows[own_rows < len(rows)][:DEEP_SELFTEST_PROFILES].astype(np.int32)
    reference = deep_torch_reference(model, rows, own_rows)
    block = counts[np.concatenate([[r] + index[r].tolist() for r in rows]).astype(np.int64)].tocsr()
    block.sort_indices()
    deep = {"format": np.array(DEEP_FORMAT), "genes": np.asarray(enc["genes"], dtype=str),
            **{name: np.ascontiguousarray(zs[key], dtype=np.float32) for key, name in Z_LAYERS.items()},
            **{name: np.ascontiguousarray(ds[key], dtype=np.float32) for key, name in D_LAYERS.items()},
            "ln_eps": np.float32(eps), "z_var_eps": np.float32(mod.z_encoder.var_eps),
            "background": background[0].astype(np.float32), "typical": typical.astype(np.float32),
            "median_distance": np.float32(float(mod.guide.median_distance)), "clamp": np.float32(DEEP_CLAMP),
            "diffuse_scale": np.float32(_pstore("diffuse_scale_est").reshape(-1)[0]),
            "train_logit": train_logit, "train_index": index,
            "rank_logit": train_logit.mean(axis=0, dtype=np.float64).astype(np.float32),
            "train_ids_sha256": np.array(ids_sha256(ids)),
            "st_rows": rows.astype(np.int32), "st_indptr": block.indptr.astype(np.int64),
            "st_indices": block.indices.astype(np.int32), "st_data": block.data.astype(np.float32),
            "st_d2": np.asarray(model.adata.obsm["distance_neighbor"])[rows].astype(np.float32),
            "st_own_rows": own_rows, **reference}
    np.savez(out / "deep.npz", **deep)
    try:
        loaded = load_deep(out / "deep.npz", enc["genes"], enc=enc)
    except ValueError as exc:
        raise TrainerError("unexpected", f"the exported details were refused on reading: {exc}") from exc
    parity = deep_self_test_parts(loaded, enc, np.float64)
    worker = deep_self_test_parts(loaded, enc, np.float32)
    if not parity["max"] <= DEEP_SELFTEST_TOLERANCE:
        raise TrainerError("unexpected", f"the exported details do not reproduce the model ({parity})")
    if not worker["max"] <= DEEP_WORKER_TOLERANCE:
        raise TrainerError("unexpected", f"the exported details fail the worker's float32 self-test ({worker})")
    others = [deep_self_test(loaded, other, np.float64) for other in also]
    if others and not max(others) <= DEEP_SELFTEST_TOLERANCE:
        raise TrainerError("unexpected", f"the exported details do not reproduce the installed calculator ({max(others):.2e})")
    selftest = {"cells": int(len(rows)), "max_abs_diff": parity["max"], "tolerance": DEEP_SELFTEST_TOLERANCE,
                "parts": parity, "float32": worker, "worker_tolerance": DEEP_WORKER_TOLERANCE}
    if others:
        selftest["installed_max_abs_diff"] = max(others)
    return {"format": DEEP_FORMAT, "file": "deep.npz", "bytes": (out / "deep.npz").stat().st_size, "selftest": selftest,
            "from": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.monotonic() - t0, 2),
            "cells": int(train_logit.shape[0]), "genes": int(len(enc["genes"])), "neighbours": int(train_logit.shape[1]),
            "median_distance": float(deep["median_distance"]), "diffuse_scale": float(deep["diffuse_scale"]),
            "train_logit_abs_mean": float(np.abs(train_logit).mean())}


def installed_model(folder: Path, job: dict) -> dict:
    """The installed model a detail job adds deep.npz to: its model.json, its calculator (read strictly and
    self-tested) and the sha256 of its learning table's cell ids (cells.parquet). Refuses with detail_mismatch a model
    of another id, table or recipe than the job's (it could not be learned again the same way), with unexpected one
    that cannot be read."""
    folder = Path(folder)
    try:
        meta = json.loads((folder / "model.json").read_text())
        enc = load_encoder(folder / "encoder.npz")
        import pyarrow.parquet as pq
        ids = pq.read_table(folder / "cells.parquet", columns=["cell_id"]).column("cell_id").to_pylist()
    except Exception as exc:                          # OSError, ValueError, BadZipFile, an arrow error ...
        raise TrainerError("unexpected", f"the installed model in {folder} could not be read: "
                                         f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise TrainerError("unexpected", f"{folder / 'model.json'} does not describe a model of this app")
    parity = self_test(enc)
    if not parity <= SELFTEST_TOLERANCE:
        raise TrainerError("unexpected", f"the installed calculator fails its self-test ({parity:.2e})")
    if job.get("model_id") is not None and meta.get("id") != job["model_id"]:
        raise TrainerError("detail_mismatch", f"the installed model is {meta.get('id')}, not {job['model_id']}")
    if (meta.get("table") or {}).get("key") != job["table"]:
        raise TrainerError("detail_mismatch", f"the installed model learned from table "
                                              f"'{(meta.get('table') or {}).get('key')}', not '{job['table']}'")
    if meta.get("recipe") != job["recipe"]:
        raise TrainerError("detail_mismatch", "the installed model was learned with another recipe")
    return {"folder": folder, "meta": meta, "enc": enc, "ids_sha256": ids_sha256(ids), "cells": len(ids)}


def _write_json(path: Path, data: dict):
    """model.json, whole or not at all (a temporary file renamed over it)."""
    part = path.with_name(f".{path.name}.part-{os.getpid()}")
    part.write_text(json.dumps(data, indent=1, default=float))
    os.replace(part, path)


_RESOLVI = None


def resolvi_class():
    """scviva's ResolVI with one change, for panels of fewer than 50 genes: ResolVI.__init__
    computes its own dataset priors before it takes the ones passed in (which then replace two of them), and those
    average the n_genes // 50 lowest-expressed genes: none below 50 genes, a division by zero. Here at least one
    (low_genes). At 50 genes or more it asks for exactly n_genes // 50, the library's own number, so the model is the
    library's. Its mean and spread of the log library size (the downsampling) do not depend on it. Built once, in the
    sopa environment; setup_anndata must be called on this class (scvi keeps each model class's registrations apart)."""
    global _RESOLVI
    if _RESOLVI is None:
        from scvi import REGISTRY_KEYS
        from scviva.model import ResolVI

        class ResolVIFewGenes(ResolVI):
            def compute_dataset_dependent_priors(self, n_small_genes=None):
                if n_small_genes is None:            # the library's own n_genes // 50, at least one
                    genes = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY).shape[1]
                    n_small_genes = low_genes(genes, 50)
                return super().compute_dataset_dependent_priors(n_small_genes)
        _RESOLVI = ResolVIFewGenes
    return _RESOLVI


def fit(job: dict, report: Reporter) -> dict:
    started = time.monotonic()
    deadline = started + float(job.get("deadline_s", 1800))
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    detail_for = job.get("detail_for")
    report.log_path = None if detail_for else out / "train.log"     # a detail job writes deep.npz and model.json only
    rec = job["recipe"]
    zarr_path = Path(job["zarr"]).resolve()
    table_key = job["table"]
    others = [k for k in job.get("other_tables", []) if k != table_key]
    mixed_at, judge_min = float(job.get("mixed_at", 0.3)), int(job.get("judge_min", 100))
    seconds = {}
    # Tests only: "cpu_for_tests" learns on the CPU, and only where no GPU can be seen at all (CUDA_VISIBLE_DEVICES set
    # and empty; the engine always sets "0"), so it can never reach GPU 0 or the worker's GPU.
    cpu_test = bool(job.get("cpu_for_tests"))
    if cpu_test and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise TrainerError("gpu", "a CPU test fit needs CUDA_VISIBLE_DEVICES set and empty (no GPU visible)")
    installed = installed_model(Path(detail_for), job) if detail_for else None     # checked before anything is read

    # Read, never while a save into the sample writes it (the lock is held only for the read): the tables, and each
    # one's cell centres in the transcripts' frame (table_centres: the sample's outlines, read with spatialdata).
    report.progress("lock")
    t = time.monotonic()
    skipped = {}                                     # the other table, when it could not be used
    centres = {}                                     # how each table's cell centres were found (model.json "centres")

    def read():
        report.progress("reading", table=table_key)
        counts_, obs_, genes_, ids_, spatial = read_table(zarr_path, table_key)
        sdata, points_key = sample_frame(zarr_path, job.get("points_key"))
        extent = None                                # the engine's (its open sample's), else read here
        if points_key is not None:
            given = job.get("extent") if points_key == job.get("points_key") else None
            extent = [float(v) for v in given] if isinstance(given, list) and len(given) == 4 and all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in given) else \
                points_extent(sdata, points_key, zarr_path)
        xy_, centres["table"] = table_centres(zarr_path, table_key, obs_, spatial, sdata, points_key, extent)
        other, proseg = None, False
        for key in others:
            if (zarr_path / "tables" / key).is_dir():
                try:
                    o_counts, o_obs, o_genes, o_ids, o_spatial = read_table(zarr_path, key)
                    o_xy, centres["other"] = table_centres(zarr_path, key, o_obs, o_spatial, sdata, points_key, extent)
                    other = (key, o_counts, o_obs, o_genes, o_ids, o_xy)
                    proseg = proseg_table(zarr_path, key)
                except TrainerError as exc:          # context only: never fails the fit
                    skipped.update(table=key, why=f"it could not be used: {exc}")
                    print(f"qc_trainer: the other table '{key}' is neither pooled nor scored: {exc}", file=sys.stderr,
                          flush=True)
                break
        return counts_, obs_, genes_, ids_, xy_, other, proseg
    counts, obs, genes, gene_ids, centroids, batch, proseg = read_consistently(zarr_path, read, report)
    if batch is not None and not proseg:
        print(f"qc_trainer: the other table '{batch[0]}' holds no proseg run record: pooled into the priors, not "
              "scored (its cells are not proseg's own count)", file=sys.stderr, flush=True)
    seconds["read"] = round(time.monotonic() - t, 2)

    # Prepare, as the reference recipe does (cells >= fit_min counts with a centroid, exact kNN within each table, the
    # genes with counts and the priors pooled over both tables when the recipe says so; the learned table alone
    # otherwise).
    report.progress("preparing")
    t = time.monotonic()
    ids, area, totals = _cell_columns(obs, counts)
    finite = np.isfinite(centroids).all(axis=1)
    keep = (totals >= int(rec["fit_min_counts"])) & finite
    n_input = int(len(totals))
    if int(keep.sum()) < MIN_CELLS:
        raise TrainerError("too_few_cells", f"only {int(keep.sum())} cells have >= {rec['fit_min_counts']} counts "
                                            f"and a centroid: at least {MIN_CELLS} are needed")
    counts, ids, area, totals, xy = counts[keep], ids[keep], area[keep], totals[keep], centroids[keep]
    indices, distances2 = knn_graph(xy, int(rec["n_neighbors"]))
    partner, pooling = None, {"table": None, "why": "the recipe uses the learned table alone"}
    if rec.get("priors", "table") == "pooled":
        partner, pooling = priors_partner(batch, genes, gene_ids, rec)
        if batch is None and skipped:
            pooling = dict(skipped)
    gene_totals = np.asarray(counts.sum(axis=0)).ravel()
    if partner is not None:                          # the reference recipe drops only genes absent from BOTH tables
        gene_totals = gene_totals + np.asarray(partner[0].sum(axis=0)).ravel()
    expressed = gene_totals > 0
    counts, genes, gene_ids = counts[:, expressed], genes[expressed], gene_ids[expressed]
    parts = [(counts, distances2)] + ([(partner[0][:, expressed], partner[1])] if partner is not None else [])
    priors = pooled_priors(parts, gene_totals[expressed], rec)
    priors_from = {"rule": "pooled" if partner is not None else "table", "tables": [table_key], "cells": [int(len(ids))]}
    if partner is not None:
        priors_from["tables"].append(pooling["table"])
        priors_from["cells"].append(pooling["cells"])
    elif pooling.get("why"):
        priors_from["why"] = pooling["why"]
    if rec.get("priors") == "pooled" and partner is None:
        print(f"qc_trainer: priors from '{table_key}' alone: {pooling.get('why')}", file=sys.stderr, flush=True)
    epochs = epochs_for(counts.shape[0], rec)
    steps = epochs * math.ceil(counts.shape[0] / int(rec["batch_size"]))
    if installed is not None:                        # a detail job: what the installed model learned from, or nothing
        if ids_sha256(ids) != installed["ids_sha256"]:
            raise TrainerError("detail_mismatch", "the sample's cells are not the ones the installed model learned "
                                                  "from; nothing was changed")
        if [str(g) for g in genes] != list(installed["enc"]["genes"]):
            raise TrainerError("detail_mismatch", "the sample's genes are not the ones the installed model learned "
                                                  "from; nothing was changed")
        if installed["meta"].get("priors") != priors:
            raise TrainerError("detail_mismatch", "the priors differ from the installed model's; nothing was changed")
    seconds["prepare"] = round(time.monotonic() - t, 2)
    report.progress("preparing", cells=int(counts.shape[0]), genes=int(counts.shape[1]), epochs=epochs, steps=steps,
                    priors=priors, priors_from=priors_from, centres=centres)
    if job.get("prepare_only"):                      # tests: everything before the GPU
        return {"prepared": True, "cells": int(counts.shape[0]), "genes": int(counts.shape[1]), "epochs": epochs,
                "steps": steps, "priors": priors, "priors_from": priors_from, "input_cells": n_input,
                "knn": {"first": indices[0].tolist(), "d2": distances2[0].tolist()}, "centres": centres,
                "scored": batch[0] if batch is not None and proseg else None,
                **({"detail_for": str(installed["folder"])} if installed is not None else {})}

    # Learn (GPU 0, or Apple's GPU; a CPU test fit sees no GPU at all).
    mps = job.get("device") == "mps" and not cpu_test
    if not cpu_test and not mps:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise TrainerError("gpu", "the mixing check learns on GPU 0 only (CUDA_VISIBLE_DEVICES must be 0)")
        if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":  # else "0" is CUDA's fastest GPU, not nvidia-smi's GPU 0
            raise TrainerError("gpu", "the mixing check learns on GPU 0 only (CUDA_DEVICE_ORDER must be PCI_BUS_ID)")
    import anndata
    import torch
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if cpu_test:
        if torch.cuda.is_available():
            raise TrainerError("gpu", "a CPU test fit must see no GPU")
        device = {"name": "cpu", "pci_bus_id": None, "memory_mb": None}
    elif mps:
        from ..zarr_bridge.devices import MPS_LEARNING_TEST, apple_gpu_problem
        problem = apple_gpu_problem(torch, MPS_LEARNING_TEST)
        if problem:
            raise TrainerError("gpu", problem)
        device = apple_gpu(torch)
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise TrainerError("gpu", f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
        props = torch.cuda.get_device_properties(0)
        device = {"name": props.name, "pci_bus_id": int(getattr(props, "pci_bus_id", -1)),
                  "memory_mb": int(props.total_memory // 2 ** 20)}
    report.progress("learning", 0, epochs, device=device, cells=int(counts.shape[0]))
    import scvi
    ResolVI = resolvi_class()                        # scviva's ResolVI, its own priors from >= 1 low gene
    scvi.settings.dl_num_workers = 0
    t = time.monotonic()
    adata = anndata.AnnData(X=counts.toarray() if mps else counts)      # MPS holds no sparse tensor
    adata.obs_names = [str(i) for i in ids]
    adata.var_names = [str(g) for g in genes]
    adata.obs_names_make_unique()
    adata.obsm["X_spatial"] = xy.astype(np.float32)
    adata.obsm["index_neighbor"] = indices
    adata.obsm["distance_neighbor"] = distances2
    scvi.settings.seed = int(rec["seed"])
    torch.manual_seed(int(rec["seed"]))
    tracker = _Epochs(report, epochs, deadline)
    try:
        ResolVI.setup_anndata(adata, prepare_data=False)
        model = ResolVI(adata, semisupervised=bool(rec["semisupervised"]), n_latent=int(rec["n_latent"]),
                        n_hidden_encoder=int(rec["n_hidden_encoder"]), **priors)
        # device "auto" = the one GPU this process can see (GPU 0: checked above); scvi hands an int to Lightning
        # as a device count, so 0 would mean none.
        model.train(max_epochs=epochs, batch_size=int(rec["batch_size"]), lr=float(rec["lr"]),
                    accelerator="cpu" if cpu_test else "mps" if mps else "gpu", device="auto",
                    enable_checkpointing=False, enable_progress_bar=False, callbacks=[tracker.callback()])
        if mps:
            torch.mps.synchronize()
        elif not cpu_test:
            torch.cuda.synchronize()
    except TrainerError:
        raise
    except Exception as exc:
        if _is_oom(exc):
            raise TrainerError("oom", "the graphics card ran out of memory") from exc
        raise
    seconds["train"] = round(time.monotonic() - t, 2)
    if tracker.done != epochs:
        raise TrainerError("unexpected", f"the learning stopped after {tracker.done} of {epochs} passes")
    history = model.history.get("elbo_train") if hasattr(model, "history") else None
    curve = [float(v) for v in history.iloc[:, 0].to_numpy()] if history is not None and len(history) else tracker.curve
    settled, why = loss_settled(curve)
    if not settled:
        raise TrainerError("not_settled", f"the learning did not settle: {why}")

    # Check and export the calculator (a detail job: beside the installed one, to compare, then removed).
    report.progress("checking")
    t = time.monotonic()
    guide = model.module.guide
    encoder = guide.diffusion_encoder
    state = {k: v.detach().float().cpu().numpy() for k, v in encoder.state_dict().items()}
    if set(state) != set(ENCODER_LAYERS) | ENCODER_UNUSED:
        raise TrainerError("unexpected", f"the model's layers are not the expected ones: {sorted(state)}")
    arrays = {name: np.ascontiguousarray(state[key], dtype=np.float32) for key, name in ENCODER_LAYERS.items()}
    norms = [m for m in encoder.modules() if isinstance(m, torch.nn.LayerNorm)]
    if any(isinstance(m, (torch.nn.BatchNorm1d,)) for m in encoder.modules()) or len(norms) != 2 \
            or any(m.elementwise_affine for m in norms):
        raise TrainerError("unexpected", "the model's normalisation layers are not the expected ones")
    arrays.update(ln_eps=np.float32(norms[0].eps), var_eps=np.float32(encoder.var_eps),
                  diffusion_eps=np.float32(float(guide.diffusion_eps)))
    encoder.eval()
    dev = next(encoder.parameters()).device

    def torch_weights(rows):
        out = []
        with torch.inference_mode():
            for s in range(0, rows.shape[0], 4096):
                xb = torch.as_tensor(rows[s:s + 4096].toarray(), device=dev)
                _, w, _ = encoder(torch.log1p(xb), torch.zeros((xb.shape[0], 1), device=dev))
                w = w.clone()
                w[:, 1] += guide.diffusion_eps
                out.append(w.float().cpu().numpy())
        return np.concatenate(out)

    weights = torch_weights(counts)
    share = neighbour_share(weights.astype(np.float64)).astype(np.float32)
    # A calculator that sees no mixing anywhere is refused before anything is exported (a detail job only adds
    # the details to a model already installed: it says nothing new about the sample).
    floor_fraction = (collapse_check if installed is None else at_floor)(
        share, float(arrays["var_eps"]), float(arrays["diffusion_eps"]))
    chosen = _reference_cells(totals, share)
    reference = counts[chosen].tocsr()
    reference.sort_indices()
    enc = {"format": np.array(FORMAT), "genes": np.asarray(genes, dtype=str), "gene_ids": np.asarray(gene_ids, dtype=str),
           **arrays, "selftest_indptr": reference.indptr.astype(np.int64),
           "selftest_indices": reference.indices.astype(np.int32), "selftest_data": reference.data.astype(np.float32),
           "selftest_expected": weights[chosen].astype(np.float32)}
    enc_path = out / (DETAIL_ENCODER if installed is not None else "encoder.npz")
    np.savez(enc_path, **enc)
    try:
        loaded = load_encoder(enc_path)
        parity = self_test(loaded)
        if not parity <= SELFTEST_TOLERANCE:
            raise TrainerError("unexpected", f"the exported calculator does not reproduce the model ({parity:.2e})")
        if installed is not None:                    # the parity rule, then the details
            detail = detail_parity(installed["enc"], loaded)
            seconds["check"] = round(time.monotonic() - t, 2)
            report.progress("exporting", file="deep.npz")
            t = time.monotonic()
            try:
                deep_block = export_deep(model, out, counts, ids, totals, share, chosen, loaded, judge_min, mixed_at,
                                         "detail", also=(installed["enc"],))
            except BaseException:
                (out / "deep.npz").unlink(missing_ok=True)
                raise
    except BaseException:
        if installed is not None:
            enc_path.unlink(missing_ok=True)
        raise
    # A detail job leaves its new calculator beside deep.npz in staging: the engine compares it with the installed one
    # once more before anything is moved (engine/qc.py _check_detail), and removes staging with it.
    if installed is not None:                        # deep.npz and the installed model.json with its "deep" block
        seconds["export_deep"] = round(time.monotonic() - t, 2)
        seconds["total"] = round(time.monotonic() - started, 2)
        model_json = {k: v for k, v in installed["meta"].items() if k != "deep_error"}
        model_json["deep"] = deep_block
        model_json["detail"] = {**detail, "installed": str(installed["folder"]), "created": deep_block["created"],
                                "device": device, "epochs": epochs, "versions": _versions(), "seconds": seconds}
        model_json["provenance"] = provenance(model_json)
        _write_json(out / "model.json", model_json)
        return model_json
    seconds["check"] = round(time.monotonic() - t, 2)

    report.progress("exporting")
    t = time.monotonic()
    _write_cells(out / "cells.parquet", ids, xy, area, totals, share)
    seconds["export_cells"] = round(time.monotonic() - t, 2)
    # Go deeper's details after the calculator. A failure never leaves a wrong deep.npz: the model keeps its
    # calculator alone (the check scores as before) and model.json says why.
    report.progress("exporting", file="deep.npz")
    t = time.monotonic()
    deep_block, deep_error = None, None
    try:
        deep_block = export_deep(model, out, counts, ids, totals, share, chosen, loaded, judge_min, mixed_at, "fit")
    except Exception as exc:
        (out / "deep.npz").unlink(missing_ok=True)
        if not isinstance(exc, TrainerError):
            traceback.print_exc(file=sys.stderr)
        deep_error = {"reason": getattr(exc, "reason", "unexpected"), "message": f"{type(exc).__name__}: {exc}"[:500]}
        print(f"qc_trainer: deep.npz not written: {deep_error['message']}", file=sys.stderr, flush=True)
        report.progress("exporting", file="deep.npz", note="the details were not kept: " + deep_error["message"][:200])
    seconds["export_deep"] = round(time.monotonic() - t, 2)
    t = time.monotonic()
    learned = _per100(share, totals, mixed_at, judge_min)
    batch_summary = None
    if batch is not None and proseg:                 # the earlier proseg cells: only a table of proseg's own count
        report.progress("scoring", table=batch[0])
        key, b_counts, b_obs, b_genes, _, b_xy = batch
        b_ids, b_area, b_totals = _cell_columns(b_obs, b_counts)
        b_keep = (b_totals >= int(rec["fit_min_counts"])) & np.isfinite(b_xy).all(axis=1)
        column = {g: i for i, g in enumerate(genes)}
        mapped = np.array([column.get(g, -1) for g in b_genes])
        from scipy import sparse
        projector = sparse.csr_matrix((np.ones(int((mapped >= 0).sum()), np.float32),
                                       (np.nonzero(mapped >= 0)[0], mapped[mapped >= 0])), shape=(len(b_genes), len(genes)))
        b_model = (b_counts[b_keep] @ projector).tocsr()
        b_share = neighbour_share(mixing_weights(loaded, b_model, np, np.float32).astype(np.float64)).astype(np.float32)
        _write_cells(out / "batch.parquet", b_ids[b_keep], b_xy[b_keep], b_area[b_keep], b_totals[b_keep], b_share)
        batch_summary = {"table": key, "genes_matched": int((mapped >= 0).sum()),
                         **_per100(b_share, b_totals[b_keep], mixed_at, judge_min)}
    seconds["export"] = round(time.monotonic() - t, 2)
    seconds["total"] = round(time.monotonic() - started, 2)
    model_json = {
        "format": FORMAT, "id": job.get("model_id"), "sample": job.get("sample"),
        "zarr": str(zarr_path), "table": {"key": table_key, **(job.get("fingerprint") or {})},
        "recipe": rec, "versions": _versions(), "python": sys.executable, "device": device,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": seconds,
        "cells": {"input": n_input, "fitted": int(counts.shape[0]), "genes": int(counts.shape[1]), **learned,
                  "at_floor": round(floor_fraction, 4)},
        "thresholds": {"mixed_at": mixed_at, "judge_min": judge_min},
        "epochs": epochs, "steps": steps, "priors": priors, "priors_from": priors_from, "centres": centres, "loss": curve,
        "selftest": {"cells": int(len(chosen)), "max_abs_diff": parity, "tolerance": SELFTEST_TOLERANCE},
        "batch": batch_summary, "deep": deep_block}
    if deep_error is not None:
        model_json["deep_error"] = deep_error
    model_json["provenance"] = provenance(model_json)
    _write_json(out / "model.json", model_json)
    return model_json


def main() -> int:
    report = Reporter()
    _die_with_parent()
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))     # never Lightning's "graceful" stop and an export
    from .. import portable
    try:
        portable.nicest()                       # niceness 19 (Windows: the idle priority class)
    except OSError:
        pass
    report.send(ev="hello", pid=os.getpid(), python=sys.executable, versions=_versions(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), nice=portable.niceness(),
                threads=os.environ.get("OMP_NUM_THREADS"))
    line = sys.stdin.readline()
    try:
        job = json.loads(line)
        if not isinstance(job, dict) or not {"zarr", "table", "out_dir", "recipe"} <= set(job):
            raise ValueError("a job needs zarr, table, out_dir and recipe")
        if job.get("detail_for") is not None and (not isinstance(job["detail_for"], str) or not job["detail_for"]):
            raise ValueError("detail_for must name the installed model's folder")
    except ValueError as exc:
        report.send(ev="error", reason="unexpected", message=f"bad job: {exc}")
        return 2
    _watch_stdin()
    try:
        model = fit(job, report)
    except TrainerError as exc:
        report.send(ev="error", reason=exc.reason, message=str(exc))
        return 1
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reason = "oom" if _is_oom(exc) else "unexpected"
        report.send(ev="error", reason=reason, message=f"{type(exc).__name__}: {exc}"[:500])
        return 1
    report.send(ev="done", model=model)
    try:
        report.out.close()
    finally:
        os._exit(0)                                   # no CUDA/pyro teardown: the engine has what it needs


if __name__ == "__main__":
    raise SystemExit(main())
