"""Analysis worker: live clustering, top genes and a warm-started 2-D UMAP of the cells in a proseg-live frame, and the
mixing check (ResolVI's per-cell own / neighbour / background split).

Started by engine/analysis.py and spoken to with one JSON object per line on stdin/stdout. Runs on the GPU (cupy, cuML,
cuGraph) when that stack and a CUDA device are there, else on the CPU (analysis/cpu.py: scipy, scikit-learn, igraph,
umap-learn, numba): its hello says which (`pick_backend`; [analysis] backend in config.toml can choose). The steps, as
the GPU takes them (the CPU's own recipe for clusters and the UMAP is analysis/cpu.py's, with the same files):

  the frame's counts file (proseg's own count of each cell, gene by gene, written with the counted frame:
  analysis/counts_file.py; the request names it, "counts_file")
    -> cell x gene counts (sparse): proseg's count of the frame, averaged over its random background draw (e; the one
       switch is counts_file.LIVE_COUNT), or the count proseg saves (f, its draw) when the request names "count":
       "sampled" (a frame proseg counted for a pause, the hold, a finish or the end: counts_file.count_of), read once
       for every analysis the request asks for ("ops"). The outline the viewer draws is never counted: a frame without a
       counts file is not analysed (reason "no_counts")
  "umap":
    -> cells with >= min_counts -> library-size normalisation, log1p -> the n_top_genes most variable genes (all of a
       smaller panel) -> PCA (n_pcs components, never more than those genes or the cells allow: `principal_components`)
    -> kNN graph (cuML or scikit-learn) -> Leiden (cuGraph, or igraph started from the last clusters) -> clusters
       matched to the previous update (stable ids and colours)
    -> top genes per cluster (mean log-expression in vs out, expressed in >= 20 % of the cluster)
    -> 2-D UMAP (cuML or umap-learn) on the same kNN graph, initialised from the previous embedding so it morphs, not
       jumps; laid out afresh (then rotated onto the previous one) when the map's own settings changed
       (`layout_settings_changed`)
    -> cluster centroids and the layout's view box (2nd-98th percentile per axis), so the viewer never sorts points;
       the files and the summary are written by one writer for both backends (`Worker.finish_umap`)
  "qc" (no PCA, kNN, Leiden or UMAP, and no minimum number of cells):
    -> the model's calculator, learned once per sample by the QC trainer (encoder.npz, `load_encoder`): the run's
       genes are matched to the model's by name once per run, by gathering the first layer's weights into the run's
       gene order, so the counts above are used as they are
    -> cells with >= fit_min transcripts: log1p -> 128 -> 128 (LayerNorm, ReLU) -> 3-way softmax, + var_eps on every
       part and + diffusion_eps on the neighbour part -> each cell's neighbour share (the neighbour part / the sum)
    -> one byte per cell (floor(share x 250); 255 = not judged: < judge_min transcripts) and sums per bin of a grid
       anchored at the run's raster origin: judged and mixed now; the same cells in the frame's start block and the
       discordant pairs; depth bands; growth, tiny and shrunk cells, transcripts now and in the start block, boundary
       changes, assigned, foreground. A cell sits in the bin of its centroid in the frame's cell statistics, else of
       its starting footprint (start.npz): positions only, never a count
  "qc_start" (once per run and model): the same on the init frame's count (proseg's count at iteration 0: frame 0's
    counts file), plus each cell's starting footprint (area and centroid) on the prior's label raster (tiled runs:
    the 2-D label plane of prior.tiles; detail runs: the 3-D labels of prior.bin.zst): the chart's start, fixed
    -> every summary (frames and the start) counts all of the run's cells by depth: judged (>= judge_min transcripts of
       the model's genes), "empty" (0), "under20" (1 .. fit_min - 1), "under100" (fit_min .. judge_min - 1; the four
       add up to "ncells"), and "tx_sum" (the model's transcripts in all cells, to 0.1: e is fractional)
    -> every frame with a start line is read against its start block (the counts file's: proseg's iteration-0
       assignment, kept as proseg keeps RNA at this frame, on the init frame's rows; "start_block", at its own depth),
       and gets the like-for-like change ("matched", `matched_change`): each cell judged now and in the start block
       has both its averaged profiles scaled to the shallower of the two totals, both are scored with the frozen
       calculator, and the change in cells that look mixed per 100 is taken over those cells (on the averaged count e
       whatever count the frame is read on: a frame read on f is scored on e too, its bins and bytes on e beside the
       summary and bytes on f, `Worker.score_frame`). No draws: an unchanged
       assignment reads exactly 0, and so does transcript loss alone (every gene of a cell losing the same share) while
       the stored profiles stay proportional (the file's steps of 1/32 can leave a few hundredths per 100)
  "deeper" (Go deeper): one cell of the frame (or of the start
    line) explained with the model's details (deep.npz beside encoder.npz): how its RNA splits into its own, from
    neighbours and loose RNA, which genes go where, and which of its 10 nearest cells the neighbour part seems to come
    from. Plain numpy on the host, from 11 rows of the counts the pass already holds (`explain_cell`); with a start
    line, the cell against its own start-block row, like for like (`like_for_like`)
  "compare" (Compare; stateless: never a run's clusters, layout or colours): several
    segmentations of one sample read side by side with the sample's model, one column each: the run's frame (proseg's
    count of it, from its counts file; its like-for-like cut from proseg's sampled count f where the frame has it,
    else from Binomial(n, e / n) draws) or a counts file the sopa helper made of a shapes layer (`read_counts_file`)
    -> the area every column covers (AREA_TILE_UM tiles holding AREA_MIN_CELLS of each column's cells; a cell is in it
       when its centre is) and per column on it: cells, judged, mixed per 100 (the check's calculator), like for like
       (every judged cell cut down to exactly LIKE_DEPTH transcripts without replacement, LIKE_DRAWS draws), the
       from-neighbours histogram (HIST_EDGES), the median transcripts per cell and the share under judge_min
    -> the model's kinds (kinds.npz beside the model, made once per deep.npz: k-means of Go deeper's description of the
       model's own cells, at most KINDS_MAX kinds of KINDS_MIN_SHARE or more each, named by the top genes of each kind's
       decoded profile) and each point's nearest kind
    -> ONE 2-D UMAP of the columns' judged cells on the area (at most max_points, each column subsampled evenly) from
       their log-normalised genes of the model (PCA, neighbours, a cold UMAP as umap() lays a run out afresh; the CPU:
       umap-learn; numpy alone (tests): the first two components), split back into one points file per column (POINT)

Between requests the worker is blocked reading stdin, and it waits for the GPU by sleeping, not spinning a core. Once
it has been idle for WARM_IDLE_S it warms clusters and UMAP up, one step at a time (cuML's and cuGraph's imports, then
one tiny fit of each part; `Worker.warm_umap_steps`), so the first UMAP pass no longer pays their ~1.4 s start-up, and
a request waits for at most one step. On the CPU the libraries are loaded before the hello (cpu.enable: a library that
cannot load is named in it), then a tiny fit of each part runs the same way (umap-learn's layout is compiled at its
first call: ~3.5 s).

GPU memory: cuML switches cupy to RMM's allocator when it is first imported, and RMM's resource here does no caching
(every new array a cudaMalloc, every free a cudaFree that waits for the GPU). The frame's counts and the whole mixing
check therefore take their arrays from cupy's own pool whatever the allocator (`Worker.pooled`); the pool's freed blocks
go back before the UMAP part of a pass and after each request (`Worker.release_stale_blocks`).

Requests:
  {"op": "analyze", "ops": ["umap"] | ["qc"] | ["umap", "qc"] (default ["umap"]), "run", "seq", "iter", "counts_file"
   (the frame's counts file: proseg's own count, analysis/counts_file.py), "count"? ("averaged" | "sampled": which of
   its counts every op reads, `request_count`; default counts_file.LIVE_COUNT), "frame", "format", "nx", "ny", "nz", "x0",
   "y0", "voxel", "z_origin", "z_spacing", "transcripts", "genes", "ncells", and with a start line "init_counts_file"
   (frame 0's counts file: the start block's rows) and "start_counts_file" (the counts file holding the frame's start
   block: its own, else the newest earlier one with a block; null for none), umap: "out_json", "out_cells", "params":
   {...};  qc: "cells_file" (the frame's per-cell statistics, or null), "qc": {"model" (encoder.npz or its folder),
   "model_id", "mixed_at", "judge_min", "fit_min", "grid": {"x0", "y0", "bin", "nx", "ny"} (or "bin_um", "origin",
   "box"), "start" (the start line's start.npz, or its start.bin; null before it exists: the starting footprints),
   "out_cells", "out_cells_e"? (on a request read on f: the bytes on the averaged count e), "out_bins"? (per-bin sums,
   always on e: only when asked for), "out_json"? (the summary, written last)}}
  {"op": "qc", ...}: the same with ops ["qc"]
  {"op": "qc_start", "run", "counts_file" (frame 0's), "transcripts", "genes", "ncells", "z_origin", "z_spacing",
   "prior": {"file", "format": "tiles" | "dense", "nx", "ny", "nz", "x0", "y0", "voxel"} (the starting footprints only),
   "qc": {"model", "model_id", "mixed_at", "judge_min", "fit_min", "grid" (or "bin_um", "origin", "box"), "out_cells",
   "out_start"? (default: start.npz beside out_cells), "out_bins"?, "out_json"?}}
  "deeper" in "ops" (with "qc" and/or "umap", or alone; {"op": "deeper", ...} is the same as ops ["deeper"]):
   "cells_file" and "deeper": {"model" (the model folder: encoder.npz, deep.npz, cells.parquet), "model_id", "cell"
   (label - 1), "start" (true: the start line's cell, on the init frame's count: asked alone, the request's
   "counts_file" is frame 0's; riding on a frame's ops, its "init_counts_file" is, and none is reason "no_counts";
   centres and bytes from start.npz, "start_file" or beside "codes", else the prior's footprints: the request's
   "prior" as qc_start's, or live/info.json's), "start_line" (true: the cell against its start-block row, like for like;
   the request then names "init_counts_file" and "start_counts_file"), "k" (10), "fit_min", "judge_min", "codes" (this
   frame's .qc.bin, or start.bin, when it exists: the card's byte; null otherwise), "cell_ids" (live/cells.txt), "v"
   (1)}
  {"op": "compare", "run", "model" (the model folder: encoder.npz, deep.npz, cells.parquet), "model_id", "kinds":
   {"path" (kinds.npz: read, else made and written there), "table": {"zarr", "key"} (the model's learning table, read
   only to make kinds.npz) or "counts" (the learning cells as a counts file)}, or the path alone (or null: beside
   deep.npz), the learning table then the one model.json names, "columns": [{"key", "name", "counts" (a counts file)
   | "frame" (an analyze request's frame fields: run, counts_file, genes, ncells, cells_file; the others as sent),
   "cells"? (its cells on the area as the engine worked the area
   out: an .npy of row numbers; when every column names them, the area is the request's, said back as it came),
   "out_points"? (its points file)}], "area" (a box [x0, y0, x1, y1] the area stays inside, the engine's area {box,
   size_um, tiles, tile_um}, or null), "max_points" (COMPARE_MAX_POINTS), "seed" (0), "mixed_at", "judge_min",
   "fit_min" (or "thresholds": {...}), "out_dir" (points files <column index>.bin)?, "out_json" (the reply's
   "compare")?}
  {"op": "forget", "run"}: drop what is kept for the run (no reply)
At start (the hello): {"ok": true, "ready": true, "pci_bus_id": the GPU's PCI bus ("0000:01:00.0") or null, "backend":
  "gpu" | "cpu", "why": why not the GPU, in plain words for the Log (null on the GPU), "threads": OMP_NUM_THREADS as
  given, or null}; or, when it cannot run here, {"ok": false, "ready": false, "missing": [package names this Python
  lacks], "lasting": whether this Python answers the same for the rest of the session (missing packages, CPU libraries
  that cannot load; not a GPU that cannot be reached), "error": the reason} and an exit code of 1.
Replies (one line of compact JSON per request but "forget"): {"ok": true, "run", "seq", "summary": {...} (umap), "qc":
  {...} (qc and qc_start), "deeper": {...} (deeper: the v1 result, `explain_cell`), "compare": {...} (compare:
  `Worker.compare`; seq null)}; when an op failed, "ok": false,
  "errors": {op: "Type: message"} and "error" (the first of them), with the results of the ops that worked,
  and "reasons": {op: "too_few_cells" (too few cells hold enough transcripts to cluster: later frames of a small region
  fail alike) | "run_deleted" (its run was deleted meanwhile: one plain line in the log, no traceback) | "no_counts"
  (no counts file named, the file gone, or one that does not fit the run: one plain line in the log) | "unknown_op"
  (an op this worker does not know: answered at once, never dropped, so the engine never waits for it)} when known.
  Summaries (umap, qc, qc_start) and Go deeper's results carry "basis": "proseg" (not Go deeper's) and "count": the
  count they read (the request's, `request_count`: "averaged" or "sampled"; qc_start and the start line's answers
  counts_file.LIVE_COUNT). A qc summary read on f also carries "averaged": {judged, mixed, tx_sum, median_tx} on e.
  A summary's clusters carry "color": an index into the viewer's 40-colour palette (`assign_colours`), never shared by
  two clusters on screen.
  Summaries carry "gpu": {pool_used_mb, pool_total_mb, allocator, released_mb, pool_kept_mb} (cupy's own pool at the
  end of the pass, whether cupy's new arrays outside the mixing check still come from it, "pool", or from RMM, "rmm":
  cuML switches cupy to RMM when it is first imported (the check's own arrays come from the pool either way); what the
  pool gave back during and after the request, and its size kept between passes; null without a GPU).
Per run the worker keeps its genes, its clusters and layout, and the init frame's count (the start block's rows); it
loads no transcripts. At most RUNS_KEPT runs (CPU_RUNS_KEPT on the CPU), least recently used first out.
Files of the mixing check: out_cells = zstd frame (level 1) of u8[ncells] (cell = label - 1), on the count read;
  out_cells_e the same on e (a request read on f); out_bins = .npz, on e always
  (`write_bins`: only the bins holding a cell; with a start block also the like-for-like sums MATCHED_SUMS); out_start
  = .npz of the start's per-cell values (`START_FIELDS`); out_json = the reply's "qc" summary (qc_tables, plus kind,
  seq, iter, model_id, cells, genes, basis, count, start_block, matched, ms, ...).
Files of Compare: out_dir/<column index>.bin = POINT records (10 bytes: f32 x, f32 y, u8 kind, u8 share code), the
  column's points in its cell order, written only into a folder that exists (a column whose folder is gone has no
  points); kinds.npz beside the model's deep.npz (`write_kinds`).
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
import ctypes
import errno
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from .. import portable
from . import backend
from . import counts_file
from .counts_file import CountsError

DEFAULTS = {"min_counts": 10, "n_top_genes": 2000, "n_pcs": 30, "n_neighbors": 15, "resolution": 1.0,
            "min_dist": 0.3, "cold_epochs": 300, "warm_epochs": 60, "warm_learning_rate": 0.3, "refresh_every": 150,
            "top_genes": 3, "min_pct": 0.2}
# A clusters and UMAP summary's keys, on both backends (Worker.finish_umap writes it).
SUMMARY_KEYS = frozenset({"seq", "iter", "ncells", "nclusters", "umap_dims", "bounds", "clusters", "warm", "ms",
                          "params", "transcripts_in_cells", "transcripts", "gpu", "basis", "count"})
BASIS = "proseg"             # every summary's "basis": proseg's own count (its "count": the one read, request_count)
RADIUS = 10.0   # every layout is normalised so that 95 % of cells lie within this radius of the median
INDEX = np.dtype([("offset", "<u8"), ("length", "<u4")])         # one entry of a tile container's index
DECODE_THREADS = max(1, min(8, (os.cpu_count() or 1) // 8))     # zstd releases the GIL: tiles decode in parallel
OPS = ("analyze", "qc", "qc_start", "deeper", "compare")    # requests answered; "forget" is taken without a reply
FRAME_OPS = ("qc", "deeper", "umap")            # what an analyze request's "ops" may name, in the order they are done
RUNS_KEPT = 4                # runs whose clusters, layout and start rows the worker keeps; the least recently used goes
CPU_RUNS_KEPT = 2            # on the CPU: the run followed live and one looked at (host memory)
LOOKUP_CHUNK = 1 << 22       # transcripts per step of a lookup: its temporaries stay ~50 MB (all at once: ~0.6 GB)
WARM_IDLE_S = 0.25           # each step of the UMAP warm-up waits until the worker has been idle this long
# Cluster colours: a summary's cluster "color" is an index into the viewer's palette (frontend/src/clusters.js
# CLUSTER_PALETTE: the same 40 colours, in the same order), chosen per run by `assign_colours`, so no two clusters on
# screen share one: ids only grow as clusters come and go, and the old rule (colour = id mod 40) repeated colours.
PALETTE = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#f0e442", "#d55e00", "#999999",
           "#332288", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677", "#aa4499",
           "#661100", "#6699cc", "#aa4466", "#4477aa", "#228833", "#ee6677", "#bbbbbb", "#66ccee",
           "#aa3377", "#eecc66", "#004488", "#bb5566", "#ddaadd", "#77aadd", "#99ddff", "#44bb99",
           "#bbcc33", "#aaaa00", "#eedd88", "#ee8866", "#ffaabb", "#663399", "#aaccaa", "#dddddd")
COLOUR_GRACE_S = 5.0         # a vanished cluster's colour goes to no new cluster for this long while another is free
DISTINCT = 25.0              # colours at least this far apart (CIE76 delta E) are all distinct enough: the palette's
                             # order decides among them (its first eight are the colour-blind safe Okabe-Ito set)

# ---- the mixing check -----------------------------------------------------------------------------------------------
QC_DEFAULTS = {"mixed_at": 0.30, "judge_min": 100, "fit_min": 20}   # config.toml [qc] mixed_at, judge_min_counts, ...
SHARE_SCALE = 250            # one byte per cell: floor(neighbour share x 250), so code >= k exactly when share >= k / 250
NOT_JUDGED = 255             # ... or this: fewer than judge_min transcripts, or not scored
SOME_AT = 0.10               # "some mixing" (colours only; never a flag)
DEPTH_EDGES = (100, 250, 500, 1000)      # depth bands of judged cells: 100-249, 250-499, 500-999, >= 1,000 transcripts
GROWTH_EDGES = (0.25, 0.5, 0.67, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)   # current / starting area buckets
GROWN, SHRUNK = 1.10, 0.90   # a cell grew (area >= 1.1 x its start) or shrank (< 0.9 x)
TINY_UM2 = 10.0              # a cell smaller than this has shrunk to almost nothing
SELFTEST_TOLERANCE = 1e-4    # largest difference from the exported reference outputs a model may show here
MODELS_KEPT = 4              # calculators kept in memory (3 MB each)
# Per-bin fields of a bins file: counts of cells, then sums (transcripts, µm², shares, ...). The pairs, judged_start,
# mixed_start and tx_start read the frame's start block at its own depth; tiny_start and area_start_um2 the starting
# footprints (start.npz).
BIN_COUNTS = ("cells", "scored", "judged", "mixed", "some", "paired", "paired_mixed_now", "paired_mixed_start", "n10",
              "n01", "judged_start", "mixed_start", "growth_n", "grown", "shrunk", "lost_half", "tiny", "tiny_start")
BIN_SUMS = ("share_sum", "growth_log_sum", "tx", "tx_start", "area_um2", "area_start_um2", "changed", "assigned",
            "foreground")
START_FIELDS = ("code", "share", "totals", "area_um2", "cx", "cy")    # start.npz, one value per cell
# The calculator's parameters: our names, and the names in the torch state_dict of ResolVI's guide.diffusion_encoder.
ENCODER_LAYERS = {"W0": "encoder.fc_layers.Layer 0.0.weight", "b0": "encoder.fc_layers.Layer 0.0.bias",
                  "W1": "encoder.fc_layers.Layer 1.0.weight", "b1": "encoder.fc_layers.Layer 1.0.bias",
                  "Wv": "var_encoder.weight", "bv": "var_encoder.bias"}
GENE_KEYS = ("genes", "gene_names", "gene_symbols", "var_names", "gene_ids")   # whichever matches the run's genes best
OFFSETS = {"var_eps": 1e-3, "diffusion_eps": 1e-2, "ln_eps": 1e-5}           # ResolVI's (scviva 0.1.7) values
# Model format ids keep the app's earlier name (analysis/qc_trainer.py FORMAT says why).
FORMATS = ("proseg-live-mixing-encoder/1",)  # the QC trainer's export (analysis/qc_trainer.py FORMAT), when it names one
SELFTEST_SETS = ("selftest", "test", "reference")   # reference cells: <set>_indptr/_indices/_data (or _counts) as CSR,
                                                    # or <set>_counts as a dense [cells, genes] array
SELFTEST_OUTPUTS = ("selftest_expected", "test_weights", "test_expected", "reference_expected")
# Coverage: every summary counts all of the run's cells by depth: "empty" 0, "under20" 1 .. fit_min - 1, "under100"
# fit_min .. judge_min - 1 transcripts of the model's genes; with "judged" they add up to "ncells".
# The like-for-like change (matched_change): per-bin float sums of each compared cell's 0 or 1.
MATCHED_SUMS = ("matched_paired", "matched_mixed_now", "matched_mixed_start")
SPLITMIX = 0x9E3779B97F4A7C15                    # the thinning's hash (SplitMix64): the same draws with numpy and cupy

# ---- Go deeper -------------------------------------------------------------------------------------------------------
DEEP_FORMATS = ("proseg-live-mixing-deep/1",)    # analysis/qc_trainer.py DEEP_FORMAT
DEEP_SELFTEST_TOLERANCE = 1e-4   # the worker's float32 recipe vs torch's outputs for the export's 16 cells (the trainer
                                 # checks its float64 numpy within 1e-5; about 1-3e-6 was measured in float32)
DEEP_K = 10                      # neighbours, as the model learned them
DEEP_RULES = {"map_rounds": 100,          # MAP rounds for a cell the model never learned ("learned no")
              "same_at": 0.9,             # same kind: cos(own profile, neighbour profile) >= 0.9 ...
              "same_enter": 0.92,         # ... and a neighbour seen before keeps its class while its likeness stays
              "same_leave": 0.88,         # between these two (the request's same_prev)
              "tag_ratio": 4.0, "tag_min_tx": 2.0, "tag_share": 0.5,   # "a neighbour's gene"
              "own_share": 0.5, "own_min_tx": 2.0,                     # "its own genes" (step 13)
              "scale_tol": 1e-3, "scale_cap": 50,                      # step 9: within 0.001 transcripts, <= 50 rounds
              "rough_at": 0.10}                                        # the gene split had to move > 10 points
DEEP_TOO_FEW_GENES = 8           # genes listed (unsplit) for a cell under judge_min
DEEP_OWN_GENES = 8               # "its own genes here"
DEEP_SOURCE_GENES = 4            # per neighbour row: the genes it gives most
DEEP_NEAR_UM = 60.0              # Go deeper's neighbour search looks this near first (DeepFrame.neighbours) ...
DEEP_NEAR_MIN = 2000             # ... in frames with more candidates than this
DEEP_KIND_WORDS = (4, 3)         # what the cell's own RNA looks like; what each neighbour's looks like
DEEP_GIVER_MIN = 0.5             # a gene row names its main giving neighbour (its 6th field) from this many transcripts
DEEP_START_BAND = 5.0            # the cell against itself at the start, like for like: "about the same" within this
                                 # many points (measured on proseg's count, 2026-10-31: a followed mixed cell that did
                                 # not change says "up" or "down" in about one answer in ten at 5, one in four at 3; a
                                 # real change of 3 points or more still gets its word 70-77 % of the time)
DEEP_START_KEEP = 0.5            # a word since the start, once shown, is kept while its points stay at this part of the
                                 # band on its side (start_word)
CENTRES_KEPT = 4                 # frames whose cell centres the worker keeps for Go deeper (frame_centres)
DEEP_DIGITS = 3                  # Go deeper's gene numbers (and the neighbours'), rounded once here: a gene's transcripts
                                 # are one entry of the counts file (e, a multiple of 1/32; or f), and 3 decimals keep
                                 # every multiple of 1/32 on the whole number it rounds to (1 decimal sent 15/32 = 0.469
                                 # as 0.5, which the page read one too high); the page rounds them once more, to whole
DEEP_REPLY_MAX = 32 * 1024       # a result's compact JSON; past it the 1-transcript genes fold into "genes_rest"
DEEP_REPLY_ROOM = 64             # ... less the room its timings ("ms", "worker_ms") take, added after the split
STAT_TTL_S = 2.0                 # a model's details and the run's cell ids are looked up on disk at most this often
DEEP_Z_LAYERS = ("Z0", "zb0", "Z1", "zb1", "Zm", "zbm", "Zv", "zbv")
DEEP_D_LAYERS = ("D0", "db0", "D1", "db1", "Ds", "dbs")
DEEP_SELFTEST_KEYS = ("st_rows", "st_indptr", "st_indices", "st_data", "st_d2", "st_z", "st_pi", "st_own_rows", "st_own",
                      "st_genes", "st_nb_split")
DEEP_KEYS = frozenset({"format", "genes", *DEEP_Z_LAYERS, *DEEP_D_LAYERS, "ln_eps", "z_var_eps", "background",
                       "typical", "median_distance", "clamp", "diffuse_scale", "train_logit", "train_index",
                       "rank_logit", "train_ids_sha256", *DEEP_SELFTEST_KEYS})

# ---- Compare --------------------------------------------------------------------
COMPARE_MAX_POINTS = 60000       # points of the shared UMAP, all columns together (each column subsampled evenly)
AREA_TILE_UM = 100.0             # the area every column covers: the tiles of this size ...
AREA_MIN_CELLS = 3               # ... holding at least this many cells (outline centres) of every column
LIKE_DEPTH = 100                 # like for like: every judged cell cut down to exactly this many transcripts ...
LIKE_DRAWS = 3                   # ... drawn without replacement, this many times; the share that looks mixed averaged
HIST_EDGES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)   # the from-neighbours histogram over judged cells: 0-2, 2-5, 5-10,
                                                    # 10-20, 20-30, 30-50, 50+ % (on the byte codes, as "mixed" is)
CUT_CHUNK = 1 << 22              # transcripts per step of a cut (its temporaries stay ~0.2 GB)
DENSE_ROWS = 4096                # rows made dense at a time to meet a dense matrix (~80 MB at 5,101 genes)
KINDS_FILE = "kinds.npz"         # the model's kinds, kept beside its deep.npz
KINDS_FORMAT = "proseg-live-kinds/1"
KINDS_VERSION = 1                # the kinds' recipe: a kinds.npz of another version (or other details) is made again
KINDS_MAX = 16                   # at most this many kinds ...
KINDS_MIN_SHARE = 0.01           # ... each holding at least this share of the model's own cells (fewer kinds otherwise)
KINDS_WORDS = 3                  # a kind's name: the top genes of its decoded profile (Go deeper's kind words)
KINDS_SEED = 0                   # the kinds' k-means start: fixed, so a model's kinds are made the same every time
KINDS_LOCK_WAIT_S = 3.0          # the learning table is read only while no save into the sample writes it: a save that
                                 # holds it longer (Finish & save holds it for its whole run) leaves this op without kinds
                                 # ("busy"; a later Compare makes them), never the worker waiting for the save
KMEANS_ROUNDS = 100              # Lloyd rounds at most (they stop once no cell changes kind)
NO_KIND = 255                    # a point's kind when the model has none (no details yet) or the cell has no RNA
# Kind colours by index (the most common kind first): the cluster palette's colour-blind safe first colours, then each
# the colour farthest from those taken; every one >= 25 (CIE76) from the mixing colours (stage.js MIXING_COLORS: the
# pinks of "from neighbours" and the unjudged grey) and >= 26 from each other, and >= 3:1 against Compare's plots
# (#0b0e13: #7a7aff rather than #332288, which reached only 1.6:1 there). The one kind palette (engine/compare.py).
KIND_PALETTE = ("#0072b2", "#e69f00", "#009e73", "#56b4e9", "#f0e442", "#d55e00", "#00ff7f", "#8c564b",
                "#7a7aff", "#dddddd", "#7cfc00", "#806000", "#dbdb8d", "#2ca02c", "#ff9896", "#9467bd")
POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("kind", "u1"), ("share", "u1")])   # a points file's record: 10 bytes,
                                                    # share = the cell's byte code (floor(share x 250); 255 not judged)
# A counts file (zarr_bridge/helper.py count): a cells x genes CSR and its cells' outline centres, under these names
# (the first one present).
COUNTS_IDS = ("ids", "cell_ids", "cells", "obs_names")
COUNTS_X, COUNTS_Y = ("x", "cx"), ("y", "cy")
COUNTS_GENES = ("genes", "gene_names", "var_names")


def principal_components(n_pcs, cells: int, genes: int) -> int:
    """How many principal components a pass computes: n_pcs, but never more than the genes it uses (a panel smaller
    than n_top_genes gives all its genes) nor than its cells less one, which is all PCA can give; at least one."""
    return max(1, int(min(int(n_pcs), cells - 1, genes)))


def normalise(embedding: np.ndarray) -> np.ndarray:
    """Centre on the median and scale so the 95th-percentile radius is RADIUS: robust to stragglers, so repeated
    warm starts cannot slowly inflate the layout or let a few outliers dictate the view."""
    centre = np.median(embedding, axis=0)
    radius = float(np.percentile(np.linalg.norm(embedding - centre, axis=1), 95)) or 1.0
    return ((embedding - centre) * (RADIUS / radius)).astype(np.float32)


def rotate_to(embedding: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes (rotation or reflection, no scaling) onto the previous layout of the same cells, so the
    picture holds still between updates and only real changes move points."""
    u, _, vt = np.linalg.svd(embedding.T @ previous)
    return (embedding @ (u @ vt)).astype(np.float32)


def layout_bounds(embedding) -> dict | None:
    """The view box of a 2-D layout: the 2nd and 98th percentile of each axis over the points with finite coordinates
    (a few stragglers must not shrink the map). The viewer fits to it instead of sorting every point itself."""
    points = np.asarray(embedding, dtype=np.float64)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return {"x": [round(float(lo[0]), 3), round(float(hi[0]), 3)], "y": [round(float(lo[1]), 3), round(float(hi[1]), 3)]}


# ---- Compare's "each its own" map: every column laid out and clustered alone, turned to line up with the run's ---------
OWN_MIN_SIMILARITY = 0.5         # a column's cluster takes the run's cluster's id when their mean profiles correlate this well
OWN_MIN_LANDMARK = 10            # a kind (else a matched cluster) lines two layouts up with at least this many cells in each
NO_CLUSTER = 65535               # a point's cluster when it has none
CLUSTER = np.dtype("<u2")        # a clusters file's record: one u16 per point, in the column's points' order


def cluster_profiles(dense: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ids, mean profile of each cluster's rows of `dense` [n, genes]), the ids sorted."""
    ids = np.unique(clusters)
    return ids, np.stack([dense[clusters == c].mean(axis=0) for c in ids]) if len(ids) else np.zeros((0, dense.shape[1]))


def match_profiles(ref: np.ndarray, own: np.ndarray, minimum: float = OWN_MIN_SIMILARITY) -> dict[int, int]:
    """{row of `own`: row of `ref`}: each cluster's mean profile paired with the reference's it correlates best with,
    one to one (the pairs taken greedily from the best), a pair only at `minimum` or above."""
    if not len(ref) or not len(own):
        return {}
    a = own - own.mean(axis=1, keepdims=True)
    b = ref - ref.mean(axis=1, keepdims=True)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (a @ b.T) / np.maximum(np.outer(na, nb), 1e-12)
    pairs, used_a, used_b = {}, set(), set()
    for flat in np.argsort(-corr, axis=None):
        i, j = divmod(int(flat), corr.shape[1])
        if corr[i, j] < minimum:
            break
        if i in used_a or j in used_b:
            continue
        pairs[i] = j; used_a.add(i); used_b.add(j)
    return pairs


def align_onto(points: np.ndarray, src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """`points` turned or flipped and moved (never scaled or reshaped) so that the landmarks `src` land as near as they
    can on `dst` (weighted orthogonal Procrustes, a reflection allowed, as rotate_to). Every own map is already the same
    size (normalise); a fitted scale would shrink a map whose groups sit in another arrangement than the yardstick's,
    which independent layouts usually do."""
    w = np.asarray(weights, np.float64) / max(float(np.sum(weights)), 1e-12)
    mu_s, mu_d = w @ src, w @ dst
    cov = ((dst - mu_d) * w[:, None]).T @ (src - mu_s)
    u, _, vt = np.linalg.svd(cov)
    return ((points - mu_s) @ (u @ vt).T + mu_d).astype(np.float32)


def landmarks(a_pts, a_ids, b_pts, b_ids, pairs=None, least: int = OWN_MIN_LANDMARK):
    """The landmarks two layouts share: for each id in both (or each pair of `pairs` {a id: b id}) with at least `least`
    points on each side, the per-axis median of its points in each: (src [k, 2], dst [k, 2], weights [k])."""
    src, dst, weights = [], [], []
    for a, b in (pairs.items() if pairs is not None else ((i, i) for i in np.intersect1d(np.unique(a_ids), np.unique(b_ids)))):
        ma, mb = a_ids == a, b_ids == b
        na, nb = int(ma.sum()), int(mb.sum())
        if na >= least and nb >= least:
            src.append(np.median(a_pts[ma], axis=0)); dst.append(np.median(b_pts[mb], axis=0)); weights.append(min(na, nb))
    return np.asarray(src, np.float64).reshape(-1, 2), np.asarray(dst, np.float64).reshape(-1, 2), np.asarray(weights, np.float64)


def cluster_centroids(clusters: np.ndarray, embedding: np.ndarray) -> dict:
    """{cluster id: per-axis median of its points} over the points with finite coordinates (rows of `embedding` match
    `clusters`); one sort instead of a scan per cluster."""
    order = np.argsort(clusters, kind="stable")
    ids, starts = np.unique(clusters[order], return_index=True)
    centroids = {}
    for cid, start, end in zip(ids.tolist(), starts, np.append(starts[1:], len(order))):
        points = embedding[order[start:end]]
        points = points[np.isfinite(points).all(axis=1)]
        centroids[cid] = np.median(points, axis=0) if len(points) else None
    return centroids


def log(*parts):
    print("analysis:", *parts, file=sys.stderr, flush=True)


def _lab(colour: str) -> tuple[float, float, float]:
    """CIE L*a*b* (D65) of an sRGB hex colour."""
    rgb = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    xyz = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, 0.2126 * r + 0.7152 * g + 0.0722 * b, \
        (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = [t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116 for t in xyz]
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


PALETTE_LAB = tuple(_lab(c) for c in PALETTE)


def colour_distance(i: int, j: int) -> float:
    """How far apart palette colours i and j look (CIE76 delta E)."""
    return math.dist(PALETTE_LAB[i], PALETTE_LAB[j])


def assign_colours(run, ids, now: float) -> dict[int, int]:
    """{cluster id: palette index} for the clusters of a result (`ids`), kept in the run (`run.colours`, `run.freed`):
    no two clusters on screen share a colour while there are at most 40 of them. A cluster that stays keeps its colour;
    a vanished one frees its own; a new one (in id order) takes the free colour most distinct from those on screen
    (DISTINCT apart counts as enough: among those, one no vanished cluster freed within COLOUR_GRACE_S, since the tissue
    would show a new cluster in an old one's colour at once, then the palette's order). Two clusters on screen in
    near-twin colours are worse than a recently freed colour: distinctness decides first. Beyond 40 clusters the least
    used colours are shared. `now`: seconds on any steady clock."""
    present = set(ids)
    for cid in [c for c in run.colours if c not in present]:
        run.freed[run.colours.pop(cid)] = now
    used = [run.colours[c] for c in ids if c in run.colours]

    def score(i):
        near = min((colour_distance(i, j) for j in used), default=math.inf)
        settled = now - run.freed.get(i, -math.inf) >= COLOUR_GRACE_S
        return min(near, DISTINCT), settled, -i

    for cid in ids:
        if cid in run.colours:
            continue
        free = [i for i in range(len(PALETTE)) if i not in used]
        if free:
            colour = max(free, key=score)
        else:
            colour = min(range(len(PALETTE)), key=lambda i: (used.count(i), i))
        run.colours[cid] = colour
        run.freed.pop(colour, None)
        used.append(colour)
    return {c: run.colours[c] for c in ids}


class TooFewCells(ValueError):
    """Too few cells in the frame hold enough transcripts to be clustered (a small region): passes of later frames of the
    same region fail the same way until it holds more cells, so the reply names the reason (`reasons`) and the engine
    stops its automatic passes (engine/analysis.py)."""
    reason = "too_few_cells"


class NoCounts(ValueError):
    """The frame has no proseg count to read: the request names no counts file, or its file is gone (pruned from the
    spool, or a run made before proseg's own counts were kept). Like counts_file.CountsError (a file that does not fit
    the run), the reply names the reason and the log gets one plain line."""
    reason = "no_counts"


def run_deleted(request) -> bool:
    """The request's run folder is gone: the run was deleted while one of its frames was being analysed."""
    path = request.get("transcripts") or request.get("genes")
    return bool(path) and not Path(path).parent.parent.exists()


def failure_reason(request, exc) -> str | None:
    """Why an op failed, when it is known: a deleted run first (its files went with it), else the exception's own
    reason ("too_few_cells", "no_counts")."""
    reason = getattr(exc, "reason", None)
    if reason != "too_few_cells" and run_deleted(request):
        return "run_deleted"
    return reason


def report(request, exc=None):
    """A failed op in the worker's log: its traceback, or one plain line when its run was deleted meanwhile (an expected
    end: the files it was writing went with the run's folder) or its frame has no proseg count to read."""
    if run_deleted(request):
        log(f"run {request.get('run')} was deleted while frame {request.get('seq')} was being analysed: nothing kept")
    elif getattr(exc, "reason", None) == "no_counts":
        log(f"run {request.get('run')}: frame {request.get('seq')} has no proseg count to read: {exc}")
    else:
        log(traceback.format_exc())


def gpu_waits_block() -> bool:
    """Make every wait for the GPU sleep instead of spin. With one context per process CUDA's default scheduling
    spins the waiting CPU thread at 100 %, so each analysis pinned a core for its whole duration. Must run before the
    GPU context exists (before the first cupy allocation); later, the flag is only recorded."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        set_flags = getattr(cuda, "cuDevicePrimaryCtxSetFlags_v2", None) or cuda.cuDevicePrimaryCtxSetFlags
        device = ctypes.c_int()
        if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), 0) or set_flags(device, 0x04):  # SCHED_BLOCKING_SYNC
            raise OSError("CUDA refused the blocking-sync flag")
        return True
    except (OSError, AttributeError) as exc:
        log("GPU waits will spin:", exc)
        return False


def and_list(items) -> str:
    """Words as a sentence lists them: "a", "a and b", "a, b and c"."""
    items = [str(item) for item in items]
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def gpu_problem() -> str | None:
    """Why cupy cannot run on a CUDA device in this process, in plain words; None when it can. A device count of 0 (or
    one that raised) means no device; else a tiny sum on the device makes the context and launches a kernel, which
    catches a driver or a GPU too old for the wheels at the hello rather than at the first pass."""
    try:
        import cupy
    except Exception as exc:                              # found, but it does not load
        return f"cupy could not be loaded ({problem(exc)})"
    error = None
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        count, error = 0, problem(exc)
    if count == 0:
        return "no CUDA device is visible" + (f" ({error})" if error else
                                              f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    try:
        float(cupy.arange(4, dtype=cupy.float32).sum())
    except Exception as exc:
        return f"the GPU could not run cupy's code ({problem(exc)})"
    return None


def pick_backend(setting: str = "auto") -> dict:
    """{"backend": "gpu" | "cpu" | None, "why": in plain words for the Log (why not the GPU; or why neither),
    "missing": package names neither backend could import, "lasting": whether this Python will answer the same for the
    rest of the session}. The GPU when cupy, cuML, cuGraph and cuDF are installed (found, not imported), CUDA shows a
    device and cupy can run code on it (gpu_problem); checked in that order, the GPU's waits set to sleep
    (gpu_waits_block) before anything makes a CUDA context, as before. Else the CPU when its libraries are installed
    (backend.CPU_MODULES; found is not loadable: main() loads them next). `setting` ([analysis] backend): "cpu" never
    looks at the GPU, "gpu" never falls back to the CPU. Missing packages do not appear during a session (lasting); a
    device that cannot be reached may come back (not lasting: the engine tries again), also where the CPU's packages
    are missing (gpu_closed)."""
    why, gpu_missing = None, []
    if setting == "cpu":
        why = '[analysis] backend is "cpu"'
    else:
        gpu_missing = backend.missing(backend.GPU_MODULES)
        if gpu_missing:
            why = f"{and_list(gpu_missing)} {'is' if len(gpu_missing) == 1 else 'are'} not installed"
        else:
            gpu_waits_block()                             # before anything creates the GPU context
            why = gpu_problem()
            if why is None:
                return {"backend": "gpu", "why": None, "missing": [], "lasting": False}
    if setting == "gpu":
        return {"backend": None, "why": why, "missing": backend.distributions(gpu_missing),
                "lasting": bool(gpu_missing)}
    cpu_missing = backend.missing(backend.CPU_MODULES)
    if not cpu_missing:
        return {"backend": "cpu", "why": why, "missing": [], "lasting": False}
    return {"backend": None, "why": why, "missing": backend.distributions(cpu_missing), "lasting": gpu_closed(setting)}


def gpu_closed(setting: str) -> bool:
    """Whether the GPU stays out of this worker's reach for the session: [analysis] backend says "cpu", or a package of
    the GPU's stack is not installed. A stack that is installed may only have missed its device for now (busy, out of
    memory while another job holds it): what the CPU lacks then lasts no longer than that."""
    return setting == "cpu" or bool(backend.missing(backend.GPU_MODULES))


def cannot(choice: dict) -> str:
    """Why Clusters & UMAP cannot run in this Python (a pick_backend answer without a backend), for the hello's error:
    the packages it lacks, with why the GPU is not used either when those are the CPU's; else that reason alone (backend
    "gpu" with no device, for example). Where the GPU's stack is installed and only its device failed (not lasting), its
    reason comes first: that is what may change."""
    missing, why = choice["missing"], choice["why"]
    if not missing:
        return why or "no reason given"
    if why and not choice["lasting"]:
        return f"{why}, and it lacks {and_list(missing)} to run on the CPU"
    gpu_own = set(missing) <= set(backend.distributions(backend.GPU_MODULES))
    return f"it lacks {and_list(missing)}" + (f" (and the GPU: {why})" if why and not gpu_own else "")


def threads() -> int | None:
    """The threads the engine gives the worker (OMP_NUM_THREADS), for its hello; None when it gives none."""
    try:
        return int(os.environ.get("OMP_NUM_THREADS") or 0) or None
    except ValueError:
        return None


def read_frame(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if path.endswith(".zst"):
        from numcodecs import Zstd
        raw = Zstd().decode(raw)
    elif path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return np.frombuffer(raw, "<u4")


def read_container(path: str):
    """A proseg-live tile container (native/proseg/src/live.rs), read in one go: (header, index, bytes)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"PLT1":
        raise ValueError("not a proseg-live tile container")
    size = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8:8 + size])
    count = header["tiles_x"] * header["tiles_y"] * len(header["planes"]) + 1
    return header, np.frombuffer(raw, INDEX, count, 8 + size), raw


def plane_tiles(path: str, plane: str = "consensus", buffer: np.ndarray | None = None, pool=None):
    """The non-empty tiles of one plane of a tile container, decoded straight into a tile-major buffer (reused when it
    is big enough): (header, tile ids [n], buffer) with buffer[k] = tile ids[k]; a partial edge tile fills the top-left
    corner of its slot. Empty tiles (length 0 = all zero) are skipped: most of a whole sample's grid is empty space
    around the tissue."""
    from numcodecs import Zstd
    codec = Zstd()
    header, index, raw = read_container(path)
    nx, ny, tile, tiles_x, planes = header["nx"], header["ny"], header["tile"], header["tiles_x"], len(header["planes"])
    entries = index[header["planes"].index(plane):tiles_x * header["tiles_y"] * planes:planes]
    ids = np.nonzero(entries["length"])[0]
    if buffer is None or len(buffer) < len(ids) or buffer.shape[1:] != (tile, tile):
        buffer = np.empty((len(ids) + len(ids) // 8 + 1, tile, tile), np.uint32)      # headroom: the tissue may grow
    data = memoryview(raw)

    def decode(k):
        t = int(ids[k])
        h, w = min(tile, ny - t // tiles_x * tile), min(tile, nx - t % tiles_x * tile)
        start = int(entries[t]["offset"])
        blob = data[start:start + int(entries[t]["length"])]
        if h == tile and w == tile:
            codec.decode(blob, out=buffer[k])
        else:
            buffer[k, :h, :w] = np.frombuffer(codec.decode(blob), "<u4").reshape(h, w)

    if pool is None:
        for k in range(len(ids)):
            decode(k)
    else:
        list(pool.map(decode, range(len(ids))))
    return header, ids, buffer


def read_tiled_consensus(path: str) -> np.ndarray:
    """The whole-grid 2-D projection of a tiled frame as one (ny, nx) raster (for inspection and tests)."""
    header, ids, tiles = plane_tiles(path)
    nx, ny, tile, tiles_x = header["nx"], header["ny"], header["tile"], header["tiles_x"]
    out = np.zeros((ny, nx), np.uint32)
    for k, t in enumerate(ids.tolist()):
        r, c = t // tiles_x * tile, t % tiles_x * tile
        h, w = min(tile, ny - r), min(tile, nx - c)
        out[r:r + h, c:c + w] = tiles[k, :h, :w]
    return out


# Where each transcript sits in a frame's grid: positions only (the where engine's owners and near flags, the settings
# test's places, engine/held.py's buckets); no count is ever made from it (proseg's own count is the frame's counts
# file). A lookup is computed LOOKUP_CHUNK transcripts at a time into its outputs, so a whole sample's temporaries (three
# arrays the size of the transcripts) never sit in memory at once. `xp` is cupy or numpy.
def grid_columns(xp, x, y, x0, y0, voxel, nx, ny):
    i = xp.floor((x - x0) / voxel).astype(xp.int32)
    j = xp.floor((y - y0) / voxel).astype(xp.int32)
    return i, j, (i >= 0) & (i < nx) & (j >= 0) & (j < ny)


def tile_lookup(xp, x, y, x0, y0, voxel, nx, ny, tile, tiles_x, tiles_y, chunk: int = LOOKUP_CHUNK):
    """(tile id, offset inside the tile) of each transcript's column in a tiled plane; tile id tiles_x * tiles_y (one
    past the last tile) for transcripts outside the grid."""
    n = int(x.shape[0])
    ids, offsets = xp.empty(n, xp.int32), xp.empty(n, xp.int32)
    for start in range(0, n, chunk):
        part = slice(start, start + chunk)
        i, j, inside = grid_columns(xp, x[part], y[part], x0, y0, voxel, nx, ny)
        ids[part] = xp.where(inside, (j // tile) * tiles_x + i // tile, tiles_x * tiles_y)
        offsets[part] = xp.where(inside, (j % tile) * tile + i % tile, 0)
    return ids, offsets


def tile_labels(xp, tiles, present, ntiles, ids, offsets):
    """Each transcript's label in a tiled plane from its non-empty tiles: tiles[0] is all zero and tiles[1 + k] holds
    tile present[k]; (ids, offsets) from tile_lookup."""
    slot = xp.zeros(ntiles + 1, dtype=xp.int64)
    slot[xp.asarray(present)] = xp.arange(1, len(present) + 1, dtype=xp.int64)
    return tiles.reshape(-1)[slot[ids] * (tiles.shape[1] * tiles.shape[2]) + offsets]


def voxel_lookup(xp, x, y, z, x0, y0, voxel, nx, ny, nz, z_origin, z_spacing, chunk: int = LOOKUP_CHUNK):
    """Index of each transcript's voxel in layer-major labels [nz, ny, nx] (proseg's coords_to_voxel); nz * ny * nx
    (one past the end) for transcripts outside the grid."""
    n = int(x.shape[0])
    index = xp.empty(n, xp.int64)
    for start in range(0, n, chunk):
        part = slice(start, start + chunk)
        i, j, inside = grid_columns(xp, x[part], y[part], x0, y0, voxel, nx, ny)
        k = xp.clip(xp.floor((z[part] - z_origin) / z_spacing), 0, nz - 1).astype(xp.int64)
        index[part] = xp.where(inside, (k * ny + j) * nx + i, nz * ny * nx)
    return index


def host(array) -> np.ndarray:
    """A device (cupy) or host array as a numpy array."""
    return array.get() if hasattr(array, "get") and not isinstance(array, np.ndarray) else np.asarray(array)


def tally(xp, values, length: int, weights=None):
    """bincount of non-negative ids < length, exactly `length` long, also for no ids at all (cupy refuses an empty
    input, numpy does not): int64 counts, or float64 sums of `weights`."""
    if not int(values.size):
        return xp.zeros(length, xp.int64 if weights is None else xp.float64)
    return xp.bincount(values, weights=weights, minlength=length)[:length]


def upload(destination, source: np.ndarray):
    """Copy a host array into a slice of a device array (one transfer; a plain copy for numpy)."""
    if hasattr(destination, "set"):
        destination.set(np.ascontiguousarray(source))
    else:
        destination[...] = source


class Stopwatch:
    """Milliseconds per step, each taken once the GPU has finished it; `ms` carries earlier steps (shared ones)."""

    def __init__(self, sync, ms: dict | None = None):
        self.sync, self.ms, self.t = sync, dict(ms or {}), time.perf_counter()

    def mark(self, name: str):
        self.sync()
        now = time.perf_counter()
        self.ms[name] = round((now - self.t) * 1e3, 1)
        self.t = now


# ---- the mixing check: the calculator ------------------------------------------------------------------------------
def load_encoder(path) -> dict:
    """ResolVI's per-cell own / neighbour / background calculator as the QC trainer exports it (encoder.npz, or the
    model folder holding it; analysis/qc_trainer.py writes format "proseg-live-mixing-encoder/1"), as float32 numpy
    arrays:
      W0 [H, G], b0 [H], W1 [H, H], b1 [H], Wv [3, H], bv [3]  (torch Linear layout: out x in; the names of the torch
        state_dict of guide.diffusion_encoder, e.g. "encoder.fc_layers.Layer 0.0.weight", are accepted too)
      the model's gene names [G], under one or more of GENE_KEYS (the one matching a run's genes best is used)
      var_eps, diffusion_eps, ln_eps (scalars; ResolVI's values when absent)
      reference cells, the trainer's way: test_indptr / test_indices / test_counts (CSR of raw counts) and test_weights
        [n, 3] (the weights torch gave: softmax + var_eps, + diffusion_eps on column 1); or selftest_counts [n, G]
        (dense) and selftest_expected [n, 3] (rows summing to 1 are taken as proportions) or [n] (neighbour shares)
      format (optional): one of FORMATS
    Anything else is refused rather than guessed: an unknown format, a third hidden layer, batch norm, a LayerNorm with
    weights, a missing or misshapen array, non-finite weights, no gene names."""
    path = Path(path)
    if path.is_dir():
        path = path / "encoder.npz"
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    if "format" in data and str(data["format"]) not in FORMATS:
        raise ValueError(f"{path.name}: unknown calculator format {str(data['format'])!r} (this worker reads "
                         f"{', '.join(FORMATS)})")
    weights = {}
    for key, value in data.items():
        if key in OFFSETS or key == "layer_norm_eps":
            continue
        layer = re.search(r"fc_layers\.Layer (\d+)\.(\d+)\.", key)
        if (layer and (int(layer[1]) > 1 or int(layer[2]) != 0)) or re.fullmatch(r"[Wb][2-9]", key) \
                or re.search(r"(batch|layer)_?norm|running_(mean|var)", key, re.IGNORECASE):
            raise ValueError(f"{path.name}: unexpected array {key!r}: not ResolVI's two-layer mixing calculator")
        for name, torch_name in ENCODER_LAYERS.items():
            if key == name or key.endswith(torch_name):
                if name in weights:
                    raise ValueError(f"{path.name}: two arrays for {name}")
                weights[name] = np.asarray(value, np.float32)
    missing = [name for name in ENCODER_LAYERS if name not in weights]
    if missing:
        raise ValueError(f"{path.name}: no {', '.join(missing)}")
    if weights["W0"].ndim != 2:
        raise ValueError(f"{path.name}: W0 must be 2-D, not {weights['W0'].shape}")
    hidden, ngenes = weights["W0"].shape
    shapes = {"W0": (hidden, ngenes), "b0": (hidden,), "W1": (hidden, hidden), "b1": (hidden,), "Wv": (3, hidden),
              "bv": (3,)}
    for name, shape in shapes.items():
        if weights[name].shape != shape:
            raise ValueError(f"{path.name}: {name} has shape {weights[name].shape}, expected {shape}")
        if not np.isfinite(weights[name]).all():
            raise ValueError(f"{path.name}: {name} holds non-finite values")
    genes = {key: [str(g) for g in data[key].tolist()] for key in GENE_KEYS
             if key in data and data[key].shape == (ngenes,)}
    if not genes:
        raise ValueError(f"{path.name}: no gene names for its {ngenes} genes ({', '.join(GENE_KEYS)})")
    encoder = {**weights, "genes": genes, "path": str(path), "hidden": hidden, "ngenes": ngenes, "selftest": None}
    for name, default in OFFSETS.items():
        value = data.get(name, data.get("layer_norm_eps") if name == "ln_eps" else None)
        encoder[name] = default if value is None else float(np.asarray(value).reshape(-1)[0])
    inputs = None
    for name in SELFTEST_SETS:
        if f"{name}_indptr" in data:                  # CSR: the trainer calls its values test_counts
            indptr, indices = data[f"{name}_indptr"].astype(np.int64), data[f"{name}_indices"].astype(np.int64)
            values = data[f"{name}_data"] if f"{name}_data" in data else data[f"{name}_counts"]
            if len(indptr) < 2 or indptr[-1] != len(indices) or len(indices) != len(values) \
                    or (len(indices) and (indices.min() < 0 or indices.max() >= ngenes)):
                raise ValueError(f"{path.name}: malformed reference cells")
            inputs = np.zeros((len(indptr) - 1, ngenes), np.float32)
            inputs[np.repeat(np.arange(len(indptr) - 1), np.diff(indptr)), indices] = values
            break
        if f"{name}_counts" in data:
            inputs = data[f"{name}_counts"]
            break
    expected = next((data[k] for k in SELFTEST_OUTPUTS if k in data), None)
    if inputs is not None and expected is not None:
        inputs, expected = np.asarray(inputs, np.float32), np.asarray(expected, np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != ngenes or expected.shape not in ((len(inputs), 3), (len(inputs),)):
            raise ValueError(f"{path.name}: self-test shapes {inputs.shape} -> {expected.shape} do not fit {ngenes} genes")
        encoder["selftest"] = (inputs, expected)
    return encoder


def on_device(xp, encoder: dict) -> dict:
    """The calculator's arrays on the device (`xp`), with its offsets."""
    return {**{k: xp.asarray(encoder[k]) for k in ENCODER_LAYERS}, **{k: encoder[k] for k in OFFSETS}}


def layer_norm_relu(xp, h, eps: float):
    """torch's LayerNorm without scale or shift (biased variance), then ReLU."""
    mean = h.mean(axis=1, keepdims=True)
    centred = h - mean
    return xp.maximum(centred / xp.sqrt((centred * centred).mean(axis=1, keepdims=True) + eps), 0)


def mixing_weights(xp, h0, encoder: dict):
    """ResolVI's own / neighbour / background weights per cell (n x 3), from the first matrix product h0 = log1p(counts)
    @ W0^T (n x H, before its bias): + b0, LayerNorm + ReLU, W1, LayerNorm + ReLU, the 3-way head, softmax, + var_eps on
    every part and + diffusion_eps on the neighbour part (column 1). Float32 throughout, as torch computes it."""
    eps = encoder["ln_eps"]
    h = layer_norm_relu(xp, h0 + encoder["b0"], eps)
    h = layer_norm_relu(xp, h @ encoder["W1"].T + encoder["b1"], eps)
    z = h @ encoder["Wv"].T + encoder["bv"]
    z = xp.exp(z - z.max(axis=1, keepdims=True))
    weights = z / z.sum(axis=1, keepdims=True) + xp.float32(encoder["var_eps"])
    weights[:, 1] += xp.float32(encoder["diffusion_eps"])
    return weights


def neighbour_share(weights):
    """The share of a cell's RNA that seems to come from neighbours: the neighbour weight over the three (sum 1.013)."""
    return weights[:, 1] / weights.sum(axis=1)


def selftest_error(xp, encoder: dict, device: dict | None = None) -> float | None:
    """Largest difference between this calculator and the outputs torch gave for the model's reference cells (None
    when the export has none)."""
    if encoder.get("selftest") is None:
        return None
    counts, expected = encoder["selftest"]
    device = device or encoder
    weights = mixing_weights(xp, xp.log1p(xp.asarray(counts, dtype=xp.float32)) @ device["W0"].T, device)
    expected = xp.asarray(expected, dtype=xp.float32)
    if expected.ndim == 1:
        got = neighbour_share(weights)
    elif abs(float(expected.sum(axis=1).mean()) - 1.0) < 1e-3:      # proportions, not the raw weights (sum 1.013)
        got = weights / weights.sum(axis=1, keepdims=True)
    else:
        got = weights
    return float(xp.abs(got - expected).max())


def gene_index(encoder: dict, run_genes: list[str]) -> tuple[np.ndarray, str, int]:
    """(model gene of each run gene, -1 when the model has none; the gene-name key used; genes matched), by name,
    with whichever of the model's name lists matches the run's genes best. A run gene the model lacks is dropped; a
    model gene the run lacks counts 0 (a region's genes.txt lists only genes with modelled transcripts there)."""
    best = None
    for key, names in encoder["genes"].items():
        position = {name: i for i, name in enumerate(names)}
        index = np.fromiter((position.get(gene, -1) for gene in run_genes), np.int64, len(run_genes))
        matched = int((index >= 0).sum())
        if best is None or matched > best[2]:
            best = (index, key, matched)
    return best


def first_layer(encoder: dict, index: np.ndarray) -> np.ndarray:
    """W0^T gathered into the run's gene order [G_run, H] (zero rows for run genes the model lacks), so that
    log1p(run counts) @ it equals log1p(model counts) @ W0^T without moving any count."""
    out = np.zeros((len(index), encoder["hidden"]), np.float32)
    known = index >= 0
    out[known] = encoder["W0"].T[index[known]]
    return out


def log1p_rows(xp, counts, rows):
    """log1p of the given rows of a cell x gene count matrix (CSR on the GPU, or dense)."""
    picked = counts[rows]
    if hasattr(picked, "indices"):            # sparse: a new matrix, so the shared counts stay as they are
        picked.data = xp.log1p(picked.data)
        return picked
    return xp.log1p(picked)


def code_at(share: float) -> int:
    """The smallest byte code of a share >= `share` (codes are floor(share x 250))."""
    return int(math.ceil(share * SHARE_SCALE - 1e-9))


def share_codes(xp, share, judged):
    """One byte per cell: floor(share x 250) (0..250) for judged cells, NOT_JUDGED for the others."""
    code = xp.clip(xp.floor(xp.nan_to_num(share) * SHARE_SCALE), 0, SHARE_SCALE).astype(xp.uint8)
    return xp.where(judged, code, xp.uint8(NOT_JUDGED)).astype(xp.uint8)


def grid_bins(xp, x, y, grid: dict):
    """Each position's bin in a grid anchored at (x0, y0) with square bins of `bin` µm, row-major (by * nx + bx);
    positions beyond the edge go to the edge bin, -1 where there is no position (NaN)."""
    size, nx, ny = float(grid["bin"]), int(grid["nx"]), int(grid["ny"])
    known = xp.isfinite(x) & xp.isfinite(y)
    bx = xp.clip(xp.floor((xp.nan_to_num(x) - float(grid["x0"])) / size), 0, nx - 1).astype(xp.int64)
    by = xp.clip(xp.floor((xp.nan_to_num(y) - float(grid["y0"])) / size), 0, ny - 1).astype(xp.int64)
    return xp.where(known, by * nx + bx, -1)


def read_cellstats(path, ncells: int, voxel: float) -> dict | None:
    """A frame's per-cell statistics (frames/NNNNNN.cells.zst, native/proseg/src/live.rs: 7 x ncells little-endian
    u32/f32) as {area_um2, assigned, foreground, changed, cx, cy}; None for frames without them."""
    if not path:
        return None
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    from numcodecs import Zstd
    stats = np.frombuffer(Zstd().decode(raw), "<u4")
    if stats.size != 7 * ncells:
        raise ValueError(f"the frame's cell statistics hold {stats.size // 7} cells, the run {ncells}")
    stats = stats.reshape(7, ncells)
    return {"area_um2": (stats[0] * (voxel * voxel)).astype(np.float32), "assigned": stats[2], "foreground": stats[3],
            "changed": stats[4], "cx": stats[5].view("<f4"), "cy": stats[6].view("<f4")}


def read_centres(path, ncells: int) -> dict | None:
    """Only the centroids {cx, cy} of a frame's per-cell statistics (read_cellstats), None for frames without them."""
    if not path:
        return None
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    from numcodecs import Zstd
    stats = np.frombuffer(Zstd().decode(raw), "<u4")
    if stats.size != 7 * ncells:
        raise ValueError(f"the frame's cell statistics hold {stats.size // 7} cells, the run {ncells}")
    return {"cx": stats[5 * ncells:6 * ncells].view("<f4"), "cy": stats[6 * ncells:].view("<f4")}


def footprint(xp, raster, ncells: int, x0: float, y0: float, voxel: float):
    """Each cell's starting 2-D footprint on the prior's label raster: (area µm², centroid x, centroid y), float32,
    NaN centroids for cells without one. A tiled prior's plane is 2-D already (per column the strongest prior layer);
    a dense prior counts each column once for every label any of its layers holds. `raster` as Worker.raster gives it."""
    area = xp.zeros(ncells + 1, xp.float64)
    sx, sy = xp.zeros(ncells + 1, xp.float64), xp.zeros(ncells + 1, xp.float64)
    if raster[0] == "tiles":
        _, header, present, tiles = raster
        nx, ny, tile, tiles_x = header["nx"], header["ny"], header["tile"], header["tiles_x"]
        rows, cols = xp.meshgrid(xp.arange(tile), xp.arange(tile), indexing="ij")
        for k, t in enumerate(np.asarray(present).tolist()):                 # tile by tile: little memory on the GPU
            r0, c0 = t // tiles_x * tile, t % tiles_x * tile
            labels = tiles[1 + k]
            inside = (labels > 0) & (labels <= ncells) & (rows + r0 < ny) & (cols + c0 < nx)
            cells = labels[inside].astype(xp.int64)
            area += tally(xp, cells, ncells + 1)
            sx += tally(xp, cells, ncells + 1, (cols[inside] + (c0 + 0.5)).astype(xp.float64))
            sy += tally(xp, cells, ncells + 1, (rows[inside] + (r0 + 0.5)).astype(xp.float64))
    else:
        _, layers, nx, ny, nz = raster
        plane = nx * ny
        labels = layers[:nz * plane].reshape(nz, plane).astype(xp.int64)
        inside = (labels > 0) & (labels <= ncells)
        columns = xp.broadcast_to(xp.arange(plane, dtype=xp.int64), (nz, plane))[inside]
        keys = xp.unique(labels[inside] * plane + columns)                    # (cell, column) pairs, each once
        cells, columns = keys // plane, keys % plane
        area = tally(xp, cells, ncells + 1).astype(xp.float64)
        sx = tally(xp, cells, ncells + 1, (columns % nx + 0.5).astype(xp.float64))
        sy = tally(xp, cells, ncells + 1, (columns // nx + 0.5).astype(xp.float64))
    area, sx, sy = area[1:], sx[1:], sy[1:]
    seen = area > 0
    cx = xp.where(seen, x0 + voxel * sx / xp.maximum(area, 1), xp.nan)
    cy = xp.where(seen, y0 + voxel * sy / xp.maximum(area, 1), xp.nan)
    return (area * voxel * voxel).astype(xp.float32), cx.astype(xp.float32), cy.astype(xp.float32)


def _median(values: np.ndarray, digits: int = 1):
    return round(float(np.median(values)), digits) if values.size else None


def qc_tables(xp, share, totals, qc: dict, grid: dict, stats: dict | None = None, start: dict | None = None,
              extra: dict | None = None, hosted: dict | None = None):
    """What the mixing check keeps of one frame (or of the start), from each cell's neighbour share (NaN where not
    scored) and its transcripts of the model's genes (`totals`, proseg's count: fractional): (byte per cell, per-bin
    table, whole-run summary).

    stats: the frame's per-cell statistics (read_cellstats; at the start: the prior footprint's area_um2, cx, cy);
    start: the start, by parts, when the frame has them: "code" and "totals" (the frame's start block read at its own
    depth: the same cells at the start, judged_start / mixed_start, the discordant pairs, transcripts at the start) and
    "area_um2", "cx", "cy" (the starting footprints, start.npz: growth and tiny cells at the start, positions);
    extra: more per-cell float sums by name (the like-for-like change's MATCHED_SUMS), summed per bin and over the run;
    hosted: receives the totals' host copy made here ("totals"), for Go deeper in the same pass.
    A cell sits in the bin of its centroid in this frame, else its starting centroid (positions only: never a count).
    The table holds only bins with a cell: "bin" (row-major id) and BIN_COUNTS (int32) and BIN_SUMS (float32) as far as
    the inputs allow, judged_band / mixed_band [k, 4] (DEPTH_EDGES) and growth_hist [k, 13] (GROWTH_EDGES). Every
    per-bin sum is made in one scatter-add over a cells x fields matrix, read back once: the whole run's numbers are the
    same sums with the cells that have no position. The summary's coverage ("empty", "under20", "under100" with
    "judged": every cell once) and "tx_sum" (to 0.1) count all of the run's cells, placed or not."""
    n = int(totals.shape[0])
    start = start or {}
    paired_with, footprints = "code" in start, "area_um2" in start
    fit_min, judge_min = float(qc["fit_min"]), float(qc["judge_min"])
    mixed_code, some_code = code_at(float(qc["mixed_at"])), code_at(SOME_AT)
    scored = (totals >= fit_min) & xp.isfinite(share)
    judged = scored & (totals >= judge_min)
    empty = ~(totals > 0)
    under20 = (totals > 0) & (totals < fit_min)
    under100 = ~judged & ~empty & ~under20          # fit_min .. judge_min - 1 (and any cell not scored: none in practice)
    codes = share_codes(xp, share, judged)
    mixed = judged & (codes >= mixed_code)
    some = judged & (codes >= some_code) & ~mixed
    band = xp.searchsorted(xp.asarray(DEPTH_EDGES[1:], dtype=totals.dtype), totals, side="right")

    x = xp.full(n, xp.nan, xp.float32) if stats is None else xp.asarray(stats["cx"], xp.float32).copy()
    y = xp.full(n, xp.nan, xp.float32) if stats is None else xp.asarray(stats["cy"], xp.float32).copy()
    if "cx" in start:
        gap = ~(xp.isfinite(x) & xp.isfinite(y))
        if bool(gap.any()):
            x, y = xp.where(gap, start["cx"], x), xp.where(gap, start["cy"], y)
    bins = grid_bins(xp, x, y, grid)
    located = bins >= 0

    counts = {"cells": located, "scored": scored, "judged": judged, "mixed": mixed, "some": some}
    sums = {"share_sum": xp.where(judged, share, 0), "tx": totals}
    if paired_with:
        start_judged = start["code"] != NOT_JUDGED
        start_mixed = start_judged & (start["code"] >= mixed_code)
        paired = judged & start_judged
        counts.update(paired=paired, paired_mixed_now=paired & mixed, paired_mixed_start=paired & start_mixed,
                      n10=paired & mixed & ~start_mixed, n01=paired & ~mixed & start_mixed,
                      judged_start=start_judged, mixed_start=start_mixed)
        sums["tx_start"] = start["totals"]
    if footprints:
        counts["tiny_start"] = start["area_um2"] < TINY_UM2
        sums["area_start_um2"] = start["area_um2"]
    ratio = had = None
    if stats is not None:
        area = xp.asarray(stats["area_um2"], xp.float32)
        counts["tiny"] = area < TINY_UM2
        sums["area_um2"] = area
        sums.update({k: stats[k] for k in ("changed", "assigned", "foreground") if k in stats})
        if footprints:
            begun = start["area_um2"]
            had = begun > 0
            ratio = xp.where(had, area / xp.where(had, begun, 1), 1).astype(xp.float32)
            counts.update(growth_n=had, grown=had & (ratio >= GROWN), shrunk=had & (ratio < SHRUNK),
                          lost_half=had & (area < 0.5 * begun))
            sums["growth_log_sum"] = xp.where(had, xp.log(xp.clip(ratio, 1 / 16, 16)), 0)
    for name, values in (extra or {}).items():
        sums[name] = values
    counts = {name: counts[name] for name in BIN_COUNTS if name in counts}     # the bins file's own order
    columns = {**counts, **sums, "_empty": empty, "_under20": under20, "_under100": under100}
    for b in range(len(DEPTH_EDGES)):
        columns[f"judged_band{b}"], columns[f"mixed_band{b}"] = judged & (band == b), mixed & (band == b)
    growth_buckets = len(GROWTH_EDGES) + 1
    if ratio is not None:
        bucket = xp.searchsorted(xp.asarray(GROWTH_EDGES, dtype=xp.float32), ratio, side="right")
        for k in range(growth_buckets):
            columns[f"growth_hist{k}"] = had & (bucket == k)

    nbins = int(grid["nx"]) * int(grid["ny"])
    names = list(columns)
    where = {name: i for i, name in enumerate(names)}
    sums_of = xp.zeros((nbins + 1, len(names)), xp.float64)   # the last row: cells without a position
    xp.add.at(sums_of, xp.where(located, bins, nbins), xp.stack([xp.asarray(columns[k]).astype(xp.float64)
                                                                 for k in names], axis=1))
    sums_of = host(sums_of)
    everyone = sums_of.sum(axis=0)
    keep = np.nonzero(sums_of[:nbins, where["cells"]] > 0)[0]
    out = {"bin": keep.astype(np.int32)}
    for name in counts:
        out[name] = np.rint(sums_of[keep, where[name]]).astype(np.int32)
    for name in sums:
        out[name] = sums_of[keep, where[name]].astype(np.float32)
    for name, prefix, width in (("judged_band", "judged_band", len(DEPTH_EDGES)),
                                ("mixed_band", "mixed_band", len(DEPTH_EDGES)),
                                ("growth_hist", "growth_hist", growth_buckets if ratio is not None else 0)):
        if width:
            out[name] = np.rint(sums_of[keep][:, [where[f"{prefix}{k}"] for k in range(width)]]).astype(np.int32)

    def total(name):
        return int(round(float(everyone[where[name]])))
    totals_h, share_h = host(totals), host(share)
    if hosted is not None:
        hosted["totals"] = totals_h
    scored_h = (totals_h >= fit_min) & np.isfinite(share_h)
    judged_h = scored_h & (totals_h >= judge_min)
    j, m = total("judged"), total("mixed")
    tx = round(float(everyone[where["tx"]]), 1)
    summary = {
        "ncells": n, "located": total("cells"), "scored": total("scored"), "judged": j, "mixed": m,
        "some": total("some"), "per100": round(100.0 * m / j, 2) if j else None,
        "share_mean": round(float(everyone[where["share_sum"]]) / j, 4) if j else None,
        "median_tx": _median(totals_h[scored_h]), "median_tx_judged": _median(totals_h[judged_h]),
        "transcripts": tx, "tx_sum": tx,
        "empty": total("_empty"), "under20": total("_under20"), "under100": total("_under100"),
        "bands": {"edges": list(DEPTH_EDGES), "judged": [total(f"judged_band{b}") for b in range(len(DEPTH_EDGES))],
                  "mixed": [total(f"mixed_band{b}") for b in range(len(DEPTH_EDGES))]},
        "codes": {"scale": SHARE_SCALE, "not_judged": NOT_JUDGED, "mixed": mixed_code, "some": some_code},
        "grid": {k: grid[k] for k in ("x0", "y0", "bin", "nx", "ny")}, "bins": int(keep.size)}
    if paired_with:
        summary["paired"] = {k: total(name) for k, name in (("n", "paired"), ("mixed_now", "paired_mixed_now"),
                                                             ("mixed_start", "paired_mixed_start"), ("n10", "n10"),
                                                             ("n01", "n01"))}
    if footprints:
        summary["tiny_start"] = total("tiny_start")
    if stats is not None:
        summary["tiny"] = total("tiny")
        summary.update({k: total(k) for k in ("changed", "assigned", "foreground") if k in sums})
    if ratio is not None:
        summary["growth"] = {"n": total("growth_n"), "median": _median(host(ratio)[host(had)], 3),
                             "grown": total("grown"), "shrunk": total("shrunk"), "lost_half": total("lost_half")}
    return codes, out, summary


# ---- binomial draws: Compare's run column where its frame holds no sampled count -------------------------------------
# Each of an entry's n transcripts is kept when a SplitMix64 draw for (seed, entry, transcript) falls under its keep p:
# Binomial(n, p), the same draws on numpy, cupy and numba. Compare cuts a column to LIKE_DEPTH whole transcripts per cell,
# and proseg's averaged count is fractional: the run's column is cut from proseg's sampled count f where its frame has
# one, else from Binomial(n, e / n), a draw like proseg's own.
THIN_SOURCE = r"""
unsigned long long key = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
int kept = 0;
for (int t = 0; t < n; ++t) {
    key += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = key;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    kept += ((float)(z >> 40) * (1.0f / 16777216.0f)) < p;
}
out = (float)kept;
"""
_thin_kernel = None
CPU_THIN = None    # the CPU worker's numba twin of thin_values (analysis/cpu_kernels.py thin), set by cpu.enable()
# Go deeper's rows on the GPU (Worker.deep_fetch): one block per requested cell copies its CSR row into model-gene
# columns of a dense output (each run gene once per row, as a counts file holds them; at most one model gene per run
# gene, as the numpy path assumes); counts of genes the model lacks are summed apart (proseg's counts are multiples of
# 1/32: exact in float32).
DEEP_ROWS_SOURCE = r"""
extern "C" __global__ void sopa_companion_deep_rows(const int* indptr, const int* indices, const float* data,
        const int* rows, const int nrows, const int* gene_map, const int ngenes, float* out, float* outside) {
    const int r = blockIdx.x;
    if (r >= nrows) return;
    const int row = rows[r];
    for (int p = indptr[row] + threadIdx.x; p < indptr[row + 1]; p += blockDim.x) {
        const int g = gene_map[indices[p]];
        if (g >= 0) out[(long long)r * ngenes + g] = data[p];
        else atomicAdd(&outside[r], data[p]);
    }
}
"""
_deep_rows = None


def deep_rows_kernel(xp):
    """The compiled DEEP_ROWS_SOURCE (cupy compiles it once and keeps it in its on-disk kernel cache)."""
    global _deep_rows
    if _deep_rows is None:
        _deep_rows = xp.RawKernel(DEEP_ROWS_SOURCE, "sopa_companion_deep_rows")
    return _deep_rows


def thin_values(xp, values, p, seed: int):
    """Binomial thinning of whole counts (float32 `values`, one keep probability `p` per entry): each of an entry's
    transcripts is kept when a SplitMix64 draw for (seed, entry, transcript) falls under p, so a count n becomes
    Binomial(n, p). numpy expands the counts into transcripts; cupy runs one kernel, each thread looping over its entry's
    transcripts, and numba (the CPU worker: CPU_THIN) loops the same way. numba, cupy and numpy make the same draws as
    the kernel, bit for bit: float32 thinned values."""
    global _thin_kernel
    if xp is not np:
        if _thin_kernel is None:
            _thin_kernel = xp.ElementwiseKernel("int32 n, float32 p, uint64 seed", "float32 out", THIN_SOURCE,
                                                "sopa_companion_thin")
        return _thin_kernel(values.astype(xp.int32), p.astype(xp.float32), xp.uint64(seed))
    if CPU_THIN is not None:
        return CPU_THIN(values, p, seed)
    n = np.asarray(values).astype(np.int64).ravel()
    if not n.size:
        return np.zeros(0, np.float32)
    entry = np.repeat(np.arange(n.size, dtype=np.int64), n)
    trial = np.arange(entry.size, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)      # 0 .. n - 1 within an entry
    mix = np.uint64(SPLITMIX)
    with np.errstate(over="ignore"):
        key = (np.uint64(seed) ^ (mix * (entry.astype(np.uint64) + np.uint64(1)))) \
            + mix * (trial.astype(np.uint64) + np.uint64(1))
        z = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    kept = u < np.asarray(p, np.float32).ravel()[entry]
    return np.bincount(entry[kept], minlength=n.size).astype(np.float32)


def stack_rows(xp, blocks: list, ncols: int):
    """Count matrices of the same kind one under the other (CSR: one new CSR over the blocks' arrays)."""
    if hasattr(blocks[0], "indptr"):
        lengths = xp.concatenate([xp.diff(b.indptr) for b in blocks])
        indptr = xp.concatenate([xp.zeros(1, lengths.dtype), xp.cumsum(lengths)])
        return type(blocks[0])((xp.concatenate([b.data for b in blocks]), xp.concatenate([b.indices for b in blocks]),
                                indptr), shape=(sum(int(b.shape[0]) for b in blocks), ncols))
    return xp.concatenate(blocks, axis=0)


def score_rows(xp, rows, weights, in_model, device, inplace: bool = False):
    """(transcripts of the model's genes, neighbour share) of every row of a count matrix over the run's genes; with
    `inplace` a CSR's values become their log1p (a matrix made for this call: no copy)."""
    totals = xp.asarray(rows @ in_model).ravel().astype(xp.float32)
    if hasattr(rows, "indptr"):
        logged = rows if inplace else rows.copy()
        logged.data = xp.log1p(logged.data)
    else:
        logged = xp.log1p(rows)
    return totals, neighbour_share(mixing_weights(xp, logged @ weights, device))


def same_rows(xp, a, b):
    """Whether row i of `a` is row i of `b` bit for bit (bool per row): a CSR's the same genes with the same values, a
    dense matrix's the same row."""
    if not hasattr(a, "indptr"):
        return (a == b).all(axis=1)
    la, lb = xp.diff(a.indptr), xp.diff(b.indptr)
    same = la == lb
    nnz = int(a.data.shape[0])
    if not nnz:
        return same
    row = xp.searchsorted(xp.asarray(a.indptr[1:]), xp.arange(nnz), side="right")
    keep = same[row]
    at, r = xp.arange(nnz)[keep], row[keep]
    other = xp.asarray(b.indptr, dtype=xp.int64)[r] + (at - xp.asarray(a.indptr, dtype=xp.int64)[r])
    wrong = r[(a.indices[at] != b.indices[other]) | (a.data[at] != b.data[other])]
    if not int(wrong.shape[0]):                          # cupy's bincount refuses an empty input
        return same
    return same & (xp.bincount(wrong, minlength=int(la.shape[0])) == 0)


def scaled_rows(xp, counts, rows, factor):
    """Rows `rows` of a cell x gene count matrix (CSR, or dense with numpy alone) as a new matrix of the same kind, each
    row multiplied by its float32 `factor`."""
    picked = counts[rows]
    factor = xp.asarray(factor, dtype=xp.float32)
    if hasattr(picked, "indptr"):
        nnz = int(picked.data.shape[0])
        row = xp.searchsorted(xp.asarray(picked.indptr[1:]), xp.arange(nnz), side="right")
        picked.data = (picked.data * factor[row]).astype(xp.float32)
        return picked
    return (picked * factor[:, None]).astype(xp.float32)


# ---- the like-for-like change -------------------------------------------------------------------------------------
# A cell that loses transcripts tends to score lower whatever its boundary did, so the headline alone rewards transcript
# loss. The like-for-like change compares each cell with itself in the frame's start block (proseg's iteration-0
# assignment, kept as proseg keeps RNA at this frame): for every cell judged now and in the start block, both averaged
# profiles are scaled to the shallower of their two totals (of the model's genes), both are scored with the frozen
# calculator in one batch, and the change is per100(now) - per100(start) over those cells. No draws: an unchanged
# assignment (the start block is taken at the frame's keep) gives the same two rows, and a row alike bit for bit takes
# its now row's share (a sparse product on the GPU may score two copies of a row a few bits apart, and differently from
# one pass to the next), so it reads exactly 0 on every backend. Pure
# transcript loss (every gene of a cell losing the same share) scales both profiles to the same row as long as the stored
# profiles are proportional; the counts file keeps both in steps of 1/32 of a transcript, so a real loss can leave a few
# hundredths per 100 (a cell at the cut-off flipping either way).
def matched_change(xp, counts, totals, share, start_counts, start_totals, start_share, qc: dict, weights, in_model,
                   device):
    """The like-for-like change of a frame against its start block (see above): (per-cell sums for the bins,
    MATCHED_SUMS: 1 for each cell compared, 1 where it looks mixed now, 1 where it looked mixed at the start, at their
    shared depth; the summary {paired: the cells judged now and in the start block, used (= paired: every compared cell
    is judged at that depth), per100_now, per100_start: cells that look mixed per 100 of them at that depth, change =
    per100_now - per100_start (4 decimals, from the unrounded values), sd: None, seeds: 0 (no draws)}).
    counts / start_counts: the frame's and the start block's cell x gene averaged counts over the run's genes (the
    same columns); totals / start_totals: their transcripts of the model's genes; share / start_share: each side's own
    neighbour share (NaN where not scored)."""
    n = int(totals.shape[0])
    judge_min, mixed_code = float(qc["judge_min"]), code_at(float(qc["mixed_at"]))
    sums = {name: xp.zeros(n, xp.float64) for name in MATCHED_SUMS}
    pair = (totals >= judge_min) & xp.isfinite(share) & (start_totals >= judge_min) & xp.isfinite(start_share)
    cells = xp.nonzero(pair)[0]
    paired = int(cells.size)
    summary = {"paired": paired, "used": paired, "per100_now": None, "per100_start": None, "change": None, "sd": None,
               "seeds": 0}
    if not paired:
        return sums, summary
    now_tx, start_tx = totals[cells].astype(xp.float32), start_totals[cells].astype(xp.float32)
    depth = xp.minimum(now_tx, start_tx)
    now_rows = scaled_rows(xp, counts, cells, depth / now_tx)
    start_rows = scaled_rows(xp, start_counts, cells, depth / start_tx)
    alike = same_rows(xp, now_rows, start_rows)          # before score_rows logs the stacked copy in place
    both = stack_rows(xp, [now_rows, start_rows], int(counts.shape[1]))
    _, shares = score_rows(xp, both, weights, in_model, device, inplace=True)
    shares[paired:] = xp.where(alike, shares[:paired], shares[paired:])    # one row, one share (cuSPARSE may differ)
    codes = share_codes(xp, shares, xp.ones(2 * paired, dtype=bool))
    mixed_now, mixed_start = codes[:paired] >= mixed_code, codes[paired:] >= mixed_code
    for name, flags in zip(MATCHED_SUMS, (xp.ones(paired, dtype=bool), mixed_now, mixed_start)):
        sums[name][cells] = flags.astype(xp.float64)
    now, then = (100.0 * int(flags.sum()) / paired for flags in (mixed_now, mixed_start))
    summary.update(per100_now=round(now, 4), per100_start=round(then, 4), change=round(now - then, 4))
    return sums, summary


def write_atomic(path, data: bytes):
    temporary = f"{path}.tmp"
    Path(temporary).write_bytes(data)
    os.replace(temporary, path)


def write_codes(path, codes):
    """One byte per cell (cell = label - 1) as a zstd frame (level 1)."""
    from numcodecs import Zstd
    write_atomic(path, Zstd(level=1).encode(np.ascontiguousarray(host(codes), np.uint8).tobytes()))


def write_npz(path, arrays: dict):
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def write_bins(path, table: dict, grid: dict):
    """A bins file: the table of qc_tables plus grid [x0, y0, bin, nx, ny] (float64), depth_edges, growth_edges."""
    write_npz(path, {"grid": np.array([grid["x0"], grid["y0"], grid["bin"], grid["nx"], grid["ny"]], np.float64),
                     "depth_edges": np.array(DEPTH_EDGES, np.int32), "growth_edges": np.array(GROWTH_EDGES, np.float32),
                     **table})


def request_grid(qc: dict) -> dict:
    """The bin grid of a request's qc block: `grid` {x0, y0, bin, nx, ny}, or the run's `box` with `bin_um` (and an
    `origin`, the box's corner by default), as engine/qc.py's QcManager.fields names them."""
    if qc.get("grid"):
        return qc["grid"]
    size, box = float(qc["bin_um"]), [float(v) for v in qc["box"]]
    x0, y0 = (float(v) for v in (qc.get("origin") or box[:2]))
    return {"x0": x0, "y0": y0, "bin": size, "nx": max(1, math.ceil((box[2] - x0) / size)),
            "ny": max(1, math.ceil((box[3] - y0) / size))}


def start_file(path) -> str | None:
    """The start line's per-cell values (start.npz), also when a request names the start's byte file (start.bin)."""
    if not path:
        return None
    return str(Path(path).with_suffix(".npz")) if Path(path).suffix == ".bin" else str(path)


def slim(counts: counts_file.Counts, which: str) -> counts_file.Counts:
    """What the worker keeps of a counts file between passes: its rows (indptr, genes) and one of its arrays (`which`:
    "e" or "f", the live values, or "start_e", its start block, which needs no rows of its own), the others left empty
    (n) or None: about a third of the file's memory."""
    keep = {"e": np.zeros(0, np.float32), "f": None, "start_e": None}
    keep[which] = getattr(counts, which)
    rows = which != "start_e"
    return counts_file.Counts(ncells=counts.ncells, ngenes=counts.ngenes, iteration=counts.iteration,
                              saturated=counts.saturated, flags=counts.flags, indptr=counts.indptr,
                              genes=counts.genes if rows else np.zeros(0, np.int32), n=np.zeros(0, np.int32), **keep)


def without_zeros(arrays: counts_file.CsrArrays) -> counts_file.CsrArrays:
    """Host CSR arrays without their stored zeros (a pair proseg assigns but averages to under 1/64 of a transcript, or
    none of whose transcripts a draw kept: never a gene the cell expresses)."""
    data, indices, indptr = (np.asarray(a) for a in (arrays.data, arrays.indices, arrays.indptr))
    kept = data != 0
    if kept.all():
        return arrays
    n = int(arrays.shape[0])
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))[kept]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=n))]).astype(np.int32)
    return counts_file.CsrArrays(data[kept], indices[kept], indptr, arrays.shape)


def model_row(genes, values, gene_map: np.ndarray, ngenes: int) -> np.ndarray:
    """One cell's values over run genes `genes` as a dense float32 row over the model's genes (gene_map: each run gene's
    model gene, -1 for none; genes the model lacks are left out)."""
    out = np.zeros(int(ngenes), np.float32)
    mapped = np.asarray(gene_map)[np.asarray(genes, np.int64)]
    inside = mapped >= 0
    out[mapped[inside]] = np.asarray(values, np.float32)[inside]
    return out


def prior_request(request) -> dict | None:
    """The prior's raster as a frame request names one (format, nx, ny, nz, x0, y0, voxel, z_origin, z_spacing and
    the file as "frame"): the request's "prior" (qc_start's), else the run's live/info.json beside its transcripts,
    exactly as the engine builds qc_start's (engine/analysis.py _start_line). None when the run has no prior."""
    prior = request.get("prior") or (request.get("qc") or {}).get("prior") or (request.get("deeper") or {}).get("prior")
    z_origin, z_spacing = request.get("z_origin"), request.get("z_spacing")
    if not prior:
        live = Path(request["transcripts"]).parent
        try:
            info = json.loads((live / "info.json").read_text())
        except (OSError, ValueError):
            return None
        stored = info.get("prior") or {}
        if not stored.get("file"):
            return None
        prior = {"file": str(live / stored["file"]), "format": stored.get("format", "dense"),
                 **{k: stored[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel") if k in stored}}
        z = info.get("z") or {}
        z_origin = z_origin if z_origin is not None else z.get("grid_origin", 0.0)
        z_spacing = z_spacing if z_spacing is not None else z.get("grid_spacing", 1.0)
    return {"frame": prior["file"], "format": "tiles" if prior.get("format") == "tiles" else "dense",
            "nx": prior["nx"], "ny": prior["ny"], "nz": prior.get("nz", 1), "x0": prior["x0"], "y0": prior["y0"],
            "voxel": prior["voxel"], "z_origin": z_origin, "z_spacing": z_spacing}


# ---- Go deeper: one cell's RNA, explained -------------------------------
# ResolVI explains a cell's counts as   expected x_g = L (rho_own s_g + rho_nb sum_k pi_k s_k,g + rho_bg beta_g):
# rho the check's own three parts (the calculator, encoder.npz), s what its own kind makes (the z encoder's mean of
# log1p(x / mean x), decoded), s_k what neighbour k's kind makes (its counts normalised by ONE mean over all the
# neighbours' counts, decoded, at this cell's depth), pi how the neighbour part is shared between its 10 nearest cells
# (the guide's learned value of each (cell, neighbour) pair, carried over by Xenium cell id, minus a distance term), and
# beta the loose-RNA profile. The numpy below matches torch within 3e-6 on real frames: networks in float32, the gene
# split and its scaling in float64 (float32 sums stall the scaling at ~1e-3 transcripts in deep cells).
def load_deep(path, genes=None) -> dict:
    """The model's details for Go deeper (deep.npz beside encoder.npz, analysis/qc_trainer.py's export: DEEP_FORMATS),
    read strictly: exactly DEEP_KEYS, the format, float32 networks of consistent shapes,
    finite values, the gene order of encoder.npz (`genes`), learned values within (-10, 10), training neighbour rows in
    range, rank_logit the column means of train_logit, profiles that sum to 1, a well-formed self-test block. Anything
    else raises ValueError (refused, never guessed at)."""
    path = Path(path)
    if path.is_dir():
        path = path / "deep.npz"
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if names != DEEP_KEYS:
            raise ValueError(f"{path.name}: unexpected arrays: missing {sorted(DEEP_KEYS - names)}, "
                             f"unknown {sorted(names - DEEP_KEYS)}")
        deep = {k: data[k] for k in data.files}
    if str(deep["format"]) not in DEEP_FORMATS:
        raise ValueError(f"{path.name}: unknown format {str(deep['format'])!r} (this worker reads "
                         f"{', '.join(DEEP_FORMATS)})")
    names = [str(g) for g in deep["genes"].tolist()]
    if genes is not None and names != [str(g) for g in genes]:
        raise ValueError(f"{path.name}: its genes differ from encoder.npz's (names or order)")
    G, H, Hd = len(names), deep["Z0"].shape[0], deep["D0"].shape[0]
    n_latent = deep["Zm"].shape[0]
    if deep["train_logit"].ndim != 2 or deep["st_genes"].ndim != 2 or deep["Z0"].ndim != 2 or deep["D0"].ndim != 2:
        raise ValueError(f"{path.name}: misshapen arrays")
    N, K = deep["train_logit"].shape
    S, T = deep["st_genes"].shape
    shapes = {"Z0": (H, G), "zb0": (H,), "Z1": (H, H), "zb1": (H,), "Zm": (n_latent, H), "zbm": (n_latent,),
              "Zv": (n_latent, H), "zbv": (n_latent,), "D0": (Hd, n_latent), "db0": (Hd,), "D1": (Hd, Hd), "db1": (Hd,),
              "Ds": (G, Hd), "dbs": (G,), "ln_eps": (), "z_var_eps": (), "background": (G,), "typical": (G,),
              "median_distance": (), "clamp": (), "diffuse_scale": (), "train_logit": (N, K), "rank_logit": (K,),
              "st_d2": (S, K), "st_z": (S, K + 1, n_latent), "st_pi": (S, K),
              "st_own": (len(deep["st_own_rows"]), G), "st_nb_split": (S, T)}
    for key, shape in shapes.items():
        array = deep[key]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{path.name}: {key} is {array.dtype}{list(array.shape)}, expected float32{list(shape)}")
    for key, shape in {"train_index": (N, K), "st_rows": (S,), "st_genes": (S, T)}.items():
        if deep[key].shape != shape or deep[key].dtype != np.int32:
            raise ValueError(f"{path.name}: {key} is {deep[key].dtype}{list(deep[key].shape)}, expected int32{list(shape)}")
    if not (0 < float(deep["ln_eps"]) < 1 and 0 <= float(deep["z_var_eps"]) < 1 and float(deep["median_distance"]) > 0
            and float(deep["clamp"]) > 0 and float(deep["diffuse_scale"]) > 0):
        raise ValueError(f"{path.name}: scalars out of range")
    for key in ("background", "typical"):
        if (deep[key] < 0).any() or abs(float(deep[key].sum(dtype=np.float64)) - 1.0) > 1e-3:
            raise ValueError(f"{path.name}: {key} is not a profile summing to 1")
    if (np.abs(deep["train_logit"]) > 10.0).any():
        raise ValueError(f"{path.name}: learned neighbour values outside (-10, 10)")
    if N and (deep["train_index"].min() < 0 or deep["train_index"].max() >= N):
        raise ValueError(f"{path.name}: training neighbour rows out of range")
    if S and (deep["st_rows"].min() < 0 or deep["st_rows"].max() >= N or deep["st_genes"].min() < 0
              or deep["st_genes"].max() >= G):
        raise ValueError(f"{path.name}: self-test indices out of range")
    if N and np.abs(deep["rank_logit"] - deep["train_logit"].mean(axis=0, dtype=np.float64)).max() > 1e-5:
        raise ValueError(f"{path.name}: rank_logit is not the column mean of train_logit")
    if len(str(deep["train_ids_sha256"])) != 64:
        raise ValueError(f"{path.name}: train_ids_sha256 is not a sha256")
    indptr, indices, values = deep["st_indptr"], deep["st_indices"], deep["st_data"]
    if (indptr.dtype != np.int64 or indices.dtype != np.int32 or values.dtype != np.float32
            or len(indptr) != S * (K + 1) + 1 or indptr[0] != 0 or indptr[-1] != len(indices)
            or len(indices) != len(values) or (np.diff(indptr) < 0).any()
            or (len(indices) and (indices.min() < 0 or indices.max() >= G)) or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError(f"{path.name}: malformed self-test counts")
    own_rows = deep["st_own_rows"]
    if own_rows.dtype != np.int32 or own_rows.ndim != 1 or (len(own_rows) and (own_rows.min() < 0
                                                                                or own_rows.max() >= S)):
        raise ValueError(f"{path.name}: malformed self-test profile rows")
    deep["genes"] = names
    deep["train_ids_sha256"] = str(deep["train_ids_sha256"])
    deep["path"] = str(path)
    return deep


def _deep_ln_relu(xp, h, eps: float):
    """torch's LayerNorm without scale or shift (biased variance) over the last axis, then ReLU; any leading shape."""
    mean = h.mean(axis=-1, keepdims=True)
    centred = h - mean
    return xp.maximum(centred / xp.sqrt((centred * centred).mean(axis=-1, keepdims=True) + eps), 0)


def _deep_softmax(xp, a):
    a = xp.exp(a - a.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


def deep_z_mean(xp, deep, normalised):
    """The z encoder's mean of log1p(normalised counts) (rows x the model's genes; any leading shape)."""
    eps = float(deep["ln_eps"])
    h = _deep_ln_relu(xp, xp.log1p(normalised) @ deep["Z0"].T + deep["zb0"], eps)
    h = _deep_ln_relu(xp, h @ deep["Z1"].T + deep["zb1"], eps)
    return h @ deep["Zm"].T + deep["zbm"]


def deep_profile(xp, deep, z):
    """What a kind makes: the decoder's px_scale (a gene profile summing to 1) of latent rows z."""
    eps = float(deep["ln_eps"])
    h = _deep_ln_relu(xp, z @ deep["D0"].T + deep["db0"], eps)
    h = _deep_ln_relu(xp, h @ deep["D1"].T + deep["db1"], eps)
    return _deep_softmax(xp, h @ deep["Ds"].T + deep["dbs"])


def deep_pi(xp, deep, d2, logit):
    """The guide's neighbour shares: softmax_k(logit_k - min(sqrt(d2_k / median_distance), clamp))."""
    return _deep_softmax(xp, logit - xp.minimum(xp.sqrt(d2 / float(deep["median_distance"])), float(deep["clamp"])))


def deep_map(xp, deep, x, L, rho, s, sk, d2, rounds: int = 100, floor: float = 1e-6):
    """Neighbour shares for a cell the model never learned (3.2 step 6, "learned no"): from distance alone, then
    `rounds` MAP-EM steps under the model's Dirichlet prior alpha_k = 30 (exp(-min(diffuse_scale d2_k / median, 20))
    + 0.001), the profiles and parts fixed. Only genes with x_g > 0 enter the expected counts n_k."""
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
    """What a profile looks like: its top genes by s_g ln(s_g / typical_g) among genes with s_g >= 2 typical_g (ties
    by gene index)."""
    return deep_kind_words_rows(xp, deep, s[None], (n,))[0]


def deep_kind_words_rows(xp, deep, profiles, ns) -> list[list[int]]:
    """deep_kind_words of each row of `profiles` [R, G] (ns: how many words each), the scores of all rows worked out
    in one pass and only where a gene qualifies (s_g >= 2 typical_g): a big cell's 11 profiles cost ~0.2 ms gene by
    gene over all genes, these ~0.05 ms. The same float32 arithmetic per gene, so the same words."""
    typical = deep["typical"]
    qualifies = profiles >= 2.0 * typical
    if xp is np:                                      # numpy's 2-D nonzero took ~0.1 ms of a big cell's op: the flat
        flat = np.flatnonzero(qualifies)              # one and a division give the same pairs, row by row
        rows, genes = np.divmod(flat, qualifies.shape[1])
        s = profiles.reshape(-1)[flat]
    else:
        rows, genes = xp.nonzero(qualifies)
        s = profiles[rows, genes]
    t = typical[genes]
    score = np.asarray(s * xp.log(xp.maximum(s, 1e-30) / xp.maximum(t, 1e-30)), np.float64)
    rows, genes = np.asarray(rows), np.asarray(genes)
    bounds = np.searchsorted(rows, np.arange(len(ns) + 1))
    finite = bool(np.isfinite(score).all())
    out = []
    for r, n in enumerate(ns):
        a, b = int(bounds[r]), int(bounds[r + 1])
        g, v = genes[a:b], score[a:b]
        if not finite:
            keep = np.isfinite(v)
            g, v = g[keep], v[keep]
        out.append([int(x) for x in g[top_order(v, g, int(n))]] if len(g) else [])
    return out


def same_kind_classes(similar, prev=None, rules: dict | None = None):
    """Each neighbour of the same kind as the cell or not, from the likeness of their profiles (3.2 step 11): at first
    sight (no class before: prev None, or -1) at the cut-off, similar >= same_at (0.9); a neighbour classed before keeps
    its class until its likeness clearly crosses the cut-off: the same kind from same_enter (0.92) up, another kind below
    same_leave (0.88). A neighbour sitting at 0.9 otherwise switched class from one pass to the next, and the genes it
    gives came and went with it (17 flips in 153 frames)."""
    rules = {**DEEP_RULES, **(rules or {})}
    xp = np if isinstance(similar, (np.ndarray, list, tuple)) else _array_module(similar)
    similar = xp.asarray(similar)
    first = similar >= rules["same_at"]
    if prev is None:
        return first
    prev = xp.asarray(prev)
    kept = xp.where(similar >= rules["same_enter"], True, xp.where(similar < rules["same_leave"], False, prev > 0))
    return xp.where(prev < 0, first, kept)


def _array_module(a):
    """numpy, or cupy for a cupy array (Go deeper's recipe also runs on cupy in tests)."""
    module = type(a).__module__.split(".")[0]
    if module == "cupy":
        import cupy
        return cupy
    return np


def deep_forward(xp, deep, x, xn, d2, logit, rho, rules: dict | None = None, profiles: bool = True,
                 same_prev=None) -> dict:
    """3.2 steps 3-13 for one cell. x [G]: its counts over the model's genes; xn [K, G]: its K nearest cells' counts;
    d2 [K]: squared centre distances (µm²); logit [K]: the learned neighbour values (carried over by id, the rank's mean
    for pairs the model never saw), or None for a cell the model never learned (MAP); rho [3]: the calculator's raw
    weights (own, neighbour, loose; they sum to 1.013); same_prev [K]: each neighbour's same-kind class as last shown (1
    same, 0 another kind, -1 not seen; same_kind_classes), or None. Returns the parts, the per-gene split before and
    after scaling (genes with x_g > 0), per-neighbour amounts, tags, own genes, kind words and diagnostics."""
    rules = {**DEEP_RULES, **(rules or {})}
    L = float(x.sum())
    parts = rho / rho.sum()
    m = deep_z_mean(xp, deep, (x / x.mean())[None])[0]
    s = deep_profile(xp, deep, m)
    mk = deep_z_mean(xp, deep, xn / xn.mean())               # ONE mean over all K x G neighbour counts
    sk = deep_profile(xp, deep, mk)
    if logit is not None:
        pi = deep_pi(xp, deep, d2, logit)
    else:
        pi = deep_map(xp, deep, x, L, rho, s, sk, d2, rules["map_rounds"])
    nz = np.nonzero(np.asarray(x) > 0)[0]
    xs = x[nz]
    mu_own = rho[0] * L * s[nz]
    mu_k = (rho[1] * L) * pi[:, None] * sk[:, nz]            # [K, n]: each neighbour's kind at THIS cell's depth
    mu_nb = mu_k.sum(axis=0)
    mu_bg = rho[2] * L * deep["background"][nz]
    mu = mu_own + mu_nb + mu_bg
    A = (xs[:, None] * xp.stack([mu_own, mu_nb, mu_bg], axis=1) / mu[:, None]).astype(np.float64)   # [n, 3]: raw
    raw = A.copy()
    xs64 = xs.astype(np.float64)
    raw_gap = float(abs(A[:, 1].sum() / L - float(parts[1])))
    target = L * parts.astype(np.float64)
    rounds = 0
    cols = A.sum(axis=0)
    for rounds in range(1, int(rules["scale_cap"]) + 1):     # step 9: columns to the card's parts, rows to the genes
        A *= (target / cols)[None, :]
        A *= (xs64 / A.sum(axis=1))[:, None]
        cols = A.sum(axis=0)                                 # this round's check, and the next round's columns
        if float(xp.abs(cols - target).max()) <= rules["scale_tol"]:
            break
    col_err = float(xp.abs(cols - target).max())
    share_k = (mu_k / xp.maximum(mu_nb, 1e-30)[None, :]).astype(np.float64)
    Ak = A[:, 1][None, :] * share_k                          # [K, n]
    tx_k = Ak.sum(axis=1)
    similar = (sk @ s) / (xp.sqrt((sk * sk).sum(axis=1)) * xp.sqrt((s * s).sum()))
    same = same_kind_classes(similar, same_prev, rules)
    nb_total = float(tx_k.sum())
    same_kind_share = float(tx_k[same].sum() / nb_total) if nb_total > 0 else 0.0
    giver = Ak.argmax(axis=0)                                # the neighbour giving most of each gene
    ratio_ok = sk[giver, nz] >= rules["tag_ratio"] * s[nz]
    tag = (xs64 >= rules["tag_min_tx"]) & (A[:, 1] / xs64 >= rules["tag_share"]) & ~same[giver] & ratio_ok
    own_ok = (xs64 >= rules["own_min_tx"]) & (A[:, 0] / xs64 >= rules["own_share"])
    out = {"L": L, "rho": rho, "parts": parts, "pi": pi, "genes_nz": nz, "x_nz": xs, "A_raw": raw, "A": A, "Ak": Ak,
           "tx_k": tx_k, "similar": similar, "same": same, "same_kind_share": same_kind_share, "giver": giver,
           "tag": tag, "own_ok": own_ok, "raw_gap": raw_gap, "rough": raw_gap > rules["rough_at"],
           "scale_rounds": rounds, "scale_col_err": col_err, "m": m, "mk": mk}
    if profiles:
        words = deep_kind_words_rows(xp, deep, xp.concatenate([s[None], sk], axis=0),
                                     (DEEP_KIND_WORDS[0],) + (DEEP_KIND_WORDS[1],) * int(sk.shape[0]))
        out.update(s=s, sk=sk, kind=words[0], kind_k=words[1:])
    return out


def deep_calculator(encoder: dict, x) -> np.ndarray:
    """The check's raw weights (own, neighbour, loose) of one cell's counts over the model's genes, float32 on the host:
    the same calculator as a frame's batch (the card's number)."""
    return mixing_weights(np, np.log1p(np.asarray(x, np.float32)[None]) @ encoder["W0"].T, encoder)[0]


def calculator_shares(encoder: dict, w0t: np.ndarray, rows, genes) -> np.ndarray:
    """The frozen calculator's share of neighbours' RNA (float64) for count rows [n, len(genes)] over the model's genes
    `genes` (w0t: its first layer by gene [G, H], so only the genes held are multiplied), all in one batch."""
    return neighbour_share(mixing_weights(np, np.log1p(rows) @ w0t[genes], encoder)).astype(np.float64)


def like_for_like(encoder: dict, w0t: np.ndarray, x, x0, judge_min: float = 100) -> dict | None:
    """The cell against itself at the start, like for like (the tissue's like-for-like change, for one cell): its
    averaged count now (x) and its start-block row (x0, proseg's iteration-0 assignment kept as proseg keeps RNA now),
    both over the model's genes, scaled to the shallower of their two totals and scored by the frozen calculator in one
    batch. {points (now - start, in points of its RNA), band (what counts as up or down: DEEP_START_BAND), draws (0:
    none), now_tx, start_tx (their totals, to 0.1), now, start (the cell's share from neighbours now and at the start,
    in percent to 2 decimals, both at that matched depth: the two numbers `points` compares)}; None when either side
    holds fewer than judge_min transcripts.
    An unchanged assignment gives the same two rows, so exactly 0 points; so does pure transcript loss (every gene
    losing the same share) while the two stored rows stay proportional: the file's steps of 1/32 can leave up to a
    point or two on a small cell (under half a point for 99 cells in 100 of a synthetic test), inside DEEP_START_BAND.
    w0t: the calculator's first layer by gene [G, H] (only the genes either row holds are multiplied)."""
    x, x0 = np.asarray(x, np.float32), np.asarray(x0, np.float32)
    L, L0 = float(x.sum()), float(x0.sum())
    if L < judge_min or L0 < judge_min:
        return None
    depth = min(L, L0)
    rows = np.stack([x * np.float32(depth / L), x0 * np.float32(depth / L0)])
    genes = np.nonzero(rows.any(axis=0))[0]
    found = calculator_shares(encoder, w0t, rows[:, genes], genes)
    return {"points": round(100.0 * float(found[0] - found[1]), 2), "band": DEEP_START_BAND, "draws": 0,
            "now_tx": round(L, 1), "start_tx": round(L0, 1), "now": round(100.0 * float(found[0]), 2),
            "start": round(100.0 * float(found[1]), 2)}


def start_word(points: float, band: float, prev=None) -> tuple[str, bool]:
    """(The window's word since the start for a like-for-like change, whether it was kept): "more" or "fewer" past
    its band, else "same"; a word shown before (prev, "more" or "fewer": the engine's start_prev) is kept while the
    points stay on its side at DEEP_START_KEEP of the band or more. A followed cell whose change sits at its band
    flipped its word from one answer to the next (about one answer in four, in a test), as the
    same-kind classes did at their cut-off; the engine also holds a followed cell's word for 5 s
    (engine/deeper.py _steady_start)."""
    own = "more" if points >= band else "fewer" if points <= -band else "same"
    if own == "same" and prev == "more" and points >= DEEP_START_KEEP * band:
        return "more", True
    if own == "same" and prev == "fewer" and points <= -DEEP_START_KEEP * band:
        return "fewer", True
    return own, False


def deep_selftest_counts(deep) -> np.ndarray:
    """The self-test cells and their training neighbours as a dense float32 array [S, K + 1, G] (row 0: the cell)."""
    S, K = deep["st_d2"].shape
    G = deep["Ds"].shape[0]
    indptr, indices, values = deep["st_indptr"], deep["st_indices"], deep["st_data"]
    dense = np.zeros((S * (K + 1), G), np.float32)
    dense[np.repeat(np.arange(S * (K + 1)), np.diff(indptr)), indices] = values
    return dense.reshape(S, K + 1, G)


def deep_self_test(deep: dict, encoder: dict) -> dict:
    """Largest |this worker's float32 recipe - torch (noise off)| over the export's self-test cells, per output: the z
    means [S, K + 1, n_latent], the neighbour shares [S, K], the decoded own profiles of a few cells [P, G] and the top
    genes' observed neighbour split before scaling (A_nb / x) [S, T]; "max" the largest of them."""
    counts = deep_selftest_counts(deep)
    diff = {"z": 0.0, "pi": 0.0, "own": 0.0, "nb_split": 0.0}
    own_at = {int(r): p for p, r in enumerate(deep["st_own_rows"].tolist())}
    for i in range(counts.shape[0]):
        x, xn = counts[i, 0], counts[i, 1:]
        r = deep_forward(np, deep, x, xn, deep["st_d2"][i], deep["train_logit"][int(deep["st_rows"][i])],
                         deep_calculator(encoder, x), profiles=False)
        z = np.concatenate([r["m"][None], r["mk"]], axis=0)
        diff["z"] = max(diff["z"], float(np.abs(z - deep["st_z"][i]).max()))
        diff["pi"] = max(diff["pi"], float(np.abs(r["pi"] - deep["st_pi"][i]).max()))
        if i in own_at:
            s = deep_profile(np, deep, r["m"])
            diff["own"] = max(diff["own"], float(np.abs(s - deep["st_own"][own_at[i]]).max()))
        split = np.zeros(x.shape[0], np.float64)
        split[r["genes_nz"]] = r["A_raw"][:, 1] / r["x_nz"]
        diff["nb_split"] = max(diff["nb_split"], float(np.abs(split[deep["st_genes"][i]] - deep["st_nb_split"][i]).max()))
    diff["max"] = max(diff.values())
    return diff


_blas_control = None


@contextmanager
def no_collection():
    """A block without Python's cycle collector (it runs again after): Go deeper's op allocates ~20,000 small objects
    (its gene rows), and a collection falling inside it over the worker's ~100,000 long-lived ones paused it for 2-20 ms
    (the tail of its time per pass); refcounting still frees everything at once."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def one_blas_thread():
    """A block in which numpy's BLAS uses one thread (threadpoolctl, as the RAPIDS env ships it; a no-op without it).
    Go deeper's matrices are tiny (11 rows x the model's genes x 128), and a 4-thread BLAS stalls whenever one of its
    threads is preempted, as on a busy computer with this worker at nice 10: on a whole sample with many busy processes
    the op took p95 78 ms with 4 threads and 9 ms with one (equal, 4 ms, on an idle computer); switching costs ~3 µs."""
    global _blas_control
    if _blas_control is None:
        try:
            from threadpoolctl import ThreadpoolController
            _blas_control = ThreadpoolController()
        except Exception:                                 # not installed (the app's own venv: tests, the fake worker)
            _blas_control = False
    return _blas_control.limit(limits=1, user_api="blas") if _blas_control else nullcontext()


def ids_sha256(ids) -> str:
    """sha256 of a learning table's cell ids (cells.parquet order) joined by newlines: deep.npz train_ids_sha256."""
    return hashlib.sha256("\n".join(str(i) for i in ids).encode()).hexdigest()


def read_cell_ids(path) -> list[str]:
    """The model's learned cells (cells.parquet "cell_id", training-row order), read with pyarrow."""
    import pyarrow.parquet as pq
    return [str(v) for v in pq.read_table(path, columns=["cell_id"]).column("cell_id").to_pylist()]


class DeepFrame:
    """One frame (or the start line) as Go deeper searches it (3.2 step 2): each cell's transcripts of the model's genes
    (proseg's count) and its centre (the frame's cell statistics, the starting footprints where those are NaN; the start
    footprint's for the start line: positions only). A neighbour candidate holds >= fit_min transcripts and has a finite
    centre; the cell itself never is."""

    def __init__(self, totals, centres, fit_min: float, k: int = DEEP_K):
        self.totals = np.asarray(totals)
        self.centres = np.asarray(centres, np.float64)
        self.k = int(k)
        x, y = self.centres[:, 0], self.centres[:, 1]
        self.cand = np.flatnonzero((self.totals >= fit_min) & np.isfinite(x) & np.isfinite(y))
        self.cand_x, self.cand_y = x[self.cand], y[self.cand]

    def neighbours(self, c: int):
        """The k nearest candidates of cell c by squared distance, ties by index: (cells [<= k], d2 [<= k]). In a big
        frame it looks within DEEP_NEAR_UM of the cell first: when at least k candidates lie that close, the k nearest
        are among them, so the answer is the same as over all candidates (a whole sample: ~0.2 ms less per cell)."""
        cand = self.cand
        dx, dy = self.cand_x - self.centres[c, 0], self.cand_y - self.centres[c, 1]
        if len(cand) > DEEP_NEAR_MIN:
            near = np.flatnonzero((np.abs(dx) <= DEEP_NEAR_UM) & (np.abs(dy) <= DEEP_NEAR_UM))
            nx, ny = dx[near], dy[near]
            d2 = nx * nx + ny * ny
            inside = (d2 <= DEEP_NEAR_UM * DEEP_NEAR_UM) & (cand[near] != c)
            if int(inside.sum()) >= self.k:
                near, d2 = near[inside], d2[inside]
                order = np.lexsort((cand[near], d2))[:self.k]
                return cand[near][order], d2[order]
        d2 = dx * dx + dy * dy
        at = int(np.searchsorted(cand, c))
        if at < len(cand) and cand[at] == c:
            d2[at] = np.inf
        if len(cand) > self.k:
            part = np.argpartition(d2, self.k - 1)[:self.k]
            part = np.nonzero(d2 <= d2[part].max())[0]       # ties at the edge: decided by index below
        else:
            part = np.arange(len(cand))
        part = part[np.isfinite(d2[part])]
        order = np.lexsort((cand[part], d2[part]))[:self.k]
        return cand[part][order], d2[part][order]


def learned_logits(run_row: np.ndarray, deep: dict, c: int, nbrs: np.ndarray):
    """(logits [k] or None, known [k]): the model's learned value of each (cell, neighbour) pair, carried over by Xenium
    cell id (3.3); a pair the model never saw gets the learned mean of its rank; None when the cell itself was never
    learned (its shares are then re-estimated: MAP)."""
    r = int(run_row[c])
    if r < 0:
        return None, np.zeros(len(nbrs), bool)
    rows = run_row[nbrs]
    match = (deep["train_index"][r][None, :] == rows[:, None]) & (rows[:, None] >= 0)     # [k, slots]
    known = match.any(axis=1)
    slot = match.argmax(axis=1)
    logits = np.where(known, deep["train_logit"][r][slot], deep["rank_logit"][:len(nbrs)])
    return logits.astype(np.float32), known


def _row_bytes(row) -> int:
    """A gene row's compact JSON length, ["ACTB",39.2,7.9,31.1,1], worked out without encoding it when it is the usual
    plain row (a name without characters JSON escapes, a finite number, two finite floats, an int: what json.dumps
    writes as str/repr), else by json.dumps itself."""
    if type(row) is list and len(row) in (5, 6):
        name, tx, own, nb, tag = row[:5]
        giver = row[5] if len(row) == 6 else 0
        if type(name) is str and type(tx) in (int, float) and type(own) is float and type(nb) is float \
                and type(tag) is int and type(giver) is int and name.isascii() and name.isprintable() \
                and '"' not in name and "\\" not in name and math.isfinite(tx) and math.isfinite(own) \
                and math.isfinite(nb):
            extra = len(str(giver)) + 1 if len(row) == 6 else 0
            return len(name) + len(repr(tx)) + len(repr(own)) + len(repr(nb)) + len(str(tag)) + 8 + extra
    return len(json.dumps(row, separators=(",", ":")))


def fit_reply(result: dict, cap: int = DEEP_REPLY_MAX - DEEP_REPLY_ROOM) -> dict:
    """A result within `cap` bytes of compact JSON: past it, the genes with the fewest transcripts (the last rows) fold
    into "genes_rest": [genes, transcripts, its own, from neighbours] (sums, to DEEP_DIGITS decimals), so the split
    still adds up."""
    genes = result.get("genes") or []
    if not genes:
        return result
    size = len(json.dumps(result, separators=(",", ":")))
    if size <= cap:
        return result
    keep, over = len(genes), size - cap + 64             # room for "genes_rest"
    while keep > 1 and over > 0:                         # only the rows that go are measured (a big cell's reply:
        keep -= 1                                        # ~2 ms less than measuring all of its ~1,400 rows)
        over -= _row_bytes(genes[keep]) + 1
    rest = genes[keep:]
    result["genes"] = genes[:keep]          # (the cap keeps DEEP_REPLY_ROOM free for the timings added afterwards)
    result["genes_rest"] = [len(rest), round(sum(r[1] for r in rest), DEEP_DIGITS),
                            round(sum(r[2] for r in rest), DEEP_DIGITS), round(sum(r[3] for r in rest), DEEP_DIGITS)]
    return result


_NAME_BYTES: dict = {}            # id(a model's genes array) -> (that array, each name's JSON length): Go deeper's rows


def top_order(values: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """np.lexsort((keys, -values))[:k] (the k largest values, ties by key) without sorting all of them: a neighbour's
    top genes among a big cell's ~2,000 (10 full sorts were ~1 ms of the op)."""
    n = len(values)
    if n <= k:
        return np.lexsort((keys, -values))
    part = np.argpartition(-values, k - 1)[:k]
    idx = np.nonzero(values >= values[part].min())[0]    # every value tied at the edge, decided by key below
    return idx[np.lexsort((keys[idx], -values[idx]))][:k]


def top_orders(values: np.ndarray, keys: np.ndarray, k: int) -> list[np.ndarray]:
    """top_order of each row of values [R, n] (keys [n], shared by the rows) with one partition for all rows: the
    neighbours' top genes (10 calls of top_order were ~0.15 ms of a big cell's op). A row with a tie at its edge (more
    than k values at or above its k-th largest), and any row when a value is not finite, takes top_order itself, so
    every answer is top_order's."""
    values = np.asarray(values)
    R, n = values.shape
    if n <= k or R == 0 or not np.isfinite(values).all():
        return [top_order(values[r], keys, k) for r in range(R)]
    part = np.argpartition(-values, k - 1, axis=1)[:, :k]             # each row's k largest, in no order
    picked = np.take_along_axis(values, part, axis=1)
    ties = (values >= picked.min(axis=1)[:, None]).sum(axis=1) > k
    tops = np.take_along_axis(part, np.lexsort((keys[part], -picked), axis=-1), axis=1)
    return [top_order(values[r], keys, k) if ties[r] else tops[r] for r in range(R)]


def name_bytes(genes) -> np.ndarray:
    """Each gene name's JSON length (a reply row's first field), worked out once per gene list (a model's: ~3 ms for
    5,000 names, done when the model's details are loaded, never inside a click)."""
    cached = _NAME_BYTES.get(id(genes))
    if cached is None or cached[0] is not genes:
        cached = (genes, np.fromiter((len(json.dumps(str(g))) for g in genes), np.int64, len(genes)))
        _NAME_BYTES[id(genes)] = cached
        while len(_NAME_BYTES) > MODELS_KEPT + 2:
            _NAME_BYTES.pop(next(iter(_NAME_BYTES)))
    return cached[1]


def gene_names(genes, idx) -> list[str]:
    """The names of genes `idx` (a model's gene list, load_deep's list of str; or any sequence of names): Python ints
    into a list (a big cell's ~1,400 numpy-scalar lookups and str() calls were ~0.1 ms of the op)."""
    idx = idx.tolist() if hasattr(idx, "tolist") else list(idx)
    if type(genes) is list:
        if _STR_LISTS.get(id(genes)) is not genes:      # checked once per list: a model's (load_deep) is all str
            if not all(type(g) is str for g in genes):
                return [str(genes[g]) for g in idx]
            _STR_LISTS[id(genes)] = genes
            while len(_STR_LISTS) > MODELS_KEPT + 2:
                _STR_LISTS.pop(next(iter(_STR_LISTS)))
        return list(map(genes.__getitem__, idx))
    return [str(genes[g]) for g in idx]


_STR_LISTS: dict = {}             # id(a gene list) -> that list, once seen to hold only str (gene_names)


_POWERS = 10.0 ** np.arange(1, 10)


def _decimal_digits(v: np.ndarray) -> np.ndarray:
    """Digits of the whole part of non-negative numbers below 1e10 (exact comparisons, no logarithms): one plus how many
    of 10, 100, ..., 1e9 it reaches (one search, not nine passes)."""
    return np.searchsorted(_POWERS, np.asarray(v, np.float64), side="right").astype(np.int64) + 1


def _amount_bytes(v: np.ndarray) -> np.ndarray:
    """How long Python writes each non-negative number below 1e9 sent at DEEP_DIGITS decimals (repr: its whole part, a
    point and its decimals without the trailing zeros, at least one: "2.469", "2.5", "3.0")."""
    scale = 10 ** DEEP_DIGITS
    k = np.rint(np.asarray(v, np.float64) * scale).astype(np.int64)
    whole, frac = k // scale, k % scale
    decimals = np.ones(k.shape, np.int64)
    for d in range(1, DEEP_DIGITS + 1):                  # d decimals when the d-th one is the last that is not 0
        decimals = np.where(frac % 10 ** (DEEP_DIGITS - d + 1) != 0, d, decimals)
    return _decimal_digits(whole.astype(np.float64)) + 1 + decimals


def fitted_gene_rows(result: dict, genes, g_idx, tx, own, nb, tag, cap: int = DEEP_REPLY_MAX - DEEP_REPLY_ROOM,
                     giver=None):
    """The reply's gene rows [[gene, transcripts, its own, from neighbours, tag(, giver)]] (genes already in order; the
    three amounts proseg's count, to DEEP_DIGITS decimals; `giver` the main giving neighbour's row, -1 for none: a 6th
    field only where it is >= 0), built only for the rows
    that fit: the same rows and genes_rest as fit_reply gives for the whole list, with every row's JSON length worked
    out from the arrays (a big cell's ~2,000 rows: ~1.5 ms less than encoding them). Rows of other shapes than the usual
    (a negative or huge value, a tag other than 0/1) take fit_reply itself. Mutates `result`."""
    n = int(len(g_idx))
    tx_l, own_l, nb_l = tx.tolist(), own.tolist(), nb.tolist()
    giver = np.full(n, -1, np.int64) if giver is None else np.asarray(giver, np.int64)

    def build(stop: int) -> list:
        rows = list(map(list, zip(gene_names(genes, g_idx[:stop]), tx_l[:stop], own_l[:stop], nb_l[:stop],
                                  tag[:stop].tolist())))
        for i in np.nonzero(giver[:stop] >= 0)[0].tolist():
            rows[i].append(int(giver[i]))
        return rows
    amounts = (tx, own, nb)
    usual = n == 0 or (all(float(a.min()) >= 0.0 and float(a.max()) < 1e9 and not np.signbit(a).any()
                           for a in amounts) and np.asarray(tx).dtype.kind == "f"
                       and set(np.unique(tag).tolist()) <= {0, 1} and int(giver.max()) < 10 ** 9)
    if not usual:
        result["genes"] = build(n)
        return fit_reply(result, cap)
    # ["ACTB",39.219,7.9,31.125,1,0]: the name's JSON, three numbers as Python writes them (at DEEP_DIGITS decimals:
    # _amount_bytes), a one-digit tag, two brackets and four commas, and the giver with its comma when named.
    row = name_bytes(genes)[g_idx] + _amount_bytes(tx) + _amount_bytes(own) + _amount_bytes(nb) + 1 + 6 + \
        np.where(giver >= 0, _decimal_digits(np.maximum(giver, 0).astype(np.float64)) + 1, 0)
    result["genes"] = []
    size = len(json.dumps(result, separators=(",", ":"))) + (int(row.sum()) + n - 1 if n else 0)
    keep = n
    if size > cap and n:
        over = size - cap + 64                           # room for "genes_rest"
        dropped = np.cumsum((row + 1)[::-1])             # the bytes the last k rows take, each with its comma
        k = int(np.searchsorted(dropped, over)) + 1      # the fewest rows from the end that free them ...
        keep = max(1, n - k)                             # ... keeping at least one
    result["genes"] = build(keep)
    if size > cap and n:                                 # as fit_reply: said once trimming was needed (even if
        result["genes_rest"] = [n - keep, round(sum(tx_l[keep:]), DEEP_DIGITS),                      # nothing
                                round(sum(own_l[keep:]), DEEP_DIGITS), round(sum(nb_l[keep:]), DEEP_DIGITS)]   # could go)
    return result


def explain_cell(c: int, frame: DeepFrame, fetch, deep: dict, encoder: dict, names, run_row, code=None,
                 judge_min: float = 100, base: dict | None = None, rules: dict | None = None,
                 same_prev: dict | None = None, start: dict | None = None) -> dict:
    """The v1 result of Go deeper for cell c (label - 1) of a frame, by the recipe of 3.2:
    fetch(cells) -> (their counts over the model's genes [n, G] float32: proseg's count, their transcripts of run genes
    the model lacks [n]); names: the run's cell ids (cells.txt); run_row: each run cell's training row (-1: not
    learned); code: the cell's byte in this frame (the card's number), None when there is none; same_prev: {neighbour
    cell (text): same kind?} as last shown (same_kind_classes); start: {x0 (the cell's start-block row over the model's
    genes [G]), x? (its averaged count now [G], when the rows fetched are another count), w0t, prev? (its word since the
    start as last shown)} for its like-for-like change since the start (like_for_like, start_word), or None. States:
    "empty" (no transcripts), "too_few" (< judge_min: its genes, unsplit), "not_checked" (no neighbour to compare with),
    "ready". Every amount is proseg's count (fractional: averaged over proseg's background draw, or the count proseg
    saves on a frame counted for a pause, the hold, a finish or the end), transcripts and outside exact (the counts
    file's 32nds: the window's whole number is then the stage readout's), the rest to DEEP_DIGITS decimals (each gene's
    transcripts are then the file's entry on the whole number it rounds to).
    Ready: {transcripts, outside, code, parts [own, neighbours, loose] (rho / sum), rough, genes [[gene, transcripts,
    its own, from neighbours (DEEP_DIGITS decimals; rounded down for a neighbour's gene, so its chip never claims more
    than the estimate), 1 = a neighbour's gene, the neighbour giving most of it (its row in `neighbours`; only from
    DEEP_GIVER_MIN transcripts from neighbours)]] (every gene it holds, most first; loose = the rest), genes_total,
    genes_rest? (fit_reply), kind (what its own RNA looks like), own_genes [[gene, its own, transcripts]],
    same_kind_share, neighbours [{cell (label - 1), name, um, tx, genes [[gene, tx] x 4], kind, similar, same (the class
    used), known}] (all k, most first), learned "yes" | "partly" | "no", start_change? (like_for_like, with its word:
    direction "more" | "fewer" | "same", held when start_word kept it), gap (points the
    split was moved to match parts), ms}."""
    t0 = time.perf_counter()
    genes = deep["genes"]
    result = {"v": 1, **(base or {}), "cell": int(c), "name": str(names[c]) if names is not None else None}
    L = float(frame.totals[c])
    if L <= 0:
        result.update(state="empty", transcripts=0.0)
        result["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return result

    def unsplit(row, outside):
        idx = np.nonzero(row > 0)[0]
        order = idx[np.lexsort((idx, -row[idx]))][:DEEP_TOO_FEW_GENES]
        return {"transcripts": round(L, 5), "outside": round(float(outside), 5),
                "genes": [[str(genes[g]), round(float(row[g]), DEEP_DIGITS)] for g in order]}
    nbrs, d2 = (np.zeros(0, np.int64), np.zeros(0)) if L < judge_min else frame.neighbours(c)
    if L < judge_min or not len(nbrs):
        X, outside = fetch(np.array([c]))
        result.update(state="too_few" if L < judge_min else "not_checked", **unsplit(X[0], outside[0]))
        if L >= judge_min:
            result["reason"] = "no_neighbours"
        result["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return result
    X, outside = fetch(np.concatenate([[c], nbrs]))
    x, xn = X[0], X[1:]
    logits, known = learned_logits(run_row, deep, c, nbrs)
    prev = None
    if same_prev:
        prev = np.array([(1 if same_prev[str(int(j))] else 0) if str(int(j)) in same_prev else -1 for j in nbrs], np.int8)
    r = deep_forward(np, deep, x, xn, d2.astype(np.float32), logits, deep_calculator(encoder, x), rules, same_prev=prev)
    parts = r["parts"]
    if code is None or int(code) == NOT_JUDGED:      # step 3: the card's byte when the frame has one
        code = int(share_codes(np, np.asarray(parts[1:2], np.float32), np.array([True]))[0])
    nz, xs, A = r["genes_nz"], r["x_nz"], r["A"]
    order = np.lexsort((nz, -xs))
    own_ok = np.nonzero(r["own_ok"])[0]
    own_ok = own_ok[top_order(A[own_ok, 0], nz[own_ok], DEEP_OWN_GENES)]     # lexsort's first few, unsorted rest
    rows = []
    ranked = np.lexsort((np.arange(len(nbrs)), -r["tx_k"]))
    tops = top_orders(r["Ak"], nz, DEEP_SOURCE_GENES)
    for k in ranked:
        top = tops[k]
        j = int(nbrs[k])
        rows.append({"cell": j, "name": str(names[j]) if names is not None else None,
                     "um": round(float(np.sqrt(d2[k])), 1), "tx": round(float(r["tx_k"][k]), DEEP_DIGITS),
                     "genes": [[str(genes[nz[g]]), round(float(r["Ak"][k, g]), DEEP_DIGITS)] for g in top],
                     "kind": [str(genes[g]) for g in r["kind_k"][k]], "similar": round(float(r["similar"][k]), 3),
                     "same": bool(r["same"][k]), "known": bool(known[k])})
    result.update(state="ready", transcripts=round(L, 5), outside=round(float(outside[0]), 5), code=int(code),
                  parts=[round(float(p), 4) for p in parts], rough=bool(r["rough"]), genes=[],
                  genes_total=int(len(order)), kind=[str(genes[g]) for g in r["kind"]],
                  own_genes=[[str(genes[nz[i]]), round(float(A[i, 0]), DEEP_DIGITS), round(float(xs[i]), DEEP_DIGITS)]
                             for i in own_ok],
                  same_kind_share=round(float(r["same_kind_share"]), 3), neighbours=rows,
                  learned="no" if logits is None else ("yes" if bool(known.all()) else "partly"),
                  gap=round(100.0 * float(r["raw_gap"]), 2))
    if start is not None:                              # the cell against itself at the start, like for like
        found = like_for_like(encoder, start["w0t"], start.get("x", x), start["x0"], judge_min)
        if found is not None:                          # its word, kept near the band as last shown (start_word)
            found["direction"], kept = start_word(found["points"], found["band"], start.get("prev"))
            if kept:
                found["held"] = True
            result["start_change"] = found
    tag = r["tag"][order]
    nb = A[order, 1]
    # A neighbour's gene sends its transcripts from neighbours rounded DOWN at DEEP_DIGITS decimals, so the window's
    # chip ("ACTB 31 of 39", its first number rounded down) never claims more than the estimate (2.9996 of 3 sent as 3.0
    # would read "3 of 3"); every other number rounded at DEEP_DIGITS decimals, once (the page rounds it to whole).
    scale = 10.0 ** DEEP_DIGITS
    nb_sent = np.where(tag, np.floor(nb * scale + 1e-9) / scale, np.round(nb, DEEP_DIGITS))
    position = np.empty(len(nbrs), np.int64)
    position[ranked] = np.arange(len(nbrs))
    giver = np.where(nb >= DEEP_GIVER_MIN, position[r["giver"][order]], -1)   # its row in `neighbours`
    result = fitted_gene_rows(result, genes, nz[order], np.round(xs[order].astype(np.float64), DEEP_DIGITS),
                              np.round(A[order, 0], DEEP_DIGITS), nb_sent, tag.astype(np.int64), giver=giver)
    result["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    return result


# ---- Compare: several segmentations of one sample, side by side ------------------
# Every column is read by its own count (the run's frame by proseg's, from its counts file; a shapes layer by the counts
# file the sopa helper made of it), on the one area all of them cover, scored by the mixing check's own calculator, cut
# down to one depth of whole transcripts for the like-for-like number and laid out in ONE UMAP coloured by the model's
# own kinds. Columns are count matrices over the model's genes in compressed rows (`Csr`), on the GPU or with numpy: the
# same code.
class Csr:
    """A cells x genes count matrix in compressed rows, on the GPU (cupy) or the host (numpy): data (float32 counts:
    whole, or proseg's averaged count in steps of 1/32), indices (int32 genes), indptr (int64), shape (rows, genes). The
    compare op's own: numpy and cupy run the same code (the app's environment has no scipy, and cupyx's matrices have no
    numpy twin)."""

    def __init__(self, data, indices, indptr, shape):
        self.data, self.indices, self.indptr = data, indices, indptr
        self.shape = (int(shape[0]), int(shape[1]))


def entry_rows(xp, indptr, start: int = 0, stop: int | None = None):
    """The row of each stored entry start .. stop - 1 of a compressed-row matrix (cupy's repeat takes no array of
    repeats, so the rows come from a search of the row ends)."""
    stop = int(indptr[-1]) if stop is None else stop
    return xp.searchsorted(indptr[1:], xp.arange(start, stop, dtype=xp.int64), side="right")


def csr_take(xp, m: Csr, rows) -> Csr:
    """Rows `rows` of a Csr, in that order."""
    rows = xp.asarray(rows, dtype=xp.int64)
    starts = m.indptr[rows]
    lengths = m.indptr[rows + 1] - starts
    indptr = xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(lengths)])
    total = int(indptr[-1])
    if not total:
        return Csr(xp.zeros(0, xp.float32), xp.zeros(0, xp.int32), indptr, (int(rows.size), m.shape[1]))
    owner = entry_rows(xp, indptr)
    at = starts[owner] + (xp.arange(total, dtype=xp.int64) - indptr[owner])
    return Csr(m.data[at], m.indices[at], indptr, (int(rows.size), m.shape[1]))


def csr_stack(xp, blocks: list) -> Csr:
    """Csrs of the same genes one under the other."""
    lengths = xp.concatenate([xp.diff(b.indptr) for b in blocks])
    return Csr(xp.concatenate([b.data for b in blocks]), xp.concatenate([b.indices for b in blocks]),
               xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(lengths)]),
               (sum(b.shape[0] for b in blocks), blocks[0].shape[1]))


def csr_columns(xp, m: Csr, index, width: int) -> Csr:
    """A Csr's genes renamed by `index` (its gene -> the new one, -1: dropped), into `width` genes."""
    mapped = xp.asarray(host(index).astype(np.int32))[m.indices]
    keep = mapped >= 0
    lengths = tally(xp, entry_rows(xp, m.indptr)[keep], m.shape[0])
    return Csr(m.data[keep], mapped[keep], xp.concatenate([xp.zeros(1, xp.int64), xp.cumsum(lengths)]),
               (m.shape[0], width))


def csr_row_sums(xp, m: Csr):
    """Each row's transcripts (float32; whole counts, and proseg's in steps of 1/32, sum exactly)."""
    if not int(m.data.size):
        return xp.zeros(m.shape[0], xp.float32)
    return tally(xp, entry_rows(xp, m.indptr), m.shape[0], m.data.astype(xp.float64)).astype(xp.float32)


def row_blocks(indptr_h: np.ndarray, chunk: int = DENSE_ROWS):
    """(first row, row after the last, first entry, entry after the last) of each block of `chunk` rows."""
    n = len(indptr_h) - 1
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        yield start, stop, int(indptr_h[start]), int(indptr_h[stop])


def log1p_matmul(xp, m: Csr, weights, scale=None, chunk: int = DENSE_ROWS):
    """log1p(each row's counts, times its `scale` when given) @ weights [genes, k] -> float32 [rows, k], DENSE_ROWS rows
    made dense at a time: the calculator's first layer (no scale) and Go deeper's description's (each cell over its own
    mean)."""
    n, width = m.shape
    out = xp.empty((n, int(weights.shape[1])), xp.float32)
    for start, stop, a, b in row_blocks(host(m.indptr), chunk):
        block = xp.zeros((stop - start, width), xp.float32)
        if b > a:
            row = entry_rows(xp, m.indptr, a, b) - start
            values = m.data[a:b]
            if scale is not None:
                values = values * scale[start:stop][row]
            block[row, m.indices[a:b]] = xp.log1p(values)
        out[start:stop] = block @ weights
    return out


def score_csr(xp, m: Csr, w0t, device: dict, fit_min: float):
    """(transcripts of the model's genes, neighbour share: NaN under fit_min) of every row of a Csr over the model's
    genes, by the mixing check's calculator (w0t: its first layer by gene [G, H] on the device)."""
    totals = csr_row_sums(xp, m)
    share = xp.full(m.shape[0], xp.nan, dtype=xp.float32)
    rows = xp.nonzero(totals >= fit_min)[0]
    if int(rows.size):
        share[rows] = neighbour_share(mixing_weights(xp, log1p_matmul(xp, csr_take(xp, m, rows), w0t), device))
    return totals, share


def splitmix(xp, seed: int, entry, trial):
    """SplitMix64 of (seed, entry, transcript), as the thinning draws its keys (THIN_SOURCE, thin_values): uint64."""
    mix = xp.uint64(SPLITMIX)
    with np.errstate(over="ignore"):
        key = (xp.uint64(seed) ^ (mix * (entry.astype(xp.uint64) + xp.uint64(1)))) \
            + mix * (trial.astype(xp.uint64) + xp.uint64(1))
        z = (key ^ (key >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)
        return z ^ (z >> xp.uint64(31))


def compare_seed(seed, key, draw: int) -> int:
    """The seed of one like-for-like draw of a column: fixed per request seed, column and draw."""
    digest = hashlib.sha256(f"proseg-live compare|{seed}|{key}|{draw}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def cut_to(xp, m: Csr, depth: int, seeds, chunk: int = CUT_CHUNK) -> list:
    """Every row of a count matrix (each holding `depth` transcripts or more) cut down to exactly `depth` of them, drawn
    without replacement, once per seed: a row keeps its transcripts with the `depth` smallest SplitMix64 keys of (seed,
    entry, transcript), so numpy and cupy draw alike, any chunking gives the same draw, and a row of exactly `depth`
    stays whole. -> one Csr per seed (the same rows and genes; a gene none of whose transcripts was drawn keeps a stored
    0). Rows are taken `chunk` transcripts at a time. Whole transcripts only: a fractional count is cut by its whole
    part."""
    n = m.shape[0]
    indptr = host(m.indptr).astype(np.int64)
    counts = m.data.astype(xp.int64)
    whole = Csr(counts.astype(xp.float32), m.indices, m.indptr, m.shape)
    before = np.concatenate([[0], np.cumsum(host(csr_row_sums(xp, whole)).astype(np.int64))])
    outs = [xp.zeros(int(m.data.shape[0]), xp.float32) for _ in seeds]
    r0 = 0
    while r0 < n:
        r1 = min(n, max(r0 + 1, int(np.searchsorted(before, before[r0] + chunk, side="right")) - 1))
        a, b = int(indptr[r0]), int(indptr[r1])
        total = int(before[r1] - before[r0])
        if total:
            reps = counts[a:b]
            ends = xp.cumsum(reps)
            entry = xp.searchsorted(ends, xp.arange(total, dtype=xp.int64), side="right")   # each transcript's entry
            trial = xp.arange(total, dtype=xp.int64) - (ends - reps)[entry]                # its place in the entry
            row = xp.searchsorted(xp.asarray(indptr[r0 + 1:r1 + 1] - a), entry, side="right")
            high = row.astype(xp.uint64) << xp.uint64(40)                  # rows stay apart, keys order a row's own
            first = xp.asarray(before[r0:r1] - before[r0])                 # each row's first place once sorted
            for out, seed in zip(outs, seeds):
                order = xp.argsort(high | (splitmix(xp, seed, entry + a, trial) >> xp.uint64(24)))
                rank = xp.arange(total, dtype=xp.int64) - first[row[order]]
                out[a:b] = tally(xp, entry[order][rank < depth], b - a).astype(xp.float32)
        r0 = r1
    return [Csr(out, m.indices, m.indptr, m.shape) for out in outs]


def like_for_like_at(xp, m: Csr, rows, w0t, device: dict, qc: dict, seeds, depth: int = LIKE_DEPTH) -> dict:
    """Like for like on a column: its rows `rows` (judged, `depth` transcripts or more) each cut down to exactly
    `depth` transcripts once per seed (cut_to) and scored by the calculator: {per100 (cells that look mixed per 100 of
    them, the draws' mean), sd (the draws' spread), cells, depth, draws}; per100 None without rows."""
    out = {"per100": None, "sd": None, "cells": int(len(rows)), "depth": int(depth), "draws": len(seeds)}
    if not len(rows):
        return out
    mixed_code = code_at(float(qc["mixed_at"]))
    found = []
    for cut in cut_to(xp, csr_take(xp, m, rows), depth, seeds):
        _, share = score_csr(xp, cut, w0t, device, float(qc["fit_min"]))
        codes = share_codes(xp, share, xp.isfinite(share))          # every cut cell holds `depth`: judged alike
        found.append(100.0 * int(((codes >= mixed_code) & (codes != NOT_JUDGED)).sum()) / len(rows))
        del cut
    out["per100"] = round(float(np.mean(found)), 2)
    out["sd"] = round(float(np.std(found, ddof=1)), 2) if len(found) > 1 else None
    return out


def compare_numbers(totals: np.ndarray, share: np.ndarray, inside: np.ndarray, qc: dict):
    """A column's numbers on the area, from each cell's transcripts of the model's genes and its share (host arrays;
    inside: its cells in the area): ({cells_in_area, judged, mixed, mixed_per100, histogram (HIST_EDGES: judged cells
    per bin of the byte codes, so its last two bins hold the mixed ones), median_tx (over the cells in the area),
    under100_share (of them, under judge_min)}, the judged cells, every cell's byte code)."""
    fit_min, judge_min = float(qc["fit_min"]), float(qc["judge_min"])
    judged = inside & (totals >= fit_min) & np.isfinite(share) & (totals >= judge_min)
    codes = share_codes(np, share, judged)
    mixed = judged & (codes >= code_at(float(qc["mixed_at"])))
    cells, j, m = int(inside.sum()), int(judged.sum()), int(mixed.sum())
    edges = np.array([code_at(e) for e in HIST_EDGES])
    hist = np.bincount(np.searchsorted(edges, codes[judged], side="right"), minlength=len(edges) + 1)
    return {"cells_in_area": cells, "judged": j, "mixed": m, "mixed_per100": round(100.0 * m / j, 2) if j else None,
            "histogram": [int(v) for v in hist], "median_tx": _median(totals[inside]),
            "under100_share": round(float((totals[inside] < judge_min).mean()), 4) if cells else None}, \
        np.nonzero(judged)[0], codes


def area_rows(cells, n: int) -> np.ndarray:
    """A column's cells on the area as the engine lists them (engine/compare.py: an .npy of its row numbers, or a list)
    as a mask of its n rows."""
    rows = np.asarray(np.load(cells, allow_pickle=False) if isinstance(cells, (str, os.PathLike)) else cells,
                      np.int64).ravel()
    if rows.size and (rows.min() < 0 or rows.max() >= n):
        raise ValueError(f"its cells on the area are not among its {n} cells")
    mask = np.zeros(n, bool)
    mask[rows] = True
    return mask


def given_area(area) -> dict:
    """The area as the engine worked it out ({box, size_um, tiles, tile_um}, or a box), when it names every column's
    cells on it: said back as it came."""
    if isinstance(area, dict):
        return {"box": area.get("box"), "size_um": area.get("size_um"), "tiles": area.get("tiles"),
                "tile_um": area.get("tile_um", AREA_TILE_UM)}
    box = area_limit(area)
    return {"box": box, "size_um": None if box is None else [round(box[2] - box[0], 1), round(box[3] - box[1], 1)],
            "tiles": None, "tile_um": AREA_TILE_UM}


def area_limit(area) -> list | None:
    """The box [x0, y0, x1, y1] a request's "area" names (a box, or {"box": ...}), or None."""
    if isinstance(area, dict):
        area = area.get("box") or area.get("limit")
    if area is None:
        return None
    box = [float(v) for v in area]
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"the area must be a box [x0, y0, x1, y1], not {area!r}")
    return box


def shared_area(centres: list, limit=None, tile: float = AREA_TILE_UM, least: int = AREA_MIN_CELLS):
    """The area every column covers: the tiles (`tile` µm, on a grid anchored at the limit box's corner, else at the
    multiple of `tile` below every centre) holding at least `least` cells of each column, inside the box `limit`
    [x0, y0, x1, y1] when given. centres: per column (x, y) host arrays, NaN for a cell without one. -> (the area
    {box (the kept tiles' bounds, within the limit; None when no tile is kept), size_um [w, h], tiles, tile_um}, per
    column which of its cells are in it (by their centres), per column how many tiles hold `least` of its cells)."""
    xs = [np.asarray(x, np.float64) for x, _ in centres]
    ys = [np.asarray(y, np.float64) for _, y in centres]
    valid = [np.isfinite(x) & np.isfinite(y) for x, y in zip(xs, ys)]
    if limit is not None:
        lx0, ly0, lx1, ly1 = limit
        valid = [v & (x >= lx0) & (x < lx1) & (y >= ly0) & (y < ly1) for v, x, y in zip(valid, xs, ys)]
    empty = {"box": None, "size_um": [0.0, 0.0], "tiles": 0, "tile_um": float(tile)}
    seen_x = np.concatenate([x[v] for x, v in zip(xs, valid)]) if xs else np.zeros(0)
    if not seen_x.size:
        return empty, [np.zeros(len(x), bool) for x in xs], [0] * len(xs)
    seen_y = np.concatenate([y[v] for y, v in zip(ys, valid)])
    ox, oy = (limit[0], limit[1]) if limit is not None else (math.floor(seen_x.min() / tile) * tile,
                                                              math.floor(seen_y.min() / tile) * tile)
    nx, ny = int((seen_x.max() - ox) // tile) + 1, int((seen_y.max() - oy) // tile) + 1
    ids = [np.where(v, (np.floor((np.where(v, y, oy) - oy) / tile).astype(np.int64) * nx
                        + np.floor((np.where(v, x, ox) - ox) / tile).astype(np.int64)), 0)
           for x, y, v in zip(xs, ys, valid)]
    counts = [np.bincount(i[v], minlength=nx * ny) for i, v in zip(ids, valid)]
    kept = np.logical_and.reduce([c >= least for c in counts])
    inside = [v & kept[i] for i, v in zip(ids, valid)]
    own = [int((c >= least).sum()) for c in counts]
    if not kept.any():
        return empty, inside, own
    k = np.nonzero(kept)[0]
    bx, by = k % nx, k // nx
    box = [ox + bx.min() * tile, oy + by.min() * tile, ox + (bx.max() + 1) * tile, oy + (by.max() + 1) * tile]
    if limit is not None:
        box = [max(box[0], limit[0]), max(box[1], limit[1]), min(box[2], limit[2]), min(box[3], limit[3])]
    box = [round(float(v), 2) for v in box]
    return {"box": box, "size_um": [round(box[2] - box[0], 1), round(box[3] - box[1], 1)],
            "tiles": int(kept.sum()), "tile_um": float(tile)}, inside, own


def even_quotas(sizes, cap: int) -> list[int]:
    """Points per column: all of them while they fit under `cap`, else the same share of every column (whole numbers
    by the largest remainders, ties to the first column)."""
    total = int(sum(sizes))
    if total <= cap:
        return [int(s) for s in sizes]
    exact = [s * cap / total for s in sizes]
    quotas = [int(math.floor(e)) for e in exact]
    for i in sorted(range(len(sizes)), key=lambda i: (quotas[i] - exact[i], i))[:cap - sum(quotas)]:
        quotas[i] += 1
    return quotas


def picked(n: int, quota: int, seed, index: int) -> np.ndarray:
    """`quota` of a column's n points, chosen at random (numpy's PCG64 from the request's seed and the column), in
    their own order."""
    if quota >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng([int(seed), int(index)]).choice(n, quota, replace=False))


def log_normalised(xp, m: Csr, totals, top: int):
    """The rows' counts normalised to 1e4 per cell and log1p'd, as umap() takes them, on their `top` most variable
    genes (plain variance over the rows): a dense float32 [rows, top or fewer]."""
    n, width = m.shape
    scale = (1e4 / xp.maximum(totals, 1)).astype(xp.float32)
    indptr_h = host(m.indptr)
    sums, squares = xp.zeros(width, xp.float64), xp.zeros(width, xp.float64)
    for start, stop, a, b in row_blocks(indptr_h):
        if b > a:
            values = xp.log1p(m.data[a:b] * scale[entry_rows(xp, m.indptr, a, b)]).astype(xp.float64)
            sums += tally(xp, m.indices[a:b], width, values)
            squares += tally(xp, m.indices[a:b], width, values * values)
    mean = sums / max(n, 1)
    keep = xp.argsort(-(squares / max(n, 1) - mean * mean))[:min(int(top), width)]
    column = xp.full(width, -1, xp.int64)
    column[keep] = xp.arange(int(keep.size), dtype=xp.int64)
    dense = xp.zeros((n, int(keep.size)), xp.float32)
    for start, stop, a, b in row_blocks(indptr_h):
        if b > a:
            row = entry_rows(xp, m.indptr, a, b)
            at = column[m.indices[a:b]]
            ok = at >= 0
            dense[row[ok], at[ok]] = xp.log1p(m.data[a:b][ok] * scale[row[ok]])
    return dense


# ---- Compare: the model's kinds ----------------------------------------------------------------
# A kind is defined once per model (its deep.npz) from Go deeper's description of the model's own cells (the z encoder's
# mean of log1p(counts over their mean), deep_z_mean): k-means with at most KINDS_MAX groups, fewer until each holds at
# least KINDS_MIN_SHARE of the cells; each named by the top genes of its centre's decoded profile, as Go deeper names
# "own kind". Any column's cell gets the nearest centre. Kinds are the model's, not cell types.
def deep_device(xp, deep: dict) -> dict:
    """The z encoder's arrays on the device, its first layer by gene (description's first layer, log1p_matmul)."""
    return {"Z0T": xp.asarray(np.ascontiguousarray(deep["Z0"].T)),
            **{k: xp.asarray(deep[k]) for k in ("zb0", "Z1", "zb1", "Zm", "zbm")}, "ln_eps": float(deep["ln_eps"])}


def z_means(xp, dd: dict, m: Csr, totals):
    """Go deeper's description of each row (deep_z_mean of its counts over the model's genes divided by their mean),
    its first layer worked out from the compressed rows; NaN for a row without transcripts."""
    scale = xp.where(totals > 0, xp.float32(m.shape[1]) / xp.maximum(totals, 1), 0).astype(xp.float32)
    eps = dd["ln_eps"]
    h = _deep_ln_relu(xp, log1p_matmul(xp, m, dd["Z0T"], scale) + dd["zb0"], eps)
    h = _deep_ln_relu(xp, h @ dd["Z1"].T + dd["zb1"], eps)
    z = h @ dd["Zm"].T + dd["zbm"]
    return xp.where((totals > 0)[:, None], z, xp.float32(np.nan))


def nearest_kind(xp, z, centres) -> np.ndarray:
    """Each row's nearest kind (Euclidean, in the description) as u8 on the host; NO_KIND for a row without one."""
    if not int(z.shape[0]):
        return np.zeros(0, np.uint8)
    known = xp.isfinite(z).all(axis=1)
    z = xp.where(known[:, None], z, 0)
    d2 = (z * z).sum(axis=1)[:, None] - 2 * (z @ centres.T) + (centres * centres).sum(axis=1)[None, :]
    return host(xp.where(known, d2.argmin(axis=1), NO_KIND).astype(xp.uint8))


def kmeans_start(points: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ on the host: each next centre drawn by squared distance among a few candidates (2 + ln k, as
    scikit-learn does), keeping the one that leaves the least spread."""
    n = len(points)
    centres = [points[int(rng.integers(n))]]
    closest = ((points - centres[0]) ** 2).sum(axis=1)
    trials = 2 + int(math.log(k)) if k > 1 else 1
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0:                                    # every point sits on a centre: any point will do
            centres.append(points[int(rng.integers(n))])
            continue
        candidates = np.minimum(np.searchsorted(np.cumsum(closest), rng.random(trials) * total), n - 1)
        spread = np.minimum(closest[None], ((points[None] - points[candidates][:, None]) ** 2).sum(axis=2))
        best = int(spread.sum(axis=1).argmin())
        centres.append(points[candidates[best]])
        closest = spread[best]
    return np.array(centres)


def kmeans(xp, points, k: int, seed: int, rounds: int = KMEANS_ROUNDS):
    """Lloyd's k-means of points [n, d] (xp, float32) from a k-means++ start drawn on the host (numpy's PCG64 from
    `seed`): (centres [k, d] float32 on the host, each point's group [n] on the host, sizes [k]); a group left without
    points restarts at the point farthest from its own centre."""
    n = int(points.shape[0])
    centres = xp.asarray(kmeans_start(host(points).astype(np.float64), k, np.random.default_rng(seed)), xp.float64)
    p = points.astype(xp.float64)
    p2 = (p * p).sum(axis=1)
    labels = None
    for _ in range(rounds):
        d2 = p2[:, None] - 2 * (p @ centres.T) + (centres * centres).sum(axis=1)[None, :]
        new = d2.argmin(axis=1)
        if labels is not None and bool((new == labels).all()):
            break
        labels = new
        sizes = tally(xp, labels, k)
        sums = xp.zeros((k, p.shape[1]), xp.float64)
        xp.add.at(sums, labels, p)
        centres = xp.where((sizes > 0)[:, None], sums / xp.maximum(sizes, 1)[:, None], centres)
        empty = np.nonzero(host(sizes == 0))[0]
        if len(empty):                                    # each restarts at one of the points farthest from their centres
            far = host(xp.argsort(-d2[xp.arange(n), labels])[:len(empty)])
            for g, f in zip(empty.tolist(), far.tolist()):
                centres[g] = p[f]
    sizes = host(tally(xp, labels, k))
    return host(centres).astype(np.float32), host(labels), sizes


def choose_kinds(xp, z, seed: int = KINDS_SEED):
    """k-means of the model's cells' descriptions with KINDS_MAX groups, then fewer until every group holds at least
    KINDS_MIN_SHARE of the cells: (centres, groups, sizes) with the most common kind first."""
    n = int(z.shape[0])
    need = math.ceil(KINDS_MIN_SHARE * n)
    for k in range(min(KINDS_MAX, n), 0, -1):
        centres, labels, sizes = kmeans(xp, z, k, seed)
        if sizes.min() >= need or k == 1:
            break
    order = np.lexsort((np.arange(len(sizes)), -sizes))     # the most common first (ties: the first found)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return centres[order], rank[labels], sizes[order]


def kind_names(deep: dict, centres: np.ndarray, words: int = KINDS_WORDS) -> tuple[list[str], list[list[int]]]:
    """Each kind's name: the top genes of its centre's decoded profile (Go deeper's kind words, deep_kind_words; genes
    by s ln(s / typical) after those when fewer qualify), up to two genes more while it would name the same genes as an
    earlier kind. -> (names "G1 · G2 · G3", each name's genes as model gene indices)."""
    profiles = deep_profile(np, deep, np.asarray(centres, np.float32))
    typical, genes = deep["typical"], deep["genes"]
    names, chosen, taken = [], [], []
    for s in profiles:
        first = deep_kind_words(np, deep, s, words + 2)
        score = s.astype(np.float64) * np.log(np.maximum(s, 1e-30) / np.maximum(typical, 1e-30))
        ranked = first + [int(g) for g in top_order(score, np.arange(len(s)), words + 2 + len(first))
                          if int(g) not in first]
        n = min(words, len(ranked))
        while n < min(len(ranked), words + 2) and set(ranked[:n]) in taken:
            n += 1
        taken.append(set(ranked[:n]))
        names.append(" · ".join(str(genes[g]) for g in ranked[:n]))
        chosen.append(ranked[:n])
    return names, chosen


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_kinds(path, sha: str, n_latent: int, ngenes: int) -> dict | None:
    """kinds.npz when it belongs to these details (deep.npz's sha256) and this recipe (KINDS_FORMAT, KINDS_VERSION),
    else None: missing, stale or malformed (made again)."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            data = {k: stored[k] for k in stored.files}
        if str(data["format"]) != KINDS_FORMAT or int(data["version"]) != KINDS_VERSION \
                or str(data["deep_sha256"]) != sha:
            return None
        centres, sizes, genes = data["centres"], data["sizes"], data["genes"]
        k = len(centres)
        if not 1 <= k <= KINDS_MAX or centres.shape != (k, n_latent) or not np.isfinite(centres).all() \
                or sizes.shape != (k,) or len(data["names"]) != k or len(data["colors"]) != k \
                or genes.ndim != 2 or len(genes) != k or (genes >= ngenes).any():
            return None
    except Exception:                                    # unreadable: made again
        return None
    return {"centres": centres.astype(np.float32), "sizes": sizes.astype(np.int64),
            "names": [str(v) for v in data["names"].tolist()], "colors": [str(v) for v in data["colors"].tolist()],
            "genes": [[int(g) for g in row if g >= 0] for row in genes], "cells": int(data["cells"]), "sha": sha}


def write_kinds(path, kinds: dict):
    """kinds.npz (atomic): format, version, deep_sha256, centres [K, n_latent], sizes, names, colors, genes [K, w]
    (model gene indices of each name, -1 after its last), cells (the model's cells the kinds were made from), seed,
    min_share."""
    width = max(len(g) for g in kinds["genes"])
    write_npz(path, {"format": np.array(KINDS_FORMAT), "version": np.int32(KINDS_VERSION),
                     "deep_sha256": np.array(kinds["sha"]), "centres": kinds["centres"].astype(np.float32),
                     "sizes": np.asarray(kinds["sizes"], np.int64), "names": np.array(kinds["names"]),
                     "colors": np.array(kinds["colors"]),
                     "genes": np.array([g + [-1] * (width - len(g)) for g in kinds["genes"]], np.int32),
                     "cells": np.int64(kinds["cells"]), "seed": np.int64(KINDS_SEED),
                     "min_share": np.float32(KINDS_MIN_SHARE)})


def read_counts_file(path):
    """A counts file of a shapes layer (zarr_bridge/helper.py count; .npz): data, indices, indptr, shape (a cells x
    genes CSR of whole counts), the cells' ids (COUNTS_IDS), outline centres x, y in µm (COUNTS_X, COUNTS_Y; NaN for
    none) and gene names (COUNTS_GENES). -> (Csr on the host, ids or None, x, y, genes), checked strictly."""
    with np.load(path, allow_pickle=False) as stored:
        names = set(stored.files)

        def first(keys, needed=True):
            key = next((k for k in keys if k in names), None)
            if key is None and needed:
                raise ValueError(f"{Path(path).name}: no {' / '.join(keys)}")
            return None if key is None else stored[key]
        data, indices, indptr = first(("data",)), first(("indices",)), first(("indptr",))
        shape = tuple(int(v) for v in first(("shape",)))
        ids, x, y, genes = first(COUNTS_IDS, False), first(COUNTS_X), first(COUNTS_Y), first(COUNTS_GENES)
    n, width = shape
    data = np.asarray(data, np.float32)
    if len(indptr) != n + 1 or int(indptr[0]) != 0 or int(indptr[-1]) != len(indices) or len(indices) != len(data) \
            or (np.diff(indptr) < 0).any() or (len(indices) and (indices.min() < 0 or indices.max() >= width)) \
            or not np.isfinite(data).all() or (data < 0).any() or len(genes) != width or len(x) != n or len(y) != n \
            or (ids is not None and len(ids) != n):
        raise ValueError(f"{Path(path).name}: malformed counts")
    return (Csr(data, np.asarray(indices, np.int32), np.asarray(indptr, np.int64), shape),
            None if ids is None else [str(v) for v in ids.tolist()], np.asarray(x, np.float64),
            np.asarray(y, np.float64), [str(g) for g in genes.tolist()])


def learning_counts(zarr_path, key: str, wait_s: float | None = None):
    """The model's learning table as the trainer reads it (analysis/qc_trainer.py read_table, read-only, while no save
    into the sample writes it: read_consistently): (Csr on the host, cell ids, gene names). `wait_s`: TimeoutError when
    a save into the sample still holds it after that many seconds (None: wait for the save, as the trainer does). Needs
    anndata and scipy: the RAPIDS environment has both."""
    from sopa_companion.analysis import qc_trainer as trainer
    counts, obs, genes, _, _ = trainer.read_consistently(zarr_path, lambda: trainer.read_table(Path(zarr_path), key),
                                                         timeout=wait_s)
    counts = counts.tocsr()
    ids = np.asarray((obs["cell_id"] if "cell_id" in obs else obs.index).astype(str))
    return (Csr(np.asarray(counts.data, np.float32), np.asarray(counts.indices, np.int32),
                np.asarray(counts.indptr, np.int64), counts.shape), [str(v) for v in ids.tolist()],
            [str(g) for g in genes.tolist()])


def request_ops(request) -> list[str]:
    """What a frame request asks for, in the order it is done (the mixing check first: it is the quick one; Go deeper
    next, from the same counts; clusters and UMAP last)."""
    ops = request.get("ops")
    if not ops:
        return {"qc": ["qc"], "deeper": ["deeper"]}.get(request.get("op"), ["umap"])
    return [op for op in FRAME_OPS if op in ops]


def unknown_ops(request) -> list[str]:
    """The ops a frame request names that this worker does not know (each is answered with an error, never dropped)."""
    ops = request.get("ops")
    if not isinstance(ops, (list, tuple)):
        return []
    return [str(op) for op in ops if op not in FRAME_OPS]


def request_count(request) -> str:
    """Which count of its frame's counts file a request reads (its "count": "averaged", e, or "sampled", f, the count
    proseg saves: the engine's counts_file.count_of of the frame's counts line); counts_file.LIVE_COUNT when it names
    none. ValueError for any other."""
    name = request.get("count") or counts_file.LIVE_COUNT
    if name not in counts_file.ARRAY:
        raise ValueError(f"no count named {name!r}: averaged or sampled")
    return name


def problem(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


# ---- clusters and UMAP: the parts both backends share ----------------------------------------------------------------
# The GPU's umap() and the CPU's (analysis/cpu.py umap) find the kept cells, the clusters' top genes and the layout's
# start with the same code, and write their files and summary with one writer (Worker.finish_umap), so the page cannot
# tell which backend made a result. `xp` is cupy or numpy, `sparse` cupyx.scipy.sparse or scipy.sparse.
def umap_cells(xp, counts, p: dict):
    """(transcripts per cell, the kept cells: those with >= min_counts); TooFewCells under max(30, n_neighbors + 2)."""
    totals = xp.asarray(counts.sum(axis=1)).ravel()
    keep = xp.nonzero(totals >= p["min_counts"])[0]
    if int(keep.size) < max(30, p["n_neighbors"] + 2):
        raise TooFewCells(f"only {int(keep.size)} cells have >= {p['min_counts']} transcripts in this frame")
    return totals, keep


def cluster_markers(xp, sparse, x, member: np.ndarray, p: dict):
    """Top genes per cluster on the normalised expression (x: the kept cells' log-normalised counts, a CSR of
    `sparse`; member: each kept cell's cluster id, on the host): every cluster at once, one transfer back. -> host
    arrays (ids, best genes [clusters, top_genes], their scores, the share of the cluster expressing them, sizes)."""
    n = int(member.shape[0])
    ids = np.unique(member)
    member = xp.asarray(member)
    indicator = sparse.csr_matrix((xp.ones(n, dtype=xp.float32), (xp.searchsorted(xp.asarray(ids), member),
                                                                   xp.arange(n))), shape=(len(ids), n))
    sums = (indicator @ x).toarray()
    binary = x.copy()
    binary.data = xp.ones_like(binary.data)
    expressing = (indicator @ binary).toarray()
    sizes = xp.asarray(indicator.sum(axis=1)).ravel()
    mean_in = sums / sizes[:, None]
    mean_out = (sums.sum(axis=0) - sums) / xp.maximum(n - sizes, 1)[:, None]
    pct = expressing / sizes[:, None]
    score = xp.where(pct >= p["min_pct"], mean_in - mean_out, -xp.inf)
    best = xp.argsort(-score, axis=1)[:, : p["top_genes"]]
    best_score = host(xp.take_along_axis(score, best, axis=1))
    best_pct = host(xp.take_along_axis(pct, best, axis=1))
    best, sizes = host(best), host(sizes)
    return ids, best, best_score, best_pct, sizes


def summary_clusters(run, ids, best, best_score, best_pct, sizes) -> list[dict]:
    """The summary's clusters (cluster_markers' arrays): {id, size, color, label, top}, each with a palette colour
    none on screen shares (assign_colours, at this moment of the pass: before the layout)."""
    colours = assign_colours(run, ids.tolist(), time.monotonic())   # palette indices: none shared on screen
    found = []
    for row, cid in enumerate(ids.tolist()):
        top_genes = [{"gene": run.genes[g], "score": round(s, 3), "pct": round(f, 3)}
                     for g, s, f in zip(best[row].tolist(), best_score[row].tolist(), best_pct[row].tolist())
                     if np.isfinite(s)]
        found.append({"id": int(cid), "size": int(sizes[row]), "color": colours[cid],
                      "label": " · ".join(t["gene"] for t in top_genes) or f"cluster {cid}", "top": top_genes})
    return found


def umap_start(run, keep: np.ndarray, neighbours: np.ndarray, ncells: int, p: dict):
    """(the layout's start for the kept cells or None, whether it is warm, the previous layout to rotate onto): counts
    the update, then starts from the previous layout (keep, neighbours: on the host)."""
    # 2-D UMAP, warm-started from the previous layout (cell ids are stable for the whole run). Every
    # `refresh_every` updates it is laid out afresh (spectral init) so small distortions cannot accumulate; the new
    # layout is still rotated onto the previous one, so the refresh does not jump.
    run.updates += 1
    init, warm = Worker.initial_embedding(run, keep, neighbours, ncells)
    previous = init
    if run.updates % int(p["refresh_every"]) == 0:
        warm = False
    # Other neighbours, spacing, fewest transcripts or genes make another map: laid out afresh (300 iterations, a
    # spectral start) and rotated onto the one on screen, so its dots fly to where the new map really puts them. A
    # new resolution only regroups the cells: the layout stays warm.
    if Worker.layout_settings_changed(run, p):
        warm = False
    return init, warm, previous


class RunState:
    """What the worker keeps of a run between its passes: its genes, its clusters and layout, and the init frame's count
    (the start block's rows). No transcripts: every count is a frame's counts file, proseg's own."""

    def __init__(self, genes_path: str, transcripts: int | None = None):
        self.genes = Path(genes_path).read_text().split("\n")[:-1]
        self.transcripts = transcripts                # the run's transcripts (live/info.json), for the UMAP summary
        self.embedding: np.ndarray | None = None      # [ncells, 2] float32 (normalised), NaN where unknown
        self.layout_key: tuple | None = None          # the settings of the last layout (layout_settings_changed)
        self.updates = 0
        self.clusters: np.ndarray | None = None       # [ncells] int32, -1 where unknown
        self.leiden_key: tuple | None = None          # the settings the clusters were found with (the CPU's Leiden
                                                      #  starts from them while these stay: analysis/cpu.py)
        self.next_cluster = 0
        self.colours: dict[int, int] = {}             # cluster id -> palette index, the clusters on screen (assign_colours)
        self.freed: dict[int, float] = {}             # palette index -> when a vanished cluster gave it up
        self.tile_buffer: np.ndarray | None = None    # host buffer the non-empty tiles of a tiled prior decode into
        self.qc_genes: OrderedDict = OrderedDict()    # (model file, mtime) -> (first layer in this run's gene order,
                                                      #  which run genes the model has, a description, and the model
                                                      #  gene of each run gene on the host)
        self.qc_start: tuple | None = None            # (start.npz, mtime, its per-cell values on the device)
        self.init: tuple | None = None                # ((frame 0's counts file, mtime), its count: the start block's
                                                      #  rows and the live values, nothing else: slim)
        self.init_totals: OrderedDict = OrderedDict()   # (init key, model file, mtime) -> its transcripts of the
                                                        #  model's genes, checked once against start.npz
        self.block: tuple | None = None               # ((counts file, mtime), its start block): a frame's start block
                                                      #  kept in an earlier counted frame's file than its own
        self.cell_ids: tuple | None = None            # (cells.txt, mtime, the run's cell ids: line k names label k + 1)
        self.deep_rows: OrderedDict = OrderedDict()   # (cells.txt, its mtime, deep.npz, its stamps) -> training row of
                                                      #  each run cell (-1: the model did not learn it)
        self.deep_start_frame: tuple | None = None    # (key, counts, totals_h, centres, codes or None): deeper_start's
                                                      #  frame, the init frame's count


def run_transcripts(request) -> int | None:
    """The run's transcripts as live/info.json (beside genes.txt) says, None when it does not."""
    try:
        info = json.loads((Path(request["genes"]).parent / "info.json").read_text())
        return int(info["transcripts"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Worker:
    """`xp`: cupy, the GPU worker; numpy runs the mixing check on the CPU. `backend` "cpu" (the CPU worker, with numpy):
    Clusters & UMAP, counts as a scipy CSR, Compare's layout and the warm-up on the CPU (analysis/cpu.py); numpy alone
    ("numpy", the default with numpy: tests, Test a setting, tests/fake_analysis_worker.py) keeps dense counts and no
    Clusters & UMAP. Every count is a frame's counts file (proseg's own: `count`)."""
    outer_allocator = None       # inside pooled(): the allocator cupy's new arrays come from outside it
    released = 0                 # bytes the pool gave back during the current request (before its UMAP part)
    backend = "gpu"              # "gpu", "cpu" or "numpy" (__init__)
    runs_kept = RUNS_KEPT        # runs kept at most (CPU_RUNS_KEPT on the CPU)

    def __init__(self, xp=None, backend=None):
        if xp is None:
            gpu_waits_block()                         # before anything creates the GPU context
            import cupy as xp  # noqa: F811  (fail fast if the GPU stack is missing)
        self.xp = xp
        self.backend = backend or ("gpu" if xp is not np else "numpy")
        self.runs_kept = CPU_RUNS_KEPT if self.backend == "cpu" else RUNS_KEPT
        self.runs: OrderedDict[str, RunState] = OrderedDict()  # least recently used first
        self.models: OrderedDict[str, tuple] = OrderedDict()   # encoder.npz -> (mtime, host arrays, device arrays)
        self.deeps: OrderedDict[str, dict] = OrderedDict()     # deep.npz -> its details, self-tested (deep_model)
        self.noted: set = set()                                # problems already logged once (the start's comparison)
        self.stats: OrderedDict[str, tuple] = OrderedDict()    # path -> (when looked up, mtime): stat_ns
        self.device_maps: OrderedDict[int, tuple] = OrderedDict()   # Go deeper's gene maps on the GPU (device_gene_map)
        self.rows_kernel: bool | None = None                   # Go deeper's GPU row kernel: works, failed, not tried
        self.centres: OrderedDict = OrderedDict()              # (cells file, ncells) -> its centres {cx, cy}: the
                                                               #  newest CENTRES_KEPT frames' (frame_centres)
        self.codes: OrderedDict = OrderedDict()                # (.qc.bin, ncells) -> its bytes: the newest frames'
                                                               #  (frame_codes)
        self.totals: OrderedDict = OrderedDict()               # (run, seq, model folder, ncells) -> its cells'
                                                               #  transcripts of the model's genes (Go deeper)
        self.kind_sets: OrderedDict = OrderedDict()            # kinds.npz -> the model's kinds (compare_kinds)
        self.stat_ttl_s = STAT_TTL_S
        # stat_ns renews an old lookup on this thread (the worker process only, on either backend: tests and the fake
        # worker ask at once).
        served = self.backend in ("gpu", "cpu")
        self.stat_pool = ThreadPoolExecutor(1, thread_name_prefix="stat") if served else None
        self.stat_pending: set = set()
        self.pool = ThreadPoolExecutor(DECODE_THREADS, thread_name_prefix="tiles") if DECODE_THREADS > 1 else None
        self.warm_steps: list = []                             # warm_umap_steps() still to take (main() fills it)

    def sync(self):
        if self.xp is not np:
            self.xp.cuda.Device().synchronize()

    @contextmanager
    def pooled(self):
        """Inside this block cupy's new arrays come from its own memory pool, also once cuML has switched cupy to
        RMM's allocator, whose resource here does no caching: each array was a cudaMalloc and each free a cudaFree that
        waits for the GPU, so passes of the mixing check took up to twice as long once any UMAP pass had run (a small
        region: counts 4.6 -> 7.5 ms, score 1.1 -> 3.7, bins 2.3 -> 4.2). The frame's counts and the check run in it;
        clusters and UMAP do not (cuML and cuGraph allocate through RMM anyway). A no-op with numpy."""
        if self.xp is np or self.outer_allocator is not None:
            yield
            return
        pool = self.xp.get_default_memory_pool()
        self.outer_allocator = self.xp.cuda.get_allocator()
        try:
            with self.xp.cuda.using_allocator(pool.malloc):
                yield
        finally:
            self.outer_allocator = None

    def state(self, request) -> RunState:
        """What the worker keeps of the run (its genes read once; RunState). At most RUNS_KEPT runs are kept
        (CPU_RUNS_KEPT on the CPU, whose runs sit in host memory), and the least recently used one goes first: the run
        being followed live is used by every pass, so the start lines and backfill of runs that are only looked at (the
        mixing check scores them too) never push its clusters and layout out."""
        key = request["run"]
        if key in self.runs:
            self.runs.move_to_end(key)
            return self.runs[key]
        while len(self.runs) >= self.runs_kept:       # bound GPU (or host) memory
            self.runs.popitem(last=False)
        self.runs[key] = RunState(request["genes"], run_transcripts(request))
        return self.runs[key]

    def pool_mb(self) -> dict | None:
        """GPU memory of this worker's cupy pool in MB: in use, and in total (in use plus the freed blocks it keeps
        for reuse; nvidia-smi counts both), and which allocator cupy's new arrays come from outside pooled(): "pool", or
        "rmm" once cuML is loaded (its RMM resource here does no caching: what it frees goes back at once). None without
        a GPU."""
        if self.xp is np:
            return None
        pool = self.xp.get_default_memory_pool()
        allocator = self.outer_allocator or self.xp.cuda.get_allocator()
        return {"pool_used_mb": round(pool.used_bytes() / 2 ** 20, 1), "pool_total_mb": round(pool.total_bytes() / 2 ** 20, 1),
                "allocator": "pool" if allocator == pool.malloc else "rmm"}

    def release_stale_blocks(self, qc_only: bool = False) -> int:
        """Give the blocks cupy's pool freed back to the driver; returns the bytes released.
        While the pool is cupy's allocator (cuML not loaded yet) a UMAP pass's blocks are kept for the next one. After
        a request of the mixing check alone (`qc_only`: no UMAP in it, or a run dropped), and after every request (and
        before a pass's UMAP part) once cuML has switched cupy to RMM (the first UMAP pass, or the warm-up), they go
        back, so a pass's transient high-water never stays on the GPU between passes (a whole sample: ~1.3 GB), nor
        beside cuML's arrays (otherwise blocks cached before the switch could hold several GB for the worker's whole
        life). Allocating them again costs a pass a few ms."""
        if self.xp is np:
            return 0
        pool = self.xp.get_default_memory_pool()
        free = pool.free_bytes()
        if not free or (self.xp.cuda.get_allocator() == pool.malloc and not qc_only):
            return 0
        pool.free_all_blocks()
        return free

    # ---- proseg's count (shared by every op) ---------------------------------------------------------------------
    def raster(self, run: RunState, request, plane: str = "consensus", layers_at: int = 2):
        """A frame's (or the prior's) labels on the device: ("tiles", header, tile ids, tiles) with tiles[0] all zero and
        tiles[1 + k] = tile ids[k] (only the non-empty tiles of the 2-D `plane`), or ("dense", layers, nx, ny, nz) with
        layers [nz * ny * nx + 1] (the extra last entry 0); a dense file holds `layers_at` planes before its per-layer
        labels (frames: consensus and last_change; a dense prior: none). Positions only: the starting footprints."""
        xp = self.xp
        if request.get("format") == "tiles":
            header, present, run.tile_buffer = plane_tiles(request["frame"], plane, run.tile_buffer, self.pool)
            tile = header["tile"]
            tiles = xp.empty((1 + len(present), tile, tile), dtype=xp.uint32)
            tiles[0] = 0
            if len(present):
                upload(tiles[1:], run.tile_buffer[:len(present)])   # one copy of the tissue's tiles, not the whole grid
            return ("tiles", header, present, tiles)
        nx, ny, nz = int(request["nx"]), int(request["ny"]), int(request["nz"])
        cells = nx * ny
        layers = xp.empty(nz * cells + 1, dtype=xp.uint32)                # + one zero past the grid
        layers[-1] = 0
        upload(layers[:-1], read_frame(request["frame"])[layers_at * cells:(layers_at + nz) * cells])
        return ("dense", layers, nx, ny, nz)

    def read_counts(self, run: RunState, request, key: str = "counts_file") -> counts_file.Counts:
        """The counts file the request names under `key` (a frame's: proseg's own count, analysis/counts_file.py), checked
        against the run's cells and genes. NoCounts when the request names none or it is gone; CountsError when it is
        malformed or not this run's (both: reason "no_counts")."""
        path = request.get(key)
        if not path:
            raise NoCounts("the request names no counts file: this frame has no proseg count")
        try:
            return counts_file.read(path, ncells=int(request["ncells"]), ngenes=len(run.genes))
        except FileNotFoundError:
            raise NoCounts(f"its counts file is gone ({Path(path).name})") from None

    def count(self, run: RunState, request, watch: Stopwatch | None = None):
        """(the frame's cell x gene matrix of proseg's count, the one the request names (`request_count`: the averaged
        count e by default, the sampled count f, the count proseg saves, on a frame counted for a pause, the hold, a
        finish or the end; float32), over the run's genes; its counts file as read: counts_file.Counts, n and e beside
        it): a cupy CSR on the GPU, a scipy CSR on the CPU worker, a dense matrix with numpy alone (tests, Test a setting:
        small runs only). Nothing is counted here: the file is proseg's own count. NoCounts / CountsError (reason
        "no_counts") without a counts file that fits the run (or holds that count). `watch` times reading the file
        ("load") and making the matrix ("counts")."""
        which = counts_file.ARRAY[request_count(request)]
        read = self.read_counts(run, request)
        if watch is not None:
            watch.mark("load")
        matrix = self.matrix(read, which)
        if watch is not None:
            watch.mark("counts")
        return matrix, read

    def matrix(self, read: counts_file.Counts, which: str | None = None):
        """One count of a counts file ("e", "n", "f"; LIVE_ARRAY by default: the start line's) as this backend's cell x
        gene matrix."""
        return self.wrap(counts_file.to_csr(np, read, which or counts_file.LIVE_ARRAY))

    def wrap(self, arrays: counts_file.CsrArrays):
        """Host CSR arrays (counts_file.CsrArrays) as this backend's matrix, without the stored zeros (a pair proseg
        assigns but averages to under 1/64 of a transcript, or none of whose transcripts its draw kept: never a gene the
        cell expresses): a cupy CSR (int32 indices) on the GPU, a scipy CSR on the CPU worker, dense with numpy alone."""
        arrays = without_zeros(arrays)
        data, indices, indptr = (np.asarray(a) for a in (arrays.data, arrays.indices, arrays.indptr))
        data = data.astype(np.float32, copy=False)
        if self.backend == "cpu":
            import scipy.sparse as sparse
            return sparse.csr_matrix((data, indices, indptr), shape=arrays.shape)
        if self.xp is np:
            dense = np.zeros(arrays.shape, np.float32)
            dense[np.repeat(np.arange(int(arrays.shape[0]), dtype=np.int64), np.diff(indptr)), indices] = data
            return dense
        import cupyx.scipy.sparse as cps
        xp = self.xp
        return cps.csr_matrix((xp.asarray(data), xp.asarray(indices.astype(np.int32)),
                               xp.asarray(indptr.astype(np.int32))), shape=arrays.shape)

    def handle(self, request) -> dict:
        """One request -> its reply (see the module docstring). Go deeper on the start line ("deeper" with start: true)
        reads the init frame's count instead of the frame's, apart from the request's other ops."""
        op = request.get("op")
        if op == "compare":
            return self.compare_reply(request)
        ops = ["qc_start"] if op == "qc_start" else request_ops(request)
        reply = {"run": request.get("run"), "seq": request.get("seq")}
        self.released = 0
        reasons, errors = {}, {}
        at_start = "deeper" in ops and bool((request.get("deeper") or {}).get("start"))
        frame_ops = [o for o in ops if not (at_start and o == "deeper")]
        try:
            if op == "qc_start":
                reply["qc"] = self.qc_start(request)
            elif frame_ops:
                result = self.process({**request, "ops": frame_ops})
                errors, reasons = result.pop("errors"), result.pop("reasons", {})
                reply.update(result)
        except Exception as exc:                          # the frame's count itself (or qc_start as a whole)
            report(request, exc)
            errors = dict.fromkeys(frame_ops, problem(exc))
            reason = failure_reason(request, exc)
            if reason:
                reasons = dict.fromkeys(frame_ops, reason)
        if at_start:
            try:
                reply["deeper"] = self.deeper_start(request)
            except Exception as exc:
                report(request, exc)
                errors["deeper"] = problem(exc)
                reason = failure_reason(request, exc)
                if reason:
                    reasons["deeper"] = reason
        for name in unknown_ops(request) if op != "qc_start" else ():
            errors[name] = f"ValueError: unknown op {name!r}"
            reasons[name] = "unknown_op"
        released = self.released + self.release_stale_blocks(qc_only="umap" not in ops)
        if isinstance(reply.get("qc"), dict) and isinstance(reply["qc"].get("gpu"), dict):
            after = self.pool_mb() or {}                 # the pool between passes (its numbers above: at the pass's end)
            reply["qc"]["gpu"].update(released_mb=round(released / 2 ** 20, 1),
                                      pool_kept_mb=after.get("pool_total_mb"))
        reply["ok"] = not errors
        if errors:
            reply["errors"] = errors
            reply["error"] = next(iter(errors.values()))
            if reasons:
                reply["reasons"] = reasons               # {op: "too_few_cells" | "run_deleted" | "no_counts" | ...}
        return reply

    def process(self, request) -> dict:
        """A frame's count read once (its counts file), then each op it asks for: {"qc"?, "deeper"?, "summary"? (umap),
        "errors": {op: message}}. The counts, the mixing check and Go deeper take their arrays from cupy's pool
        (pooled()); what they freed goes back before the UMAP part once RMM is cupy's allocator, so it never sits beside
        cuML's arrays at the pass's peak. A frame without a counts file that fits the run raises NoCounts / CountsError
        here: every op fails with reason "no_counts" (handle)."""
        ops = request_ops(request)
        out, errors, reasons = {}, {}, {}

        def failed(op, exc):
            errors[op] = problem(exc)
            reason = failure_reason(request, exc)
            if reason:
                reasons[op] = reason
            if reason != "too_few_cells":                 # an expected end, said by the reply: no traceback
                report(request, exc)
        with self.pooled():
            run = self.state(request)
            watch = Stopwatch(self.sync)
            counts, read = self.count(run, request, watch)
            kept = {"read": read} if "deeper" in ops else None   # the check's totals, bytes and cell statistics, and
            if "qc" in ops:                                      # the frame's file, for Go deeper
                try:
                    out["qc"] = self.score_frame(run, request, counts, read, watch.ms, kept)
                except Exception as exc:
                    failed("qc", exc)
            if "deeper" in ops:
                try:
                    out["deeper"] = self.deeper(run, request, counts, kept)
                except Exception as exc:
                    failed("deeper", exc)
            del kept
        if "umap" in ops:
            self.released += self.release_stale_blocks()
            try:
                out["summary"] = self.umap(run, request, counts, read, watch.ms)
                self.warm_steps = []                  # warm now: the warm-up has nothing left to do
            except Exception as exc:
                failed("umap", exc)
        out["errors"] = errors
        if reasons:
            out["reasons"] = reasons
        return out

    # ---- clusters and UMAP ----------------------------------------------------------------------------------------
    def umap(self, run: RunState, request, counts, read, ms: dict) -> dict:
        """One pass of clusters and UMAP on a frame's counts (proseg's count; `read` its counts file): writes out_cells
        and out_json (finish_umap) and returns the summary. On the CPU, analysis/cpu.py's recipe; here the GPU's (cuML,
        cuGraph)."""
        if self.backend == "cpu":
            from .cpu import umap
            return umap(self, run, request, counts, read, ms)
        import cupy as cp
        import cupyx.scipy.sparse as cps
        p = {**DEFAULTS, **(request.get("params") or {})}
        watch = Stopwatch(self.sync, ms)
        mark = watch.mark
        ncells, ngenes = int(request["ncells"]), len(run.genes)
        totals, keep = umap_cells(cp, counts, p)

        # Normalise, select variable genes, PCA.
        x = counts[keep]
        scale = 1e4 / cp.maximum(totals[keep], 1)
        x = cps.diags(scale.astype(cp.float32)) @ x
        x.data = cp.log1p(x.data)
        mean = cp.asarray(x.mean(axis=0)).ravel()
        sq = x.copy()
        sq.data **= 2
        var = cp.asarray(sq.mean(axis=0)).ravel() - mean ** 2
        top = cp.argsort(-var)[: min(int(p["n_top_genes"]), ngenes)]
        dense = x[:, top].toarray()
        dense -= dense.mean(axis=0)
        from cuml.decomposition import PCA
        n_pcs = principal_components(p["n_pcs"], dense.shape[0], dense.shape[1])
        pcs = PCA(n_components=n_pcs, output_type="cupy").fit_transform(dense)
        mark("pca")

        # kNN graph, shared by Leiden and UMAP.
        from cuml.neighbors import NearestNeighbors
        nn = int(min(p["n_neighbors"], keep.size - 1))
        distances, neighbours = NearestNeighbors(n_neighbors=nn, output_type="cupy").fit(pcs).kneighbors(pcs)
        n = int(keep.size)
        rows = cp.repeat(cp.arange(n, dtype=cp.int32), nn)
        knn = cps.csr_matrix((distances.ravel().astype(cp.float32), (rows, neighbours.ravel().astype(cp.int32))), shape=(n, n))
        mark("knn")

        import cudf
        import cugraph
        edges = cudf.DataFrame({"src": rows, "dst": neighbours.ravel().astype(cp.int32),
                                "w": cp.ones(n * nn, dtype=cp.float32)})
        graph = cugraph.Graph()
        graph.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="w", renumber=False, symmetrize=True)
        parts, _ = cugraph.leiden(graph, resolution=float(p["resolution"]), random_state=0)
        parts = parts.sort_values("vertex")
        local = parts["partition"].values.get().astype(np.int32)
        keep_host = keep.get()
        clusters = self.match_clusters(run, keep_host, local, ncells)
        mark("leiden")

        # Top genes per cluster on the normalised expression: every cluster at once, one transfer back.
        found = summary_clusters(run, *cluster_markers(cp, cps, x, clusters[keep_host], p))
        mark("markers")

        # 2-D UMAP, warm-started from the previous layout (umap_start), on the same kNN graph.
        from cuml.manifold import UMAP
        init, warm, previous = umap_start(run, keep_host, neighbours.get(), ncells, p)
        embedding = UMAP(n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]),
                         init=normalise(init) if warm else "spectral",
                         n_epochs=int(p["warm_epochs"] if warm else p["cold_epochs"]),
                         learning_rate=float(p["warm_learning_rate"]) if warm else 1.0,
                         output_type="numpy").fit_transform(pcs, knn_graph=knn)
        return self.finish_umap(run, request, p, keep_host, clusters, found, embedding, previous, warm, watch, n, read)

    def finish_umap(self, run: RunState, request, p: dict, keep: np.ndarray, clusters: np.ndarray,
                    summary_clusters: list, embedding, previous, warm: bool, watch: Stopwatch, n: int,
                    read: counts_file.Counts) -> dict:
        """The end of a clusters and UMAP pass, the same on both backends: the layout of the kept cells (`keep`, host;
        `embedding` as the layout gave it) normalised and rotated onto the previous one, kept for the next warm start;
        the cells file (int16 clusters, -1 where not kept, then float32 [ncells, 3] with z = 0 and NaN rows: out_cells)
        and the summary (SUMMARY_KEYS: out_json; transcripts_in_cells = proseg's count of the frame's cells, the count
        the pass clustered (`request_count`: round(sum of e), or of f on a frame read on the count proseg saves), from its
        counts file `read`, and `count` that count's name; transcripts = the run's, live/info.json), each written
        atomically; marks "umap" and "write"."""
        mark = watch.mark
        ncells = int(request["ncells"])
        embedding = normalise(embedding.astype(np.float32))
        if previous is not None:
            embedding = rotate_to(embedding, normalise(previous))
        layout = np.full((ncells, 2), np.nan, dtype=np.float32)
        layout[keep] = embedding
        run.embedding = layout
        full = np.zeros((ncells, 3), dtype=np.float32)        # the wire format keeps 3 components (z = 0)
        full[:, :2] = layout
        full[np.isnan(layout[:, 0]), 2] = np.nan
        mark("umap")

        # Per-cluster medians and the view box: the viewer places labels and fits the view without sorting points.
        centroids = cluster_centroids(clusters[keep], embedding)
        for item in summary_clusters:
            centre = centroids.get(item["id"])
            item["centroid"] = None if centre is None else [round(float(centre[0]), 3), round(float(centre[1]), 3), 0.0]
        payload = clusters.astype(np.int16).tobytes() + full.tobytes()
        tmp = request["out_cells"] + ".tmp"
        Path(tmp).write_bytes(payload)
        os.replace(tmp, request["out_cells"])
        count = request_count(request)
        summary = {"seq": request["seq"], "iter": request.get("iter"), "ncells": n, "nclusters": len(summary_clusters),
                   "umap_dims": 2, "bounds": layout_bounds(embedding),
                   "clusters": summary_clusters, "warm": warm, "ms": watch.ms, "params": p,
                   "transcripts_in_cells": int(round(float(np.sum(getattr(read, counts_file.ARRAY[count]),
                                                                  dtype=np.float64)))),
                   "transcripts": run.transcripts, "gpu": self.pool_mb(), "basis": BASIS, "count": count}
        tmp = request["out_json"] + ".tmp"
        Path(tmp).write_text(json.dumps(summary))
        os.replace(tmp, request["out_json"])
        mark("write")
        summary["ms"] = watch.ms
        return summary

    def warm_umap_steps(self) -> list:
        """Clusters and UMAP warmed up before their first pass, in steps that main() takes one at a time whenever the
        worker has been idle WARM_IDLE_S: importing cuML (~0.6 s; it also switches cupy to RMM's allocator) and cuGraph
        (~0.4 s), then one tiny fit of each part a pass uses, called as umap() calls it (PCA and neighbours, Leiden,
        UMAP from a layout and from scratch), whose first calls load the GPU's code and handles (~0.35 s). Without
        them the first UMAP pass in a worker took ~1.4 s longer (a whole sample: 1.9 s instead of 0.55 s, and the mixing
        check's result riding on it waited as long). Its cost, measured on a whole sample with the mixing check alone
        (Clusters & UMAP never switched on): +0.2 GB of GPU memory while idle, +0.9 GB once the check's passes run
        (cuML's RMM state beside cupy's pool) and +1.0 GB of the worker's host memory, in every session with the check
        on. The CPU's own four steps: analysis/cpu.py warm_steps. None with numpy alone."""
        if self.backend == "cpu":
            from .cpu import warm_steps
            return warm_steps(self)
        if self.xp is np:
            return []
        made = {}

        def import_cuml():
            import cuml  # noqa: F401

        def import_cugraph():
            import cudf  # noqa: F401
            import cugraph  # noqa: F401

        def pca_neighbours():
            import cupy as cp
            from cuml.decomposition import PCA
            from cuml.neighbors import NearestNeighbors
            p, n = DEFAULTS, 512
            dense = cp.asarray(np.random.default_rng(0).normal(size=(n, 64)).astype(np.float32))
            made["pcs"] = PCA(n_components=int(p["n_pcs"]), output_type="cupy").fit_transform(dense)
            made["knn"] = NearestNeighbors(n_neighbors=int(p["n_neighbors"]), output_type="cupy").fit(made["pcs"]) \
                .kneighbors(made["pcs"])

        def leiden():
            import cupy as cp
            import cudf
            import cugraph
            _, neighbours = made["knn"]
            n, nn = neighbours.shape
            edges = cudf.DataFrame({"src": cp.repeat(cp.arange(n, dtype=cp.int32), nn),
                                    "dst": neighbours.ravel().astype(cp.int32), "w": cp.ones(n * nn, dtype=cp.float32)})
            graph = cugraph.Graph()
            graph.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="w", renumber=False,
                                     symmetrize=True)
            cugraph.leiden(graph, resolution=float(DEFAULTS["resolution"]), random_state=0)

        def layouts():
            import cupy as cp
            import cupyx.scipy.sparse as cps
            from cuml.manifold import UMAP
            distances, neighbours = made.pop("knn")
            pcs = made.pop("pcs")
            n, nn = neighbours.shape
            knn = cps.csr_matrix((distances.ravel().astype(cp.float32),
                                  (cp.repeat(cp.arange(n, dtype=cp.int32), nn), neighbours.ravel().astype(cp.int32))),
                                 shape=(n, n))
            p = DEFAULTS
            first = UMAP(n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]), init="spectral", n_epochs=30,
                         learning_rate=1.0, output_type="numpy").fit_transform(pcs, knn_graph=knn)
            UMAP(n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]),
                 init=normalise(first.astype(np.float32)), n_epochs=int(p["warm_epochs"]),
                 learning_rate=float(p["warm_learning_rate"]), output_type="numpy").fit_transform(pcs, knn_graph=knn)

        return [import_cuml, import_cugraph, pca_neighbours, leiden, layouts]

    def warm_umap(self) -> dict:
        """Every warm-up step at once: {step: ms}."""
        took = {}
        for step in self.warm_umap_steps():
            t = time.perf_counter()
            step()
            took[step.__name__] = round((time.perf_counter() - t) * 1e3, 1)
        return took

    @staticmethod
    def layout_settings_changed(run, p: dict) -> bool:
        """Whether the settings the map is made with (neighbours, spacing, fewest transcripts, principal components,
        genes) differ from those of the run's previous layout; records them for the next pass. A first layout is no
        change; resolution is not among them (it regroups the cells on the same map)."""
        key = tuple(p.get(name) for name in ("n_neighbors", "min_dist", "min_counts", "n_pcs", "n_top_genes"))
        previous, run.layout_key = getattr(run, "layout_key", None), key
        return previous is not None and previous != key

    @staticmethod
    def initial_embedding(run: RunState, keep: np.ndarray, neighbours: np.ndarray, ncells: int):
        """Previous coordinates for cells seen before; new cells start at the mean of their seen neighbours."""
        if run.embedding is None or run.embedding.shape[1] != 2:
            return None, False
        previous = run.embedding[keep]
        known = ~np.isnan(previous[:, 0])
        if known.mean() < 0.5:                   # mostly new cells: start fresh
            return None, False
        init = previous.copy()
        new = np.nonzero(~known)[0]
        if len(new):
            near = neighbours[new]
            seen = known[near]                   # [new cells, neighbours]: which neighbours have a position
            total = np.where(seen[..., None], previous[near], 0).sum(axis=1)
            count = seen.sum(axis=1)[:, None]
            init[new] = np.where(count > 0, total / np.maximum(count, 1), np.nanmean(previous[known], axis=0))
        init += np.random.default_rng(0).normal(0, 1e-3, init.shape).astype(np.float32)
        return init.astype(np.float32), True

    @staticmethod
    def match_clusters(run: RunState, keep: np.ndarray, local: np.ndarray, ncells: int) -> np.ndarray:
        """Give new Leiden clusters the ids of the previous clusters they overlap most (stable colours and names)."""
        from scipy.optimize import linear_sum_assignment
        new_ids = np.unique(local)
        mapping = {}
        if run.clusters is not None:
            previous = run.clusters[keep]
            seen = previous >= 0
            old_ids = np.unique(previous[seen])
            if len(old_ids):
                overlap = np.zeros((len(new_ids), len(old_ids)), dtype=np.int64)
                np.add.at(overlap, (np.searchsorted(new_ids, local[seen]), np.searchsorted(old_ids, previous[seen])), 1)
                rows, cols = linear_sum_assignment(-overlap)
                for r, c in zip(rows, cols):
                    if overlap[r, c] > 0:
                        mapping[int(new_ids[r])] = int(old_ids[c])
        used = set(mapping.values())
        for cid in new_ids.tolist():
            if cid not in mapping:
                while run.next_cluster in used:
                    run.next_cluster += 1
                mapping[cid] = run.next_cluster
                used.add(run.next_cluster)
        run.next_cluster = max(used) + 1 if used else 0
        clusters = np.full(ncells, -1, dtype=np.int32)
        if len(local):
            clusters[keep] = np.array([mapping[cid] for cid in new_ids.tolist()], np.int32)[np.searchsorted(new_ids, local)]
        run.clusters = clusters
        return clusters

    # ---- the mixing check -----------------------------------------------------------------------------------------
    def model(self, path) -> tuple[dict, dict, int]:
        """(the calculator, its arrays on the device, the file's mtime), loaded once per file version and checked
        against its own reference cells here (the export was checked against torch where it was learned)."""
        path = Path(path)
        if path.suffix != ".npz":                     # a model folder (no stat of its own: every pass asks)
            path = path / "encoder.npz"
        stamp = self.stat_ns(path)                    # looked up at most every stat_ttl_s (a stat can take ~0.1 ms)
        cached = self.models.get(str(path))
        if cached is not None and cached[0] == stamp:
            self.models.move_to_end(str(path))
            return cached[1], cached[2], stamp
        encoder = load_encoder(path)
        device = on_device(self.xp, encoder)
        error = selftest_error(self.xp, encoder, device)
        if error is not None and error > SELFTEST_TOLERANCE:
            raise ValueError(f"the model failed its self-test here (largest difference {error:.3g}, "
                             f"allowed {SELFTEST_TOLERANCE:g})")
        encoder["selftest_error"] = error
        self.models[str(path)] = (stamp, encoder, device)
        while len(self.models) > MODELS_KEPT:
            self.models.popitem(last=False)
        return encoder, device, stamp

    def run_genes(self, run: RunState, encoder: dict, stamp: int):
        """(first layer in the run's gene order, 1 for run genes the model has, a description), once per run and model."""
        return self._run_genes(run, encoder, stamp)[:3]

    def gene_map(self, run: RunState, encoder: dict, stamp: int) -> np.ndarray:
        """The model gene of each run gene (-1 where the model has none), on the host (Go deeper's rows)."""
        return self._run_genes(run, encoder, stamp)[3]

    def _run_genes(self, run: RunState, encoder: dict, stamp: int):
        key = (encoder["path"], stamp)
        found = run.qc_genes.get(key)
        if found is None:
            index, names, matched = gene_index(encoder, run.genes)
            if not matched:
                raise ValueError("none of this run's genes is in the model (is it this sample's model?)")
            found = (self.xp.asarray(first_layer(encoder, index)), self.xp.asarray((index >= 0).astype(np.float32)),
                     {"run": len(run.genes), "model": encoder["ngenes"], "matched": matched, "names": names}, index)
            run.qc_genes[key] = found
            while len(run.qc_genes) > 2:
                run.qc_genes.popitem(last=False)
        return found

    def score(self, run: RunState, counts, qc: dict):
        """(transcripts of the model's genes per cell (proseg's count), neighbour share per cell (NaN below fit_min),
        genes, model)."""
        xp = self.xp
        encoder, device, stamp = self.model(qc["model"])
        weights, in_model, genes = self.run_genes(run, encoder, stamp)
        totals = xp.asarray(counts @ in_model).ravel().astype(xp.float32)
        share = xp.full(totals.shape[0], xp.nan, dtype=xp.float32)
        rows = xp.nonzero(totals >= float(qc["fit_min"]))[0]
        if int(rows.size):
            share[rows] = neighbour_share(mixing_weights(xp, log1p_rows(xp, counts, rows) @ weights, device))
        return totals, share, genes, encoder

    def start_values(self, run: RunState, path, ncells: int) -> dict | None:
        """The start line's per-cell values (start.npz) on the device, kept per run; None without a start line."""
        if not path:
            return None
        try:
            stamp = Path(path).stat().st_mtime_ns
        except OSError:
            return None
        if run.qc_start is not None and run.qc_start[:2] == (str(path), stamp):
            return run.qc_start[2]
        with np.load(path, allow_pickle=False) as stored:
            values = {k: self.xp.asarray(stored[k]) for k in START_FIELDS}
        if int(values["code"].shape[0]) != ncells:
            raise ValueError(f"the start line has {int(values['code'].shape[0])} cells, the run {ncells}")
        run.qc_start = (str(path), stamp, values)
        return values

    def score_frame(self, run: RunState, request, counts, read, ms: dict, kept: dict | None = None) -> dict:
        """The mixing check of one frame (proseg's count of it; `read` its counts file): writes qc.out_cells (and
        qc.out_bins, qc.out_json when asked for) and returns the whole run's summary. With a start line the frame is read
        against its start block ("start_block": its numbers at its own depth; the pairs and the per-bin start sums) and
        gets the like-for-like change ("matched", `start_comparison`: its time is part of "bins", whose per-bin sums it
        adds to, and is also given as matched.ms). `kept` receives the frame's totals, shares, bytes and cell
        statistics for Go deeper in the same pass.
        A frame read on the count proseg saves (`request_count` "sampled": `counts` is f) is scored on f, its summary's
        numbers (judged, mixed, per100, tx_sum, median_tx, the bands, the coverage counts and the rest) and its bytes
        (qc.out_cells); it is scored on its averaged count e too, which the like-for-like change reads as on any frame,
        and which gives the bins file (qc.out_bins: byte for byte an averaged request's of the same frame: the areas read
        e), its bytes on e (qc.out_cells_e) and the summary's `averaged` {judged, mixed, tx_sum, median_tx} (e, the
        summary's keys and rounding); `kept` then holds f's totals and bytes."""
        xp = self.xp
        qc = {**QC_DEFAULTS, **request["qc"]}
        grid = request_grid(qc)
        ncells = int(request["ncells"])
        count = request_count(request)
        watch = Stopwatch(self.sync, ms)
        totals, share, genes, encoder = self.score(run, counts, qc)
        # The averaged count beside f on a frame read on f: the like for like, the bins and the bytes on e.
        averaged = None if counts_file.ARRAY[count] == "e" else self.matrix(read, "e")
        e_totals, e_share = (totals, share) if averaged is None else self.score(run, averaged, qc)[:2]
        watch.mark("score")
        raw_stats = read_cellstats(request.get("cells_file"), ncells, float(request["voxel"]))
        if raw_stats is not None:                        # Go deeper on this frame later (paused, scrubbed) reads them
            self.keep_centres(request.get("cells_file"), ncells, raw_stats)
        stats = None if raw_stats is None else {k: xp.asarray(v) for k, v in raw_stats.items()}
        footprints = self.start_values(run, start_file(qc.get("start")), ncells)   # start.npz: positions, growth
        start, start_block, matched, extra = {}, None, None, None
        if request.get("init_counts_file") and request.get("start_counts_file"):
            began = time.perf_counter()
            try:
                start, start_block, matched, extra = self.start_comparison(
                    run, request, qc, counts if averaged is None else averaged, read, e_totals, e_share, footprints)
            except Exception as exc:                     # the start's comparison never costs the frame its check
                # A counts file gone or not this run's (no_counts) is one plain line, as everywhere; else the traceback.
                why = problem(exc) if getattr(exc, "reason", None) == "no_counts" else traceback.format_exc()
                self.note_once(f"matched:{request.get('run')}:{problem(exc)}",
                               f"no comparison with the start for run {request.get('run')}: {why}")
                start, matched = {}, {"error": problem(exc)}
            self.sync()
            matched["ms"] = round((time.perf_counter() - began) * 1e3, 1)
        if footprints is not None:
            start.update({k: footprints[k] for k in ("area_um2", "cx", "cy")})
        hosted = {} if kept is not None else None
        codes_e = None
        if averaged is None:
            codes, table, summary = qc_tables(xp, share, totals, qc, grid, stats, start, extra, hosted)
        else:                                            # the bins and e's bytes from e alone; the numbers from f
            codes_e, table, on_e = qc_tables(xp, e_share, e_totals, qc, grid, stats, start, extra)
            codes, _, summary = qc_tables(xp, share, totals, qc, grid, stats, start, None, hosted)
            if "paired" in on_e:                         # the pairs with the start block: like for like, on e
                summary["paired"] = on_e["paired"]
            summary["averaged"] = {k: on_e[k] for k in ("judged", "mixed", "tx_sum", "median_tx")}
        watch.mark("bins")
        codes_h = host(codes)                            # the one read-back write_codes always made
        write_codes(qc["out_cells"], codes_h)
        self.keep_codes(qc["out_cells"], ncells, codes_h)  # Go deeper on this frame later reads them from here
        if codes_e is not None and qc.get("out_cells_e"):
            write_codes(qc["out_cells_e"], host(codes_e))
        if qc.get("out_bins"):
            write_bins(qc["out_bins"], table, grid)
        watch.mark("write")
        summary.update(kind="frame", seq=request.get("seq"), iter=request.get("iter"), model=qc.get("model_id"),
                       model_id=qc.get("model_id"), cells=ncells, genes=genes, basis=BASIS, count=count,
                       has_start=footprints is not None, cellstats=stats is not None,
                       selftest=encoder.get("selftest_error"), start_block=start_block, matched=matched,
                       files={"cells": qc["out_cells"], "bins": qc.get("out_bins"),
                              **({"cells_e": qc.get("out_cells_e")} if codes_e is not None else {})},
                       ms=watch.ms, gpu=self.pool_mb())
        if qc.get("out_json"):                                # after the bytes: a reader takes both once this exists
            write_atomic(qc["out_json"], json.dumps(summary).encode())
        if kept is not None:
            kept.update(totals=totals, totals_h=hosted.get("totals"), codes=codes_h, stats=raw_stats, model=qc["model"])
        return summary

    def note_once(self, key: str, text: str):
        """A problem in the worker's log once per key (a frame's optional part that keeps failing the same way)."""
        if key not in self.noted:
            self.noted.add(key)
            log(text)

    def init_counts(self, run: RunState, request) -> counts_file.Counts:
        """The init frame's count (the request's "init_counts_file": frame 0's), kept with the run while its file stays
        the same: its rows (the start block's) and its live values, nothing else (slim). NoCounts without it."""
        path = request.get("init_counts_file")
        if not path:
            raise NoCounts("the request names no counts file of the init frame")
        try:
            key = (str(path), self.stat_ns(path))
        except FileNotFoundError:
            raise NoCounts(f"the init frame's counts file is gone ({Path(path).name})") from None
        if run.init is None or run.init[0] != key:
            run.init = (key, slim(self.read_counts(run, request, "init_counts_file"), counts_file.LIVE_ARRAY))
            run.init_totals.clear()
        return run.init[1]

    def block_counts(self, run: RunState, request, read: counts_file.Counts | None = None) -> counts_file.Counts:
        """The counts file holding the frame's start block (the request's "start_counts_file"): the frame's own (`read`,
        when it names the same file), else an earlier counted frame's, kept with the run while it stays the same.
        CountsError when that file holds no start block."""
        path = request.get("start_counts_file")
        if not path:
            raise NoCounts("the request names no counts file with the frame's start block")
        if read is not None and str(path) == str(request.get("counts_file")):
            block = read
        else:
            try:
                key = (str(path), self.stat_ns(path))
            except FileNotFoundError:
                raise NoCounts(f"the start block's counts file is gone ({Path(path).name})") from None
            if run.block is None or run.block[0] != key:
                run.block = (key, slim(self.read_counts(run, request, "start_counts_file"), "start_e"))
            block = run.block[1]
        if block.start_e is None:
            raise CountsError("the frame's start block is not in the counts file named for it")
        return block

    def start_comparison(self, run: RunState, request, qc: dict, counts, read, totals, share, footprints):
        """The frame against its start block: (the start for qc_tables {code, totals: the start block's cells at their
        own depth}, the frame's start summary {per100, judged, mixed, tx_sum, median_tx, iteration: the start block's own
        numbers at its own depth, and the iteration of the file it came from}, the like-for-like change (matched_change),
        its per-cell sums). The start block is proseg's iteration-0 assignment kept as proseg keeps RNA at this frame (the
        counts file's, on the init frame's rows), never the outlines. The like-for-like change compares averaged counts
        whatever count the frame is read on: `counts`, `totals` and `share` are the frame's averaged count (e) and its
        scores (score_frame reads e beside f on a frame read on f). It is refused (matched {"error"}) when the run's
        start.npz is not the init frame's count (it was made another way), the start's other numbers stand."""
        xp = self.xp
        init, block = self.init_counts(run, request), self.block_counts(run, request, read)
        start_counts = self.wrap(counts_file.start_csr(np, init, block))
        start_totals, start_share, _, _ = self.score(run, start_counts, qc)
        fit_min, judge_min = float(qc["fit_min"]), float(qc["judge_min"])
        scored = (start_totals >= fit_min) & xp.isfinite(start_share)
        judged = scored & (start_totals >= judge_min)
        start_codes = share_codes(xp, start_share, judged)
        mixed = judged & (start_codes >= code_at(float(qc["mixed_at"])))
        totals_h, scored_h = host(start_totals), host(scored)
        j, m = int(judged.sum()), int(mixed.sum())
        start_block = {"per100": round(100.0 * m / j, 2) if j else None, "judged": j, "mixed": m,
                       "tx_sum": round(float(totals_h.sum(dtype=np.float64)), 1),
                       "median_tx": _median(totals_h[scored_h]), "iteration": int(block.iteration)}
        try:
            self.check_start(run, qc, footprints, init)
        except ValueError as exc:
            self.note_once(f"start-npz:{request.get('run')}:{problem(exc)}",
                           f"no like-for-like change for run {request.get('run')}: {problem(exc)}")
            return {"code": start_codes, "totals": start_totals}, start_block, {"error": problem(exc)}, None
        encoder, device, stamp = self.model(qc["model"])
        weights, in_model, _ = self.run_genes(run, encoder, stamp)
        extra, matched = matched_change(xp, counts, totals, share, start_counts, start_totals, start_share, qc,
                                        weights, in_model, device)
        return {"code": start_codes, "totals": start_totals}, start_block, matched, extra

    def check_start(self, run: RunState, qc: dict, footprints: dict | None, init: counts_file.Counts):
        """The run's start.npz (when it exists) is the init frame's count: its totals equal the init frame's, once per
        init file, model and start.npz (a start line made another way gives no like-for-like change, never a wrong
        one). ValueError when they differ."""
        if footprints is None or run.qc_start is None:
            return
        encoder, _, stamp = self.model(qc["model"])
        key = (run.init[0], encoder["path"], stamp, run.qc_start[:2])
        found = run.init_totals.get(key)
        if found is None:
            _, in_model, _ = self.run_genes(run, encoder, stamp)
            matrix = self.wrap(counts_file.to_csr(np, init, counts_file.LIVE_ARRAY))
            init_totals = host(self.xp.asarray(matrix @ in_model).ravel()).astype(np.float32)
            found = run.init_totals[key] = bool(np.array_equal(init_totals,
                                                               host(footprints["totals"]).astype(np.float32)))
            while len(run.init_totals) > 2:
                run.init_totals.popitem(last=False)
        if not found:
            raise ValueError("the start line's counts differ from the init frame's (start.npz was made another way)")

    def qc_start(self, request) -> dict:
        """The start line: proseg's count of the init frame (the request's "counts_file": frame 0's), scored, plus each
        cell's starting footprint on the prior's labels (positions only). Writes qc.out_cells (byte per cell),
        qc.out_start (START_FIELDS; by default start.npz beside out_cells) and, when asked for, qc.out_bins (cells placed
        at their starting centroids) and qc.out_json. Its arrays come from cupy's pool (pooled())."""
        with self.pooled():
            return self._qc_start(request)

    def _qc_start(self, request) -> dict:
        xp = self.xp
        run = self.state(request)
        qc = {**QC_DEFAULTS, **request["qc"]}
        grid = request_grid(qc)
        out_start = qc.get("out_start") or start_file(qc["out_cells"])
        ncells = int(request["ncells"])
        watch = Stopwatch(self.sync)
        read = self.read_counts(run, request)
        geometry = prior_request(request)
        if geometry is None:
            raise ValueError("this run has no starting cells to place (no prior)")
        area, cx, cy = footprint(xp, self.raster(run, geometry, plane="label", layers_at=0), ncells,
                                 float(geometry["x0"]), float(geometry["y0"]), float(geometry["voxel"]))
        watch.mark("load")
        counts = self.matrix(read)
        watch.mark("counts")
        totals, share, genes, encoder = self.score(run, counts, qc)
        watch.mark("score")
        codes, table, summary = qc_tables(xp, share, totals, qc, grid, {"area_um2": area, "cx": cx, "cy": cy})
        watch.mark("bins")
        write_codes(qc["out_cells"], codes)
        write_npz(out_start, {"code": host(codes), "share": host(share).astype(np.float32),
                              "totals": host(totals).astype(np.float32), "area_um2": host(area),
                              "cx": host(cx), "cy": host(cy)})
        if qc.get("out_bins"):
            write_bins(qc["out_bins"], table, grid)
        watch.mark("write")
        summary.update(kind="start", seq=None, model=qc.get("model_id"), model_id=qc.get("model_id"), cells=ncells,
                       genes=genes, basis=BASIS, count=counts_file.LIVE_COUNT, selftest=encoder.get("selftest_error"),
                       footprints=int((area > 0).sum()),
                       files={"cells": qc["out_cells"], "start": out_start, "bins": qc.get("out_bins")}, ms=watch.ms,
                       gpu=self.pool_mb())
        if qc.get("out_json"):
            write_atomic(qc["out_json"], json.dumps(summary).encode())
        return summary

    # ---- Go deeper ---------------------------------------------------------------------------------------------------
    def deep_model(self, folder) -> dict:
        """The model's calculator and details for Go deeper (encoder.npz, deep.npz and cells.parquet of the model
        folder), loaded once per file version and checked here: deep.npz strictly (load_deep), its self-test in float32
        within DEEP_SELFTEST_TOLERANCE, and cells.parquet's ids against its train_ids_sha256. {encoder, device, stamp,
        deep, rows ({learned cell id: training row}), selftest, stamps, key}; kept beside the calculators (MODELS_KEPT).
        A model learned before Go deeper (no deep.npz) raises, naming the missing file."""
        folder = model_folder(folder)
        encoder, device, stamp = self.model(folder / "encoder.npz")
        path, ids = folder / "deep.npz", folder / "cells.parquet"
        try:
            stamps = (stamp, self.stat_ns(path), self.stat_ns(ids))
        except FileNotFoundError as exc:
            raise ValueError(f"this sample's model has no details for Go deeper ({Path(exc.filename).name} is "
                             "missing): learn the details") from None
        cached = self.deeps.get(str(path))
        if cached is not None and cached["stamps"] == stamps:
            self.deeps.move_to_end(str(path))
            return cached
        deep = load_deep(path, encoder["genes"].get("genes") or next(iter(encoder["genes"].values())))
        with one_blas_thread():
            error = deep_self_test(deep, encoder)
        if not error["max"] <= DEEP_SELFTEST_TOLERANCE:
            raise ValueError(f"the model's details failed their self-test here (largest difference {error['max']:.3g}, "
                             f"allowed {DEEP_SELFTEST_TOLERANCE:g})")
        learned = read_cell_ids(ids)
        if len(learned) != deep["train_logit"].shape[0] or ids_sha256(learned) != deep["train_ids_sha256"]:
            raise ValueError("the model's cells.parquet does not match its details (train_ids_sha256)")
        name_bytes(deep["genes"])                    # what the first answer would otherwise work out (~3 ms) ...
        gene_names(deep["genes"], [])
        if self.xp is not np and getattr(self, "rows_kernel", None) is None:   # ... and the GPU's row kernel (cupy's
            try:                                                                # disk cache, else compiled)
                deep_rows_kernel(self.xp).compile()
                self.rows_kernel = True
            except Exception as exc:                 # deep_fetch then uses cupyx's row indexing
                self.rows_kernel = False
                self.note_once("deep-rows-kernel", f"Go deeper: the GPU row kernel could not compile ({problem(exc)}); "
                                                   "rows are fetched with cupyx")
        entry = {"encoder": encoder, "device": device, "stamp": stamp, "deep": deep, "stamps": stamps,
                 "rows": {cell: row for row, cell in enumerate(learned)}, "selftest": error, "key": str(path),
                 "w0t": np.ascontiguousarray(encoder["W0"].T)}   # the calculator's first layer by gene (like_for_like)
        self.deeps[str(path)] = entry
        while len(self.deeps) > MODELS_KEPT:
            self.deeps.popitem(last=False)
        return entry

    def deep_cells(self, run: RunState, model: dict, path, ncells: int):
        """(the run's cell ids, each run cell's training row): the ids from live/cells.txt (line k names label k + 1),
        matched by id to the model's learned cells (cells.parquet), once per run and model. A run keeps its Xenium cell
        ids for its whole life, so the model's learned values carry over to every frame (3.3)."""
        if not path:
            raise ValueError("Go deeper needs the run's cell ids (live/cells.txt)")
        stamp = self.stat_ns(path)
        if run.cell_ids is None or run.cell_ids[:2] != (str(path), stamp):
            names = Path(path).read_text().split("\n")
            if names and names[-1] == "":
                names = names[:-1]
            if len(names) != ncells:
                raise ValueError(f"cells.txt names {len(names)} cells, the run has {ncells}")
            run.cell_ids = (str(path), stamp, names)
        names = run.cell_ids[2]
        key = (str(path), stamp, model["key"], model["stamps"])
        rows = run.deep_rows.get(key)
        if rows is None:
            learned = model["rows"]
            rows = run.deep_rows[key] = np.fromiter((learned.get(n, -1) for n in names), np.int64, len(names))
            while len(run.deep_rows) > 2:
                run.deep_rows.popitem(last=False)
        return names, rows

    def deep_fetch(self, counts, index: np.ndarray, ngenes: int):
        """fetch(cells) for explain_cell: those cells' rows of a count matrix over the run's genes (CSR on the GPU,
        dense with numpy) as dense rows over the model's genes on the host, and their transcripts of run genes the
        model lacks. On the GPU one kernel writes the rows straight into model-gene columns and one read-back brings
        them over (deep_rows_kernel: ~0.1 ms; cupyx's row indexing plus a packed read-back took ~0.4 ms of the op, and
        its toarray would also sort them first: ~1.7 ms). Other index types: the rows' three arrays, packed into one
        read-back."""
        xp = self.xp
        known = index >= 0

        def fetch(rows):
            picked = counts[xp.asarray(rows)]
            out = np.zeros((len(rows), ngenes), np.float32)
            if hasattr(picked, "indptr"):                 # each run gene once per row; one model gene per run gene
                n, nnz = len(rows), int(picked.indices.shape[0])
                if xp is np:
                    indptr, columns, values = picked.indptr, picked.indices, picked.data
                else:                                     # one read-back (each waits for the GPU) for the three arrays
                    packed = host(xp.concatenate([picked.indptr.astype(xp.int32), picked.indices.astype(xp.int32),
                                                  picked.data.astype(xp.float32).view(xp.int32)]))
                    indptr, columns = packed[:n + 1], packed[n + 1:n + 1 + nnz]
                    values = packed[n + 1 + nnz:].view(np.float32)
                row = np.repeat(np.arange(len(rows)), np.diff(indptr))
                genes = index[columns]
                inside = genes >= 0
                out[row[inside], genes[inside]] = values[inside]
                return out, np.bincount(row[~inside], weights=values[~inside], minlength=len(rows))
            dense = host(picked).astype(np.float32)
            out[:, index[known]] = dense[:, known]
            return out, dense[:, ~known].sum(axis=1)

        if xp is not np and getattr(self, "rows_kernel", None) is not False and hasattr(counts, "indptr") \
                and counts.indptr.dtype == xp.int32 and counts.indices.dtype == xp.int32 \
                and counts.data.dtype == xp.float32:

            def fetch_rows(rows):
                try:
                    kernel, gene_map = deep_rows_kernel(xp), self.device_gene_map(index)
                    n = len(rows)
                    buffer = xp.zeros(n * ngenes + n, xp.float32)        # the rows, then each row's outside sum
                    kernel((n,), (128,), (counts.indptr, counts.indices, counts.data,
                                          xp.asarray(np.asarray(rows, np.int32)), np.int32(n), gene_map,
                                          np.int32(ngenes), buffer[:n * ngenes], buffer[n * ngenes:]))
                    both = host(buffer)
                    return both[:n * ngenes].reshape(n, ngenes), both[n * ngenes:].astype(np.float64)
                except Exception as exc:              # the kernel could not compile or launch (an NVRTC or CUDA
                    self.rows_kernel = False          # toolkit problem): cupyx's row indexing from now on, said once
                    self.note_once("deep-rows-kernel", f"Go deeper: the GPU row kernel failed ({problem(exc)}); "
                                                       "rows are fetched with cupyx from now on")
                    return fetch(rows)
            return fetch_rows
        return fetch

    def device_gene_map(self, index: np.ndarray):
        """The run-gene -> model-gene map (-1: a gene the model lacks) on the GPU as int32, kept for the newest few
        maps (gene_map's own arrays: the same object while the run's genes and the model stay)."""
        hit = self.device_maps.get(id(index))
        if hit is not None and hit[0] is index:
            return hit[1]
        mapped = self.xp.asarray(np.asarray(index, np.int32))
        self.device_maps[id(index)] = (index, mapped)
        while len(self.device_maps) > 4:
            self.device_maps.popitem(last=False)
        return mapped

    def keep_centres(self, path, ncells: int, stats: dict):
        """Keep a frame's cell centres (its statistics' cx, cy) for frame_centres."""
        if not path:
            return
        self.centres[(str(path), int(ncells))] = {"cx": stats["cx"], "cy": stats["cy"]}
        self.centres.move_to_end((str(path), int(ncells)))
        while len(self.centres) > CENTRES_KEPT:
            self.centres.popitem(last=False)

    def frame_centres(self, path, ncells: int) -> dict | None:
        """read_centres of a frame's cell statistics, kept for the newest CENTRES_KEPT frames: proseg writes a frame's
        file once, and Go deeper on a paused or stepped frame read it again for every cell it was asked about (~0.1 ms
        each on a small region) though the check's pass of that frame had read it already."""
        if not path:
            return None
        key = (str(path), int(ncells))
        found = self.centres.get(key)
        if found is not None:
            self.centres.move_to_end(key)
            return found
        found = read_centres(path, ncells)
        if found is not None:
            self.keep_centres(path, ncells, found)
        return found

    def keep_codes(self, path, ncells: int, codes):
        """Keep a frame's bytes (its .qc.bin, as written) for frame_code."""
        if not path:
            return
        key = (str(path), int(ncells))
        self.codes[key] = np.asarray(codes)
        self.codes.move_to_end(key)
        while len(self.codes) > CENTRES_KEPT:
            self.codes.popitem(last=False)

    def frame_code(self, path, c: int, ncells: int) -> int | None:
        """stored_code from the newest CENTRES_KEPT frames' bytes kept here (each .qc.bin is written once per frame and
        model, by the check's pass of it): Go deeper without the check in its pass (a paused frame, another cell clicked)
        read the file from the disk for every answer, ~1 ms under load on a FUSE-mounted disk."""
        found = self.codes.get((str(path), int(ncells))) if path else None
        if found is not None:
            self.codes.move_to_end((str(path), int(ncells)))
            return int(found[c]) if len(found) == ncells else None
        return stored_code(path, c, ncells)

    def stat_ns(self, path) -> int:
        """A file's mtime (ns), looked up at most once per stat_ttl_s per path: a run folder on a FUSE-mounted disk
        answers a stat in ~0.1 ms, and a followed cell needs several files on every pass. A file replaced in place (the
        details learned for a model) is seen within that time (and one answer). With a stat thread (the GPU worker) an
        older lookup is renewed there while the answer goes on with the one it has: under a playing run a stat on such a
        mount waited up to ~4 ms behind proseg's frame writes, inside Go deeper's answer. A file never looked up is
        asked at once, and one found gone raises FileNotFoundError as the disk does."""
        now = time.monotonic()
        key = str(path)
        hit = self.stats.get(key)
        if hit is not None and (now - hit[0] < self.stat_ttl_s or self.stat_pool is not None):
            if now - hit[0] >= self.stat_ttl_s and key not in self.stat_pending:
                self.stat_pending.add(key)
                self.stat_pool.submit(self._restat, key)
            if hit[1] is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), key)
            return hit[1]
        value = Path(path).stat().st_mtime_ns
        self.stats[key] = (now, value)
        if len(self.stats) > 64:
            self.stats.popitem(last=False)
        return value

    def _restat(self, key: str):
        """stat_ns's renewal on the stat thread: the file's new mtime, or None once it is gone."""
        try:
            value = Path(key).stat().st_mtime_ns
        except OSError:
            value = None
        self.stats[key] = (time.monotonic(), value)
        self.stat_pending.discard(key)

    def deeper(self, run: RunState, request, counts, kept: dict) -> dict:
        """Go deeper on one cell of this frame (`explain_cell`), from the counts the pass holds (proseg's count): with
        the check in the same pass, its totals and the cell's byte (the card's number); else the frame's stored byte
        (deeper.codes) and totals of its own. Centres: the frame's cell statistics, the starting footprints where those
        are NaN (the run's start.npz, when a pass of the check read it: positions only). With "start_line" in the block,
        the cell against its own start-block row (`like_for_like`: the request names "init_counts_file" and
        "start_counts_file"; kept["read"] is the frame's counts file when the pass read it)."""
        with no_collection():
            return self._deeper(run, request, counts, kept)

    def _deeper(self, run: RunState, request, counts, kept: dict) -> dict:
        began = time.perf_counter()
        xp = self.xp
        block = {**QC_DEFAULTS, "k": DEEP_K, **request["deeper"]}
        ncells = int(request["ncells"])
        c = deeper_cell(block, ncells)
        loading = time.perf_counter()
        fresh = str(model_folder(block["model"]) / "deep.npz") not in self.deeps
        model = self.deep_model(block["model"])
        load_ms = (time.perf_counter() - loading) * 1e3 if fresh else 0.0     # once per model version: apart
        encoder, stamp = model["encoder"], model["stamp"]
        _, in_model, _ = self.run_genes(run, encoder, stamp)
        count = request_count(request)
        shared = kept.get("totals") is not None and model_folder(kept["model"]) == model_folder(block["model"])
        frame = (request.get("run"), request.get("seq"), str(model_folder(block["model"])), ncells, count)
        if shared and kept.get("totals_h") is not None:   # the check's own host copy: no read-back here
            totals_h = kept["totals_h"]
        elif not shared and request.get("seq") is not None and frame in self.totals:   # this frame's, worked out
            totals_h = self.totals[frame]                                             # before (another cell)
            self.totals.move_to_end(frame)
        else:
            totals_h = host(kept["totals"] if shared else xp.asarray(counts @ in_model).ravel().astype(xp.float32))
        if request.get("seq") is not None and frame not in self.totals:
            self.totals[frame] = totals_h
            while len(self.totals) > CENTRES_KEPT:
                self.totals.popitem(last=False)
        code = int(kept["codes"][c]) if shared else self.frame_code(block.get("codes"), c, ncells)
        stats = kept["stats"] if "stats" in kept else self.frame_centres(request.get("cells_file"), ncells)
        centres = np.full((ncells, 2), np.nan) if stats is None else \
            np.stack([stats["cx"], stats["cy"]], axis=1).astype(np.float64)
        gap = ~np.isfinite(centres).all(axis=1)
        if gap.any() and run.qc_start is not None and int(run.qc_start[2]["cx"].shape[0]) == ncells:
            begun = np.stack([host(run.qc_start[2]["cx"]), host(run.qc_start[2]["cy"])], axis=1).astype(np.float64)
            centres[gap] = begun[gap]
        names, rows = self.deep_cells(run, model, block.get("cell_ids"), ncells)
        gene_map = self.gene_map(run, encoder, stamp)
        start = None
        if block.get("start_line"):                   # the cell against itself at the start, like for like
            loading = time.perf_counter()
            fresh = run.init is None
            try:
                start = self.start_rows(run, request, kept.get("read"), c, gene_map, encoder["ngenes"])
                start.update(w0t=model["w0t"], prev=block.get("start_prev"))
            except Exception as exc:                  # no start to compare with: the window says no direction
                self.note_once(f"deeper-start:{request.get('run')}", f"Go deeper: no start to compare with "
                                                                       f"({problem(exc)})")
                start = None
            if fresh:                                 # the init frame's count read once for the run: apart too
                load_ms += (time.perf_counter() - loading) * 1e3
        with one_blas_thread():
            result = explain_cell(c, DeepFrame(totals_h, centres, float(block["fit_min"]), int(block["k"])),
                                  self.deep_fetch(counts, gene_map, encoder["ngenes"]),
                                  model["deep"], encoder, names, rows, code, float(block["judge_min"]),
                                  {"run": request.get("run"), "seq": request.get("seq"), "iter": request.get("iter"),
                                   "model": block.get("model_id"), "count": count},
                                  same_prev=block.get("same_prev") or None, start=start)
        # worker_ms: the op's own share of its pass (its frame's preparation and the recipe); loading the model's details
        # or the init frame's count, once per model version or run, is load_ms (the harness's budget is the former).
        result["worker_ms"] = round((time.perf_counter() - began) * 1e3 - load_ms, 2)
        if load_ms:
            result["load_ms"] = round(load_ms, 2)
        return result

    def start_rows(self, run: RunState, request, read, c: int, gene_map: np.ndarray, ngenes: int) -> dict:
        """Cell c's rows for its like-for-like change since the start, over the model's genes: {"x0": its start-block row
        (proseg's iteration-0 assignment kept as proseg keeps RNA at this frame, on the init frame's rows), "x"?: its
        averaged count now, when the request reads another count (`request_count`: f, the count proseg saves, or D0's
        sampled count)}. `read`: the frame's counts file when the pass read it."""
        init, block = self.init_counts(run, request), self.block_counts(run, request, read)
        if (init.ncells, init.ngenes) != (block.ncells, block.ngenes) or block.start_e.size != init.nnz:
            raise CountsError(f"the start block holds {block.start_e.size} entries, the init frame's count {init.nnz}")
        a, b = int(init.indptr[c]), int(init.indptr[c + 1])
        out = {"x0": model_row(init.genes[a:b], block.start_e[a:b], gene_map, ngenes)}
        if counts_file.ARRAY[request_count(request)] != "e":
            now = read if read is not None else self.read_counts(run, request)
            e = counts_file.rows(now, [c], "e")["e"][0]
            genes = np.nonzero(e)[0]
            out["x"] = model_row(genes, e[genes], gene_map, ngenes)
        return out

    def deeper_start(self, request) -> dict:
        """Go deeper on one cell of the start line: proseg's count of the init frame (kept with the run): asked alone
        (as the scheduler asks), the request's "counts_file" (frame 0's) unless it names "init_counts_file"; riding on a
        frame's request, its "init_counts_file" only (NoCounts without one: the frame's own file is never read as the
        start). Centres and bytes from start.npz (beside deeper.codes, start.bin); without start.npz, the prior
        footprint's centres and no stored byte."""
        with self.pooled(), no_collection():
            began = time.perf_counter()
            xp = self.xp
            run = self.state(request)
            block = {**QC_DEFAULTS, "k": DEEP_K, **request["deeper"]}
            ncells = int(request["ncells"])
            c = deeper_cell(block, ncells)
            loading = time.perf_counter()
            fresh = str(model_folder(block["model"]) / "deep.npz") not in self.deeps
            model = self.deep_model(block["model"])
            load_ms = (time.perf_counter() - loading) * 1e3 if fresh else 0.0     # once per model version: apart
            encoder, stamp = model["encoder"], model["stamp"]
            _, in_model, _ = self.run_genes(run, encoder, stamp)
            loading = time.perf_counter()
            alone = request_ops(request) == ["deeper"]
            field = "init_counts_file" if request.get("init_counts_file") or not alone else "counts_file"
            source = request.get(field)
            if not source:
                raise NoCounts("the request names no counts file of the init frame")
            try:
                counted = (str(source), self.stat_ns(source))
            except FileNotFoundError:
                raise NoCounts(f"the init frame's counts file is gone ({Path(source).name})") from None
            path = block.get("start_file") or start_file(block.get("codes"))
            try:
                stored = (str(path), self.stat_ns(path)) if path else None
            except OSError:
                stored = None
            # The start line's frame (its counts, totals, centres and bytes) never changes within a run: worked out once
            # per run and model, and that time is load_ms too.
            key = (counted, encoder["path"], stamp, stored, ncells)
            if run.deep_start_frame is not None and run.deep_start_frame[0] == key:
                _, counts, totals_h, centres, codes = run.deep_start_frame
            else:
                run.deep_start_frame = None              # the old frame's counts go before the new ones are made
                counts = self.matrix(self.read_counts(run, request, field))
                totals_h = host(xp.asarray(counts @ in_model).ravel()).astype(np.float64)
                values = None
                if stored is not None:
                    with np.load(path, allow_pickle=False) as found:
                        values = {k: found[k] for k in ("code", "cx", "cy")}
                if values is not None and len(values["code"]) == ncells:
                    centres = np.stack([values["cx"], values["cy"]], axis=1).astype(np.float64)
                    codes = np.asarray(values["code"])
                else:
                    geometry = prior_request(request)
                    if geometry is None:
                        raise ValueError("this run has no starting cells to place (no prior)")
                    _, cx, cy = footprint(xp, self.raster(run, geometry, plane="label", layers_at=0), ncells,
                                          float(geometry["x0"]), float(geometry["y0"]), float(geometry["voxel"]))
                    centres = np.stack([host(cx), host(cy)], axis=1).astype(np.float64)
                    codes = None
                run.deep_start_frame = (key, counts, totals_h, centres, codes)
                load_ms += (time.perf_counter() - loading) * 1e3
            code = int(codes[c]) if codes is not None else stored_code(block.get("codes"), c, ncells)
            names, rows = self.deep_cells(run, model, block.get("cell_ids"), ncells)
            with one_blas_thread():
                result = explain_cell(c, DeepFrame(totals_h, centres, float(block["fit_min"]), int(block["k"])),
                                      self.deep_fetch(counts, self.gene_map(run, encoder, stamp), encoder["ngenes"]),
                                      model["deep"], encoder, names, rows, code, float(block["judge_min"]),
                                      {"run": request.get("run"), "seq": "start", "iter": None,
                                       "model": block.get("model_id"), "count": counts_file.LIVE_COUNT},
                                      same_prev=block.get("same_prev") or None)
            # As for a frame's cell (deeper): loading the model's details and the init frame's count and working out its
            # frame are load_ms, once per model version or run; worker_ms is the rest.
            result["worker_ms"] = round((time.perf_counter() - began) * 1e3 - load_ms, 2)
            if load_ms:
                result["load_ms"] = round(load_ms, 2)
            return result

    # ---- Compare ------------------------------------------------------------------
    def compare_reply(self, request) -> dict:
        """The reply to op "compare": {ok, run, seq (null: the engine tells replies from a restarted worker's hello by
        it), compare (`compare`)}; when the op failed as a whole (the model could not be read, no column could be
        counted), "ok": false with "errors": {"compare": message} and "error". Its blocks go back to the driver after
        it: a compare is rare, and its peak must not stay on the GPU."""
        reply = {"run": request.get("run"), "seq": request.get("seq")}
        self.released = 0
        try:
            reply["compare"] = self.compare(request)
            reply["ok"] = True
        except Exception as exc:
            report(request)
            reply.update(ok=False, errors={"compare": problem(exc)}, error=problem(exc))
        self.release_stale_blocks(qc_only=True)
        return reply

    def compare(self, request) -> dict:
        """Op "compare" (the module docstring; the Compare section): the columns counted, the area all of them cover
        (the engine's, when it names every column's cells on it), each column's numbers on it, the model's kinds and one
        shared layout of the columns' judged cells on the area, split back into one points file per column (its
        out_points, else out_dir/<column index>.bin: POINT records, in the column's cell order). Returns {model, area
        {box, size_um, tiles, tile_um}, kinds [{index, name, color, genes, share}] | None,
        kinds_reason (None | "needs_detail": no deep.npz yet | "no_source": no kinds.npz and nothing to make it from |
        "busy": a save into the sample held its learning table, kinds next time | "failed"), kinds_error, columns [{key,
        name, state ("ready" | "failed"), error, cells, cells_in_area, judged, mixed, mixed_per100, like100_per100,
        like100_sd, like100_cells, histogram, median_tx, under100_share, tiles (holding AREA_MIN_CELLS of its cells),
        umap_points, sampled, points_file (None: no layout, or its folder is gone)}], points, layout ("umap" | "pca" |
        None), layout_error ("too_few_cells" | a message | None), bounds (the layout's view box: one for every column),
        seed, max_points, ms, gpu}. A run's live clusters, layout and colours are never read nor changed."""
        xp = self.xp
        thresholds = {**(request.get("thresholds") or {}), **{k: request[k] for k in QC_DEFAULTS if k in request}}
        qc = {**QC_DEFAULTS, **{k: v for k, v in thresholds.items() if k in QC_DEFAULTS and v is not None}}
        folder = model_folder(request["model"])
        seed = int(request.get("seed") or 0)
        cap = int(request.get("max_points") or COMPARE_MAX_POINTS)
        specs = list(request.get("columns") or [])
        if not specs:
            raise ValueError("nothing to compare: no columns")
        # The engine may work the area out itself (engine/compare.py) and name each column's cells on it ("cells");
        # else it is worked out here (shared_area), inside the request's box when it names one.
        given = all(spec.get("cells") is not None for spec in specs)
        limit = None if given else area_limit(request.get("area"))
        watch = Stopwatch(self.sync)
        columns = [{"key": str(spec.get("key")), "name": spec.get("name"), "state": "failed", "error": None,
                    "cells": 0, "cells_in_area": 0, "judged": 0, "mixed": 0, "mixed_per100": None,
                    "like100_per100": None, "like100_sd": None, "like100_cells": 0,
                    "histogram": [0] * (len(HIST_EDGES) + 1), "median_tx": None, "under100_share": None, "tiles": 0,
                    "umap_points": 0, "sampled": False, "points_file": None, "own_points_file": None, "clusters_file": None,
                    "clusters": 0, "clusters_matched": 0} for spec in specs]
        with self.pooled():
            encoder, device, _ = self.model(folder / "encoder.npz")
            w0t = xp.ascontiguousarray(device["W0"].T)              # the calculator's first layer by gene
            counted = {}
            for i, spec in enumerate(specs):
                try:
                    column = self.compare_counts(spec, encoder, seed)
                    if given:
                        column["inside"] = area_rows(spec["cells"], column["csr"].shape[0])
                    counted[i] = column
                except Exception as exc:                            # one column failing never costs the others
                    columns[i]["error"] = problem(exc)              # (a frame without proseg's count: one line)
                    why = problem(exc) if getattr(exc, "reason", None) == "no_counts" else traceback.format_exc()
                    log(f"compare: column {columns[i]['key']!r} could not be counted: {why}")
            if not counted:
                raise ValueError("no column could be counted: " + "; ".join(c["error"] for c in columns))
            watch.mark("counts")
            if given:
                area, inside, tiles = given_area(request.get("area")), [c["inside"] for c in counted.values()], \
                    [None] * len(counted)
            else:
                area, inside, tiles = shared_area([(c["x"], c["y"]) for c in counted.values()], limit)
            judged = {}
            for (i, c), mask, own in zip(counted.items(), inside, tiles):
                c["totals"], c["share"] = score_csr(xp, c["csr"], w0t, device, float(qc["fit_min"]))
                numbers, judged[i], c["codes"] = compare_numbers(host(c["totals"]), host(c["share"]), mask, qc)
                columns[i].update(numbers, state="ready", cells=c["csr"].shape[0], tiles=own)
            watch.mark("score")
            for i, c in counted.items():                            # cut from whole transcripts (the run's: f, or
                whole = c.get("whole", c["csr"])                    # Binomial(n, e / n) draws)
                depth = c["totals"] if whole is c["csr"] else csr_row_sums(xp, whole)
                rows = judged[i][host(depth)[judged[i]] >= LIKE_DEPTH]
                like = like_for_like_at(xp, whole, rows, w0t, device, qc,
                                        [compare_seed(seed, columns[i]["key"], d) for d in range(LIKE_DRAWS)])
                columns[i].update(like100_per100=like["per100"], like100_sd=like["sd"], like100_cells=like["cells"])
            watch.mark("like")
            kinds, kinds_reason, kinds_error = self.compare_kinds(request, folder, encoder)
            watch.mark("kinds")
            blocks, parts = [], []                                  # the points: each column's share of the cap
            for (i, c), quota in zip(counted.items(), even_quotas([len(judged[i]) for i in counted], cap)):
                rows = judged[i][picked(len(judged[i]), quota, seed, i)]
                block = csr_take(xp, c["csr"], rows)
                totals = c["totals"][xp.asarray(rows)]
                kind = np.full(len(rows), NO_KIND, np.uint8) if kinds is None else \
                    nearest_kind(xp, z_means(xp, kinds["device"], block, totals), kinds["centres"])
                blocks.append(block)
                parts.append((i, kind, c["codes"][rows], totals))
                columns[i].update(umap_points=int(len(rows)), sampled=bool(len(rows) < len(judged[i])))
            del counted, c, block, totals                           # the columns' counts go before the layout
            n_points = sum(len(kind) for _, kind, _, _ in parts)
            points = csr_stack(xp, blocks) if n_points else None
            point_totals = xp.concatenate([totals for *_, totals in parts]) if n_points else None
            del blocks
            watch.mark("points")
        self.released += self.release_stale_blocks()             # before the layout's own arrays (RMM), as umap()
        layout = how = layout_error = bounds = None
        own = {"layouts": [None] * len(parts), "clusters": [None] * len(parts), "counts": [0] * len(parts),
               "matched": [0] * len(parts), "how": None, "bounds": None, "error": None}
        try:
            if n_points < max(30, int(DEFAULTS["n_neighbors"]) + 2):
                raise TooFewCells(f"only {n_points} judged cells to lay out")
            dense = log_normalised(xp, points, point_totals, DEFAULTS["n_top_genes"])
            del points, point_totals
            layout, how = self.compare_layout(dense)
            bounds = layout_bounds(layout)
            # "Each its own": every column laid out and clustered alone, from its own rows of the same genes, turned
            # to line up with the run's column (`own_maps`); the shared map stands if this fails. Only when the request
            # asks (`own`: the engine's call with every column, so the run's column comes first without waiting).
            if request.get("own", True):
                try:
                    at, host_parts = 0, []
                    for _, kind, _, _ in parts:
                        host_parts.append(np.asarray(host(dense[at:at + len(kind)]), np.float32)); at += len(kind)
                    reference = next((k for k, (i, *_rest) in enumerate(parts) if specs[i].get("frame")), 0)
                    own = own_maps(self, host_parts, [kind for _, kind, _, _ in parts], reference, self.compare_own)
                except Exception as exc:
                    own["error"] = problem(exc)
                    log(f"compare: no own maps: {traceback.format_exc()}")
            del dense
        except TooFewCells:
            layout_error = "too_few_cells"
        except Exception as exc:                                    # the numbers stand without the picture
            layout_error = problem(exc)
            log(f"compare: no layout: {traceback.format_exc()}")
        watch.mark("layout")
        out_dir = Path(request["out_dir"]) if request.get("out_dir") else None
        if out_dir is not None:
            with suppress(OSError):
                out_dir.mkdir(exist_ok=True)                        # the folder itself, never its parents
        at = 0
        for part, (i, kind, codes, _) in enumerate(parts):
            # Each column's points go to its own "out_points" when the request names one, else to out_dir/<index>.bin;
            # never into a folder that is gone (a run deleted meanwhile must stay deleted): no points for it then.
            path = Path(specs[i]["out_points"]) if specs[i].get("out_points") else \
                None if out_dir is None else out_dir / f"{i}.bin"
            if layout is None:
                columns[i].update(umap_points=0, sampled=False)
            elif path is not None:
                records = np.zeros(len(kind), POINT)
                records["x"], records["y"] = layout[at:at + len(kind), 0], layout[at:at + len(kind), 1]
                records["kind"], records["share"] = kind, codes
                try:
                    write_atomic(path, records.tobytes())
                    columns[i]["points_file"] = str(path)
                except OSError as exc:
                    columns[i].update(umap_points=0, sampled=False)
                    log(f"compare: the points of {columns[i]['key']!r} were not written: {problem(exc)}")
                # its own map (the same points, in the same order: x, y its own; kind and share as above) and its own
                # clusters (u16 per point), beside the shared map's file
                mine, ids = own["layouts"][part], own["clusters"][part]
                if columns[i]["points_file"] and mine is not None:
                    records["x"], records["y"] = mine[:, 0], mine[:, 1]
                    try:
                        write_atomic(own_path(path), records.tobytes()); write_atomic(clusters_path(path), ids.tobytes())
                        columns[i].update(own_points_file=str(own_path(path)), clusters_file=str(clusters_path(path)),
                                          clusters=own["counts"][part], clusters_matched=own["matched"][part])
                    except OSError as exc:
                        log(f"compare: the own map of {columns[i]['key']!r} was not written: {problem(exc)}")
            at += len(kind)
        for i, column in enumerate(columns):                        # no older picture for a column without one now
            if column["points_file"] is None:
                stale = specs[i].get("out_points") or (out_dir / f"{i}.bin" if out_dir is not None else None)
                if stale:
                    with suppress(OSError):
                        Path(stale).unlink()
            if column["own_points_file"] is None:                   # nor an older own map
                stale = specs[i].get("out_points") or (out_dir / f"{i}.bin" if out_dir is not None else None)
                if stale:
                    for gone in (own_path(Path(stale)), clusters_path(Path(stale))):
                        with suppress(OSError):
                            gone.unlink()
        result = {"model": request.get("model_id"), "area": area, "kinds": None if kinds is None else kinds["public"],
                  "kinds_reason": kinds_reason, "kinds_error": kinds_error, "columns": columns,
                  "points": sum(c["umap_points"] for c in columns), "layout": how, "layout_error": layout_error,
                  "bounds": bounds, "seed": seed, "max_points": cap,
                  "own_layout": own["how"], "own_bounds": own["bounds"], "own_error": own["error"]}
        watch.mark("write")
        result.update(ms=watch.ms, gpu=self.pool_mb())
        if request.get("out_json"):
            write_atomic(request["out_json"], json.dumps(result).encode())
        return result

    def compare_counts(self, spec: dict, encoder: dict, seed: int = 0) -> dict:
        """A column's counts over the model's genes (Csr on the device) and its cells' centres (x, y: host float64,
        NaN for none): the run's frame (compare_frame: proseg's count, from its counts file) or a counts file
        (read_counts_file). `seed`: the request's (the run's binomial draws)."""
        if spec.get("frame"):
            return self.compare_frame(dict(spec["frame"]), encoder, compare_seed(seed, spec.get("key"), "whole"))
        if not spec.get("counts"):
            raise ValueError("a column needs a counts file or the run's frame")
        xp = self.xp
        m, _, x, y, genes = read_counts_file(spec["counts"])
        index, _, matched = gene_index(encoder, genes)
        if not matched:
            raise ValueError("none of its genes is in the model (is it this sample's?)")
        m = Csr(xp.asarray(m.data), xp.asarray(m.indices), xp.asarray(m.indptr), m.shape)
        return {"csr": csr_columns(xp, m, index, encoder["ngenes"]), "x": x, "y": y}

    def compare_frame(self, frame: dict, encoder: dict, seed: int) -> dict:
        """The run's column: proseg's count of the frame (its counts file: the count the frame names, `request_count`: e,
        or f, the count proseg saves, on a frame counted for a pause, the hold, a finish or the end; the column's own
        count, as each column reads its own), and whole transcripts for its like-for-like cut ("whole": proseg's sampled
        count f where the frame has it, else one Binomial(n, e / n) draw from `seed`: thin_values); centres from the
        frame's cell statistics (NaN for a cell without one). Its clusters, layout and colours are never read nor changed
        (only its genes: RunState)."""
        xp = self.xp
        run = self.state(frame)
        ncells = int(frame["ncells"])
        index, _, matched = gene_index(encoder, run.genes)
        if not matched:
            raise ValueError("none of this run's genes is in the model (is it this sample's model?)")
        read = self.read_counts(run, frame)
        stats = read_centres(frame.get("cells_file"), ncells)
        x = np.full(ncells, np.nan) if stats is None else np.asarray(stats["cx"], np.float64)
        y = np.full(ncells, np.nan) if stats is None else np.asarray(stats["cy"], np.float64)
        width = int(encoder["ngenes"])

        def column(arrays: counts_file.CsrArrays, data=None) -> Csr:
            arrays = without_zeros(arrays._replace(data=arrays.data if data is None else data))
            m = Csr(xp.asarray(arrays.data), xp.asarray(arrays.indices), xp.asarray(arrays.indptr.astype(np.int64)),
                    arrays.shape)
            return csr_columns(xp, m, index, width)
        live = column(counts_file.to_csr(np, read, counts_file.ARRAY[request_count(frame)]))
        if read.f is not None:
            whole = column(counts_file.to_csr(np, read, "f"))
        else:
            n = counts_file.to_csr(np, read, "n")
            keep = np.minimum(np.asarray(read.e, np.float32) / np.maximum(n.data, 1), 1).astype(np.float32)
            drawn = host(thin_values(xp, xp.asarray(n.data), xp.asarray(keep), seed)).astype(np.float32)
            whole = column(n, drawn)
        return {"csr": live, "whole": whole, "x": x, "y": y}

    def compare_kinds(self, request, folder: Path, encoder: dict):
        """(the model's kinds, None, None), or (None, reason, message). The kinds: kinds.npz (the request's kinds
        path, else beside deep.npz) when it belongs to the model's details, else made once (make_kinds) from the
        model's own cells (kinds_source) and written there; kept here per file version: {centres and device (the z
        encoder's arrays) on the device, public [{index, name, color, genes, share}], cells, sha, stamp}. Reasons:
        "needs_detail" (no deep.npz: Go deeper's Learn the details), "no_source" (nothing to make kinds.npz from),
        "busy" (a save into the sample held its learning table past KINDS_LOCK_WAIT_S: nothing is made nor kept, a later
        Compare makes them), "failed" (logged)."""
        xp = self.xp
        spec = request.get("kinds")
        spec = {"path": spec} if isinstance(spec, str) else dict(spec or {})
        if not (folder / "deep.npz").is_file():
            return None, "needs_detail", "the model has no details yet (deep.npz)"
        path = Path(spec.get("path") or folder / KINDS_FILE)
        try:
            model = self.deep_model(folder)                         # its details, checked (once per version)
            deep = model["deep"]
            try:
                written = path.stat().st_mtime_ns
            except OSError:
                written = None
            cached = self.kind_sets.get(str(path))
            if cached is not None and cached["stamp"] == (model["stamps"], written):
                self.kind_sets.move_to_end(str(path))
                return cached, None, None
            sha = file_sha256(folder / "deep.npz")
            stored = None if written is None else load_kinds(path, sha, deep["Zm"].shape[0], len(deep["genes"]))
            if stored is None:
                source = self.kinds_source(spec, folder)
                if source is None:
                    return None, "no_source", "nothing to make the model's kinds from (its learning cells)"
                stored = self.make_kinds(deep, encoder, source, sorted(model["rows"], key=model["rows"].get), sha)
                try:
                    write_kinds(path, stored)
                    written = path.stat().st_mtime_ns
                except OSError as exc:                              # kept here only: made again by the next worker
                    self.note_once(f"kinds:{path}", f"compare: the kinds could not be kept in {path}: {problem(exc)}")
            kinds = {"stamp": (model["stamps"], written), "centres": xp.asarray(stored["centres"]),
                     "device": deep_device(xp, deep), "cells": stored["cells"], "sha": sha,
                     "public": [{"index": k, "name": stored["names"][k], "color": stored["colors"][k],
                                 "genes": [str(deep["genes"][g]) for g in stored["genes"][k]],
                                 "share": round(float(stored["sizes"][k]) / max(stored["cells"], 1), 4)}
                                for k in range(len(stored["names"]))]}
        except TimeoutError as exc:                                 # a save into the sample: not now
            log(f"compare: no kinds this time: {problem(exc)}")
            return None, "busy", problem(exc)
        except Exception as exc:
            log(f"compare: no kinds: {traceback.format_exc()}")
            return None, "failed", problem(exc)
        self.kind_sets[str(path)] = kinds
        while len(self.kind_sets) > MODELS_KEPT:
            self.kind_sets.popitem(last=False)
        return kinds, None, None

    @staticmethod
    def kinds_source(spec: dict, folder: Path):
        """The model's own cells to make its kinds from, (Csr on the host, cell ids, gene names): the request's
        kinds.counts (a counts file naming its cells' ids), else its learning table (learning_counts, waiting at most
        KINDS_LOCK_WAIT_S for a save into the sample: TimeoutError after that): the request's kinds.table {zarr, key},
        else the one the model's model.json names (the trainer's "zarr" and "table" key); None without any."""
        if spec.get("counts"):
            m, ids, _, _, genes = read_counts_file(spec["counts"])
            if ids is None:
                raise ValueError("the learning cells' counts file names no cell ids")
            return m, ids, genes
        table = spec.get("table") or {}
        if not (table.get("zarr") and table.get("key")):
            try:
                learned = json.loads((folder / "model.json").read_text())
                key = learned.get("table")
                table = {"zarr": learned.get("zarr"), "key": key.get("key") if isinstance(key, dict) else key}
            except (OSError, ValueError, AttributeError):
                return None
        if table.get("zarr") and table.get("key"):
            return learning_counts(table["zarr"], table["key"], wait_s=KINDS_LOCK_WAIT_S)
        return None

    def make_kinds(self, deep: dict, encoder: dict, source, learned: list, sha: str) -> dict:
        """The model's kinds from its own cells (`learned`: their ids in deep.npz's training-row order) as `source`
        counts them: each one's description (z_means) over the model's genes, grouped (choose_kinds), named
        (kind_names) and coloured by index (KIND_PALETTE)."""
        xp = self.xp
        m, ids, genes = source
        position = {str(c): i for i, c in enumerate(ids)}
        rows = np.fromiter((position.get(str(c), -1) for c in learned), np.int64, len(learned))
        rows = rows[rows >= 0]
        if not len(rows) or len(rows) < 0.5 * len(learned):         # another table than the one it learned from
            raise ValueError(f"only {len(rows)} of the model's {len(learned)} cells were found to make its kinds from")
        index, _, matched = gene_index(encoder, genes)
        if not matched:
            raise ValueError("none of the learning cells' genes is in the model")
        m = Csr(xp.asarray(m.data), xp.asarray(m.indices), xp.asarray(m.indptr), m.shape)
        m = csr_columns(xp, csr_take(xp, m, rows), index, encoder["ngenes"])
        totals = csr_row_sums(xp, m)
        z = z_means(xp, deep_device(xp, deep), m, totals)
        z = z[xp.isfinite(z).all(axis=1)]
        centres, _, sizes = choose_kinds(xp, z)
        names, chosen = kind_names(deep, centres)
        return {"centres": centres, "sizes": sizes, "names": names, "genes": chosen, "cells": int(z.shape[0]),
                "colors": [KIND_PALETTE[k % len(KIND_PALETTE)] for k in range(len(names))], "sha": sha}

    def compare_layout(self, dense) -> tuple[np.ndarray, str]:
        """The shared layout of the points (dense [n, genes]: their log-normalised top genes, centred here): cuML's
        PCA, neighbours and a cold UMAP, as umap() lays a run out afresh (spectral start, DEFAULTS' cold epochs), then
        normalised to RADIUS: (layout [n, 2], "umap"); on the CPU the same with umap-learn (analysis/cpu.py
        compare_layout); with numpy alone (tests, the fake worker) the first two principal components stand in for the
        UMAP ("pca")."""
        if self.backend == "cpu":
            from .cpu import compare_layout
            return compare_layout(dense)
        xp, p = self.xp, DEFAULTS
        n = int(dense.shape[0])
        if n < max(30, int(p["n_neighbors"]) + 2):
            raise TooFewCells(f"only {n} judged cells to lay out")
        dense -= dense.mean(axis=0)
        if xp is np:
            u, s, _ = np.linalg.svd(dense.astype(np.float64), full_matrices=False)
            two = np.zeros((n, 2), np.float32)
            two[:, :min(2, len(s))] = u[:, :2] * s[:2]
            return normalise(two), "pca"
        import cupyx.scipy.sparse as cps
        from cuml.decomposition import PCA
        from cuml.manifold import UMAP
        from cuml.neighbors import NearestNeighbors
        n_pcs = principal_components(p["n_pcs"], n, dense.shape[1])
        pcs = PCA(n_components=n_pcs, output_type="cupy").fit_transform(dense)
        nn = int(min(p["n_neighbors"], n - 1))
        distances, neighbours = NearestNeighbors(n_neighbors=nn, output_type="cupy").fit(pcs).kneighbors(pcs)
        rows = xp.repeat(xp.arange(n, dtype=xp.int32), nn)
        knn = cps.csr_matrix((distances.ravel().astype(xp.float32), (rows, neighbours.ravel().astype(xp.int32))),
                             shape=(n, n))
        embedding = UMAP(n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]), init="spectral",
                         n_epochs=int(p["cold_epochs"]), learning_rate=1.0,
                         output_type="numpy").fit_transform(pcs, knn_graph=knn)
        return normalise(np.asarray(embedding, np.float32)), "umap"

    def compare_own(self, dense) -> tuple[np.ndarray, np.ndarray, str]:
        """One column's own map (Compare's "each its own"): the recipe of compare_layout on its points alone, and its
        own clusters, Leiden on the same neighbour graph with the run's Clusters defaults: (layout [n, 2] normalised,
        clusters int32 [n], "umap"); on the CPU analysis/cpu.py compare_own; with numpy alone (tests, the fake worker)
        the first two principal components, and their four quadrants about the median for clusters ("pca")."""
        if self.backend == "cpu":
            from .cpu import compare_own
            return compare_own(dense)
        xp, p = self.xp, DEFAULTS
        n = int(dense.shape[0])
        if n < max(30, int(p["n_neighbors"]) + 2):
            raise TooFewCells(f"only {n} judged cells to lay out")
        dense -= dense.mean(axis=0)
        if xp is np:
            u, sv, _ = np.linalg.svd(dense.astype(np.float64), full_matrices=False)
            two = np.zeros((n, 2), np.float32)
            two[:, :min(2, len(sv))] = u[:, :2] * sv[:2]
            med = np.median(two, axis=0)
            return normalise(two), ((two[:, 0] > med[0]) * 2 + (two[:, 1] > med[1])).astype(np.int32), "pca"
        import cudf
        import cugraph
        import cupyx.scipy.sparse as cps
        from cuml.decomposition import PCA
        from cuml.manifold import UMAP
        from cuml.neighbors import NearestNeighbors
        n_pcs = principal_components(p["n_pcs"], n, dense.shape[1])
        pcs = PCA(n_components=n_pcs, output_type="cupy").fit_transform(dense)
        nn = int(min(p["n_neighbors"], n - 1))
        distances, neighbours = NearestNeighbors(n_neighbors=nn, output_type="cupy").fit(pcs).kneighbors(pcs)
        rows = xp.repeat(xp.arange(n, dtype=xp.int32), nn)
        knn = cps.csr_matrix((distances.ravel().astype(xp.float32), (rows, neighbours.ravel().astype(xp.int32))),
                             shape=(n, n))
        edges = cudf.DataFrame({"src": rows, "dst": neighbours.ravel().astype(xp.int32), "w": xp.ones(n * nn, dtype=xp.float32)})
        graph = cugraph.Graph()
        graph.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="w", renumber=False, symmetrize=True)
        parts, _ = cugraph.leiden(graph, resolution=float(p["resolution"]), random_state=0)
        clusters = parts.sort_values("vertex")["partition"].values.get().astype(np.int32)
        embedding = UMAP(n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]), init="spectral",
                         n_epochs=int(p["cold_epochs"]), learning_rate=1.0,
                         output_type="numpy").fit_transform(pcs, knn_graph=knn)
        return normalise(np.asarray(embedding, np.float32)), clusters, "umap"


def own_path(path: Path) -> Path:
    """A column's own-map points file, beside its shared-map one (<stem>.own.bin)."""
    return path.with_name(path.stem + ".own.bin")


def clusters_path(path: Path) -> Path:
    """A column's own clusters file, beside its shared-map one (<stem>.clusters.bin)."""
    return path.with_name(path.stem + ".clusters.bin")


def own_maps(worker, dense_host_parts: list, kinds: list, reference: int, lay) -> dict:
    """Compare's "each its own": every column's points (`dense_host_parts`, numpy [n_i, genes] each, in the columns'
    order; `kinds` their nearest kinds) laid out and clustered alone (`lay`: Worker.compare_own), each column's clusters
    renumbered to the reference column's ids where their mean profiles match (`match_profiles`; the reference's own by
    size, the others' unmatched ones given ids of their own after every reference id), each layout but the reference's
    (every one the same size: normalise) turned or flipped and moved onto the reference's on the kinds' medians they
    share (else on the matched clusters'). {layouts, clusters (uint16 ids per point), counts [clusters per column],
    matched [matched per column], how, bounds, error}; a column too small for a map has none (its layout None)."""
    out = {"layouts": [None] * len(dense_host_parts), "clusters": [None] * len(dense_host_parts),
           "counts": [0] * len(dense_host_parts), "matched": [0] * len(dense_host_parts), "how": None, "bounds": None, "error": None}
    raw = []
    for dense in dense_host_parts:
        try:
            layout, local, how = lay(dense.copy())
            raw.append((layout, np.asarray(local, np.int64)))
            out["how"] = how
        except TooFewCells:
            raw.append(None)
    if all(r is None for r in raw):
        out["error"] = "too_few_cells"
        return out
    ref = reference if raw[reference] is not None else next(i for i, r in enumerate(raw) if r is not None)
    ref_layout, ref_local = raw[ref]
    ids, sizes = np.unique(ref_local, return_counts=True)
    by_size = ids[np.argsort(-sizes, kind="stable")]
    renumber = {int(c): k for k, c in enumerate(by_size)}
    ref_ids = np.array([renumber[int(c)] for c in ref_local], np.int64)
    rid, ref_prof = cluster_profiles(dense_host_parts[ref], ref_ids)
    next_id = len(by_size)
    for i, r in enumerate(raw):
        if r is None:
            continue
        layout, local = r
        if i == ref:
            ids_i, matched = ref_ids, len(by_size)
        else:
            cid, prof = cluster_profiles(dense_host_parts[i], local)
            pairs = match_profiles(ref_prof, prof)
            mapping = {int(cid[a]): int(rid[b]) for a, b in pairs.items()}
            order = cid[np.argsort(-np.array([int((local == c).sum()) for c in cid]), kind="stable")]
            for c in order.tolist():
                if c not in mapping:
                    mapping[c] = next_id; next_id += 1
            ids_i, matched = np.array([mapping[int(c)] for c in local], np.int64), len(pairs)
            src, dst, w = landmarks(layout, kinds[i], ref_layout, kinds[ref])
            if len(w) < 3:
                src, dst, w = landmarks(layout, ids_i, ref_layout, ref_ids)
            if len(w) >= 3:
                layout = align_onto(layout, src, dst, w)
        out["layouts"][i] = layout
        out["clusters"][i] = np.where(ids_i < NO_CLUSTER, ids_i, NO_CLUSTER).astype(CLUSTER)
        out["counts"][i], out["matched"][i] = int(len(np.unique(ids_i))), int(matched)
    every = np.concatenate([lay_ for lay_ in out["layouts"] if lay_ is not None])
    out["bounds"] = layout_bounds(every)
    return out


def model_folder(path) -> Path:
    """A model folder, also when a request names its encoder.npz (or deep.npz)."""
    path = Path(path)
    return path.parent if path.suffix == ".npz" else path


def deeper_cell(block: dict, ncells: int) -> int:
    """The cell a Go deeper request asks for (label - 1), checked against the run's cells."""
    cell = block.get("cell")
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < ncells:
        raise ValueError(f"no cell {cell!r} in this run ({ncells} cells)")
    return cell


def stored_code(path, c: int, ncells: int) -> int | None:
    """Cell c's byte in a stored .qc.bin or start.bin (a zstd frame, or raw), None when there is none."""
    if not path:
        return None
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    if raw[:4] == b"\x28\xb5\x2f\xfd":                   # a zstd frame, as write_codes writes them
        from numcodecs import Zstd
        raw = Zstd().decode(raw)
    return int(raw[c]) if len(raw) == ncells else None


def device_bus(xp) -> str | None:
    """The PCI bus id of the GPU this process uses ("0000:01:00.0"), None when it cannot be told."""
    try:
        return str(xp.cuda.Device().pci_bus_id)
    except Exception:                                     # numpy, or a cupy without the attribute
        return None


class Lines:
    """The lines of a stream (the engine's requests), each read only once it is complete, with a timeout: the
    descriptor is read directly, so no line ever waits in a buffer that select() cannot see."""

    def __init__(self, stream):
        self.fd = stream.fileno()
        self.buffer = b""
        self.ended = False

    def get(self, timeout: float | None = None) -> str | None:
        """The next line (without its newline); None when none came within `timeout` seconds (None: wait as long as
        it takes); EOFError once the stream has ended and every line was given."""
        while b"\n" not in self.buffer:
            if self.ended:
                if not self.buffer:
                    raise EOFError
                line, self.buffer = self.buffer, b""
                return line.decode()
            if not portable.readable(self.fd, timeout):      # select on POSIX; Windows' select takes sockets only
                return None
            chunk = os.read(self.fd, 1 << 16)
            self.ended = not chunk
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode()


def refusal(request, message: str, reason: str) -> dict:
    """The reply to a request the worker cannot take (an op it does not know, a line that is not a request): every op
    it names fails at once with a reason, so the engine never waits for an answer that would not come."""
    request = request if isinstance(request, dict) else {}
    named = request.get("ops") if isinstance(request.get("ops"), list) else []
    ops = [str(op) for op in named] or [str(request.get("op") or "request")]
    return {"ok": False, "run": request.get("run"), "seq": request.get("seq"), "error": message,
            "errors": dict.fromkeys(ops, message), "reasons": dict.fromkeys(ops, reason)}


def answer(worker: Worker, line: str) -> dict | None:
    """The reply to one request line (None for "forget", which is never answered, and for a blank line)."""
    if not line.strip():
        return None
    try:
        request = json.loads(line)
    except ValueError:
        return refusal(None, "ValueError: the request is not JSON", "unreadable")
    if not isinstance(request, dict):
        return refusal(None, "ValueError: the request is not a JSON object", "unreadable")
    if request.get("op") == "forget":
        worker.runs.pop(request.get("run"), None)
        worker.release_stale_blocks(qc_only=True)         # its transcripts, if cupy's pool held them
        return None
    if request.get("op") not in OPS:
        return refusal(request, f"ValueError: unknown op {request.get('op')!r}", "unknown_op")
    return worker.handle(request)


def serve(worker: Worker, stream, out=None, idle_s: float = WARM_IDLE_S):
    """Requests from `stream`, one JSON object per line, until it ends; each answered on `out` (stdout) with one line
    of compact JSON, also a request the worker cannot take (`answer`: the engine reads one reply per request). Whenever
    no request has come for idle_s, the next of worker.warm_steps runs, one at a time, so a request waits for at most
    one step; a failing step ends the warm-up (the first UMAP pass then loads what it needs itself, as before)."""
    out = out or sys.stdout
    lines = Lines(stream)
    while True:
        try:
            line = lines.get(idle_s if worker.warm_steps else None)
        except EOFError:
            return
        if line is None:
            step = worker.warm_steps.pop(0)
            started = time.perf_counter()
            try:
                step()
                log(f"warm-up {step.__name__}: {(time.perf_counter() - started) * 1e3:.0f} ms")
            except Exception:
                log("warm-up stopped:", traceback.format_exc())
                worker.warm_steps = []
            continue
        reply = answer(worker, line)
        if reply is not None:
            print(json.dumps(reply, separators=(",", ":")), file=out, flush=True)


def not_ready(choice: dict, error: str) -> int:
    """The hello of a worker that cannot run in this Python (main): the packages it lacks, whether that holds for the
    rest of the session (the engine then stops asking), and why; exit code 1."""
    print(json.dumps({"ok": False, "ready": False, "missing": list(choice["missing"]),
                      "lasting": bool(choice["lasting"]), "error": error}), flush=True)
    return 1


def main():
    """The worker process: its backend chosen ([analysis] backend, as the engine passes it) and its libraries loaded,
    then the hello (every way this can fail prints one too: not_ready), the idle-time warm-up and the requests."""
    # A background helper: under load, the viewer, the desktop and proseg come first (cuML/cuGraph poll the GPU).
    portable.lower_priority(10)
    setting = os.environ.get(backend.BACKEND_ENV) or "auto"
    choice = {"backend": None, "why": None, "missing": [], "lasting": False}
    try:
        choice = pick_backend(setting)
        if choice["backend"] is None:
            return not_ready(choice, f"Clusters & UMAP cannot run in this Python environment: {cannot(choice)}")
        if choice["backend"] == "cpu":
            from . import cpu
            cpu.enable()      # threads first, then numba (its NumPy check), scipy, scikit-learn, igraph, umap-learn
            worker = Worker(xp=np, backend="cpu")
        else:
            import cupy
            worker = Worker(xp=cupy, backend="gpu")   # gpu_waits_block already ran in pick_backend
    except Exception as exc:                          # pick_backend itself, cpu.enable() or Worker()
        kind = {"cpu": "CPU", "gpu": "GPU"}.get(choice.get("backend"), "analysis")
        # CPU libraries that cannot load last for the session, unless the CPU was taken only because the GPU's stack,
        # installed, missed its device now: that may come back (gpu_closed), and its reason comes first.
        lasting = kind == "CPU" and gpu_closed(setting)
        reason = f"its {kind} libraries could not be loaded ({problem(exc)})"
        if kind == "CPU" and not lasting and choice.get("why"):
            reason = f"{choice['why']}, and {reason}"
        return not_ready({**choice, "lasting": lasting},
                         f"Clusters & UMAP cannot run in this Python environment: {reason}")
    # The PCI bus of the one GPU this process sees, for the engine to check against nvidia-smi's ([analysis] gpu).
    print(json.dumps({"ok": True, "ready": True, "pci_bus_id": device_bus(worker.xp), "backend": choice["backend"],
                      "why": choice["why"], "threads": threads()}), flush=True)
    worker.warm_steps = worker.warm_umap_steps()          # taken while the worker is idle
    serve(worker, sys.stdin)
    return 0


if __name__ == "__main__":
    # Run as the package's own module rather than as this __main__ copy of it: analysis/cpu.py imports the worker
    # module, and the settings it makes there (CPU_THIN) must be the ones this process runs with.
    import importlib
    raise SystemExit(importlib.import_module(__spec__.name).main() if __spec__ is not None else main())
