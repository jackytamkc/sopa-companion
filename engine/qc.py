"""Mixing check (live ResolVI QC): each sample's per-cell mixing calculator (QcModels) and what the page is told about
a run (QcManager). Files and API: docs/ARCHITECTURE.md, "The mixing check", "HTTP API" and "Files".

QcModels learns one model per sample: analysis/qc_trainer.py in the sopa environment ([qc] python), on the one GPU
[qc] gpu names (0 here; engine/children.py maps it, PCI bus order: the only GPU the trainer sees, its PCI bus checked;
one the engine does not see, or one the trainer does not learn on, TRAINER_GPU, fails with reason "gpu" before
anything is read), or on a Mac without one on Apple's GPU (MPS) where it passes the probe's test (device_of, kept with
each sample's key: its learning slower, MPS_STEP_S and MPS_TIME_FACTOR, its room the Mac's own memory, apple_memory,
never waited for, and its models apart, model_key),
nice 19, 4 threads, one fit at a time
(two lock files: one keeps engines that share this models folder from fitting together, the other every engine of this
user on this computer, e.g. a test engine next to the app: fits on GPU 0 never overlap), as soon as that GPU has at
least the free memory a fit of its table needs (room_needed_mb, from its panel; [qc] gpu_min_free_mb when set: however
busy it is, room is what a fit needs; a card smaller than that is said at once), and for at most
time_limit_s.
Whether the check can learn here at all is probed in its Python at start (analysis/qc_probe.py: an NVIDIA card, torch
with CUDA that sees it, a driver new enough for torch's CUDA, or else Apple's GPU passing a short test of the learning's
own operations; scvi-tools and scviva-tools). Until it answers the check counts as offered; a definite no makes every
sample "unavailable" with the reason ("no_gpu", "apple_gpu", "software", "torch_cpu", "driver"); a probe that cannot
answer leaves the check offered and is asked again later (capability_of, _probe_loop).
A model is cached in <models>/<sample folder>/<model key>/ (encoder.npz, model.json, cells.parquet, batch.parquet,
train.log, and for Go deeper deep.npz); <models> is [qc] models_dir when set (API mode's model_cache, shared by
sessions: a model another session used within SHARED_KEEP_S is never pruned), else <data_dir>/qc-models, which always
keeps the page's switch (prefs.json), apart from shared models. The sample folder is that of the zarr that holds its
tables (table_home's sample id), with tables.json, the tables refused before.
The key hashes the zarr that really holds the learned table (its own path; for a table linked from another zarr, an
API session's view, that zarr's: table_home), the learned table's key
and fingerprint (n_obs, n_vars, nnz, mtime), the fingerprints of the sample's other [qc] tables (the earlier proseg cells: the priors are
pooled with them, and they are scored for one line), the recipe and the sopa environment's package versions: upgrading
scviva, scvi or torch means relearning, never two yardsticks in one trend. A fit is written to a hidden staging folder,
checked here (numpy: the exported arrays, and the 256 reference cells reproduced within 1e-5; deep.npz with its own
self-test) and only then renamed into place. A model learned before Go deeper
gets its deep.npz from a *detail* job (5.4: the same queue, fit locks, GPU 0 wait, nice 19, time limit and progress as a
fit; the trainer learns the same recipe again and keeps the details only when its calculator equals the installed one):
deep.npz is moved into the installed folder and model.json rewritten, the key, the yardstick and every run's results
stay; any failure changes nothing.

QcManager keeps, per run, the switch (off by default, the last choice remembered for new runs), the start line, the
headline and the flagged areas over the GPU worker's per-frame results, every one of them on proseg's own count of the
frame (its averaged count: analysis/counts_file.py; the drawn outlines are for drawing and positions only); a frame
proseg counted for a pause, the hold, a finish or the end shows its own numbers on the count proseg saves there (f,
counts_file.count_of "sampled"), while the smoothing, the arrow's word, the areas and the cards read its averaged count
as on any frame (`ingest`). It is the
analysis scheduler's QC provider (engine/analysis.py `attach_qc`: worker ops "qc" and "qc_start", analysis/worker.py):
the scheduler asks it which model scores a run and hands it every result with its per-bin sums, and asks whether a
knob change is being judged on a run (`watching`: its counted frames come first then). In
<run>/qc/<model key[:8]>.<counts_file.BASIS>/ (a folder per count basis: results scored on another count are never
read) the worker writes <seq>.qc.bin (on the frame's count), <seq>.qc.averaged.bin (on e, beside a frame read on f),
start.bin, start.npz and start.bins.npz; this manager writes <seq>.qc.json (the
frame's summary with its own area set, each area naming its outline), <seq>.qc.grid (the per-bin share the overview
layer draws), start.json (with the start line's own areas), series.jsonl, shapes.jsonl (each outline once per run),
state.json (what an engine restart resumes: the headline's smoothing and the areas' smoothed sums, hysteresis and
letters) and trials.jsonl. The areas themselves are engine/qc_areas.py's; the suggestions, the trials' bands and
verdicts, the Undo card and the rules are engine/qc_rules.py's, fed here with every live result (`_suggest`); the
calibrated constants are engine/qc_constants.py's, by the run's mode (region, tiled or whole).

The start (only for a run started from cells alone, qc_rules.start_of "cells": in a start with nuclei, proseg's first
iteration holds only the nuclei, so such a run has no start line, no arrow and no retention): the start line shown is
proseg's count at iteration 0 (qc_start on the first frame's counts, fixed); every frame is compared with its own
start block instead (proseg's iteration-0 assignment, its RNA kept as proseg keeps it at this frame: the worker's
`start_block` and its like-for-like `matched` change), so a change shows what the boundaries and drift did.

Every summary also
says how much of the run it rests on (`coverage`: judged, empty, under 20 and 20-99 transcripts over all its cells),
whether cells lost transcripts (`retention`: proseg's count now over the frame's start block's), the GPU worker's
depth-matched change against the start (`matched`) and the comparison made on it (`change` against the start: the raw
share stays the headline, every arrow and verdict is on the matched change with the size-aware thresholds), which
count it is on (`basis`, `count`), the constants it was judged with (`constants`: a mode not yet re-measured on
proseg's count gets a caveat on the page) and the model's `provenance` (the table, priors and passes it learned with
against the reference recipe's).
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from . import children
from . import qc_areas as qa
from . import qc_constants as qk
from . import qc_rules as qr
from ..analysis import qc_probe
from ..analysis import qc_trainer as trainer
from ..analysis.counts_file import BASIS, count_of
from .. import portable
from ..names import FIT_LOCK_ENV, FIT_LOCKS, home_path, retired_isolation

KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")
LEARN_MODES = ("open", "start", "toggle")
DEFAULTS = {"python": "", "gpu": 0, "models_dir": "", "enabled_by_default": False, "learn": "start",
            "tables": ["xenium_table", "table"],
            "priors": "pooled", "mixed_at": 0.30, "judge_min_counts": 100, "fit_min_counts": 20, "epochs": 50,
            "min_steps": 3000, "bin_um_detail": 50, "bin_um_tiled": 100, "gpu_min_free_mb": "auto",
            "trainer_module": "sopa_companion.analysis.qc_trainer", "time_limit_s": 1800, "keep_models": 64,
            "probe_module": "sopa_companion.analysis.qc_probe"}
# Retired settings: a fit once also waited for GPU 0 to be under gpu_busy_util % busy for gpu_quiet_s seconds, and
# notebook_dir named a folder of outside results whose number the page never showed; config files that still set
# them are read as before and the values are not used.
RETIRED = ("gpu_busy_util", "gpu_quiet_s", "notebook_dir")
TRAINER_THREADS = "4"
# One fit on this computer at a time, whichever engine (the app, a test or harness engine, one of the previous version)
# starts it: an flock on each of FIT_LOCKS (under the home folder: the previous version's, which it takes alone, then
# this version's), released when the fit ends or its engine dies. FIT_LOCK_ENV names another file, then the only one
# (the app's tests).
HELLO_TIMEOUT_S = 180          # the trainer's first line (imports nothing heavy before it)
KILL_GRACE_S = 30              # past time_limit_s the engine kills a trainer that did not stop itself
GPU_POLL_S = 1.0
OOM_RETRY_S = 60               # out of GPU 0's memory: the one retry (a fit's or a detail job's) waits this long first
STEP_S = 0.03                  # seconds per optimisation step on an idle GPU 0 (~2.4 s per 80-batch pass)
# On Apple's GPU (MPS): measured 0.056 s a step on an M2 Ultra (2026-11-06, 20k cells, 3000 steps; a recent NVIDIA
# workstation card took 0.016, 3.5 times less), which its estimates use; smaller chips are slower still, so its fits get
# MPS_TIME_FACTOR times time_limit_s.
MPS_STEP_S = 0.06
MPS_TIME_FACTOR = 4
FIXED_S = 25                   # imports, reading, export
KEY_TTL_S = 5.0                # a sample's table fingerprint is read again after this long
# The one CUDA_VISIBLE_DEVICES the trainer learns with (analysis/qc_trainer.py run refuses any other, after reading the
# tables): a [qc] gpu that maps to another one fails before anything is read (QcModels._gpu_state). None once the
# trainer learns on whatever one GPU the job names (its "gpu").
TRAINER_GPU: str | None = "0"
SHARED_KEEP_S = 24 * 3600      # a shared models folder: a model any session used this recently is never pruned (_prune)
# Whether the check can learn here (analysis/qc_probe.py, its own process in the [qc] Python): its answer within this
# long, else none; after no answer it is asked again after these delays (the last one repeating); a press of the switch
# or Try again while the check is not offered asks again, at most this often.
PROBE_TIMEOUT_S = 120.0
PROBE_RETRY_S = (60.0, 300.0, 900.0)
PROBE_AGAIN_S = 10.0
RESULTS_KEPT = 4000            # full summaries per run held in memory (the newest; older ones are read from disk)
# What the line over a run keeps of every result (GET /runs/{id}/qc/series), held for all of them: a run's frames are
# thinned at the source (every iteration to 2,000, every 5th to 10,000, ...), so a long run has a few thousand.
POINT_KEYS = ("seq", "iter", "per100", "shown", "raw_per100", "judged", "mixed", "median_tx", "single",
              "matched_change", "count")  # matched_change: the summary's change.matched (smoothed for live results);
                                          # count: the count the frame was read on ("averaged" | "sampled")
COUNTS = ("averaged", "sampled")         # a frame's count (analysis/counts_file.py count_of): e, or f, the count proseg
                                         # saves, for a frame proseg counted for a pause, the hold, a finish or the end
# The headline's smoothing is the mode's (engine/qc_constants.py headline_weight, headline_ref_s): weight w on the
# newest live result per second of the run's clock (frame "t"; w 0.2 in a region, 0.1 in a tiled one, 0.5 in a whole
# sample), the smoothing the shown integer's margin (shown_margin) was measured with: on proseg's count in regions and
# tiled regions (once cells have formed, at most one reversal of the integer every 14 minutes in a region's repeat
# runs, none in a tiled one's), on the outline count in whole samples, not measured again yet (about one change of the
# integer every two minutes); results without a time weigh w each. The start block's level (the arrow's start) is
# smoothed the same way.
PRELEARN_POLL_S = 2.0          # the pre-learning command (main) looks at the sample being learned this often
# state.json (the headline's and the areas' smoothing, engine/qc_areas.py): written after a live result at most this
# often (s, the engine's clock), at once when the drawn areas change, and when the engine closes. A whole sample's areas
# make it ~30 KB; a run does not go on across an engine restart (engine/runs.py stops it), so at most the last second of
# smoothing is lost when the engine is killed.
STATE_SAVE_S = 1.0
# An area's cell groups (only with Clusters & UMAP on): the clusters of the frame's own UMAP result
# that hold at least this share of the area's cells, largest first, at most CLUSTERS_LISTED, each named by its first two
# top genes (a summary with 8 areas stays <= 8 KB).
CLUSTER_SHARE = 0.15
CLUSTERS_LISTED = 3
FORMING_PHASES = ("init", "burnin")    # cells still forming: no areas of their own yet (the start line's are shown)
FORMED_PHASES = ("upscale", "sampling", "recorded", "hillclimb")
# The order the page lists the drawn areas in (live results): the previous order, an area moving above the one before it
# only once it has ORDER_SWAP x that one's extra mixed cells (near-equal areas never swap rows back and forth).
ORDER_SWAP = 1.25
PUSHED_SEQS = 4                        # results pushed per WebSocket tick (the newest); the others are only named
DRAWN_KEPT = 4000                      # drawn sets held per run (drawn.jsonl keeps them all)
# Suggestions and trials (engine/qc_rules.py): the live results held for the band of the next knob change (its history
# is the 50 iterations before it), the frames before a trial's request where its change may already show (the page may
# set the knob before it registers the trial), and the frames a run's pace is read from (1,000 iterations, in seconds,
# for the "later" timing line).
RECENT_ITERS = 120
RECENT_KEPT = 2000
TRIAL_LOOKBACK = 20
PACE_FRAMES = 60
ACTIVE_STATES = ("preparing", "running", "paused", "finishing")   # engine/runs.py ACTIVE: a run that still holds proseg
ENDED_STATES = ("done", "failed", "stopped")                      # engine/runs.py TERMINAL: a trial there has ended
# A live summary as the page gets it stays under this (every summary carries its coverage, retention, matched change,
# comparisons and provenance, and every area its own comparison: ~1.7 KB with 8 ragged areas, so 9.6 KB keeps the
# outlines whole): the worker's ms and gpu
# numbers included; the last more ideas give way first, never the card or an area.
SUMMARY_BYTES = 9600
# A run's switch, <run>/qc/settings.json {"v", "enabled", "n"}: "v" since the check became opt-in. A file
# without it is from the on-by-default days (stored_switch).
SETTINGS_VERSION = 2



def stored_switch(settings, default: bool) -> tuple[bool, int, bool]:
    """A run's switch as its qc/settings.json keeps it: (enabled, n: how often it changed, current: the file has this
    version). A file from before the check was opt-in (no "v") whose switch nobody ever changed (n 0) only followed the
    old default (on): it follows `default` instead ([qc] enabled_by_default, off as shipped), so opening a run saved
    back then learns nothing and scores nothing; a switch someone changed keeps their choice. Raises (KeyError,
    TypeError, ...) for anything that is not a switch."""
    enabled = bool(settings["enabled"])
    n = settings.get("n")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else 0
    current = settings.get("v") == SETTINGS_VERSION
    return (enabled if current or n else bool(default)), n, current


class QcError(Exception):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic(path: Path, data: bytes):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    with open(temporary, "wb") as stream:
        stream.write(data)
    os.replace(temporary, path)


# ---- settings -----------------------------------------------------------------------------------------------------
class QcSettings:
    """[qc] in config.toml, checked (a bad value fails at start-up)."""

    def __init__(self, config: dict):
        values = {**DEFAULTS, **(config.get("qc") or {})}
        self.python = str(values["python"] or "")
        if type(values["gpu"]) is not int or values["gpu"] < 0:
            raise ValueError("[qc] gpu must be a GPU's number (0 or more): the one the mixing check learns on")
        self.gpu = values["gpu"]                     # 0 here: GPU 0 learns, the GPU worker has GPU 1
        models_dir = values["models_dir"]
        if not isinstance(models_dir, str) or models_dir and not Path(models_dir).expanduser().is_absolute():
            raise ValueError("[qc] models_dir must be empty or a full folder path")
        self.models_dir = str(Path(models_dir).expanduser()) if models_dir else ""
        self.enabled_by_default = bool(values["enabled_by_default"])
        if values["learn"] not in LEARN_MODES:
            raise ValueError(f"[qc] learn must be one of {', '.join(LEARN_MODES)}")
        self.learn = values["learn"]
        tables = values["tables"]
        if not isinstance(tables, list) or not tables or not all(isinstance(t, str) and KEY.fullmatch(t) for t in tables):
            raise ValueError("[qc] tables must be a list of table names")
        self.tables = list(dict.fromkeys(tables))
        if values["priors"] not in trainer.PRIORS:
            raise ValueError(f"[qc] priors must be one of {', '.join(trainer.PRIORS)}")
        self.priors = values["priors"]
        self.mixed_at = self._number(values, "mixed_at", 0, 1, open_low=True, open_high=True)
        self.judge_min = self._integer(values, "judge_min_counts", 1)
        self.fit_min = self._integer(values, "fit_min_counts", 1)
        if self.fit_min > self.judge_min:
            raise ValueError("[qc] fit_min_counts must not exceed judge_min_counts")
        self.epochs = self._integer(values, "epochs", 1)
        self.min_steps = self._integer(values, "min_steps", 0)
        self.bin_um_detail = self._number(values, "bin_um_detail", 0, 10_000, open_low=True)
        self.bin_um_tiled = self._number(values, "bin_um_tiled", 0, 10_000, open_low=True)
        # "auto": the room a fit needs, from its table's panel (room_needed_mb); a number: that many MB, whatever the panel
        self.gpu_min_free_mb = None if values["gpu_min_free_mb"] in ("auto", "", None) else \
            self._number(values, "gpu_min_free_mb", 0, 10 ** 7)
        self.module = str(values["trainer_module"])
        self.probe_module = str(values["probe_module"])
        self.time_limit_s = self._number(values, "time_limit_s", 0, 10 ** 6, open_low=True)
        self.keep_models = self._integer(values, "keep_models", 1)
        self.recipe = trainer.recipe(self.epochs, self.min_steps, self.fit_min, self.priors)

    @staticmethod
    def _number(values, name, low, high, open_low=False, open_high=False) -> float:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value < low or value > high or (open_low and value == low) or (open_high and value == high):
            raise ValueError(f"[qc] {name} must be a number in {'(' if open_low else '['}{low}, {high}{')' if open_high else ']'}")
        return float(value)

    @staticmethod
    def _integer(values, name, low) -> int:
        value = values[name]
        if type(value) is not int or value < low:
            raise ValueError(f"[qc] {name} must be a whole number >= {low}")
        return value

    @property
    def available(self) -> bool:
        return bool(self.python) and Path(self.python).is_file()


def qc_settings(config: dict) -> QcSettings:
    return QcSettings(config)


# ---- cache keys ---------------------------------------------------------------------------------------------------
def _meta(folder: Path) -> tuple[dict | None, float]:
    """A zarr node's metadata (v3 zarr.json, or v2 .zarray/.zgroup + .zattrs) as {node, shape, attributes, dtype}, and
    the newest mtime among its metadata files."""
    stamps, found = [], None
    v3 = folder / "zarr.json"
    try:
        data = json.loads(v3.read_text())
        stamps.append(v3.stat().st_mtime)
        found = {"node": data.get("node_type"), "shape": data.get("shape"), "attributes": data.get("attributes") or {},
                 "dtype": data.get("data_type")}
    except (OSError, ValueError):
        for name, node in ((".zarray", "array"), (".zgroup", "group")):
            try:
                data = json.loads((folder / name).read_text())
                stamps.append((folder / name).stat().st_mtime)
                found = {"node": node, "shape": data.get("shape"), "attributes": {}, "dtype": data.get("dtype")}
                break
            except (OSError, ValueError):
                continue
        if found is not None:
            try:
                found["attributes"] = json.loads((folder / ".zattrs").read_text())
                stamps.append((folder / ".zattrs").stat().st_mtime)
            except (OSError, ValueError):
                pass
    return found, max(stamps) if stamps else 0.0


def table_info(zarr_path, key: str) -> dict | None:
    """What a cell table looks like without reading it: {key, n_obs, n_vars, nnz, mtime, spatial, region, outlines}
    from the zarr metadata of tables/<key> (X, obsm/spatial and the group's attributes: the elements it annotates,
    `region`, and whether one of them is a shapes element of the sample, `outlines`: the trainer can place its cells
    by them when obsm['spatial'] is missing or in other units), or None when there is no such table with counts."""
    folder = Path(zarr_path) / "tables" / key
    table, stamp = _meta(folder)
    if table is None or table["node"] != "group":
        return None
    x, x_stamp = _meta(folder / "X")
    if x is None:
        return None
    nnz, stamps = None, [stamp, x_stamp]
    if x["node"] == "array":
        shape = x["shape"]
    else:
        shape = x["attributes"].get("shape")
        data, data_stamp = _meta(folder / "X" / "data")
        stamps.append(data_stamp)
        if data and data.get("shape"):
            nnz = int(data["shape"][0])
    if not isinstance(shape, list) or len(shape) != 2:
        return None
    spatial, spatial_stamp = _meta(folder / "obsm" / "spatial")
    stamps.append(spatial_stamp)
    has_spatial = bool(spatial and spatial["node"] == "array" and isinstance(spatial.get("shape"), list)
                       and len(spatial["shape"]) == 2 and spatial["shape"][0] == shape[0] and spatial["shape"][1] >= 2)
    region = (table.get("attributes") or {}).get("region")
    regions = [region] if isinstance(region, str) else [r for r in region if isinstance(r, str)] \
        if isinstance(region, list) else []
    outlines = any(KEY.fullmatch(r) and (Path(zarr_path) / "shapes" / r).is_dir() for r in regions)
    return {"key": key, "n_obs": int(shape[0]), "n_vars": int(shape[1]), "nnz": nnz,
            "mtime": round(max(stamps), 3), "spatial": has_spatial, "region": regions, "outlines": bool(outlines)}


def table_keys(zarr_path, configured) -> list[str]:
    """The tables the check may learn from, in the order it tries them: the configured names ([qc] tables), then
    every other table of the sample by name (a counts table may be called anything, e.g. "cells")."""
    try:
        others = sorted(p.name for p in (Path(zarr_path) / "tables").iterdir()
                        if p.is_dir() and KEY.fullmatch(p.name) and p.name not in configured)
    except OSError:
        others = []
    return [*configured, *others]


def first_counts(zarr_path, key: str, limit: int = 1 << 18) -> np.ndarray | None:
    """The first values of tables/<key>'s counts (X/data of a sparse X, else the first rows of a dense one), at most
    `limit` of them, read read-only with zarr (a few ms: one chunk); None when they cannot be read (the trainer reads
    them all anyway)."""
    try:
        import zarr
        node = zarr.open(str(Path(zarr_path) / "tables" / key / "X"), mode="r")
        array = node["data"] if isinstance(node, zarr.Group) else node
        if array.ndim == 1:
            return np.asarray(array[:min(int(array.shape[0]), limit)])
        rows = max(1, min(int(array.shape[0]), limit // max(1, int(array.shape[1]))))
        return np.asarray(array[:rows])
    except Exception:
        return None


def whole_counts(values) -> bool:
    """Whether counts are whole, non-negative, finite numbers (the trainer's not_counts rule)."""
    values = np.asarray(values)
    if values.dtype.kind in "iub":
        return bool(values.size == 0 or values.min() >= 0)
    return bool(np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.rint(values)).all())


def env_versions(python: str) -> dict:
    """Package versions of the trainer's environment from its dist-info folders (no interpreter is started), read where
    the interpreter finds its packages: a venv's own folder (its pyvenv.cfg beside or above bin/, by the path as given:
    a venv's bin/python is a symlink to the base interpreter, whose folder holds none of the venv's packages), then the
    base's for what the venv lacks when it shares them (include-system-site-packages); else the folder of the
    interpreter the path leads to."""
    versions = {name: None for name in trainer.PACKAGES}
    if not python:
        return versions
    place = Path(os.path.abspath(python)).parent
    venv = next((p for p in (place, place.parent) if (p / "pyvenv.cfg").is_file()), None)
    base = Path(python).resolve().parents[1]
    try:
        shared = venv is not None and re.search(r"(?im)^\s*include-system-site-packages\s*=\s*true\s*$",
                                                (venv / "pyvenv.cfg").read_text(errors="replace")) is not None
    except OSError:
        shared = False
    wanted = {re.sub(r"[-_.]+", "_", name).lower(): name for name in trainer.PACKAGES}
    for prefix in ([base] if venv is None else [venv, base] if shared else [venv]):
        found = {}
        for site in sorted(prefix.glob("lib/python3*/site-packages")):
            try:
                entries = os.listdir(site)
            except OSError:
                continue
            for entry in entries:
                match = re.fullmatch(r"(.+?)-([^-]+)\.dist-info", entry)
                if match and re.sub(r"[-_.]+", "_", match.group(1)).lower() in wanted:
                    found[wanted[re.sub(r"[-_.]+", "_", match.group(1)).lower()]] = match.group(2)
        versions = {name: versions[name] or found.get(name) for name in versions}     # the venv's own come first
    return versions


def table_home(zarr_path, key: str) -> Path:
    """The zarr that really holds tables/<key>: the sample's own (resolved) path, or, when that table is a link into
    another zarr (an API session's view of the caller's zarr), that zarr. A view of a zarr and
    the zarr opened in place therefore share their models; a desktop sample (no links) is keyed as before."""
    real = (Path(zarr_path) / "tables" / key).resolve()
    return real.parent.parent if real.parent.name == "tables" else Path(zarr_path).resolve()


def model_key(zarr_path, info: dict, recipe: dict, versions: dict, others: dict | None = None,
              device: str = "cuda") -> str:
    """The cache key of a sample's model: the zarr that holds the learned table (table_home), the learned table
    (`info`), the fingerprints of the sample's other [qc] tables ({key: {n_obs, n_vars, nnz, mtime}}: the priors are
    pooled with them), the recipe and the versions; and for a model learned on Apple's GPU (`device` "mps") the device
    too: as good as one learned on an NVIDIA card, not the same cells' numbers (another random stream), so the two are
    never taken for each other. A CUDA model's key is what it always was."""
    identity = {"zarr": str(table_home(zarr_path, info["key"])), "table": info["key"],
                "fingerprint": {k: info.get(k) for k in ("n_obs", "n_vars", "nnz", "mtime")},
                "others": {key: {k: value.get(k) for k in ("n_obs", "n_vars", "nnz", "mtime")}
                           for key, value in sorted((others or {}).items())},
                "recipe": recipe, "versions": versions}
    if device != "cuda":
        identity["device"] = device
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]


def models_root(config: dict) -> Path:
    """Where the learned models live: [qc] models_dir when set (API mode's model_cache, which sessions share), else
    <data_dir>/qc-models. The page's switch stays in prefs_path, never among shared models."""
    folder = qc_settings(config).models_dir
    return Path(folder) if folder else Path(config["paths"]["data_dir"]) / "qc-models"


def prefs_path(config: dict) -> Path:
    """The mixing check's remembered switch: <data_dir>/qc-models/prefs.json, the engine's (an API session's) own."""
    return Path(config["paths"]["data_dir"]) / "qc-models" / "prefs.json"


def fit_lock_paths() -> list[Path]:
    """The computer-wide fit locks, in the order they are taken: the file FIT_LOCK_ENV names, alone; else FIT_LOCKS
    under the home folder. RuntimeError for a private file named the previous version's way (names.retired_isolation)."""
    if problem := retired_isolation():
        raise RuntimeError(problem)
    private = os.environ.get(FIT_LOCK_ENV)
    return [Path(private).expanduser()] if private else [home_path(path) for path in FIT_LOCKS]


def estimate_seconds(cells: int, recipe: dict, device: str = "cuda") -> int:
    """How long learning a table of this many cells takes on an idle GPU 0 (a few minutes for a whole sample), or on
    Apple's GPU (`device` "mps")."""
    usable = max(1, int(cells))
    epochs = trainer.epochs_for(usable, recipe)
    return int(FIXED_S + step_seconds(device) * epochs * math.ceil(usable / recipe["batch_size"]))


def step_seconds(device: str) -> float:
    """Seconds per optimisation step on the device a fit learns on: an idle GPU 0, or Apple's GPU ("mps")."""
    return MPS_STEP_S if device == "mps" else STEP_S


def device_of(answer) -> str:
    """The device the probe's answer (analysis/qc_probe.py's line, or None) lets the trainer learn on: "mps" when no
    CUDA GPU is seen and Apple's GPU passed the probe's test, else "cuda" (also with no answer, as without the
    probe)."""
    if isinstance(answer, dict) and not answer.get("devices") and answer.get("mps") is True:
        return "mps"
    return "cuda"


def apple_memory() -> dict | None:
    """Apple's GPU's room, in the shape gpu_probe gives ({bus, total_mb, used_mb, free_mb, util}): its memory is the
    computer's own (unified), of which Metal lets one process use about three quarters (APPLE_GPU_SHARE of the
    computer's memory in all), and what is free is what the system calls available; no bus. None when it cannot be
    read."""
    try:
        memory = psutil.virtual_memory()
    except Exception:
        return None
    total = memory.total / 2 ** 20 * APPLE_GPU_SHARE
    free = min(total, memory.available / 2 ** 20)
    return {"bus": None, "total_mb": float(total), "used_mb": float(total - free), "free_mb": float(free), "util": None}


def gpu_probe(index: int | str = 0) -> dict | None:
    """GPU `index` as nvidia-smi sees it (a number in PCI bus order, the order CUDA_DEVICE_ORDER=PCI_BUS_ID gives, or a
    UUID: what engine/children.py gpu() gives): {bus, total_mb, used_mb, free_mb, util}; None when it cannot be asked
    (or "": no GPU)."""
    if index == "":
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=pci.bus_id,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             **portable.no_window())
        bus, total, used, util = (p.strip() for p in out.stdout.strip().splitlines()[0].split(","))
        return {"bus": bus, "total_mb": float(total), "used_mb": float(used), "free_mb": float(total) - float(used),
                "util": float(util)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# The free memory a fit needs on [qc] gpu, from its table: the trainer's peak as nvidia-smi sees it (CUDA's own context
# included) grows with the panel and hardly with the cells. Measured 2026-11-05 (torch 2.13 with CUDA 13):
# 40k cells with 50, 300, 1,000, 2,500, 5,101 and 10,202 genes peaked at 860, 1,030, 1,432, 2,682, 4,082 and 6,792 MB,
# 3.5k cells with 5,101 genes at 4,024 MB. ROOM_BASE_MB, ROOM_GENE_MB a gene and ROOM_CELL_MB a cell, times ROOM_MARGIN
# (another card's context, the allocator's steps), rounded up to 100 MB: 5,000 MB for 5,101 genes and 41k cells.
ROOM_BASE_MB, ROOM_GENE_MB, ROOM_CELL_MB, ROOM_MARGIN = 900, 0.62, 0.0016, 1.2
# On Apple's GPU the table's counts are learned from dense (MPS holds no sparse tensor): cells x genes float32 more,
# in the same (unified) memory. Metal's recommended working set is about this share of the computer's memory.
APPLE_GPU_SHARE = 0.75


def room_needed_mb(genes: int, cells: int = 0, device: str = "cuda") -> int:
    """The free memory (MB) a fit of a table of `genes` genes and `cells` cells needs on [qc] gpu, or on Apple's GPU
    (`device` "mps": its dense counts too)."""
    genes, cells = max(0, int(genes)), max(0, int(cells))
    raw = ROOM_BASE_MB + ROOM_GENE_MB * genes + ROOM_CELL_MB * cells
    if device == "mps":
        raw += cells * genes * 4 / 2 ** 20
    return int(math.ceil(raw * ROOM_MARGIN / 100.0) * 100)


def bus_number(bus: str | None) -> int | None:
    """00000000:11:00.0 -> 17 (what torch reports as pci_bus_id)."""
    try:
        return int(str(bus).split(":")[-2], 16)
    except (ValueError, IndexError):
        return None


def and_list(items) -> str:
    """Names as a sentence says them: "a", "a and b", "a, b and c" ("" for none)."""
    items = [str(item) for item in items]
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _major(version) -> int | None:
    """A CUDA version's major number ("13.0" -> 13), None when it is not one."""
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return None


def capability_of(answer) -> tuple[str, str | None, str]:
    """What the probe's answer (analysis/qc_probe.py's line, or None) means for the mixing check: (state, reason, why).
    The first that holds wins, in the order that helps: no answer ("unknown": the check stays offered, as without
    the probe); no NVIDIA card that anything can see: on a Mac its Apple GPU, ready where it passed the probe's test
    (scvi-tools and scviva-tools installed), "apple_gpu" with the test's words where it failed, torch missing there
    "software" on Apple silicon, else (an Intel Mac, every other computer without an NVIDIA card) "no_gpu"; torch
    missing; a torch built without CUDA;
    an NVIDIA driver older than torch's CUDA; a card torch cannot see (hidden, or a driver torch cannot start);
    scvi-tools or scviva-tools missing; else "ready". `why` is for the engine's log and the Log, never the page."""
    if not isinstance(answer, dict) or answer.get("error"):
        error = answer.get("error") if isinstance(answer, dict) else None
        return "unknown", None, f"its Python could not be asked ({error or 'no answer'}): offered, asked again later"
    cards, seen = answer.get("nvidia") or 0, answer.get("devices") or 0
    missing = [str(name) for name in answer.get("missing") or []]
    torch, build, driver = answer.get("torch"), answer.get("cuda_build"), answer.get("driver_cuda")
    lacking = (f"{and_list(missing)} {'is' if len(missing) == 1 else 'are'} not installed in "
               f"{answer.get('python') or 'its Python'}")
    if cards == 0 and seen == 0:
        if answer.get("mps") is True:
            if missing:
                return "unavailable", "software", lacking
            return "ready", None, f"torch {torch} learns on Apple's GPU (MPS): it passed the probe's test"
        if answer.get("mps_problem"):
            return "unavailable", "apple_gpu", str(answer["mps_problem"])
        if answer.get("platform") == "darwin" and answer.get("machine") == "arm64" and "torch" in missing:
            return "unavailable", "software", lacking       # Apple silicon: the packages could learn there
        return "unavailable", "no_gpu", "this computer has no NVIDIA graphics card"
    if "torch" in missing:
        return "unavailable", "software", lacking
    if build is None:
        return "unavailable", "torch_cpu", f"torch {torch} was built without CUDA"
    have, need = _major(driver) if driver else None, _major(build)
    if have is not None and need is not None and have < need:
        return "unavailable", "driver", \
            f"the NVIDIA driver supports CUDA {driver}, older than torch {torch}'s CUDA {build}"
    if seen < 1:
        hidden = answer.get("torch_warning") or f"CUDA_VISIBLE_DEVICES={answer.get('visible')!r}"
        return "unavailable", "no_gpu", f"torch sees no CUDA GPU ({hidden})"
    if missing:
        return "unavailable", "software", lacking
    return "ready", None, f"torch {torch} sees {seen} CUDA GPU(s)"


# ---- QcModels -----------------------------------------------------------------------------------------------------
REASONS = {   # failed / unavailable / waiting reasons the page puts into words
    "software", "no_table", "not_counts", "positions", "too_few_cells", "oom", "not_settled", "timeout", "unreadable",
    "gpu", "unexpected", "queue", "gpu_busy", "gpu_memory", "gpu_small", "lock", "detail_mismatch", "no_gpu", "apple_gpu",
    "torch_cpu",
    "driver"}
# Reasons that belong to a table (remembered per table fingerprint in tables.json, and the next table tried): its counts
# are not whole, or its cells cannot be placed on the transcripts (the trainer's choose_centres).
TABLE_REASONS = ("not_counts", "positions")
# The trainer's rules those reasons come from, in each remembered note: a note made under other rules is not taken (2:
# cells its outlines place need only touch the transcripts; a table of a sample whose transcripts are two elements was
# refused for "positions" under 1, and stayed refused).
TABLE_RULES = 2
DETAIL_STATES = ("none", "queued", "learning", "ready", "failed")   # a sample's Go deeper details (describe "detail")
DEEP_FILE = "deep.npz"
DEEP_TOLERANCE = getattr(trainer, "DEEP_SELFTEST_TOLERANCE", 1e-5)  # deep.npz's self-test, float64, as the trainer's
DETAIL_TOLERANCE = getattr(trainer, "DETAIL_TOLERANCE", 1e-6)      # a detail job's calculator vs the installed one


class DetailMismatch(ValueError):
    """A detail job's new learning does not match the installed model (its calculator, or its export): nothing is
    changed (reason "detail_mismatch")."""


def deep_parity(deep: dict, encoder: dict) -> float:
    """deep.npz's self-test against its calculator: the largest |numpy - torch (noise off)| over every self-test
    output (analysis/qc_trainer.py deep_self_test; float64)."""
    value = trainer.deep_self_test(deep, encoder)
    if isinstance(value, dict):
        value = value.get("max")
    return float(value)


def calculators_agree(installed: dict, new: dict) -> bool:
    """The parity rule of a detail job: the same genes and every exported array identical, or the
    installed reference cells' raw weights within DETAIL_TOLERANCE."""
    keys = ("W0", "b0", "W1", "b1", "Wv", "bv", "ln_eps", "var_eps", "diffusion_eps")
    if list(installed["genes"]) != list(new["genes"]) or any(installed[k].shape != new[k].shape for k in keys):
        return False
    if all(np.array_equal(installed[k], new[k]) for k in keys):
        return True
    ref = trainer.selftest_counts(installed).astype(np.float64)
    diff = np.abs(trainer.mixing_weights(installed, ref, np, np.float64) - trainer.mixing_weights(new, ref, np, np.float64))
    return float(diff.max()) <= DETAIL_TOLERANCE


def provenance_of(data: dict) -> dict | None:
    """model.json's "provenance" (the table learned from, the other table pooled into the priors, the
    priors' source, the passes run against the reference recipe's, the cells, notebook_equivalent and its differences):
    as the trainer wrote it, else worked out from the rest of an older model.json by the trainer's own rule
    (qc_trainer.provenance); None when neither can be had."""
    found = data.get("provenance")
    if isinstance(found, dict):
        return found
    rule = getattr(trainer, "provenance", None)
    if rule is None:
        return None
    try:
        value = rule(data)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


class QcModels:
    """One model per sample, learned on demand (see the module docstring). Thread-safe; one background thread."""

    def __init__(self, config: dict, zarr=None, *, start: bool = True, probe=None, prober=None, clock=time.monotonic,
                 log_path: Path | None = None):
        self.s = qc_settings(config)
        children.configure(config)
        self.root = models_root(config)
        self.log_path = Path(log_path) if log_path else Path(config["paths"]["state_dir"]) / "logs" / "qc-trainer.log"
        self.zarr = zarr
        self.probe = probe or self._gpu_state         # nvidia-smi's view of [qc] gpu, or why it cannot be learned on
        # Whether the check can learn here at all (capability_of): asked of `prober` (the probe in the [qc] Python),
        # in its own thread (_probe_loop), "checking" until its first answer.
        self.prober = prober or self._run_probe
        self.probe_thread: threading.Thread | None = None
        self.probe_process: subprocess.Popen | None = None
        self.probe_wanted = False                      # ask_again: the probe thread asks once more
        self.probing = False
        self.probed_at = -math.inf                     # when the last answer came (clock)
        self.unanswered = 0                            # answers in a row that were no answer (PROBE_RETRY_S)
        located = (config.get("locate") or {}).get("qc") or {}
        # Without a runnable [qc] python the check is not offered from the start; the probe still runs in this Python
        # when "auto" found the check's packages missing here (config["locate"], engine/config.py), so a computer
        # without an NVIDIA card is told that first rather than to install packages that could not learn there.
        self.no_python = self._no_python_why(located)
        self.capability_state = {"state": "checking", "reason": None, "why": None, "probe": None} if self.s.available \
            else {"state": "unavailable", "reason": "software", "why": self.no_python, "probe": None}
        self.probe_python = self.s.python if self.s.available else \
            sys.executable if located.get("how") == "auto" and located.get("missing") else None
        self.clock = clock
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.records: dict[str, dict] = {}
        self.queue: list[str] = []
        self.version = 0
        self.process: subprocess.Popen | None = None
        self.closed = False
        self.versions_cache: tuple[float, dict] = (-math.inf, {})
        self.summaries: dict[str, dict] = {}           # model dir -> model.json (read once)
        self.touched: dict[str, float] = {}            # model dir -> when its `used` stamp was last checked
        self.on_ready: list = []                       # callables, called after a sample finished learning
        self.problems: dict[tuple, tuple | None] = {}  # (path, table, fingerprint) -> what _problem found
        self.lock_warned = False                       # the computer-wide fit lock could not be used (said once)
        self.stuck: list = []                          # (trainer, fit lock handles): trainers that outlived SIGKILL
        self._sweep_staging()
        self.thread = threading.Thread(target=self._loop, name="qc-models", daemon=True)
        if start:
            self.thread.start()
            if self.probe_python:
                self.probe_thread = threading.Thread(target=self._probe_loop, name="qc-probe", daemon=True)
                self.probe_thread.start()

    # ---- public ---------------------------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether the check is offered: a runnable [qc] python, and no definite no from the probe (while it checks, or
        when it could not answer, the check stays offered, as without the probe)."""
        return self.s.available and self.capability_state["state"] != "unavailable"

    def capability(self) -> dict:
        """Whether the check can learn here: {state: "checking" | "ready" | "unavailable" | "unknown", reason (an
        unavailable one's: "no_gpu", "apple_gpu", "software", "torch_cpu", "driver"), why (the Log's words), probe (its
        answer)}."""
        with self.lock:
            return dict(self.capability_state)

    def device(self) -> str:
        """The device a fit learns on: "mps" (Apple's GPU) where the probe's answer says so (device_of), else "cuda"."""
        with self.lock:
            return device_of(self.capability_state.get("probe"))

    def ask_again(self):
        """Ask the probe again (its thread does) when the check is not offered here, or its probe gave no answer: a
        press of the switch or Try again (ensure), at most every PROBE_AGAIN_S, so a computer set up meanwhile (a
        package installed, a card's cable back) is answered without a restart. Nothing without a runnable [qc] python
        (none can appear while the engine runs), while a probe runs or before its first answer."""
        with self.lock:
            if not self.s.available or self.probe_thread is None or self.probing or self.probe_wanted \
                    or self.capability_state["state"] not in ("unknown", "unavailable") \
                    or self.clock() - self.probed_at < PROBE_AGAIN_S:
                return
            self.probe_wanted = True
            self.wake.notify_all()

    def describe(self, sample_id: str) -> dict:
        record = self._record(sample_id)
        self._refresh(record)
        return self._public(record)

    def ensure(self, sample_id: str, retry: bool = False, detail: bool = False) -> dict:
        """Learn the sample unless its model is cached, being learned or queued; a failed one only with `retry`.
        `detail` (Go deeper's *Learn the details*, also its Try again): a ready model without deep.npz gets a detail
        job (queued unless one is queued or running already); a sample without a model is learned as usual (new fits
        write deep.npz). The check not offered here, a press asks its probe again (ask_again)."""
        record = self._record(sample_id)
        self.ask_again()
        if detail:                                    # the model's files as they are now (deep.npz included)
            with self.lock:
                folder = self._folder(record)
            if folder is not None:
                self.summaries.pop(str(folder), None)
        self._refresh(record, force=retry or detail)
        with self.lock:
            if record["state"] == "failed" and retry:
                record.update(state="off", reason=None, message=None, attempts=0)
            if record["state"] == "off":
                self._queue(record)
            elif detail and record["state"] == "ready" and not (record["model"] or {}).get("deep") \
                    and not (record["detail"]["key"] == record["key"] and record["detail"]["state"] in ("queued", "learning")):
                self._queue_detail(record)
        return self._public(record)

    def adopt(self, sample_id: str, path: str | None, name: str | None = None):
        """A run's own sample (its id and zarr path from run.json), also when the open dialog forgot the sample."""
        if not path:
            return
        with self.lock:
            record = self.records.setdefault(sample_id, self._blank(sample_id))
            if not record["path"]:
                record.update(path=str(path), name=name or record["name"], rejected={})

    def register(self, path: str, name: str | None = None) -> str:
        """A sample known by its path only (the pre-learning command): its id, as the open dialog would give it."""
        from .zarr_samples import sample_id
        folder = Path(path).expanduser().resolve()
        sid = sample_id(str(folder))
        with self.lock:
            record = self.records.setdefault(sid, self._blank(sid))
            record.update(path=str(folder), name=name or re.sub(r"\.zarr/?$", "", folder.name))
        return sid

    def model(self, sample_id: str) -> dict | None:
        """The sample's ready model {id, dir, encoder, table, ...} or None (nothing is learned here)."""
        try:
            record = self._record(sample_id)
        except QcError:
            return None
        self._refresh(record)
        if record["state"] != "ready":
            return None
        folder = self._folder(record)
        if not (folder / "encoder.npz").is_file():    # removed since the last look (a shared folder's prune, a hand)
            self._refresh(record, force=True)
            return None
        self._touch(folder)
        deep = (record["model"] or {}).get("deep")
        return {"id": record["key"], "dir": str(folder), "encoder": str(folder / "encoder.npz"),
                "deep": str(folder / DEEP_FILE) if deep else None, "summary": record["model"],
                "table": record["table"]["key"]}

    def snapshot(self) -> dict[str, dict]:
        """Every sample this engine has a state for other than off (for WebSocket events)."""
        with self.lock:
            records = [r for r in self.records.values() if r["state"] != "off" or r["changed"]]
        return {r["sample"]: self._public(r) for r in records}

    def close(self):
        with self.lock:
            self.closed = True
            self.wake.notify_all()
            probe = self.probe_process
        if probe is not None and probe.poll() is None:  # its answer would come to nobody
            try:
                probe.kill()
            except OSError:
                pass
        self._kill()

    # ---- records --------------------------------------------------------------------------------------------------
    @staticmethod
    def _blank(sid: str) -> dict:
        return {"sample": sid, "path": None, "name": None, "state": "off", "reason": None, "message": None,
                "table": None, "key": None, "device": "cuda", "checked": -math.inf, "look": 0, "progress": None,
                "eta_s": None,
                "home": None, "model": None, "relearn": False, "attempts": 0, "stage": None, "learning": None,
                "estimate_s": None, "room": None,
                "shared": False, "changed": False, "rejected": {}, "detail": QcModels._no_detail()}

    @staticmethod
    def _no_detail() -> dict:
        """A sample's detail job (Go deeper's deep.npz for a model learned before it): none queued."""
        return {"state": "none", "key": None, "reason": None, "message": None, "progress": None, "eta_s": None,
                "stage": None, "learning": None, "attempts": 0}

    def _record(self, sample_id: str) -> dict:
        with self.lock:
            record = self.records.get(sample_id)
            if record is not None and record["path"]:
                return record
        path = name = None
        if self.zarr is not None:
            # Any sample the engine knows, on the open dialog's list or not (a run's sample opened again for it, one
            # past the list's 30): its detail answers, so its mixing check answers too.
            entry = self.zarr.entry(sample_id)
            if entry is not None:
                path, name = entry.get("path"), entry.get("name")
        if not path:
            raise QcError("Unknown sample: open it again", 404)
        with self.lock:
            record = self.records.setdefault(sample_id, self._blank(sample_id))
            record.update(path=path, name=name, rejected={})        # its tables.json, read again as needed (_notes)
            return record

    def _changed(self, record: dict | None = None):
        with self.lock:
            if record is not None:
                record["changed"] = True
            self.version += 1

    def _versions(self) -> dict:
        stamp, versions = self.versions_cache
        if self.clock() - stamp > 60:
            versions = env_versions(self.s.python)
            self.versions_cache = (self.clock(), versions)
        return versions

    def _refresh(self, record: dict, force: bool = False, internal: bool = False, off_as: str = "off"):
        """The table, key and cache state of an idle sample (waiting or learning ones keep their job, unless the
        learning thread itself asks: `internal`). A sample with nothing cached becomes `off_as`.
        The zarr metadata is read without holding the lock, so what this look found is written (`_settle`) only if no
        later look began meanwhile, and never over a job another thread queued meanwhile (ensure(), e.g. a run created
        while the analysis thread looked): only a model found ready replaces "waiting" (the queued job then finds it
        and stops); the queued job looks again itself before it learns (`_learn`)."""
        with self.lock:
            if record["state"] in ("waiting", "learning") and not internal:
                return
            if not force and self.clock() - record["checked"] < KEY_TTL_S:
                return
            record["checked"] = self.clock()
            record["look"] += 1
            look = record["look"]
        if not self.available:
            cap = self.capability()
            self._settle(record, look, internal, state="unavailable", reason=cap["reason"] or "software",
                         message=cap["why"] or "no Python environment is set up for it ([qc] python in config.toml)",
                         key=None, model=None)
            return
        # The first table it can learn from, in table_keys' order (the configured names, then any other). A table
        # the trainer cannot learn from is passed over here when its metadata already says so, before any wait for
        # GPU 0: no cell centres to go by, too few cells, counts that are not whole (`_problem`), or a reason the
        # trainer gave for this very table before (tables.json). The first one passed over says why when none is left.
        chosen, passed = None, []
        for key in table_keys(record["path"], self.s.tables):
            info = table_info(record["path"], key)
            if info is None:
                continue
            note = self._notes(record, self._home(record, key)).get(key)
            if note and note.get("fingerprint") == self._fingerprint(info) and note.get("rules") == TABLE_RULES:
                reason = note.get("reason") if note.get("reason") in TABLE_REASONS else "not_counts"
                passed.append((reason, f"table '{key}' was refused for this ({reason}) before"))
                continue
            problem = self._problem(record["path"], info)
            if problem is not None:
                passed.append(problem)
                continue
            chosen = info
            break
        if chosen is None:
            reason, message = passed[0] if passed else ("no_table", "no table of cells with transcript counts")
            # Too few cells is what the trainer would say after its wait ("failed": Try again looks again, at once).
            self._settle(record, look, internal, state="failed" if reason == "too_few_cells" else "unavailable",
                         reason=reason, table=None, key=None, model=None, message=message)
            return
        others = {}                                     # the sample's other tables: pooled priors, the earlier cells
        for other in self.s.tables:
            found = table_info(record["path"], other) if other != chosen["key"] else None
            if found is not None:
                others[other] = self._fingerprint(found)
        device = self.device()
        key = model_key(record["path"], chosen, self.s.recipe, self._versions(), others, device)
        home = self._home(record, chosen["key"])
        folder = self.root / home / key
        estimate = estimate_seconds(chosen["n_obs"], self.s.recipe, device)
        if (folder / "model.json").is_file() and (folder / "encoder.npz").is_file():
            summary = self._summary(folder)
            if summary is not None:
                self._settle(record, look, internal, state="ready", reason=None, message=None, table=chosen, key=key,
                             device=device, home=home, model=summary, relearn=False, progress=None, eta_s=None,
                             estimate_s=estimate, shared=False, stage=None)
                return
        with self.lock:
            same = record["key"] == key
            state = record["state"] if same and record["state"] == "failed" and not internal else off_as
        relearn = self._relearn(home, chosen, key)
        self._settle(record, look, internal, state=state, table=chosen, key=key, device=device, home=home, model=None,
                     relearn=relearn, estimate_s=estimate,
                     **({} if state == "failed" else {"reason": "queue" if state == "waiting" else None, "message": None}))

    def _settle(self, record: dict, look: int, internal: bool, **values) -> bool:
        """Write what look number `look` of `_refresh` found, unless a later look began meanwhile, or the sample was
        queued or is being learned now (by another thread, after this look began) and this look found no model."""
        with self.lock:
            if record["look"] != look:
                return False
            if not internal and record["state"] in ("waiting", "learning") \
                    and not (values.get("state") == "ready" and record["state"] == "waiting"):
                return False
            if all(record.get(k) == v for k, v in values.items()):
                return True
            record.update(values)
            record["changed"] = True
            self.version += 1
        return True

    @staticmethod
    def _fingerprint(info: dict) -> dict:
        return {k: info.get(k) for k in ("n_obs", "n_vars", "nnz", "mtime")}

    def _problem(self, path: str, info: dict) -> tuple[str, str] | None:
        """What a table's metadata already says the trainer would refuse, before any wait for GPU 0:
        ("positions", why) with neither cell centroids (obsm['spatial']) nor outlines to place its cells by;
        ("too_few_cells", why) with fewer rows than the trainer's MIN_CELLS; ("not_counts", why) when its first counts
        are not whole numbers (one chunk read once per table fingerprint: a log-normalised table); None otherwise."""
        key = info["key"]
        if not info["spatial"] and not info.get("outlines"):
            return "positions", f"table '{key}' has no cell centroids (obsm['spatial']) and annotates no outlines"
        if info["n_obs"] < trainer.MIN_CELLS:
            return "too_few_cells", f"table '{key}' holds {info['n_obs']} cells: at least {trainer.MIN_CELLS} are needed"
        stamp = (str(path), key, json.dumps(self._fingerprint(info), sort_keys=True))
        if stamp not in self.problems:
            values = first_counts(path, key)
            if len(self.problems) > 256:
                self.problems.clear()
            self.problems[stamp] = None if values is None or whole_counts(values) else \
                ("not_counts", f"table '{key}' does not hold whole transcript counts")
        return self.problems[stamp]

    @staticmethod
    def _home(record: dict, key: str) -> str:
        """The folder under the models root for the sample's table `key`: the sample id of the zarr that holds it
        (table_home): the sample's own (a desktop sample's id), or, for a table linked from another zarr (an API view),
        that zarr's, the folder an in-place session of that zarr uses, however each session named its sample."""
        from .zarr_samples import sample_id
        return sample_id(str(table_home(record["path"], key)))

    def _folder(self, record: dict, key: str | None = None) -> Path | None:
        """A sample's model folder, <root>/<home>/<key> (`key`: the record's own by default); None without a key."""
        key = key or record["key"]
        return self.root / (record.get("home") or record["sample"]) / key if key else None

    def _relearn(self, home: str, info: dict, key: str) -> bool:
        """A model of the same table exists under another key: the software (or recipe) changed since."""
        for path in (self.root / home).glob("*/model.json"):
            if path.parent.name == key:
                continue
            summary = self._summary(path.parent)
            if summary and summary.get("table") == info["key"] \
                    and self._fingerprint(summary.get("table_fingerprint") or {}) == self._fingerprint(info):
                return True
        return False

    def _summary(self, folder: Path) -> dict | None:
        """A model folder's model.json as describe uses it (read once; dropped from the cache when a detail job or a
        prune changes the folder, so the entry is held locally: another thread may drop it meanwhile)."""
        key = str(folder)
        entry = self.summaries.get(key)
        if entry is None:
            try:
                data = json.loads((folder / "model.json").read_text())
            except (OSError, ValueError):
                return None
            cells = data.get("cells") or {}
            entry = {
                "id": data.get("id") or folder.name, "table": (data.get("table") or {}).get("key"),
                "cells": cells.get("fitted"), "genes": cells.get("genes"), "created": data.get("created"),
                "seconds": (data.get("seconds") or {}).get("total"), "epochs": data.get("epochs"),
                "steps": data.get("steps"), "per100": cells.get("per100"), "versions": data.get("versions"),
                "batch": data.get("batch"), "table_info": data.get("table"),
                # Go deeper's details: model.json names them and the file is there (an encoder-only model still scores).
                "deep": isinstance(data.get("deep"), dict) and (folder / DEEP_FILE).is_file(),
                # ... or why they were left out when the fit's own export of them failed (model.json "deep_error"):
                # the window then says the details could not be added, not that the sample predates them.
                "deep_failed": ((data.get("deep_error") or {}).get("reason") or "unexpected")
                if isinstance(data.get("deep_error"), dict) and not isinstance(data.get("deep"), dict) else None,
                "provenance": provenance_of(data)}
            self.summaries[key] = entry
        summary = dict(entry)
        summary.pop("table_info", None)
        summary["table_fingerprint"] = entry.get("table_info")
        return summary

    def _set(self, record: dict, **values):
        """Update a record; the version (what the WebSocket loop watches) moves only when something changed."""
        with self.lock:
            if all(record.get(k) == v for k, v in values.items()):
                return
            record.update(values)
        self._changed(record)

    def _public(self, record: dict) -> dict:
        with self.lock:
            model = record["model"]
            return {"sample": record["sample"], "state": record["state"], "reason": record["reason"],
                    "message": record["message"], "stage": record["stage"],
                    "progress": dict(record["progress"]) if record["progress"] else None, "eta_s": record["eta_s"],
                    "estimate_s": record["estimate_s"], "relearn": record["relearn"], "shared": record["shared"],
                    "room": dict(record["room"]) if record.get("room") else None,
                    "table": (record["table"] or {}).get("key"),
                    "model": {k: model.get(k) for k in ("id", "table", "cells", "genes", "created", "seconds",
                                                        "epochs", "steps", "deep", "provenance")} if model else None,
                    "detail": self._detail_public(record)}

    @staticmethod
    def _detail_public(record: dict) -> dict:
        """Go deeper's details for the sample's model (describe "detail"): {state: "ready" (the model has deep.npz) |
        "queued" | "learning" (a detail job) | "failed" (the last one, with its reason) | "none" (no model yet, or an
        encoder-only model nobody asked to complete), progress {step, of} | null, eta_s, reason}. Lock held."""
        model, job = record["model"] or {}, record["detail"]
        if record["state"] == "ready" and model.get("deep"):
            return {"state": "ready", "progress": None, "eta_s": None, "reason": None}
        if record["state"] == "ready" and job["key"] == record["key"] and job["state"] in ("queued", "learning", "failed"):
            return {"state": job["state"], "progress": dict(job["progress"]) if job["progress"] else None,
                    "eta_s": job["eta_s"], "reason": job["reason"]}
        if record["state"] == "ready" and model.get("deep_failed"):   # the fit's own export of them failed
            return {"state": "failed", "progress": None, "eta_s": None, "reason": model["deep_failed"]}
        return {"state": "none", "progress": None, "eta_s": None, "reason": None}

    def _queue(self, record: dict):
        with self.lock:
            if record["sample"] not in self.queue:
                self.queue.append(record["sample"])
            record.update(state="waiting", reason="queue", message=None, progress=None, eta_s=None, stage=None)
            self.version += 1
            record["changed"] = True
            self.wake.notify_all()

    def _queue_detail(self, record: dict):
        """A detail job for the sample's ready model (Go deeper's *Learn the details*): the same queue as fits. Lock
        held."""
        attempts = record["detail"]["attempts"] if record["detail"]["key"] == record["key"] else 0
        record["detail"] = {**self._no_detail(), "state": "queued", "key": record["key"], "reason": "queue",
                            "eta_s": record["estimate_s"], "attempts": attempts}
        if record["sample"] not in self.queue:
            self.queue.append(record["sample"])
        self.version += 1
        record["changed"] = True
        self.wake.notify_all()

    def _set_detail(self, record: dict, **values):
        """Update a sample's detail job (the version moves only when something changed)."""
        with self.lock:
            job = record["detail"]
            if all(job.get(k) == v for k, v in values.items()):
                return
            job.update(values)
        self._changed(record)

    # ---- rejected tables (not whole counts), remembered per fingerprint ------------------------------------------
    def _notes(self, record: dict, home: str) -> dict:
        """The tables refused before, {table key: note}, as <models>/<home>/tables.json has them: kept beside the models
        of the zarr that holds the tables (_home), so a view never leaves a folder of its own behind and the zarr opened
        in place knows them too. Read once per record and home (record["rejected"]: {home: notes})."""
        with self.lock:
            notes = record["rejected"].get(home)
        if notes is None:
            try:
                data = json.loads((self.root / home / "tables.json").read_text())
                notes = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                notes = {}
            with self.lock:
                notes = record["rejected"].setdefault(home, notes)
        return notes

    def _reject(self, record: dict, info: dict, reason: str):
        home = self._home(record, info["key"])
        notes = self._notes(record, home)
        with self.lock:
            notes[info["key"]] = {"reason": reason, "fingerprint": self._fingerprint(info), "rules": TABLE_RULES}
            data = json.dumps(notes, indent=1).encode()
        folder = self.root / home
        folder.mkdir(parents=True, exist_ok=True)
        _atomic(folder / "tables.json", data)

    # ---- whether the check can learn here (analysis/qc_probe.py) ---------------------------------------------------
    def _no_python_why(self, located: dict) -> str:
        """Why there is no [qc] python to learn with, from what the engine found for it (locate.py's report)."""
        if located.get("how") == "auto" and located.get("missing"):
            names = dict(qc_probe.PACKAGES)
            return f"this Python environment lacks {and_list(names.get(m, m) for m in located['missing'])}"
        if not self.s.python or self.s.python == "off" or located.get("how") == "off":
            return "python under [qc] is off"
        return f"the python set under [qc] ({self.s.python}) is not there"

    def _run_probe(self) -> dict | None:
        """The probe's answer: its last {"ev": "probe", ...} line, asked in the probe's Python ([qc] python) with the
        trainer's own environment, so torch sees exactly the GPU the trainer would; None when it gave none (it could
        not start, timed out after PROBE_TIMEOUT_S, or printed no such line: its stderr's end goes to the log)."""
        python = self.probe_python
        if not python:
            return None
        try:
            process = subprocess.Popen([python, "-m", self.s.probe_module], cwd=children.app_path(),
                                       env=self._trainer_env(python), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
                                       **portable.no_window())
        except OSError as exc:
            self._log(f"qc: the mixing check's probe could not start in {python}: {exc}")
            return None
        with self.lock:
            self.probe_process = process
        try:
            out, err = process.communicate(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = process.communicate()
            self._log(f"qc: the mixing check's probe gave no answer within {PROBE_TIMEOUT_S:g} s")
            return None
        finally:
            with self.lock:
                self.probe_process = None
        for line in reversed((out or "").splitlines()):
            try:
                answer = json.loads(line)
            except ValueError:
                continue
            if isinstance(answer, dict) and answer.get("ev") == "probe":
                return answer
        if self.closed:                                   # stopped by close(): nobody waits for its answer
            return None
        tail = (err or "").strip()[-300:]
        self._log(f"qc: the mixing check's probe gave no answer (exit {process.returncode})"
                  + (f": {tail}" if tail else ""))
        return None

    def _probe(self):
        """Ask the probe (in this thread) and keep what its answer means (capability_of). Without a runnable [qc] python
        the answer only orders the reason: a definite no gives its own (a computer without an NVIDIA card is told that
        first), anything else leaves "software". When what is offered, or why not, changed, every sample known here is
        looked at again (its new state reaches the pages: snapshot) and the analysis scheduler asks again which model
        scores its runs (on_ready); one line in the log per change."""
        with self.lock:
            self.probing = True
        try:
            answer = self.prober()
        except Exception as exc:                          # a prober in trouble: no answer
            answer = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            state, reason, why = capability_of(answer)
        except Exception as exc:                          # a line of another shape (a count as text): no answer either
            answer = {"error": f"{type(exc).__name__} reading its answer: {exc}"}
            state, reason, why = capability_of(answer)
        if not self.s.available and state != "unavailable":
            state, reason, why = "unavailable", "software", self.no_python
        with self.lock:
            self.probing = False
            if self.closed:
                return
            before = self.capability_state
            self.capability_state = {"state": state, "reason": reason, "why": why,
                                     "probe": dict(answer) if isinstance(answer, dict) else None}
            self.probed_at = self.clock()
            self.unanswered = self.unanswered + 1 if state == "unknown" else 0
            offered = (before["state"] != "unavailable") != (state != "unavailable")
            moved = device_of(before.get("probe")) != device_of(answer)      # Apple's GPU found: other keys
            records = list(self.records.values()) if offered or moved or before["reason"] != reason else []
            if records:
                self.version += 1
            for record in records:                        # whoever looks next looks again (describe, ensure, model)
                record["checked"] = -math.inf
            self.wake.notify_all()
        if (before["state"], before["reason"]) != (state, reason):    # the check's own log (the Log shows its tail)
            self._log(f"qc: the mixing check {state}: {why}", echo=False)
        for record in records:                            # waiting or learning ones keep their job (_refresh)
            try:
                self._refresh(record, force=True)
            except Exception as exc:
                self._log(f"qc: looking at {record['sample']} again failed: {type(exc).__name__}: {exc}")
        if offered:
            for hook in list(self.on_ready):
                try:
                    hook()
                except Exception as exc:
                    self._log(f"qc: telling others that the mixing check is {state} failed: {exc}")

    def _probe_loop(self):
        """The probe at start, then again after an answer that was none (PROBE_RETRY_S, its last delay repeating) and
        when asked (ask_again), until close(). Without a runnable [qc] python one answer is all."""
        self._probe()
        while self.s.available:
            with self.lock:
                while not self.closed and not self.probe_wanted:
                    if self.capability_state["state"] != "unknown":
                        self.wake.wait()
                        continue
                    delay = PROBE_RETRY_S[min(self.unanswered, len(PROBE_RETRY_S)) - 1]
                    remaining = self.probed_at + delay - self.clock()
                    if remaining <= 0:
                        break
                    self.wake.wait(remaining)
                if self.closed:
                    return
                self.probe_wanted = False
            self._probe()

    def _wait_capability(self, record: dict, detail: bool = False) -> bool:
        """Before a fit or a detail job reads anything: whether the check can learn here. While its probe has not
        answered the job waits for it (at most PROBE_TIMEOUT_S + 10 s, then it goes on as without an answer); without
        a probe thread (an instance made with start=False, whose fit a caller runs) it asks the probe here first. A
        definite no: the sample becomes unavailable in the probe's words (a detail job is dropped) and False; else
        True (with no answer the trainer's own checks say what fails, as without the probe)."""
        with self.lock:
            own = self.capability_state["state"] == "checking" and self.probe_thread is None and not self.probing
        if own:
            self._probe()
        with self.lock:
            deadline = time.monotonic() + PROBE_TIMEOUT_S + 10
            while self.capability_state["state"] == "checking" and not self.closed and time.monotonic() < deadline:
                self.wake.wait(min(1.0, max(0.0, deadline - time.monotonic())))
            if self.capability_state["state"] != "unavailable":
                return True
            if detail:
                record["detail"] = self._no_detail()
                record["changed"] = True
                self.version += 1
        self._refresh(record, force=True, internal=True)
        return False

    # ---- the background thread ------------------------------------------------------------------------------------
    def _loop(self):
        while True:
            with self.lock:
                while not self.queue and not self.closed:
                    self.wake.wait()
                if self.closed:
                    return
                sid = self.queue[0]
                record = self.records[sid]
                detail = record["state"] == "ready" and record["detail"]["state"] == "queued"
            try:
                if detail:
                    self._learn_detail(record)
                else:
                    self._learn(record)
            except Exception as exc:                  # whatever happens, the thread survives and says so
                self._log(f"qc: {'adding the details to' if detail else 'learning'} {sid} failed unexpectedly: "
                          f"{type(exc).__name__}: {exc}")
                if detail:
                    self._set_detail(record, state="failed", reason="unexpected", message=str(exc)[:300],
                                     progress=None, eta_s=None, stage=None)
                else:
                    self._set(record, state="failed", reason="unexpected", message=str(exc)[:300], progress=None,
                              eta_s=None, stage=None)
            finally:
                with self.lock:
                    again = record["state"] == "waiting" or \
                        (record["state"] == "ready" and record["detail"]["state"] == "queued")
                    if self.queue and self.queue[0] == sid:
                        self.queue.pop(0)
                        if again:                                    # requeued (retry): to the back
                            self.queue.append(sid)

    def _wait_quietly(self, seconds: float) -> bool:
        """Sleep, waking early on close(); False once closed."""
        with self.lock:
            if not self.closed:
                self.wake.wait(seconds)
            return not self.closed

    def _back_off(self, seconds: float) -> bool:
        """Wait all of `seconds` (a job queued meanwhile wakes `wake` too, without ending it) unless close() comes
        first; False once closed."""
        with self.lock:
            return not self.wake.wait_for(lambda: self.closed, seconds)

    def _gpu_state(self) -> dict | None:
        """The probe: [qc] gpu as nvidia-smi sees it (gpu_probe), or {"refused": why} when the trainer could not learn
        on it at all, said before anything is read (as _problem says what the trainer would refuse in a table): the
        engine sees no such GPU (engine/children.py gpu gives ""), or it is not the one the trainer learns on
        (TRAINER_GPU)."""
        device = children.gpu(self.s.gpu)
        if not device:
            return {"refused": f"[qc] gpu {self.s.gpu} is not one of the GPUs this engine sees: there is none to "
                               "learn on"}
        if TRAINER_GPU is not None and device != TRAINER_GPU:
            return {"refused": f"[qc] gpu {self.s.gpu} is GPU {device} here, and the learning process learns on GPU "
                               f"{TRAINER_GPU} only"}
        return gpu_probe(device)

    def _room_mb(self, record: dict) -> tuple[int, int | None]:
        """(the free memory a fit of the record's table needs on [qc] gpu, in MB; its genes): [qc] gpu_min_free_mb when
        it is set, else room_needed_mb of the table's panel and cells."""
        table = record.get("table") or {}
        genes = table.get("n_vars")
        if self.s.gpu_min_free_mb is not None:
            return int(self.s.gpu_min_free_mb), genes
        return room_needed_mb(int(genes or 0), int(table.get("n_obs") or 0), record.get("device") or "cuda"), genes

    def _wait_for_gpu(self, record: dict, detail: bool = False) -> bool:
        """Room on [qc] gpu: at least the free memory a fit of this table needs (_room_mb, from its panel), whatever
        else runs there (sampled every second). While it waits it says why (a detail job in its own record: the sample
        stays ready), and the record's `room` how much it needs and how much is free: {need_mb, free_mb (to the nearest
        100 MB), total_mb, genes}. A card with less than that in all can never learn the table: failed, reason
        "gpu_small", at once and with nothing read. False then, when closed, or when the probe says that GPU cannot be
        learned on (failed, reason "gpu", with nothing read)."""
        need, genes = self._room_mb(record)
        if record.get("device") == "mps":
            return self._apple_room(record, need, genes, detail)
        while True:
            with self.lock:
                if self.closed:
                    return False
                if not detail and record["state"] == "ready":   # found stored meanwhile: the look after this stops it
                    return True
            probe = self.probe()
            if probe is None:
                self._log(f"qc: nvidia-smi gave no answer: learning without checking GPU {self.s.gpu}'s free memory")
                self._set(record, room=None)
                return True
            if probe.get("refused"):
                self._log(f"qc: not learning {record['sample']}: {probe['refused']}")
                failed = dict(state="failed", reason="gpu", message=probe["refused"], progress=None, eta_s=None,
                              stage=None)
                if detail:
                    self._set_detail(record, **failed)
                else:
                    self._set(record, **failed)
                return False
            total = probe.get("total_mb")
            room = {"need_mb": need, "free_mb": int(round(probe["free_mb"] / 100.0) * 100),
                    "total_mb": int(total) if total is not None else None, "genes": genes}
            if total is not None and total < need:
                words = (f"learning it needs about {need} MB free on GPU {self.s.gpu} ({genes} genes) and that card has "
                         f"{int(total)} MB in all")
                self._log(f"qc: not learning {record['sample']}: {words}")
                failed = dict(state="failed", reason="gpu_small", message=words, progress=None, eta_s=None, stage=None)
                if detail:
                    self._set_detail(record, **failed)
                    self._set(record, room=room)
                else:
                    self._set(record, room=room, **failed)
                return False
            if probe["free_mb"] >= need:
                self._set(record, room=None)
                return True
            self._set(record, room=room)
            if detail:
                self._set_detail(record, reason="gpu_memory")
            elif record["reason"] != "gpu_memory":
                self._set(record, reason="gpu_memory")
            if not self._wait_quietly(GPU_POLL_S):
                return False

    def _apple_room(self, record: dict, need: int, genes, detail: bool) -> bool:
        """On Apple's GPU, never a wait: its memory is the Mac's own, which macOS makes room in as a fit asks (what it
        calls available stays low beside other programs, so waiting for it could never end). Only a Mac whose GPU may
        use less than the fit needs in all (Metal's share, apple_memory) can never learn the table: failed, reason
        "gpu_small", at once, its room saying so ("mac")."""
        memory = apple_memory()
        total = memory["total_mb"] if memory else None
        if total is None:
            self._log("qc: the Mac's memory could not be read: learning without checking it")
        if total is None or total >= need:
            self._set(record, room=None)
            return True
        room = {"need_mb": need, "free_mb": int(round(memory["free_mb"] / 100.0) * 100), "total_mb": int(total),
                "genes": genes, "mac": True}
        words = (f"learning it needs about {need} MB on Apple's GPU ({genes} genes, its counts included) and this Mac "
                 f"lets it use {int(total)} MB in all")
        self._log(f"qc: not learning {record['sample']}: {words}")
        failed = dict(state="failed", reason="gpu_small", message=words, progress=None, eta_s=None, stage=None)
        if detail:
            self._set_detail(record, **failed)
            self._set(record, room=room)
        else:
            self._set(record, room=room, **failed)
        return False

    def _fit_lock(self):
        """The fit locks, non-blocking: the data dir's (engines sharing it) and the computer's (fit_lock_paths: every
        engine of this user, of either version, so fits on GPU 0 never overlap). A list of open handles (close them to
        release), or None while another engine holds any, or a trainer of this one that outlived SIGKILL still runs."""
        self._release_stuck()
        if self.stuck:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        machines = fit_lock_paths()
        for machine in machines:
            try:
                machine.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        held = []
        for path in (self.root / ".fit.lock", *machines):
            try:
                handle = open(path, "a")
            except OSError as exc:
                if path not in machines:              # the data dir must be writable (as for the models themselves)
                    for other in held:
                        other.close()
                    raise
                if not self.lock_warned:              # a folder we cannot write: the other locks only (said once)
                    self.lock_warned = True
                    self._log(f"qc: the computer-wide fit lock {path} cannot be used ({exc}): only the others")
                continue
            try:
                portable.lock(handle, block=False)
            except BlockingIOError:
                handle.close()
                for other in held:
                    other.close()
                return None
            held.append(handle)
        return held

    def _release_stuck(self):
        """Close the fit locks of trainers that outlived SIGKILL (stuck in the GPU driver) once they are gone."""
        with self.lock:
            stuck, self.stuck = self.stuck, []
        for process, handles in stuck:
            if process.poll() is None:
                with self.lock:
                    self.stuck.append((process, handles))
                continue
            for handle in handles:
                handle.close()

    def _ready_to_fit(self, record: dict) -> bool:
        """Look at the queued sample again (its key is not known yet when it was queued during its first look, and may
        be stale by now: a table saved again, or a model another engine or the pre-learning command stored meanwhile):
        True while it still has no model and a table to learn from."""
        self._refresh(record, force=True, internal=True, off_as="waiting")
        with self.lock:
            return record["state"] == "waiting" and bool(record["table"]) and bool(record["key"])

    def _learn(self, record: dict):
        if not self._ready_to_fit(record):                # a model meanwhile, or nothing to learn from
            return
        if not self._wait_capability(record):             # the check cannot learn here: nothing is read
            return
        if not self._ready_to_fit(record):                # again, the probe answered: the device its key is made for
            return
        while True:
            if not self._wait_for_gpu(record):
                return
            lock = self._fit_lock()
            if lock is not None:
                break
            self._set(record, reason="lock")
            if not self._wait_quietly(2.0):
                return
        try:
            if self._ready_to_fit(record):                # again, with the locks: what changed while it waited
                self._fit(record, record["table"], record["key"])
        finally:
            with self.lock:
                process = self.process
            if process is not None and process.poll() is None:     # it outlived SIGKILL: no fit until it is gone
                with self.lock:
                    self.stuck.append((process, lock))
            else:
                for handle in lock:
                    handle.close()

    def _detail_needed(self, record: dict) -> bool:
        """Look at a sample with a queued detail job again (with fresh model files): True while its model is ready,
        the one the job was asked for, and still without deep.npz; otherwise the job is dropped (a relearned table,
        another engine or the pre-learning command added the details meanwhile)."""
        with self.lock:
            folder = self._folder(record)
        if folder is not None:
            self.summaries.pop(str(folder), None)
        self._refresh(record, force=True, internal=True)
        with self.lock:
            job = record["detail"]
            needed = record["state"] == "ready" and job["key"] == record["key"] and \
                not (record["model"] or {}).get("deep")
            if not needed and job["state"] in ("queued", "learning"):
                record["detail"] = self._no_detail()
                self.version += 1
                record["changed"] = True
            return needed

    def _learn_detail(self, record: dict):
        """A detail job: the fit's GPU 0 wait and fit locks, then `_fit_detail`."""
        if not self._detail_needed(record):
            return
        if not self._wait_capability(record, detail=True):
            return
        if not self._detail_needed(record):              # again, the probe answered: the device its key is made for
            return
        while True:
            if not self._wait_for_gpu(record, detail=True):
                return
            lock = self._fit_lock()
            if lock is not None:
                break
            self._set_detail(record, reason="lock")
            if not self._wait_quietly(2.0):
                return
        try:
            if self._detail_needed(record):              # again, with the locks
                self._fit_detail(record, record["table"], record["key"])
        finally:
            with self.lock:
                process = self.process
            if process is not None and process.poll() is None:
                with self.lock:
                    self.stuck.append((process, lock))
            else:
                for handle in lock:
                    handle.close()

    def _trainer_env(self, python: str | None = None) -> dict:
        """engine/children.py's environment for the trainer's interpreter (or `python`, the probe's): [qc] gpu (mapped)
        as the only GPU it sees, in PCI bus order, no desktop, small thread pools."""
        env = children.env(children.kind(python or self.s.python), gpu=self.s.gpu)
        env.update(PYTHONUNBUFFERED="1", **dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"),
                                                         TRAINER_THREADS))
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # on Apple's GPU an operation MPS lacks runs on the CPU (else nothing)
        return env

    def _log(self, message: str, echo: bool = True):
        """A line in qc-trainer.log (with its time), and in the engine's own output unless `echo` is False (the probe's
        answer at every start: a line from a background thread at any moment of the engine's output)."""
        if echo:
            print(message, flush=True)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as log:
                log.write(f"{now_iso()} {message}\n")
        except OSError:
            pass

    def _job(self, record: dict, info: dict, key: str, staging: Path) -> dict:
        """The trainer's job line for learning `info` (a fit, or with "detail_for" a detail job), with the transcripts'
        frame of the sample when this engine has it open (`_frame`)."""
        device = record.get("device") or "cuda"         # the one its key was made for (_refresh)
        return {"zarr": record["path"], "table": info["key"], "other_tables": [t for t in self.s.tables if t != info["key"]],
                "gpu": children.gpu(self.s.gpu),               # the CUDA_VISIBLE_DEVICES it is given (_trainer_env)
                "device": device,                              # "cuda", or "mps": Apple's GPU (device_of)
                "out_dir": str(staging), "model_id": key, "sample": {"id": record["sample"], "name": record["name"]},
                "recipe": self.s.recipe, "fingerprint": self._fingerprint(info), "mixed_at": self.s.mixed_at,
                "judge_min": self.s.judge_min, "deadline_s": self._time_limit(device), **self._frame(record["sample"])}

    def _time_limit(self, device: str | None) -> float:
        """[qc] time_limit_s for a fit on `device`: MPS_TIME_FACTOR times it on Apple's GPU, else as set."""
        return self.s.time_limit_s * MPS_TIME_FACTOR if device == "mps" else self.s.time_limit_s

    def _frame(self, sid: str) -> dict:
        """{points_key, extent} of a sample this engine has open (its detail, as the picker has it: the working frame's
        points element and the transcripts' [x0, y0, x1, y1]): the trainer places the tables' cells on them
        (qc_trainer.choose_centres). {} otherwise (the pre-learning command, a sample not opened in
        this session): the trainer then reads them itself, by the same rule. Never starts the helper."""
        details = getattr(self.zarr, "details", None)
        detail = details.get(sid) if isinstance(details, dict) else None
        if not isinstance(detail, dict):
            return {}
        points, extent = detail.get("points_key"), detail.get("extent")
        if not isinstance(points, str) or not isinstance(extent, list) or len(extent) != 4 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in extent):
            return {}
        return {"points_key": points, "extent": [float(v) for v in extent]}

    def _run_trainer(self, record: dict, job: dict, title: str, progress, check):
        """Run one trainer job to its end: ((state, reason, message), gone). The trainer is always stopped (never left
        running on GPU 0 without anyone following it) and the log closed; `gone` False when it outlived SIGKILL.
        `progress(record, message)` takes its progress lines, `check()` the export once it says done."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        log.write(f"{now_iso()} qc: {title}\n".encode())
        log.flush()
        try:
            process = subprocess.Popen([self.s.python, "-m", self.s.module], cwd=children.app_path(),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=self._trainer_env(),
                                       text=True, bufsize=1, close_fds=True, **portable.new_session())
            portable.tie(process)                     # macOS and Windows: as its own death signal does on Linux
        except OSError as exc:
            log.close()
            return ("unavailable", "software", str(exc)), True
        with self.lock:
            self.process = process
        replies: queue.Queue = queue.Queue()

        def read():
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    replies.put(message)
            replies.put({"ev": "exit", "code": process.wait()})
        try:
            threading.Thread(target=read, name="qc-trainer-reader", daemon=True).start()
            try:
                process.stdin.write(json.dumps(job) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):    # it died at once, or close() closed its stdin
                pass
            try:
                outcome = self._follow(record, process, replies, progress, check)
            except Exception as exc:                          # e.g. a malformed progress line
                outcome = ("failed", "unexpected", f"following the learning process failed: {type(exc).__name__}: {exc}")
            return outcome, self._kill()
        finally:
            if self._kill():
                with self.lock:
                    if self.process is process:
                        self.process = None
            log.close()

    def _fit(self, record: dict, info: dict, key: str):
        sid = record["sample"]
        folder = self._folder(record, key).parent
        folder.mkdir(parents=True, exist_ok=True)
        staging = folder / f".{key}.part-{os.getpid()}-{threading.get_ident()}"
        shutil.rmtree(staging, ignore_errors=True)
        job = self._job(record, info, key, staging)
        estimate = estimate_seconds(info["n_obs"], self.s.recipe, job["device"])
        self._set(record, state="learning", reason=None, message=None, stage="starting", estimate_s=estimate,
                  progress={"step": 0, "of": trainer.epochs_for(info["n_obs"], self.s.recipe)}, eta_s=estimate,
                  shared=False, learning={"started": self.clock(), "cells": info["n_obs"], "steps_per_epoch": None})
        installed = False
        try:
            outcome, gone = self._run_trainer(record, job, f"learning {sid} ({record['name']}) table {info['key']} as "
                                              f"{key}", self._progress,
                                              lambda: self._check_export(staging, key, job["device"]))
            state, reason, message = outcome
            if state == "ready" and gone:
                self._install(record, staging, key)
                installed = True
                return
        finally:
            if not installed:
                shutil.rmtree(staging, ignore_errors=True)
        if state == "unavailable":                            # the [qc] python could not be started
            self._set(record, state="unavailable", reason=reason, message=message, progress=None, eta_s=None)
            return
        if state == "ready":                                  # the export was complete, but the process would not stop
            state, reason, message = "failed", "unexpected", "the learning process did not stop"
        self._log(f"qc: learning {sid} {state}: {reason}: {message}")
        if reason in TABLE_REASONS and info:              # the table's own: remembered, and the next one tried
            self._reject(record, info, reason)
            with self.lock:
                record.update(progress=None, eta_s=None, stage=None)
            # Another table remains: learn from it next ("waiting" keeps it queued), else unavailable.
            self._refresh(record, force=True, internal=True, off_as="waiting")
            return
        if reason in ("no_table",):
            self._set(record, state="unavailable", reason=reason, message=message, progress=None, eta_s=None, stage=None)
            return
        with self.lock:
            record["attempts"] += 1
            retry = reason == "oom" and record["attempts"] < 2 and not self.closed
        # GPU 0 is shared: once more, after OOM_RETRY_S, saying meanwhile that it waits for room on GPU 0 (the fit locks
        # stay held). The failed trainer's own memory is free again at once, so the wait for its room alone would
        # start the retry ~0.1 s later, into whatever filled GPU 0.
        if retry:
            self._set(record, state="waiting", reason="gpu_memory", message=None, progress=None, eta_s=None, stage=None)
            if self._back_off(OOM_RETRY_S):                  # close() ends it, and there is no retry
                self._queue(record)
            return
        if state == "closed":
            self._set(record, state="off", reason=None, message=None, progress=None, eta_s=None, stage=None)
            return
        self._set(record, state="failed", reason=reason, message=message, progress=None, eta_s=None, stage=None)

    def _fit_detail(self, record: dict, info: dict, key: str):
        """A detail job: the trainer learns the same recipe again with "detail_for" the installed folder;
        its deep.npz is checked against the installed calculator (`_check_detail`) and moved in, model.json rewritten
        (`_install_detail`). Any failure (the parity rule: "detail_mismatch") leaves the installed model as it was."""
        sid = record["sample"]
        final = self._folder(record, key)
        staging = final.with_name(f".{key}.detail-{os.getpid()}-{threading.get_ident()}")
        shutil.rmtree(staging, ignore_errors=True)
        job = {**self._job(record, info, key, staging), "detail_for": str(final)}
        estimate = estimate_seconds(info["n_obs"], self.s.recipe, job["device"])
        self._set_detail(record, state="learning", reason=None, message=None, stage="starting", eta_s=estimate,
                         progress={"step": 0, "of": trainer.epochs_for(info["n_obs"], self.s.recipe)},
                         learning={"started": self.clock(), "cells": info["n_obs"], "steps_per_epoch": None})
        try:
            outcome, gone = self._run_trainer(
                record, job, f"adding the details to {sid} ({record['name']}) model {key}",
                lambda rec, message: self._progress(rec, message, detail=True),
                lambda: self._check_detail(staging, key, final))
            state, reason, message = outcome
            if state == "ready" and gone:
                if self._install_detail(record, staging, key):
                    return
                state, reason, message = "failed", "unexpected", "the details could not be stored"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if state == "ready":
            state, reason, message = "failed", "unexpected", "the learning process did not stop"
        self._log(f"qc: adding the details to {sid} {state}: {reason}: {message}")
        with self.lock:
            record["detail"]["attempts"] += 1
            retry = reason == "oom" and record["detail"]["attempts"] < 2 and not self.closed
        # As a fit's (`_fit`): once more, after OOM_RETRY_S, so as not to start ~0.1 s later into whatever filled GPU 0.
        # Queued at once, so the job keeps this model's key (the look at its turn, `_detail_needed`, drops it if the
        # model changed meanwhile); until the pause ends it says it waits for room on GPU 0.
        if retry:
            with self.lock:
                self._queue_detail(record)
            self._set_detail(record, reason="gpu_memory")
            if self._back_off(OOM_RETRY_S):                  # close() ends it, and there is no retry
                self._set_detail(record, reason="queue")
            return
        if state == "closed":
            with self.lock:
                record["detail"] = self._no_detail()
                self.version += 1
            return
        self._set_detail(record, state="failed", reason=reason if reason in REASONS else "unexpected",
                         message=message, progress=None, eta_s=None, stage=None)

    def _follow(self, record: dict, process, replies: queue.Queue, progress, check):
        """Read the trainer until done, error, exit, the time limit or close(): (state, reason, message)."""
        started = self.clock()
        hello = False
        device = record.get("device") or "cuda"
        expected_bus = None if device == "mps" else bus_number((self.probe() or {}).get("bus"))
        limit = self._time_limit(device)
        while True:
            with self.lock:
                if self.closed:
                    return "closed", "closed", "the engine is closing"
            elapsed = self.clock() - started
            if elapsed > limit + KILL_GRACE_S:
                return "failed", "timeout", f"no result within {limit / 60:g} minutes"
            if not hello and elapsed > HELLO_TIMEOUT_S:
                return "failed", "unexpected", "the learning process did not start"
            try:
                message = replies.get(timeout=0.25)
            except queue.Empty:
                continue
            kind = message.get("ev")
            if kind == "hello":
                hello = True
                if message.get("cuda_visible_devices") != children.gpu(self.s.gpu):
                    return "failed", "gpu", f"the learning process could see a GPU other than GPU {self.s.gpu}"
                continue
            if kind == "progress":
                device = message.get("device") or {}
                if device and expected_bus is not None and device.get("pci_bus_id") not in (None, -1, expected_bus):
                    return "failed", "gpu", (f"the learning process landed on PCI bus {device.get('pci_bus_id')}, "
                                             f"not GPU {self.s.gpu}")
                progress(record, message)
                continue
            if kind == "done":
                try:
                    check()
                except DetailMismatch as exc:
                    return "failed", "detail_mismatch", str(exc)[:500]
                except Exception as exc:                  # a damaged archive is BadZipFile or EOFError, not OSError
                    return "failed", "unexpected", f"the exported model failed its check: {type(exc).__name__}: {exc}"
                return "ready", None, None
            if kind == "error":
                reason = message.get("reason") if message.get("reason") in REASONS else "unexpected"
                return "failed", reason, str(message.get("message") or "")[:500]
            if kind == "exit":
                return "failed", "unexpected", f"the learning process exited (code {message.get('code')})"

    def _progress(self, record: dict, message: dict, detail: bool = False):
        """A progress line of the trainer: the stage, the passes done and the time left, in the sample's record (a
        fit) or its detail job's (`detail`: the sample stays ready meanwhile)."""
        stage = message.get("stage")
        with self.lock:
            holder = record["detail"] if detail else record
            learning = holder["learning"] or {}
            values = {"stage": stage}
            apple = record.get("device") == "mps"
            step_s = step_seconds("mps" if apple else "cuda")
            if stage == "preparing" and message.get("epochs"):
                learning["steps_per_epoch"] = math.ceil(int(message.get("cells") or 1) / self.s.recipe["batch_size"])
                values["progress"] = {"step": 0, "of": int(message["epochs"])}
                values["eta_s"] = int(FIXED_S * 0.6 + step_s * int(message.get("steps") or 0))
            elif stage == "learning" and message.get("of"):
                step, of = int(message.get("step") or 0), int(message["of"])
                values["progress"] = {"step": step, "of": of}
                seconds = message.get("seconds")
                if step and seconds:
                    per = float(seconds) / step
                    values["eta_s"] = int(math.ceil((of - step) * per + 5))
                    batches = learning.get("steps_per_epoch")
                    if not detail:                  # Apple's chips differ too much to tell a shared GPU
                        values["shared"] = bool(not apple and batches and per / batches > 2 * step_s)
            elif stage in ("checking", "exporting", "scoring"):
                values["eta_s"] = 5
            holder["learning"] = learning
            holder.update(values)
            self.version += 1
            record["changed"] = True

    def _check_export(self, staging: Path, key: str, device: str = "cuda"):
        """A fit's export, before it is installed: the calculator (strict loader, its 256 reference cells within 1e-5),
        model.json naming this model and the device its key was made for (its "device" kind; "cuda" before 0.6.0),
        cells.parquet; and Go deeper's deep.npz when there is one (strict loader, its
        self-test against the calculator within DEEP_TOLERANCE). A deep.npz failing its check is left out (model.json
        says why, "deep_error", as the trainer does when its own export fails): the calculator still scores, and only
        Go deeper says the details are missing."""
        encoder = trainer.load_encoder(staging / "encoder.npz")
        parity = trainer.self_test(encoder)
        if not parity <= trainer.SELFTEST_TOLERANCE:
            raise ValueError(f"the reference cells differ by {parity:.2e}")
        data = json.loads((staging / "model.json").read_text())
        if data.get("id") != key or data.get("format") != trainer.FORMAT:
            raise ValueError("model.json does not describe this model")
        learned_on = (data.get("device") or {}).get("kind", "cuda") if isinstance(data.get("device"), dict) else "cuda"
        if learned_on != device:
            raise ValueError(f"the model was learned on {learned_on}, and its key is for {device}")
        if not (staging / "cells.parquet").is_file():
            raise ValueError("cells.parquet is missing")
        deep_path = staging / DEEP_FILE
        if not deep_path.is_file():
            if isinstance(data.get("deep"), dict):        # named but missing: not offered as there
                data.update(deep=None, deep_error={"reason": "unexpected", "message": "deep.npz is missing"})
                _atomic(staging / "model.json", json.dumps(data, indent=1).encode())
            return
        try:
            deep = trainer.load_deep(deep_path, encoder)
            diff = deep_parity(deep, encoder)
            if not diff <= DEEP_TOLERANCE:
                raise ValueError(f"the details' reference cells differ by {diff:.2e}")
            if not isinstance(data.get("deep"), dict):
                raise ValueError("model.json does not name deep.npz")
        except Exception as exc:                          # the details only: the calculator is fine
            deep_path.unlink(missing_ok=True)
            data.update(deep=None, deep_error={"reason": "unexpected",
                                               "message": f"the engine's check: {type(exc).__name__}: {exc}"[:500]})
            _atomic(staging / "model.json", json.dumps(data, indent=1).encode())
            self._log(f"qc: the details of {key} were left out: {type(exc).__name__}: {exc}")

    @staticmethod
    def _check_detail(staging: Path, key: str, final: Path):
        """A detail job's export, before anything is moved: model.json (this model, with its "deep" block, the parity
        record not refused), deep.npz (strict loader, its self-test against the INSTALLED calculator within
        DEEP_TOLERANCE) and, when the trainer left its new calculator beside it, the parity rule once more (identical
        arrays, or the installed reference cells within DETAIL_TOLERANCE). DetailMismatch when the new learning is
        another model."""
        data = json.loads((staging / "model.json").read_text())
        if data.get("id") != key or data.get("format") != trainer.FORMAT:
            raise ValueError("model.json does not describe this model")
        if not isinstance(data.get("deep"), dict):
            raise ValueError("model.json does not name deep.npz")
        if isinstance(data.get("detail"), dict) and data["detail"].get("accepted") is False:
            raise DetailMismatch("the new learning does not match the installed calculator")
        installed = trainer.load_encoder(final / "encoder.npz")
        spare = staging / getattr(trainer, "DETAIL_ENCODER", ".detail-encoder.npz")
        if spare.is_file() and not calculators_agree(installed, trainer.load_encoder(spare)):
            raise DetailMismatch("the new learning does not match the installed calculator")
        deep = trainer.load_deep(staging / DEEP_FILE, installed)
        diff = deep_parity(deep, installed)
        if not diff <= DEEP_TOLERANCE:
            raise ValueError(f"the details' reference cells differ by {diff:.2e} from the installed calculator")

    def _install_detail(self, record: dict, staging: Path, key: str) -> bool:
        """Move a checked detail job's deep.npz into the installed model's folder, then rewrite its model.json (the
        installed one with the job's "deep", "detail" and "provenance"; whole or not at all). The key, encoder.npz and
        the parquet files stay, so every run's results stay too. False (nothing changed) when the folder is gone."""
        final = self._folder(record, key)
        try:
            data = json.loads((staging / "model.json").read_text())
            current = json.loads((final / "model.json").read_text())
            if current.get("id") != key:
                return False
            merged = {k: v for k, v in current.items() if k != "deep_error"}
            merged["deep"] = data["deep"]
            if isinstance(data.get("detail"), dict):
                merged["detail"] = data["detail"]
            merged["provenance"] = data.get("provenance") if isinstance(data.get("provenance"), dict) \
                else provenance_of({k: v for k, v in merged.items() if k != "provenance"})
            os.replace(staging / DEEP_FILE, final / DEEP_FILE)
            _atomic(final / "model.json", json.dumps(merged, indent=1).encode())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._log(f"qc: the details of {record['sample']} could not be stored: {type(exc).__name__}: {exc}")
            return False
        with self.lock:
            self.summaries.pop(str(final), None)
        summary = self._summary(final)
        self._log(f"qc: added the details to {record['sample']} ({record['name']}) model {key}")
        # The model with its details first, then the job goes: a look in between sees "ready" (its deep), never
        # "none" (no job, no details yet), which the page would show for a moment.
        same = summary is not None and record["key"] == key
        if same:
            self._set(record, model=summary, checked=self.clock())
        with self.lock:
            record["detail"] = self._no_detail()
        if not same:
            self._changed(record)
        return True

    def _install(self, record: dict, staging: Path, key: str):
        final = self._folder(record, key)
        try:
            os.replace(staging, final)
        except OSError:
            if not (final / "model.json").is_file():
                # Not another engine's model (models arrive whole, by one rename) but what an interrupted prune or a
                # crash left of one: moved aside, then this model goes in (else every relearn would fail the same way).
                aside = final.with_name(f".{key}.stale-{os.getpid()}-{threading.get_ident()}")
                try:
                    os.replace(final, aside)
                    os.replace(staging, final)
                except OSError:
                    shutil.rmtree(staging, ignore_errors=True)
                    self._set(record, state="failed", reason="unexpected", message="the model could not be stored",
                              progress=None, eta_s=None, stage=None)
                    return
                finally:
                    shutil.rmtree(aside, ignore_errors=True)
            else:                                        # another engine stored the same model meanwhile
                shutil.rmtree(staging, ignore_errors=True)
        self._log(f"qc: learned {record['sample']} ({record['name']}) as {key}")
        self._touch(final)
        summary = self._summary(final)
        self._set(record, state="ready" if summary else "failed", reason=None if summary else "unexpected",
                  message=None if summary else "the stored model could not be read", model=summary, relearn=False,
                  progress=None, eta_s=None, stage=None, attempts=0, checked=self.clock(), shared=False)
        self._prune()
        for hook in list(self.on_ready):             # e.g. the analysis scheduler: score this sample's runs now
            try:
                hook()
            except Exception as exc:
                self._log(f"qc: telling others that {record['sample']} is learned failed: {exc}")

    def _touch(self, folder: Path):
        """The model's `used` stamp (what pruning goes by), refreshed at most once an hour (no disk access between)."""
        now = time.time()
        if now - self.touched.get(str(folder), -math.inf) < 3600:
            return
        self.touched[str(folder)] = now
        stamp = folder / "used"
        try:
            if not stamp.exists() or now - stamp.stat().st_mtime > 3600:
                stamp.write_text(now_iso())
        except OSError:
            pass

    def _prune(self):
        """Keep the keep_models most recently used models (by their `used` stamp), never one in use right now: this
        engine's, and in a shared models folder ([qc] models_dir) any used within SHARED_KEEP_S, which another session
        may be using (each stamps its models at least hourly while it uses them, _touch)."""
        folders = []
        for path in self.root.glob("*/*/model.json"):
            if path.parent.name.startswith("."):
                continue
            stamp = path.parent / "used"
            try:
                folders.append(((stamp if stamp.exists() else path).stat().st_mtime, path.parent))
            except OSError:
                continue
        folders.sort(reverse=True)
        with self.lock:
            busy = {str(self._folder(r)) for r in self.records.values() if r["key"] and r["state"] == "ready"}
        recent = time.time() - SHARED_KEEP_S if self.s.models_dir else math.inf
        for used, folder in folders[self.s.keep_models:]:
            if str(folder) not in busy and used < recent:
                shutil.rmtree(folder, ignore_errors=True)
                self.summaries.pop(str(folder), None)

    def _sweep_staging(self):
        """Staging folders of fits and detail jobs that never finished (a crash, a kill) and are older than an hour,
        and broken model folders `_install` moved aside but could not remove."""
        for folder in [*self.root.glob("*/.*.part-*"), *self.root.glob("*/.*.detail-*"),
                       *self.root.glob("*/.*.stale-*")]:
            try:
                if time.time() - folder.stat().st_mtime > 3600:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                pass

    def _kill(self) -> bool:
        """Stop the running trainer: the end of its stdin, then SIGTERM, then SIGKILL. Never raises. True once it is gone
        (or there is none); False when it outlived SIGKILL (stuck in the GPU driver): said in the log, and its fit
        locks stay held until it is gone (`_learn`, `_release_stuck`)."""
        with self.lock:
            process = self.process
        if process is None or process.poll() is not None:
            return True
        try:
            process.stdin.close()                      # the trainer leaves on the end of its stdin ...
        except (OSError, ValueError):
            pass
        for stop, seconds in ((None, 3), (process.terminate, 5), (process.kill, 10)):   # ... or on SIGTERM, or now
            if stop is not None:
                try:
                    stop()
                except OSError:
                    pass
            try:
                process.wait(timeout=seconds)
                return True
            except subprocess.TimeoutExpired:
                continue
        self._log(f"qc: the learning process (pid {process.pid}) did not exit after SIGKILL: no fit until it does")
        return False


# ---- QcManager ----------------------------------------------------------------------------------------------------
class RunQc:
    """One run's mixing check: the switch, the model it is scored by, and its results so far."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.lock = threading.RLock()
        self.enabled: bool | None = None
        self.synced: bool | None = None              # the switch as the analysis scheduler last got it
        self.key: str | None = None
        self.dir: Path | None = None                 # <run>/qc/<model key[:8]>
        self.model_dir: Path | None = None           # <data_dir>/qc-models/<sample>/<model key>
        self.start: dict | None = None
        self.results: dict[int, str] = {}            # the newest RESULTS_KEPT summaries as stored (JSON lines: areas
                                                     # name their outline's shape), oldest first (insertion order)
        self.points: dict[int, dict] = {}            # every result's point of the line (POINT_KEYS), by seq
        self.ema: list[float] | None = None          # smoothed [judged, mixed] over live results
        self.ema_matched: list[float] | None = None  # smoothed [paired, mixed now, mixed at the start] at the matched
                                                     # depth (the worker's "matched"), as the headline is smoothed
        self.ema_start: list[float] | None = None    # smoothed [judged, mixed] of the frames' own start blocks (the
                                                     # worker's "start_block"), as the headline is smoothed
        self.provenance: dict | None = None          # the model's provenance (model.json), carried by every summary
        self.shown: int | None = None
        self.last_live: int | None = None
        self.last_t: float | None = None             # the run clock (frame "t") of the last live result
        self.live_count = 0
        self.trial: dict | None = None
        self.trials = 0
        self.errors: list[dict] = []
        self.batch: dict | None = None               # {key, per100, judged, mixed, cells}
        self.tracker: qa.Tracker | None = None       # the flagged areas over the live results
        self.start_bins: dict | None = None          # the start line's per-bin sums (dense), for the tags and its areas
        self.start_areas: dict | None = None         # {areas (as stored), grid} of the start line
        self.shapes: dict[str, list] = {}            # outline of each area shape (shapes.jsonl)
        self.shapes_pending: list[str] = []          # shapes not written to shapes.jsonl yet
        self.start_drawn: list | None = None         # [(letter, bins)] of the start line's drawn areas (its grid)
        self.displayed: tuple | None = None          # (letters, bins) of the last live summary's areas: a change saves
                                                     # state.json at once (the letters resume after a restart)
        self.state_saved = -math.inf                 # when state.json was last written (the manager's clock)
        self.state_dirty = False                     # a live result since then
        self.reveal_iter: int | None = None          # areas are shown from this iteration (burn-in + 20)
        self.switch_n = 0                            # how often the switch changed (settings.json "n"): pushes and
                                                     # answers carry it, so a page keeps the newest of the two
        self.area_order: list[str] = []              # the areas' letters in the order a summary lists them (the cards
                                                     # and trials read them in that order)
        self.drawn_log: list[tuple] = []             # [(seq, [(letter, bins)], start)]: what the page drew from each
                                                     # live result on (drawn.jsonl), for the letters of single snapshots
        self.drawn_pending: list[str] = []           # drawn.jsonl lines not written yet
        self.shapes_missing: set[str] = set()        # shape ids shapes.jsonl does not hold (read once, not again)
        self.tails: set[str] = set()                 # append-only files whose last line was checked this session
        # Suggestions and trials (engine/qc_rules.py).
        self.recent: list[dict] = []                 # the newest live results, for the band of the next knob change
        self.prev_live: tuple | None = None          # (seq, knob values, values asked for) of the previous live result
        self.watch: qr.Watch | None = None           # the newest live knob change (the Undo card watches it)
        self.watch_n = 0
        self.trial_watch: qr.Watch | None = None     # the running trial's change (frozen when another change came)
        self.trial_view: dict | None = None          # the running trial as last shown (state.json keeps it)
        self.trial_closed: dict | None = None        # the trial its run's end closed (outcome "ended"), as last shown
        self.trial_n = 0                             # trial changes so far (the WebSocket's "trial" push)
        self.tried: list[dict] = []                  # ended trials {area, flag, from, to, verdict} (rule R7)
        self.keeper = qr.CardKeeper()                # one card at a time, steady
        self.steady = qr.Steady()                    # the verdicts live results show, held steady (qc_rules.Steady)


def _per100(mixed, judged):
    return round(100.0 * mixed / judged, 1) if judged else None


def counted_again(run, seq: int, stored: str) -> bool:
    """Whether frame `seq` was counted again after the result kept of it (`stored`: the count that result read): its
    counts line names another count now (counts_file.count_of) and that count's file is there, so the check scores it
    again (engine/analysis.py `_stands`) and the worker rewrites its <seq>.qc.bin before the new result is in; with the
    file gone the result kept stands, as it is."""
    with run.lock:
        frame = run.frames[seq] if isinstance(seq, int) and 0 <= seq < len(run.frames) else {}
        line, name = frame.get("counts"), frame.get("counts_file")
    return count_of(line) != stored and bool(name) and (run.live / name).exists()


def _share(mixed, judged):
    """The share in 100, unrounded (None with nothing judged): every whole number shown is rounded once, from it."""
    return 100.0 * mixed / judged if judged else None


def _whole(share):
    """A share's whole number, rounded once (half up, as the page rounds): 8.46 reads 8, never 8.5 and then 9."""
    return None if share is None else int(math.floor(share + 0.5))


def _keep(state: RunQc, seq: int, line: str, point: dict):
    """A result: its point of the line (kept for every result) and its summary as stored (`line`) among the newest
    RESULTS_KEPT (dropped oldest first, one at a time: never a scan of what is kept). State lock held."""
    state.points[seq] = point
    state.results.pop(seq, None)                     # a frame scored again counts as new
    state.results[seq] = line
    while len(state.results) > RESULTS_KEPT:
        del state.results[next(iter(state.results))]


def _point(summary: dict) -> dict:
    point = {k: summary.get(k) for k in POINT_KEYS}
    change = summary.get("change")
    point["matched_change"] = change.get("matched") if isinstance(change, dict) else None
    return point


def _int(value) -> int | None:
    """A count from the worker's summary (a whole, non-negative number), else None."""
    x = qr.number(value)
    return None if x is None or x < 0 else int(round(x))


MATCHED_KEYS = ("paired", "used", "per100_now", "per100_start", "change", "sd", "seeds")


def _matched(value) -> dict | None:
    """The GPU worker's like-for-like change of a frame, as it gave it (`matched`: each cell judged now and in the
    frame's start block, both scaled to the shallower depth and scored; deterministic, no draws): {paired (cells judged
    now and at the start), used (the cells compared: all the paired ones), per100_now, per100_start (cells that look
    mixed per 100 compared, at the matched depth; null when none could be compared), change (per100_now -
    per100_start), sd (null: there are no draws), seeds (0)}, its values checked; None when the worker gave none (no
    start block to compare with, or an older worker)."""
    if not isinstance(value, dict):
        return None
    out = {k: (_int(value[k]) if k in ("paired", "seeds") else qr.exact(value[k])) for k in MATCHED_KEYS if k in value}
    if out.get("change") is None and out.get("per100_now") is not None and out.get("per100_start") is not None:
        out["change"] = qr.exact(out["per100_now"] - out["per100_start"])
    return out or None


def _matched_counts(matched: dict | None) -> list[float] | None:
    """[cells compared, of them mixed now, mixed at the start] (at the matched depth) of a frame's matched change, for
    smoothing and the thresholds' n; None when nothing could be compared."""
    if not matched or matched.get("per100_now") is None or matched.get("per100_start") is None:
        return None
    n = matched.get("used") if qr.number(matched.get("used")) else matched.get("paired")
    if not qr.number(n) or n <= 0:
        return None
    n = float(n)
    return [n, n * matched["per100_now"] / 100.0, n * matched["per100_start"] / 100.0]


def _smoothed(old: list | None, new: list | None, weight: float | None) -> list | None:
    """An exponential average of counts (the headline's weight on the newest); the newest alone when there is no
    history (or no weight: the first live result)."""
    if new is None:
        return old
    if old is None or weight is None or len(old) != len(new):
        return [float(v) for v in new]
    return [(1 - weight) * a + weight * b for a, b in zip(old, new)]


def _coverage(result: dict, mode: str = "region") -> dict:
    """How much of the run a frame's (or the start line's) number rests on, over ALL the run's cells:
    {cells, judged (>= judge_min transcripts of the model's genes in proseg's count), empty (0), under20 (under 20),
    under100 (20 to under 100: proseg's count is fractional), judged_share, enough (>= the mode's local_min_judged,
    300 judged cells: the minimum for any verdict)}. The worker's counts; an older worker gives judged and cells
    only."""
    cells, judged = _int(result.get("ncells", result.get("cells"))), _int(result.get("judged")) or 0
    share = judged / cells if cells else None
    return {"cells": cells, "judged": judged, "empty": _int(result.get("empty")),
            "under20": _int(result.get("under20")), "under100": _int(result.get("under100")),
            "judged_share": qr.exact(share), "enough": judged >= qk.value(mode, "local_min_judged")}


START_BLOCK_KEYS = ("per100", "judged", "mixed", "tx_sum", "median_tx", "iteration")
# The per-bin sums that compare a frame with its start (its own start block, the paired cells, the like-for-like sums,
# the start's transcripts): a run whose start has nuclei has no start line, so its areas never use them, whatever a
# bins file holds (qc_areas.dense names).
START_SUMS = ("J0", "M0", "B", "N1", "S1", "n10", "n01", *qa.MATCHED, "tx_start")


def _start_block(value) -> dict | None:
    """A frame's own start as the worker gave it (`start_block`: the cells proseg started with, their RNA kept as
    proseg keeps it at this frame, scored at the block's own depth: {per100, judged, mixed, tx_sum, median_tx,
    iteration}), its counts checked; None without one (a run with no start line, a frame proseg wrote no start block
    for, or an older worker)."""
    if not isinstance(value, dict):
        return None
    judged, mixed = _int(value.get("judged")), _int(value.get("mixed"))
    if judged is None or mixed is None or mixed > judged:
        return None
    out = {k: qr.number(value.get(k)) for k in START_BLOCK_KEYS}
    out.update(judged=judged, mixed=mixed)
    return out



def _point_in_polygon(x: np.ndarray, y: np.ndarray, points) -> np.ndarray:
    """Even-odd rule, vectorised over the points (a lasso region)."""
    inside = np.zeros(len(x), bool)
    poly = np.asarray(points, dtype=np.float64)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-300, yj - yi) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


class QcManager:
    """What the page is told about each run's mixing check, and the analysis scheduler's QC provider
    (engine/analysis.py `attach_qc`): the scheduler asks qc_model(run) which model scores a run and hands every result
    of the GPU worker to qc_result(record), in its own thread. This manager keeps the scheduler's switch for each run
    in step with its own (`configure_qc`: the scheduler starts every run off, also after a restart) and tells it when a
    sample finished learning (`qc_changed`). It writes <seq>.qc.json and start.json beside the worker's files: a later
    session (of either) finds the frames done there."""

    def __init__(self, config: dict, runs, models: QcModels, analysis=None, *, clock=time.monotonic):
        self.s = models.s
        self.runs = runs
        self.models = models
        self.analysis = analysis
        self.clock = clock
        self.lock = threading.RLock()
        self.states: dict[str, RunQc] = {}
        self.local_version = 0
        self.prefs_path = prefs_path(config)          # the engine's own, also when models_dir is shared
        try:
            self.prefs = json.loads(self.prefs_path.read_text())
            if not isinstance(self.prefs.get("enabled"), bool):
                raise ValueError
        except (OSError, ValueError, AttributeError):
            self.prefs = {"enabled": self.s.enabled_by_default}
        models.on_ready.append(self._models_changed)

    @property
    def version(self) -> int:
        return self.local_version + self.models.version

    def _bump(self):
        with self.lock:
            self.local_version += 1

    def _models_changed(self):
        """A sample finished learning: the scheduler asks again at once which model scores each run."""
        if self.analysis is not None and hasattr(self.analysis, "qc_changed"):
            self.analysis.qc_changed()
        self._bump()

    def watching(self, run_id: str) -> bool:
        """Whether a knob change is being judged on the run now: a trial is registered and its change has not shown
        yet, or a change (the trial's, or the newest made in Live knobs) can still get a verdict or an Undo card
        (qc_rules.Watch.judging: it has a band and is not past its horizon or the last settling stage, not followed by
        another change, its Undo card not answered). The analysis scheduler then scores that run's newest counted
        frames before any background pass: a verdict needs their results. A change that can never have a band (made
        while cells form or in the last settling stage, too soon after cells formed, or on a run without a start line)
        gets no such priority. Cheap, and reads only what this session holds."""
        with self.lock:
            state = self.states.get(run_id)
        if state is None:
            return False
        with state.lock:
            if state.trial is not None and state.trial_watch is None:
                return True
            return any(watch is not None and watch.judging for watch in (state.trial_watch, state.watch))

    def new_runs(self) -> bool:
        """The switch new runs start with: the last choice a user made (prefs.json), else [qc] enabled_by_default (off
        as shipped). GET /samples/{id}/qc says it, so a freshly loaded picker knows whether to speak of the check."""
        return bool(self.prefs.get("enabled"))

    # ---- lifecycle hooks (server/app.py) ---------------------------------------------------------------------------
    def sample_opened(self, detail: dict):
        """Open a sample with the check on ([qc] learn = "open"): learn it now if it has no model."""
        if self.s.learn == "open" and self.prefs.get("enabled") and detail.get("kind") == "zarr":
            self.models.ensure(detail["id"])

    def run_created(self, meta: dict):
        """A new run: its switch (the remembered choice), handed to the scheduler, and its sample's model."""
        state = self._state(meta["id"])
        if state is None:
            return
        if state.enabled and self.s.learn in ("open", "start") and meta.get("sample_kind") == "zarr":
            self.models.ensure(meta["sample"])

    def forget(self, run_id: str):
        with self.lock:
            self.states.pop(run_id, None)
        self._bump()

    def close(self):
        self.models.close()                          # first: no fit goes on on GPU 0 (the save dialog may wait)
        with self.lock:
            states = list(self.states.values())
        for state in states:                         # the smoothing as it is now, for the next session
            try:
                run = self.runs.get(state.run_id)
                with state.lock:
                    if state.dir is not None and run.dir.exists():
                        self._save_state(run, state)
            except Exception:
                continue

    # ---- for the analysis scheduler (engine/analysis.py) -----------------------------------------------------------
    def qc_model(self, run) -> dict | None:
        """{id, path (the model folder holding encoder.npz), deep (its deep.npz for Go deeper, or None: a model learned
        before Go deeper)} of the model that scores this run now; None while its sample has none ready (or is no
        SpatialData sample). Cheap: the scheduler asks often."""
        model = self._model_for(run)
        return None if model is None else {"id": model["id"], "path": model["dir"], "deep": model.get("deep")}

    def qc_result(self, record: dict):
        """A result of the GPU worker (engine/analysis.py `_deliver`): {kind: frame | start | error, run, seq, model,
        dir, source, single, frame, summary, bins, cells, cells_e, count, start, error}. A result for another model than
        the run's current one is not kept (it would mix two yardsticks)."""
        state = self._state(record["run"])
        if state is None or state.key is None or record.get("model") != state.key:
            return
        kind = record.get("kind")
        if kind == "frame":
            self.ingest(record["run"], int(record["seq"]), record.get("summary") or {}, single=bool(record.get("single")),
                        frame=record.get("frame"), bins=record.get("bins"), count=record.get("count"))
        elif kind == "start":
            self.ingest_start(record["run"], record.get("summary") or {}, bins=record.get("bins"))
        elif kind == "error":
            self.fail(record["run"], record.get("seq"), record.get("error") or "the mixing check failed")

    def _sync(self, run_id: str, state: RunQc, seq: int | None = None) -> dict | None:
        """Give the scheduler this run's switch (and a frame to score now); its reply (configure_qc's: `requested`, the
        counted frame the frame asked for snaps to), or None. A frame asked for when the analysis worker cannot run
        raises (503, saying why: the scheduler's unavailable_why); the switch alone is retried at the next look."""
        analysis = self.analysis
        if analysis is None or not hasattr(analysis, "configure_qc") or not getattr(analysis, "available", True):
            if seq is not None:
                why = analysis.unavailable_why() if hasattr(analysis, "unavailable_why") else \
                    "no analysis worker is attached"
                raise QcError(f"The mixing check needs Clusters and the UMAP, which can't run here: {why}", 503)
            return                                   # nothing can be scored: no switch to hand over
        with state.lock:
            enabled = bool(state.enabled)
            if seq is None and state.synced == enabled:
                return
        try:
            reply = analysis.configure_qc(run_id, enabled=enabled, seq=seq)
        except Exception as exc:
            if seq is not None:
                raise
            print(f"qc: the analysis scheduler did not take the switch of {run_id}: {exc}", flush=True)
            return None
        with state.lock:
            state.synced = enabled
        return reply if isinstance(reply, dict) else None

    # ---- per-run state ----------------------------------------------------------------------------------------------
    def _run(self, run_id: str):
        """The run (RunError 404 for an unknown one); a deleted run's state is forgotten."""
        try:
            return self.runs.get(run_id)
        except Exception:
            with self.lock:
                self.states.pop(run_id, None)
            raise

    def _state(self, run_id: str) -> RunQc | None:
        """The run's state, loaded once per session (switch, model, results, smoothing); None for a deleted run. The
        switch: its settings.json (stored_switch: a file from before the check was opt-in follows today's default
        unless someone changed it, and is brought up to date once), else the remembered choice for new runs."""
        try:
            run = self._run(run_id)
        except Exception:
            return None
        if run.meta.get("kind") == "stain":
            return None
        with self.lock:
            state = self.states.get(run_id)
            if state is None:
                state = self.states[run_id] = RunQc(run_id)
        with state.lock:
            if state.enabled is None:
                try:
                    settings = json.loads((run.dir / "qc" / "settings.json").read_text())
                    state.enabled, state.switch_n, current = stored_switch(settings, self.s.enabled_by_default)
                    if not current:
                        self._save_settings(run, state)      # written once with its version: read as it is from now on
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    state.enabled = bool(self.prefs.get("enabled"))
                    self._save_settings(run, state)
            self._attach(run, state)
        if run.meta.get("sample_kind") == "zarr":       # other runs have no cell table to be checked against
            self._sync(run_id, state)
        return state

    def _save_settings(self, run, state: RunQc):
        if not run.dir.exists():                     # a deleted run stays deleted
            return
        try:
            (run.dir / "qc").mkdir(exist_ok=True)
            _atomic(run.dir / "qc" / "settings.json",
                    json.dumps({"v": SETTINGS_VERSION, "enabled": state.enabled, "n": state.switch_n}).encode())
        except OSError:
            pass

    def _model_for(self, run) -> dict | None:
        if run.meta.get("sample_kind") != "zarr":
            return None
        self.models.adopt(run.meta["sample"], run.meta.get("sample_path"), run.meta.get("sample_name"))
        return self.models.model(run.meta["sample"])

    def _attach(self, run, state: RunQc):
        """Point the run at its sample's ready model and load what earlier sessions left (once per model): its results
        on proseg's own count, in the model's folder for that count (qc/<model key[:8]>.<BASIS>)."""
        model = self._model_for(run)
        if model is None or model["id"] == state.key:
            return
        state.key, state.model_dir = model["id"], Path(model["dir"])
        state.dir = run.dir / "qc" / f"{model['id'][:8]}.{BASIS}"
        state.start, state.results, state.points, state.ema, state.shown = None, {}, {}, None, None
        state.ema_matched = state.ema_start = None
        state.provenance = (model.get("summary") or {}).get("provenance")
        state.trial, state.batch, state.live_count, state.errors, state.last_t = None, None, 0, [], None
        state.last_live = None
        state.tracker, state.start_bins, state.start_areas, state.shapes, state.displayed = None, None, None, {}, None
        state.shapes_pending, state.state_saved, state.state_dirty, state.reveal_iter = [], -math.inf, False, None
        state.start_drawn = None
        state.area_order, state.drawn_log, state.drawn_pending = [], [], []
        state.shapes_missing, state.tails = set(), set()
        state.recent, state.prev_live, state.watch, state.trial_watch, state.trial_view = [], None, None, None, None
        state.tried, state.trial_closed = [], None
        state.keeper.reset()
        state.steady.reset()
        saved_areas = saved_trial = None
        try:
            saved = json.loads((state.dir / "state.json").read_text())
            if saved.get("model") == state.key:
                state.ema, state.shown, state.last_live = saved.get("ema"), saved.get("shown"), saved.get("last_live")
                matched = saved.get("ema_matched")
                state.ema_matched = [float(v) for v in matched] if isinstance(matched, list) and len(matched) == 3 \
                    and all(qr.number(v) is not None for v in matched) else None
                begun = saved.get("ema_start")
                state.ema_start = [float(v) for v in begun] if isinstance(begun, list) and len(begun) == 2 \
                    and all(qr.number(v) is not None for v in begun) else None
                state.live_count = int(saved.get("live_count") or 0)
                state.last_t = _seconds(saved.get("last_t"))
                saved_areas = saved.get("areas")
                saved_trial = saved.get("trial") if isinstance(saved.get("trial"), dict) else None
                order = saved.get("area_order")
                if isinstance(order, list):
                    state.area_order = [str(L) for L in order if isinstance(L, str)]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        for line in self._read_lines(state, "series.jsonl"):   # every result's point; the newest RESULTS_KEPT as stored
            try:
                item = json.loads(line)
                _keep(state, int(item["seq"]), line, _point(item))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        self._load_shapes(state)
        for line in self._read_lines(state, "drawn.jsonl"):    # what the page drew, from each live result on
            try:
                item = json.loads(line)
                state.drawn_log.append((int(item["seq"]), [(str(L), frozenset(int(b) for b in bins))
                                                           for L, bins in item["areas"]], bool(item.get("start"))))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        state.drawn_log.sort(key=lambda entry: entry[0])
        del state.drawn_log[:-DRAWN_KEPT]
        if state.drawn_log:                          # what the page drew last (a tracker read back below says it too)
            shown = state.drawn_log[-1][1]
            state.displayed = ({L for L, _ in shown}, set().union(*(set(b) for _, b in shown)) if shown else set())
        try:
            stored = json.loads((state.dir / "start.json").read_text())
            state.start = self._start_summary(stored)
            if isinstance(stored.get("areas"), list):
                state.start_areas = {k: stored.get(k) for k in ("areas", "grid")}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        if self._start_line(run):
            state.start_bins = self._read_bins(state.dir / "start.bins.npz")
        if state.start_bins is not None and state.start_areas is None:   # a start line from before the areas
            state.start_areas = self._start_area_fields(run, state)
        self._load_tracker(run, state, saved_areas)
        for line in self._read_lines(state, "trials.jsonl"):
            try:
                item = json.loads(line)
                state.trials = max(state.trials, int(item.get("id") or 0))
                if item.get("event") == "start":
                    state.trial = {k: v for k, v in item.items() if k != "event"}
                elif item.get("event") == "end":
                    self._remember_tried(state, item)
                    if state.trial and state.trial.get("id") == item.get("id"):
                        state.trial = None
            except (ValueError, TypeError, AttributeError):
                continue
        if state.trial is not None and saved_trial is not None and saved_trial.get("id") == state.trial.get("id"):
            state.trial_view = saved_trial               # the readout as the last session showed it
        self._close_ended(run, state)                    # a run the last session ended (or a restart stopped)
        self._bump()

    def _start_source(self, run) -> str:
        """Where the run's cells started: "xenium" (Xenium's cell boundaries or cell_id), "other" (another
        segmentation) or "nuclei" (nuclei only: no start line)."""
        prior = run.meta.get("prior")
        if run.meta.get("sample_kind") != "zarr" or prior is None:
            return "xenium"
        cells = prior.get("cells")
        if not cells:
            return "nuclei"
        if cells.get("kind") == "shapes" and "xenium" in str(cells.get("key", "")).lower():
            return "xenium"
        if cells.get("kind") == "column" and cells.get("column") == "cell_id":
            return "xenium"
        return "other"

    @staticmethod
    def _start_line(run) -> bool:
        """Whether the run has a start line (the start shown, the arrow, retention): only a start from cells alone
        (qc_rules.start_of "cells"). A start with nuclei (nuclei only, or cells with their nuclei: describe's
        start_kind "nuclei" or "paired") has none: proseg's count at iteration 0 holds only the nuclei there, so there
        is no starting number to compare with."""
        return qr.start_of(run.meta) == "cells"

    def _start_shown(self, run, state: RunQc) -> dict | None:
        """The start line the page shows (proseg's count at iteration 0): none for a run whose start has nuclei."""
        return self._start_public(state.start) if self._start_line(run) else None

    def _start_full(self, run, state: RunQc) -> dict | None:
        """The start line with its own areas ({per100, judged, mixed, median_tx, areas (with outlines), grid}): the
        areas the cards and "Look at its cells" read while paused on the prior and until the run's own areas come.
        For describe, the series and the start event; a frame's summary carries only the start line's numbers."""
        start = self._start_shown(run, state)
        return None if start is None else {**start, **self._start_detail(state.start, self._mode(run)),
                                           **self._start_area_public(state)}

    # ---- describe / configure ----------------------------------------------------------------------------------------
    def describe(self, run_id: str) -> dict:
        run = self._run(run_id)
        state = self._state(run_id)
        sample = None
        if run.meta.get("sample_kind") == "zarr":
            try:
                sample = self.models.describe(run.meta["sample"])
                if state.enabled and sample["state"] == "off":     # the check is on: its sample gets learned
                    sample = self.models.ensure(run.meta["sample"])
            except QcError:
                sample = None
        if sample is None:
            sample = {"state": "unavailable", "reason": "no_table", "progress": None, "eta_s": None, "model": None,
                      "relearn": False, "estimate_s": None}
        scheduler = {}
        if self.analysis is not None and hasattr(self.analysis, "describe_qc"):
            try:
                scheduler = self.analysis.describe_qc(run_id) or {}
            except Exception:
                scheduler = {}
        source = self._start_source(run)
        no_start = not self._start_line(run) or scheduler.get("start") in ("none", "error")
        with state.lock:
            self._close_ended(run, state)
            if not state.enabled:
                status, reason = "off", None
            elif sample["state"] != "ready":
                status, reason = ("waiting", "queue") if sample["state"] == "off" else (sample["state"], sample["reason"])
            elif scheduler.get("unavailable"):           # e.g. a run made before live analysis: no transcripts
                status, reason = "unavailable", scheduler["unavailable"]
            elif state.start is None and not no_start:
                status, reason = "starting", None
            else:
                status, reason = "ready", None
            seqs = sorted(state.points)
            batch = self._batch(run, state) or {}
            # sample_reason: why the sample's check failed or cannot run, also while the run's switch is off (the off
            # panel says it, not only "turn it on"). start_kind: how the run's cells started (qc_rules.start_of:
            # "cells" has a start line; "paired", cells with their nuclei, and "nuclei" have none).
            return {"enabled": bool(state.enabled), "switch_n": state.switch_n, "state": status, "reason": reason,
                    "sample_state": sample["state"], "sample_reason": sample.get("reason"),
                    "progress": sample.get("progress"), "eta_s": sample.get("eta_s"),
                    "relearn": sample.get("relearn", False), "model": sample.get("model"),
                    "detail": sample.get("detail") or {"state": "none", "progress": None, "eta_s": None, "reason": None},
                    "start": self._start_full(run, state), "start_source": source,
                    "start_kind": qr.start_of(run.meta),
                    "batch_per100": batch.get("per100"),
                    "batch_table": batch.get("table") if batch.get("per100") is not None else None,
                    "latest_seq": seqs[-1] if seqs else None, "seqs": seqs[-2000:],
                    "thresholds": {"mixed_at": self.s.mixed_at, "judge_min": self.s.judge_min},
                    "scoring": scheduler.get("status"), "backfill": scheduler.get("backfill"),
                    "requested": scheduler.get("requested"), "trial": self._trial_shown(run, state)}

    def configure(self, run_id: str, enabled=None, seq=None) -> dict:
        """The run's switch (remembered as the choice for new runs) and/or one frame to score now (as for analysis:
        the scheduler does it once the model is ready). A refused request changes nothing."""
        run = self._run(run_id)
        if enabled is not None and not isinstance(enabled, bool):
            raise QcError("enabled must be true or false")
        if seq is not None:
            try:
                seq = int(seq)
            except (TypeError, ValueError):
                raise QcError("seq must be a frame number") from None
            with run.lock:
                if not 0 <= seq < len(run.frames):
                    raise QcError("Unknown frame", 404)
        state = self._state(run_id)
        if enabled is not None:
            with state.lock:
                if enabled != state.enabled:
                    state.switch_n += 1           # pages keep the newest switch they heard of (events: "switch")
                state.enabled = enabled
                self._save_settings(run, state)
            if enabled != self.prefs.get("enabled"):
                self.prefs = {"enabled": enabled}         # remembered: new runs and opened samples follow it
                try:
                    self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic(self.prefs_path, json.dumps(self.prefs).encode())
                except OSError:
                    pass
            if run.meta.get("sample_kind") == "zarr":
                if enabled:
                    self.models.ensure(run.meta["sample"])
                self._sync(run_id, state)
            self._bump()
        sent = self._sync(run_id, state, seq=seq) if seq is not None and run.meta.get("sample_kind") == "zarr" else None
        described = self.describe(run_id)
        # The frame the asked one snapped to, as the scheduler said it: its thread, just woken, may have taken the
        # request before describe looked (DESIGN 2.2: the reply carries its seq).
        if isinstance(sent, dict) and sent.get("requested") is not None:
            described["requested"] = sent["requested"]
        return described

    # ---- results ------------------------------------------------------------------------------------------------------
    def ingest(self, run_id: str, seq: int, result: dict, single: bool | None = None, frame: dict | None = None,
               bins: dict | None = None, count: str | None = None):
        """The worker's result for frame seq (its summary: judged, mixed, median_tx, ...) and its per-bin sums (`bins`,
        the worker's bins file as {name: array}): the headline (live results smoothed, the integer with hysteresis;
        `single` ones judged alone), the comparison with the frame's own start block (`_change`, `_retention`) and the
        areas (`_frame_areas`), written as <seq>.qc.grid, <seq>.qc.json and a series.jsonl line (state.json:
        `_save_state`). Live results also get the suggestions, the running trial's readout (`_suggest`). Every summary
        also carries the frame's own numbers, `own` {per100 (its share, unrounded to 4 decimals), shown (its whole
        number, rounded once from that share, no margin), change (its own, neither smoothed nor held)}, which the page
        shows where the run rests; every whole number is rounded once, from the unrounded share (never from per100's
        one decimal: 8.46 reads 8). Returns the summary the page gets.
        `count`: the count the engine asked the worker to read (the record's: "averaged" | "sampled"; None: the
        result's own, else "averaged"). A frame read on the count proseg saves ("sampled", f: one counted for a pause,
        the hold, a finish or the end) shows its own numbers on f, `single` and its own change released (decision A);
        its averaged count (result["averaged"], e) is what the live path takes, exactly as an averaged live result of
        the same bins would: the smoothing, the arrow's held word, the areas and the cards. A re-count that is not live
        (a pause, the hold, a finish or the end counting again a frame a token counted; a pause's since proseg-live
        3.2.0.2) keeps the stored summary's areas, grid and cards, and its <seq>.qc.grid: they read e, which a re-count
        does not change."""
        state = self._state(run_id)
        if state is None or state.key is None or not isinstance(result, dict):
            return None
        run = self._run(run_id)
        seq = int(seq)
        if count not in COUNTS:
            count = result.get("count") if result.get("count") in COUNTS else "averaged"
        sampled = count == "sampled"
        averaged = result.get("averaged") if sampled and isinstance(result.get("averaged"), dict) else None
        live_result = {**result, **averaged} if averaged is not None else result   # what the live path reads: e
        with state.lock:
            key = state.key
            mode = self._mode(run)
            judged, mixed = int(result.get("judged") or 0), int(result.get("mixed") or 0)
            e_judged, e_mixed = int(live_result.get("judged") or 0), int(live_result.get("mixed") or 0)
            if frame is None:
                with run.lock:
                    frame = dict(run.frames[seq]) if 0 <= seq < len(run.frames) else {}
            # A frame newer than the last live result is live: always for the scheduler's live passes, and for one asked
            # for (Pause or Step shows the newest frame before a live pass scored it) when it is the run's newest frame
            # and continues a live history: then the headline and the areas go on smoothly, with the same letters. A
            # frame read on f without its averaged count is never live (nothing to smooth with).
            newer = state.last_live is None or seq > state.last_live
            live = newer and (not single or (state.last_live is not None and self._newest(run, seq))) \
                and (not sampled or averaged is not None)
            raw = _per100(mixed, judged)
            share = _share(mixed, judged)             # the frame's own share, unrounded: its whole numbers come from it
            t = _seconds(frame.get("t"))
            matched = _matched(result.get("matched"))
            frame_counts = _matched_counts(matched)
            block = _start_block(result.get("start_block")) if self._start_line(run) else None
            start_counts = None if block is None else [float(block["judged"]), float(block["mixed"])]
            own_start = 100.0 * start_counts[1] / start_counts[0] if start_counts and start_counts[0] > 0 else None
            margin = qk.value(mode, "shown_margin")
            if live:
                weight = None
                if state.ema is None:
                    state.ema = [float(e_judged), float(e_mixed)]
                else:
                    weight = headline_weight(t, state.last_t, mode)
                    state.ema = [(1 - weight) * state.ema[0] + weight * e_judged,
                                 (1 - weight) * state.ema[1] + weight * e_mixed]
                if frame_counts is not None:              # the depth-matched counts, smoothed as the headline
                    state.ema_matched = _smoothed(state.ema_matched, frame_counts, weight)
                if start_counts is not None:              # the frame's own start, smoothed as the headline
                    state.ema_start = _smoothed(state.ema_start, start_counts, weight)
                per100 = _per100(state.ema[1], state.ema[0])
                exact_now = _share(state.ema[1], state.ema[0])
                state.shown = shown_integer(exact_now, state.shown, margin)
                state.last_live, state.last_t = seq, t
                state.live_count += 1
                shown = state.shown
            else:
                # Judged alone; its whole number keeps the margin against what the nearest earlier live result showed.
                per100 = raw
                shown = shown_integer(share, self._earlier_shown(state, seq), margin)
                exact_now = 100.0 * e_mixed / e_judged if e_judged else None
            counts = None if frame_counts is None else state.ema_matched if live else frame_counts
            begun = None if start_counts is None else state.ema_start if live else start_counts
            start_now = 100.0 * begun[1] / begun[0] if begun and begun[0] > 0 else None
            change = self._change(run, state, frame, exact_now, counts, e_judged, start_now)
            steady_at = t if t is not None else self.clock()     # live verdicts are held steady on the run clock
            if live:
                change = state.steady.apply("change", change, steady_at)
            # The frame's own change: its own like-for-like counts and its own start block, on e (the comparison with
            # the start is like for like whatever count the frame shows), neither smoothed nor held.
            own_change = change if not live else self._change(
                run, state, frame, 100.0 * e_mixed / e_judged if e_judged else None, frame_counts, e_judged, own_start)
            live_change = change                      # the live path's (the cards'): on e, smoothed and held
            if sampled:                               # the count proseg saves: the frame's own numbers, its own word
                per100, shown, change = raw, _whole(share), own_change
                if self._newest(run, seq):           # the word held before it is let go
                    state.steady.release("change", own_change)
            started = time.perf_counter()
            stored = self._stored(state, seq) if sampled and not live else None
            if stored is not None and stored.get("count", "averaged") != "averaged":
                stored = None
            if stored is not None:                    # a re-count: the areas, grid and cards it had (on e) stay
                areas = {"areas": stored.get("areas") or [], "area_view": stored.get("area_view") or "none",
                         "grid": stored.get("grid")}
                if stored.get("areas_carried"):
                    areas["areas_carried"] = True
                grid_bytes, save_now, drawn, grid, sm = None, False, [], None, None
            else:
                areas, grid_bytes, save_now, drawn, grid, sm = self._frame_areas(run, state, frame, bins, live, t, seq)
            if live:                                  # each drawn area's own verdict too (letters drawn now only)
                letters = set()
                for item in areas.get("areas") or []:
                    if item.get("L") and isinstance(item.get("local"), dict):
                        item["local"] = state.steady.apply(f"area:{item['L']}", item["local"], steady_at)
                        letters.add(item["L"])
                state.steady.keep("area:", letters)
            ms = {**(result.get("ms") or {}), "areas": round(1000 * (time.perf_counter() - started), 2)} \
                if areas["area_view"] != "none" else dict(result.get("ms") or {})
            summary = {"seq": seq, "iter": frame.get("iter", result.get("iter")), "phase": frame.get("phase"),
                       "model": state.key, "single": sampled or not live, "judged": judged, "mixed": mixed,
                       "raw_per100": raw, "per100": per100, "shown": shown, "start": self._start_shown(run, state),
                       "median_tx": result.get("median_tx"), "batch_per100": (self._batch(run, state) or {}).get("per100"),
                       "coverage": _coverage(result, mode), "retention": self._retention(run, live_result),
                       "matched": matched, "change": change, **_basis(result), "count": count,
                       "constants": qk.block(mode), "provenance": self._provenance(state),
                       "own": {"per100": qr.exact(share), "shown": _whole(share), "change": own_change},
                       **areas, "suggestion": None, "trial": None, "ms": ms, "gpu": _gpu(result.get("gpu"))}
            if stored is not None:
                summary.update({k: stored[k] for k in ("suggestion", "more_ideas", "trial") if k in stored})
            elif live:                                # a scrubbed or backfilled frame gets none: it is not "now"
                summary.update(self._suggest(run, state, seq, frame, live_result, areas, drawn, grid, sm, live_change,
                                             start_now))
        # The cell groups that meet in the drawn areas read the frame's own files (its UMAP result, its cell statistics):
        # outside the run's lock, so page requests and pushes never wait on that disk.
        if drawn and grid is not None:
            started = time.perf_counter()
            groups = self._area_clusters(run, seq, grid, drawn)
            if groups:
                for item in summary["areas"]:
                    if item.get("L") in groups:
                        item["clusters"] = groups[item["L"]]
                summary["ms"]["clusters"] = round(1000 * (time.perf_counter() - started), 2)
        with state.lock:
            if state.key != key:                      # a relearned model meanwhile: another yardstick
                return None
            if summary.get("more_ideas"):
                self._fit(state, summary)
            line = json.dumps(summary, separators=(",", ":"))
            _keep(state, seq, line, _point(summary))
            if run.dir.exists():
                try:
                    state.dir.mkdir(parents=True, exist_ok=True)
                    self._write_shapes(state)         # before any summary names them
                    self._write_drawn(state)
                    if grid_bytes is not None:
                        _atomic(state.dir / f"{seq:06d}.qc.grid", grid_bytes)
                    _atomic(state.dir / f"{seq:06d}.qc.json", line.encode())
                    self._append_lines(state, "series.jsonl", line + "\n")
                    if live:                          # an engine restart resumes the smoothing
                        state.state_dirty = True
                        if save_now or self.clock() - state.state_saved >= STATE_SAVE_S:
                            self._save_state(run, state)
                except OSError:
                    pass
        self._bump()
        return self._expand(state, summary)

    @staticmethod
    def _newest(run, seq: int) -> bool:
        """Whether frame `seq` is the run's newest frame."""
        with run.lock:
            if not run.frames:
                return False
            last = run.frames[-1]
            return seq >= int(last.get("seq", len(run.frames) - 1))

    @staticmethod
    def _earlier_shown(state: RunQc, seq: int) -> int | None:
        """The whole number the nearest live result before frame `seq` showed (None: none). State lock held."""
        best = None
        for s, point in state.points.items():
            if s < seq and not point.get("single") and (best is None or s > best):
                best = s
        shown = None if best is None else state.points[best].get("shown")
        return shown if isinstance(shown, int) and not isinstance(shown, bool) else None

    def _change(self, run, state: RunQc, frame: dict, now, counts, judged: int, start_now=None) -> dict:
        """The summary's comparison with the start (qc_rules.change_block): the raw change (the unrounded share,
        smoothed for live results, minus `start_now`, the frame's own start block's share, smoothed the same way: never
        the start line shown, which is proseg's count at iteration 0) and the depth-matched change (`counts`: [paired,
        mixed now, mixed at the start], the worker's like for like on the frame's start block, smoothed for live
        results) with its threshold for the run's mode at its own share and paired cells, and the two numbers it
        compares (per100_now, per100_start: mixed now and at the start per 100 paired, from the same counts). No
        verdict while cells form, nor for a run whose start has nuclei (no start line: nothing is compared with it).
        State lock held."""
        line = self._start_line(run)
        raw = None if not line or now is None or start_now is None else now - start_now
        matched = share = paired = per100_now = per100_start = None
        if line and counts and counts[0] > 0:
            paired = counts[0]
            matched, share = 100.0 * (counts[1] - counts[2]) / paired, counts[1] / paired
            per100_now, per100_start = 100.0 * counts[1] / paired, 100.0 * counts[2] / paired
        return qr.change_block(raw, matched, self._mode(run), share, paired, judged,
                               forming=line and self._early(run, state, frame), now=per100_now, start=per100_start)

    def _retention(self, run, result: dict) -> dict | None:
        """Whether cells lost RNA: {ratio (proseg's count of the model's genes in all of the run's cells now / in the
        frame's own start block: the cells proseg started with, their RNA kept the same way), start_tx, tx}; null
        without a start block to compare with (a run whose start has nuclei, a frame without one, or a worker that does
        not count them)."""
        block = _start_block(result.get("start_block")) if self._start_line(run) else None
        tx = qr.number(result.get("tx_sum", result.get("transcripts")))
        start_tx = None if block is None else block["tx_sum"]
        if tx is None or not start_tx or start_tx < 0:
            return None
        return {"ratio": qr.exact(tx / start_tx), "start_tx": int(round(start_tx)), "tx": int(round(tx))}

    def _provenance(self, state: RunQc) -> dict | None:
        """The model's provenance as its model.json says it now (a detail job may have added it), without its empty
        entries (every summary carries it). State lock held."""
        summary = self.models._summary(state.model_dir) if state.model_dir is not None else None
        if summary and summary.get("provenance") is not None:
            state.provenance = summary["provenance"]
        found = state.provenance
        return None if not isinstance(found, dict) else {k: v for k, v in found.items() if v is not None}

    def ingest_start(self, run_id: str, result: dict, bins: dict | None = None) -> dict | None:
        """The worker's qc_start result (proseg's count at iteration 0, scored once: judged, mixed, median_tx, ...;
        the start line shown, which never moves) and its per-bin sums (`bins`: the cells at their starting centroids),
        written as start.json with the start line's own areas: where the cells proseg starts from already look mixed
        (none for a run whose start has nuclei). Frames pair with their own start blocks, never with these."""
        state = self._state(run_id)
        if state is None or state.key is None or not isinstance(result, dict):
            return None
        run = self._run(run_id)
        with state.lock:
            state.start = self._start_summary(result)
            state.start_bins = None
            grid = qa.grid_of(bins) if isinstance(bins, dict) and self._start_line(run) else None
            if grid is not None:
                table = qa.dense(bins, grid)
                if "J" in table and "M" in table:
                    state.start_bins = {"grid": grid, "bins": table}
            state.start_areas = self._start_area_fields(run, state)
            if run.dir.exists():
                try:
                    state.dir.mkdir(parents=True, exist_ok=True)
                    self._write_shapes(state)
                    _atomic(state.dir / "start.json", json.dumps(
                        {**state.start, "model": state.key, **(state.start_areas or {})}).encode())
                except OSError:
                    pass
        self._bump()
        return self._start_public(state.start)

    # ---- the run's mode (engine/qc_constants.py) and areas (engine/qc_areas.py) -------------------------------------
    @staticmethod
    def _tiled(run) -> bool:
        """Whether the run is tiled (whole samples and tiled regions), by the rule AnalysisManager.qc_grid sizes its
        bins with (100 µm, else 50 µm)."""
        info = run.info or {}
        return run.meta.get("mode") == "tiled" or (info.get("prior") or {}).get("format") == "tiles"

    def _mode(self, run) -> str:
        """The run's mode, whose constants it is judged with (qc_constants.run_mode: the rule the settings test and the
        page read too): "whole" (20,000 cells or more), "tiled" or "region"."""
        return qk.run_mode(run.meta, run.info)

    def _area_kind(self, run) -> str:
        """The kind of the run's area grid: "tiled" (100 µm bins) or "detail" (50 µm), as state.json keeps it."""
        return "tiled" if self._tiled(run) else "detail"

    def _area_params(self, run) -> dict:
        """The area constants of the run's mode (qc_constants "areas"), the mode itself (an area's own minimum support
        and thresholds) and whether its grid's border is the tissue's edge ("edge_border": only for a run of the whole
        sample; a region's border is a cut through tissue). A dense run of a whole sample's size (a max_region_um
        raised past what dense regions are measured on) keeps the dense constants: its bins are 50 µm."""
        region = run.meta.get("region") if isinstance(run.meta.get("region"), dict) else {}
        mode = self._mode(run)
        areas = qk.value(mode if self._tiled(run) else "region", "areas")
        return {**areas, "mode": mode, "edge_border": region.get("kind") == "full"}

    def _reveal_iter(self, run, state: RunQc) -> int | None:
        """The iteration from which areas are shown: burn-in + 20 (the schedule's burn-in, else the first frame past
        it); None while that is unknown. State lock held."""
        if state.reveal_iter is not None:
            return state.reveal_iter
        burnin = ((run.info or {}).get("schedule") or {}).get("burnin")
        if isinstance(burnin, (int, float)) and not isinstance(burnin, bool) and burnin >= 0:
            state.reveal_iter = int(burnin) + qa.FROM_ITERS
            return state.reveal_iter
        with run.lock:
            for frame in run.frames:
                if frame.get("phase") in FORMED_PHASES and isinstance(frame.get("iter"), (int, float)):
                    state.reveal_iter = int(frame["iter"]) + qa.FROM_ITERS
                    break
        return state.reveal_iter

    def _early(self, run, state: RunQc, frame: dict) -> bool:
        """A frame while cells are still forming (before burn-in + 20 iterations): no areas of its own."""
        if frame.get("phase") in FORMING_PHASES:
            return True
        reveal, iteration = self._reveal_iter(run, state), frame.get("iter")
        return reveal is not None and isinstance(iteration, (int, float)) and not isinstance(iteration, bool) \
            and iteration < reveal

    def _tracker(self, run, state: RunQc, grid: dict) -> qa.Tracker:
        """The run's tracker on `grid`. Its areas pair with each result's own start sums (the frame's start block per
        bin), never with the start line's bins (those make only the start line's own areas)."""
        params = self._area_params(run)
        tracker = state.tracker
        if tracker is None or tracker.grid != grid or tracker.p != params:
            if tracker is not None:                  # another grid: what was drawn on the old one is not comparable
                state.displayed, state.drawn_log, state.area_order = None, [], []
            tracker = state.tracker = qa.Tracker(grid, params)
        return tracker

    def _frame_areas(self, run, state: RunQc, frame: dict, bins: dict | None, live: bool, t: float | None,
                     seq: int | None = None):
        """A frame's area fields for its summary (each area names the shape of its outline; `_expand` adds the outline),
        its overview grid (zstd, for <seq>.qc.grid), whether state.json should be saved now, the drawn areas and their
        grid (for `_area_clusters`) and the bin sums used (for `_suggest`). Live results go through the run's tracker
        (smoothed, hysteresis, stable letters: area_view "smoothed"); other results (a frame asked for, backfill) are judged alone ("single"), their areas
        named by the letters the page showed there (`_letter_single`); a frame before burn-in + 20 iterations has none
        of its own ("start": the page shows the start line's, `_expand` serves them). A tracker that starts after the
        areas are shown (a check switched on late, a model that came late, a lost state.json) decides nothing before
        the enter rule's 2 results over 2 s: until then the areas the page showed last stay, with this result's numbers
        ("areas_carried"), else none ("pending"); a live result without per-bin sums keeps the areas on screen the same
        way; a single one has none ("none"). Live results list the areas in a steady order (`_ordered`). The page
        draws no area (the cards, trials and "Look at its cells" read them): what only an area list and the stage
        would read is not made (each area's whole number, the spots left out, how many were listed, what appeared or
        cleared since the previous result). A change of the drawn set still saves
        state.json at once, so its letters resume after a restart. State lock held."""
        params = self._area_params(run)
        fields = {"areas": [], "area_view": "none", "grid": None}
        grid = qa.grid_of(bins) if isinstance(bins, dict) else None
        table = qa.dense(bins, grid) if grid is not None else {}
        if not self._start_line(run):                # nothing is compared with a start it does not have
            table = {k: v for k, v in table.items() if k not in START_SUMS}
        early = self._early(run, state, frame)
        tracker = state.tracker
        save_now, carried = False, False
        if "J" not in table or "M" not in table:
            if not (live and not early and tracker is not None and tracker.revealed and tracker.sm is not None):
                return fields, None, False, [], None, None
            grid, sm, found = tracker.grid, tracker.sm, []
            drawn = tracker.carried([(a.letter, a.bins) for a in tracker.areas], sm)
            view, carried = "smoothed", True
        elif live:
            tracker = self._tracker(run, state, grid)
            clock = t if t is not None else (0.0 if tracker.t_prev is None else tracker.t_prev + 1.0)
            found, _, sm = tracker.update(table, clock, live=True)
            if early:
                drawn, view = [], "start"
            elif not tracker.revealed and not tracker.warm:
                previous = self._last_drawn(state, start=False)
                drawn = tracker.carried(previous, sm) if previous else []
                view, carried = ("smoothed", True) if drawn else ("pending", False)
            else:
                if not tracker.revealed:             # the letters the page showed go on where those areas still are
                    tracker.reveal(self._previous_drawn(run, state, grid))
                    save_now = True
                drawn, view = list(tracker.areas), "smoothed"
        elif early:
            found, drawn, sm, view = [], [], table, "start"
        else:
            found, _, sm = qa.Tracker(grid, params).snapshot(table)
            drawn, view = _listed([a for a in found if a.letter]), "single"
            self._letter_single(run, state, grid, seq, drawn)
        if live and drawn:
            drawn = self._ordered(state, drawn)
        totals = qa.totals(sm)
        items = [self._area_item(run, state, area, grid, sm, totals) for area in drawn]
        fields.update(areas=items, area_view=view, grid={**grid, "planes": qa.OVERVIEW_PLANES})
        if carried:                                  # _suggest gives a carried area no new card
            fields["areas_carried"] = True
        if live:
            shown = self._start_drawn(run, state, grid) if early else [(area.letter, area.bins) for area in drawn]
            letters = {letter for letter, _ in shown}
            cells = set().union(*(set(np.asarray(b).tolist()) for _, b in shown)) if shown else set()
            save_now = save_now or (state.displayed is not None and state.displayed != (letters, cells))
            last = state.drawn_log[-1] if state.drawn_log else None
            if state.displayed is None or state.displayed != (letters, cells) or last is None or last[2] != early:
                self._log_drawn(state, seq, shown, early)
            state.displayed = (letters, cells)
        import zstandard
        packed = zstandard.ZstdCompressor(level=3).compress(
            qa.overview_grid(sm, (grid["ny"], grid["nx"]), int(params["window"])).tobytes())
        return fields, packed, save_now, drawn, grid, sm

    def _ordered(self, state: RunQc, drawn: list) -> list:
        """The drawn areas of a live result in the order the page lists them: the previous order, a new area at its
        rank by extra mixed cells, and an area moving above the one before it only once it has ORDER_SWAP x that one's
        extra mixed cells (one pass from the bottom: a clearly larger area can climb several places at once). Areas at
        the tissue's edge that did not get more mixed in this run come after the others (windows at the
        edge look more mixed by themselves: in a test, half again as mixed as inside, at the start), so the rows listed
        first are the ones most worth a look. State lock held."""
        by = {area.letter: area for area in drawn}
        order = [L for L in state.area_order if L in by]
        for area in sorted(drawn, key=lambda a: -a.extra):
            if area.letter in order:
                continue
            at = next((i for i, L in enumerate(order) if by[L].extra < area.extra), len(order))
            order.insert(at, area.letter)
        for i in range(len(order) - 1, 0, -1):
            if by[order[i]].extra > ORDER_SWAP * max(by[order[i - 1]].extra, 1e-9):
                order[i - 1], order[i] = order[i], order[i - 1]
        order.sort(key=lambda L: _edge_last(by[L]))                                            # stable
        state.area_order = order
        return [by[L] for L in order]

    def _log_drawn(self, state: RunQc, seq, shown: list, start: bool):
        """What the page draws from live result `seq` on ([(letter, bins)]; `start`: the start line's own areas, while
        cells form), kept for the letters of single snapshots (drawn.jsonl, written with the summary). State lock
        held."""
        if seq is None:
            return
        areas = [(str(letter), frozenset(np.asarray(bins, np.int64).tolist())) for letter, bins in shown]
        state.drawn_log.append((int(seq), areas, bool(start)))
        del state.drawn_log[:-DRAWN_KEPT]
        state.drawn_pending.append(json.dumps({"seq": int(seq), "start": bool(start),
                                               "areas": [[L, sorted(b)] for L, b in areas]}, separators=(",", ":")))

    def _write_drawn(self, state: RunQc):
        """drawn.jsonl lines not written yet (the run folder exists). State lock held."""
        if state.drawn_pending:
            self._append_lines(state, "drawn.jsonl", "".join(line + "\n" for line in state.drawn_pending))
            state.drawn_pending = []

    @staticmethod
    def _last_drawn(state: RunQc, start: bool | None = None) -> list:
        """[(letter, bins)] the page drew from the newest live result logged (`start` False: only the run's own areas,
        never the start line's), or []. State lock held."""
        if not state.drawn_log:
            return []
        entry = state.drawn_log[-1]
        return [] if start is False and len(entry) > 2 and entry[2] else list(entry[1])

    def _previous_drawn(self, run, state: RunQc, grid: dict) -> list:
        """The areas the page showed until the run's own are first shown (Tracker.reveal keeps their letters where they
        overlap): the newest logged ones, else the start line's. State lock held."""
        return self._last_drawn(state) if state.drawn_log else self._start_drawn(run, state, grid)

    def _letter_single(self, run, state: RunQc, grid: dict, seq, drawn: list):
        """A single snapshot's areas take the letters the page showed for the same places at that frame (the newest
        drawn set logged at or before `seq`, else the start line's; Jaccard of bins >= letter_jaccard, best pairs
        first, one to one), so stepping or scrubbing never renames a place; the others take the first letters that
        name nothing there. State lock held."""
        entry = None
        if seq is not None and state.drawn_log:
            i = bisect.bisect_right([e[0] for e in state.drawn_log], int(seq)) - 1
            entry = state.drawn_log[i] if i >= 0 else None
        reference = list(entry[1]) if entry is not None else self._start_drawn(run, state, grid)
        threshold = self._area_params(run)["letter_jaccard"]
        theirs = [(str(L), set(np.asarray(list(b), np.int64).tolist())) for L, b in reference]
        pairs = []
        for i, area in enumerate(drawn):
            mine = set(area.bins.tolist())
            for j, (_, other) in enumerate(theirs):
                jac = len(mine & other) / max(1, len(mine | other))
                if jac >= threshold:
                    pairs.append((jac, -i, -j))
        pairs.sort(reverse=True)
        kept, taken = {}, set()
        for _, i, j in pairs:
            if -i not in kept and -j not in taken:
                kept[-i] = -j
                taken.add(-j)
        busy = {L for L, _ in theirs}
        for i, area in enumerate(drawn):
            area.letter = theirs[kept[i]][0] if i in kept else ""
        for area in drawn:
            if not area.letter:
                area.letter = next((L for L in qa.LETTERS if L not in busy), "?")
            busy.add(area.letter)

    def _area_item(self, run, state: RunQc, area: qa.Area, grid: dict, sm: dict, totals: dict,
                   clusters: list | None = None) -> dict:
        """One drawn area as stored: {L, tag, shape, judged, per100, start_per100, cells, growth, tiny, tiny_start,
        shrunk, churn, noise, depth, depth_start, bins (how many bins its outline is drawn around), edge (it touches
        the occupied tissue's edge), local (its own comparison with the start, qc_areas.local_change: {judged, paired,
        matched_change, threshold, enough, direction}), clusters ([{id, label}], only when the frame has a Clusters &
        UMAP result: `_area_clusters`)}; its outline (qc_areas.outline, at most qc_areas.outline_points of the run's
        constants) is stored once per run under `shape`."""
        drawn = qa.drawn_bins(area.bins, sm, grid)
        shape, outline = qa.outline(drawn, grid, qa.outline_points(self._area_params(run)))
        if shape not in state.shapes:
            state.shapes[shape] = [list(point) for point in outline]
            state.shapes_pending.append(shape)
            state.shapes_missing.discard(shape)
        start = None if math.isnan(area.start_per100) else round(area.start_per100, 1)
        local = area.local if area.local else qa.local_change(area.bins, sm, self._mode(run))
        item = {"L": area.letter, "tag": area.tag, "shape": shape, "judged": int(round(area.judged)),
                "per100": round(area.per100, 1), "start_per100": start, **qa.evidence(area.bins, sm, totals),
                "bins": int(len(drawn)), "edge": bool(area.edge), "local": dict(local)}
        if clusters:
            item["clusters"] = clusters
        return item

    def _area_clusters(self, run, seq, grid: dict, areas: list) -> dict:
        """{letter: [{id, label}]}: the cell groups that meet in each area when this frame has a
        Clusters & UMAP result: its clusters holding >= CLUSTER_SHARE of the area's clustered cells, largest first, at
        most CLUSTERS_LISTED, named by their first two top genes as the UMAP pane names them. A cell sits in the bin of
        its centroid in this frame (the frame's cell statistics). {} when either is missing. Reads files: called
        without the run's lock (ingest), and only the cells' labels of the UMAP result's per-cell file."""
        analysis = self.analysis
        if not areas or seq is None or not hasattr(analysis, "cells_path") or not hasattr(analysis, "summary"):
            return {}
        ncells = int((run.info or {}).get("cells") or 0)
        with run.lock:
            record = run.frames[seq] if 0 <= seq < len(run.frames) else {}
        if not ncells or not record.get("cells_file"):
            return {}
        try:
            import zstandard
            # Both reads of one settings generation: a change between them is a 404 (no groups for this frame), never
            # one generation's labels named by another's clusters.
            gen = analysis.generation(run.meta["id"]) if hasattr(analysis, "generation") else None
            which = {} if gen is None else {"generation": gen}
            summary = analysis.summary(run.meta["id"], seq, **which)
            with open(analysis.cells_path(run.meta["id"], seq, **which), "rb") as stream:   # its labels: 2 B a cell
                clusters = np.frombuffer(stream.read(2 * ncells), "<i2")
            raw = (run.live / record["cells_file"]).read_bytes()
            stats = np.frombuffer(zstandard.ZstdDecompressor().decompressobj().decompress(raw), "<u4")
        except Exception:                            # not analysed (yet), or a file went away: no groups
            return {}
        if clusters.size != ncells or stats.size != 7 * ncells:
            return {}
        stats = stats.reshape(7, ncells)
        cx, cy = stats[5].view("<f4"), stats[6].view("<f4")
        keep = np.isfinite(cx) & np.isfinite(cy) & (clusters >= 0)
        size, nx, ny = float(grid["bin"]), int(grid["nx"]), int(grid["ny"])
        bx = np.clip(np.floor((cx[keep] - float(grid["x0"])) / size), 0, nx - 1).astype(np.int64)
        by = np.clip(np.floor((cy[keep] - float(grid["y0"])) / size), 0, ny - 1).astype(np.int64)
        cell_bins, cell_groups = by * nx + bx, clusters[keep].astype(np.int64)
        labels = {}
        for item in summary.get("clusters") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                genes = [str(t.get("gene")) for t in (item.get("top") or [])[:2] if isinstance(t, dict) and t.get("gene")]
                labels[item["id"]] = " · ".join(genes) or str(item.get("label") or f"group {item['id']}")
        out = {}
        for area in areas:
            groups = cell_groups[np.isin(cell_bins, area.bins)]
            if not groups.size:
                continue
            ids, counts = np.unique(groups, return_counts=True)
            order = np.argsort(-counts, kind="stable")[:CLUSTERS_LISTED]
            picked = [int(ids[i]) for i in order if counts[i] >= CLUSTER_SHARE * groups.size]
            if picked:
                out[area.letter] = [{"id": i, "label": labels.get(i, f"group {i}")} for i in picked]
        return out

    def _start_area_fields(self, run, state: RunQc) -> dict | None:
        """The start line's own areas (all "already": where its cells look mixed before anything moved), as stored."""
        start = state.start_bins
        state.start_drawn = None
        if start is None:
            return None
        params = self._area_params(run)
        found, _ = qa.start_areas(start["grid"], params, start["bins"])
        drawn = _listed([area for area in found if area.letter])
        state.start_drawn = [(area.letter, area.bins) for area in drawn]
        totals = qa.totals(start["bins"])
        areas = []
        for area in drawn:
            item = self._area_item(run, state, area, start["grid"], start["bins"], totals)
            item.update(tiny_start=item["tiny"], depth_start=item["depth"])   # the start is its own start
            areas.append(item)
        return {"areas": areas, "grid": start["grid"]}

    def _start_drawn(self, run, state: RunQc, grid: dict) -> list:
        """[(letter, bins)] of the start line's drawn areas when they are on `grid` (what the page draws until the
        run's own areas are shown), else []. Computed once per start line. State lock held."""
        start = state.start_bins
        if start is None or start["grid"] != grid:
            return []
        if state.start_drawn is None:
            found, _ = qa.start_areas(start["grid"], self._area_params(run), start["bins"])
            state.start_drawn = [(area.letter, area.bins) for area in found if area.letter]
        return state.start_drawn

    @staticmethod
    def _read_lines(state: RunQc, name: str) -> list[str]:
        """The complete lines of one of the run's append-only files (series.jsonl, shapes.jsonl, drawn.jsonl). A last
        line without its newline (the engine was killed while writing it) is left out and cut off the file, so the
        next line appended is whole again. State lock held."""
        path = state.dir / name
        try:
            raw = path.read_bytes()
        except OSError:
            return []
        end = raw.rfind(b"\n") + 1
        if end < len(raw):
            try:
                with open(path, "r+b") as stream:
                    stream.truncate(end)
            except OSError:
                pass
        state.tails.add(name)
        return raw[:end].decode("utf-8", "replace").splitlines()

    @staticmethod
    def _append_lines(state: RunQc, name: str, text: str):
        """Append whole lines (`text` ends with a newline) to one of the run's append-only files, after making sure,
        once per session, that the file does not end in a torn line (which would swallow the first one appended).
        State lock held; the run's qc folder exists."""
        path = state.dir / name
        if name not in state.tails:
            state.tails.add(name)
            try:
                with open(path, "rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    if stream.tell():
                        stream.seek(-1, os.SEEK_END)
                        if stream.read(1) != b"\n":
                            text = "\n" + text
            except OSError:
                pass
        with open(path, "a") as stream:
            stream.write(text)

    def _write_shapes(self, state: RunQc):
        """Outlines not written yet, one shapes.jsonl line each (the run folder exists). State lock held."""
        if not state.shapes_pending:
            return
        lines = "".join(json.dumps({"id": shape, "outline": state.shapes[shape]}, separators=(",", ":")) + "\n"
                        for shape in state.shapes_pending)
        self._append_lines(state, "shapes.jsonl", lines)
        state.shapes_pending = []

    @classmethod
    def _load_shapes(cls, state: RunQc):
        """The outlines shapes.jsonl holds; ids asked for that it does not hold are remembered (shapes_missing), so a
        summary naming a lost outline does not read the whole file again on every read."""
        for line in cls._read_lines(state, "shapes.jsonl"):
            try:
                item = json.loads(line)
                state.shapes[str(item["id"])] = [list(point) for point in item["outline"]]
            except (ValueError, KeyError, TypeError):
                continue

    @staticmethod
    def _read_bins(path: Path) -> dict | None:
        """A bins file as {grid, bins (dense)}, or None."""
        try:
            with np.load(path, allow_pickle=False) as stored:
                table = {k: stored[k] for k in stored.files}
        except (OSError, ValueError):
            return None
        grid = qa.grid_of(table)
        if grid is None:
            return None
        dense = qa.dense(table, grid)
        return {"grid": grid, "bins": dense} if "J" in dense and "M" in dense else None

    def _load_tracker(self, run, state: RunQc, saved: dict | None):
        """The areas where an earlier session left them (state.json "areas" of this model and kind of run), and the set
        drawn then (a change of it at the next result saves state.json at once). State lock held."""
        if not isinstance(saved, dict) or saved.get("kind") != self._area_kind(run):
            return
        grid = qa.grid_of({"grid": saved.get("grid")})
        tracker = None if grid is None else qa.Tracker.from_state(saved, self._area_params(run))
        if tracker is None:
            return
        state.tracker = tracker
        shown = [(a.letter, a.bins) for a in tracker.areas] if tracker.revealed else self._start_drawn(run, state, grid)
        state.displayed = ({letter for letter, _ in shown},
                           set().union(*(set(np.asarray(b).tolist()) for _, b in shown)) if shown else set())

    def _save_state(self, run, state: RunQc):
        """state.json, when a live result came since it was last written: the headline's smoothing (ema, ema_matched,
        ema_start, shown, last_live, last_t, live_count), the areas' (qc_areas.Tracker.state, with the kind of the run's
        grid) and the running trial as last shown (a run does not go on after a restart: its readout stays as it was).
        State lock held; the run folder exists."""
        if not state.state_dirty or state.dir is None:
            return
        tracker = state.tracker
        areas = None if tracker is None else {**tracker.state(), "kind": self._area_kind(run), "seq": state.last_live}
        _atomic(state.dir / "state.json", json.dumps(
            {"model": state.key, "ema": state.ema, "ema_matched": state.ema_matched, "ema_start": state.ema_start,
             "shown": state.shown, "last_live": state.last_live,
             "last_t": state.last_t, "live_count": state.live_count, "updated": now_iso(), "areas": areas,
             "area_order": state.area_order,
             "trial": state.trial_view if state.trial is not None else None},
            separators=(",", ":")).encode())
        state.state_saved, state.state_dirty = self.clock(), False

    def _expand(self, state: RunQc, summary: dict) -> dict:
        """A summary as the page gets it: each area with its outline ([[x, y] µm, ...]) beside its shape id. A frame
        before the run's own areas are shown (area_view "start": cells still form) gets the start line's areas."""
        # Older runs may still have the removed comparison in their cached summaries.
        summary = {k: v for k, v in summary.items() if k not in ("previous", "prev_change")}
        areas = summary.get("areas")
        with state.lock:
            if summary.get("area_view") == "start" and state.start_areas and state.start_areas.get("areas"):
                areas = state.start_areas["areas"]
            if not areas:
                return summary
            wanted = {area.get("shape") for area in areas} - set(state.shapes) - state.shapes_missing
            if wanted:
                self._load_shapes(state)              # written by this run's earlier session
                state.shapes_missing |= wanted - set(state.shapes)   # lost (a torn line): never read again for them
            shapes = state.shapes
            return {**summary, "areas": [{**area, "outline": shapes.get(area.get("shape"), [])} for area in areas]}

    def _start_area_public(self, state: RunQc) -> dict:
        """The start line's areas as the page gets them: {areas (with outlines), grid}."""
        with state.lock:
            fields = state.start_areas
            if not fields:
                return {"areas": [], "grid": None}
            return {**self._expand(state, {"areas": fields.get("areas") or []}), "grid": fields.get("grid")}

    def fail(self, run_id: str, seq, error: str):
        """The worker could not score a frame (or the start line, seq None): an error event for the page."""
        state = self._state(run_id)
        if state is None:
            return
        with state.lock:
            number = (state.errors[-1]["n"] if state.errors else 0) + 1
            state.errors.append({"seq": seq, "error": str(error)[:300], "n": number})
            del state.errors[:-20]
        self._bump()

    @staticmethod
    def _start_summary(raw: dict) -> dict:
        """The start line as start.json keeps it: its numbers, its coverage over all the run's cells (empty, under 20,
        20-99: the worker's counts) and its transcripts of the model's genes (`tx`: the retention's yardstick)."""
        judged, mixed = int(raw.get("judged") or 0), int(raw.get("mixed") or 0)
        return {"per100": _per100(mixed, judged), "judged": judged, "mixed": mixed, "median_tx": raw.get("median_tx"),
                "cells": raw.get("cells", raw.get("ncells")), "ms": raw.get("ms") or {},
                "empty": _int(raw.get("empty")), "under20": _int(raw.get("under20")),
                "under100": _int(raw.get("under100")),
                "tx": qr.number(raw.get("tx_sum", raw.get("transcripts", raw.get("tx"))))}

    @staticmethod
    def _start_detail(start: dict | None, mode: str = "region") -> dict:
        """The start line's coverage and transcripts, for describe, GET /qc/start and the start event."""
        if not start:
            return {}
        return {"coverage": _coverage({"ncells": start.get("cells"), **{k: start.get(k) for k in (
                    "judged", "empty", "under20", "under100")}}, mode),
                "tx": None if qr.number(start.get("tx")) is None else int(round(start["tx"]))}

    @staticmethod
    def _start_public(start: dict | None) -> dict | None:
        return None if start is None else {k: start.get(k) for k in ("per100", "judged", "mixed", "median_tx")}

    def _batch(self, run, state: RunQc) -> dict | None:
        """The earlier proseg cells (batch.parquet: the sample's other table, scored once by the same model), counted
        inside the run's region: {table, cells, judged, mixed, per100}. Only a table of proseg's own count
        (qc_trainer.proseg_table: its uns in the sample holds proseg's run record) is proseg's cells: for any other
        (e.g. Xenium Ranger's outline re-count a model learned before this rule scored) per100 is None and no line is
        shown. Read once per model and session."""
        if state.key is None:
            return None
        if state.batch is not None and state.batch.get("key") == state.key:
            return state.batch
        value = {"key": state.key}
        try:
            import pyarrow.parquet as pq
            name = ((self.models._summary(state.model_dir) or {}).get("batch") or {}).get("table")
            if not trainer.proseg_table(run.meta.get("sample_path"), name):
                raise LookupError(f"the table {name!r} is not proseg's own count")
            table = pq.read_table(state.model_dir / "batch.parquet", columns=["x", "y", "counts", "share"])
            x, y = np.asarray(table["x"], np.float64), np.asarray(table["y"], np.float64)
            counts, share = np.asarray(table["counts"], np.float64), np.asarray(table["share"], np.float64)
            region = run.meta.get("region") or {"kind": "box", "box": run.meta.get("box")}
            if region.get("kind") == "polygon":
                inside = _point_in_polygon(x, y, region["points"])
            elif region.get("kind") == "box" and region.get("box"):
                x0, y0, x1, y1 = region["box"]
                inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            else:
                inside = np.ones(len(x), bool)
            judged = inside & (counts >= self.s.judge_min)
            mixed = judged & (share >= self.s.mixed_at)
            value.update(table=name, cells=int(inside.sum()), judged=int(judged.sum()), mixed=int(mixed.sum()),
                         per100=_per100(int(mixed.sum()), int(judged.sum())))
        except Exception:
            value.update(per100=None)
        state.batch = value
        return value

    # ---- reading results back ---------------------------------------------------------------------------------------
    def _ready_state(self, run_id: str) -> RunQc:
        self._run(run_id)
        state = self._state(run_id)
        if state is None or state.key is None:
            raise QcError("No mixing check for this run yet", 404)
        return state

    def summary(self, run_id: str, seq: int) -> dict:
        state = self._ready_state(run_id)
        value = self._summary_of(state, int(seq))
        if value is None:
            raise QcError("Not checked yet", 404)
        return value

    def _summary_of(self, state: RunQc, seq: int) -> dict | None:
        """A result's full summary as the page gets it (`_expand`): as stored, in memory for the newest RESULTS_KEPT,
        else its <seq>.qc.json (None: not scored). A frame's areas are always its own: those of when it was scored."""
        value = self._stored(state, seq)
        return None if value is None else self._expand(state, value)

    @staticmethod
    def _stored(state: RunQc, seq: int) -> dict | None:
        """A result's summary as stored (areas naming their shapes, no outlines), of the run's current model: in memory
        for the newest RESULTS_KEPT, else its <seq>.qc.json; None when not scored."""
        with state.lock:
            line = state.results.get(seq)
            known, folder, key = seq in state.points, state.dir, state.key
        if line is None:
            if not known:
                return None
            try:
                line = (folder / f"{seq:06d}.qc.json").read_text()
            except OSError:
                return None
        try:
            value = json.loads(line)
        except ValueError:
            return None
        return value if isinstance(value, dict) and value.get("model") == key else None

    def bins_grid(self, run_id: str, seq: int, model: str | None = None) -> tuple[Path, dict]:
        """(<seq>.qc.grid: the frame's per-bin share, one byte per bin row-major, zstd; its grid {x0, y0, bin, nx, ny})
        for GET /qc/{seq}/bins: the smoothed share for live results, the frame's own for single ones."""
        state = self._ready_state(run_id)
        if model is not None and model != state.key:
            raise QcError("This iteration was checked by another model", 404)
        summary = self._summary_of(state, int(seq))
        path = state.dir / f"{int(seq):06d}.qc.grid"
        if summary is None or not isinstance(summary.get("grid"), dict) or not path.exists():
            raise QcError("No area map for this iteration", 404)
        return path, summary["grid"]

    @staticmethod
    def _no_start(run):
        """A start line's route asked of a run that has none (a start with nuclei, qc_rules.start_of not "cells"):
        refused for good, in Go deeper's words, 404 with reason "no_start" and its start_kind; a run from cells alone
        whose start line is not scored yet is refused by the route itself ("yet")."""
        kind = qr.start_of(run.meta)
        if kind != "cells":
            error = QcError(qr.NO_START_WORDS.get(kind, qr.NO_START_WORDS["paired"]), 404)
            error.extra = {"reason": "no_start", "start_kind": kind}
            raise error

    def start_grid(self, run_id: str, model: str | None = None) -> tuple[bytes, dict]:
        """(the start line's overview grid, qa.overview_grid; its grid with "planes") for GET /qc/start/bins."""
        run = self._run(run_id)
        self._no_start(run)
        state = self._ready_state(run_id)
        if model is not None and model != state.key:
            raise QcError("The start line was checked by another model", 404)
        with state.lock:
            start = state.start_bins
            if start is None or not self._start_line(run):
                raise QcError("No start line for this run yet", 404)
            grid = start["grid"]
            data = qa.overview_grid(start["bins"], (grid["ny"], grid["nx"]), int(self._area_params(run)["window"]))
            return data.tobytes(), {**grid, "planes": qa.OVERVIEW_PLANES}

    def cells_path(self, run_id: str, seq: int, model: str | None = None, count: str | None = None) -> Path:
        """A frame's bytes (one per cell): <seq>.qc.bin, on the count it was read on. `count` names the count the
        caller means: "sampled" (the count proseg saves) only while the frame's summary is on it; "averaged" its
        <seq>.qc.bin while the summary is averaged, its <seq>.qc.averaged.bin (the worker's out_cells_e) while it is
        sampled; any other count, or that file missing: 404 (the frame was counted again meanwhile: the caller asks the
        check again). 404 too, with a count named, while the frame is being scored again (`counted_again`: the worker
        rewrites <seq>.qc.bin on the new count before its result is in, so until then those bytes may be either
        count's). 404 too for another model, or a frame not checked yet."""
        state = self._ready_state(run_id)
        if model is not None and model != state.key:
            raise QcError("This iteration was checked by another model", 404)
        seq = int(seq)
        name = f"{seq:06d}.qc.bin"
        if count is not None:
            with state.lock:
                point = state.points.get(seq)
            stored = (point or {}).get("count") or "averaged"
            if count not in COUNTS or (count == "sampled" and stored != "sampled") or point is None \
                    or counted_again(self._run(run_id), seq, stored):
                raise QcError("This iteration's count changed", 404)
            if count == "averaged" and stored == "sampled":
                name = f"{seq:06d}.qc.averaged.bin"
        path = state.dir / name
        if not path.exists():
            raise QcError("This iteration's count changed" if count is not None else "Not checked yet", 404)
        return path

    def start_info(self, run_id: str) -> dict:
        run = self._run(run_id)
        self._no_start(run)
        state = self._ready_state(run_id)
        with state.lock:
            if state.start is None or not self._start_line(run):
                raise QcError("No start line for this run yet", 404)
            return {**state.start, "model": state.key, "start_source": self._start_source(run),
                    **self._start_detail(state.start, self._mode(run)), **self._start_area_public(state)}

    def start_cells(self, run_id: str, model: str | None = None) -> tuple[Path, int]:
        """start.bin and the run's cell count (its first ncells bytes are the per-cell values); `model`: the model id
        the caller means (404 for another one: the run was scored again by a relearned model meanwhile)."""
        run = self._run(run_id)
        self._no_start(run)
        state = self._ready_state(run_id)
        if model is not None and model != state.key:
            raise QcError("The start line was checked by another model", 404)
        path = state.dir / "start.bin"
        if not path.exists() or not self._start_line(run):
            raise QcError("No start line for this run yet", 404)
        return path, int((run.info or {}).get("cells") or 0)

    def series(self, run_id: str, points: int = 600) -> dict:
        """The line over the whole run (every result's point, thinned): it starts at the first frame scored however
        long the run has gone on."""
        run = self._run(run_id)
        state = self._ready_state(run_id)
        with state.lock:
            items = [state.points[s] for s in sorted(state.points)]
            start = self._start_full(run, state)
        return {"model": state.key, "start": start, "points": _thin(items, points)}

    # ---- suggestions (engine/qc_rules.py) ---------------------------------------------------------------------------
    def _suggest(self, run, state: RunQc, seq: int, frame: dict, result: dict, fields: dict, drawn: list, grid,
                 sm, change: dict | None = None, start_now=None) -> dict:
        """A live result's suggestion fields: it joins the history a knob change is judged against; a change of the
        frame's knob values (a trial's, or one made in Live knobs) starts a Watch (`_knob_change`); the running trial
        follows its change; the rules give the cards (none while a trial runs: registered, waiting for its frame or
        running), the keeper the one shown and the more ideas. State lock held.
        Returns {suggestion, more_ideas, trial}."""
        confirmed = fields.get("area_view") == "smoothed" and not fields.get("areas_carried")
        entry = self._entry(state, seq, frame, result, fields if confirmed else None, change)
        recent = state.recent
        recent.append(entry)
        cut = 0
        while cut < len(recent) - 1 and (len(recent) - cut > RECENT_KEPT
                                         or recent[cut]["iter"] < entry["iter"] - RECENT_ITERS):
            cut += 1
        del recent[:cut]
        change = self._knob_change(run, state, seq, frame)
        if change is not None:
            self._new_watch(run, state, change, drawn if confirmed else [], grid)
        if state.trial is not None and state.trial_watch is None:
            self._attach_trial(run, state)
        for watch in {id(w): w for w in (state.watch, state.trial_watch) if w is not None}.values():
            watch.update(entry, self._area_shares(watch, sm, grid))
        trial = self._trial_public(run, state)
        if trial is not None:
            state.trial_view = trial
        if confirmed and state.tried:                 # a tried area that cleared: its letter may name another place later
            letters = {a.letter for a in drawn}
            state.tried = [t for t in state.tried if t.get("area") in letters]
        suggestion, more = None, []
        if state.trial is not None:
            state.keeper.reset()                     # never while a trial runs; afresh after it
        else:
            early = self._early(run, state, frame)
            start_shown = fields.get("area_view") == "start" and bool((state.start_areas or {}).get("areas"))
            ctx = {"phase": frame.get("phase"), "iter": frame.get("iter"),
                   "min_judged": self._area_params(run)["enter_min_j"], "live": frame.get("live") or {},
                   "areas": fields.get("areas") if confirmed else [],
                   "whole": {} if early else self._whole(run, state, sm, result, change, start_now),
                   "flagged": early and start_shown, "tried": state.tried,
                   "harm": state.watch.harm_card() if state.watch is not None else None, "later_s": self._later_s(run),
                   "start": qr.start_of(run.meta)}
            # The letters on screen: carried areas too (a result without per-bin sums keeps them drawn), so a card does
            # not blink out with them; only confirmed areas get new cards.
            suggestion, more = state.keeper.choose(qr.evaluate(ctx), entry.get("t"),
                                                   [a.letter for a in drawn if getattr(a, "letter", "")])
        out = {"suggestion": self._card(suggestion), "trial": self._trial_compact(trial)}
        if more:                                      # only when there are some (the summary stays <= 8 KB)
            out["more_ideas"] = [self._card(card) for card in more]
        return out

    @staticmethod
    def _card(card: dict | None) -> dict | None:
        """A card as the page gets it, with its identity (`id`, qc_rules.card_id: the page keys the card shown, the idea
        it picked and the cards it set aside by it, never by the areas, which may join or leave) and without the Undo
        card's watch number (its id carries it)."""
        return None if card is None else {k: v for k, v in card.items() if k != "watch"}

    @staticmethod
    def _trial_compact(trial: dict | None) -> dict | None:
        """The running trial as each summary carries it (GET /runs/{id}/qc and the push carry the rest)."""
        return None if trial is None else {k: v for k, v in trial.items() if k not in ("started", "asked_iter")}

    def _fit(self, state: RunQc, summary: dict):
        """Keep a live summary within SUMMARY_BYTES as the page gets it (outlines added, compact JSON): the last of the
        more ideas go first (the card, the areas and the trial stay). State lock held."""
        while summary.get("more_ideas") and \
                len(json.dumps(self._expand(state, summary), separators=(",", ":"))) > SUMMARY_BYTES:
            summary["more_ideas"] = summary["more_ideas"][:-1]
            if not summary["more_ideas"]:
                del summary["more_ideas"]

    @staticmethod
    def _entry(state: RunQc, seq: int, frame: dict, result: dict, fields: dict | None, change: dict | None = None) -> dict:
        """What the history keeps of a live result: {seq, iter, t, phase, v (the smoothed share, unrounded), raw (its
        own), mc (the smoothed depth-matched change against the start, change.matched; None without one), mp, mn (the
        smoothed matched share now and paired cells: a band's threshold), tx (typical transcripts), areas {L: [share,
        judged, matched change, paired]} of its confirmed areas (their own: local)}. State lock held (after the
        headline took it in)."""
        judged, mixed = int(result.get("judged") or 0), int(result.get("mixed") or 0)
        ema = state.ema
        areas = {}
        for item in (fields or {}).get("areas") or []:
            if item.get("L") and qr.number(item.get("per100")) is not None:
                local = item.get("local") or {}
                areas[str(item["L"])] = [item.get("per100"), item.get("judged"), local.get("matched_change"),
                                         local.get("paired")]
        mc = qr.number((change or {}).get("matched"))
        counts = state.ema_matched if mc is not None else None
        return {"seq": int(seq), "iter": int(qr.number(frame.get("iter")) or 0), "t": _seconds(frame.get("t")),
                "phase": frame.get("phase"), "v": 100.0 * ema[1] / ema[0] if ema and ema[0] else None,
                "raw": 100.0 * mixed / judged if judged else None, "mc": mc,
                "mp": counts[1] / counts[0] if counts and counts[0] > 0 else None,
                "mn": counts[0] if counts else None, "tx": qr.number(result.get("median_tx")), "areas": areas}

    @staticmethod
    def _knobs(run, live) -> dict | None:
        """A frame's knob values as changes are counted: the temperature only when the user set it (a schedule may move
        it by itself)."""
        if not isinstance(live, dict):
            return None
        user = run.meta.get("live_values") or {}
        return {str(k): v for k, v in live.items() if k != "temperature" or "temperature" in user}

    @staticmethod
    def _requested(run) -> dict:
        """The knob values the run was asked for: its startup options with the live values set since over them (the
        exact numbers the user typed; frames carry proseg's 32-bit floats)."""
        options, live = run.meta.get("options"), run.meta.get("live_values")
        return {**(options if isinstance(options, dict) else {}), **(live if isinstance(live, dict) else {})}

    def _knob_change(self, run, state: RunQc, seq: int, frame: dict) -> dict | None:
        """Whether the knob values changed since the previous live result: {changes [{flag, from, to}], start (the first
        frame showing any of them: {seq, iter, phase})}, or None. Each changed flag's `from` is the previous result's
        value and its `to` this result's (two changes landing between two results are both there, so a watch of more
        than one change is "mixed up"); each the exact value asked for when the frame shows that value (`_requested`).
        State lock held."""
        knobs, asked = self._knobs(run, frame.get("live")), self._requested(run)
        previous, state.prev_live = state.prev_live, (seq, knobs, asked)
        if previous is None or knobs is None or previous[1] is None:
            return None
        before_seq, before, asked_before = previous
        flags = [f for f in sorted(set(before) | set(knobs)) if not qr.same(before.get(f), knobs.get(f))]
        if not flags:
            return None
        with run.lock:                                # (a long gap, e.g. the check off for a while: this frame)
            between = [dict(f) for f in run.frames[before_seq + 1:seq + 1]] if seq - before_seq <= RECENT_KEPT else []
        start = frame
        for candidate in between:                     # the first frame showing any of them
            shown = self._knobs(run, candidate.get("live"))
            if shown is not None and any(not qr.same(before.get(f), shown.get(f)) for f in flags):
                start = candidate
                break

        def exact(value, wanted: dict, flag: str):
            ask = wanted.get(flag)
            return ask if ask is not None and qr.same(ask, value) else value
        return {"changes": [{"flag": f, "from": exact(before.get(f), asked_before, f), "to": exact(knobs.get(f), asked, f)}
                            for f in flags],
                "start": {"seq": start.get("seq", seq), "iter": start.get("iter", frame.get("iter")),
                          "phase": start.get("phase", frame.get("phase"))}}

    def _new_watch(self, run, state: RunQc, change: dict, drawn: list, grid):
        """A knob change: the Watch that judges it (the Undo card's). A trial's change is frozen by another change:
        "mixed up", unless the new change puts the trial's back (its Undo). State lock held."""
        trial = state.trial
        if state.trial_watch is not None and not state.trial_watch.frozen:
            undone = trial is not None and any(c["flag"] == trial["flag"] and qr.same(c["to"], trial.get("from"))
                                               for c in change["changes"])
            state.trial_watch.freeze(mixed_up=not undone)
            state.trial_n += 1
        state.watch_n += 1
        mode = self._mode(run)
        areas = {a.letter: (a.bins, a.judged, (getattr(a, "local", None) or {}).get("paired")) for a in drawn
                 if a.letter and a.judged >= qk.value(mode, "local_min_judged")}
        watch = qr.Watch(state.watch_n, change["changes"], change["start"], state.recent, self._reveal_iter(run, state),
                         areas, mode=mode)
        watch.grid = dict(grid) if isinstance(grid, dict) else None
        state.watch = watch
        state.keeper.reset()                         # a new situation: cards are shown afresh (U goes at once)

    @staticmethod
    def _area_shares(watch: qr.Watch, sm: dict | None, grid) -> dict:
        """{L: {share, judged, mc, paired}} on the bins area L had when the change started, from this result's smoothed
        sums (the same place whatever the letters do): its share of judged cells that look mixed, and its own
        depth-matched change against the start (qc_areas.local_change; None without the worker's matched sums)."""
        if not watch.area_bins or sm is None or "J" not in sm or "M" not in sm or watch.grid != grid:
            return {}
        J, M = np.asarray(sm["J"]).ravel(), np.asarray(sm["M"]).ravel()
        out = {}
        for letter, bins in watch.area_bins.items():
            bins = bins[(bins >= 0) & (bins < J.size)]
            judged = float(J[bins].sum())
            if judged > 0:
                local = qa.local_change(bins, sm, watch.mode)
                out[letter] = {"share": 100.0 * float(M[bins].sum()) / judged, "judged": judged,
                               "mc": local["matched_change"], "paired": local["paired"]}
        return out

    def _whole(self, run, state: RunQc, sm: dict | None, result: dict, change: dict | None = None,
               start_now=None) -> dict:
        """The whole tissue for R2 and R3: {per100 (smoothed), start_per100 (the frame's own start block's share,
        `start_now`, smoothed as the headline), rose (the run's paired cells: (n10 - n01) /
        sqrt(n10 + n01) >= 2 and +3 points over the same cells at the start, as an area's tag; and, with the worker's
        depth-matched change, that change counting: `change`, qc_rules.change_block), growth (the median current /
        starting area - 1), change}, from the smoothed bin sums, else from the worker's whole-run numbers."""
        params = self._area_params(run)
        ema = state.ema
        out = {"per100": 100.0 * ema[1] / ema[0] if ema and ema[0] else None,
               "start_per100": None if start_now is None else round(start_now, 1), "rose": False, "growth": None,
               "change": change or {}}
        totals = qa.totals(sm) if isinstance(sm, dict) and isinstance(sm.get("J"), np.ndarray) else None
        paired = {"n10": totals.get("n10"), "n01": totals.get("n01"), "B": totals.get("B"), "N1": totals.get("N1"),
                  "S1": totals.get("S1")} if totals else {}
        if not paired.get("B") and self._start_line(run):
            worker = result.get("paired") if isinstance(result.get("paired"), dict) else {}
            paired = {"n10": worker.get("n10"), "n01": worker.get("n01"), "B": worker.get("n"),
                      "N1": worker.get("mixed_now"), "S1": worker.get("mixed_start")}
        n10, n01, B = (qr.number(paired.get(k)) or 0.0 for k in ("n10", "n01", "B"))
        if B > 0 and n10 + n01 > 0:
            z = (n10 - n01) / math.sqrt(n10 + n01)
            rise = ((qr.number(paired.get("N1")) or 0.0) - (qr.number(paired.get("S1")) or 0.0)) / B
            out["rose"] = bool(z >= params["rose_z"] and rise >= params["rose_points"])
            if out["rose"] and (change or {}).get("matched") is not None:     # a rise depth alone explains is none
                out["rose"] = change.get("direction") == "more"
        if totals:
            out["growth"] = qa.evidence(np.arange(sm["J"].size), sm, totals).get("growth")
        if out["growth"] is None:
            growth = result.get("growth") if isinstance(result.get("growth"), dict) else {}
            median = qr.number(growth.get("median"))
            out["growth"] = None if median is None else round(median - 1.0, 3)
        return out

    def _later_s(self, run) -> float | None:
        """1,000 iterations of this run once cells formed, in seconds (the timing line's "about {t} here")."""
        with run.lock:
            frames = run.frames[-PACE_FRAMES:]
        pace = qr.seconds_per_iteration(frames)
        return None if pace is None else 1000.0 * pace

    # ---- trials (the try loop: POST /runs/{id}/qc/trial and /trial/end) ----------------------------------------------
    def trial(self, run_id: str, body: dict) -> dict:
        """Register a trial ({rule, area, flag, from, to}): the page sets the knob itself (POST /control set) and tells
        the check which change to judge. It starts at the first frame whose live values show `to` (from shortly before
        it was asked for: the page may set the knob first); while it runs the page gets no card. 409 while another runs
        or when the run is not sampling."""
        run = self._run(run_id)
        state = self._ready_state(run_id)
        if not isinstance(body, dict):
            raise QcError("A trial is {rule, area, flag, from, to}")
        rule, area, flag = body.get("rule"), body.get("area"), body.get("flag")
        if not isinstance(rule, str) or not re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", rule):
            raise QcError("rule must be a rule name such as R4")
        if area is not None and (not isinstance(area, str) or not re.fullmatch(r"[A-Z]{1,2}", area)):
            raise QcError("area must be an area letter or null")
        if not isinstance(flag, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", flag):
            raise QcError("flag must be a proseg option name")
        live = self._live_flags()
        if live is not None and flag not in live:
            raise QcError("flag must be a setting Live knobs can change")
        for name in ("from", "to"):
            value = body.get(name)
            if not (value is None or isinstance(value, (bool, str)) or
                    (isinstance(value, (int, float)) and math.isfinite(value))) or (isinstance(value, str) and len(value) > 64):
                raise QcError(f"{name} must be a setting value")
        if run.meta.get("state") not in ("running", "paused"):
            raise QcError("This run is not sampling", 409)
        with state.lock:
            if state.trial is not None:
                raise QcError("A trial is already running: keep or undo it first", 409)
            with run.lock:
                last = run.frames[-1] if run.frames else {}
            state.trials += 1
            record = {"id": state.trials, "rule": rule, "area": area, "flag": flag, "from": body.get("from"),
                      "to": body.get("to"), "asked_seq": last.get("seq"), "asked_iter": last.get("iter"),
                      "started": now_iso()}
            state.trial, state.trial_watch, state.trial_view, state.trial_closed = record, None, None, None
            self._attach_trial(run, state)            # its change may already show (the knob was set first)
            state.keeper.reset()
            state.trial_n += 1
            self._append(state, "trials.jsonl", {"event": "start", **record})
            value = self._trial_public(run, state)
        self._bump()
        return value

    @staticmethod
    def _live_flags() -> set | None:
        """The settings Live knobs can change (engine/proseg_options.py), None when the schema is not installed."""
        try:
            from . import proseg_options
        except ImportError:
            return None
        return {o["flag"] for o in proseg_options.OPTIONS if o.get("live")}

    def _attach_trial(self, run, state: RunQc):
        """The running trial follows the newest change when it is the trial's (its flag set to `to`) and started at most
        TRIAL_LOOKBACK frames before the trial was asked for; its area is watched too (its bins now). State lock
        held."""
        trial, watch = state.trial, state.watch
        if trial is None or state.trial_watch is not None or watch is None or watch.frozen:
            return
        asked = qr.number(trial.get("asked_seq"))
        if not watch.matches(trial["flag"], trial.get("to")) or \
                (asked is not None and watch.start_seq < asked - TRIAL_LOOKBACK):
            return
        state.trial_watch = watch
        trial["start_seq"], trial["start_iter"] = watch.start_seq, watch.start_iter
        if trial.get("from") is None:
            trial["from"] = next((c["from"] for c in watch.changes if c["flag"] == trial["flag"]), None)
        letter = trial.get("area")
        if letter and letter not in watch.area_bins and state.tracker is not None:
            found = next((a for a in state.tracker.areas if a.letter == letter), None)
            if found is not None:
                if watch.grid is None:
                    watch.grid = dict(state.tracker.grid)
                if watch.grid == state.tracker.grid:
                    watch.add_area(letter, found.bins, found.judged,
                                   paired=(getattr(found, "local", None) or {}).get("paired"))
        state.trial_n += 1

    def _change_frame(self, run, trial: dict) -> dict | None:
        """The first frame from TRIAL_LOOKBACK frames before the trial was asked for where its flag turned `to` (a
        trial no live result has met yet, e.g. while paused), or None (a frame with no earlier one to differ from is no
        change)."""
        asked = qr.number(trial.get("asked_seq"))
        first = 0 if asked is None else max(0, int(asked) - TRIAL_LOOKBACK)
        missing = object()
        with run.lock:
            frames = run.frames[first:]
            before = (run.frames[first - 1].get("live") or {}).get(trial["flag"], missing) if first > 0 else missing
        for frame in frames:
            now = (frame.get("live") or {}).get(trial["flag"], missing)
            if now is not missing and before is not missing and qr.same(now, trial.get("to")) \
                    and not qr.same(before, trial.get("to")):
                return frame
            before = now
        return None

    def _trial_public(self, run, state: RunQc) -> dict | None:
        """The running trial as the page gets it: {id, rule, area, flag, from, to, asked_seq, asked_iter, started,
        start_seq, start_iter (the first frame showing the new value; null until one does), rows [{kind, area?,
        expected, now}], verdict ("less" | "more" | "no_change" | null: no clear change yet), on ("area": the verdict is
        the area's own, an area of >= 300 judged cells; "whole"), notes [forming | no_history | no_matched | mixed_up |
        small_area | last_stage (the run's final settling stage stopped it)], forming,
        mixed_up, small_area, depth_loss, band [{iter, expected, low, high}] | null, results, horizon_iter, final,
        timing, later_s?}. State lock held."""
        trial = state.trial
        if trial is None:
            return None
        value = {k: trial.get(k)
                 for k in ("id", "rule", "area", "flag", "from", "to", "asked_seq", "asked_iter", "started")}
        watch = state.trial_watch
        view = state.trial_view
        if watch is not None:
            value.update(start_seq=watch.start_seq, start_iter=watch.start_iter, **watch.readout(trial.get("area")))
            phase = watch.start_phase
        elif view is not None and view.get("id") == trial.get("id") and view.get("start_seq") is not None:
            return dict(view)                          # as the last session showed it (an engine restart)
        else:
            start = self._change_frame(run, trial)
            with run.lock:
                newest = dict(run.frames[-1]) if run.frames else {}
            phase = (start or newest).get("phase")
            reveal, asked = self._reveal_iter(run, state), qr.number(trial.get("asked_iter"))
            forming = phase in FORMING_PHASES or (reveal is not None and asked is not None and asked < reveal)
            value.update(start_seq=None if start is None else start.get("seq"),
                         start_iter=None if start is None else start.get("iter"), rows=[], verdict=None,
                         notes=["forming"] if forming else [], forming=forming, mixed_up=False, small_area=False,
                         depth_loss=None, band=None, results=0, horizon_iter=None, final=False)
        value["timing"] = qr.timing_of(phase)
        later = self._later_s(run) if value["timing"] == "later" else None
        if later is not None:
            value["later_s"] = round(later)
        return value

    def trial_end(self, run_id: str, outcome, rule=None) -> dict:
        """End the running trial ({outcome: kept | undone}: the page put the value back itself) and remember how it
        ended (an area whose trial ended "more" or "no_change" gets rule R7); or answer the Undo card ({rule: "U"}, or
        no trial running while U shows): "kept" keeps the change and the card goes, "undone" (the page put the value
        back) too. 404 when there is neither."""
        run = self._run(run_id)
        state = self._ready_state(run_id)
        if outcome not in ("kept", "undone"):
            raise QcError('outcome must be "kept" or "undone"')
        if rule is not None and rule not in ("U",) and not (isinstance(rule, str) and re.fullmatch(r"R[0-9]", rule)):
            raise QcError("rule must be a rule name such as U")
        with state.lock:
            if rule == "U" or (rule is None and state.trial is None):
                watch = state.watch
                card = None if watch is None else watch.harm_card()
                if card is None:
                    raise QcError("No trial is running" if rule is None else "No change to undo or keep", 404)
                watch.dismissed = True
                state.keeper.reset()
                answered = {**card, "outcome": outcome, "ended": now_iso()}
                self._append(state, "trials.jsonl", {
                    "event": "harm", "watch": watch.id, "outcome": outcome, "ended": answered["ended"],
                    "changed": card["changed"], "areas": card.get("areas"), "before_per100": card.get("before_per100"),
                    "after_per100": card.get("after_per100")})
                result = answered
            else:
                if state.trial is None:
                    raise QcError("No trial is running", 404)
                result = self._end_trial(run, state, outcome)
        self._bump()
        return result

    def _end_trial(self, run, state: RunQc, outcome: str) -> dict:
        """End the running trial with `outcome` (kept | undone: the user's; ended: its run ended): trials.jsonl gets its
        end, an area it showed rising gets rule R7, and its change gets no Undo card any more (the user has just
        answered it: Keep, or Undo with its old value already on its way). Returns it as the page last saw it, with
        the outcome. State lock held."""
        ended = {**self._trial_public(run, state), "outcome": outcome, "ended": now_iso()}
        record = {"event": "end", **{k: ended.get(k) for k in (
            "id", "outcome", "ended", "rule", "area", "flag", "from", "to", "start_seq", "start_iter", "verdict", "on",
            "rows", "notes", "results", "final")}}
        self._append(state, "trials.jsonl", record)
        self._remember_tried(state, record)
        if state.trial_watch is not None:
            state.trial_watch.dismissed = True
        state.trial, state.trial_watch, state.trial_view = None, None, None
        state.keeper.reset()
        state.trial_n += 1
        return ended

    def _close_ended(self, run, state: RunQc) -> bool:
        """A trial whose run no longer samples (finished, stopped or failed; also one restored after an engine restart
        stopped its run) ends with the outcome "ended", so trials.jsonl gets its end and the page its card slot back;
        the page still sees it (`trial_closed`: describe and the trial push). State lock held. Whether it ended one."""
        if state.trial is None or run.meta.get("state") not in ENDED_STATES:
            return False
        state.trial_closed = self._end_trial(run, state, "ended")
        return True

    def _trial_shown(self, run, state: RunQc) -> dict | None:
        """The trial the page is told of (GET /runs/{id}/qc, the trial push): the running one, else the one its run's end
        closed (its outcome "ended"). State lock held."""
        trial = self._trial_public(run, state)
        return trial if trial is not None else state.trial_closed

    @staticmethod
    def _remember_tried(state: RunQc, record: dict):
        """An ended trial on an area that made mixing rise there (its verdict "more": the area's own for an area of >=
        300 judged cells, else the whole tissue's, and the area's own row not below what was expected): rule R7 for that
        area. "No clear change" is no evidence: a shape change after cells formed shows only after ~1,000 iterations,
        never within the 300 a trial is judged over, and a change while cells form has no
        verdict. State lock held."""
        if not record.get("area") or record.get("verdict") != "more":
            return
        row = next((r for r in record.get("rows") or () if isinstance(r, dict) and r.get("kind") == "area"), None)
        now, expected = (qr.number(row.get("now")), qr.number(row.get("expected"))) if row else (None, None)
        if now is not None and expected is not None and now < expected:
            return
        state.tried.append({k: record.get(k) for k in ("area", "flag", "from", "to", "verdict")})

    @classmethod
    def _append(cls, state: RunQc, name: str, item: dict):
        try:
            state.dir.mkdir(parents=True, exist_ok=True)
            cls._append_lines(state, name, json.dumps(item, separators=(",", ":")) + "\n")
        except OSError:
            pass

    # ---- the run log (the Log button: diagnostic detail goes there, not on screen) -----------------------------------
    def log_info(self, run_id: str) -> dict | None:
        """What the run log can say about this run's check beyond the page, read only from what this session holds
        (nothing is loaded, learned or scored): {model, provenance, seq, iter, coverage, retention, matched, change,
        basis, count, constants} of the newest result (None for those before one) and how the run's cells started
        (start_kind: only "cells" has a start line to compare with), or None for a run not looked at in this
        session."""
        with self.lock:
            state = self.states.get(run_id)
        if state is None:
            return None
        try:
            kind = qr.start_of(self.runs.get(run_id).meta)
        except Exception:                               # a run deleted meanwhile: its kind is not said
            kind = None
        with state.lock:
            newest = max(state.points, default=None)
            line = state.results.get(newest) if newest is not None else None
            out = {"model": state.key, "provenance": state.provenance}
        try:
            summary = json.loads(line) if line else {}
        except ValueError:
            summary = {}
        out.update({k: summary.get(k) for k in ("seq", "iter", "coverage", "retention", "matched", "change", "basis",
                                                "count", "constants")}, start_kind=kind)
        return out

    # ---- WebSocket events --------------------------------------------------------------------------------------------
    def events(self, cursor: dict, watched: str | None, now: float) -> list[dict]:
        """QC events one page connection has not had yet. `cursor` is that connection's own dict; cursor["pending"]
        is set when a sample's progress was held back (at most one event per second per sample while it waits or
        learns): ask again shortly even if nothing changed. The watched run's switch goes out as {state: "switch",
        enabled, n} on the first look after a watch and whenever it changes (another window or the API flipped it); `n`
        counts the switch's changes (GET /runs/{id}/qc says switch_n), so the page keeps whichever of the two is newer;
        cursor["switch"] is what the connection had (a new watch drops it). Results: at most PUSHED_SEQS a tick (the
        newest), the others named in one {state: "seqs", seqs} event (the page fetches them when it needs them). A new
        watch replays only the errors no later result answered. The running trial goes out as {state: "trial", trial
        (as GET /runs/{id}/qc says it, null once ended), n} when it is registered, meets its first frame, is mixed up or
        ends (`n` counts those changes; a first look says it only while one runs): every result's summary carries it
        too."""
        out: list[dict] = []
        sent = cursor.setdefault("samples", {})
        cursor["pending"] = False
        for sid, view in self.models.snapshot().items():
            signature = (view["state"], view["reason"], (view["progress"] or {}).get("step"), view["stage"],
                         (view["model"] or {}).get("id"), view["relearn"])
            last = sent.get(sid)
            if last is not None and last[0] == signature:
                continue
            if last is not None and view["state"] in ("waiting", "learning") and last[0][0] == view["state"] \
                    and now - last[1] < 1.0:
                cursor["pending"] = True
                continue
            sent[sid] = (signature, now)
            out.append({"ev": "qc", "sample": sid, **{k: view[k] for k in ("state", "reason", "stage", "progress",
                                                                          "eta_s", "estimate_s", "relearn", "shared",
                                                                          "model")}})
        if watched:
            state = self._state(watched)
            runs = cursor.setdefault("runs", {})
            switches = cursor.setdefault("switch", {})
            if state is not None:
                with state.lock:
                    switch = (bool(state.enabled), state.switch_n)
                had = switches.get(watched)
                switches[watched] = switch
                if had != switch:                        # first look after a watch, or changed since
                    out.append({"ev": "qc", "run": watched, "state": "switch", "enabled": switch[0], "n": switch[1]})
            if state is not None and state.key is not None:
                run = self._run(watched)
                mine = runs.get(watched)
                new = mine is None or mine["model"] != state.key
                if new:
                    mine = runs[watched] = {"model": state.key, "seqs": set(), "start": None, "errors": 0}
                with state.lock:
                    start = self._start_shown(run, state)
                    # A frame counted again (a pause, the hold, a finish or the end: the count proseg saves) is news
                    # again.
                    counted = {s: point.get("count") or "averaged" for s, point in state.points.items()}
                    fresh = sorted(s for s, c in counted.items() if (s, c) not in mine["seqs"])
                    errors = [e for e in state.errors if e["n"] > mine["errors"]]
                    newest = max(state.points, default=None)
                    has_start = state.start is not None
                if new:      # a new watch: an error a later result (or the start line) answered is history
                    mine["errors"] = max((e["n"] for e in errors), default=0)
                    errors = [e for e in errors if not (
                        (e["seq"] is None and has_start) or
                        (e["seq"] is not None and newest is not None and e["seq"] <= newest))]
                if start is not None and mine["start"] != start:
                    mine["start"] = start                # with the start line's own areas (shown while paused on it)
                    out.append({"ev": "qc", "run": watched, "state": "start",
                                "summary": {**start, **self._start_detail(state.start, self._mode(run)),
                                            **self._start_area_public(state)}})
                mine["seqs"].update((s, counted[s]) for s in fresh)
                if len(fresh) > PUSHED_SEQS and not new:  # too many for this tick: named, so the page knows they exist
                    out.append({"ev": "qc", "run": watched, "state": "seqs", "seqs": fresh[:-PUSHED_SEQS]})
                for seq in fresh[-PUSHED_SEQS:]:
                    summary = self._summary_of(state, seq)
                    if summary is None:
                        continue
                    out.append({"ev": "qc", "run": watched, "state": "ready", "seq": seq, "iter": summary.get("iter"),
                                "summary": summary})
                for error in errors:
                    mine["errors"] = max(mine["errors"], error["n"])
                    out.append({"ev": "qc", "run": watched, "state": "error", "seq": error["seq"], "error": error["error"]})
                with state.lock:
                    self._close_ended(run, state)        # a run that ended with a trial running: it ends too
                if mine.get("trial_n") != state.trial_n:
                    with state.lock:
                        trial_n, trial = state.trial_n, self._trial_shown(run, state)
                    first = "trial_n" not in mine
                    mine["trial_n"] = trial_n
                    if not (first and trial is None):
                        out.append({"ev": "qc", "run": watched, "state": "trial", "trial": trial, "n": trial_n})
        return out


def _edge_last(area) -> bool:
    """An area listed after the others: at the tissue's edge and not got more mixed in this run (windows
    at the edge look more mixed by themselves)."""
    return bool(getattr(area, "edge", False)) and getattr(area, "tag", "") != "rose"


def _listed(areas: list) -> list:
    """Drawn areas in the order the page lists them where no earlier order holds (a single snapshot, the start line):
    by rank, the ones at the tissue's edge that did not rise after the others."""
    return sorted(areas, key=_edge_last)


def _gpu(value) -> dict | None:
    """The GPU worker's memory numbers of a pass ({pool_used_mb, pool_total_mb}: its cupy pool) and which allocator
    its arrays came from ("pool" | "rmm"); nothing else."""
    if not isinstance(value, dict):
        return None
    out = {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)}
    if value.get("allocator") in ("pool", "rmm"):
        out["allocator"] = value["allocator"]
    return out or None


def _basis(result: dict) -> dict:
    """Which count a frame was scored on, as the worker says it: {basis ("proseg": proseg's own count of the frame),
    count ("averaged" | "sampled": the count the engine asked it to read, analysis/counts_file.py count_of; ingest
    sets the engine's own over it)}, each None when the worker does not say."""
    return {k: result.get(k) if isinstance(result.get(k), str) else None for k in ("basis", "count")}


def _seconds(value) -> float | None:
    """A frame's time on the run clock, or None when it has none (older runs, tests)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def headline_weight(t: float | None, last_t: float | None, mode: str = "region") -> float:
    """The newest live result's weight in the headline's smoothing (the mode's headline_weight w per headline_ref_s of
    the run clock): w per second of the run clock since the previous live result (1 - (1 - w) ** (dt / 1 s): at a
    region's 0.2, its results 0.25 s apart weigh 0.05 each; at a whole sample's 0.5, its results once a second 0.5; the
    first after a long pause almost 1); w when either result has no time."""
    weight = qk.value(mode, "headline_weight")
    if t is None or last_t is None or not t > last_t:
        return weight
    return 1.0 - (1.0 - weight) ** ((t - last_t) / qk.value(mode, "headline_ref_s"))


def shown_integer(per100: float | None, shown: int | None, margin: float | None = None) -> int | None:
    """The headline's whole number: it moves only once the smoothed value has passed a rounding boundary (n +- 0.5)
    by `margin` (the mode's shown_margin; the region's 0.3 by default), so it never flickers between 8 and 9. `per100`
    is the unrounded share: the whole number is rounded once."""
    margin = qk.value("region", "shown_margin") if margin is None else margin
    if per100 is None:
        return shown
    if shown is None or abs(per100 - shown) >= 0.5 + margin:
        return int(math.floor(per100 + 0.5))
    return shown


def _thin(items: list[dict], points: int) -> list[dict]:
    """At most `points` items, evenly spaced, always keeping the last."""
    points = max(2, min(int(points or 600), 5000))
    if len(items) <= points:
        chosen = items
    else:
        index = np.unique(np.linspace(0, len(items) - 1, points).round().astype(int))
        chosen = [items[i] for i in index]
    return [{k: item.get(k) for k in POINT_KEYS} for item in chosen]


# ---- pre-learning (learn samples ahead, e.g. a whole batch overnight) ------------------------------------------------
def app_running(config: dict) -> str | None:
    """Why the app of this configuration seems to be in use (its runtime file names a live engine, or something listens
    on its port), or None. Read only."""
    import socket
    from ..server.runtime_file import runtime_path
    try:
        record = json.loads(runtime_path(config).read_text())
        pid = int(record["pid"])
        portable.probe(pid)
        return f"an engine (pid {pid}) runs with the state dir {config['paths']['state_dir']}"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    port = int(config["server"]["port"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return f"something listens on port {port} (the app's)"
    return None


def main(argv=None, *, probe=None) -> int:
    """python -m sopa_companion.engine.qc learn [--detail] <sample.zarr> [...] [--config F]: learn each sample's model into
    the configured models folder ([qc] models_dir, else <data_dir>/qc-models: where the engine using them looks), one
    after another, with the engine's own GPU 0 rules. --detail also gives a model learned before Go deeper its details
    (deep.npz, a detail job: the same learning again, kept only when it matches; the mixing check's numbers stay the
    same). Prints one line per sample. Run it from the installed app's tree while the app is closed: it refuses while
    the app of the configured state dir runs (its runtime file, its port) or any engine owns the data dir (it holds that
    data dir's engine lock itself for the whole pass, so no engine starts on it meanwhile), and it writes nothing into
    the state dir: its log goes next to the models (<models>/prelearn.log). Exit 0 all learned (with
    --detail: with their details), 1 some failed, 2 refused."""
    import argparse
    import sys
    from .config import load_config
    from ..server.runtime_file import acquire_engine_lock
    parser = argparse.ArgumentParser(prog="python -m sopa_companion.engine.qc")
    parser.add_argument("command", choices=["learn"])
    parser.add_argument("samples", nargs="+")
    parser.add_argument("--config", default=None)
    parser.add_argument("--detail", action="store_true",
                        help="also add Go deeper's details (deep.npz) to models learned before it")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    busy = app_running(config)
    if busy:
        print(f"refused: {busy}: close the app first", file=sys.stderr)
        return 2
    try:
        engine_lock = acquire_engine_lock(config)
    except RuntimeError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    failed = 0
    try:
        root = models_root(config)
        models = QcModels(config, probe=probe, log_path=root / "prelearn.log")
        print(f"learning into {root} (the engine whose data dir is {config['paths']['data_dir']} uses these models)",
              flush=True)
        try:
            for path in args.samples:
                sid = models.register(path)
                view = models.ensure(sid, detail=args.detail)
                again = args.detail
                while True:
                    while view["state"] in ("waiting", "learning") or \
                            (args.detail and view["detail"]["state"] in ("queued", "learning")):
                        time.sleep(PRELEARN_POLL_S)
                        view = models.describe(sid)
                    # Learned just now (the first ensure queued a fit) by a trainer that kept no details: ask once.
                    # A fit whose own export of the details failed ("failed", model.json deep_error) gets one too.
                    if again and view["state"] == "ready" and view["detail"]["state"] in ("none", "failed"):
                        view, again = models.ensure(sid, detail=True), False
                        continue
                    break
                model = view.get("model") or {}
                detail = view.get("detail") or {}
                print(f"{path}: {view['state']}" + (f" ({view['reason']}: {view.get('message') or ''})"
                                                    if view["reason"] else "")
                      + (f" model {model.get('id')} in {model.get('seconds')} s" if model else "")
                      + (f"; details {detail.get('state')}" + (f" ({detail.get('reason')})" if detail.get("reason") else "")
                         if args.detail else ""), flush=True)
                failed += view["state"] != "ready" or (args.detail and detail.get("state") != "ready")
        finally:
            models.close()
    finally:
        os.close(engine_lock)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
