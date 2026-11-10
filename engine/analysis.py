"""Live clustering, 2-D UMAP and the mixing check: schedules frames onto the analysis worker (analysis/worker.py: the
GPU with RAPIDS, else the CPU) and keeps the results.

One worker process ([analysis] python: this environment by default; [analysis] backend) serves every run, one request
at a time. A request names the analyses ("ops") it wants of one frame, and the worker reads the frame's proseg count
(proseg's own count of each cell, gene by gene: analysis/counts_file.py) once for all:
  * "umap": clusters and a 2-D UMAP. Clusters & UMAP are always on: every active run of the engine (preparing,
    running, paused, finishing) has its newest counted frame analysed from its first frame on, with no switch on
    screen. Only a tool turns a run's analysis off (`configure(enabled=False)`, e.g. the harness measuring the mixing
    check alone); an ended run gets no automatic pass unless switched on, but every frame asked for is analysed;
  * "qc": the mixing check, for runs with the check on (`configure_qc`) whose sample has
    a learned model, which the QC provider names (`attach_qc`, engine/qc.py). Without a provider nothing is scored;
  * "qc_start": the check's start line, once per run and model: proseg's count at iteration 0 (frame 0's), for runs
    that started from cells alone (engine/qc_rules.py start_of "cells"); a start with nuclei has none;
  * "deeper": Go deeper: one cell's RNA explained by the run's model
    (its own, from neighbours, loose), for runs whose check is on and whose model holds the details (deep.npz). The
    deeper provider (`attach_deeper`, engine/deeper.py) keeps the answers: a cached one never reaches the worker.
Only frames with proseg's count are analysed, never an outline count in their place. Proseg-live counts a frame when
the engine asks for one (a counts token: engine/runs.py request_counts) and by itself at its first frame, upscales,
pauses, the hold, the finish and the end; its frame record then names the file (`counts_file`: in the run's spool on
the RAM disk, engine/counts_spool.py, or for the forced ones in the run folder). A run without proseg's count gets no
job at all, and describe says why: "old_proseg" (its proseg-live cannot keep it) or "no_counts" (made before it was
kept).
Jobs, in this order:
  0. explicit Go deeper requests (a window waits for them; the newest per run wins): ops ["qc", "deeper"] when the
     frame is not scored yet (the card and the window change together), else ["deeper"]; never "umap" (it would hold
     the answer for up to 2 s). While a window follows a cell (`follow_deeper`), every live job of its run also explains
     it (no extra job, no second read), and the counts it asks for carry the where export of the window's box
     (`follow_where`);
  1. explicit requests (the viewer scrubbing or stepping to a frame): at once, whatever the cadence; the newest request
     per run wins; a UMAP request also scores the frame when the run's check is on; then Compare's ops
     (`submit_compare`: worker op "compare"), in the order they came; each answers
     its Future (a cancelled one is dropped), and a failure never holds the scheduler back. A frame asked for (also of
     the check, Go deeper and Compare) is snapped to the nearest counted frame: the newest at or before it whose file is
     there or whose results are kept, else the oldest after it; its results are stored under that frame;
  2. the newest counted frame of each run. Clusters & UMAP: by default as fast as the worker goes on a GPU
     (`min_interval_s = 0`: the next analysis starts on the newest counted frame as soon as the last one ends; on the
     CPU, 0 means "auto"); a number of seconds, or "auto" (the run's cells / 15,000 s, between 0.5 and 6 s), spaces the
     starts of a run's automatic analyses instead. The mixing
     check rides along with every UMAP pass; on its own it goes back to back for runs under 20,000 cells and at most once
     a second for larger ones (the viewer shows their results at most once a second). Frames that arrive meanwhile are
     skipped, never queued. When a live pass of a sampling run takes the newest counted frame, the next count is asked
     for (one token outstanding at most; for spaced passes, as long before the next pass is due as its frame takes to
     come), so the next counted frame is there about an iteration after the pass ends; never sooner than ms.extract
     / [analysis] counts_extract_share after the newest counted frame (proseg's own counting stays within that share of
     its time), nor, with [analysis] counts_mb_s, than its bytes take at that pace: a token held back is asked for when
     its time comes, while a pass waits for the worker too; none while the run is paused, held or finishing (it counts
     that state itself), nor while its spool is full (said once in the log until tokens go again for a while). While
     a Watch runs on a run (the QC provider's `watching`) and its live pass is due, no background pass goes. A run
     whose pass found too few cells to cluster (the worker's reason "too_few_cells": a small region) gets no automatic
     pass until its newest frame holds materially more cells (TOO_FEW_GROWTH); describe says so (`unavailable`), as it
     says "no_transcripts" for a run made before live analysis;
  3. the start line, then backfill: when a run's check starts while it already has frames (the model arrived mid-run,
     or the check was switched on), at most 40 evenly spaced earlier counted frames whose file is there (the forced
     ones and the spool's reserve), coarse to fine, so its line starts at iteration 0. One such background pass goes
     after each live pass, and back to back when no live pass is due.
A frame asked of the check while its sample has no model yet (still learning) waits for the model, then goes first.
After each job of a run with a spool, its spooled files that no job needs any more go (`_prune`): it always keeps
those of the frames asked for or still to analyse (and any newer than the frames it read), of its where exports, of
Compare's run column and the newest with the start block; then its KEEP_NEWEST newest and an evenly spaced reserve of
up to BACKFILL_MAX (a check switched on later backfills from it), as long as they fit with two more frames in the run's
share of the spool: its cap, or what the engine's cap leaves beside the other runs' files when that is less. An ended
run gives way: once the other runs' files leave a sampling run less than its cap, or the engine's spool is full, it
keeps only the files a job needs (ended runs are pruned first). The forced files in the run folder always stay. A
test's own runs are never asked for counts nor pruned here (engine/qc_test.py does both).
Each run's Clusters & UMAP settings (resolution, n_neighbors, min_dist, min_counts, n_top_genes) can be changed while it
runs: a change starts a new generation of results, and new runs start with the settings last applied.
Results are written beside the run (analysis/pc1/<seq>.json + .cells.bin, <seq>-s<n> once the run's Clusters & UMAP
settings changed, files of earlier settings being removed; qc/<model8>.pc1/<seq>.qc.bin, start.*) and reused when asked
again, op by op: a cached UMAP result never skips the check, nor the reverse. The folders are proseg's count's
(counts_file.BASIS): results earlier versions made on outline counts (analysis/, qc/<model8>/) are never read. Each
scored frame's per-bin sums (the areas' input) reach the provider through a transient file in a private folder on a RAM
disk (/dev/shm), read back and deleted at once: never the run folder's disk, whose round trip (on a slow disk) slowed
whole-sample UMAP passes.
The worker keeps what it holds of at most 4 runs (2 on the CPU), least recently used out first. An ended run whose
check has scored what it scores by itself (start line, backfill, newest frame) and that gets no automatic Clusters &
UMAP pass is dropped from the worker once ("forget"; a later request loads it again; not while a Go deeper window
follows it or waits for an answer), and so is a deleted run. A run that is not live whose own frame a Compare op read
(which loads the run on the worker) is released again COMPARE_HOLD_S after its comparison's last op, the same way; the
count file of the frame a comparison takes (`compare_frame`) stays in the spool COMPARE_FILE_HOLD_S after it was
taken and after each op that read it. An
ended run without exported transcripts (made before live analysis existed) is not scored at all (describe_qc:
unavailable "no_transcripts").
The scheduler sleeps until it has work: a request, a setting or close() wakes it at once, the next automatic start is
timed, a run that is still active and caught up is checked for a newer frame every POLL_S (0.1 s), and with nothing to
do it looks for a new active run every IDLE_LOOK_S (1 s; never without the worker's environment). The worker starts with
the first active run (preparing included), not with its first frame, so its warm-up runs while the run prepares; a
Python that cannot run Clusters & UMAP at all (its worker's hello says so: packages missing, or libraries that do not
load) turns them off for the session, with one line in the log naming what is missing (`unavailable_why`).
What happened to the worker and to each run's analyses (the worker starting or failing, errors, a run switched off) is
kept for the page's Log button (`events`, `log_info`; engine/runlog.py writes the log).
"""
from __future__ import annotations

import bisect
from collections import deque
from concurrent.futures import CancelledError, Future, TimeoutError as FutureTimeout
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import weakref

import numpy as np

from .. import names, portable
from ..analysis.backend import BACKEND_ENV, CPU_CACHE_ENV, distributions
from ..analysis.counts_file import BASIS, count_of
from ..analysis.worker import DEFAULTS as WORKER_DEFAULTS
from . import children
from .config import cores
from .qc import QcSettings, _atomic, and_list, bus_number, gpu_probe
from .qc_rules import start_of
from .runs import TERMINAL, is_test_run

AUTO_INTERVAL = (15000.0, 0.5, 6.0)   # min_interval_s = "auto": the run's cells / 15,000 seconds, within 0.5 .. 6 s
DEFAULT_MIN_INTERVAL = 0              # [analysis] min_interval_s when unset: back to back on a GPU; the CPU worker
                                      # spaces its passes by "auto" instead (interval)
POLL_S = 0.1                          # how often an active run whose newest frame is analysed is checked for a newer one
IDLE_LOOK_S = 1.0                     # with nothing to do, how often the scheduler looks for a new active run (it is
                                      # analysed from its first frame on: a run prepares for seconds before that frame)
MAX_SLEEP_S = 3600.0                  # longest single sleep of the scheduler (it only re-checks when it wakes), so a huge
                                      # min_interval_s cannot exceed what Condition.wait accepts (threading.TIMEOUT_MAX)
# The mixing check: [qc] in config.toml (mixed_at, judge_min_counts, fit_min_counts, bin_um_detail, bin_um_tiled, checked
# by engine/qc.py QcSettings with the rest of [qc]); none of them is a UI option.
QC_LARGE_RUN = 20000                  # runs of this many cells or more: QC-only passes at most once per ...
QC_LARGE_INTERVAL_S = 1.0             # ... this many seconds (smaller runs: back to back)
QC_MODEL_POLL_S = 0.5                 # an active run whose check is on asks for its model again this often until it has one
BACKFILL_MAX = 40                     # earlier frames scored when a run's check starts mid-run (the spool's reserve)
QC_RETRY_S = (1.0, 60.0)              # after a failed QC pass: no automatic one of that run for 1 s, doubling to a minute
START_ATTEMPTS = 5                    # a start line that failed this often is given up (until the model changes)
# Proseg's own count of each frame. [analysis] counts_extract_share, counts_mb_s and counts_spool_mb (engine/config.py
# checks them; none is a UI option): a run's next counts token is asked for no sooner than its newest counted frame's
# ms.extract / counts_extract_share after that frame (proseg's own counting stays within that share of its time) nor,
# with counts_mb_s above 0, than that frame's bytes take at that pace; the spool's reserve stays within
# counts_spool_mb (a run's cap, engine/counts_spool.py).
COUNTS_EXTRACT_SHARE = 0.10
COUNTS_MB_S = 0.0
COUNTS_SPOOL_MB = 1024
KEEP_NEWEST = 8                       # a run's newest counted files its spool keeps while they fit (_keep)
NEWEST_LOOK = 16                      # _newest_counted looks at this many of a run's counted frames at a time
COUNTS_LEAD_S = (0.5, 3.0)            # a token's wait for its counted frame before one is learnt per run, and its most
COUNTS_RETRY_S = 1.0                  # a token the run store refused is not asked for again sooner
PRUNE_RETRY_S = 1.0                   # a run whose full spool stops its tokens is pruned at most this often
COUNTS_NOTE_S = 60.0                  # a full spool's stop is said again only once tokens went this long since the last
AHEAD_FRAMES = 32                     # a prune keeps the spooled files of this many frames around the newest it read
                                      # (their counts lines merge while it runs: none goes before it is analysed)
ASK_SLICE_S = 0.01                    # a pass waiting for the worker looks for a held-back token at most this often
# Proseg-live's history thinning (live.rs is_kept): every frame up to iteration 2,000, then every 5th, every 25th past
# 10,000, every 100th past 50,000. A counts token makes it write a frame whatever thinning would keep, so past 2,000 a
# token goes only once the run is a step past its newest counted frame (`thinning_allows`): at most twice the frames
# thinning keeps, whose files stay in the run folder, never one per iteration.
THINNED_FROM = 2000
THINNING_STEPS = ((10_000, 5), (50_000, 25), (math.inf, 100))
# A run without proseg's count: no job at all (describe's `unavailable`, its status; Go deeper's dropped requests).
COUNTS_REASONS = {"old_proseg": "this run's proseg-live cannot keep proseg's own count",
                  "no_counts": "this run was made before proseg's own count was kept"}
COUNTS_WAITS = {"run_cap": "this run's spool is full", "engine_cap": "the engine's spool is full",
                "low_space": "the disk the spool is on has less than twice a run's spool cap free",
                "spool_lost": "this run's spool folder could not be made"}
NO_COUNT = "No proseg count for this frame"
COUNTS_NAME = re.compile(r"^([0-9]{6,9})\.counts\.zst$")   # a counted frame's file as proseg-live names it
# The transient folder of each frame's per-bin sums: <RAM disk>/<prefix><pid>-* (names.BINS_PREFIXES: dead engines'
# folders of the previous version's prefix are swept too).
BINS_PREFIX, BINS_PREFIXES = names.BINS_PREFIX, names.BINS_PREFIXES
RAM_DISK = "/dev/shm"
EVENTS_KEPT = 300                     # the newest worker and run events kept for the log (`events`)
# A run whose pass found too few cells to cluster (the worker's reason "too_few_cells": a small region) gets no
# automatic pass until its newest frame holds at least this many times as many cells as the one that failed, and this
# many more (frame "cells"); explicit requests still go.
TOO_FEW_GROWTH = (1.25, 10)
# Go deeper (worker op "deeper"): the request block's neighbours and result
# version, the model's details file beside encoder.npz, and the back-off of a followed cell whose explanation failed
# (live passes stop carrying it for 1 s, doubling to a minute; explicit requests still go). A Go deeper failure never
# fails a pass nor holds the scheduler back: Clusters & UMAP and the check keep their cadence.
DEEPER_K = 10
DEEPER_V = 1
DEEP_FILE = "deep.npz"
DEEPER_RETRY_S = (1.0, 60.0)
# A followed cell is a lease: the window asks again at least every 30 s while it stays open on a live run, and a follow
# nobody renewed for this long ends by itself (a window closed before its answer came, a
# reloaded or closed page, left every later pass explaining a cell nobody looked at, and kept an ended run on the GPU).
DEEPER_FOLLOW_TTL_S = 75.0
# The box a followed cell's where exports cover is a lease of its own: Go deeper's where engine hands it over with each
# reply of a playing run's newer frame while the view is on (about once a pass: at most every 6 s on the CPU worker's
# own spacing), so a view turned off while its window stays open stops its exports this long after (follow_where).
WHERE_BOX_TTL_S = 20.0
# Compare on a run that is not live (ended, failed or deleted): its op loads the run on the GPU worker, which keeps it
# this long after the comparison's last op (its second op comes once its other columns are counted: several seconds on
# a whole sample), then lets it go as Go deeper's runs go (`_release_compared`). The count file of the run's frame a
# comparison takes stays in the spool longer, after it was taken and after each op that read it (the first time on a
# whole sample, the other columns can be counted for minutes); one such file per run, the newest comparison's.
COMPARE_HOLD_S = 60.0
COMPARE_FILE_HOLD_S = 600.0
# [analysis] backend: "auto" (the GPU when RAPIDS and a CUDA GPU are there, else the CPU), "gpu" (never the CPU) or
# "cpu" (never the GPU); passed to the worker (analysis/backend.py BACKEND_ENV), which chooses at its start.
BACKENDS = ("auto", "gpu", "cpu")
PRESTART_RETRY_S = 60.0               # a worker that could not start is not started early again for this long
# Clusters & UMAP settings a run can change while it runs (the UMAP pane's Advanced settings), in the order the page
# lists them and a request is checked: key -> (a whole number, lowest, highest, the page's step). Their defaults are the
# worker's (analysis/worker.py DEFAULTS) overlaid with the same keys of [analysis], checked by the same rules. A change
# starts a new generation of the run's results (`_paths`). [analysis] n_pcs passes to the worker as it is (FIXED_KEYS);
# whatever else the worker does stays its own. Genes used goes up to 10,000 on the GPU; on the CPU its PCA grows with
# the genes squared (40k cells: 1.5 s at 5,000 genes, 24-57 s at 10,000), so a worker on the CPU takes at most
# CPU_MAX_GENES (the same number as analysis/cpu.py MAX_GENES, repeated: the engine never imports the CPU module), sent
# to the page as the field's limit (`limits`) and held by the worker whatever it is sent.
SETTINGS = {"resolution": (False, 0.1, 3.0, 0.05), "n_neighbors": (True, 5, 50, 1), "min_dist": (False, 0.0, 1.0, 0.05),
            "min_counts": (True, 1, 500, 1), "n_top_genes": (True, 50, 10000, 50)}
CPU_MAX_GENES = 5000
# How a worker's not-ready hello starts its error when this Python cannot run Clusters & UMAP (analysis/worker.py
# main): the engine keeps the reason after it.
LASTING_PREFIX = "Clusters & UMAP cannot run in this Python environment: "
LIMITS = {key: {"min": low, "max": high, "step": step, "whole": whole}
          for key, (whole, low, high, step) in SETTINGS.items()}
FIXED_KEYS = ("n_pcs",)
RUN_SETTINGS = "settings.json"            # <run>/analysis/: the run's settings (all of them) and their generation
PREFS_FILE = "analysis-settings.json"     # <data_dir>/: the settings new runs start with (those not the defaults)
SETTINGS_VERSION = 1
SETTINGS_MAX_BYTES = 65536                # a settings file larger than this is not read
# A result's files in <run>/analysis/ (and the worker's .tmp while it writes them): <seq>.json, <seq>.cells.bin at
# generation 0, <seq>-s<n>.* at generation n. The sweep removes those of any other generation, nothing else.
RESULT_NAME = re.compile(r"^\d{6}(?:-s(\d+))?\.(?:json|cells\.bin)(?:\.tmp)?$")


class AnalysisError(Exception):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def min_interval_setting(value) -> float | None:
    """[analysis] min_interval_s: "auto" (None: from each run's cell count) or a number of seconds >= 0."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if isinstance(value, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f'[analysis] min_interval_s must be "auto" or a number of seconds >= 0, not {value!r}')
    return seconds


def worker_threads() -> str:
    """The worker's BLAS/OpenMP pools (OMP_NUM_THREADS and the like): at most 4 of the cores this engine may use (its
    host-side maths is small on the GPU; on the CPU these are its cores, beside proseg)."""
    return str(max(1, min(4, cores())))


def auto_interval(cells) -> float:
    """Minimum time between automatic analyses of a run with this many cells (seconds)."""
    per_second, low, high = AUTO_INTERVAL
    return min(max(float(cells or 0) / per_second, low), high)


def qc_settings(section: dict | None) -> dict:
    """[qc] mixed_at, judge_min_counts, fit_min_counts, bin_um_detail, bin_um_tiled, with the same rules and defaults as
    the rest of [qc] (engine/qc.py QcSettings, the one validator): a value refused there is refused here too."""
    s = QcSettings({"qc": section or {}})
    return {"mixed_at": s.mixed_at, "judge_min_counts": s.judge_min, "fit_min_counts": s.fit_min,
            "bin_um_detail": s.bin_um_detail, "bin_um_tiled": s.bin_um_tiled}


def counts_settings(section: dict | None) -> dict:
    """[analysis] counts_extract_share (a share above 0, at most 1), counts_mb_s (MB a second, 0 or more: 0 sets no
    pace by bytes) and counts_spool_mb (a run's spool cap in MB, at least 64), with their defaults, as floats; a bad
    value stops the engine (ValueError naming it), as engine/config.py says at start."""
    section = section or {}
    found = {}
    for key, default, low, high, words in (
            ("counts_extract_share", COUNTS_EXTRACT_SHARE, 0.0, 1.0, "a number above 0 and at most 1"),
            ("counts_mb_s", COUNTS_MB_S, 0.0, math.inf, "a number of MB a second, 0 or more"),
            ("counts_spool_mb", COUNTS_SPOOL_MB, 64.0, math.inf, "a number of MB, 64 or more")):
        value = section.get(key, default)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high or \
                (key == "counts_extract_share" and number == 0):
            raise ValueError(f"[analysis] {key} must be {words}, not {value!r}")
        found[key] = number
    return found


def settings_problem(key: str, value, limits: dict = LIMITS) -> str | None:
    """Why `value` cannot be the Clusters & UMAP setting `key` (SETTINGS) within `limits` (LIMITS, or a worker's own:
    AnalysisManager.limits), as the end of a sentence that starts with the key ("must be a number"), or None. A bool is
    no number, nor are NaN and the infinities; a whole-number setting takes 15.0 as 15."""
    whole = SETTINGS[key][0]
    low, high = limits[key]["min"], limits[key]["max"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    try:
        number = float(value)
    except OverflowError:                                 # a whole number beyond any float: out of range below
        number = math.inf if value > 0 else -math.inf
    if isinstance(value, float) and not math.isfinite(number):
        return "must be a number"
    if whole and math.isfinite(number) and not number.is_integer():
        return "must be a whole number"
    if not low <= number <= high:
        return f"must be from {low:g} to {high:g}"
    return None


def setting_value(key: str, value):
    """A checked value as the run keeps it: whole-number settings as int, the others as float (never rounded)."""
    return int(value) if SETTINGS[key][0] else float(value)


def check_settings(params, current: dict, limits: dict = LIMITS, defaults: dict | None = None) -> dict:
    """A request's `params` (some of the settings) merged over `current`, every value checked (settings_problem, within
    `limits`) and kept as setting_value makes it. A value equal to its default (`defaults`, checked within LIMITS when
    the engine started) is taken whatever `limits` say: a worker with a lower limit holds to its own whatever it is
    sent, and the page's Reset sends the defaults. AnalysisError 422 names the first problem: a key that is no setting
    first, then the settings in SETTINGS' order."""
    if not isinstance(params, dict):
        raise AnalysisError(f"params must be an object of settings ({', '.join(SETTINGS)})")
    for key in params:
        if key not in SETTINGS:
            *rest, last = SETTINGS
            raise AnalysisError(f"unknown setting {str(key)[:40]!r} (the settings are {', '.join(rest)} and {last})")
    merged = dict(current)
    for key in SETTINGS:
        if key in params:
            problem = settings_problem(key, params[key], limits)
            if problem and defaults is not None and settings_problem(key, params[key]) is None \
                    and _same(params[key], defaults.get(key)):
                problem = None
            if problem:
                raise AnalysisError(f"{key} {problem}")
            merged[key] = setting_value(key, params[key])
    return merged


def cluster_defaults(section: dict | None) -> dict:
    """The settings every run starts from: the worker's own (analysis/worker.py DEFAULTS) overlaid with the same keys of
    [analysis], checked by a request's rules: a bad value stops the engine at start (ValueError naming it)."""
    section = section or {}
    defaults = {}
    for key in SETTINGS:
        value = section.get(key, WORKER_DEFAULTS[key])
        problem = settings_problem(key, value)
        if problem:
            raise ValueError(f"[analysis] {key} {problem}, not {value!r}")
        defaults[key] = setting_value(key, value)
    return defaults


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


def same_settings(a: dict, b: dict) -> bool:
    """Whether two sets of settings agree on every setting (within 1e-9 of its value)."""
    return all(_same(a.get(key), b.get(key)) for key in SETTINGS)


def canonical(values: dict, defaults: dict) -> dict:
    """The settings of `values` that differ from `defaults`: what a run (or the next one) keeps of its own."""
    return {key: values[key] for key in SETTINGS if key in values and not _same(values[key], defaults.get(key))}


def _number(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def settings_words(values: dict) -> str:
    """Settings as the log says them: "resolution 1.5, n_neighbors 30"."""
    return ", ".join(f"{key} {_number(values[key])}" for key in SETTINGS if key in values)


def _highest_generation(folder: Path) -> int:
    """The highest settings generation among the result files in `folder` (RESULT_NAME; 0 without any, or no folder)."""
    highest = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                match = RESULT_NAME.match(entry.name)
                if match is not None and match.group(1):
                    highest = max(highest, int(match.group(1)))
    except OSError:
        pass
    return highest


def _read_settings_file(path: Path) -> dict:
    """A settings file as JSON (an object), refused (ValueError) when larger than SETTINGS_MAX_BYTES."""
    with open(path, "rb") as stream:
        raw = stream.read(SETTINGS_MAX_BYTES + 1)
    if len(raw) > SETTINGS_MAX_BYTES:
        raise ValueError(f"larger than {SETTINGS_MAX_BYTES // 1024} KB")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("v") != SETTINGS_VERSION:
        raise ValueError("not a settings file of this version")
    if not isinstance(data.get("settings"), dict):
        raise ValueError("it holds no settings")
    return data


def model_key(model_id) -> str:
    """A model's short name inside a run: the first 8 characters of its id (a hash)."""
    return (re.sub(r"[^A-Za-z0-9_-]", "", str(model_id)) or "model")[:8]


def qc_folder(run_dir, model_id) -> Path:
    """Where a run's results with one model live: <run>/qc/<model8>.pc1/, the folder of proseg's count
    (counts_file.BASIS), so the outline-era results earlier versions left in qc/<model8>/ are never read. Go deeper's
    history and disk cache (engine/deeper.py) read the same folder."""
    return Path(run_dir) / "qc" / f"{model_key(model_id)}.{BASIS}"


def coarse_to_fine(items: list) -> list:
    """The items reordered so that every prefix spreads over the whole list: first, middle, quarters, eighths, ..."""
    order, seen = [], set()
    step = 1 << max(0, (len(items) - 1).bit_length())
    while step:
        for i in range(0, len(items), step):
            if i not in seen:
                seen.add(i)
                order.append(items[i])
        step //= 2
    return order


def nearest_each(pool, wanted) -> list[int]:
    """For each value of `wanted` in turn, the member of `pool` nearest to it that no earlier value took (a tie: the
    lower), as a sorted list: evenly spaced picks among the frames there are."""
    free, picked = sorted(pool), []
    for value in wanted:
        if not free:
            break
        i = bisect.bisect_left(free, value)
        if i == len(free) or (i > 0 and value - free[i - 1] <= free[i] - value):
            i -= 1
        picked.append(free.pop(i))
    return sorted(picked)


def backfill_plan(newest: int, done, limit: int = BACKFILL_MAX, available=None) -> list[int]:
    """At most `limit` evenly spaced frames before `newest` with no result near them yet (within half the spacing),
    coarse to fine, so the mixing line covers the run from iteration 0 as soon as possible. `available`: the frames that
    can be scored (counted frames whose file is there: the forced ones and the spool's reserve); each target is then
    the nearest of them not taken yet. None: every frame."""
    if newest <= 0:
        return []
    pool = None if available is None else sorted({int(seq) for seq in available if 0 <= int(seq) < newest})
    if pool is not None and not pool:
        return []
    count = min(limit, newest if pool is None else len(pool))
    spacing = newest / count
    done = sorted(done)
    wanted = sorted({int(round(v)) for v in np.linspace(0, newest - 1, count)})
    targets = []
    for seq in (wanted if pool is None else nearest_each(pool, wanted)):
        i = bisect.bisect_left(done, seq - spacing / 2)
        if i < len(done) and done[i] <= seq + spacing / 2:
            continue
        targets.append(seq)
    return coarse_to_fine(targets)


def _soonest(wait, seconds):
    return seconds if wait is None else min(wait, seconds)


def _spooled_seq(path) -> int:
    """The frame number a counted frame's file name holds (NNNNNN.counts.zst), -1 for any other name."""
    match = COUNTS_NAME.match(os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def thinning_step(iteration) -> int:
    """How many iterations apart proseg-live keeps frames at `iteration` (live.rs is_kept): 1 up to THINNED_FROM."""
    if iteration <= THINNED_FROM:
        return 1
    return next(step for upto, step in THINNING_STEPS if iteration <= upto)


def transient_dir(root=None) -> Path:
    """A private folder for the frames' per-bin sums: on the RAM disk when there is one (the files are written by the
    worker and read back and deleted by this engine at once), else in the system's temporary folder. Folders that
    engines which are gone (a crash) left there, under either version's prefix, are removed first."""
    if root is None:
        root = RAM_DISK if os.path.isdir(RAM_DISK) and os.access(RAM_DISK, os.W_OK) else tempfile.gettempdir()
    root = Path(root)
    for prefix in BINS_PREFIXES:
        for folder in root.glob(prefix + "*"):
            try:
                if not portable.WINDOWS and folder.stat().st_uid != os.getuid():
                    continue                                # (Windows: the user's own temporary folder)
                portable.probe(int(folder.name[len(prefix):].split("-")[0]))
            except ProcessLookupError:                      # its engine is gone
                shutil.rmtree(folder, ignore_errors=True)
            except (ValueError, OSError):                   # alive (or not ours to judge): kept
                continue
    return Path(tempfile.mkdtemp(prefix=f"{BINS_PREFIX}{os.getpid()}-", dir=root))


class Job(tuple):
    """What the scheduler hands to the worker next. It is the pair (run id, seq) (and compares equal to it; seq is None
    for a start line), plus `ops` ("umap", "qc", "qc_start", "deeper", or "compare" alone), `kind` (request | live |
    start | backfill | compare), `qc`, the run's QcState when the job scores or explains anything, `deeper` {cell, start,
    explicit} when it explains a cell (Go deeper; `deeper_done` once its answer or failure went to the deeper provider),
    and `compare` (the worker request, its Future) for a Compare op; `count`, the count of its frame the worker reads
    ("averaged" | "sampled": counts_file.count_of of the frame's counts line as the job starts)."""

    def __new__(cls, run: str, seq, ops=("umap",), kind: str = "request", qc=None, deeper: dict | None = None,
                compare: tuple | None = None):
        job = super().__new__(cls, (run, seq))
        job.ops, job.kind, job.qc = tuple(ops), kind, qc
        job.deeper, job.deeper_done = deeper, False
        job.compare = compare
        job.count = None                  # the count its frame is read on (counts_file.count_of), set as it starts
        return job

    @property
    def run(self) -> str:
        return self[0]

    @property
    def seq(self):
        return self[1]


class QcState:
    """One run's mixing check with one model: where its files go and what the scheduler knows about them."""

    def __init__(self, model: str, path: str, directory: Path):
        self.model, self.path, self.dir = model, path, directory
        self.done: dict[int, str | None] = {}   # frames with a result -> the count it was read on ("averaged" |
                                              # "sampled"; None: found on disk, its <seq>.qc.json read when asked)
        self.start = "pending"                # pending | done | none (no prior to start from) | error (given up)
        self.backfill: list[int] = []         # earlier frames still to score, coarse to fine
        self.retry_at = 0.0                   # no automatic pass of the run before this (clock), after a failure
        self.failures = 0
        self.start_failures = 0
        self.unavailable: str | None = None   # an ended run's look at its files: "" (scorable) | "no_transcripts"
        self.released = False                 # the worker was told it may drop the run (an ended run, all scored)


class AnalysisManager:
    """The QC provider (`attach_qc`; engine/qc.py's QcManager) is any object with
      qc_model(run) -> {"id": model id (a hash; its first 8 characters name the run's qc/<model8>.pc1/ folder),
        "path": the model folder holding encoder.npz (or the file)} | None: the model that scores this run now, None
        while its sample has none ready. Called in the analysis thread for every run whose check is on (configure_qc)
        at every scheduling pass (up to ~10 times a second), never under this manager's lock: it must be cheap and must
        not wait on this manager.
      watching(run_id) -> bool (optional): whether a Watch (a trial of a setting) runs on the run now: its live passes
        then go before any background pass. Called as qc_model is, for runs whose check is on with a model.
      qc_result(record) (optional): every result, in the analysis thread, as soon as the worker has it:
        {"kind": "frame" | "start" | "error", "run", "seq" (None for the start line), "model", "dir" (the run's
         qc/<model8>.pc1/ folder), "source": "live" | "request" | "backfill" | "start", "single" (not the newest live
         frame: judge it without smoothing), "frame": {seq, iter, phase, t, live, voxel} (frames), "summary" (the worker's:
         analysis/worker.py qc_tables plus kind, seq, iter, model_id, cells, genes, has_start, ms ...), "bins" ({name:
         numpy array}: the per-bin table, worker.write_bins: the start line's from start.bins.npz, which is kept; a
         frame's from its transient file, read and deleted before the provider gets it, always on the averaged count;
         None when the worker wrote none),
         "cells" (<seq>.qc.bin or start.bin: u8 per cell, zstd), "count" (frames: the count the worker read, "averaged"
         or "sampled", counts_file.count_of of the frame's counts line), "cells_e" (a sampled frame's bytes on the
         averaged count, <seq>.qc.averaged.bin; else None), "start" (start.npz or None), "error" (kind "error")}.
        A frame counts as done here once qc_result has returned, for the count it was read on: a frame counted again
        (a request's count, then a pause's, the hold's, a finish's or the end's: the count proseg saves) is scored again
        while its
        counts file is there; a later session reuses it from disk only if qc/<model8>.pc1/<seq>.qc.json exists next to
        <seq>.qc.bin and names that count (the start line: start.json beside start.bin and start.npz), so the provider
        writes those files.
    The provider switches each run's check on and off (configure_qc: every run starts off) and calls qc_changed() when
    models change (a sample finished learning); active runs whose check is on are also asked again every
    QC_MODEL_POLL_S while their model is not ready.
    Clusters & UMAP need no switch: `runs.active()` names the active runs, and each has its newest counted frame
    analysed unless a tool switched it off (`enabled`, False; explicit settings only, the default being on).
    The run store (`runs`; engine/runs.py Runs) also tells each run's proseg count (counts_status: mode, reason, the
    tokens issued and served, the newest counted frame, when it came, its ms.extract and bytes, whether a token can be
    issued), takes the tokens asked for (request_counts, request_where) and prunes the spool (`runs.spool`); its runs'
    frame records name their counted frame's files (counts_file, and the counts line as `counts`).
    The deeper provider (`attach_deeper`; engine/deeper.py's DeeperManager) asks for cells (`request_deeper`,
    `follow_deeper`, `follow_where`, `stop_deeper`) and is any object with
      deeper_cached(run_id, seq, cell, model, explicit) -> bool: whether it holds that answer already (seq: a frame
        number or "start"; cell: label - 1; model: the run's model id); when True it hands the answer to whoever waits
        itself and the worker is not asked. Called in the analysis thread, never under this manager's lock.
      deeper_result(record): every answer or failure, in the analysis thread, never under this manager's lock: {run,
        seq (a frame number | "start"), iter (None for the start line), cell, model, explicit (asked for, or a followed
        cell riding along a live pass), result (the worker's "deeper" reply) | error (text) and reason (a
        word: worker, needs_detail, no_start, no_transcripts, no_counts, old_proseg, off, run_deleted, or None)}.
        A frame asked for is answered for the counted frame it snaps to: `seq` is that frame's (request_deeper returns
        it).
    A provider must never wait on this manager while holding its own lock."""

    def __init__(self, config: dict, runs, start: bool = True, probe=None, prestart: bool = True):
        settings = config.get("analysis", {})
        children.configure(config)
        self.python = settings.get("python", "")
        # What the engine found for [analysis] python (engine/config.py load_config, locate.py), for the Log.
        self.located = dict(((config.get("locate") or {}).get("analysis")) or {})
        self.gpu = settings.get("gpu", 0)
        self.module = settings.get("worker_module", "sopa_companion.analysis.worker")
        self.backend_setting = settings.get("backend", "auto")
        if self.backend_setting not in BACKENDS:
            raise ValueError(f'[analysis] backend must be "auto", "gpu" or "cpu", not {self.backend_setting!r}')
        # What the running worker said in its hello: its backend ("gpu", "cpu"; None before one), why not the GPU, and
        # its threads. The packages this Python lacks for Clusters & UMAP: from "auto" (known before any worker
        # starts, so the page never opens the UMAP pane only to close it), or from a worker's lasting hello, whose
        # reason is `lasting`: this Python cannot run them at all, and the worker is not started again this session.
        self.backend: str | None = None
        self.backend_why: str | None = None
        self.worker_threads: int | None = None
        auto = self.located.get("how") == "auto" and not self.python
        self.missing: list[str] = distributions(self.located.get("missing") or []) if auto else []
        self.lasting: str | None = None
        # The worker starts with the first active run, while it prepares (`_loop`); one that could not start is not
        # started early again for PRESTART_RETRY_S (a job still starts it as before).
        self.prestart = prestart
        self.prestart_failed = -math.inf
        # Where the CPU worker's numba keeps its compiled code: the engine's cache folder (a hand-made config without
        # one: the state folder's), made by the worker when it needs it.
        paths = config["paths"]
        self.cpu_cache = Path(paths.get("cache_dir") or Path(paths["state_dir"]) / "cache") / "numba"
        # [analysis]'s own worker settings (for the log); the settings every run starts from (`defaults`: the worker's
        # overlaid with these, checked: a bad one stops the engine here), and those passed to the worker as they are.
        self.params = {k: settings[k] for k in ("resolution", "min_counts", "n_pcs", "n_neighbors", "n_top_genes", "min_dist")
                       if k in settings}
        self.defaults = cluster_defaults(settings)
        self.fixed = {k: settings[k] for k in FIXED_KEYS if k in settings}
        self.min_interval = min_interval_setting(settings.get("min_interval_s", DEFAULT_MIN_INTERVAL))
        counts = counts_settings(settings)                  # proseg's count: the token budget and the spool's cap
        self.extract_share, self.counts_mb_s = counts["counts_extract_share"], counts["counts_mb_s"]
        self.spool_cap_mb = counts["counts_spool_mb"]
        self.qc_settings = qc_settings(config.get("qc"))
        self.log_path = Path(config["paths"]["state_dir"]) / "logs" / "analysis.log"
        self.runs = runs
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)      # requests, settings and close() wake the scheduler at once
        self.clock = time.monotonic
        self.enabled: dict[str, bool] = {}              # explicit switches (a tool's); a run not listed is on
        self.events: deque = deque(maxlen=EVENTS_KEPT)  # (unix time, run id or None, text) for the log
        self.noted: dict[str, str] = {}                 # the last failure noted per kind and run ("umap:<run>")
        self.worker_since: float | None = None          # when the running worker said it was ready (unix time)
        self.requested: dict[str, int] = {}
        self.started: dict[str, float] = {}             # when each run's last automatic analysis started (clock)
        self.run_params: dict[str, dict] = {}           # each run's settings that differ from the defaults
        self.gen: dict[str, int] = {}                   # each run's settings generation (none: 0, the defaults)
        # Runs whose folder held result files of a later generation than their settings file named when this engine
        # first read them (the file lost, or older than the files): that highest generation, which the run's next one is
        # numbered above, so a change never takes the name of files made with other settings.
        self.taken: dict[str, int] = {}
        self.loaded: set[str] = set()                   # runs whose settings this engine has read (_ensure_settings)
        self.sweeps: set[str] = set()                   # runs whose files of other generations the scheduler removes
        self.save_lock = threading.Lock()               # one writer of the settings files at a time (_save)
        self.results: dict[str, dict[int, dict]] = {}
        self.status: dict[str, dict] = {}
        self.too_few: dict[str, int] = {}               # runs held back for too few cells: the failed frame's cells
        self.exported: dict[str, bool] = {}             # whether a run (with its info) exported its transcripts
        self.version = 0
        self.qc_provider = None                         # attach_qc: which model scores a run, and who keeps the results
        self.qc_enabled: dict[str, bool] = {}           # each run's own switch for the mixing check
        self.qc_requested: dict[str, int] = {}
        self.qc_started: dict[str, float] = {}          # when each run's last automatic QC pass started (clock)
        self.qc_status: dict[str, dict] = {}
        self.qc_runs: dict[str, QcState] = {}
        self.qc_version = 0
        self.bg_turn = False                            # a background QC pass may go before the next live one
        self.deeper_provider = None                     # attach_deeper: who keeps Go deeper's answers
        self.deeper_requested: dict[str, tuple] = {}    # run -> (seq | "start", cell): the newest explicit request
        self.deeper_follow: dict[str, int] = {}         # run -> the cell every live pass of the run also explains
        self.deeper_follow_at: dict[str, float] = {}    # run -> when its follow was last asked for (the lease)
        self.deeper_turn = False                        # an explicit Go deeper job just went: a UMAP request goes next
        self.deeper_retry: dict[str, tuple] = {}        # run -> (clock, failures): a followed cell's back-off
        self.deeper_dropped: list[dict] = []            # failures found under the lock, handed over after it
        self.to_forget: set[str] = set()                # runs the GPU worker may drop (sent by the scheduler's thread)
        self.compare_queue: deque = deque()             # (worker request, Future): Compare's ops (submit_compare)
        self.compared: dict[str, float] = {}            # runs that are not live a Compare op loaded -> when they may go
        # Proseg's count. Each run's frame its last live pass took (its newer counted frames are still to analyse); the
        # box a followed Go deeper window wants where exports of (follow_where); tokens left to ask for after the lock
        # (run -> a where box or None: `_ask_counts`) and runs whose full spool is pruned then; tokens held back until
        # a time (run -> (clock, run, Clusters & UMAP want frames, its QcState): `_ask_due` asks them, during a pass
        # too); when each run's last token was asked for (its wait for the counted frame is learnt: `leads`) and when
        # the next may be; when a full spool last stopped a run's tokens; a run's newest counted frame whose file is
        # there, kept until its frames change; the frame Compare took, kept in the spool until then; the runs whose
        # spool was pruned here, and the ended ones among them that keep only the files a job needs (`_squeeze`).
        self.live_seq: dict[str, int] = {}
        self.where_box: dict[str, list] = {}
        self.where_box_at: dict[str, float] = {}
        self.to_ask: dict[str, list | None] = {}
        self.to_prune: set[str] = set()
        self.ask_plan: dict[str, tuple] = {}
        self.asked: dict[str, float] = {}
        self.leads: dict[str, float] = {}
        self.ask_after: dict[str, float] = {}
        self.pruned: dict[str, float] = {}
        self.stopped_at: dict[str, float] = {}
        self.newest_seen: dict[str, tuple] = {}
        self.compare_files: dict[str, dict[int, float]] = {}
        self.spooled: set[str] = set()
        self.squeezed: set[str] = set()
        self.bins_dir: Path | None = None               # transient per-bin sums of scored frames (transient_dir)
        self.probe = probe or (lambda: gpu_probe(children.gpu(int(self.gpu))))   # nvidia-smi's [analysis] gpu (its bus)
        self.process: subprocess.Popen | None = None
        self.replies: queue.Queue = queue.Queue()
        self.failed_at = 0.0
        self.closed = False
        # The settings new runs start with: the last ones applied, kept in the data folder (an engine without one, as in
        # tests, keeps them in memory only); those that differ from the defaults, as the file was last written.
        data_dir = (config.get("paths") or {}).get("data_dir")
        self.prefs_path = Path(data_dir) / PREFS_FILE if data_dir else None
        self.prefs = self._read_prefs()
        self.prefs_written = dict(self.prefs)
        self.thread = threading.Thread(target=self._loop, name="analysis", daemon=True)
        if start:
            self.thread.start()

    # ---- public --------------------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether Clusters & UMAP can run here: [analysis] python is a file, and no worker said that it cannot run them
        at all (`lasting`)."""
        return bool(self.python) and Path(self.python).is_file() and self.lasting is None

    def unavailable_why(self) -> str:
        """Why Clusters & UMAP cannot run here, in plain words for every refusal and the Log: the packages this Python
        lacks; else a worker's lasting reason; else [analysis] python off, or not there."""
        if self.missing:
            return f"this Python environment lacks {and_list(self.missing)}"
        if self.lasting is not None:
            return self.lasting
        if not self.python or self.python == "off":
            return "python under [analysis] is off"
        return f"the python set under [analysis] ({self.python}) is not there"

    def refusal(self, what: str) -> AnalysisError:
        """The 503 of a request that needs Clusters and the UMAP where they cannot run (`what`: the feature asked
        for)."""
        if what == "Clusters and the UMAP":
            return AnalysisError(f"{what} can't run here: {self.unavailable_why()}", 503)
        return AnalysisError(f"{what} needs Clusters and the UMAP, which can't run here: {self.unavailable_why()}", 503)

    def limits(self) -> dict:
        """The settings' limits as the page's fields take them (LIMITS), with the running worker's own: on the CPU,
        Genes used stops at CPU_MAX_GENES. Before the worker's first hello the GPU's stand."""
        limits = {key: dict(limit) for key, limit in LIMITS.items()}
        if self.backend == "cpu":
            limits["n_top_genes"]["max"] = CPU_MAX_GENES
        return limits

    def interval(self, run) -> float:
        """Minimum time between the starts of two automatic analyses of this run (seconds): [analysis]
        min_interval_s; on the CPU worker "auto" also when that is 0 (back to back there would keep its cores busy for
        the whole run)."""
        cells = (run.info or {}).get("cells")
        if self.min_interval is None or (self.backend == "cpu" and self.min_interval == 0):
            return auto_interval(cells)
        return self.min_interval

    @staticmethod
    def qc_interval(run) -> float:
        """Minimum time between the starts of two QC-only passes of this run (seconds)."""
        return QC_LARGE_INTERVAL_S if ((run.info or {}).get("cells") or 0) >= QC_LARGE_RUN else 0.0

    def describe(self, run_id: str) -> dict:
        """The run's analysis for the page (GET /runs/{id}/analysis, the WebSocket's pushes); `unavailable` names a
        reason no new attempt can fix by itself, the first that holds of "old_proseg" (its proseg-live cannot keep
        proseg's own count: an older one), "no_counts" (a run made before proseg's own count was kept), "no_transcripts"
        (a run made before live analysis exported its transcripts) and "too_few_cells" (its last pass found too few
        cells to cluster: a small region), else None; with either of the first two the run gets no job at all and its
        status is {"state": "unavailable", "reason": it}. `seqs`: the counted frames analysed (a frame asked for is
        analysed as the counted frame it snaps to: the page shows the newest at or before the frame on screen);
        `requested`: the counted frame an explicit request waits for (None).
        `settings` (the run's Clusters & UMAP settings), `defaults`, `limits` (SETTINGS: {min, max, step, whole}, the
        running worker's own: `limits()`), `changed` (they differ from the defaults), `generation` (results of other
        settings are never served) and `params` (what the worker gets: the settings and [analysis]'s own n_pcs);
        `backend` (the running worker's: "gpu", "cpu", None before its hello) and `missing` (the packages this Python
        lacks for Clusters & UMAP, when it cannot run them)."""
        try:
            run = self.runs.get(run_id)
            interval = round(self.interval(run), 2)
            self._ensure_settings(run_id, run)
        except Exception:
            run, interval = None, None
        exported = self._exported(run_id, run) if run is not None else None
        reason = self._counts_reason(run_id) if run is not None else None
        with self.lock:
            done = sorted(self.results.get(run_id, {}))
            unavailable = reason or ("no_transcripts" if exported is False else
                                     "too_few_cells" if run_id in self.too_few else None)
            status = {"state": "unavailable", "reason": reason} if reason else \
                self.status.get(run_id, {"state": "idle"})
            settings = self.settings_of(run_id)
            return {"available": self.available, "enabled": self.enabled.get(run_id, True),
                    "status": status, "latest_seq": done[-1] if done else None,
                    "seqs": done[-2000:], "interval_s": interval, "unavailable": unavailable,
                    "requested": self.requested.get(run_id),
                    "params": {**self.fixed, **settings}, "settings": settings, "defaults": dict(self.defaults),
                    "limits": self.limits(), "changed": bool(self.run_params.get(run_id)),
                    "generation": self.gen.get(run_id, 0), "backend": self.backend, "missing": list(self.missing)}

    def settings_of(self, run_id: str) -> dict:
        """The run's Clusters & UMAP settings: the defaults, with what it changed."""
        return {**self.defaults, **self.run_params.get(run_id, {})}

    def generation(self, run_id: str) -> int:
        """The run's settings generation (0 while it has the defaults it never changed): what its results now are."""
        self._ensure_settings(run_id)
        with self.lock:
            return self.gen.get(run_id, 0)

    def run_created(self, meta: dict):
        """A run the engine just made (POST /runs) starts with the settings last applied (`prefs`), as a generation
        of its own; with none remembered, or where Clusters & UMAP cannot run, it keeps the defaults."""
        run_id = str(meta.get("id") or "")
        if not run_id or not self.available:
            return
        run = self.runs.get(run_id)
        self._ensure_settings(run_id, run)
        with self.lock:
            values = {**self.defaults, **self.prefs}
            if not self.prefs or same_settings(values, self.settings_of(run_id)):
                return
            self._new_generation(run_id, values, "Clusters & UMAP start with the settings last applied: "
                                 + settings_words(canonical(values, self.defaults)))
            self.version += 1
            self.wake.notify_all()
        self._save(run_id)

    def _exported(self, run_id: str, run) -> bool | None:
        """Whether the run exported its transcripts (proseg-live writes them before info.json, so a run with its info
        has them or never will): looked at once per run; None before its info exists."""
        known = self.exported.get(run_id)
        if known is None and run.info:
            known = self.exported[run_id] = (run.live / "transcripts.bin.gz").exists()
        return known

    def configure(self, run_id: str, enabled: bool | None = None, seq: int | None = None, params: dict | None = None):
        """`enabled`: the run's switch, on by default (the page has none; a tool switches a run off, or on again);
        `seq`: one frame to analyse now, as the counted frame it snaps to (`_snap`: the reply's `requested` says which,
        also once the scheduler took it; none for a run without proseg's count); `params`: Clusters & UMAP settings
        (some or all of SETTINGS, merged over
        the run's): settings that differ from the run's start a new generation of its results (`_new_generation`), and
        new runs start with them from now on, changed or not (so the page's Reset also holds for new runs). A refused
        request changes nothing. Settings are checked within the running worker's limits (`limits`: Genes used stops
        lower on the CPU), a value equal to its default always taken (the worker holds to its own limit)."""
        if not self.available:
            raise self.refusal("Clusters and the UMAP")
        run = self.runs.get(run_id)
        self._ensure_settings(run_id, run)
        # Everything is checked before anything changes: a refused request leaves the run as it was (and the scheduler,
        # which sleeps until woken, never misses a change that was made).
        given = check_settings(params, {}, self.limits(), self.defaults) if params else None
        seq = self._frame_number(run, seq)
        if seq is not None and self._counts_reason(run_id) is not None:
            seq = None                                        # no proseg count: no job at all (describe says why)
        if seq is not None:                                   # results kept count only while the settings stay theirs
            seq = self._snap(run, seq, None if given else self._umap_kept(run_id, run))
        run_changed = prefs_changed = False
        with self.lock:
            if enabled is not None:
                if bool(enabled) != self.enabled.get(run_id, True):
                    self._note(f"Clusters & UMAP switched {'on' if enabled else 'off'} through the API", run_id)
                    # The page has no switch and follows this one: the WebSocket pushes every status but "ready" (with
                    # the switch), so a change sets the run's status to idle and the page hears of it at once.
                    self.status[run_id] = {"state": "idle"}
                self.enabled[run_id] = bool(enabled)
            if given is not None:
                current = self.settings_of(run_id)
                values = {**current, **given}
                if not same_settings(values, current):
                    note = "Clusters & UMAP settings changed: " + ", ".join(
                        f"{key} {_number(current[key])} -> {_number(values[key])}" for key in SETTINGS
                        if not _same(current[key], values[key]))
                    self._new_generation(run_id, values, note + ("" if canonical(values, self.defaults) else
                                                                 "; back to the defaults"))
                    run_changed = True
                remembered = canonical(values, self.defaults)
                if remembered != self.prefs:
                    self.prefs, prefs_changed = remembered, True
            if seq is not None:
                self.requested[run_id] = seq
            self.version += 1
            self.wake.notify_all()
        if run_changed or prefs_changed:
            self._save(run_id if run_changed else None)
        reply = self.describe(run_id)
        if seq is not None:                                   # the scheduler, just woken, may have taken it already
            reply["requested"] = seq
        return reply

    def summary(self, run_id: str, seq: int, generation: int | None = None) -> dict:
        """A frame's result of the run's current settings; `generation`: only when that is still theirs (404 for any
        other: an earlier generation's files may stay on disk until they are swept, never served)."""
        run = self.runs.get(run_id)
        gen = self._current(run_id, run, generation)
        with self.lock:
            value = self.results.get(run_id, {}).get(seq) if self.gen.get(run_id, 0) == gen else None
        if value is None:
            path = self._paths(run, seq, gen)[0]
            if not path.exists():
                raise AnalysisError("Not analysed yet", 404)
            value = json.loads(path.read_text())
        return value

    def cells_path(self, run_id: str, seq: int, generation: int | None = None) -> Path:
        """A frame's per-cell file of the run's current settings (`generation` as for summary)."""
        run = self.runs.get(run_id)
        path = self._paths(run, seq, self._current(run_id, run, generation))[1]
        if not path.exists():
            raise AnalysisError("Not analysed yet", 404)
        return path

    def _current(self, run_id: str, run, generation: int | None) -> int:
        """The run's settings generation, checked against the one a request names (404 when another)."""
        self._ensure_settings(run_id, run)
        with self.lock:
            gen = self.gen.get(run_id, 0)
        if generation is not None and generation != gen:
            raise AnalysisError("Not analysed yet", 404)
        return gen

    def log_info(self, run_id: str | None = None) -> dict:
        """What the log says about live analysis (engine/runlog.py): the worker's settings and state, the Clusters &
        UMAP settings runs start from (`defaults`, with [analysis]'s own n_pcs: `fixed`) and those new runs start with
        (`new_runs`), its events (the run's own and the worker's when `run_id` is given), and for the run its switch,
        status, settings (`settings`, `changed`, `generation`; `params`: what the worker gets) and every result kept
        here with its timings (`results`: [(seq, summary)], oldest first); the worker's backend, why, threads and the
        setting (`backend`, `backend_why`, `threads`, `backend_setting`), and why Clusters & UMAP cannot run here
        (`why_not`, None when they can) with the packages missing. A copy, read under the lock; nothing is computed or
        started."""
        process = self.process
        if run_id is not None:
            try:
                self._ensure_settings(run_id)
            except Exception:                                 # a deleted run: what memory holds
                pass
        with self.lock:
            # One copy first, in C (atomic under the GIL): _note appends from the scheduler's thread without the lock, and
            # a deque that grows while Python code iterates it raises.
            events = tuple(self.events)
            info = {"available": self.available, "python": self.python, "located": dict(self.located), "gpu": self.gpu,
                    "module": self.module, "min_interval": self.min_interval, "params": dict(self.params),
                    "backend": self.backend, "backend_why": self.backend_why, "threads": self.worker_threads,
                    "backend_setting": self.backend_setting, "missing": list(self.missing),
                    "why_not": None if self.available else self.unavailable_why(),
                    "defaults": dict(self.defaults), "fixed": dict(self.fixed), "new_runs": {**self.defaults, **self.prefs},
                    "worker": {"pid": process.pid if process is not None and process.poll() is None else None,
                               "since": self.worker_since, "log": str(self.log_path)},
                    "runs": {"switched_off": sorted(r for r, on in self.enabled.items() if not on),
                             "with_results": len(self.results)},
                    "events": [e for e in events if run_id is None or e[1] in (None, run_id)]}
            if run_id is not None:
                settings = self.settings_of(run_id)
                info["run"] = {"enabled": self.enabled.get(run_id, True), "explicit": run_id in self.enabled,
                               "status": dict(self.status.get(run_id, {"state": "idle"})),
                               "params": {**self.fixed, **settings}, "settings": settings,
                               "changed": bool(self.run_params.get(run_id)), "generation": self.gen.get(run_id, 0),
                               "interval_s": None, "requested": self.requested.get(run_id),
                               "results": sorted(self.results.get(run_id, {}).items()),
                               "deeper": {"follow": self.deeper_follow.get(run_id),
                                          "requested": self.deeper_requested.get(run_id),
                                          "failures": self.deeper_retry.get(run_id, (0.0, 0))[1]}}
        if run_id is not None:
            try:
                info["run"]["interval_s"] = self.interval(self.runs.get(run_id))
            except Exception:                                 # a deleted run: no cadence to tell
                pass
        return info

    def _note(self, text: str, run_id: str | None = None, once: str | None = None):
        """An event for the log (`events`, the newest EVENTS_KEPT). `once`: a key under which the same text is noted
        only once in a row (a failure repeated by every pass; a success clears the key)."""
        if once is not None:
            if self.noted.get(once) == text:
                return
            self.noted[once] = text
        self.events.append((time.time(), run_id, text))

    # ---- Clusters & UMAP settings --------------------------------------------------------------------------------
    def _read_prefs(self) -> dict:
        """The settings new runs start with, as the last change left them (PREFS_FILE: those that differ from the
        defaults), read once at start: {} without the file, and with one that cannot be read (said once in the log)."""
        if self.prefs_path is None:
            return {}
        try:
            data = _read_settings_file(self.prefs_path)
            return canonical(check_settings(data.get("settings"), self.defaults), self.defaults)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, AnalysisError) as exc:
            self._note(f"The Clusters & UMAP settings for new runs could not be read ({exc}): new runs start with the "
                       "defaults", once="settings-prefs")
            return {}

    def _ensure_settings(self, run_id: str, run=None):
        """The run's settings and generation as its analysis/settings.json keeps them, read once per run in this engine
        (every way in calls this first): without the file the defaults at generation 0, and so with a file that cannot
        be read (said once in the log). A run whose settings differ from the defaults is never at generation 0, so a
        lost file never shows results of other settings: the files of every other generation are swept (also what an
        engine that was killed left), and the next change is numbered above the highest of them (`taken`), so results
        of the new settings never take the names of files the sweep has not removed yet. Read outside the lock; kept
        unless another thread was first."""
        if run_id in self.loaded:
            return
        if run is None:
            run = self.runs.get(run_id)
        values, generation = dict(self.defaults), 0
        highest = _highest_generation(self._results_dir(run))
        try:
            data = _read_settings_file(run.dir / "analysis" / RUN_SETTINGS)
            stored = check_settings(data.get("settings"), self.defaults)
            generation = data.get("generation")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
                raise ValueError("its generation is not a whole number >= 0")
            if generation == 0 and canonical(stored, self.defaults):
                raise ValueError("changed settings at generation 0")
            values = stored
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AnalysisError) as exc:
            values, generation = dict(self.defaults), 0
            self._note(f"The Clusters & UMAP settings of this run could not be read ({exc}): the defaults are used",
                       run_id, once=f"settings:{run_id}")
        with self.lock:
            if run_id in self.loaded:
                return
            changed = canonical(values, self.defaults)
            if changed:
                self.run_params[run_id] = changed
            else:
                self.run_params.pop(run_id, None)
            if generation:
                self.gen[run_id] = generation
            else:
                self.gen.pop(run_id, None)
            if highest > generation:
                self.taken[run_id] = highest
            else:
                self.taken.pop(run_id, None)
            self.loaded.add(run_id)
            self.sweeps.add(run_id)
            self.wake.notify_all()

    def _new_generation(self, run_id: str, values: dict, note: str):
        """The run's settings become `values`, a new generation of its results (numbered above any its folder's files
        had when first read: `taken`): what was shown of the earlier ones is dropped (results, status, the too-few
        hold, the cadence's wait), their files are swept, and the frame asked for or the newest one goes next with these
        settings. Lock held; the caller bumps `version` and saves."""
        changed = canonical(values, self.defaults)
        if changed:
            self.run_params[run_id] = changed
        else:
            self.run_params.pop(run_id, None)
        self.gen[run_id] = max(self.gen.get(run_id, 0), self.taken.pop(run_id, 0)) + 1
        self.results.pop(run_id, None)
        self.status[run_id] = {"state": "idle"}
        self.too_few.pop(run_id, None)
        self.started.pop(run_id, None)
        self.noted.pop(f"umap:{run_id}", None)
        self.sweeps.add(run_id)
        self._note(note, run_id)

    def _save(self, run_id: str | None = None):
        """Write what is current when this writer's turn comes (one at a time, so two changes close together leave the
        files on the later one), each file atomically: the run's settings and generation (`run_id`; only into the run's
        folder as it is, never made again for a run deleted meanwhile) and the settings new runs start with, when they
        changed since last written. A write that fails is said in the log; the settings in memory stand."""
        with self.save_lock:
            with self.lock:
                prefs = dict(self.prefs)
                settings = self.settings_of(run_id) if run_id is not None else None
                generation = self.gen.get(run_id, 0) if run_id is not None else 0
            try:
                if run_id is not None:
                    try:
                        run = self.runs.get(run_id)
                    except Exception:                         # deleted meanwhile: nothing to keep
                        run = None
                    if run is not None and run.dir.is_dir():
                        try:
                            (run.dir / "analysis").mkdir(exist_ok=True)
                            _atomic(run.dir / "analysis" / RUN_SETTINGS, json.dumps(
                                {"v": SETTINGS_VERSION, "settings": settings, "generation": generation}).encode())
                        except FileNotFoundError:             # its folder went meanwhile: it stays gone
                            pass
                if self.prefs_path is not None and prefs != self.prefs_written:
                    self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic(self.prefs_path, json.dumps({"v": SETTINGS_VERSION, "settings": prefs}).encode())
                    self.prefs_written = prefs
            except OSError as exc:
                self._note(f"The Clusters & UMAP settings could not be saved ({exc})", once="settings-save")

    def _sweep_generations(self, run_id: str):
        """Remove the run's result files (analysis/pc1/) of every generation but its current one (RESULT_NAME: nothing
        else; files and links only, a link itself and never what it points to). In the scheduler's thread,
        which also runs the passes, so it never meets a pass writing."""
        try:
            run = self.runs.get(run_id)
        except Exception:                                     # deleted: its folder went with it
            return
        with self.lock:
            current = self.gen.get(run_id, 0)
        try:
            entries = list(os.scandir(self._results_dir(run)))
        except OSError:
            return
        for entry in entries:
            match = RESULT_NAME.match(entry.name)
            if match is None or int(match.group(1) or 0) == current:
                continue
            try:
                if entry.is_file(follow_symlinks=False) or entry.is_symlink():
                    os.unlink(entry.path)
            except OSError:
                pass

    def _umap_outcome(self, run_id: str, seq: int, gen: int, frame_cells, summary: dict | None = None,
                      error: str | None = None, reason: str | None = None, count: str | None = None) -> bool:
        """What a pass's Clusters & UMAP decided, set under one lock: its result (`summary`), or its error status (with
        the `count` the pass read when it was the count proseg saves, "sampled"; none: the averaged count: the same frame
        is not tried again by itself on that count) and, for too few cells, the hold on automatic passes. A pass whose
        settings changed meanwhile (`gen` is not the run's generation any more) sets nothing: False, and its files are
        the caller's to remove."""
        with self.lock:
            if gen != self.gen.get(run_id, 0):
                return False
            if summary is not None:
                self._store_locked(run_id, seq, summary)
            else:
                if reason == "too_few_cells":                 # held back until the run holds more cells (_too_few)
                    self.too_few[run_id] = int(frame_cells or 0)
                self._status_locked(run_id, {"state": "error", "seq": seq, "error": error,
                                             **({"reason": reason} if reason else {}),
                                             **({"count": count} if count == "sampled" else {})})
            return True

    # ---- the mixing check ----------------------------------------------------------------------------------
    def attach_qc(self, provider):
        """Set the QC provider (see the class docstring) and look at every run again."""
        with self.lock:
            self.qc_provider = provider
            self.qc_version += 1
            self.wake.notify_all()

    def qc_changed(self):
        """Models changed (a sample finished learning, or learned again): ask the provider again at once."""
        with self.lock:
            self.qc_version += 1
            self.wake.notify_all()

    # ---- Go deeper (engine/deeper.py) ----------------------------------------------------------------------------
    def attach_deeper(self, provider):
        """Set the deeper provider (see the class docstring)."""
        with self.lock:
            self.deeper_provider = provider
            self.wake.notify_all()

    def request_deeper(self, run_id: str, seq, cell: int, follow: bool | None = None):
        """Explain cell `cell` (label - 1) of frame `seq` (a frame number, or "start": the start line) now, before any
        other request; the newest request per run wins. A frame number is snapped to the nearest counted frame whose
        file is there (`_snap`): the answer is that frame's, and this returns it (the frame number, or "start"). `follow`:
        True also explains that cell with every live pass of the run from now on, False stops following, None leaves
        it. Only runs whose check is on are explained (a request for one switched off meanwhile is dropped), and only
        runs with proseg's count (another's is dropped, its reason handed over: "no_counts", "old_proseg"). Refused
        requests (503 without the worker's environment, 404 unknown run or frame, 422 not a frame number or cell) change
        nothing."""
        if not self.available:
            raise self.refusal("Go deeper")
        run = self.runs.get(run_id)
        if seq != "start":
            seq = self._frame_number(run, seq)
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise AnalysisError("cell must be a cell number (label - 1)")
        if seq != "start":
            found = self._snap(run, seq)
            seq = seq if found is None else found             # none yet: the scheduler says why when it drops it
        with self.lock:
            self.deeper_requested[run_id] = (seq, cell)
            self._set_follow(run_id, cell if follow else None, follow is None)
            self.wake.notify_all()
        return seq

    def snap_deeper(self, run_id: str, seq, kept=None):
        """The frame Go deeper answers frame `seq` with, as request_deeper snaps it (`_snap`: the newest counted frame at
        or before it whose file is there, or whose answer `kept(seq)` says is kept, else the oldest such after it): a
        frame without proseg's count of its own (between counted frames, its file pruned from the spool, or gone with
        an engine that closed) is answered as that counted frame, under its seq, so the caller names and finds the answer
        there. "start" as it is, and `seq` itself while there is none (the scheduler says why when it drops the
        request). 404 for an unknown frame, 422 for one that is not a frame number."""
        if seq == "start":
            return seq
        run = self.runs.get(run_id)
        seq = self._frame_number(run, seq)
        found = self._snap(run, seq, kept)
        return seq if found is None else found

    def follow_deeper(self, run_id: str, cell: int | None):
        """The cell every live pass of the run also explains (a window open on it); None stops following."""
        with self.lock:
            self._set_follow(run_id, cell, False)
            self.wake.notify_all()

    def follow_where(self, run_id: str, box) -> None:
        """The box (µm: x0, y0, x1, y1) around the followed cell whose where export the Go deeper window wants
        (engine/deeper_where.py, while it follows a live cell with the view on); None clears it. While the run follows
        a cell and has a box, the counts tokens asked for it carry that where export (runs.request_where), so its pink
        dots stay exact at every counted frame. Ignored for a run that follows no cell; cleared with the follow
        (stop_deeper, follow_deeper(None), its lease running out), and when no box came for WHERE_BOX_TTL_S (the view
        was turned off). 422 for a box that is not four finite numbers with x1 > x0 and y1 > y0."""
        if box is not None:
            try:
                box = [float(v) for v in box]
            except (TypeError, ValueError, OverflowError):
                box = []
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise AnalysisError("box must be four numbers x0, y0, x1, y1 (µm) with x1 > x0 and y1 > y0")
        with self.lock:
            if box is None or run_id not in self.deeper_follow:
                self.where_box.pop(run_id, None)
                self.where_box_at.pop(run_id, None)
            else:
                self.where_box[run_id] = box
                self.where_box_at[run_id] = self.clock()

    def stop_deeper(self, run_id: str):
        """The window closed: no more following, and a request not started yet is dropped."""
        with self.lock:
            self._set_follow(run_id, None, False)
            self.deeper_requested.pop(run_id, None)
            self.wake.notify_all()

    def _set_follow(self, run_id: str, cell: int | None, keep: bool):
        """Lock held. `keep`: leave the run's following as it is. Following a cell (again) renews its lease
        (DEEPER_FOLLOW_TTL_S); following none clears the where box too (follow_where)."""
        if keep:
            return
        if cell is not None:
            self.deeper_follow_at[run_id] = self.clock()
        if self.deeper_follow.get(run_id) == cell:
            return
        if cell is None:
            self.deeper_follow.pop(run_id, None)
            self.deeper_follow_at.pop(run_id, None)
            self.where_box.pop(run_id, None)
            self.where_box_at.pop(run_id, None)
        else:
            self.deeper_follow[run_id] = int(cell)
        self.deeper_retry.pop(run_id, None)              # another cell (or none): its back-off starts afresh

    # ---- Compare (the compare manager builds the requests) -------------------------
    def submit_compare(self, request: dict) -> Future:
        """Queue one worker op "compare" (analysis/worker.py Worker.compare; `request` without "op", as the compare
        manager builds it) and return a Future of the worker's reply: {ok, run, seq, compare: {...}} or {ok: false,
        errors: {compare}, error}; it raises AnalysisError when no reply came (the worker stopped or timed out). It goes
        after the explicit requests and before every other pass, in the order submitted; a Future cancelled before it
        starts is dropped, and close() cancels those still queued. Refused (503) where Clusters & UMAP cannot run."""
        if not self.available:
            raise self.refusal("Compare")
        future = Future()
        with self.lock:
            if self.closed:
                raise AnalysisError("Live analysis has stopped", 503)
            self.compare_queue.append((dict(request), future))
            self.wake.notify_all()
        return future

    def compare(self, request: dict, timeout: float = 600.0) -> dict:
        """Compare's one blocking call (engine/compare.py: CompareManager._score): submit_compare, then the worker's
        reply as the compare manager reads it, the op's result ("compare": columns, kinds, area, layout, ...) also at
        its top level beside ok, error, errors, run and seq. AnalysisError: 503 without the worker's environment or
        once closed; when no reply came within `timeout` (a request not started by then is dropped), or the worker
        stopped or timed out."""
        future = self.submit_compare(request)
        try:
            reply = future.result(timeout)
        except FutureTimeout:
            if future.cancel():                               # not started: out of the queue at once
                with self.lock:
                    self.compare_queue = deque(item for item in self.compare_queue if item[1] is not future)
            raise AnalysisError("the GPU worker did not compare in time") from None
        except CancelledError:
            raise AnalysisError("Live analysis has stopped", 503) from None
        return {**(reply.get("compare") or {}), **reply}

    def compare_frame(self, run_id: str, seq: int | None = None) -> dict:
        """The run's own column of a Compare op: the counted frame `seq` snaps to (`_snap`; the newest counted frame
        when None) as the worker's op takes it (a column's "frame": the fields a pass of that frame sends, `_process`,
        with its proseg count's file, `counts_file`); `seq` and `iter` are that counted frame's, whose file the spool
        keeps for COMPARE_FILE_HOLD_S, and as long again after each op that reads it (`_hold_compared`; the run's
        comparison before, which a new one replaces, holds its file no more). Refused: 404 an unknown frame, 409 a run
        with no frame yet, with no frame with proseg's count, or without exported transcripts."""
        run = self.runs.get(run_id)
        if seq is not None:
            seq = self._frame_number(run, seq)
        with run.lock:
            newest = len(run.frames) - 1
        if newest < 0:
            raise AnalysisError("This run has no frame yet", 409)
        found = self._snap(run, newest if seq is None else seq)
        if found is None:
            raise AnalysisError("This run has no frame with proseg's count", 409)
        with run.lock:
            frame = dict(run.frames[found])
        info = run.info
        if not info or not (run.live / "transcripts.bin.gz").exists():
            raise AnalysisError("This run has no exported transcripts (made before live analysis existed)", 409)
        with self.lock:                                       # its file stays in the spool until the ops read it
            self.compare_files[run_id] = {found: self.clock() + COMPARE_FILE_HOLD_S}
        return {"run": run_id, "seq": found, "iter": frame.get("iter"),
                "transcripts": str(run.live / "transcripts.bin.gz"), "genes": str(run.live / "genes.txt"),
                "frame": str(run.live / frame["file"]), "format": frame.get("format", "dense"),
                **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
                "z_origin": info["z"]["grid_origin"], "z_spacing": info["z"]["grid_spacing"], "ncells": info["cells"],
                "cells_file": str(run.live / frame["cells_file"]) if frame.get("cells_file") else None,
                "counts_file": str(run.live / frame["counts_file"])}

    def configure_qc(self, run_id: str, enabled: bool | None = None, seq: int | None = None) -> dict:
        """The run's own switch for the mixing check (`enabled`; off until switched on: apply [qc] enabled_by_default
        by switching it on when a run is created or opened), and/or one frame to score now (`seq`, as for analysis:
        whatever the switch says, once a model is ready; as the counted frame it snaps to, `_snap`: the reply's
        `requested` says which, also once the scheduler took it; none for a run without proseg's count). Refused
        requests (404 unknown frame, 422 not a frame number) change nothing."""
        if not self.available:
            raise self.refusal("The mixing check")
        run = self.runs.get(run_id)
        seq = self._frame_number(run, seq)
        if seq is not None and self._counts_reason(run_id) is not None:
            seq = None                                        # no proseg count: no job at all (describe_qc says why)
        if seq is not None:
            with self.lock:
                qc = self.qc_runs.get(run_id)
                done = set(qc.done) if qc is not None else set()
            seq = self._snap(run, seq, done.__contains__)
        with self.lock:
            if enabled is not None:
                self.qc_enabled[run_id] = bool(enabled)
            if seq is not None:
                self.qc_requested[run_id] = seq
            self.qc_version += 1
            self.wake.notify_all()
        reply = self.describe_qc(run_id)
        if seq is not None:                                   # the scheduler, just woken, may have taken it already
            reply["requested"] = seq
        return reply

    def describe_qc(self, run_id: str) -> dict:
        """{available (a worker and a provider), enabled, model, dir, status {state: idle|running|ready|error, seq,
        error?, start?}, start (pending|done|none|error, None before a model), latest_seq, seqs (counted frames scored
        with the current model), backfill (frames still to backfill), requested (a counted frame asked for that waits,
        or None), unavailable (None; "old_proseg" or "no_counts": a run without proseg's count, active ones too, which
        gets no job at all; "no_transcripts": an ended run nothing can be scored for), settings {mixed_at, judge_min,
        fit_min}}."""
        reason = self._counts_reason(run_id)
        with self.lock:
            qc = self.qc_runs.get(run_id)
            done = sorted(qc.done) if qc is not None else []
            return {"available": self.available and self.qc_provider is not None,
                    "enabled": self.qc_enabled.get(run_id, False),
                    "model": None if qc is None else qc.model, "dir": None if qc is None else str(qc.dir),
                    "status": dict(self.qc_status.get(run_id, {"state": "idle"})),
                    "start": None if qc is None else qc.start, "latest_seq": done[-1] if done else None,
                    "seqs": done[-2000:], "backfill": 0 if qc is None else len(qc.backfill),
                    "requested": self.qc_requested.get(run_id),
                    "unavailable": reason or ((qc.unavailable or None) if qc is not None else None),
                    "settings": self.qc_thresholds()}

    def qc_thresholds(self) -> dict:
        s = self.qc_settings
        return {"mixed_at": s["mixed_at"], "judge_min": int(s["judge_min_counts"]), "fit_min": int(s["fit_min_counts"])}

    def qc_dir(self, run, model_id: str) -> Path:
        """Where a run's results with this model live: var/runs/<id>/qc/<model8>.pc1/ (qc_folder)."""
        return qc_folder(run.dir, model_id)

    def qc_grid(self, run) -> dict:
        """The run's bin grid {x0, y0, bin, nx, ny}: anchored at the run's raster origin (the x0, y0 of its prior and of
        every frame, so bin ids hold for the whole run; the areas were calibrated on this grid) and covering the
        raster, with 100 µm bins in whole-sample (tiled) runs and 50 µm in detail runs ([qc] bin_um_tiled,
        bin_um_detail). Without a raster yet: the run box."""
        info = run.info or {}
        tiled = run.meta.get("mode") == "tiled" or (info.get("prior") or {}).get("format") == "tiles"
        size = float(self.qc_settings["bin_um_tiled" if tiled else "bin_um_detail"])
        raster = info.get("prior") or {}
        if not all(k in raster for k in ("x0", "y0", "nx", "ny", "voxel")):
            with run.lock:
                raster = dict(run.frames[0]) if run.frames else {}
        try:
            x0, y0 = float(raster["x0"]), float(raster["y0"])
            x1, y1 = x0 + float(raster["nx"]) * float(raster["voxel"]), y0 + float(raster["ny"]) * float(raster["voxel"])
        except (KeyError, TypeError, ValueError):             # nothing exported yet: the run box
            x0, y0, x1, y1 = (float(v) for v in run.meta.get("box"))
        return {"x0": x0, "y0": y0, "bin": size, "nx": max(1, math.ceil((x1 - x0) / size)),
                "ny": max(1, math.ceil((y1 - y0) / size))}

    def close(self):
        with self.lock:
            self.closed = True
            queued, self.compare_queue = list(self.compare_queue), deque()
            self.wake.notify_all()
        for _, future in queued:                              # nobody will run them: their waiters hear at once
            future.cancel()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if self.bins_dir is not None:
            shutil.rmtree(self.bins_dir, ignore_errors=True)

    def _bins_file(self, run_id: str, seq: int) -> Path:
        """Where the worker writes a frame's per-bin sums for the provider (`transient_dir`, made when first needed and
        removed by close(), or when this manager is gone or the process exits)."""
        if self.bins_dir is None or not self.bins_dir.is_dir():
            self.bins_dir = transient_dir()
            weakref.finalize(self, shutil.rmtree, str(self.bins_dir), True)
        return self.bins_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', run_id)}-{seq:06d}.bins.npz"

    # ---- scheduling ----------------------------------------------------------------------------------------
    @staticmethod
    def _frame_number(run, seq) -> int | None:
        if seq is None:
            return None
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            raise AnalysisError("seq must be a frame number") from None
        with run.lock:
            if not 0 <= seq < len(run.frames):
                raise AnalysisError("Unknown frame", 404)
        return seq

    @staticmethod
    def _results_dir(run) -> Path:
        """Where a run's Clusters & UMAP results live: <run>/analysis/pc1/ (counts_file.BASIS: those of the outline
        counts earlier versions made, in analysis/, are never read); its settings.json stays in analysis/."""
        return run.dir / "analysis" / BASIS

    def _paths(self, run, seq: int, gen: int | None = None) -> tuple[Path, Path]:
        """A frame's result files of settings generation `gen` (None: the run's current one): <seq>.json and
        <seq>.cells.bin at generation 0 (a run that never changed its settings), <seq>-s<n>.* at generation n."""
        if gen is None:
            gen = self.gen.get(run.meta["id"], 0)
        base, suffix = self._results_dir(run), f"-s{gen}" if gen else ""
        return base / f"{seq:06d}{suffix}.json", base / f"{seq:06d}{suffix}.cells.bin"

    # ---- proseg's count of each frame -------------------------------------------------------------------------
    @staticmethod
    def _count_path(run, seq: int) -> Path | None:
        """Frame `seq`'s proseg count file (its record's counts_file: in the spool, absolute, or in the live folder),
        None for a frame without one."""
        with run.lock:
            name = run.frames[seq].get("counts_file") if 0 <= seq < len(run.frames) else None
        return run.live / name if name else None

    def _has_count(self, run, seq: int) -> bool:
        """Whether frame `seq` has proseg's count in a file that is there."""
        path = self._count_path(run, seq)
        return path is not None and path.exists()

    @staticmethod
    def _frame_count(run, seq: int) -> str:
        """The count frame `seq` is read on now (counts_file.count_of of its counts line, the newest of its seq):
        "sampled" (the count proseg saves) for a frame proseg counted for a pause, the hold, a finish or the end, else
        "averaged" (also without a line)."""
        with run.lock:
            line = run.frames[seq].get("counts") if 0 <= seq < len(run.frames) else None
        return count_of(line)

    def _stands(self, run, seq: int, kept) -> bool:
        """Whether a result kept on count `kept` (None: none kept; a summary without one: "averaged") stands for frame
        `seq` now: its count is the one the frame is read on, or the frame's counts file is gone (it cannot be read again:
        the result kept is served as it is, with its own count's words)."""
        if kept is None:
            return False
        return kept == self._frame_count(run, seq) or not self._has_count(run, seq)

    def _umap_current(self, run_id: str, run, seq: int) -> bool:
        """Whether frame `seq`'s Clusters & UMAP result kept here stands now (`_stands`). Lock held."""
        summary = self.results.get(run_id, {}).get(seq)
        return summary is not None and self._stands(run, seq, summary.get("count") or "averaged")

    def _qc_count(self, qc: QcState, seq: int) -> str | None:
        """The count frame `seq`'s check was kept on (QcState.done; None: not scored); one an earlier session left is
        read from its <seq>.qc.json once. Lock held."""
        if seq not in qc.done:
            return None
        if qc.done[seq] is None:
            qc.done[seq] = self._stored_count(qc.dir / f"{seq:06d}.qc.json") or "averaged"
        return qc.done[seq]

    def _qc_current(self, qc: QcState, run, seq: int) -> bool:
        """Whether frame `seq`'s check kept stands now (`_stands`). Lock held."""
        return self._stands(run, seq, self._qc_count(qc, seq))

    @staticmethod
    def _stored_count(path) -> str | None:
        """The count a stored summary names (a <seq>.qc.json, a UMAP result's json): "averaged" without one (an older
        one), None when it cannot be read."""
        try:
            found = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None
        return (found.get("count") or "averaged") if isinstance(found, dict) else None

    def result_counts(self, run_id: str) -> dict[int, str]:
        """{seq: the count its Clusters & UMAP result was read on} of the results kept: the WebSocket announces a frame
        analysed again on another count (server/app.py)."""
        with self.lock:
            return {seq: (summary.get("count") or "averaged") for seq, summary in self.results.get(run_id, {}).items()}

    @staticmethod
    def _snap(run, seq: int, kept=None) -> int | None:
        """Frame `seq` as an explicit request takes it: the newest counted frame at or before it whose file is there or
        whose results are kept already (`kept(seq)`, None: none count), else the oldest such after it; None when there
        is none (a run without proseg's count, or before its first counted frame). Looked for from `seq` in the run's
        counted frames (run.counted, in order), down then up, NEWEST_LOOK at a time, their files outside the run's
        lock: never every frame of a long run."""
        for down in (True, False):
            edge = None                                       # the farthest one looked at, this way
            while True:
                with run.lock:
                    counted = run.counted
                    if down:
                        end = bisect.bisect_right(counted, seq) if edge is None else bisect.bisect_left(counted, edge)
                        chunk = counted[max(0, end - NEWEST_LOOK):end][::-1]
                    else:
                        start = bisect.bisect_right(counted, seq if edge is None else edge)
                        chunk = counted[start:start + NEWEST_LOOK]
                    names = [(s, run.frames[s].get("counts_file")) for s in chunk if s < len(run.frames)]
                if not chunk:
                    break
                edge = chunk[-1]
                for s, name in names:
                    if name and ((kept is not None and kept(s)) or (run.live / name).exists()):
                        return s
        return None

    def _umap_kept(self, run_id: str, run, locked: bool = False):
        """kept(seq) for _snap: whether that frame's Clusters & UMAP result of the run's current settings is kept (here,
        or on disk). `locked`: the caller holds the lock."""
        if locked:
            gen, done = self.gen.get(run_id, 0), set(self.results.get(run_id, {}))
        else:
            with self.lock:
                gen, done = self.gen.get(run_id, 0), set(self.results.get(run_id, {}))
        return lambda seq: seq in done or all(path.exists() for path in self._paths(run, seq, gen))

    def _newest_counted(self, run) -> int | None:
        """The run's newest counted frame whose file is there (its seq), None without one. Kept per run until its
        frames change or that file goes (one look at it each time); looked for from the end of the run's counted frames
        (run.counted, in order), NEWEST_LOOK at a time, their files outside the run's lock: never every frame of a long
        run."""
        run_id = run.meta.get("id")
        with run.lock:
            key = (len(run.frames), getattr(run, "version", None))
        seen = self.newest_seen.get(run_id)
        if seen is not None and seen[0] == key and (seen[2] is None or seen[2].exists()):
            return seen[1]
        found, below = (None, None), None
        while found[0] is None:
            with run.lock:                                    # the next ones back, below the last looked at
                counted = run.counted
                end = len(counted) if below is None else bisect.bisect_left(counted, below)
                chunk = counted[max(0, end - NEWEST_LOOK):end]
                names = [(s, run.frames[s].get("counts_file")) for s in reversed(chunk) if s < len(run.frames)]
            if not chunk:
                break
            below = chunk[0]
            found = next(((s, run.live / name) for s, name in names if name and (run.live / name).exists()),
                         (None, None))
        self.newest_seen[run_id] = (key, *found)
        return found[0]

    def _counted_with_files(self, run, below: int) -> list[int]:
        """The run's counted frames before `below` whose file is there (backfill's choice), of run.counted. Outside the
        run's lock, a spooled file (an absolute path) by one listing of its folder, the others (a few forced frames, in
        the run's folder) one by one: a long run has thousands of counted frames, most of their spooled files pruned,
        and a look at each took about 100 ms at 20,000, with the scheduler's lock held (a check starting: _qc_state)."""
        with run.lock:
            counted = run.counted
            names = [(s, run.frames[s].get("counts_file")) for s in counted[:bisect.bisect_left(counted, below)]
                     if s < len(run.frames)]
        listed, found = {}, []
        for s, name in names:
            if not name:
                continue
            if os.path.isabs(name):
                folder, file = os.path.split(name)
                if folder not in listed:
                    try:
                        listed[folder] = set(os.listdir(folder))
                    except OSError:
                        listed[folder] = set()
                there = file in listed[folder]
            else:
                there = (run.live / name).exists()
            if there:
                found.append(s)
        return found

    @staticmethod
    def _init_count(run) -> Path | None:
        """Frame 0's proseg count (its counts line's reason "init": proseg's count at iteration 0) when its file is
        there, else None."""
        with run.lock:
            first = run.frames[0] if run.frames else {}
            name, reason = first.get("counts_file"), (first.get("counts") or {}).get("reason")
        if not name or reason != "init":
            return None
        path = run.live / name
        return path if path.exists() else None

    def _start_line_of(self, run) -> Path | None:
        """A run with a start line (it started from cells alone, engine/qc_rules.py start_of "cells", and its frame 0
        is counted): frame 0's count file, else None (a start with nuclei has no start line)."""
        return self._init_count(run) if start_of(run.meta) == "cells" else None

    @staticmethod
    def _start_counts(run, seq: int) -> str | None:
        """The file whose start block frame `seq` pairs with (proseg's iteration-0 assignment at that frame's keep): its
        own when its counts line has the block, else the newest counted frame before it with one whose file is there;
        None without one. Looked for from `seq` down the run's counted frames (run.counted, in order), NEWEST_LOOK at a
        time, their files outside the run's lock: proseg puts the block on every K-th counted frame (the engine's runs:
        K = 1) and the prune keeps the newest with one, so the first few have one."""
        below = None
        while True:
            with run.lock:                                    # the next ones down, below the last looked at
                counted = run.counted
                end = bisect.bisect_right(counted, seq) if below is None else bisect.bisect_left(counted, below)
                chunk = counted[max(0, end - NEWEST_LOOK):end]
                records = [run.frames[s] for s in reversed(chunk) if s < len(run.frames)]
                names = [record.get("counts_file") for record in records if (record.get("counts") or {}).get("has_start")]
            if not chunk:
                return None
            below = chunk[0]
            found = next((str(run.live / name) for name in names if name and (run.live / name).exists()), None)
            if found is not None:
                return found

    def _counts_state(self, run_id: str) -> dict | None:
        """The run's proseg count as the run store tells it (engine/runs.py counts_status: mode, reason, issued, served,
        outstanding, newest, newest_at, extract_ms, bytes, spool_mb, can_issue, why), or None when it cannot (a deleted
        run)."""
        try:
            state = self.runs.counts_status(run_id)
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    def _counts_reason(self, run_id: str) -> str | None:
        """Why the run has no proseg count to analyse ("old_proseg", "no_counts"), else None."""
        reason = (self._counts_state(run_id) or {}).get("reason")
        return reason if reason in COUNTS_REASONS else None

    @staticmethod
    def _reason(run_id: str, seen: dict) -> str | None:
        """The counts reason a pass's look (`_look`) found for the run, else None."""
        reason = ((seen.get(run_id) or {}).get("counts") or {}).get("reason")
        return reason if reason in COUNTS_REASONS else None

    def _mark_unavailable(self, run_id: str, reason: str):
        """A run without proseg's count: its Clusters & UMAP status says why (the WebSocket pushes it). Lock held."""
        status = {"state": "unavailable", "reason": reason}
        if self.status.get(run_id) != status:
            self.status[run_id] = status
            self.version += 1

    def _look(self, models: dict, locked: bool = False) -> dict:
        """What a scheduling pass needs of the run store and the QC provider, asked without holding the lock (`locked`:
        the caller holds it; tests): {run id: {"counts": its proseg count (`_counts_state`) or None, "watched": a Watch
        runs on it (the provider's optional `watching`, for runs whose check has a model)}} for every run a pass may
        schedule."""
        try:
            active = [run.meta["id"] for run in self.runs.active() if run.meta.get("kind") != "stain"] \
                if self.available else []
        except Exception:                                     # a store that cannot list its runs: those named here
            active = []

        def wanted():                                         # asked for, or with Clusters & UMAP or the check on
            on = [run_id for run_id in dict.fromkeys([*active, *self.enabled, *self.qc_enabled])
                  if self.enabled.get(run_id, run_id in active) or self.qc_enabled.get(run_id, False)]
            return [*self.requested, *self.qc_requested, *self.deeper_requested, *on]
        if locked:
            ids = wanted()
        else:
            with self.lock:
                ids = wanted()
        watching = getattr(self.qc_provider, "watching", None)
        seen = {}
        for run_id in dict.fromkeys(ids):
            watched = False
            if watching is not None and models.get(run_id) is not None:
                try:
                    watched = bool(watching(run_id))
                except Exception:                             # a provider in trouble: no priority
                    watched = False
            seen[run_id] = {"counts": self._counts_state(run_id), "watched": watched}
        return seen

    def _qc_models(self, locked: bool = False) -> dict:
        """{run id: (model id, model path), or None while it has none} for the runs whose check is on and those that
        asked for a frame, asked of the provider without holding the lock (`locked`: the caller holds it; tests)."""
        provider = self.qc_provider
        if provider is None:
            return {}
        if locked:
            wanted = [run_id for run_id, on in self.qc_enabled.items() if on] + list(self.qc_requested)
        else:
            with self.lock:
                wanted = [run_id for run_id, on in self.qc_enabled.items() if on] + list(self.qc_requested)
        found = {}
        for run_id in dict.fromkeys(wanted):
            try:
                model = provider.qc_model(self.runs.get(run_id))
            except Exception:                                 # a deleted run, or a provider in trouble: not now
                model = None
            if isinstance(model, dict):
                model = (model.get("id"), model.get("path"))
            found[run_id] = (str(model[0]), str(model[1])) if model and model[0] and model[1] else None
        return found

    def _qc_state(self, run_id: str, models: dict, run=None) -> QcState | None:
        """The run's QcState for the model that scores it now, a new one when the model changed (its results never mix
        with another model's), None without a model. Lock held."""
        model = models.get(run_id)
        if model is None:
            return None
        state = self.qc_runs.get(run_id)
        if state is None or (state.model, state.path) != model:
            run = run if run is not None else self.runs.get(run_id)
            state = QcState(model[0], model[1], self.qc_dir(run, model[0]))
            self._qc_found(state)
            newest = self._newest_counted(run)              # the live pass's; backfill: earlier counted files there
            if newest is not None:
                state.backfill = backfill_plan(newest, state.done, available=self._counted_with_files(run, newest))
            self.qc_runs[run_id] = state
            self.qc_status.pop(run_id, None)                  # the previous model's last pass says nothing of this one
            self.bg_turn = False                              # a check that starts: its newest frame first
        return state

    @staticmethod
    def _qc_found(state: QcState):
        """Results an earlier session left in the run's folder for this model (one listing)."""
        try:
            names = set(os.listdir(state.dir))
        except OSError:
            return
        state.done = {int(name[:-8]): None for name in names     # their counts read when asked (_qc_count)
                      if name.endswith(".qc.json") and name[:-8].isdigit() and name[:-8] + ".qc.bin" in names}
        if {"start.npz", "start.bin", "start.json"} <= names:
            state.start = "done"

    def _start_ready(self, run, qc: QcState) -> bool:
        """Whether the run's start line can be computed now: a run that started from cells alone (engine/qc_rules.py
        start_of "cells"; any other start, or a prior that was not exported, has none: "none") whose frame 0 is counted
        (proseg's count at iteration 0, reason "init", its file there) and whose prior is written (its footprints
        place the starting cells). Lock held."""
        info = run.info
        if not info:
            return False
        prior = info.get("prior") or {}
        if start_of(run.meta) != "cells" or not prior.get("file"):
            qc.start = "none"                                 # nuclei in the start (proseg's iteration 0 holds them
            return False                                      # alone), or no starting cells exported
        return self._init_count(run) is not None and (run.live / prior["file"]).exists()

    def _background(self, run_id: str, run, qc: QcState, now: float) -> Job | None:
        """The run's next background pass (its start line, then backfill), or None; a backfill frame whose count's file
        went meanwhile (the spool's reserve moved on) is skipped. Lock held."""
        if now < qc.retry_at:
            return None
        if qc.start == "pending" and self._start_ready(run, qc):
            return Job(run_id, None, ("qc_start",), "start", qc)
        while qc.backfill and (qc.backfill[0] in qc.done or not self._has_count(run, qc.backfill[0])):
            qc.backfill.pop(0)
        return Job(run_id, qc.backfill[0], ("qc",), "backfill", qc) if qc.backfill else None

    def _forget(self, run_id: str):
        """A deleted run: its settings and its proseg-count state go, and the GPU worker drops the run. Lock held."""
        for table in (self.enabled, self.qc_enabled, self.qc_requested, self.qc_started, self.qc_status, self.qc_runs,
                      self.too_few, self.exported, self.deeper_requested, self.deeper_follow, self.deeper_follow_at,
                      self.deeper_retry, self.compared, self.run_params, self.gen, self.taken, self.live_seq,
                      self.where_box, self.where_box_at, self.to_ask, self.ask_plan, self.asked, self.leads,
                      self.ask_after, self.pruned,
                      self.stopped_at, self.newest_seen, self.compare_files):
            table.pop(run_id, None)
        self.to_prune.discard(run_id)
        self.spooled.discard(run_id)
        self.squeezed.discard(run_id)
        self.loaded.discard(run_id)
        self.sweeps.discard(run_id)
        for kind in ("umap", "qc", "deeper"):
            self.noted.pop(f"{kind}:{run_id}", None)
        self.to_forget.add(run_id)

    def _active(self) -> list[str]:
        """The runs analysed by default: every active run of the engine (preparing, running, paused, finishing), while
        Clusters & UMAP can run here (where they cannot, nothing is started). Lock held; the runs' own lock is taken
        inside it, as runs.get does."""
        if not self.available:
            return []
        try:
            return [run.meta["id"] for run in self.runs.active() if run.meta.get("kind") != "stain"]
        except Exception as exc:                          # a store that cannot list its runs: explicit switches only
            self._note(f"The active runs could not be listed ({type(exc).__name__}: {exc}): only runs switched on are "
                       "analysed by themselves", once="active")
            return []

    def _too_few(self, run_id: str, run, newest: int) -> bool:
        """The run's last pass found too few cells to cluster (a small region: every later frame would fail alike, a GPU
        pass each): no automatic pass until its newest frame holds materially more cells (TOO_FEW_GROWTH) than the one
        that failed. Lock held; the run's own lock is taken inside it."""
        before = self.too_few.get(run_id)
        if before is None:
            return False
        with run.lock:
            cells = run.frames[newest].get("cells") if 0 <= newest < len(run.frames) else None
        factor, more = TOO_FEW_GROWTH
        if isinstance(cells, (int, float)) and not isinstance(cells, bool) and cells >= max(before * factor, before + more):
            del self.too_few[run_id]
            return False
        return True

    def _without_transcripts(self, run, qc: QcState) -> bool:
        """An ended run whose live folder has no transcripts.bin.gz (made before live analysis existed): nothing can
        be scored, so it gets no start line, backfill or automatic pass (looked at once per model). Lock held."""
        if qc.unavailable is None:
            qc.unavailable = "" if (run.live / "transcripts.bin.gz").exists() else "no_transcripts"
            if qc.unavailable:
                qc.start, qc.backfill = "none", []
        return bool(qc.unavailable)

    def _release(self, run_id: str, qc: QcState):
        """An ended run without automatic Clusters & UMAP passes (not switched on explicitly) whose check has scored all
        it scores by itself (start line, backfill, newest frame), nothing asked of it: the GPU worker may drop the run,
        once (it loads it again for a later request). Runs that are only looked at must not hold the
        worker GPU's memory; a run a Go deeper window follows, or waits on, stays (each click would load it again). Lock
        held."""
        if qc.released or qc.start == "pending" or qc.backfill or run_id in self.requested or run_id in self.qc_requested \
                or run_id in self.deeper_follow or run_id in self.deeper_requested:
            return
        qc.released = True
        self.to_forget.add(run_id)

    def _release_compared(self) -> float | None:
        """Runs a Compare op loaded (`_hold_compared`) whose hold is over leave the GPU worker as Go deeper's do: one
        whose check is on once it has nothing more to score (`_release`; its `released` reset here), any other at once
        unless something is asked of it or its Clusters & UMAP are switched on, a deleted one with its settings
        (`_forget`), never a live one; one with a Compare op still queued waits for that op (it holds it again).
        Returns the seconds until the next hold ends, None without any. Lock held."""
        now, soonest = self.clock(), None
        queued = {str(request.get("run") or "") for request, _ in self.compare_queue}
        for run_id, until in list(self.compared.items()):
            if now < until:
                soonest = _soonest(soonest, until - now)
                continue
            del self.compared[run_id]
            if run_id in queued:
                continue
            try:
                run = self.runs.get(run_id)
            except Exception:                                 # deleted meanwhile
                self._forget(run_id)
                continue
            if run.meta.get("state") not in TERMINAL:         # live: its passes keep it
                continue
            qc = self.qc_runs.get(run_id)
            if qc is not None and self.qc_enabled.get(run_id, False):
                qc.released = False                           # loaded again: released again once unwatched (_release)
            elif not (self.enabled.get(run_id, False) or run_id in self.requested or run_id in self.qc_requested
                      or run_id in self.deeper_follow or run_id in self.deeper_requested):
                self.to_forget.add(run_id)
        return soonest

    def _next_job(self, models: dict | None = None, seen: dict | None = None):
        """(Job, None) to run now, else (None, seconds to sleep; None = until woken). Lock held. `models`: _qc_models(),
        `seen`: _look(models), both asked before the lock was taken (None: asked here, as tests do).
        Of the runs due for an automatic pass, the one whose last one started longest ago goes first: a run whose
        analyses take (or fail within) its whole interval would otherwise be due again first every time and starve the
        others. A frame asked of the check whose sample has no model yet stays asked until the model is ready (the
        provider is asked again every QC_MODEL_POLL_S, and at once when it says models changed). Clusters & UMAP: every
        active run (`_active`) unless switched off, and an ended run only when switched on explicitly. Explicit Go deeper
        requests go before everything (`_deeper_job`), and a followed cell rides along each live job of its run.
        Only counted frames go: a live pass takes the run's newest counted frame whose file is there, a frame asked for
        goes as the counted frame it snaps to (`_snap`), and a run without proseg's count gets no job at all (its status
        says why). A watched run's live pass goes before any background pass. The counts tokens due are left in
        `to_ask` (`_plan_counts`) for the scheduler's thread to ask for after the lock."""
        if models is None:
            models = self._qc_models(locked=True)
        if seen is None:
            seen = self._look(models, locked=True)
        self._expire_follows()
        compared = self._release_compared()              # seconds until a run Compare loaded may go, or None
        if self.deeper_turn and self.requested:
            # One explicit Clusters & UMAP request between explicit Go deeper jobs: while a
            # window is scrubbed, a new Go deeper request came during every job, and the UMAP pane of the frames on
            # screen waited until the scrubbing stopped.
            self.deeper_turn = False
            wait = None
        else:
            job, wait = self._deeper_job(models, seen)
            if job is not None:
                self.deeper_turn = True
                return job, None
            self.deeper_turn = False
        for run_id, seq in list(self.requested.items()):     # explicit requests: at once, whatever the cadence
            del self.requested[run_id]
            job = self._request_job(run_id, seq, models, seen)
            if job is not None:
                return job, None
        while self.compare_queue:                             # Compare: its view waits; a cancelled one is dropped
            request, future = self.compare_queue.popleft()
            if future.set_running_or_notify_cancel():
                return Job(str(request.get("run") or ""), None, ("compare",), "compare",
                           compare=(request, future)), None
        now, best, background = self.clock(), None, None
        if not self.available:                                # turned off for the session (a lasting hello): what was
            return None, compared                             # asked before failed once; no automatic pass is due
        for run_id, seq in list(self.qc_requested.items()):
            try:
                run = self.runs.get(run_id)
            except Exception:                                 # the run was deleted
                self._forget(run_id)
                continue
            if self._reason(run_id, seen) is not None:        # no proseg count: nothing to score, ever
                del self.qc_requested[run_id]
                continue
            if run_id not in models:                          # asked after the provider was: ask it now
                if self.qc_provider is not None:
                    wait = 0.0
                continue
            qc = self._qc_state(run_id, models, run)
            if qc is None:                                    # its sample is still learning: kept until it has a model
                wait = _soonest(wait, QC_MODEL_POLL_S)
                continue
            del self.qc_requested[run_id]
            seq = self._snap(run, seq, qc.done.__contains__)  # again: its file may have gone while it waited
            if seq is not None and not self._qc_current(qc, run, seq):     # (or it was counted again meanwhile)
                return Job(run_id, seq, ("qc",), "request", qc), None
        needing = {}                                          # runs whose next pass needs a newer counted frame
        watched = None                                        # the live pass of a watched run that is due, if any
        for run_id in list(dict.fromkeys([*self.enabled, *self.qc_enabled, *self._active()])):
            if not (self.enabled.get(run_id, True) or self.qc_enabled.get(run_id, False)):
                continue
            try:
                run = self.runs.get(run_id)
            except Exception:                                 # the run was deleted
                self._forget(run_id)
                continue
            ended = run.meta.get("state") in TERMINAL         # read before the frames: a run indexes its last frames first
            # Clusters & UMAP are on for every active run (the default) and for an ended one switched on explicitly.
            umap_on, qc_on = self.enabled.get(run_id, not ended), self.qc_enabled.get(run_id, False)
            if not (umap_on or qc_on):
                continue
            reason = self._reason(run_id, seen)
            if reason is not None:                            # no proseg count: no job at all, its status says why
                self._mark_unavailable(run_id, reason)
                continue
            with run.lock:
                last_frame = len(run.frames) - 1
            if self.live_seq.get(run_id, -1) > last_frame:   # started again (its frames from 0): none taken yet
                del self.live_seq[run_id]
            qc = self._qc_state(run_id, models, run) if qc_on else None
            if qc_on and qc is None and self.qc_provider is not None and not ended:
                wait = _soonest(wait, QC_MODEL_POLL_S)        # its model is not ready yet: ask again in a while
            if qc is not None and ended and self._without_transcripts(run, qc):
                qc = None                                     # nothing to score it with: no automatic pass at all
            if not umap_on and qc is None:                    # the check alone, with nothing to score it with yet
                continue
            if qc is not None and background is None:
                background = self._background(run_id, run, qc, now)
            newest = self._newest_counted(run)
            if newest is None or not run.info:                # no counted frame yet (frame 0's comes by itself)
                if not ended:
                    wait = _soonest(wait, POLL_S)
                continue
            # Too few cells: held back until the run's newest frame (counted or not) holds materially more. A result
            # kept on another count than the frame's now (counted again: a request's line, then a pause's, the hold's,
            # a finish's or the end's) is analysed again; a failure is not tried again for the same frame and count.
            few = umap_on and self._too_few(run_id, run, last_frame)
            count = self._frame_count(run, newest)
            status = self.status.get(run_id, {})
            umap_todo = umap_on and not few and not self._umap_current(run_id, run, newest) \
                and not (status.get("state") == "error" and status.get("seq") == newest
                         and status.get("count", "averaged") == count)
            failed = self.qc_status.get(run_id, {})
            qc_todo = qc is not None and not self._qc_current(qc, run, newest) \
                and not (failed.get("state") == "error" and failed.get("seq") == newest
                         and failed.get("count", "averaged") == count)
            held = qc_todo and now < qc.retry_at
            if held:                                          # failed a moment ago: no automatic pass until then
                qc_todo, wait = False, _soonest(wait, qc.retry_at - now)
            if not (umap_todo or qc_todo):
                if ended and qc is not None and not umap_on and not held:
                    self._release(run_id, qc)                 # an ended run, all scored: off the GPU worker
                if not ended and (umap_on or not held):       # caught up: look for a newer frame shortly
                    wait = _soonest(wait, POLL_S)
                if not ended and (umap_on and not few or qc is not None):
                    needing[run_id] = (run, umap_on and not few, qc)
                continue
            ops, last = None, None
            if umap_todo:
                last = self.started.get(run_id, -math.inf)
                due = last + self.interval(run)
                if now >= due:
                    ops = ["umap", "qc"] if qc_todo else ["umap"]   # the check rides along with every UMAP pass
                else:
                    wait = _soonest(wait, due - now)
            if ops is None and qc_todo:
                last = self.qc_started.get(run_id, -math.inf)
                due = last + self.qc_interval(run)
                if now >= due:
                    ops = ["qc"]
                else:
                    wait = _soonest(wait, due - now)
            if ops is not None:
                candidate = (last, run_id, newest, ops, qc, run, umap_on and not few)
                if best is None or last < best[0]:
                    best = candidate
                if (seen.get(run_id) or {}).get("watched"):
                    watched = candidate                       # a Watch runs on it: no background pass meanwhile
        job = None
        if background is not None and (best is None or self.bg_turn and watched is None):
            self.bg_turn = False
            if background.kind == "backfill":
                background.qc.backfill.pop(0)
            job = background
        elif best is not None:
            _, run_id, newest, ops, qc, run, umap_wants = best
            if "umap" in ops:
                self.started[run_id] = now
            if "qc" in ops:
                self.qc_started[run_id] = now
            self.bg_turn = True
            self.live_seq[run_id] = newest
            if run.meta.get("state") not in TERMINAL:         # taken: its next pass needs a newer counted frame
                needing[run_id] = (run, umap_wants, qc)
            follow = self._deeper_follow(run_id, qc, now)
            job = Job(run_id, newest, [*ops, "deeper"] if follow else ops, "live", qc, deeper=follow)
        for run_id in [run_id for run_id in self.ask_plan if run_id not in needing]:
            del self.ask_plan[run_id]                         # its next pass needs no newer counted frame now
        wait = self._plan_counts(needing, seen, now, wait)
        if job is not None:
            return job, None
        return None, wait if compared is None else _soonest(wait, compared)

    def _request_job(self, run_id: str, seq: int, models: dict, seen: dict) -> Job | None:
        """An explicit Clusters & UMAP request as its job (the check rides along when the run's check is on and the
        frame is not scored yet), the frame snapped again (its file may have gone while it waited; results of the run's
        current settings kept count); None when it cannot go: a deleted run, one without proseg's count (its status
        says why) or with no counted frame yet. Lock held."""
        reason = self._reason(run_id, seen)
        if reason is not None:
            self._mark_unavailable(run_id, reason)
            return None
        try:
            run = self.runs.get(run_id)
        except Exception:                                     # the run was deleted
            self._forget(run_id)
            return None
        seq = self._snap(run, seq, self._umap_kept(run_id, run, locked=True))
        if seq is None:
            return None
        qc = self._qc_state(run_id, models, run)
        ops = ["umap"]
        if qc is not None and not self._qc_current(qc, run, seq) and self._has_count(run, seq):   # the frame on screen
            ops.append("qc")                                                       # gets both: one read of its count
            if self.qc_requested.get(run_id) == seq:
                del self.qc_requested[run_id]
        return Job(run_id, seq, ops, "request", qc)

    def _plan_counts(self, needing: dict, seen: dict, now: float, wait):
        """The counts tokens of the runs whose next live pass needs a newer counted frame (`needing`: run id -> (run,
        Clusters & UMAP want frames, its QcState or None)): for a sampling run in proseg's "request" mode (a paused or
        held run counts its state by itself; "every" counts each frame; a test's own runs are asked by the test) with
        no token outstanding, one is left in `to_ask` (with the where box a followed Go deeper window wants) once the
        extraction budget allows it (`_budget`) and the run's next pass is due within the wait for its frame (`leads`:
        at the pass's start when passes go back to back); a later one is timed (`wait`, the seconds to sleep, returned)
        and kept in `ask_plan`, so that it is asked for when its time comes while a pass runs too (`_ask_due`). A full
        spool stops tokens (said once in the log until they went for COUNTS_NOTE_S) and has the run pruned. Past
        iteration 2,000 a token goes only once the run is a thinning step past its newest counted frame
        (`_thinning_allows`): proseg-live writes the frame a token asks for whatever its history thinning keeps. Lock
        held."""
        for run_id, (run, umap_wants, qc) in needing.items():
            self.ask_plan.pop(run_id, None)
            if run.meta.get("state") != "running" or is_test_run(run.meta):
                self.asked.pop(run_id, None)                  # a token read while paused: served with no frame
                continue
            counts = (seen.get(run_id) or {}).get("counts") or {}
            if counts.get("mode") != "request":
                continue
            self._learn_lead(run_id, counts)
            if counts.get("outstanding"):                     # one at a time: its frame is on its way
                continue
            if counts.get("can_issue") is False:
                self._counts_wait(run_id, counts.get("why"), now)
                continue
            if not self._thinning_allows(run, counts.get("newest")):   # past 2,000: a step after the last count
                continue
            if now - self.stopped_at.get(run_id, -math.inf) >= COUNTS_NOTE_S:
                self.noted.pop(f"counts:{run_id}", None)      # tokens went for a while: a later stop is news again
            due = self._next_due(run_id, run, umap_wants, qc)
            at = max(self._budget(counts), due - self.leads.get(run_id, COUNTS_LEAD_S[0]),
                     self.ask_after.get(run_id, -math.inf))
            if now >= at:
                self.to_ask[run_id] = list(self.where_box[run_id]) if run_id in self.where_box else None
            elif math.isfinite(at):
                wait = _soonest(wait, at - now)
                self.ask_plan[run_id] = (at, run, umap_wants, qc)
        return wait

    @staticmethod
    def _thinning_allows(run, newest) -> bool:
        """Whether a token may go as history thinning goes (THINNING_STEPS): always up to iteration THINNED_FROM, past
        it once the run's newest frame is a thinning step past its newest counted frame (`newest`, its seq): a token
        forces a frame thinning may not keep, and such frames are never pruned. Lock held (the run's own is taken)."""
        with run.lock:
            if not run.frames:
                return True
            now = run.frames[-1].get("iter")
            then = run.frames[newest].get("iter") if isinstance(newest, int) and 0 <= newest < len(run.frames) else None
        if not isinstance(now, (int, float)) or now <= THINNED_FROM or not isinstance(then, (int, float)):
            return True
        return now - then >= thinning_step(now)

    def _next_due(self, run_id: str, run, umap_wants: bool, qc: QcState | None) -> float:
        """When the run's next live pass may start (clock): its Clusters & UMAP interval after the last one, or the
        check's alone (after a failure's back-off), whichever comes first. Lock held."""
        dues = []
        if umap_wants:
            dues.append(self.started.get(run_id, -math.inf) + self.interval(run))
        if qc is not None:
            dues.append(max(self.qc_started.get(run_id, -math.inf) + self.qc_interval(run), qc.retry_at))
        return min(dues) if dues else math.inf

    def _budget(self, counts: dict) -> float:
        """The earliest time (clock) the run's next token may be asked for: its newest counted frame's arrival
        (`newest_at`) plus that frame's ms.extract / counts_extract_share and, with counts_mb_s, its bytes at that pace;
        -inf before its first counted frame."""
        at = counts.get("newest_at")
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            return -math.inf
        spacing = max(0.0, float(counts.get("extract_ms") or 0.0)) / 1000.0 / self.extract_share
        if self.counts_mb_s > 0:
            spacing = max(spacing, max(0.0, float(counts.get("bytes") or 0)) / 1e6 / self.counts_mb_s)
        return float(at) + spacing

    def _learn_lead(self, run_id: str, counts: dict):
        """Once the run's last token was served (its counted frame came at `newest_at`), how long it waited joins the
        run's lead (half the last, half the new, at most COUNTS_LEAD_S[1]). Lock held."""
        asked, came = self.asked.get(run_id), counts.get("newest_at")
        if asked is None or counts.get("outstanding") or isinstance(came, bool) or not isinstance(came, (int, float)) \
                or came < asked:
            return
        del self.asked[run_id]
        took = min(COUNTS_LEAD_S[1], float(came) - asked)
        before = self.leads.get(run_id)
        self.leads[run_id] = took if before is None else (before + took) / 2

    def _counts_wait(self, run_id: str, why, now: float):
        """The run store issues no token for the run now (its spool or the engine's is full, or the RAM disk low): said
        once in the log (again only once its tokens went for COUNTS_NOTE_S since); a full spool has the run pruned (at
        most every PRUNE_RETRY_S), and a full engine's spool every run with a spool too, the ended ones first and to the
        files a job needs (`_squeeze`). Lock held."""
        self.stopped_at[run_id] = now
        self._note(f"Proseg's count is not asked for while {COUNTS_WAITS.get(why, 'the spool cannot take more')}",
                   run_id, once=f"counts:{run_id}")
        if why in ("run_cap", "engine_cap") and now - self.pruned.get(run_id, -math.inf) >= PRUNE_RETRY_S:
            self.pruned[run_id] = now
            self.to_prune.add(run_id)
            if why == "engine_cap":                           # the other runs' files: theirs go too
                self._squeeze(run_id)
                self.to_prune |= self.spooled

    def _squeeze(self, run_id: str):
        """The other runs' spooled files leave run `run_id` less than its cap, or fill the engine's spool: every ended
        run with a spool (it gets no new frame) keeps only the files a job needs from now on (`_prune`, lean), pruned
        so after the job (`_prune_runs`: the ended runs first). Lock held; the runs' own lock is taken inside it, as
        runs.get does."""
        for other in sorted(self.spooled - self.squeezed - {run_id}):
            try:
                ended = self.runs.get(other).meta.get("state") in TERMINAL
            except Exception:                                 # deleted: its spool went with it
                continue
            if ended:
                self.squeezed.add(other)
                self.to_prune.add(other)

    def _deeper_job(self, models: dict, seen: dict | None = None):
        """The next explicit Go deeper request as (Job, None), else (None, seconds to sleep or None). Lock held.
        A request waits while its sample's model is not ready (asked again every QC_MODEL_POLL_S), and a start line's
        until the run's start line is done (none to come: dropped, its failure handed over); a run whose check is off
        has nothing to explain with, nor has one without proseg's count: its request is dropped. A frame goes as the
        counted frame it snaps to (again: its file may have gone while it waited)."""
        wait = None
        for run_id, (seq, cell) in list(self.deeper_requested.items()):
            try:
                run = self.runs.get(run_id)
            except Exception:                                 # the run was deleted
                self._forget(run_id)
                continue
            reason = self._reason(run_id, seen or {})
            if reason is not None:                            # no proseg count: nothing to explain it with, ever
                del self.deeper_requested[run_id]
                self.deeper_dropped.append({"run": run_id, "seq": seq, "cell": cell, "model": None,
                                            "error": COUNTS_REASONS[reason], "reason": reason})
                continue
            if not self.qc_enabled.get(run_id, False):        # switched off meanwhile: Go deeper needs the check
                del self.deeper_requested[run_id]
                self.deeper_dropped.append({"run": run_id, "seq": seq, "cell": cell, "model": None,
                                            "error": "the mixing check is off for this run", "reason": "off"})
                continue
            if run_id not in models:                          # asked after the provider was: ask it now
                if self.qc_provider is not None:
                    wait = 0.0
                continue
            qc = self._qc_state(run_id, models, run)
            if qc is None:                                    # its sample is being learned: kept until it has a model
                wait = _soonest(wait, QC_MODEL_POLL_S)
                continue
            if seq == "start" and qc.start != "done":
                if qc.start == "pending":                     # the start line is scored first (a background pass)
                    wait = _soonest(wait, POLL_S)
                    continue
                del self.deeper_requested[run_id]
                self.deeper_dropped.append({"run": run_id, "seq": "start", "cell": cell, "model": qc.model,
                                            "error": "this run has no start line", "reason": "no_start"})
                continue
            del self.deeper_requested[run_id]
            if seq != "start":
                found = self._snap(run, seq)
                if found is None:
                    self.deeper_dropped.append({"run": run_id, "seq": seq, "cell": cell, "model": qc.model,
                                                "error": NO_COUNT, "reason": "no_counts"})
                    continue
                seq = found
            qc.released = False                               # loaded again for this: released again once unwatched
            want = {"cell": cell, "start": seq == "start", "explicit": True}
            if seq == "start":
                return Job(run_id, None, ("deeper",), "request", qc, deeper=want), None
            ops = ["deeper"] if self._qc_current(qc, run, seq) else ["qc", "deeper"]   # the card and the window change
                                                                                      # together
            if "qc" in ops and self.qc_requested.get(run_id) == seq:
                del self.qc_requested[run_id]
            return Job(run_id, seq, ops, "request", qc, deeper=want), None
        return None, wait

    def _expire_follows(self):
        """Followed cells whose lease nobody renewed for DEEPER_FOLLOW_TTL_S stop being followed (the window that
        asked is gone: closed before its answer came, or its page reloaded or closed), and so do their where boxes; a
        where box no reply renewed for WHERE_BOX_TTL_S goes alone (the view is off). While a run is paused or held
        (its ack "paused") both clocks start again: the page renews neither then, so both leases count from Resume, and
        the first token after it asks for the box the paused frame's reply armed. Lock held; the runs' own lock is
        taken inside it, as runs.get does."""
        now = self.clock()
        for run_id in list(self.deeper_follow_at):
            if self._paused(run_id):
                self.deeper_follow_at[run_id] = now
                if run_id in self.where_box_at:
                    self.where_box_at[run_id] = now
        for run_id, at in list(self.deeper_follow_at.items()):
            if now - at >= DEEPER_FOLLOW_TTL_S:
                self.deeper_follow_at.pop(run_id, None)
                self.deeper_follow.pop(run_id, None)
                self.deeper_retry.pop(run_id, None)
                self.where_box.pop(run_id, None)
                self.where_box_at.pop(run_id, None)
        for run_id, at in list(self.where_box_at.items()):
            if now - at >= WHERE_BOX_TTL_S:
                self.where_box_at.pop(run_id, None)
                self.where_box.pop(run_id, None)

    def _paused(self, run_id: str) -> bool:
        """Whether the run rests now, paused or held at the end of its schedule: proseg's ack says "paused" and the run
        is running or paused (engine/deeper_where.py `_follow`'s rule; done, finishing or stopped: no lease is kept).
        False for a deleted run."""
        try:
            run = self.runs.get(run_id)
        except Exception:
            return False
        with run.lock:
            ack = getattr(run, "ack", None)
            return isinstance(ack, dict) and ack.get("state") == "paused" \
                and run.meta.get("state") in ("running", "paused")

    def _deeper_follow(self, run_id: str, qc: QcState | None, now: float) -> dict | None:
        """The followed cell a live job of this run also explains, or None (none followed, nothing to explain it with,
        or backing off after it failed). Lock held."""
        cell = self.deeper_follow.get(run_id)
        if cell is None or qc is None or now < self.deeper_retry.get(run_id, (0.0, 0))[0]:
            return None
        return {"cell": cell, "start": False, "explicit": False}

    def _loop(self):
        while True:
            models = self._qc_models()                        # before the lock: the provider has locks of its own,
            seen = self._look(models)                         # and so has the run store
            with self.lock:
                if self.closed:
                    return
                try:
                    job, wait = self._next_job(models, seen)
                except Exception:                             # whatever a run's state, the scheduler must survive
                    job, wait = None, 0.5
                forget, self.to_forget = self.to_forget, set()
                dropped, self.deeper_dropped = self.deeper_dropped, []
                sweeps, self.sweeps = self.sweeps, set()
                asks, self.to_ask = self.to_ask, {}
                prunes, self.to_prune = self.to_prune, set()
                idle = job is None and not forget and not dropped and not sweeps and not asks and not prunes
                prestart = idle and self._prestart_due()
                if idle and not prestart:                     # until a request, a setting, close()
                    if wait is None and self.available:       # or the next start, or a new run: every active run is
                        wait = IDLE_LOOK_S                    # analysed
                    self.wake.wait(None if wait is None else min(wait, MAX_SLEEP_S))
                    continue
            if prestart:
                # The worker starts (imports, its hello, then its warm-up while idle) with the first active run, while
                # it prepares, not with its first frame. Whatever this start does, the scheduler goes on.
                try:
                    self._ensure_worker()
                except Exception as exc:
                    self.prestart_failed = self.clock()
                    if not isinstance(exc, AnalysisError):
                        self.failed_at = time.monotonic()
                        self._note(f"The analysis worker could not start: {type(exc).__name__}: {exc}", once="worker")
                continue
            for record in dropped:                            # Go deeper requests dropped under the lock
                self._deeper_hand({"iter": None, "explicit": True, **record})
            for run_id in sorted(forget):                     # this thread is the only one that writes to the worker
                self._send({"op": "forget", "run": run_id})
            for run_id in sorted(sweeps):                     # files of earlier settings: here, between passes
                self._sweep_generations(run_id)
            if asks:                                          # proseg's next counted frames, during the pass
                self._ask_counts(asks)
            if job is None:
                self._prune_runs(prunes)                      # a spool full enough to stop the run's tokens
                continue
            try:
                if job.ops == ("umap",):
                    self._analyse(job.run, job.seq)           # clusters and UMAP alone, as always
                    failure = None
                elif job.ops == ("compare",):
                    failure = self._compare(job)
                else:
                    failure = self._process(job)
            except Exception as exc:  # keep the loop alive whatever happens
                failure = str(exc)
                # A failure the pass recorded itself, or one of a pass whose settings changed meanwhile, sets no status.
                self._failed(job, job.ops, failure, generation=getattr(exc, "generation", None),
                             recorded=getattr(exc, "recorded", False))
            # After the job (its file read), the files no job needs go from its run's spool, and from a full one's
            # (those a held-back token's ask found full during the pass too).
            with self.lock:
                prunes |= self.to_prune
                self.to_prune = set()
            self._prune_runs(prunes | ({job.run} if job.run else set()))
            if failure is not None:
                with self.lock:                               # no tight retry loop; a new request still goes at once
                    if not self.requested and not self.qc_requested and not self.deeper_requested and not self.closed:
                        self.wake.wait(0.5)

    def _ask_counts(self, asks: dict):
        """Ask the run store for the counts tokens `_plan_counts` left (run -> a where box: runs.request_where without
        waiting, a counts token with it when none is outstanding; None: runs.request_counts), in the scheduler's thread
        after the lock (each writes the run's control file). One refused (None) is not asked again for COUNTS_RETRY_S;
        a where export refused asks for the count alone; a failure is said once in the log."""
        for run_id, box in asks.items():
            token, where = None, None
            if box is not None:
                try:
                    self.runs.request_where(run_id, box, wait_s=0)
                    token = where = True                      # its answer comes as the run's next where export
                except Exception as exc:                      # a box the store refuses: the count alone, said once
                    where = False
                    self._note(f"Proseg's where export could not be asked for: {type(exc).__name__}: {exc}", run_id,
                               once=f"where:{run_id}")
            if token is None:
                try:
                    token = self.runs.request_counts(run_id)
                except Exception as exc:                      # a run deleted meanwhile, a store in trouble: not now
                    self._note(f"Proseg's count could not be asked for: {type(exc).__name__}: {exc}", run_id,
                               once=f"ask:{run_id}")
            now = self.clock()
            with self.lock:
                if token is not None:
                    self.asked[run_id] = now
                    self.noted.pop(f"ask:{run_id}", None)
                if where:
                    self.noted.pop(f"where:{run_id}", None)
                self.ask_after[run_id] = now + (POLL_S if token is not None else COUNTS_RETRY_S)

    def _ask_due(self):
        """Ask for the counts tokens held back until now (`ask_plan`), as a pass waits for the worker (`_await_reply`):
        each run's counts are looked at again first and `_plan_counts` decides as between passes (a token out or
        served meanwhile, a run paused or a full spool: none; a newer counted frame moves the budget on). In the
        scheduler's thread; a failure is said once in the log and never fails the pass."""
        now = self.clock()
        with self.lock:
            due = {run_id: plan[1:] for run_id, plan in self.ask_plan.items() if plan[0] <= now}
            for run_id in due:
                del self.ask_plan[run_id]
        if not due:
            return
        try:
            seen = {run_id: {"counts": self._counts_state(run_id)} for run_id in due}
            with self.lock:
                if self.closed:
                    return
                self._plan_counts(due, seen, self.clock(), None)
                asks, self.to_ask = self.to_ask, {}
            if asks:
                self._ask_counts(asks)
        except Exception as exc:
            self._note(f"Proseg's count could not be asked for during a pass: {type(exc).__name__}: {exc}",
                       once="ask-due")

    def _await_reply(self, timeout: float) -> dict:
        """The worker's reply to the request just sent (queue.Empty when none came within `timeout` seconds).
        Meanwhile the counts tokens `_plan_counts` held back (the extraction budget, a spaced pass's lead) are asked
        for as their time comes (`_ask_due`), so a sampling run's next counted frame is on its way while the pass
        runs, not only once it ends."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            with self.lock:
                due = min((plan[0] for plan in self.ask_plan.values()), default=None)
            wait = left if due is None else min(left, max(ASK_SLICE_S, due - self.clock()))
            try:
                return self.replies.get(timeout=max(0.0, wait))
            except queue.Empty:
                if due is None or time.monotonic() >= deadline:
                    raise
            self._ask_due()

    def _prune_runs(self, run_ids):
        """`_prune` each of these runs, the ended ones first: what they free is room the active runs' prunes see. In
        the scheduler's thread, which a failure never stops (said once in the log)."""
        found = []
        for run_id in run_ids:
            try:
                run = self.runs.get(run_id)
                found.append((run.meta.get("state") not in TERMINAL, run_id, run))
            except Exception:                                 # deleted: its spool folder went with it
                continue
        for _, run_id, run in sorted(found, key=lambda item: item[:2]):
            try:
                self._prune(run_id, run)
            except Exception as exc:
                self._note(f"Proseg's spooled counts could not be pruned: {type(exc).__name__}: {exc}", run_id,
                           once=f"prune:{run_id}")

    def _prune(self, run_id: str, run=None):
        """Remove the run's spooled count files that no job needs any more (engine/counts_spool.py prune, keeping
        `_keep`'s), after each of its jobs and while a full spool stops tokens. Its share of the spool: its cap, or what
        the engine's cap (twice that) leaves beside the other runs' files when that is less; when the others hold more
        than its cap, the ended runs give way first (`_squeeze`), and an ended run squeezed keeps only the files a job
        needs. Runs with a spool only; never a test's own runs (engine/qc_test.py prunes theirs) nor the forced files
        in the run folder. In the scheduler's thread, outside the lock; a failure is said once in the log."""
        spool = getattr(self.runs, "spool", None)
        if run is None:
            try:
                run = self.runs.get(run_id)
            except Exception:                                 # deleted: its spool folder went with it
                return
        if spool is None or is_test_run(run.meta) or not ((run.info or {}).get("counts") or {}).get("spool"):
            return
        ended = run.meta.get("state") in TERMINAL
        try:                                                  # the other runs' files (a spool that cannot tell: none)
            others = max(0.0, float(spool.usage_mb()) - float(spool.usage_mb(run_id)))
        except Exception:
            others = 0.0
        try:
            with run.lock:                                    # a frame counted from here on may be missing from the
                since = run.counted[-1] if run.counted else -1    # listing: _keep looks at those too
            there = {os.path.normpath(str(path)) for path in spool.files(run_id)}
            with self.lock:
                self.spooled.add(run_id)
                if not ended:
                    self.squeezed.discard(run_id)             # sampling again (started anew)
                    if others > self.spool_cap_mb:
                        self._squeeze(run_id)
                lean = ended and run_id in self.squeezed
            keep = self._keep(run_id, run, there, others, lean, since)
            if there - keep:
                spool.prune(run_id, keep)
        except Exception as exc:
            self._note(f"Proseg's spooled counts could not be pruned: {type(exc).__name__}: {exc}", run_id,
                       once=f"prune:{run_id}")
            return
        with self.lock:
            self.noted.pop(f"prune:{run_id}", None)
            self.newest_seen.pop(run_id, None)                # its newest file is looked at again

    def _keep(self, run_id: str, run, there: set, others: float = 0.0, lean: bool = False,
              since: int | None = None) -> set[str]:
        """The run's spooled count files to keep (absolute paths; `there`: those in its spool now). Always: those of
        the frames still to analyse (newer than its last live pass, while Clusters & UMAP or the check is on), of the
        frames asked for, waiting for backfill or taken by Compare (its queued ops and the frame compare_frame gave
        within COMPARE_FILE_HOLD_S), of its where exports, the newest with the start block, and those of frames newer
        than the newest counted one read here (their counts lines merge meanwhile: AHEAD_FRAMES). Then, unless `lean`
        (an ended run squeezed: those alone), its KEEP_NEWEST newest and an evenly spaced reserve of up to BACKFILL_MAX
        of the others (a check switched on later backfills from it), newest first, as long as they fit with the next
        two counted frames in its share of the spool: its cap, or twice that less the other runs' files (`others`, MB)
        when that is less.
        Only the frames whose file a prune could remove are looked at, never every counted frame of a long run: those
        of the files in `there`, the frames held as above, the KEEP_NEWEST newest and the newest with the start block
        (from the end of run.counted), and those counted after `since` (the newest counted frame when `there` was
        listed, -1 for none: their files may have landed since; None: nothing was counted since)."""
        ended = run.meta.get("state") in TERMINAL
        counts = (run.info or {}).get("counts") or {}
        folder = counts.get("spool")
        now = self.clock()
        with self.lock:
            held = set()
            for asked in (self.requested.get(run_id), self.qc_requested.get(run_id),
                          (self.deeper_requested.get(run_id) or (None,))[0]):
                if isinstance(asked, int) and not isinstance(asked, bool):
                    held.add(asked)
            qc = self.qc_runs.get(run_id)
            if qc is not None and self.qc_enabled.get(run_id, False):
                held.update(qc.backfill)
            taken = self.compare_files.get(run_id, {})
            for seq, until in list(taken.items()):
                if until <= now:
                    del taken[seq]
            held.update(taken)
            queued = {os.path.normpath(str(column["frame"]["counts_file"]))
                      for request, _ in self.compare_queue for column in request.get("columns") or []
                      if isinstance(column, dict) and isinstance(column.get("frame"), dict)
                      and column["frame"].get("run") == run_id and column["frame"].get("counts_file")}
            on = self.enabled.get(run_id, not ended) or self.qc_enabled.get(run_id, False)
            live = self.live_seq.get(run_id, -1)
        with run.lock:
            frames, counted = run.frames, run.counted
            last = len(frames) - 1
            held.update(w.get("seq") for w in (getattr(run, "wheres", None) or []) if isinstance(w, dict))
            newest = counted[-KEEP_NEWEST:]
            fresh = counted[bisect.bisect_right(counted, since):] if since is not None else []
            # Proseg keeps the start block only when info.json says so; then the newest has one (every K-th counted).
            start = next((s for s in reversed(counted)
                          if s <= last and (frames[s].get("counts") or {}).get("has_start")), None) \
                if counts.get("start") else None
            top = counted[-1] if counted else -1              # files of later frames: their lines merge meanwhile
            looked = {_spooled_seq(name) for name in there} | held | {*newest, *fresh, start}
            records = [(s, frames[s]) for s in looked if isinstance(s, int) and 0 <= s <= last]
            lines = {s: (record["counts_file"], record.get("counts") or {}) for s, record in records
                     if record.get("counts_file")}
        due_after = (live if live <= last else -1) if on else math.inf   # above its frames: started again
        path = {s: os.path.normpath(str(run.live / name)) for s, (name, _) in lines.items()}
        keep = set(queued)
        keep.update(path[s] for s in path if s in held or s > due_after)
        if start in path:
            keep.add(path[start])
        keep.update(name for name in there if _spooled_seq(name) > top)
        if folder:
            keep.update(os.path.normpath(os.path.join(str(folder), f"{s:06d}.counts.zst"))
                        for s in range(max(top, last - AHEAD_FRAMES) + 1, last + AHEAD_FRAMES + 1))
        if lean:
            return keep
        # Its share of the spool, less what is kept and room for the next two counted frames (the size of the largest
        # of its newest): the newest go first, then the reserve, evenly spaced over the others, BACKFILL_MAX at most.
        size = {s: max(0, int(line.get("bytes") or 0)) for s, (_, line) in lines.items()}
        frame = max((size[s] for s in newest if s in size), default=0)
        share = min(self.spool_cap_mb, 2 * self.spool_cap_mb - others) * 1e6
        room = share - 2 * frame - sum(size[s] for s in path if path[s] in there and path[s] in keep)
        for s in reversed(newest):
            if s not in path or path[s] in keep or path[s] not in there:
                continue
            if size[s] > room:
                break
            keep.add(path[s])
            room -= size[s]
        rest = sorted(s for s in path if path[s] in there and path[s] not in keep)
        if rest and room > 0:
            mean = sum(size[s] for s in rest) / len(rest)
            count = min(BACKFILL_MAX, len(rest), int(room // mean) if mean > 0 else BACKFILL_MAX)
            if count > 0:
                wanted = np.linspace(rest[0], rest[-1], count) if count > 1 else [(rest[0] + rest[-1]) / 2]
                keep.update(path[s] for s in nearest_each(rest, wanted))
        return keep

    def _prestart_due(self) -> bool:
        """Whether the worker should start now, before any job: no worker runs, Clusters & UMAP can run here, an
        active run has them (or its check) on, and no early start failed in the last PRESTART_RETRY_S. A worker that
        ended by itself (a crash, in its warm-up too) counts as an early start that failed: one that keeps dying is not
        started again and again while nothing is asked of it (a job still starts it as before). Lock held; this thread
        is the only one that starts or talks to the worker."""
        if not self.prestart or not self.available or (self.process is not None and self.process.poll() is None):
            return False
        if self.process is not None:                          # it ended by itself
            self.process = None
            self.prestart_failed = self.clock()
            return False
        if self.clock() - self.prestart_failed < PRESTART_RETRY_S:
            return False
        return any(self.enabled.get(run_id, True) or self.qc_enabled.get(run_id, False) for run_id in self._active())

    def _set_status(self, run_id: str, status: dict, gen: int | None = None):
        """The run's Clusters & UMAP status; one of a pass of settings generation `gen` only while that is the run's."""
        with self.lock:
            if gen is None or gen == self.gen.get(run_id, 0):
                self._status_locked(run_id, status)

    def _status_locked(self, run_id: str, status: dict):
        self.status[run_id] = status
        self.version += 1
        if status.get("state") == "error":
            self._note(f"Clusters & UMAP failed for frame {status.get('seq')}: {status.get('error')}", run_id,
                       once=f"umap:{run_id}")

    def _set_qc_status(self, run_id: str, status: dict):
        with self.lock:
            self.qc_status[run_id] = status
            self.qc_version += 1

    def _store(self, run_id: str, seq: int, summary: dict, gen: int | None = None) -> bool:
        """Keep a result; one of settings generation `gen` only while that is the run's (False when it is not)."""
        with self.lock:
            if gen is not None and gen != self.gen.get(run_id, 0):
                return False
            self._store_locked(run_id, seq, summary)
            return True

    def _store_locked(self, run_id: str, seq: int, summary: dict):
        bucket = self.results.setdefault(run_id, {})
        bucket[seq] = summary
        while len(bucket) > 2000:
            bucket.pop(min(bucket))
        self.status[run_id] = {"state": "ready", "seq": seq}
        self.noted.pop(f"umap:{run_id}", None)            # a failure after this result is news again
        self.too_few.pop(run_id, None)                    # clustered: enough cells after all
        self.version += 1

    def _analyse(self, run_id: str, seq: int):
        """Clusters and UMAP of one frame (raises when that failed: the pass recorded its status already)."""
        failure = self._process(Job(run_id, seq, ("umap",), "request"))
        if failure is not None:
            error = AnalysisError(failure)
            error.recorded = True
            raise error

    def _process(self, job: Job) -> str | None:
        """One job: the ops its frame still lacks (a result already on disk is reused, op by op; a cell Go deeper has
        explained already is handed over from the deeper provider's cache), in one worker request that names the
        frame's proseg count (`counts_file`; with the run's start line, also frame 0's and the file of the start block
        the frame pairs with: `init_counts_file`, `start_counts_file`) and which of its counts every op reads (`count`:
        counts_file.count_of of the frame's counts line now: "sampled", the count proseg saves, for a frame counted for a
        pause, the hold, a finish or the end; a sampled check also writes its bytes on the averaged count,
        <seq>.qc.averaged.bin). A result kept on another count than that is worked out again. A frame without its
        count's file fails its ops with the reason "no_counts" (never an outline count in its place); a reply on another
        count than the one asked fails its op with the reason "count_mismatch" (nothing kept: not asked again for that
        frame and count automatically). Sets each op's status; returns the first failure of Clusters & UMAP or the
        check, None when they worked (a Go deeper failure goes to the deeper provider and never holds the scheduler
        back)."""
        run = self.runs.get(job.run)
        if job.ops == ("qc_start",):
            return self._start_line(job, run)
        if job.seq is None and "deeper" in job.ops:
            return self._deeper_start(job, run)
        self._ensure_settings(job.run, run)
        with self.lock:                                       # the pass's settings: its files and result are theirs
            gen = self.gen.get(job.run, 0)
            params = {**self.fixed, **self.settings_of(job.run)}
        seq, ops, qc = job.seq, list(job.ops), job.qc
        with run.lock:
            frame = dict(run.frames[seq])
        count = job.count = count_of(frame.get("counts"))
        counts = run.live / frame["counts_file"] if frame.get("counts_file") else None
        there = counts is not None and counts.exists()
        out_json, out_cells = self._paths(run, seq, gen)
        if "umap" in ops and out_json.exists() and out_cells.exists():   # computed before (e.g. while scrubbing)
            kept = json.loads(out_json.read_text())
            if (kept.get("count") or "averaged") == count or not there:    # on another count: worked out again
                self._store(job.run, seq, kept, gen)
                ops.remove("umap")
        if "qc" in ops:
            kept = None if qc is None else self._qc_cached(qc, seq)
            if qc is None or (kept is not None and (kept == count or not there)):
                if qc is not None:
                    with self.lock:
                        qc.done[seq] = kept
                ops.remove("qc")
        info = run.info
        block = self._deeper_block(job, run, frame, scoring="qc" in ops) if "deeper" in ops else None
        if "deeper" in ops and block is None:                # cached (handed over), or nothing to explain it with
            ops.remove("deeper")
        if not ops:
            return None
        if not info or not (run.live / "transcripts.bin.gz").exists():
            failure = "This run has no exported transcripts (made before live analysis existed)"
            self._failed(job, ops, failure, reason="no_transcripts" if info else None, generation=gen)
            return failure
        if not there:                                        # never an outline count in its place
            self._failed(job, ops, NO_COUNT, reason="no_counts", generation=gen)
            return NO_COUNT
        request = {"op": "analyze", "ops": ops, "run": job.run, "seq": seq, "iter": frame["iter"],
                   "frame": str(run.live / frame["file"]), "format": frame.get("format", "dense"),
                   "nx": frame["nx"], "ny": frame["ny"], "nz": frame["nz"], "x0": frame["x0"], "y0": frame["y0"],
                   "voxel": frame["voxel"], "z_origin": info["z"]["grid_origin"], "z_spacing": info["z"]["grid_spacing"],
                   "transcripts": str(run.live / "transcripts.bin.gz"), "genes": str(run.live / "genes.txt"),
                   "ncells": info["cells"], "counts_file": str(counts), "count": count}
        if "umap" in ops:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            request.update(out_json=str(out_json), out_cells=str(out_cells), params=params)
            self._set_status(job.run, {"state": "running", "seq": seq}, gen)
        if "qc" in ops or "deeper" in ops:                   # the frame's cell statistics (centres for Go deeper)
            request["cells_file"] = str(run.live / frame["cells_file"]) if frame.get("cells_file") else None
        if "qc" in ops or (block or {}).get("start_line"):
            init = self._start_line_of(run)                  # like for like: proseg's iteration-0 assignment at this
            if init is not None:                             # frame's keep (its start block, on frame 0's pattern)
                request.update(init_counts_file=str(init), start_counts_file=self._start_counts(run, seq))
        if "qc" in ops:
            qc.dir.mkdir(parents=True, exist_ok=True)
            # The frame's per-bin sums (the areas' input) go to a transient file on the RAM disk, read back and deleted
            # below: on a slow run folder disk that round trip (~1 MB on a whole sample) slowed UMAP passes.
            request["qc"] = {**self.qc_thresholds(), "model": qc.path, "model_id": qc.model, "grid": self.qc_grid(run),
                             "start": str(qc.dir / "start.npz") if qc.start == "done" else None,
                             "out_cells": str(qc.dir / f"{seq:06d}.qc.bin"),
                             "out_bins": str(self._bins_file(job.run, seq))}
            if count == "sampled":                           # its bytes on the averaged count too (the test reads them)
                request["qc"]["out_cells_e"] = str(qc.dir / f"{seq:06d}.qc.averaged.bin")
            self._set_qc_status(job.run, {"state": "running", "seq": seq})
        if "deeper" in ops:
            request["deeper"] = block
        try:
            reply = self._call(request)
        except Exception as exc:
            if "qc" in ops:
                self._load_bins(request["qc"]["out_bins"], keep=False)   # nothing left behind
            if "deeper" in ops:
                self._deeper_deliver(job, frame, error=str(exc), reason="worker")
            try:                                              # the loop's _failed sets no status of other settings
                exc.generation = gen
            except AttributeError:
                pass
            raise
        errors = self._errors(reply, ops)
        reasons = dict(reply.get("reasons") or {})
        for op, message in self._mismatched(reply, ops, count, errors).items():
            errors[op], reasons[op] = message, "count_mismatch"
        failure = None
        if "umap" in ops:
            # A pass of settings changed while it ran is thrown away: never stored, announced or noted, its files gone;
            # so are a pass's on another count than asked.
            if "umap" in errors:
                current = self._umap_outcome(job.run, seq, gen, frame.get("cells"), error=errors["umap"],
                                             reason=reasons.get("umap"), count=count)
                failure = errors["umap"] if current else None
            else:
                current = self._umap_outcome(job.run, seq, gen, frame.get("cells"), summary=reply["summary"])
            if not current or reasons.get("umap") == "count_mismatch":
                for path in (out_json, out_cells):
                    for name in (path, path.with_name(path.name + ".tmp")):
                        try:
                            name.unlink()
                        except OSError:
                            pass
        if "qc" in ops:
            bins = self._load_bins(request["qc"]["out_bins"], keep=False)   # read and deleted whatever the outcome
            if "qc" in errors:
                failure = failure or errors["qc"]
                if reasons.get("qc") == "count_mismatch":    # its bytes are another count's: not this frame's any more
                    with self.lock:
                        qc.done.pop(seq, None)
                self._qc_error(job, errors["qc"])
            else:
                summary = reply["qc"]
                self._qc_ready(job, {"kind": "frame", "summary": summary,
                                     "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                                     "bins": bins, "cells": request["qc"]["out_cells"], "count": count,
                                     "cells_e": request["qc"].get("out_cells_e"), "start": request["qc"]["start"]})
        if "deeper" in ops:                                  # after the check: the card first, then the window
            if "deeper" in errors:
                self._deeper_deliver(job, frame, error=errors["deeper"], reason=reasons.get("deeper"))
            else:
                self._deeper_deliver(job, frame, result=reply["deeper"])
        return failure

    @staticmethod
    def _mismatched(reply: dict, ops, count: str, errors: dict) -> dict:
        """{op: message} for each op whose result says it read another count than `count`, the one asked for (a summary
        or answer without one: "averaged"); ops that failed already are left as they are."""
        found = {}
        for op, key in (("umap", "summary"), ("qc", "qc"), ("deeper", "deeper")):
            result = reply.get(key)
            if op in ops and op not in errors and isinstance(result, dict):
                got = result.get("count") or "averaged"
                if got != count:
                    found[op] = f"the worker read the {got} count, not the {count} one asked for"
        return found

    def _compare(self, job: Job) -> None:
        """One Compare op in the worker: its reply goes to the job's Future, or the error when none came (the worker
        stopped, timed out or could not start). Never a failure of the scheduler's: nothing else waits on it. A run
        that is not live whose frame it counted is held on the worker for a while, then released (`_hold_compared`)."""
        request, future = job.compare
        try:
            reply = self._call({**request, "op": "compare"})
        except Exception as exc:
            self._note(f"Compare failed: {exc}", job.run or None, once=f"compare:{job.run}")
            self._hold_compared(request)
            future.set_exception(exc)
            return None
        if reply.get("ok"):
            self.noted.pop(f"compare:{job.run}", None)        # a failure after this answer is news again
        else:
            self._note(f"Compare failed: {reply.get('error') or 'no result'}", job.run or None, once=f"compare:{job.run}")
        self._hold_compared(request)
        future.set_result(reply)
        return None

    def _hold_compared(self, request: dict):
        """After a Compare op: the run whose own frame it read is loaded on the GPU worker now. One
        that is not live (ended, failed or deleted) stays there COMPARE_HOLD_S more, for its comparison's next op, then
        goes as Go deeper's runs go (`_release_compared`): runs that are only looked at must not hold the worker's GPU.
        A live run's state is never touched. The frame's count file, while compare_frame's hold on it stands, stays in
        the spool COMPARE_FILE_HOLD_S more, live run or not."""
        frames = [column["frame"] for column in request.get("columns") or []
                  if isinstance(column, dict) and isinstance(column.get("frame"), dict)]
        now = self.clock()
        with self.lock:
            for frame in frames:
                held, seq = self.compare_files.get(str(frame.get("run") or request.get("run") or "")), frame.get("seq")
                if held is not None and isinstance(seq, int) and seq in held:
                    held[seq] = now + COMPARE_FILE_HOLD_S
        counted = {str(frame.get("run") or request.get("run") or "") for frame in frames}
        for run_id in counted - {""}:
            try:
                live = self.runs.get(run_id).meta.get("state") not in TERMINAL
            except Exception:                                 # deleted
                live = False
            if not live:
                with self.lock:
                    self.compared[run_id] = self.clock() + COMPARE_HOLD_S
                    self.wake.notify_all()

    @staticmethod
    def _prior_block(run, info: dict) -> dict | None:
        """The run's starting cells as the worker reads them (op qc_start's "prior"): the exported prior's file and
        raster, or None before it exists."""
        prior = (info or {}).get("prior") or {}
        if not prior.get("file"):
            return None
        return {"file": str(run.live / prior["file"]), "format": prior.get("format", "dense"),
                **{k: prior[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel") if k in prior}}

    # ---- Go deeper ----------------------------------------------------------------------------------------------------
    def _deeper_block(self, job: Job, run, frame: dict | None, scoring: bool) -> dict | None:
        """The request's "deeper" block for the job's cell, or None when the
        worker is not needed: the deeper provider holds the answer already (it hands it over itself), or the model holds
        no details (no deep.npz: that failure is handed over). `scoring`: the same request scores the frame, so its
        bytes (the card's numbers) exist by the time the cell is explained. The block names the neighbours' classes and
        the cell's word since the start as last shown (`same_prev`, `start_prev`), which the worker keeps near their
        cut-offs; never that word for a frame read on the count proseg saves (counted for a pause, the hold, a finish or
        the end: the run rests there), whose word is its own, the band's alone (analysis/worker.py start_word)."""
        want, qc = job.deeper, job.qc
        if want is None or qc is None:
            return None
        start = bool(want.get("start"))
        seq = "start" if start else job.seq
        folder = Path(qc.path)
        folder = folder if folder.suffix != ".npz" else folder.parent       # the model folder (or its encoder.npz)
        if not (folder / DEEP_FILE).is_file():
            self._deeper_deliver(job, frame, error="this sample's model holds no details for Go deeper (deep.npz)",
                                 reason="needs_detail")
            return None
        provider = self.deeper_provider
        if provider is not None:
            try:
                cached = provider.deeper_cached(job.run, seq, int(want["cell"]), qc.model, bool(want.get("explicit")))
            except Exception as exc:                          # its cache in trouble: the worker answers instead
                print(f"SoPA companion: Go deeper's cache failed (run {job.run}, {seq}): {type(exc).__name__}: {exc}",
                      flush=True)
                cached = False
            if cached:
                job.deeper_done = True
                return None
        codes = qc.dir / ("start.bin" if start else f"{job.seq:06d}.qc.bin")
        th = self.qc_thresholds()
        block = {"model": str(folder), "model_id": qc.model, "cell": int(want["cell"]), "start": start, "k": DEEPER_K,
                 "fit_min": th["fit_min"], "judge_min": th["judge_min"],
                 "codes": str(codes) if scoring or codes.exists() else None,
                 "cell_ids": str(run.live / "cells.txt"), "v": DEEPER_V}
        if start:
            block["start_file"] = str(qc.dir / "start.npz")  # the start line's centres (cx, cy) and values
        elif self._start_line_of(run) is not None:           # the cell against itself at the start, like for like (its
            block["start_line"] = True                        # row of the start block: the request names its files)
        if provider is not None and hasattr(provider, "deeper_prev"):
            try:                                              # the neighbours' classes as last shown
                block["same_prev"] = provider.deeper_prev(job.run, int(want["cell"]), qc.model)
            except Exception:
                pass
        resting = frame is not None and count_of(frame.get("counts")) == "sampled"
        if block.get("start_line") and not resting and provider is not None \
                and hasattr(provider, "deeper_start_prev"):
            try:                                              # its word since the start as last shown
                word = provider.deeper_start_prev(job.run, int(want["cell"]), qc.model)
                if word in ("more", "fewer", "same"):
                    block["start_prev"] = word
            except Exception:
                pass
        return block

    def _deeper_start(self, job: Job, run) -> str | None:
        """Go deeper on the start line: the cell in proseg's count at iteration 0 (frame 0's file; centres from
        start.npz, the prior placing the starting cells). A failure is handed over and never holds the scheduler
        back."""
        info = run.info or {}
        prior = self._prior_block(run, info)
        init = self._init_count(run)
        if prior is None or init is None or not (run.live / "transcripts.bin.gz").exists():
            self._deeper_deliver(job, None, error="this run has no start line to explain", reason="no_start")
            return None
        block = self._deeper_block(job, run, None, scoring=False)
        if block is None:
            return None
        request = {"op": "analyze", "ops": ["deeper"], "run": job.run, "seq": None, "iter": None,
                   "counts_file": str(init), "transcripts": str(run.live / "transcripts.bin.gz"),
                   "genes": str(run.live / "genes.txt"), "ncells": info["cells"], "z_origin": info["z"]["grid_origin"],
                   "z_spacing": info["z"]["grid_spacing"], "prior": prior, "deeper": block}
        try:
            reply = self._call(request)
        except Exception as exc:
            self._deeper_deliver(job, None, error=str(exc), reason="worker")
            raise
        errors = self._errors(reply, ["deeper"])
        if errors:
            self._deeper_deliver(job, None, error=errors["deeper"], reason=(reply.get("reasons") or {}).get("deeper"))
        else:
            self._deeper_deliver(job, None, result=reply["deeper"])
        return None

    def _deeper_deliver(self, job: Job, frame: dict | None, result: dict | None = None, error: str | None = None,
                        reason: str | None = None):
        """Hand the job's Go deeper answer (or failure) to the deeper provider, once per job; a followed cell that failed
        backs off (DEEPER_RETRY_S), one that worked starts afresh."""
        if job.deeper is None or job.deeper_done:
            return
        job.deeper_done = True
        want = job.deeper
        start, explicit = bool(want.get("start")), bool(want.get("explicit"))
        failed = error is not None or not isinstance(result, dict)
        with self.lock:
            if not explicit and failed:
                _, failures = self.deeper_retry.get(job.run, (0.0, 0))
                failures += 1
                self.deeper_retry[job.run] = (self.clock() + min(DEEPER_RETRY_S[1], DEEPER_RETRY_S[0] * 2 ** (failures - 1)),
                                              failures)
            elif not failed:
                self.deeper_retry.pop(job.run, None)
        record = {"run": job.run, "seq": "start" if start else job.seq,
                  "iter": None if frame is None else frame.get("iter"), "cell": int(want["cell"]),
                  "model": None if job.qc is None else job.qc.model, "explicit": explicit}
        if failed:
            record.update(error=error or "the GPU worker sent no result", reason=reason)
        else:
            record["result"] = result
        self._deeper_hand(record)

    def _deeper_hand(self, record: dict):
        """Give a Go deeper record to the deeper provider (never under the lock); failures are noted for the log once."""
        where = "the start line" if record["seq"] == "start" else f"frame {record['seq']}"
        if "error" in record:
            if record.get("reason") not in ("off",):
                self._note(f"Go deeper failed for {where}, cell {record['cell']}: {record['error']}", record["run"],
                           once=f"deeper:{record['run']}")
        else:
            self.noted.pop(f"deeper:{record['run']}", None)    # a failure after this answer is news again
        provider = self.deeper_provider
        if provider is None:
            return
        try:
            provider.deeper_result(record)
        except Exception as exc:
            print(f"SoPA companion: Go deeper could not keep its answer (run {record['run']}, {where}): "
                  f"{type(exc).__name__}: {exc}", flush=True)

    def _start_line(self, job: Job, run) -> str | None:
        """The run's start line with its model (op qc_start: proseg's count at iteration 0, frame 0's file; the prior's
        footprints place the starting cells), unless an earlier session left it complete."""
        qc = job.qc
        if self._start_cached(qc):
            with self.lock:
                qc.start = "done"
            return None
        info = run.info or {}
        prior = self._prior_block(run, info)
        init = self._init_count(run)
        if prior is None or init is None:                    # gone since it was ready: tried again later
            message = "This run has no proseg count at iteration 0" if init is None else "This run has no prior"
            self._qc_error(job, message)
            return message
        qc.dir.mkdir(parents=True, exist_ok=True)
        settings = {**self.qc_thresholds(), "model": qc.path, "model_id": qc.model, "grid": self.qc_grid(run),
                    "out_cells": str(qc.dir / "start.bin"), "out_start": str(qc.dir / "start.npz"),
                    "out_bins": str(qc.dir / "start.bins.npz")}
        request = {"op": "qc_start", "run": job.run, "seq": None, "counts_file": str(init),
                   "transcripts": str(run.live / "transcripts.bin.gz"), "genes": str(run.live / "genes.txt"),
                   "ncells": info["cells"], "z_origin": info["z"]["grid_origin"],
                   "z_spacing": info["z"]["grid_spacing"], "prior": prior, "qc": settings}
        self._set_qc_status(job.run, {"state": "running", "seq": None, "start": True})
        reply = self._call(request)
        errors = self._errors(reply, ["qc_start"])
        if errors:
            self._qc_error(job, errors["qc_start"])
            return errors["qc_start"]
        self._qc_ready(job, {"kind": "start", "summary": reply["qc"],
                             "bins": self._load_bins(settings["out_bins"], keep=True),
                             "cells": settings["out_cells"], "start": settings["out_start"]})
        return None

    def _qc_ready(self, job: Job, record: dict):
        """Hand a result to the provider, then count it as done."""
        qc = job.qc
        problem = self._deliver({"run": job.run, "seq": job.seq, "model": qc.model, "dir": str(qc.dir),
                                 "source": job.kind, "single": job.kind != "live", **record})
        with self.lock:
            if record["kind"] == "start":
                qc.start = "done"
            else:
                qc.done[job.seq] = record.get("count") or "averaged"     # kept for the count it was read on
            qc.failures, qc.retry_at = 0, 0.0
            self.qc_status[job.run] = {"state": "error", "seq": job.seq, "error": problem} if problem else \
                {"state": "ready", "seq": job.seq, **({"start": True} if record["kind"] == "start" else {})}
            self.qc_version += 1
            if not problem:
                self.noted.pop(f"qc:{job.run}", None)         # a failure after this result is news again

    def _qc_error(self, job: Job, message: str):
        """A QC pass failed: no automatic pass of the run for a while (doubling); a start line is retried a few times."""
        qc = job.qc
        with self.lock:
            if qc is not None:
                qc.failures += 1
                qc.retry_at = self.clock() + min(QC_RETRY_S[1], QC_RETRY_S[0] * 2 ** (qc.failures - 1))
                if job.kind == "start":
                    qc.start_failures += 1
                    if qc.start_failures >= START_ATTEMPTS:
                        qc.start = "error"
            self.qc_status[job.run] = {"state": "error", "seq": job.seq, "error": message,
                                       **({"start": True} if job.kind == "start" else {}),
                                       **({"count": job.count} if job.count == "sampled" else {})}
            self.qc_version += 1
        self._note(f"Mixing check failed for {'the start line' if job.kind == 'start' else f'frame {job.seq}'}: {message}",
                   job.run, once=f"qc:{job.run}")
        if qc is not None:
            self._deliver({"kind": "error", "run": job.run, "seq": job.seq, "model": qc.model, "dir": str(qc.dir),
                           "source": job.kind, "single": job.kind != "live", "error": message})

    def _failed(self, job: Job, ops, message: str, reason: str | None = None, generation: int | None = None,
                recorded: bool = False):
        """A job that failed as a whole, for each of its ops. The Clusters & UMAP status is set unless the pass recorded
        it itself (`recorded`) or ran with settings the run no longer has (`generation`); where they cannot run at all
        (a lasting hello) it is "unavailable", not an error."""
        if "umap" in ops and not recorded:
            if self.lasting is not None:
                self._set_status(job.run, {"state": "unavailable"}, generation)
            else:
                self._set_status(job.run, {"state": "error", "seq": job.seq, "error": message,
                                           **({"reason": reason} if reason else {}),
                                           **({"count": job.count} if job.count == "sampled" else {})}, generation)
        if "qc" in ops or "qc_start" in ops:
            self._qc_error(job, message)
        if "deeper" in ops:                                   # unless its answer or failure went over already
            self._deeper_deliver(job, None, error=message, reason=reason or "worker")

    def _deliver(self, record: dict) -> str | None:
        """Give a record to the provider's qc_result; its failure is logged and returned, never raised."""
        handler = getattr(self.qc_provider, "qc_result", None)
        if handler is None:
            return None
        try:
            handler(record)
        except Exception as exc:
            print(f"SoPA companion: the mixing check could not keep its {record['kind']} result "
                  f"(run {record['run']}, frame {record.get('seq')}): {type(exc).__name__}: {exc}", flush=True)
            return f"{type(exc).__name__}: {exc}"
        return None

    def _qc_cached(self, qc: QcState, seq: int) -> str | None:
        """The count a frame's check left on disk was read on (its <seq>.qc.json beside its <seq>.qc.bin), None when
        it left none."""
        if not (qc.dir / f"{seq:06d}.qc.bin").exists():
            return None
        return self._stored_count(qc.dir / f"{seq:06d}.qc.json")

    @staticmethod
    def _start_cached(qc: QcState) -> bool:
        return all((qc.dir / name).exists() for name in ("start.npz", "start.bin", "start.json"))

    @staticmethod
    def _load_bins(path, keep: bool) -> dict | None:
        """A bins file as {name: array}; a frame's (transient) file is deleted once read."""
        if not path:
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                return {key: stored[key] for key in stored.files}
        except (OSError, ValueError):
            return None
        finally:
            if not keep:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @staticmethod
    def _errors(reply: dict, ops) -> dict:
        """{op: message} for the ops of a request that failed (a reply without per-op errors fails all of them)."""
        errors = {op: message for op, message in (reply.get("errors") or {}).items() if op in ops}
        if not reply.get("ok") and not errors:
            errors = dict.fromkeys(ops, reply.get("error", "analysis failed"))
        for op, key in (("umap", "summary"), ("qc", "qc"), ("qc_start", "qc"), ("deeper", "deeper")):
            if op in ops and op not in errors and not isinstance(reply.get(key), dict):
                errors[op] = reply.get("error") or "the GPU worker sent no result"
        return errors

    # ---- worker process ------------------------------------------------------------------------------------
    def _worker_env(self) -> dict:
        """The worker's environment (engine/children.py: the app on PYTHONPATH as its environment needs it, no desktop):
        only [analysis] gpu visible, numbered in PCI bus order as nvidia-smi numbers GPUs (the mixing check's trainer
        uses the same order for its GPU: the two never meet on one card, whatever CUDA's default order would say),
        BLAS/OpenMP pools of worker_threads(), [analysis] backend, and the folder for the CPU path's compiled code (one
        the engine's own environment names is kept). The worker reads the last two only on the CPU path."""
        env = children.env(children.kind(self.python), gpu=int(self.gpu))
        threads = worker_threads()
        env.update(PYTHONUNBUFFERED="1",
                   **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), threads))
        env[BACKEND_ENV] = self.backend_setting
        env.setdefault(CPU_CACHE_ENV, str(self.cpu_cache))
        return env

    def _ensure_worker(self):
        """Start the worker unless it runs, and wait for its hello (at most 3 minutes: its imports, and on the GPU the
        card opened). Raises AnalysisError when it could not start: a process that could not be made (said once), a
        not-ready hello (retried after 5 s at the next request, as a GPU that cannot be reached may come back), or a
        lasting one: this Python cannot run Clusters & UMAP at all, and they are off for the session (`_turn_off`)."""
        if self.process is not None and self.process.poll() is None:
            return
        if self.lasting is not None:                           # this session's answer: never started again
            raise self.refusal("Clusters and the UMAP")
        if time.monotonic() - self.failed_at < 5:
            raise AnalysisError("The GPU worker failed recently; retrying shortly")
        started = time.monotonic()
        self.worker_since = None
        log = None
        try:                                                   # a python that cannot run, a state folder not writable
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "ab")
            self.process = subprocess.Popen([self.python, "-m", self.module], cwd=children.app_path(),
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                            env=self._worker_env(), text=True, bufsize=1, close_fds=True,
                                            **portable.no_window())
            portable.tie(self.process)                         # macOS and Windows: it ends with the engine
        except OSError as exc:
            if log is not None:
                log.close()
            self.process = None
            self.failed_at = time.monotonic()
            text = f"The analysis worker could not start: {type(exc).__name__}: {exc}"
            self._note(text, once="worker")
            raise AnalysisError(text) from None
        self.replies = queue.Queue()                           # this worker's own: a dead one's reader never writes here
        threading.Thread(target=self._read, args=(self.process, self.replies), name="analysis-reader",
                         daemon=True).start()
        try:
            hello = self.replies.get(timeout=180)              # its libraries' imports, and on the GPU the card opened
        except queue.Empty:
            hello = {"ok": False, "error": "the GPU worker did not start within 3 minutes"}
        refused = None
        if hello.get("ready") and (hello.get("backend") or "gpu") == "gpu":
            refused = self._wrong_gpu(hello.get("pci_bus_id"))
            if refused:
                hello = {"ok": False, "error": refused}
        if not hello.get("ready"):
            self.failed_at = time.monotonic()
            self.process.kill()
            self.process = None
            if hello.get("lasting"):
                self._turn_off(hello)
                raise self.refusal("Clusters and the UMAP")
            # A worker of the previous version (no "lasting") is retried, as before.
            who = "The GPU worker" if refused else "The analysis worker"
            self._note(f"{who} could not start: {hello.get('error', 'no reason given')}", once="worker")
            raise AnalysisError(hello.get("error", "the GPU worker could not start"))
        # A worker of the previous version says no backend: it ran on the GPU only.
        self.backend = hello.get("backend") or "gpu"
        self.backend_why = hello.get("why")
        self.worker_threads = hello.get("threads")
        self.worker_since = time.time()
        self.noted.pop("worker", None)
        took = time.monotonic() - started
        if self.backend == "cpu":
            threads = self.worker_threads or worker_threads()
            self._note(f"Analysis worker started on the CPU with {threads} threads (pid {self.process.pid}) in "
                       f"{took:.1f} s" + (f": {self.backend_why}" if self.backend_why else ""))
        else:
            bus = f", PCI bus {hello['pci_bus_id']}" if hello.get("pci_bus_id") else ""
            self._note(f"GPU worker started on GPU {self.gpu}{bus} (pid {self.process.pid}) in {took:.1f} s")

    def _turn_off(self, hello: dict):
        """A lasting not-ready hello: this Python cannot run Clusters & UMAP at all (packages missing, or its libraries
        do not load), and the same Python gives the same answer until the app's settings or packages change. They are
        off for the session: `available` is false, nothing is scheduled or started again, every run analysed here and
        every active one gets the status "unavailable" (the WebSocket pushes it; the page reads describe again), and
        the log says it once, naming what is missing."""
        error = str(hello.get("error") or "its libraries could not be loaded")
        if error.startswith(LASTING_PREFIX):                   # the worker's own sentence: its reason alone
            error = error[len(LASTING_PREFIX):] or error
        try:
            active = [run.meta["id"] for run in self.runs.active() if run.meta.get("kind") != "stain"]
        except Exception:                                      # a store that cannot list its runs: those known here
            active = []
        with self.lock:
            self.lasting = error
            self.missing = [str(name) for name in hello.get("missing") or []]
            for run_id in {*self.status, *active}:
                self.status[run_id] = {"state": "unavailable"}
            self.version += 1
            self.wake.notify_all()
        self._note(f"Clusters & UMAP are off for this session: {self.unavailable_why()}", once="worker")

    def _wrong_gpu(self, bus) -> str | None:
        """Why the worker's GPU (the PCI bus its hello names) is not [analysis] gpu as nvidia-smi numbers it, or None
        (also when either side cannot tell)."""
        got = bus_number(bus) if bus else None
        if got is None:
            return None
        expected = bus_number((self.probe() or {}).get("bus"))
        if expected is None or got == expected:
            return None
        return f"the GPU worker landed on PCI bus {got:02x}, not on GPU {self.gpu} (bus {expected:02x})"

    def _send(self, message: dict):
        """A message the worker does not answer ("forget"), to a worker that runs (none is started for it)."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    @staticmethod
    def _read(process: subprocess.Popen, replies: queue.Queue):
        for line in process.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                continue
        replies.put({"ok": False, "error": f"the GPU worker exited (code {process.wait()})"})

    def _call(self, request: dict) -> dict:
        self._ensure_worker()
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.failed_at = time.monotonic()
            self._note("The GPU worker stopped (it no longer reads requests)", once="worker")
            raise AnalysisError("the GPU worker stopped") from None
        try:
            reply = self._await_reply(300)                     # held-back tokens asked meanwhile
        except queue.Empty:
            self.failed_at = time.monotonic()
            self.process.kill()
            self._note("The GPU worker timed out (no answer within 5 minutes) and was stopped", once="worker")
            raise AnalysisError("the GPU worker timed out") from None
        if "ready" in reply and "seq" not in reply:
            return self._call(request)                          # stray hello from a restart
        return reply
