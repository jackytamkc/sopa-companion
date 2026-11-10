"""Runs: crop a region, drive the live proseg process, and index the snapshot it writes after every iteration.

Each run lives in var/runs/<id>/:
  run.json                  what was asked for and the run's state
  input/transcripts.parquet the region's transcripts in the Xenium schema
  image/                    morphology crops (c<k>.png + image.json), full resolution
  live/info.json            written once by proseg-live: schedule, counts, voxel/z geometry, file layout
  live/frames.jsonl         one line per snapshot (iteration, phase, statistics, raster geometry)
  live/frames/<seq>.bin.zst the snapshot rasters (.tiles in whole-sample runs; layout in docs/ARCHITECTURE.md)
  live/transcripts.bin.gz   the modelled transcripts (x, y, normalised z, gene)
  live/control              pause / step / speed / finish / continue / live knobs, and the counts and where tokens,
                            polled by proseg-live between iterations
  live/counts.jsonl         one line per frame proseg counted (its own count of each cell, gene by gene: --live-counts),
                            each merged into its frame's record; the files of the frames the engine asks for are in the
                            run's spool folder (engine/counts_spool.py), the durable ones in live/frames
  live/where.json           proseg's pools in a box the engine asked about (where-<token>.bin.zst), for pink dots
  live/schedule_done        written by proseg-live when it completes its schedule and holds there (--live-hold)
  output/                   proseg's own results, written when sampling finishes (SpatialData zarr, polygons, counts)
  proseg.log                proseg's console output

Every run follows proseg's own schedule (its defaults, or the lengths set under Advanced proseg options: burn-in at the
coarse voxels, sampling, recorded sampling, hill-climb) and starts paused on its prior (frame 0) until Resume or Step.
At the end of the schedule proseg holds instead of finishing: the run pauses on proseg's own result there; Finish & save
writes it, and Resume or Step samples on, open-ended ("continued"). Finish & save at any earlier iteration saves that
state.

A test's own run (engine/qc_test.py; meta `qc_test` {of, role, side, test}: is_test_run) is the exception: it reads the
test's one shared input where it is (meta `input`), so it has no input/ or image/ of its own; proseg writes only its
per-transcript file (TEST_OUTPUTS) and ends plainly at the end of the schedule, without holding there; it is never
offered for a save, the disk guard aborts it instead of pausing it, and its folder goes in the background
(delete(later=True)).

Proseg's own count (proseg-live's --live-counts, which a binary whose --help lacks it does not have: its runs start and
segment as before, without counts, meta live_counts "old_proseg"): proseg counts the frames it stops on by itself (its
first, the upscale, a pause, the hold, a finish, the end) and the next frame after each counts token the engine writes
(request_counts, for the scheduler); a where token asks for proseg's pools in a box (request_where, for Go deeper and a
test's targets), served at once while paused or held, with the next counted frame while sampling.
"""
from __future__ import annotations

import bisect
from collections.abc import Mapping
from datetime import datetime, timezone
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid

from .. import dirs, portable
from ..analysis.counts_file import count_of
from . import qc_constants, vectors
from .counts_spool import ROOT_ENV as COUNTS_ROOT_ENV, CountsSpool
from .samples import _zstd
from . import leftovers, tiles

# Voxel budget for one run: columns of the 2-D grid and voxels of the 3-D grid (each snapshot holds 2 + layers
# u32 planes, plus the extraction buffers in proseg-live).
MAX_COLUMNS = 9_000_000
MAX_VOXELS = 40_000_000

ACTIVE = ("preparing", "running", "paused", "finishing")
TERMINAL = ("done", "failed", "stopped")
SPEEDS = {"max": 0, "fast": 120, "slow": 500}  # minimum milliseconds per iteration
MIN_FREE_GB = 15
OUTPUTS = ["--output-spatialdata", "proseg-output.zarr", "--exclude-spatialdata-transcripts", "--overwrite",
           "--output-cell-polygons", "cell-polygons.geojson.gz",
           "--output-cell-metadata", "cell-metadata.csv.gz"]
# Per-transcript and per-count tables: worth having for a region, too slow and large for a whole sample (the counts
# are in proseg-output.zarr's table either way).
DETAIL_OUTPUTS = ["--output-counts", "counts.mtx.gz", "--output-transcript-metadata", "transcript-metadata.csv.gz"]
# A test's run: proseg's per-transcript table only, as parquet, which is all the test reads (engine/qc_readout.py); no
# counts, polygons or cell table. The zarr cannot be left out without a change to proseg, so it is written without its
# transcripts (about 0.13 s a run).
TEST_OUTPUTS = ["--output-spatialdata", "proseg-output.zarr", "--exclude-spatialdata-transcripts", "--overwrite",
                "--output-transcript-metadata", "transcript-metadata.parquet",
                "--output-transcript-metadata-fmt", "parquet"]
MAX_POLYGON_VERTICES = 5000
# Whole-sample runs: each 2-D plane of the snapshot grid is 4 bytes a column and proseg-live keeps a few of them
# (a full Xenium sample at 1 µm can be over 100 million columns).
MAX_TILED_COLUMNS = 250_000_000
# Time to a run's first frame (run.eta_s), fitted on test runs of 3 M to 80 M transcripts (about 1 s per million
# transcripts for each of the two steps on a many-core computer). Preparing is the helper scanning the sample's
# transcripts and assigning the region's to the prior; loading is proseg-live reading its input until it writes frame 0.
PREPARE_S = (1.5, 0.1e-6, 0.86e-6)    # seconds: fixed, per transcript of the sample, per transcript of the region
LOAD_S = (2.0, 1.0e-6)                # seconds: fixed, per transcript of the region
ETA_TICK_S = 1.0                      # while preparing or loading, the run goes out this often (eta_s counts down)
# Proseg's own count (the module docstring; native/proseg/src/live.rs).
COUNTS_FLAG = "--live-counts"         # what a proseg-live that keeps it lists in its --help (Runs.counts_capable)
HELP_TIMEOUT_S = 10.0                 # its --help is waited for this long, HELP_TRIES times HELP_RETRY_S apart (no
HELP_TRIES = 2                        # answer: taken for an older one, said in the log, and asked again next run)
HELP_RETRY_S = 1.0
RECENT_COUNTED = 64                   # the newest counted frames a run's public view names (proseg_counts.recent)
WHERES_KEPT = 4                       # proseg keeps its 4 newest where exports: the engine keeps their records
TOKEN_POLL_S = 0.02                   # request_where looks for its export this often
TOKEN_STALE_S = 10.0                  # a token still not served after this long no longer holds a new one back
WHERE_FILE = re.compile(r"where-[0-9]+\.bin\.zst")


# A proseg run asked of a sample that holds images only: the sample's detail says it (transcripts null, tools
# without "proseg"), and its stain runs need no transcripts.
IMAGE_ONLY = "This sample has no transcripts: only stain segmentation runs here"
# A stain run with no cell left to save: it found none, or every one was removed.
NO_CELLS = "There are no cells to save: this run found none, or all were removed"
ONE_RUN = "Another run is open: save it to the sample or discard it first."
# A run's start without its proseg-live, by what makes the program (locate.py's fix), in the README's own words.
MISSING_PROSEG = {"build": "proseg-live is not built yet: run native/build.sh",
                  "install": "proseg-live is not installed: install it as the README's Install section says",
                  "setting": "proseg-live is not where [proseg] binary says"}
# Every run this version makes carries it in run.json: a run.json without it was made before the one-run model.
MARKER = ("session_model", 2)
OLDER_RECHECK_S = 30.0       # a folder the last session's save may still use is looked at again this often
OLDER_DONE_KEPT = 64         # older runs answered (saved or discarded) still listed under `done` for the page's words
SAVE_WAIT_S = 3600.0         # a save into a sample is waited for this long while closing, then left to finish by itself
RECEIVE_S = 600.0            # API mode: a delivery's `received` is waited for this long (the notebook reads it)
# The older runs that cannot be saved: why (a stable reason), its words and its tooltip ({path}, {folder}, {sample}).
OLDER_WHY = {
    "sample_gone": ("{sample} is no longer where it was.",
                    "It was at {path}. Put it back and start the app again to save this run."),
    "xenium": ("Made on a Xenium folder, which cannot hold its cells.",
               "Its results are in {folder}: copy them before you discard it."),
    "patches": ("Its patch boxes were never exported.", "Its boxes are in {folder}: copy them before you discard it."),
    "missing": ("Its results are missing from its folder.", ""),
    "filtered": ("Its filters leave no cells, so there is nothing to save.", ""),
    "too_old": ("Made by a much older version: it cannot be saved.", ""),
}


def is_test_run(meta: dict) -> bool:
    """Whether a run is one of a test's own (engine/qc_test.py): its `qc_test` is {of, role, side, test}, or anything
    else but a dict. The parent's own qc_test {state, progress} (its run detail's status card) is not that mark."""
    test = meta.get("qc_test")
    return bool(test) and (not isinstance(test, dict) or bool(test.get("of")))


def _given_input(meta: dict) -> str | None:
    """The input a run reads where it is, instead of a crop of its own: a test's runs share the test's one input (meta
    `input` {path, transcripts, ids, owned, prepared}: qc_test.shared_input). None for every other run."""
    given = meta.get("input")
    path = given.get("path") if isinstance(given, dict) else None
    return str(path) if path else None


def leave_kind(meta: dict) -> dict:
    """What leaving would lose. Shared with the page through tests/data/leave_cases.json.

    `why` is a stable reason, not display text. The parent's qc_test {state, progress} is a
    status card; only a test's own run is marked as one (is_test_run).
    """
    def result(kind, why, can_add=False):
        return {"kind": kind, "can_add": can_add, "why": why}

    if is_test_run(meta):
        return result("none", "test")
    saved = meta.get("zarr_save") or {}
    if saved.get("state") == "saving" or saved.get("state") == "queued" and not saved.get("written_to"):
        return result("saving", "saving")
    kind, state = meta.get("kind", "proseg"), meta.get("state")
    if kind == "patches":
        return result("none", "patches")
    current = saved.get("state") in ("saved", "queued")
    if kind == "stain":
        progress = meta.get("progress") or {}
        model = meta.get("summary_model") or meta.get("summary") or {}
        if meta.get("stain_started") is False or not progress.get("done"):
            return result("none", "not_started")
        if model.get("cells") == 0:
            return result("none", "no_cells")
        if (meta.get("summary") or {}).get("cells") == 0:
            return result("ask", "filtered")
        exclusions = meta.get("exclusions") or {}
        same_filters = (saved.get("exclusions_digest") == exclusions.get("digest")
                        if saved.get("exclusions_digest") is not None and exclusions.get("digest") is not None
                        else saved.get("exclusions_version", 0) == exclusions.get("version", 0))
        current = current and saved.get("result_revision") == progress.get("revision") and same_filters
        if current:
            return result("saved", "saved")
    else:
        if current:
            return result("saved", "saved")
        if state in ("stopped", "failed"):
            return result("none", "ended")
        if state == "preparing" or meta.get("phase") == "loading" or meta.get("frames") == 0 and state in ACTIVE:
            return result("none", "not_started")
        if (state == "paused" and meta.get("frames") == 1 and not meta.get("steps") and
                not meta.get("finish_requested") and not meta.get("schedule_done") and
                (meta.get("last") or meta.get("ack") or {}).get("iter", 0) == 0):
            return result("none", "not_started")
    if meta.get("sample_kind") == "xenium":
        return result("ask", "sample")
    return result("ask", "changed" if saved.get("state") in ("saved", "queued", "outdated") else "unsaved", True)


def finish_and_save(meta: dict) -> bool:
    """A proseg run whose Finish & save was pressed, with no save tried since (zarr_save null) and its Save dialog not
    cancelled (`leave stay` sets save_cancelled), finishing or done: decided. Closing waits for it and saves it under new
    names whatever the answer. Pinned beside leave_kind by tests/data/leave_cases.json (each case's `decided`)."""
    if is_test_run(meta):
        return False
    return (meta.get("kind", "proseg") == "proseg" and bool(meta.get("finish_requested")) and
            meta.get("zarr_save") is None and not meta.get("save_cancelled") and
            meta.get("state") in ("finishing", "done") and meta.get("sample_kind", "zarr") == "zarr")


def sample_locked(path) -> bool:
    """Whether a save holds the sample's save lock now (zarr_bridge/helper.py _store_lock: one lock file per store in
    each lock folder, held for a save's whole build and write; a save of the previous version holds only the first),
    each tried without blocking and without touching the file's time: True when any is held."""
    if not path:
        return False
    try:
        from ..zarr_bridge.helper import _lock_paths
        targets = _lock_paths(Path(path).resolve(), create=False)
    except (OSError, ImportError, RuntimeError):
        return False
    for target in targets:
        try:
            fd = os.open(target, os.O_RDONLY | portable.O_NOFOLLOW)
        except OSError:
            continue
        try:
            portable.lock(fd, block=False)
        except BlockingIOError:
            return True
        except OSError:
            pass
        else:
            if portable.WINDOWS:
                portable.unlock(fd)  # Windows lets go of a lock at close only "when it can"
        finally:
            os.close(fd)             # closing lets go of the lock this try may have taken
    return False


def stored_meta(folder: Path, meta: dict) -> dict:
    """A cheap durable leave check for startup and the CLI, without loading a run's full frame index."""
    value = dict(meta)
    try:
        value["ack"] = json.loads((folder / "live" / "ack").read_text())
    except (OSError, ValueError):
        pass
    try:
        with (folder / "live" / "frames.jsonl").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read()
        end = tail.rfind(b"\n")
        if end >= 0:
            last = json.loads(tail[:end].splitlines()[-1])
            value.update(last=last, frames=last["seq"] + 1)
    except (OSError, ValueError, KeyError, IndexError):
        pass
    if "frames" not in value and isinstance(value.get("ack"), dict) and "seq" in value["ack"]:
        value["frames"] = value["ack"]["seq"] + 1
    return value


def has_transcripts(detail: dict) -> bool:
    """Whether a SpatialData sample's detail has transcripts for proseg: `transcripts` is null and `tools` lacks "proseg"
    for a sample that holds images only (a detail made before either key was there has transcripts)."""
    return detail.get("transcripts", 0) is not None and "proseg" in (detail.get("tools") or ("proseg",))


def prepare_seconds(sample: float, region: float) -> float:
    return PREPARE_S[0] + PREPARE_S[1] * sample + PREPARE_S[2] * region


def load_seconds(region: float) -> float:
    return LOAD_S[0] + LOAD_S[1] * region


class RunError(Exception):
    def __init__(self, message, status=422, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra          # more of the answer's JSON, beside its words (a save's name clash: save_to_zarr)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode())


def atomic_bytes(path: Path, data: bytes, durable: bool = False) -> None:
    """Write through a temporary file in the same folder (never creates the folder: a deleted run stays deleted).
    `durable`: on disk before it is renamed into place, so a crash or power loss leaves the old state or the whole
    file, never a torn one under the final name (for files served as immutable: the browser would keep a torn one)."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    with open(temporary, "wb") as stream:
        stream.write(data)
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    portable.replace(temporary, path)       # Windows: proseg may be reading the control file this very moment


def _jsonable(value):
    """`value` as strict JSON holds it: numbers that are not finite become None, anything else unknown its text (a
    delivery's provenance crosses the events pipe)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _apart(a: str, b: str) -> tuple[str, str]:
    """Two paths by their shortest ends that still tell them apart (whole folder names), for a message on the page:
    sample.zarr in two folders of the same layout is a/data/sample.zarr and b/data/sample.zarr."""
    pa, pb = Path(a).parts, Path(b).parts
    for n in range(1, max(len(pa), len(pb)) + 1):
        ends = str(Path(*pa[-n:])), str(Path(*pb[-n:]))
        if ends[0] != ends[1]:
            return ends
    return a, b


def _without_outlines(value):
    """A save's provenance without its outlines' point lists (a lasso region's, the covered part's, every removed area's):
    the deliver event stays a few kilobytes however much was drawn, while the staged result and the sample keep them
    whole on the shapes (api/deliver.py reads them from the staged shapes)."""
    if isinstance(value, dict):
        return {key: _without_outlines(item) for key, item in value.items()
                if not (key == "points" and isinstance(item, list))}
    if isinstance(value, list):
        return [_without_outlines(item) for item in value]
    return value


def user_cache() -> Path:
    """The user's own cache of the companion's tools ($XDG_CACHE_HOME/sopa-companion, else ~/.cache/sopa-companion:
    dirs.user_cache), shared by their API sessions: nothing derived from their data goes there."""
    return dirs.user_cache()


def budget_columns(values: dict, width: float, height: float) -> tuple[int, int, float, float]:
    """A run's snapshot grid for options `values` over a `width` x `height` µm region: (columns of its 2-D grid, depth
    layers, the finer of its voxel size and starting grid, quad size), each option as asked else proseg's default. The
    memory checks (Runs._check_budget) and the test of a setting's depth layers (qc_test.layers_max) count alike."""
    defaults = {o["flag"]: o["default"] for o in (_options_module().OPTIONS if _options_module() else [])}
    get = lambda flag, fallback: float(values.get(flag, defaults.get(flag)) or fallback)
    voxel = min(get("voxel-size", 1.0), get("burnin-voxel-size", 2.0))
    quad = get("quad-size", 150.0)
    layers = int(get("voxel-layers", 4))
    return math.ceil((width + quad) / voxel) * math.ceil((height + quad) / voxel), layers, voxel, quad


def _options_module():
    try:
        from . import proseg_options
        return proseg_options
    except ImportError:
        return None


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def counts_reason(run) -> str | None:
    """Why a proseg run has no proseg's own count: "old_proseg" when its proseg-live cannot keep it, "no_counts" when
    proseg started without keeping it (a run made before it was kept); None when it is kept, or proseg has not said yet
    (no info.json), and for stain runs and training patches."""
    meta = run.meta
    if meta.get("kind", "proseg") != "proseg":
        return None
    if meta.get("live_counts") == "old_proseg":
        return "old_proseg"
    if isinstance(run.info, dict) and not isinstance(run.info.get("counts"), dict):
        return "no_counts"
    return None


def where_box(box) -> list[float]:
    """A where request's box [x0, y0, x1, y1] (µm) as the control file gives it to proseg (2 decimals), checked as a
    view is (RunError 422)."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise RunError("box must be [x0, y0, x1, y1] in µm") from None
    # proseg reads them as float32: a line with a number beyond that is ignored, its token never served
    if not all(math.isfinite(v) and abs(v) < 1e30 for v in (x0, y0, x1, y1)):
        raise RunError("The box is empty")
    x0, y0, x1, y1 = (round(v, 2) for v in (x0, y0, x1, y1))
    if x1 <= x0 or y1 <= y0:
        raise RunError("The box is empty")
    return [x0, y0, x1, y1]


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _counts_line(line) -> bool:
    """A counts.jsonl line the engine can merge: its frame's seq, its file and its sidecar's name."""
    return isinstance(line, dict) and _whole(line.get("seq")) and line["seq"] >= 0 \
        and isinstance(line.get("file"), str) and bool(line["file"]) and isinstance(line.get("counted_file"), str)


# A counted frame's record keys only the engine reads (read_counts): its whole counts line and its count file (for a
# frame counted when asked, an absolute path in the spool on the RAM disk). The page reads counted_file alone (main.js
# frameCounted).
ENGINE_FRAME_KEYS = ("counts", "counts_file")


def page_frame(record):
    """A frame record as pages get it (GET /runs/{id}/frames, the socket's frames, a run's `last`): without
    ENGINE_FRAME_KEYS, which add about half again to a counted frame's record, and with `count` on a counted one: the
    count every number of it reads (counts_file.count_of of its counts line: "sampled", the count proseg saves, for a
    frame proseg counted for a pause, the hold, a finish or the end, else "averaged"). The run keeps its records whole;
    a record is replaced, never changed in place (read_counts), so a counted one is copied and the others go as they
    are."""
    if isinstance(record, dict) and any(key in record for key in ENGINE_FRAME_KEYS):
        out = {key: value for key, value in record.items() if key not in ENGINE_FRAME_KEYS}
        if "counts" in record:
            out["count"] = count_of(record["counts"])
        return out
    return record


def page_info(info):
    """proseg's info.json as pages get it (a run's `info`): without its counts block's spool folder, a path on the RAM
    disk only the engine reads (engine/analysis.py's prune). The run keeps it whole."""
    counts = info.get("counts") if isinstance(info, dict) else None
    if isinstance(counts, dict) and "spool" in counts:
        return {**info, "counts": {key: value for key, value in counts.items() if key != "spool"}}
    return info


def _where_record(record, live: Path) -> dict | None:
    """A where export's record (where.json, or a counts line's "where") with "path" (its file, absolute) added; None for
    one the engine cannot use (a field missing or of the wrong kind, a file name that is not proseg's)."""
    if not isinstance(record, dict) or not WHERE_FILE.fullmatch(str(record.get("file", ""))):
        return None
    try:
        if not all(_whole(record[key]) for key in ("token", "seq", "nx", "ny", "nz", "i0", "j0", "i1", "j1")) \
                or record["token"] < 1 or record["seq"] < 0:
            return None
        if not all(math.isfinite(float(record[key])) for key in ("x0", "y0", "voxel")):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {**record, "path": str(live / record["file"])}


class Run:
    def __init__(self, directory: Path, meta: dict):
        self.dir, self.meta = directory, meta
        self.frames: list[dict] = []
        self.info: dict | None = None
        self.offset = 0
        self.process: subprocess.Popen | None = None
        self.stop_requested = False
        self.lock = threading.RLock()
        self.version = 0
        self.frame_bytes = 0
        self.ack: dict | None = None
        self.ack_mtime = 0
        self.window: dict | None = None
        self.window_mtime = 0
        self.worker: threading.Thread | None = None
        self.saver: threading.Thread | None = None
        self.leave_worker: threading.Thread | None = None
        self.leave_token = None
        self.leave_lock = threading.Lock()
        self.leave_revisions: dict = {}           # page -> the newest revision of its answers taken (Runs.leave)
        self.follower: threading.Thread | None = None   # API mode: a delivery's received and applied (Runs._follow)
        self.removed = False                   # deleted (Runs._delete_run): its folder goes, nothing is written there
        self.expected: dict = {}               # transcripts before preparing counts them: {"sample", "region"}
        self.started_at: float | None = None   # time.monotonic() when preparing began
        self.spawned_at: float | None = None   # ... and when proseg-live started (loading)
        self.ticked = 0.0
        self.spool: CountsSpool | None = None  # the engine's spool (Runs sets it): the files of read lines go there
        self.spool_lost = False                # its spool folder could not be made: no counted frame is asked for
        self.reset_counts()

    def reset_counts(self):
        """Nothing of proseg's counts read yet (proseg starts, or starts again: the caller holds the lock). `wheres`:
        the records of the WHERES_KEPT newest where exports (by token), each with its "path"; `counted`: the frames
        whose counts lines are merged, in order; `counts_served`, `where_served`: the newest tokens a line or an export
        carried; `newest_at`: time.monotonic() when the newest counted frame's line was merged; `sampled`: the counted
        frames read on the count proseg saves (counts_file.count_of "sampled"), in order."""
        self.counts_offset = 0
        self.held_counts: dict[int, tuple[dict, int]] = {}   # seq -> (line, bytes it adds), until its frame is read
        self.counts_bytes: dict[int, int] = {}               # seq -> what its count adds to frame_bytes
        self.counted: list[int] = []
        self.sampled: list[int] = []
        self.counts_served = self.where_served = 0
        self.counts_issued_at = self.where_issued_at = 0.0
        self.newest_at: float | None = None
        self.wheres: list[dict] = []
        self.where_stamp = None

    @property
    def live(self) -> Path:
        return self.dir / "live"

    def save(self):
        atomic_text(self.dir / "run.json", json.dumps(self.meta, indent=1))

    def touch(self):
        self.version += 1

    def phase(self) -> str:
        """What the run is doing, for the page: preparing (its input), loading (proseg-live started, no frame yet: it
        is reading its input, whether a pause or Finish & save was asked for or not: it can do neither before its first
        frame), sampling, paused, finishing, or its final state (done, failed, stopped). Not the frame records' own
        `phase` (burnin, ...): this one says what the engine is waiting for."""
        state = self.meta.get("state")
        if self.meta.get("kind") == "stain":
            return state
        if state in ("running", "paused", "finishing") and not self.frames and self.spawned_at is not None:
            return "loading"
        return "sampling" if state == "running" else state

    def eta_s(self) -> int | None:
        """Expected seconds until the first frame (PREPARE_S, LOAD_S) while preparing or loading; None otherwise,
        without a transcript count, or once the estimate has passed."""
        phase = self.phase()
        region = self.meta.get("transcripts") or self.expected.get("region")
        if phase not in ("preparing", "loading") or not region:
            return None
        if phase == "preparing":
            if self.started_at is None:
                return None
            spent = time.monotonic() - self.started_at
            left = max(0.0, prepare_seconds(self.expected.get("sample") or region, region) - spent) + load_seconds(region)
        else:
            if self.spawned_at is None:
                return None
            left = load_seconds(region) - (time.monotonic() - self.spawned_at)
        return math.ceil(left) if left > 0 else None

    def tick(self):
        """Preparing or loading: send the run again (its eta_s) at most every ETA_TICK_S, while there is an estimate
        to count down (past it the page just says loading: nothing new to send)."""
        now = time.monotonic()
        if now - self.ticked >= ETA_TICK_S and self.eta_s() is not None:
            self.ticked = now
            self.touch()

    def public(self) -> dict:
        """The run as the page reads it (GET /runs, GET /runs/{id}, the socket's run pushes, the run actions' answers).
        `proseg_counts`: why it has no proseg's count ({reason}: counts_reason), its newest counted frame and the
        RECENT_COUNTED newest, in order (one counted after the page read its record), and the RECENT_COUNTED newest read
        on the count proseg saves (`sampled`: counted for a pause, the hold, a finish or the end), in order. `constants`:
        the mixing check's constants block of the run's mode (engine/qc_constants.py block: mode, id, measured, and the
        constants kept, not measured, there: the page's caveats). `last`, its newest frame's record,
        and `info` as pages get them (page_frame, page_info: without what only the engine reads, which the engine's own
        readers of this record never need: leave_kind, the closing rows and the run log read last's iter and info's
        schedule)."""
        with self.lock:
            last = page_frame(self.frames[-1]) if self.frames else None
            disk = self.frame_bytes + self.meta.get("input_bytes", 0) + sum(o.get("bytes", 0) for o in self.meta.get("outputs", []))
            counts = {"reason": counts_reason(self), "newest": self.counted[-1] if self.counted else None,
                      "recent": self.counted[-RECENT_COUNTED:], "sampled": self.sampled[-RECENT_COUNTED:]}
            return {**self.meta, "frames": len(self.frames), "last": last, "info": page_info(self.info), "disk_bytes": disk,
                    "ack": self.ack, "window": self.window, "phase": self.phase(), "eta_s": self.eta_s(),
                    "proseg_counts": counts, "constants": qc_constants.block(qc_constants.run_mode(self.meta, self.info))}

    def read_new(self) -> bool:
        """Consume complete lines appended to frames.jsonl and counts.jsonl (and info.json once it exists, the ack,
        window.json and where.json as they change). A counts line is merged into its frame's record once that record is
        read (read_counts)."""
        changed = False
        try:   # tiled runs: per-layer labels for the paused state after the view moved
            stamp = (self.live / "window.json").stat().st_mtime_ns
            if stamp != self.window_mtime:
                self.window, self.window_mtime, changed = json.loads(portable.read_text(self.live / "window.json")), stamp, True
        except (OSError, ValueError):
            pass
        try:   # proseg's acknowledgement of pause/finish: written only once the state on screen is committed
            stamp = (self.live / "ack").stat().st_mtime_ns
            if stamp != self.ack_mtime:
                self.ack, self.ack_mtime, changed = json.loads(portable.read_text(self.live / "ack")), stamp, True
        except (OSError, ValueError):
            pass
        if self.info is None:
            try:
                self.info = json.loads(portable.read_text(self.live / "info.json"))
                changed = True
            except (OSError, ValueError):
                pass
        changed = self.read_where() or changed
        chunk = b""
        try:
            with open(self.live / "frames.jsonl", "rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
        except OSError:
            pass
        end = chunk.rfind(b"\n")
        frames, size = [], 0
        for line in chunk[:end].splitlines() if end >= 0 else ():
            try:
                frame = json.loads(line)
            except ValueError:
                continue
            frames.append(frame)
            try:
                size += (self.live / frame["file"]).stat().st_size
            except (OSError, KeyError, TypeError):
                pass
        if end >= 0:
            with self.lock:
                self.frame_bytes += size
                self.offset += end + 1
                self.frames.extend(frames)
        merged = self.read_counts()           # after the frames: a line whose frame was just read is merged at once
        with self.lock:
            self.served_counts()              # a paused state's line, or its ack, may have come
        return changed or bool(frames) or merged

    def read_counts(self) -> bool:
        """Consume complete lines appended to counts.jsonl (proseg's commit point for each counted frame: its file, its
        sidecar and any where export it carries are in place) and merge each into its frame's record once that record
        is read: "counts_file" (the line's file: relative to live/ in the run's folder, else absolute in the spool),
        "counted_file" (its sidecar, always in live/frames) and the whole line as "counts". A record is replaced, never
        changed in place (a reader may hold the old one), and a later line of the same frame replaces the earlier
        (proseg counts a frame again: with f for a pause on a frame a token counted, since proseg-live 3.2.0.2, and
        durably for the hold, a finish or the end). The files in the run's folder count toward its disk bytes; the
        spool's are noted for pruning (CountsSpool.note). True when a line was merged."""
        chunk, found = b"", []
        try:
            with open(self.live / "counts.jsonl", "rb") as stream:
                stream.seek(self.counts_offset)
                chunk = stream.read()
        except OSError:
            pass
        end = chunk.rfind(b"\n")
        for text in chunk[:end].splitlines() if end >= 0 else ():
            try:
                line = json.loads(text)
            except ValueError:
                continue
            if not _counts_line(line):
                continue
            size = 0 if os.path.isabs(line["file"]) else max(0, line["bytes"]) if _whole(line.get("bytes")) else 0
            try:
                size += (self.live / line["counted_file"]).stat().st_size
            except (OSError, ValueError):
                pass
            found.append((line, size))
        wheres = [_where_record(line.get("where"), self.live) for line, _ in found]
        notes, merged = [], False
        with self.lock:
            if end >= 0:
                self.counts_offset += end + 1
            for line, size in found:
                self.held_counts[line["seq"]] = (line, size)
            for record in wheres:
                if record is not None:
                    self._keep_where(record)
            for seq in sorted(seq for seq in self.held_counts if seq < len(self.frames)):
                line, size = self.held_counts.pop(seq)
                self.frames[seq] = {**self.frames[seq], "counts_file": line["file"],
                                    "counted_file": line["counted_file"], "counts": line}
                self.frame_bytes += size - self.counts_bytes.get(seq, 0)
                self.counts_bytes[seq] = size
                at = bisect.bisect_left(self.counted, seq)
                if at == len(self.counted) or self.counted[at] != seq:
                    self.counted.insert(at, seq)
                at = bisect.bisect_left(self.sampled, seq)
                listed = at < len(self.sampled) and self.sampled[at] == seq
                if count_of(line) == "sampled" and not listed:     # a later line of a seq counts (a pause's, a hold's)
                    self.sampled.insert(at, seq)
                elif count_of(line) != "sampled" and listed:
                    del self.sampled[at]
                if seq == self.counted[-1]:
                    self.newest_at = time.monotonic()
                if _whole(line.get("token")) and line["token"] > self.counts_served:
                    self.counts_served = line["token"]
                if os.path.isabs(line["file"]):
                    notes.append(line["file"])
                merged = True
        if self.spool is not None:
            for path in notes:
                self.spool.note(self.meta.get("id"), path)
        return merged

    def read_where(self) -> bool:
        """where.json, proseg's record of its newest where export (written after its file: its commit point), read when
        it changed; one being replaced (an ntfs-3g rename gap) or half there is read again next time."""
        path = self.live / "where.json"
        try:
            info = path.stat()
        except OSError:
            return False
        stamp = (info.st_mtime_ns, info.st_ino, info.st_size)
        if stamp == self.where_stamp:
            return False
        try:
            record = _where_record(json.loads(path.read_text()), self.live)
        except (OSError, ValueError):
            return False
        with self.lock:
            self.where_stamp = stamp
            return record is not None and self._keep_where(record)

    def _keep_where(self, record: dict) -> bool:
        """Keep a where export's record (the caller holds the lock): the WHERES_KEPT newest by token, one per token, in
        a new list (a reader may hold the old one). Its token is served. True when the records kept changed."""
        kept = sorted([w for w in self.wheres if w["token"] != record["token"]] + [record], key=lambda w: w["token"])
        kept = kept[-WHERES_KEPT:]
        self.where_served = max(self.where_served, record["token"])
        if kept == self.wheres:
            return False
        self.wheres = kept
        return True

    def served_counts(self) -> int:
        """The newest counts token served (the caller holds the lock): the newest a merged line carried, and every one
        issued while proseg is paused (or finishing) on a frame whose counts line is merged, which proseg serves with
        that frame's count, counting nothing more for it (an ack left from an earlier pause, while the run samples
        again, serves none). Kept as served from then on."""
        ack = self.ack or {}
        state, seq = ack.get("state"), ack.get("seq")
        if state in ("paused", "finishing") and self.meta.get("state") == state and _whole(seq) \
                and 0 <= seq < len(self.frames) and "counts" in self.frames[seq]:
            self.counts_served = max(self.counts_served, int(self.meta.get("counts_token") or 0))
        return self.counts_served


def _polygon_area(points) -> float:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(points)))) / 2


def region_spec(region, extent) -> dict:
    """A validated region in the working frame: {kind, box | points, bbox}. Boxes and bounding boxes are clamped
    to the sample's extent."""
    if not isinstance(region, dict):
        raise RunError("The region must be a rectangle, a lasso polygon or the full sample")
    ex0, ey0, ex1, ey1 = (float(v) for v in extent)
    kind = region.get("kind")
    if kind == "full":
        return {"kind": "full", "bbox": [ex0, ey0, ex1, ey1]}
    if kind == "box":
        try:
            x0, y0, x1, y1 = (float(v) for v in region.get("box"))
        except (TypeError, ValueError):
            raise RunError("The region must be four numbers: x0, y0, x1, y1 (µm)") from None
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            raise RunError("The region is empty")
        x0, y0, x1, y1 = max(ex0, x0), max(ey0, y0), min(ex1, x1), min(ey1, y1)
        if min(x1 - x0, y1 - y0) < 20:
            raise RunError("Draw a region at least 20 µm across, over the tissue")
        return {"kind": "box", "box": [x0, y0, x1, y1], "bbox": [x0, y0, x1, y1]}
    if kind == "polygon":
        points = region.get("points")
        try:
            points = [[round(float(x), 2), round(float(y), 2)] for x, y in points]
        except (TypeError, ValueError):
            raise RunError("A lasso must be a list of [x, y] points (µm)") from None
        if not 3 <= len(points) <= MAX_POLYGON_VERTICES or not all(math.isfinite(v) for p in points for v in p):
            raise RunError(f"A lasso needs 3 to {MAX_POLYGON_VERTICES} points")
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, y0, x1, y1 = max(ex0, min(xs)), max(ey0, min(ys)), min(ex1, max(xs)), min(ey1, max(ys))
        if min(x1 - x0, y1 - y0) < 20 or _polygon_area(points) < 400:
            raise RunError("Draw a lasso at least 20 µm across, over the tissue")
        return {"kind": "polygon", "points": points, "bbox": [x0, y0, x1, y1]}
    raise RunError("Unknown region kind")


def region_fraction(spec: dict, extent) -> float:
    """The share of the sample's extent a validated region covers (its transcripts are estimated in proportion)."""
    ex0, ey0, ex1, ey1 = (float(v) for v in extent)
    whole = (ex1 - ex0) * (ey1 - ey0)
    if spec.get("kind") == "full" or whole <= 0:
        return 1.0
    if spec.get("kind") == "polygon":
        area = _polygon_area(spec["points"])
    else:
        x0, y0, x1, y1 = spec["bbox"]
        area = (x1 - x0) * (y1 - y0)
    return min(1.0, max(0.0, area / whole))


class Runs:
    def __init__(self, config: dict, samples, zarr=None, noted=None, session=None):
        """`noted`: where what went wrong is noted for the Log (engine/runlog.py Noted; server/app.py makes one).
        `session`: an API session's engine.session.Session, None on the desktop: a save then also goes to the
        notebook's object (the deliver path, _save_job), and closing follows the session's answer (close_session)."""
        self.noted = noted
        self.session = session
        # API mode: [session] patches_dir: where the export dialog suggests a folder per sample and resolution, and
        # where earlier sets are found; None: engine/patch_folders.py default_home.
        self.patches_dir = (config.get("session") or {}).get("patches_dir") or None
        self.root = Path(config["paths"]["data_dir"]) / "runs"
        self.state_dir = Path(config["paths"]["state_dir"])      # the export dialog refuses it, as the data folder
        self.root.mkdir(parents=True, exist_ok=True)
        self.binary = config["proseg"]["binary"]
        # What makes a missing program (locate.py): its path is where that puts it, looked at again at every start.
        self.binary_fix = ((config.get("locate") or {}).get("proseg") or {}).get("fix") or "build"
        self.threads = config["proseg"]["threads"]
        self.tiled_threads = int(config["proseg"].get("tiled_threads", self.threads))
        self.max_region = float(config["proseg"]["max_region_um"])
        self.max_active = int(config["proseg"].get("max_active_runs", 2))
        self.samples = samples
        self.zarr = zarr
        self.save_locks: dict[str, threading.Lock] = {}
        self.vector_guards: dict[str, threading.Lock] = {}     # one prior-vectors build per run at a time
        self.savers: list[threading.Thread] = []
        self.runs: dict[str, Run] = {}
        self.indexes: dict = {}                # run id -> TranscriptIndex (mapped) while the run exists
        self.held = leftovers.holders          # who still holds files of a deleted run (tests replace it)
        self.lock = threading.Lock()
        self.creation_lock = threading.Lock()
        self.version = 0
        self.frames_version = 0
        self.closing = False
        self.keep_unsaved = bool(config["proseg"].get("keep_unsaved", False))
        self.one_run = not self.keep_unsaved
        # Proseg's own count: the spool of the frames the engine asks proseg to count (engine/counts_spool.py; made now,
        # sweeping what engines that are gone left there), whether each proseg-live keeps it (counts_capable: its
        # answer per file), and whether the spool's disk was said to be low (said once until there is room again).
        self.spool = CountsSpool(os.environ.get(COUNTS_ROOT_ENV) or None,
                                 int((config.get("analysis") or {}).get("counts_spool_mb", 1024)))
        self.capable: dict[tuple, bool] = {}
        self.spool_low = False
        self.leftovers = None
        self.gone: dict[str, str] = {}
        self.departed: dict[str, dict] = {}
        self.on_delete = None
        # API mode: the results a save built, staged for the notebook (deleted once it has read them).
        self.deliveries = self.root.parent / "deliveries"
        self.followers: list[threading.Thread] = []
        self.receipts: dict[str, threading.Event] = {}      # delivery id -> set once the notebook received it (or not)
        self.followers_done = threading.Event()             # set by the final close: no delivery is followed after it
        # A save of the last session still at work (tests replace them): a process naming a folder, a save lock taken.
        self.named = leftovers.named
        self.locked = sample_locked
        # Runs made before the one-run model, apart from the registry (the one-run check, the boot, pending() and
        # close never see them), with their answers (older_state: answer, stage, message, later, still_saving,
        # can_add, why), those answered for good (older_done) and the one thread that carries the answers out, one
        # run at a time.
        self.older_runs: dict[str, Run] = {}
        self.older_state: dict[str, dict] = {}
        self.older_done: dict[str, dict] = {}
        self.older_queue: list[str] = []
        self.older_lock = threading.RLock()
        self.older_worker: threading.Thread | None = None
        self.older_current: str | None = None
        self.older_stop = False
        self.older_version = 0
        # Folders a save of the last session may still use, left as they are until it ends (folder -> entry).
        self.still_saving: dict[str, dict] = {}
        self.recheck_wake = threading.Event()
        self.rechecker: threading.Thread | None = None
        for folder in (self.root / ".trash").glob("*"):
            if folder.is_dir() and not folder.is_symlink():
                leftovers.remove_later(folder, held=self.held)
        from .stain import StainRuns
        self.stain = StainRuns(self, config.get("stain", {}), config.get("zarr", {}).get("python"))
        if session is not None:
            # A session's own folder goes when it closes: InstanSeg's downloaded weights are kept in the user's cache
            # instead, so a session neither fetches them again nor, offline, goes without them.
            self.stain.instanseg_models = user_cache() / "instanseg-models"
        self._start()
        # The safe point for what deleted and discarded runs left (engine/leftovers.py): nothing of this engine holds
        # a file yet, and the engine lock keeps any other engine away from these runs.
        swept = leftovers.sweep(self.root, held=lambda folders: self.held(folders))
        if swept["hidden"] or swept["folders"]:
            print(f"SoPA companion: removed {len(swept['folders'])} folder(s) and {len(swept['hidden'])} file(s) "
                  "(.fuse_hidden, unfinished builds) left by deleted runs", flush=True)
        if swept["left"]:      # never removed (not a run, nor only what a deleted run leaves): someone should look
            sizes = []
            for folder in swept["left"]:
                try:
                    sizes.append(f"{Path(folder).name} ({_size(Path(folder)) / 1e6:,.0f} MB)")
                except OSError:
                    sizes.append(Path(folder).name)
            print(f"SoPA companion: kept {len(sizes)} folder(s) in {self.root} that are not runs (no run.json) or are "
                  f"still in use: {', '.join(sizes)}", flush=True)

    # ---- start: what the last session left, and runs from before the one-run model -----------------------------------
    def _start(self):
        """Sort the run folders found at start. A test engine (`keep_unsaved`) loads every run as before (harness fixtures
        made before the marker keep working). A user engine keeps none in its registry: a folder this version made (the
        marker) is a leftover, removed and reported once (health.leftovers); an older run (made before) is listed for
        the user's answer when it holds a result never saved (Runs.older), else removed and counted. A folder that a
        save of the last session may still use is left as it is, said so, and sorted once it is free."""
        found = []
        for path in sorted(self.root.glob("*/run.json")):
            if path.parent.is_symlink() or path.parent.name == ".trash":
                continue
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and isinstance(meta.get("id"), str):
                found.append((path.parent, meta))
        if not self.one_run:
            for folder, meta in found:
                self.runs[meta["id"]] = self._loaded(folder, meta)
            return
        busy = self._in_use(found)
        rows, removed = [], 0
        for folder, meta in found:
            marked = meta.get(MARKER[0]) == MARKER[1]
            if str(folder) in busy:
                self.still_saving[str(folder)] = {"folder": folder, "meta": meta, "marked": marked}
                print(f"SoPA companion: left run {meta['id']} of {meta.get('sample_name') or 'a sample'} as it is: the last "
                      "session may still be saving it", flush=True)
                if marked:
                    rows.append(self._leftover_row(folder, meta, still_saving=True))
                else:
                    self._list_older(Run(folder, meta), None, still_saving=True)
            elif marked:
                row = self._leftover(folder, meta)
                if row is not None:
                    rows.append(row)
            elif not self._sort_older(folder, meta):
                removed += 1
        if rows or removed:
            self.leftovers = {"stamp": f"{now()}-{time.time_ns()}", "runs": rows, "older_removed": removed}
        if self.still_saving:
            self.rechecker = threading.Thread(target=self._watch_still_saving, name="still-saving", daemon=True)
            self.rechecker.start()

    @staticmethod
    def _fixed(run: Run) -> Run:
        """A run of the last session as it is now: one that was active was interrupted (stopped), a save under way failed
        (each written back); its `source` and `leaving` go."""
        meta = run.meta
        meta.pop("source", None)
        meta.pop("leaving", None)
        if meta.get("kind") == "stain" and meta.get("stain_started") is False and meta.get("state") == "paused":
            run.ack = {"state": "paused", "tiles": 0}
        elif meta.get("state") in ACTIVE:
            meta.update(state="stopped", message="Interrupted: the engine stopped while this run was active.")
            run.save()
        if (meta.get("zarr_save") or {}).get("state") == "saving":
            meta["zarr_save"].update(state="failed", message="Interrupted: the engine stopped while saving. Save again.")
            run.save()
        return run

    def _loaded(self, folder: Path, meta: dict) -> Run:
        """A test engine's run, kept from the last session as before."""
        run = self._fixed(Run(folder, meta))
        run.spool = self.spool
        run.read_new()
        return run

    def _in_use(self, entries) -> set[str]:
        """Which of [(folder, meta)] a save of the last session may still use: the folders a process of this user holds
        (open files, mappings, working folder: leftovers.holders), names on its command line (a save helper gets the
        run's output folder as an argument: leftovers.named) or whose sample's save lock is taken (sample_locked). A save helper of the last
        session has no pdeathsig and carries on alone: it may still read a run's output by path."""
        if not entries:
            return set()
        folders = [str(folder) for folder, _ in entries]
        busy: set[str] = set()
        try:
            held = self.held([Path(f) for f in folders])
        except Exception:                      # no /proc to read: nothing can be told apart
            held = set()
        for folder in folders:
            places = {folder, os.path.realpath(folder)}
            if any(p == place or p.startswith(place + os.sep) for p in held for place in places):
                busy.add(folder)
        try:
            busy |= {str(f) for f in self.named(folders)}
        except Exception:
            pass
        for folder, meta in entries:
            if str(folder) not in busy and self.locked(meta.get("sample_path")):
                busy.add(str(folder))
        return busy

    @staticmethod
    def _leftover_row(folder: Path, meta: dict, failed: str | None = None, still_saving: bool = False) -> dict:
        """A line of the start notice (health.leftovers.runs): {sample, sample_name, kind, state, iter, tiles, failed: the
        reason a save failed while the app closed, else None, still_saving}."""
        durable = stored_meta(folder, meta)
        progress = durable.get("progress") or {}
        return {"id": meta["id"], "sample": meta.get("sample"), "sample_name": meta.get("sample_name"),
                "kind": meta.get("kind", "proseg"), "state": meta.get("state"),
                "iter": (durable.get("last") or durable.get("ack") or {}).get("iter"),
                "tiles": {"done": progress.get("done", 0), "of": progress.get("total", 0)},
                "failed": failed, "still_saving": still_saving}

    def _leftover(self, folder: Path, meta: dict) -> dict | None:
        """A folder this version made, found at start: moved into .trash/ and emptied in the background (a start with
        dozens of GB of runs stays within the launcher's health wait). Its notice line when it held something to lose
        or its save failed while the app closed; None otherwise."""
        failed = None
        if meta.get("closed") == "save failed":
            failed = (meta.get("zarr_save") or {}).get("message") or "the save failed"
        kind = leave_kind(stored_meta(folder, meta))["kind"]
        if kind == "saving" and failed is None:
            # A save the engine left under way (closed "save running", or a crash), whose helper has ended since: in the
            # sample, nothing was lost; else it is said as a save that failed.
            if self._save_landed(meta):
                kind = "saved"
            else:
                failed = "it did not finish"
        noted = (failed is not None or kind in ("ask", "saving")) and not is_test_run(meta)
        row = self._leftover_row(folder, meta, failed) if noted else None
        try:
            folder = leftovers.trash(folder)
        except OSError:
            pass
        leftovers.remove_later(folder, held=self.held)
        print(f"SoPA companion: removed leftover run {meta['id']} of {meta.get('sample_name') or 'a sample'} when the "
              f"engine started" + (f" (its save failed when the app closed: {failed})" if failed else ""), flush=True)
        return row

    @staticmethod
    def _save_landed(meta: dict) -> bool:
        """Whether the save a run's record says was under way (zarr_save "saving") is in its sample now, committed: the
        element it was writing names the run (a stain save's shapes, at the revision it saved; a proseg save's table)
        and no save cut off before it committed lists it (zarr_samples.unfinished_writes). Read from the sample's own
        files, without the helper; False when they cannot say."""
        from .zarr_samples import shapes_provenance, table_run, unfinished_writes
        saved, path = meta.get("zarr_save") or {}, meta.get("sample_path")
        stain = meta.get("kind") == "stain"
        kind, key = ("shapes", saved.get("shapes_key")) if stain else ("tables", saved.get("table_key"))
        if saved.get("state") != "saving" or not path or not key:
            return False
        try:
            if stain:
                record = shapes_provenance(path, key) or {}
                made = record.get("run") == meta.get("id") and record.get("result_revision") == saved.get("result_revision")
            else:
                made = table_run(path, key) == meta.get("id")
            return made and (kind, key) not in unfinished_writes(path)
        except Exception:
            return False

    # ---- runs made before the one-run model -------------------------------------------------------------------------
    def _sort_older(self, folder: Path, meta: dict, later: bool = False) -> bool:
        """An older run found at start (or freed by the last session's save, `later`: its row was set aside by Later):
        listed when it holds a result never saved (True), else removed (False)."""
        run = self._fixed(Run(folder, meta))
        verdict = self._older_verdict(run)
        if verdict is None:
            self._remove_older(run)
            print(f"SoPA companion: removed run {meta['id']} of {meta.get('sample_name') or 'a sample'} from an earlier "
                  "version: it was saved already or had nothing to save", flush=True)
            return False
        self._list_older(run, verdict, later=later)
        return True

    def _list_older(self, run: Run, verdict, still_saving: bool = False, later: bool = False):
        can_add, why = verdict if verdict is not None else (False, "still_saving")
        durable = stored_meta(run.dir, run.meta)
        with self.older_lock:
            self.older_runs[run.meta["id"]] = run
            self.older_state[run.meta["id"]] = {
                "answer": None, "stage": None, "message": None, "later": later, "still_saving": still_saving,
                "can_add": can_add, "why": why, "iter": (durable.get("last") or durable.get("ack") or {}).get("iter")}
        self._older_changed()

    def _older_verdict(self, run: Run):
        """(can_add, why) of an older run that holds a result never saved; None for one with nothing to save. Listed: a
        proseg run done and not saved or whose save failed; a stain run with finished tiles (joined or not; stopped, done
        or failed), not saved, whose save failed or that changed since it was saved; a finished run on a Xenium folder
        and a Training patches item whose boxes were never exported, with Discard only. What can be saved is checked
        without opening a zarr: the first refusal a save would give (an unjoined stain run is joined first), or the
        sample's folder gone."""
        meta = run.meta
        kind = meta.get("kind", "proseg")
        if kind == "patches":
            return (False, "patches") if self._unexported(run) else None
        predicate = leave_kind(stored_meta(run.dir, meta))
        if predicate["kind"] != "ask":
            return None
        if predicate["why"] in ("sample", "filtered"):
            return False, "xenium" if predicate["why"] == "sample" else "filtered"
        if not meta.get("sample") or not meta.get("sample_path") or \
                kind == "stain" and not isinstance(meta.get("stain"), dict):
            return False, "too_old"
        try:
            if not Path(meta["sample_path"]).is_dir():
                return False, "sample_gone"
        except OSError:
            return False, "sample_gone"
        progress = meta.get("progress") or {}
        if kind == "stain":
            needed = [run.dir / "output" / "boundaries.geojson"] if progress.get("resolved") else \
                [run.dir / "stain" / "job.json", run.dir / "stain" / "tiles.json"]
        else:
            needed = [run.dir / "output" / "proseg-output.zarr"]
        if not all(path.exists() for path in needed):
            return False, "missing"
        return True, predicate["why"]

    @staticmethod
    def _unexported(run: Run) -> bool:
        """A Training patches item holding boxes that were never exported (its draft has patches and no exported set)."""
        try:
            draft = json.loads((run.dir / "patches.json").read_text())
        except (OSError, ValueError):
            return False
        return isinstance(draft, dict) and bool(draft.get("patches")) and not draft.get("set")

    def _older_changed(self):
        self.older_version += 1

    def _older_row(self, run_id: str) -> dict:
        """One older run's row (GET /runs/older): the run as its run.json holds it, what can be done and what is
        under way. `why` is a stable reason; `reason` and `detail` its words and tooltip when it cannot be saved."""
        run, state = self.older_runs[run_id], self.older_state[run_id]
        meta = run.meta
        kind = meta.get("kind", "proseg")
        progress = meta.get("progress") or {}
        method = (meta.get("stain") or {}).get("method", "cellpose") if kind == "stain" else kind
        reason, detail = ("Still being saved by the last session", "") if state["still_saving"] else \
            OLDER_WHY.get(state["why"], ("", "")) if not state["can_add"] else ("", "")
        folder = run.dir / "output" if state["why"] == "xenium" else run.dir
        words = {"sample": meta.get("sample_name") or "The sample", "path": meta.get("sample_path") or "",
                 "folder": str(folder)}
        return {"id": run_id, "sample": meta.get("sample"), "sample_name": meta.get("sample_name"),
                "sample_path": meta.get("sample_path"), "kind": kind, "method": method, "state": meta.get("state"),
                "iter": state["iter"], "tiles": {"done": progress.get("done", 0), "of": progress.get("total", 0)},
                "created": meta.get("created"), "when": meta.get("created"), "region": meta.get("region"),
                "prior": meta.get("prior"), "options": meta.get("options"), "stain": meta.get("stain"),
                "can_add": state["can_add"] and not state["still_saving"], "why": state["why"],
                "reason": reason.format(**words), "detail": detail.format(**words), "folder": str(run.dir),
                "still_saving": state["still_saving"], "answer": state["answer"], "stage": state["stage"],
                "message": state["message"], "zarr_save": deepcopy(meta.get("zarr_save")), "later": state["later"]}

    def older(self) -> dict:
        """The older runs (GET /runs/older, health.older): {runs: [row], newest first; done: [rows answered for good
        this session: stage "saved", "saved_already" or "discarded", with their message]; count: the rows not set aside
        by Later; version: grows with every change (push {ev: "older", runs} when it does)}. Empty on a test engine and
        in an API session."""
        with self.older_lock:
            rows = [self._older_row(run_id) for run_id in self.older_runs]
            done = list(self.older_done.values())
            version = self.older_version
        rows.sort(key=lambda r: r.get("created") or "", reverse=True)
        return {"runs": rows, "done": done, "count": sum(not r["later"] for r in rows), "version": version}

    def answer_older(self, run_id: str, answer: str) -> dict:
        """POST /runs/older/{id} {answer: "add" | "discard"}: one older run's answer, carried out in the older-run thread,
        one run at a time; answers the row at once. Add saves an inert run as a leave would: a layer of the sample made
        from this run already ("Saved already as X") removes it with no second copy; else an unjoined stain run is
        joined, then it is saved under its next free names, never as the sample's segmentation (set_default False),
        then removed. A failed
        save keeps the row (zarr_save failed: Try again, Discard or Later). Discard removes the folder. 409 while the
        last session may still be saving it, for add on a run that cannot be saved, or while another answer is under
        way; 404 for a run that is not listed. `run_id` None: every row, with "add", "discard" or "later"
        (answer_all_older, which answers older())."""
        if run_id is None:
            return self.answer_all_older(answer)
        if answer not in ("add", "discard"):
            raise RunError("Choose Save to the sample or Discard")
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        with self.older_lock:
            if run_id not in self.older_runs:
                raise RunError("Unknown run", 404)
            self._queue_older(run_id, answer)
            row = self._older_row(run_id)
        self._older_changed()
        return row

    def answer_all_older(self, answer: str) -> dict:
        """POST /runs/older {answer: "add" | "discard" | "later"}: every row at once (Save all skips the rows that cannot
        be saved, and both skip a row still being saved by the last session or already under way), one run at a time,
        newest first. Later sets aside, for this engine session, every row that is not under way (health.older 0; they
        stay on disk, untouched, and are listed again at the next start; one the last session was still saving stays
        aside once free) and stops a Save all after the row under way. Answers older()."""
        if answer not in ("add", "discard", "later"):
            raise RunError("Choose Save all, Discard all or Later")
        if self.closing and answer != "later":
            raise RunError("The engine is shutting down", 503)
        with self.older_lock:
            if answer == "later":
                for run_id, state in self.older_state.items():
                    state["later"] = True              # the row under way goes on; its end is a notice
                    if run_id == self.older_current:
                        continue
                    if run_id in self.older_queue:
                        self.older_queue.remove(run_id)
                        state.update(answer=None, stage=None, message=None)
            else:
                ordered = sorted(self.older_runs, key=lambda i: self.older_runs[i].meta.get("created") or "", reverse=True)
                for run_id in ordered:
                    state = self.older_state[run_id]
                    if state["still_saving"] or (answer == "add" and not state["can_add"]) or self._under_way(run_id):
                        continue
                    self._queue_older(run_id, answer)
        self._older_changed()
        return self.older()

    def _under_way(self, run_id: str) -> bool:
        """An older run whose answer is queued or being carried out (a failed one waits for another answer)."""
        state = self.older_state[run_id]
        return state["answer"] is not None and state["stage"] != "failed"

    def _queue_older(self, run_id: str, answer: str):
        """Queue an answer (the caller holds older_lock) and make sure the older-run thread runs."""
        state = self.older_state[run_id]
        if state["still_saving"]:
            raise RunError("The last session is still saving this run: wait until it has finished.", 409)
        if self._under_way(run_id):
            if state["answer"] == answer:
                return
            raise RunError("It is being saved already: wait for it." if state["answer"] == "add" else
                           "It is being discarded already.", 409)
        if answer == "add" and not state["can_add"]:
            raise RunError(OLDER_WHY.get(state["why"], ("This run cannot be saved into its sample.", ""))[0].format(
                sample=self.older_runs[run_id].meta.get("sample_name") or "The sample"), 409)
        state.update(answer=answer, stage="waiting", message="Waiting…", later=False)
        self.older_queue.append(run_id)
        if self.older_worker is None or not self.older_worker.is_alive():
            self.older_stop = False
            self.older_worker = threading.Thread(target=self._older_loop, name="older-runs", daemon=True)
            self.older_worker.start()

    def _older_stage(self, run_id: str, stage: str, message: str | None):
        with self.older_lock:
            state = self.older_state.get(run_id)
            if state is not None:
                state.update(stage=stage, message=message)
        self._older_changed()

    def _older_loop(self):
        """The older-run thread: the queued answers, one run at a time, until none is left, Later, or the engine closes
        (then the run under way finishes and the queued ones stay on disk, listed again at the next start)."""
        while True:
            with self.older_lock:
                if self.older_stop or not self.older_queue:
                    self.older_worker = self.older_current = None
                    return
                run_id = self.older_current = self.older_queue.pop(0)
                run, state = self.older_runs.get(run_id), self.older_state.get(run_id)
                answer = state["answer"] if state is not None else None
            if run is None or answer is None:
                continue
            try:
                if answer == "add":
                    stage, message = self._older_add(run)
                else:
                    self._older_stage(run_id, "removing", "Discarding…")
                    self._remove_older(run)
                    stage, message = "discarded", "Discarded"
            except Exception as exc:
                self.note(f"{'saving' if answer == 'add' else 'discarding'} the older run {run_id} failed: {exc}", run_id)
                with self.older_lock:
                    state.update(stage="failed", message=f"Couldn't {'save' if answer == 'add' else 'discard'}: {exc}")
            else:
                with self.older_lock:
                    row = self._older_row(run_id)
                    self.older_runs.pop(run_id, None)
                    self.older_state.pop(run_id, None)
                    self.older_done[run_id] = {**row, "stage": stage, "message": message, "answer": answer}
                    while len(self.older_done) > OLDER_DONE_KEPT:
                        self.older_done.pop(next(iter(self.older_done)))
            finally:
                with self.older_lock:
                    self.older_current = None
                self._older_changed()

    def _older_add(self, run: Run) -> tuple[str, str]:
        """Save an older run into its sample (answer_older): (stage, message) once it is in the sample."""
        meta, run_id = run.meta, run.meta["id"]
        run.read_new()                                # its frames and schedule, for the save's provenance
        self._older_stage(run_id, "checking", "Looking for it in the sample…")
        if self.zarr is None:
            raise RunError("SpatialData samples are not available here", 503)
        if not self.zarr.known(meta["sample"]):
            self.zarr.open(meta["sample_path"], recent=False)
        already = self._saved_already(run)
        if already:
            self._remove_older(run)
            return "saved_already", f"Saved already as {' + '.join(already)}"
        if meta.get("kind") == "stain" and not (meta.get("progress") or {}).get("resolved"):
            self._older_stage(run_id, "joining", "Joining the finished tiles…")
            self.stain.join(run)
        self._older_stage(run_id, "writing", "Saving to the sample…")
        saved = self._save_with_retry(run, set_default=False)
        self._remove_older(run)
        name = saved.get("shapes_key") if not saved.get("table_key") else f"{saved['shapes_key']} + {saved['table_key']}"
        return "saved", f"Saved as {name}"

    def _saved_already(self, run: Run) -> list[str]:
        """The sample's layers that hold an older run's result as it is now ("Saved already": no second copy): those
        the sample's own record names as made from the run by a save that committed (ZarrSamples.saved_from). A proseg
        run's result never changes once done. A stain run's does (resumed for more tiles, filtered again), so one of its
        layers counts only when made at its revision and filters now: a row "changed since you saved it" or whose later
        save failed holds a newer result, saved under new names."""
        meta = run.meta
        saved_from = getattr(self.zarr, "saved_from", None)
        layers = saved_from(meta["sample"], meta["id"]) if callable(saved_from) else []
        if not layers or meta.get("kind") != "stain":
            return layers
        read = getattr(self.zarr, "saved_provenance", None)
        progress = meta.get("progress") or {}
        if not callable(read) or not progress.get("resolved"):
            return []
        now = self.stain.exclusions.digest(self.stain.exclusions.load(run))
        held = []
        for key in layers:
            record = read(meta["sample"], key)
            if isinstance(record, dict) and record.get("result_revision") == progress.get("revision") and \
                    self._filters_digest(record) == now:
                held.append(key)
        return held

    def _filters_digest(self, record: dict) -> str | None:
        """The digest of the filters a stain save left cells out by (its provenance's `filters`; a save by a version
        before they were recorded whole has its areas and minimums under their older keys); None when unreadable."""
        filters = record.get("filters")
        if not isinstance(filters, dict):
            areas = record.get("removed_areas") or []
            filters = {"areas": [{"id": i + 1, "points": points} for i, points in enumerate(areas)],
                       "min_area_um2": record.get("min_area_um2"), "min_brightness": record.get("min_brightness")}
        try:
            return self.stain.exclusions.digest(filters)
        except Exception:
            return None

    def _save_with_retry(self, run: Run, set_default=None, timeout: float | None = None) -> dict:
        """Save a run under its next free names (`set_default` None: _whole's rule), once more after a clash in the
        sample (a save into it since its names were worked out); its zarr_save once saved (or queued, in API mode),
        else RunError with the save's words. A clash in the notebook's object (API mode: a name made there after the
        add) is never retried: the result is in the object's zarr already, and the run stays to be asked again."""
        for attempt in range(2):
            try:
                self._start_save(run, None, set_default)
            except RunError as exc:
                if attempt == 0 and exc.extra.get("clash"):
                    continue
                raise
            self._wait_save(run, timeout)
            saved = run.meta.get("zarr_save") or {}
            if saved.get("state") != "failed":
                return saved
            if attempt or not saved.get("clash") or saved.get("delivery"):
                raise RunError(saved.get("message") or "The save failed.", 409)
        raise RunError("The save failed.", 409)

    def _remove_older(self, run: Run):
        """An older run's folder goes: moved into .trash/, then emptied in the background, as leftovers are."""
        self._release(run)
        if run.meta.get("kind") in ("stain", "patches"):
            try:
                self.stain.patches.forget(run)
            except Exception:
                pass
        try:
            folder = leftovers.trash(run.dir)
        except OSError:
            folder = run.dir
        leftovers.remove_later(folder, held=self.held)

    def _stop_older(self, timeout: float):
        """Closing: the older-run row under way finishes (its save waited for, as a decided leave is); the queued ones
        and the unanswered ones stay on disk as they are."""
        with self.older_lock:
            self.older_stop = True
            worker = self.older_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)

    def _watch_still_saving(self):
        while not self.closing and self.still_saving:
            self.recheck_wake.wait(OLDER_RECHECK_S)
            self.recheck_wake.clear()
            if self.closing:
                return
            self.recheck_still_saving()

    def recheck_still_saving(self) -> int:
        """The folders left to the last session's save, looked at again (every OLDER_RECHECK_S): one free now is
        sorted as at start (this version's removed; an older run listed or removed). Answers how many are still in use."""
        with self.older_lock:
            entries = [(entry["folder"], entry["meta"]) for entry in self.still_saving.values()]
        busy = self._in_use(entries)
        for folder, meta in entries:
            if str(folder) in busy:
                continue
            with self.older_lock:
                entry = self.still_saving.pop(str(folder), None)
            if entry is None:
                continue
            try:                                       # as the last session's save left it
                meta = json.loads((folder / "run.json").read_text())
            except (OSError, ValueError):
                pass
            if entry["marked"]:
                self._leftover(folder, meta)
                continue
            with self.older_lock:
                self.older_runs.pop(meta["id"], None)
                state = self.older_state.pop(meta["id"], None) or {}
            if folder.exists():                        # a row set aside by Later stays aside once free
                self._sort_older(folder, meta, later=bool(state.get("later")))
            self._older_changed()
        with self.older_lock:
            return len(self.still_saving)

    # ---- queries -------------------------------------------------------------------------------------------
    def list(self) -> list[dict]:
        with self.lock:
            runs = list(self.runs.values())
        return sorted((r.public() for r in runs), key=lambda m: m["created"], reverse=True)

    def get(self, run_id: str) -> Run:
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise RunError("Unknown run", 404)
        return run

    def frame_path(self, run_id: str, seq: int) -> Path:
        run = self.get(run_id)
        with run.lock:
            if not 0 <= seq < len(run.frames):
                raise RunError("Unknown frame", 404)
            name = run.frames[seq].get("file", "")
        if not re.fullmatch(r"frames/[0-9]{6,9}\.(bin\.(gz|zst)|tiles)", name):
            raise RunError("Malformed frame index", 500)
        return run.live / name

    def frame_record(self, run_id: str, seq: int) -> dict:
        run = self.get(run_id)
        with run.lock:
            if not 0 <= seq < len(run.frames):
                raise RunError("Unknown frame", 404)
            return dict(run.frames[seq])

    def active(self) -> list[Run]:
        with self.lock:
            return [r for r in self.runs.values() if r.meta.get("state") in ACTIVE]

    def changed(self):
        self.version += 1

    # ---- creation and execution ----------------------------------------------------------------------------
    def create(self, sample_id: str, box=None, options: dict | None = None, region: dict | None = None,
               prior: dict | None = None, kind: str = "proseg", stain=None, extra: dict | None = None) -> dict:
        """One creation at a time, including stain and patch items. Test squares may share their parent's engine."""
        with self.creation_lock:
            self.check_open(extra)
            value = Runs._create(self, sample_id, box, options, region, prior, kind, stain, extra)
            if kind in ("stain", "patches"):
                run = self.get(value["id"])
                self._set(run, session_model=2)
                value = run.public()
            self.leftovers = None
            return value

    def check_open(self, extra=None):
        """The early answer to a new run (one run, closing); the same checks are made again where the run is registered
        (_room), under the registry lock."""
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        if self.one_run and not (extra or {}).get("qc_test"):
            with self.lock:
                if any(not r.meta.get("qc_test") for r in self.runs.values()):
                    raise RunError(ONE_RUN, 409)

    def _room(self, extra=None, active: bool = True):
        """One run per user engine (a test square's runs, `extra` qc_test, get in) and the active-run limit, checked
        where a run is registered, with closing checked again there: the CALLER HOLDS self.lock (a check made before
        it let two windows each start a run, and a run registered after close() took its list was never closed).
        Runs are counted here, never through active(), which takes the lock. Called as
        Runs._room(self, ...) so a test's _WithRoom keeps its own max_active."""
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        if self.one_run and not (extra or {}).get("qc_test") and any(not r.meta.get("qc_test") for r in self.runs.values()):
            raise RunError(ONE_RUN, 409)
        if active and sum(r.meta.get("state") in ACTIVE for r in self.runs.values()) >= self.max_active:
            raise RunError(f"{self.max_active} runs are already active: finish or abort one before starting another", 409)

    def _create(self, sample_id, box, options, region, prior, kind, stain, extra):
        if kind in ("stain", "patches"):
            return self.stain.create(sample_id, region or {"kind": "box", "box": box}, stain, kind=kind)
        if kind != "proseg":
            raise RunError("Unknown run kind")
        if options is not None and (not isinstance(options, dict) or not all(isinstance(k, str) for k in options)):
            raise RunError("Options must be an object of proseg flags")
        options = dict(options or {})
        zarr = self.zarr is not None and self.zarr.known(sample_id)
        if region is None:
            if box is None:
                raise RunError("Choose a region: a rectangle, a lasso or the full sample")
            region = {"kind": "box", "box": box}
        if zarr:
            detail = self.zarr.detail(sample_id)
            if not has_transcripts(detail):        # an image-only sample: its stain runs are made above
                raise RunError(IMAGE_ONLY, 409)
            spec = region_spec(region, detail["extent"])
            prior = self.zarr.validate_prior(sample_id, prior)
        else:
            detail = self.samples.detail(sample_id)
            if region.get("kind") != "box":
                raise RunError("Xenium bundles take a rectangle; open the sample as a SpatialData zarr for lassos and full runs")
            spec = region_spec(region, [0, 0, detail["width_um"], detail["height_um"]])
            prior = None
        x0, y0, x1, y1 = spec["bbox"]
        mode = "detail" if max(x1 - x0, y1 - y0) <= self.max_region else "tiled"
        if mode == "tiled" and not zarr:
            raise RunError(f"Regions are limited to {self.max_region:g} µm across for Xenium bundles")
        if not Path(self.binary).is_file():
            raise RunError(MISSING_PROSEG.get(self.binary_fix, MISSING_PROSEG["build"]), 503)
        if len(self.active()) >= self.max_active:
            raise RunError(f"{self.max_active} runs are already active: finish or abort one before starting another", 409)
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        values = self._values(options)
        if mode == "tiled" and "nthreads" not in options:
            values["nthreads"] = self.tiled_threads
        args = self._proseg_args(values)            # validates before anything is created
        if mode == "detail":
            self._check_budget(values, x1 - x0, y1 - y0)
        else:
            self._check_tiled_budget(values, x1 - x0, y1 - y0)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{re.sub(r'[^A-Za-z0-9-]+', '-', detail['name'])[:40].strip('-')}"
        with self.lock:
            Runs._room(self, extra)
            run_id, n = base, 1
            while run_id in self.runs or (self.root / run_id).exists():
                n += 1
                run_id = f"{base}-{n}"
            directory = self.root / run_id
            directory.mkdir()
            meta = {"id": run_id, "kind": "proseg", "sample": sample_id, "sample_name": detail["name"], "box": [x0, y0, x1, y1],
                    "region": spec, "mode": mode, "prior": prior,
                    "sample_kind": "zarr" if zarr else "xenium", "sample_path": detail.get("path"),
                    "options": options, "proseg_args": args, "threads": int(values.get("nthreads", self.threads)),
                    "speed": "fast" if mode == "detail" else "max", "state": "preparing", "message": "Queued", "created": now(),
                    "live_values": {}, "outputs": [], "paused": True, "zarr_save": None}
            meta.update(extra or {})
            meta.pop("source", None)
            meta["session_model"] = 2
            run = Run(directory, meta)
            run.spool = self.spool
            run.save()
            total = detail.get("transcripts")
            if total:       # the region's share of the sample, until preparing counts it (run.eta_s)
                extent = detail["extent"] if zarr else [0, 0, detail["width_um"], detail["height_um"]]
                run.expected = {"sample": int(total), "region": int(total * region_fraction(spec, extent))}
            run.started_at = time.monotonic()
            self.runs[run_id] = run
        self.changed()
        run.worker = threading.Thread(target=self._execute, args=(run,), name=f"run-{run_id}", daemon=True)
        run.worker.start()
        self._prebuild_vectors(run)
        return run.public()

    def _check_budget(self, values: dict, width: float, height: float):
        """Refuse geometry whose snapshots could not be held in memory (proseg-live also refuses, later)."""
        columns, layers, voxel, quad = budget_columns(values, width, height)
        if columns > MAX_COLUMNS or columns * layers > MAX_VOXELS:
            raise RunError(f"This region at {voxel:g} µm voxels, {quad:g} µm quads and {layers} layers needs about "
                           f"{columns * layers / 1e6:.0f} million voxels per snapshot (limit {MAX_VOXELS / 1e6:.0f} million): "
                           "draw a smaller region or use larger voxels")

    def _check_tiled_budget(self, values: dict, width: float, height: float):
        """Tiled snapshots still hold a few whole-grid 2-D planes (u32) and a per-layer window: bound both."""
        defaults = {o["flag"]: o["default"] for o in (_options_module().OPTIONS if _options_module() else [])}
        get = lambda flag, fallback: float(values.get(flag, defaults.get(flag)) or fallback)
        voxel = min(get("voxel-size", 1.0), get("burnin-voxel-size", 2.0))
        quad = get("quad-size", 150.0)
        layers = int(get("voxel-layers", 4))
        columns = math.ceil((width + quad) / voxel) * math.ceil((height + quad) / voxel)
        if columns > MAX_TILED_COLUMNS:
            raise RunError(f"This region at {voxel:g} µm voxels needs a {columns / 1e6:.0f} million column snapshot grid "
                           f"(limit {MAX_TILED_COLUMNS / 1e6:.0f} million): use larger voxels or a smaller region")
        if layers * 1600 * 1600 > MAX_VOXELS * 2.5:
            raise RunError(f"{layers} voxel layers are too many for live 3-D windows (at most {int(MAX_VOXELS * 2.5 / 1600 ** 2)})")

    def _values(self, options: dict) -> dict:
        """The engine's thread count, overridden by the user's proseg options. Nothing else is passed: proseg's own
        defaults (its schedule among them) apply to every option the user did not set."""
        return {"nthreads": self.threads, **options}

    def _proseg_args(self, values: dict) -> list[str]:
        module = _options_module()
        if module is None:   # schema not installed: only the thread count is passed
            return ["--nthreads", str(values["nthreads"])]
        try:
            return module.to_args(values)
        except ValueError as exc:
            raise RunError(str(exc)) from None

    def _set(self, run: Run, **values):
        with run.lock:
            run.meta.update(values)
            run.save()
            run.touch()
        self.changed()
        if run.meta.get("id") in self.older_runs:      # an older run being joined or saved: its row moves on
            self._older_changed()

    def _command(self, run: Run, extra: list[str]) -> list[str]:
        meta = run.meta
        tiled = meta.get("mode") == "tiled"
        # The app's input rule after the user's options (proseg_options.input_args): every platform's control probes
        # stay out, as Compare counts (proseg's own --xenium default leaves out Xenium's only).
        module = _options_module()
        rule = module.input_args(meta.get("options") or {}) if module is not None else []
        source = _given_input(meta) or str(run.dir / "input" / "transcripts.parquet")
        counts = self._counts_args(run)
        if is_test_run(meta):
            # A test's run ends plainly at the end of proseg's schedule (no --live-hold): proseg then writes its
            # per-transcript file, all the test reads (TEST_OUTPUTS). Never tiled: a test's area is at most 500 µm.
            return [self.binary, "--xenium", source, *meta["proseg_args"], *rule, *extra,
                    "--output-path", str(run.dir / "output"), *TEST_OUTPUTS,
                    "--live-dir", str(run.live), "--live-control", str(run.live / "control"), *counts]
        command = [self.binary, "--xenium", source, *meta["proseg_args"], *rule, *extra,
                   "--output-path", str(run.dir / "output"), *OUTPUTS, *([] if tiled else DETAIL_OUTPUTS),
                   "--live-dir", str(run.live), "--live-control", str(run.live / "control"), *counts, "--live-hold"]
        if tiled:
            command.append("--live-tiled")
        return command

    def counts_capable(self, binary: str, run_id: str | None = None) -> bool:
        """Whether the proseg-live at `binary` keeps proseg's own count: its --help lists --live-counts. Asked once per
        file (its real path, modification time and size: a new build or install is asked again). A program that does
        not answer (an error, or nothing within HELP_TIMEOUT_S: a build replacing it, a slow first read) is asked once
        more HELP_RETRY_S later; still none: taken for an older one, said in the engine's log (the run's, `run_id`:
        else its Log would only say to update a proseg-live that may be new), and asked again at the next run."""
        try:
            real = os.path.realpath(binary)
            info = os.stat(real)
        except (OSError, TypeError, ValueError):
            return False
        key = (real, info.st_mtime_ns, info.st_size)
        if key not in self.capable:
            for attempt in range(HELP_TRIES):
                if attempt:
                    time.sleep(HELP_RETRY_S)
                try:
                    done = subprocess.run([str(binary), "--help"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, timeout=HELP_TIMEOUT_S, check=False,
                                          **portable.no_window())
                except (OSError, subprocess.SubprocessError) as exc:
                    failed = exc
                    continue
                self.capable[key] = COUNTS_FLAG.encode() in done.stdout
                break
            else:
                self.note(f"{f'run {run_id}: ' if run_id else ''}proseg-live did not answer --help ({failed}): "
                          "started without proseg's own count, as an older proseg-live is", run_id)
                return False
        return self.capable[key]

    def _counts_args(self, run: Run) -> list[str]:
        """proseg-live's flags for proseg's own count, and the run's meta `live_counts` for them: "request" (counted
        when the engine asks, into the run's spool folder; a user's run whose cells start without nuclei also gets the
        start block: --live-counts-start, never a test's run), or "old_proseg" for a proseg-live without them (no flag:
        the run starts and segments as before). A spool folder that cannot be made: proseg's own frames are counted into
        the run's folder, and none is asked for."""
        meta = run.meta
        if not self.counts_capable(self.binary, meta.get("id")):
            with run.lock:
                meta["live_counts"] = "old_proseg"
            return []
        args = ["--live-counts", "request"]
        try:
            args += ["--live-counts-dir", str(self.spool.run_dir(meta["id"]))]
            run.spool_lost = False
        except OSError as exc:
            run.spool_lost = True
            self.note(f"run {meta['id']}: proseg's counts have no spool folder ({exc}): only the frames proseg "
                      "counts by itself are kept", meta["id"])
        if not is_test_run(meta):
            from .qc_rules import start_of
            if start_of(meta) == "cells":
                args.append("--live-counts-start")
        with run.lock:
            meta["live_counts"] = "request"
        return args

    def _execute(self, run: Run):
        meta = run.meta
        try:
            if meta.get("input") is not None:
                self._read_given(run)
            elif meta.get("sample_kind") == "zarr":
                self._prepare_zarr(run)
            else:
                self._set(run, message="Reading the region's transcripts…")
                count = self.samples.crop_transcripts(meta["sample"], meta["box"], run.dir / "input" / "transcripts.parquet")
                if count == 0:
                    raise RunError("There are no transcripts in this region")
                self._set(run, transcripts=count, message="Reading the morphology image at full resolution…")
                try:
                    image = self.samples.crop_images(meta["sample"], meta["box"], run.dir / "image")
                except Exception as exc:  # the masks are still worth watching without an image
                    image = {"channels": [], "error": str(exc)}
                self._set(run, image=image, message="Starting proseg…",
                          input_bytes=_size(run.dir / "input") + (_size(run.dir / "image") if (run.dir / "image").exists() else 0))
            extra: list[str] = []
            for attempt in range(2):
                if run.stop_requested:
                    self._set(run, state="stopped", message="Stopped before proseg started", finished=now())
                    return
                code = self._run_proseg(run, extra)
                if run.stop_requested and not meta.get("finish_requested"):
                    # Why, when the engine stopped it (the disk guard: _guard_disk), else what it stopped with.
                    self._set(run, state="stopped", finished=now(), exit_code=code, message=meta.get("stop_reason") or
                              f"Stopped after {len(run.frames)} snapshots, without results")
                    return
                if code == 0:
                    defaults = None
                    if meta.get("sample_kind") == "zarr" and not is_test_run(meta):    # a test's run is never saved
                        try:
                            defaults = self._save_defaults(run)
                        except Exception:
                            defaults = None
                    self._set(run, state="done", finished=now(), exit_code=0, outputs=self._outputs(run),
                              message=self._saved_message(run), save_defaults=defaults)
                    return
                # Small crops can make proseg's high-plex feature selection fail before sampling starts (upstream
                # issue #160). Retry once without factorisation, as the proseg author suggests.
                if attempt == 0 and not run.frames and "--no-factorization" not in meta["proseg_args"]:
                    self._set(run, message="Proseg could not start on this region with factorisation; retrying without it…",
                              retried_without_factorization=True)
                    extra = ["--no-factorization"]
                    with run.lock:
                        shutil.rmtree(run.live, ignore_errors=True)
                    continue
                break
            message = self._failure(run, code)
            self.note(f"run {meta['id']} failed: {message}", meta["id"])       # noted before anyone sees it failed
            self._set(run, state="failed", message=message, finished=now(), exit_code=code)
        except Exception as exc:
            try:
                if run.stop_requested:
                    self._set(run, state="stopped", message="Stopped before proseg started", finished=now())
                else:
                    self.note(f"run {meta['id']} failed: {exc}", meta["id"])
                    self._set(run, state="failed", message=str(exc), finished=now())
            except Exception:
                pass
        finally:
            process = run.process
            if process is not None and process.poll() is None:   # never leave an unwatched proseg behind
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            run.process = None

    def _read_given(self, run: Run):
        """A test's run: nothing to prepare. It reads the test's one shared input where it is (meta `input`, made from
        the parent's own input: qc_test.shared_input), so there is no helper, crop or image (the test reads no image),
        and the input's bytes are the test's, not the run's. Its count and preparation are the shared input's."""
        given = run.meta["input"] if isinstance(run.meta["input"], dict) else {}
        path = _given_input(run.meta)
        if path is None or not Path(path).is_file():
            raise RunError("The test's input is missing")
        self._set(run, transcripts=given.get("transcripts"), prepared=given.get("prepared"), image={"channels": []},
                  input_bytes=0, message="Starting proseg…")

    def _prepare_zarr(self, run: Run):
        """The region's transcripts with the chosen prior (sopa's join, in the sopa environment), then the image."""
        meta = run.meta
        last = [0.0]

        def progress(text, fraction):
            if time.monotonic() - last[0] > .25 or fraction == 1:
                last[0] = time.monotonic()
                self._set(run, message=text)

        def should_stop() -> bool:     # polled every 0.1 s while the helper prepares
            run.tick()                 # ... which is also when the page's countdown (eta_s) goes out again
            return run.stop_requested

        self._set(run, message="Reading the region's transcripts…")
        destination = run.dir / "input" / "transcripts.parquet"
        stats = self.zarr.prepare(meta["sample"], meta["region"], meta["prior"], destination, progress=progress,
                                  should_stop=should_stop)
        if not stats.get("transcripts"):
            raise RunError("There are no transcripts in this region")
        if not stats.get("assigned") and not stats.get("nuclear"):
            raise RunError("No transcript falls inside the chosen prior in this region: choose another prior or region")
        self._set(run, transcripts=stats["transcripts"], prepared=stats)
        if meta.get("mode") == "detail":
            self._set(run, message="Reading the morphology image at full resolution…")
            try:
                image = self.zarr.crop_images(meta["sample"], meta["box"], run.dir / "image")
            except Exception as exc:  # the masks are still worth watching without an image
                image = {"channels": [], "error": str(exc)}
        else:   # whole-sample runs read the image pyramid on demand, for the part on screen
            detail = self.zarr.detail(meta["sample"])
            image = {"source": "sample", "image": detail.get("image_key"), "channels": detail.get("channels", []),
                     "overview": detail.get("overview")}
        self._set(run, image=image, message="Starting proseg…",
                  input_bytes=_size(run.dir / "input") + (_size(run.dir / "image") if (run.dir / "image").exists() else 0))

    def save_to_zarr(self, run_id: str, shapes_key: str, table_key: str, set_default: bool | None = None,
                     overwrite: bool = False) -> dict:
        """Save resolved shapes, and a table for proseg, into the sample through the transactional helper.
        `set_default` None: the run's own default (_whole). A name the sample already has is refused (409) unless
        `overwrite`, the answer naming the names taken (`clash`) and every name in the sample now (`existing`), so the
        Save dialog can offer Replace; the run's save defaults are worked out again then too."""
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        run = self.get(run_id)
        with run.lock:
            if run.meta.get("leaving"):
                raise RunError("It is being saved already: wait, or press Stay.", 409)
        return self._start_save(run, {"shapes_key": shapes_key, "table_key": table_key}, set_default,
                                overwrite=overwrite)

    def _start_save(self, run: Run, names=None, set_default=None, *, overwrite=False) -> dict:
        """The save primitive for the dialog, leave and shutdown. Automatic names never replace elements."""
        if self.closing and self.keep_unsaved:
            raise RunError("The engine is shutting down", 503)
        meta = run.meta
        if not self.zarr or meta.get("sample_kind") != "zarr":
            raise RunError("Only runs on a SpatialData sample can be saved into it", 409)
        if not self.zarr.known(meta["sample"]):
            self.zarr.open(meta["sample_path"], recent=False)
        names = self._save_defaults(run) if names is None else names
        shapes_key, table_key = names.get("shapes_key"), names.get("table_key")
        run_id = meta["id"]
        stain = meta.get("kind") == "stain"
        if meta.get("kind") == "patches":
            raise RunError("Training patches are saved in their own folder", 409)
        if meta.get("sample_kind") != "zarr" or self.zarr is None:
            raise RunError("Only runs on a SpatialData sample can be saved into it", 409)
        ended_stain = stain and (meta.get("state") in ("done", "stopped") or
                                 meta.get("state") == "failed" and meta.get("progress", {}).get("complete"))
        if meta.get("state") != "done" and not ended_stain:
            raise RunError("Finish the run first (Finish & save), then save it into the sample", 409)
        if stain and not meta.get("progress", {}).get("resolved"):
            raise RunError("Resume and stop this run to join its finished tiles before saving", 409)
        result = "boundaries.geojson" if stain else "proseg-output.zarr"
        if not (run.dir / "output" / result).exists():
            raise RunError(f"This run has no {result} to save", 409)
        # The cells the run shows (its summary leaves out the removed ones): none to save is said plainly here, before
        # anything is written (the helper failed on the empty shapes with a raw IndexError).
        if stain and (meta.get("summary") or {}).get("cells") == 0:
            raise RunError(NO_CELLS, 409)
        if set_default is None:
            set_default = self._whole(run)
        if stain:
            table_key = None
        keys = (shapes_key,) if stain else (shapes_key, table_key)
        for key in keys:
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}", key) or key.startswith("__"):
                raise RunError(f"'{key}' is not a valid element name: letters, digits, _ . - (not starting with __)")
        if not stain and shapes_key == table_key:
            raise RunError("The shapes layer and the table need different names")
        if not self.zarr.known(meta["sample"]):
            # Forgotten from the open dialog's list: the zarr is still there; opened again for the run, off the list.
            self.zarr.open(meta["sample_path"], recent=False)
        existing = self._existing(meta["sample"])
        if not stain and shapes_key.lower() == table_key.lower():
            raise RunError("The shapes layer and the table need different names (ignoring case)")
        for key in keys:
            twin = next((name for name in existing if name.lower() == key.lower() and name != key), None)
            if twin:
                self._renew_defaults(run)
                raise RunError(f"'{key}' differs from the existing element '{twin}' only by case: choose another name",
                               existing=sorted(existing))
        clashes = [k for k in keys if k in existing]
        if clashes and not overwrite:
            # The dialog's names were the run's as it ended: another save may have taken them since.
            self._renew_defaults(run)
            message = (f"{' and '.join(clashes)} already exist{'s' if len(clashes) == 1 else ''} in the sample: "
                       "choose other names or allow replacing")
            raise RunError(message, 409, error=message, clash=clashes, existing=sorted(existing))
        with run.lock:
            # A stain run can resume while the sample/name checks above are in flight.
            if stain and (not (meta.get("state") in ("done", "stopped") or
                               meta.get("state") == "failed" and meta["progress"].get("complete")) or
                          not meta["progress"].get("resolved")):
                raise RunError("Stop this run and wait for its finished tiles to be joined before saving", 409)
            previous = meta.get("zarr_save")
            if (previous or {}).get("state") == "saving":
                raise RunError("This run is already being saved", 409)
            value = {"state": "saving", "shapes_key": shapes_key, "table_key": table_key,
                     "set_default": bool(set_default), "message": "Waiting for the sample…", "at": now()}
            exclusions = None
            if stain:
                value["result_revision"] = meta["progress"]["revision"]
                # The removed areas and ranges as of this save: a later change makes the saved shapes outdated (the
                # page compares). The digest is the frozen state's own (stain_rules.digest): counting every cell here
                # read every outline file, and a failure then left the run "saving" for good (closing waited an hour).
                exclusions = deepcopy(self.stain.exclusions.load(run))
                value.update(exclusions_version=exclusions["version"],
                             exclusions_digest=self.stain.exclusions.digest(exclusions),
                             covered=deepcopy(meta.get("covered")))
            failed_leave = meta.get("leave_error")
            meta.update(zarr_save=value, **({"leave_error": None} if failed_leave else {}))   # that failure is past
            try:
                run.save()
            except Exception:
                meta.update(zarr_save=previous, **({"leave_error": failed_leave} if failed_leave else {}))
                raise                                  # nothing started: the run is as it was
            # The save's thread is the run's before anyone can see it "saving" (_save_open: a "saving" with no thread
            # of this engine to write it is a record nobody is writing, never waited for).
            worker = run.saver = threading.Thread(target=self._save_job, name=f"save-{run_id}", daemon=True, args=(
                run, shapes_key, table_key, bool(set_default), bool(overwrite), exclusions))
            run.touch()
        self.changed()
        with self.lock:
            self.savers = [t for t in self.savers if t.is_alive()] + [worker]
        try:
            worker.start()
        except RuntimeError as exc:                    # no thread to save with: said, never left "saving"
            self._save_update(run, state="failed", message=f"The save could not start: {exc}", at=now())
            raise RunError(f"The save could not start: {exc}", 503) from None
        return run.public()

    def _existing(self, sid: str) -> set[str]:
        """Every element name a save must not take: the sample's; in an API session also the notebook's object's and its
        zarr's (_extra_names)."""
        return set(self.zarr.existing_elements(sid)) | self._extra_names()

    def _extra_names(self) -> set[str]:
        """API mode: the notebook's object's names (as the handle last pushed them) and its zarr's when writing is on
        (the view's are the sample's own); empty on the desktop."""
        session = self.session
        if session is None:
            return set()
        names = {str(name) for name in session.names()}
        target = session.target()
        if target:
            from .zarr_samples import ZarrSamples
            names |= set(ZarrSamples.elements_of(target))
        return names

    def _save_update(self, run: Run, **values) -> bool:
        """A save's state moves on (zarr_save, written to run.json and pushed); False for a run that was deleted. A run
        that left meanwhile (API mode: a run goes at queued once its result is in a zarr, and while the session closes
        once the notebook received it) has its folder removed (_delete_run marks it `removed` first, under its lock):
        its record goes with it, and nothing is written there any more (a run.json written while rmtree empties the
        folder failed with the folder still there). A folder gone otherwise (an older run's, moved to .trash/) alike;
        one kept while closing (_keep_closed: its save still running) is written as before."""
        with run.lock:
            run.meta["zarr_save"] = {**(run.meta.get("zarr_save") or {}), **values}
            if run.removed:
                return False
            try:
                run.save()
            except OSError:
                if run.dir.exists():
                    raise
            run.touch()
        self.changed()
        if run.meta.get("id") in self.older_runs:
            self._older_changed()
        return True

    def _save_job(self, run: Run, shapes_key: str, table_key: str, set_default: bool, overwrite: bool,
                  exclusions: dict | None = None):
        meta = run.meta
        last = run.frames[-1] if run.frames else {}
        provenance = {"app": "SoPA companion", "run": meta["id"], "iteration": last.get("iter"), "phase": last.get("phase"),
                      "saved": now(), "region": meta.get("region"), "prior": meta.get("prior"),
                      "options": meta.get("options"), "live_values": meta.get("live_values"),
                      "proseg_args": meta.get("proseg_args"), "schedule": (run.info or {}).get("schedule")}
        if self.session is not None:
            provenance["session"] = "api"
        if meta.get("kind") == "stain":
            provenance.update(kind="stain", method=meta["stain"].get("method", "cellpose"), model=meta["stain"].get("model", "cpsam"),
                              model_path=meta["stain"].get("model_path"), model_sha256=meta["stain"].get("model_sha256"),
                              parameters=meta["stain"],
                              runtime=meta.get("stain_runtime"), fallback=meta.get("stain_fallback"),
                              progress=meta["progress"], result_revision=meta["zarr_save"]["result_revision"],
                              tissue=meta.get("tissue"), tiling=meta.get("tiling"),
                              resolution=deepcopy(meta["stain"].get("resolution")),
                              covered=deepcopy(meta["zarr_save"].get("covered")))
        if exclusions is not None:
            provenance["filters"] = deepcopy(exclusions)

        def update(**values):
            self._save_update(run, **values)

        lock = self.save_locks.setdefault(meta["sample"], threading.Lock())
        delivery = staged = None
        try:
            with lock:      # one writer per sample zarr at a time
                output = run.dir / "output"
                keep = self._stain_keep(run, exclusions) if exclusions is not None else None
                where = {}
                if self.session is not None:            # API mode: the notebook's object too
                    delivery = uuid.uuid4().hex[:16]
                    staged = self.deliveries / delivery
                    staged.mkdir(parents=True, exist_ok=True)
                    where = {"deliver": self._delivery_where(meta["sample"], staged / "staged.zarr")}
                result = self.zarr.save(meta["sample"], output, shapes_key, table_key, set_default, overwrite,
                                        provenance, progress=lambda text, fraction: update(message=text),
                                        **({"keep": str(keep)} if keep is not None else {}), **where)
            # The names it took are no longer free: this run's and every other run's of the sample offer new ones
            # (a run that ended before this save still offered them, without Replace).
            self._refresh_defaults(meta["sample"])
            name = shapes_key if table_key is None else f"{shapes_key} + {table_key}"
            if delivery is not None:
                self._deliver(run, delivery, staged, result, provenance, set_default=set_default, overwrite=overwrite,
                              shapes_key=shapes_key, table_key=table_key, name=name)
            else:
                update(state="saved", cells=result.get("cells"), message=f"Saved {result.get('cells', 0):,} cells as {name}",
                       at=now())
            if callable(getattr(self.zarr, "detail", None)):
                def warm():
                    try:
                        self.zarr.detail(meta["sample"])
                    except Exception as exc:
                        self.note(f"refreshing sample {meta['sample']} after saving failed: {exc}")
                threading.Thread(target=warm, name=f"saved-sample-{meta['sample']}", daemon=True).start()
        except Exception as exc:
            self.note(f"saving run {meta['id']} into its sample failed: {exc}", meta["id"])
            if staged is not None and (meta.get("zarr_save") or {}).get("state") != "queued":
                shutil.rmtree(staged, ignore_errors=True)
            if getattr(exc, "status", None) == 409 and "clash" in getattr(exc, "extra", {}):
                # The helper found the names taken (a save into the sample since this one was checked): the dialog
                # asks again as it opens, and now hears them (Replace offered).
                self._renew_defaults(run)
            update(state="failed", message=str(exc), at=now(), clash=getattr(exc, "extra", {}).get("clash"))

    # ---- API mode: a save goes to the notebook's object --------------------------------------------------------------
    def _delivery_where(self, sid: str, staged: Path) -> dict:
        """Where an API session's save goes: `path` the object's zarr when writing is on (today's transaction; None: no
        target), `view` the engine's own zarr when that is not it (the same transaction, unconsolidated; never an
        in-place view: that is the object's own store, written only as its target), `frame_from` the view (its
        transformations frame the result) and `stage_to`, where the notebook reads the built elements from."""
        session = self.session
        view = str(Path(self.zarr.path(sid)).resolve())
        target = session.target()
        target = str(Path(target).resolve()) if target else None
        try:
            in_place = (session.public().get("sample") or {}).get("view") == "in_place"
        except Exception:                 # not known: the engine's zarr may be the object's own store, which is only
            in_place = True               # ever written as the target (writing on), so never as a view
        if in_place and target is not None and target != view:
            # In place, the object's zarr is the sample open here, and a save goes into it or into no zarr at all.
            # Another one means the object now points elsewhere (written to another folder since): never written here.
            there, here = _apart(target, view)
            raise RunError(f"Not saved: your SpatialData now belongs to {there}, not to {here}, which is open here. "
                           "Open the app again from your notebook to save there.", 409)
        return {"path": target, "view": None if in_place or view == target else view, "frame_from": view,
                "stage_to": str(staged), "consolidate": True}

    def _deliver(self, run: Run, delivery: str, staged: Path, result: dict, provenance: dict, *, set_default: bool,
                 overwrite: bool, shapes_key: str, table_key: str | None, name: str):
        """Hand a built result to the notebook (deliver), then zarr_save {state: "queued", written_to}: the save is
        over (in the object's zarr when it has one, or ready for memory). Its staged folder goes once the notebook has
        read it, and the run's zarr_save becomes saved (added) or failed once the notebook's main thread applied it
        (_follow)."""
        from ..api import protocol
        stain = run.meta.get("kind") == "stain"
        written_to = result.get("written_to")
        event = protocol.build_event(
            "deliver", id=delivery, run=run.meta["id"], kind="stain" if stain else "proseg",
            staged=str(result.get("staged") or staged / "staged.zarr"), shapes_key=shapes_key, table_key=table_key,
            set_default=bool(set_default), overwrite=bool(overwrite), replaced=[str(n) for n in result.get("replaced") or []],
            written_to=written_to or None, cells=int(result.get("cells") or 0),
            genes=None if stain else int(result.get("genes") or 0), provenance=_jsonable(_without_outlines(provenance)))
        received = threading.Event()
        with self.lock:
            self.receipts = {k: v for k, v in self.receipts.items() if not v.is_set()}
            self.receipts[delivery] = received
        self.session.deliver(event)
        # view_written false: the engine's own view could not take it, so the app's layers do not show it; it is
        # in the object (and its zarr) all the same. Kept through `saved` (_save_update merges).
        self._save_update(run, state="queued", written_to=written_to or None, delivery=delivery, cells=event["cells"],
                          view_written=result.get("view_written"), view_error=result.get("view_error"),
                          message=(f"Written to {Path(written_to).name} · in your notebook as soon as it is free"
                                   if written_to else "Ready · in your notebook as soon as it is free"), at=now())
        follower = threading.Thread(target=self._follow, args=(run, delivery, staged, received, written_to, name),
                                    name=f"deliver-{delivery}", daemon=True)
        with self.lock:
            self.followers = [t for t in self.followers if t.is_alive()] + [follower]
        run.follower = follower
        follower.start()

    def _follow(self, run: Run, delivery: str, staged: Path, received: threading.Event, written_to, name: str):
        """A delivery's way to the notebook: received (its staged folder goes), then applied by the notebook's main
        thread, each written into the run's zarr_save. A run deleted meanwhile (it leaves at queued once its result is
        in a zarr, or once received while the session closes) is followed no further: its folder is going."""
        session = self.session
        try:
            got = session.received(delivery, RECEIVE_S)
        except Exception:
            got = False
        shutil.rmtree(staged, ignore_errors=True)
        if not got:
            received.set()
            self._save_update(run, state="failed", written_to=written_to or None, received=False, at=now(),
                              message="Your notebook did not receive it" + (f": it is in {Path(written_to).name} only"
                                                                            if written_to else ""))
            return
        here = self._save_update(run, received=True)
        received.set()
        while here and not self.followers_done.is_set():
            applied = session.applied(delivery)
            if applied is not None:
                if applied.get("state") == "added":
                    cells = int((run.meta.get("zarr_save") or {}).get("cells") or 0)
                    self._save_update(run, state="saved", added=True, cells=cells, written_to=written_to or None,
                                      at=now(), message=f"Added {cells:,} cells as {name}")
                else:
                    clash = [str(k) for k in applied.get("clash") or []]
                    self._save_update(run, state="failed", written_to=written_to or None, clash=clash or None, at=now(),
                                      message=applied.get("message") or (
                                          f"Not added to your notebook's object: it already has {', '.join(clash)}"
                                          if clash else "Not added to your notebook's object"))
                return
            if getattr(session, "kernel_gone", False):      # nothing reaches memory any more: what is on disk stays
                return
            self.followers_done.wait(0.1)
            here = not run.removed

    def _stain_keep(self, run, state):
        """Freeze the committed rules over the joined cells in their output order (I-C1). A brightness rule is kept by
        its channel's name (stain_rules.check), which is what file_brightness measures (an index there failed, and the
        rule was dropped from the save as unreadable)."""
        import numpy as np
        from . import stain_rules
        rules = stain_rules.normalise(state)
        masks = []
        for file in self.stain.exclusions.files(run):
            _, (centres, areas) = self.stain.exclusions.file_cells(file)
            bright = {}
            for channel in rules.get("brightness", {}):
                try:
                    bright[channel] = np.asarray(self.stain.exclusions.file_brightness(run, file, channel),
                                                 dtype=np.float32)
                except Exception as exc:
                    self.note(f"brightness of run {run.meta['id']} could not be read: {exc}", run.meta["id"])
                    bright[channel] = np.full(len(areas), np.nan, dtype=np.float32)
            mask = stain_rules.keep(np.asarray(areas, dtype=np.float32), bright,
                                    stain_rules.in_area(centres, rules["areas"]), rules)
            if np.shape(mask) != (len(areas),):
                raise RunError("The cells changed since the save was asked for. Save again.", 409)
            masks.append(np.asarray(mask, dtype=bool))
        kept = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
        if not kept.any():
            raise RunError(NO_CELLS, 409)
        path = run.dir / "stain" / "keep.npy"
        with path.open("wb") as stream:
            np.save(stream, kept, allow_pickle=False)
        return path

    # ---- the names a save is offered (the Save dialog) -------------------------------------------------------------
    @staticmethod
    def _can_save(run: Run) -> bool:
        """Whether the run can be saved into its sample now (save_to_zarr's first checks): a proseg run done, a stain run
        ended with its tiles joined, on a SpatialData sample. Never a test's own run (is_test_run): it holds no result
        of its own, and goes once the test has read it."""
        meta = run.meta
        if meta.get("kind") == "patches" or meta.get("sample_kind") != "zarr" or is_test_run(meta):
            return False
        if meta.get("kind") == "stain":
            progress = meta.get("progress") or {}
            return bool(progress.get("resolved")) and (meta.get("state") in ("done", "stopped") or
                                                       meta.get("state") == "failed" and bool(progress.get("complete")))
        return meta.get("state") == "done"

    @staticmethod
    def _whole(run: Run) -> bool:
        """Whether a save of the run starts with "Make it the sample's segmentation" ticked (sopa's boundaries attribute:
        sopa's later steps take that layer as the sample's cells): never for a stain run that covers part of the sample
        only, a box, a lasso, or tiles left undone. Proseg uses the same full-region rule."""
        meta = run.meta
        if meta.get("kind") != "stain":
            return (meta.get("region") or {}).get("kind") == "full"
        return (meta.get("region") or {}).get("kind") == "full" and bool((meta.get("progress") or {}).get("complete"))

    def _save_defaults(self, run: Run, found: dict | None = None) -> dict:
        """The names a save of the run is offered, worked out against the sample's element names as they are now
        (ZarrSamples.save_defaults: {shapes_key, table_key, existing}), with `set_default` (_whole). `found`: what was
        worked out for other runs already, by kind and method (one listing of the sample for many runs)."""
        meta = run.meta
        kind = "stain" if meta.get("kind") == "stain" else "proseg"
        method = (meta.get("stain") or {}).get("method", "cellpose")
        key = (kind, method if kind == "stain" else None)
        defaults = (found or {}).get(key)
        if defaults is None:
            also = self._extra_names() if self.session is not None else None
            defaults = self.zarr.save_defaults(meta["sample"], kind=kind, method=method,
                                               **({"also": sorted(also)} if also else {}))
            if found is not None:
                found[key] = defaults
        return {**defaults, "existing": list(defaults.get("existing") or []), "set_default": self._whole(run)}

    def _keep_defaults(self, run: Run, defaults: dict) -> bool:
        """Keep the run's save defaults if they changed (its meta and run.json; the run goes out again): whether they
        did."""
        with run.lock:
            if run.meta.get("save_defaults") == defaults:
                return False
            run.meta["save_defaults"] = defaults
            run.save()
            run.touch()
        return True

    def _renew_defaults(self, run: Run):
        """The run's save defaults worked out again (a save refused for its names: they were taken since); if the
        sample cannot be read now, the kept ones stand."""
        try:
            if self._keep_defaults(run, self._save_defaults(run)):
                self.changed()
        except Exception:
            pass

    def _refresh_defaults(self, sample: str):
        """After a save into `sample`: every run of it that can be saved gets its save names worked out again."""
        with self.lock:
            runs = [r for r in self.runs.values() if r.meta.get("sample") == sample and self._can_save(r)]
        found, changed = {}, False
        for run in runs:
            try:
                changed = self._keep_defaults(run, self._save_defaults(run, found)) or changed
            except Exception:
                continue
        if changed:
            self.changed()

    def current_save_defaults(self, run_id: str) -> dict | None:
        """GET /runs/{id}: the save defaults of a run that can be saved into its sample, worked out against the
        sample's element names as they are now: the Save dialog asks each time it opens, and a save into the sample
        since the run ended may have taken the names it was offered. Kept in the run when they changed.
        None for a run that cannot be saved, or when the sample cannot be read now (the kept ones stand)."""
        run = self.get(run_id)
        if self.zarr is None or not self._can_save(run):
            return None
        try:
            if not self.zarr.known(run.meta["sample"]):
                # Forgotten from the open dialog's list: opened again for its run, the list left as it is.
                self.zarr.open(run.meta["sample_path"], recent=False)
            defaults = self._save_defaults(run)
            if self._keep_defaults(run, defaults):
                self.changed()
        except Exception:
            return None
        return defaults

    def note(self, text: str, run_id: str | None = None):
        """Something that went wrong: a line in the engine log, and an event the Log counts in its summary ("Engine
        noted", `noted`: engine/runlog.py Noted), the run's when it is about one (after failures the session
        Log said nothing went wrong)."""
        try:
            print(f"SoPA companion: {text}", flush=True)
        except (OSError, ValueError):          # no terminal any more (a dropped SSH session): the event still counts
            pass
        if self.noted is not None:
            self.noted.add(text, run_id)

    def _run_proseg(self, run: Run, extra: list[str]) -> int:
        run.live.mkdir(exist_ok=True)
        (run.dir / "output").mkdir(exist_ok=True)
        self.spool.remove_run(run.meta["id"])      # a (re)start: its counted frames went with its frames
        with run.lock:
            run.frames, run.offset, run.info, run.frame_bytes = [], 0, None, 0
            run.reset_counts()
            for key in ("counts_token", "where_token", "where_box"):    # proseg serves its tokens from 0 again
                run.meta.pop(key, None)
        self._write_control(run)
        command = self._command(run, extra)
        with open(run.dir / "proseg.log", "ab") as log, run.lock:
            if run.stop_requested:          # an abort that arrived while preparing: never start proseg
                return -1
            log.write(("$ " + " ".join(command) + "\n").encode())
            log.flush()
            # setpriv: proseg gets SIGTERM if the engine dies, so it never outlives the process watching it. Its own
            # session: a terminal's hang-up or Ctrl+C (sent to the whole foreground group) reaches the engine only, which
            # then closes as on SIGTERM, instead of killing proseg mid-run, a finish and save included.
            # The engine signals proseg by its pid; the parent-death signal does not depend on the group. macOS and
            # Windows have no setpriv: the engine's reaper ends proseg there should the engine die (portable.tie).
            launcher = ["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []
            run.process = subprocess.Popen(launcher + command, cwd=run.dir, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, close_fds=True, **portable.new_session())
            portable.tie(run.process)
            run.spawned_at = time.monotonic()
        state = "finishing" if run.meta.get("finish_requested") else ("paused" if run.meta.get("paused") else "running")
        self._set(run, state=state, pid=run.process.pid, started=now(), message="")
        finishing, last_disk_check, signalled, indexing, held = False, 0.0, False, False, False
        while True:
            if run.read_new():
                run.touch()
                self.frames_version += 1
            elif not run.frames:
                run.tick()                  # loading: the page counts eta_s down
            if not indexing and run.info and run.meta.get("mode") == "tiled":
                indexing = True    # whole-sample runs: index the transcripts now, before the user first zooms in
                threading.Thread(target=self._index_transcripts, args=(run,), name=f"index-{run.meta['id']}", daemon=True).start()
            if run.stop_requested and not run.meta.get("finish_requested") and not signalled:
                signalled = True            # abort: proseg does not need to save anything
                try:
                    run.process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if not held and (run.live / "schedule_done").exists():
                held = True
                # proseg acknowledged its pause before it wrote the file: read that first, so the pause carries its ack.
                if run.read_new():
                    run.touch()
                    self.frames_version += 1
                self._schedule_done(run)
            if not finishing and (run.live / "sampling_done").exists():
                finishing = True
                self._set(run, state="finishing", message="Sampling finished; proseg is writing its results…")
            if time.monotonic() - last_disk_check > 5:
                last_disk_check = time.monotonic()
                self._guard_disk(run)
            code = run.process.poll()
            if code is not None:
                if run.read_new():
                    run.touch()
                return code
            time.sleep(.04)

    @staticmethod
    def _index_transcripts(run: Run):
        try:
            from .transcript_index import TranscriptIndex
            TranscriptIndex(run.live).close()    # the files only: the viewer maps them when it needs them
        except Exception as exc:       # the viewer builds it on demand instead
            print(f"SoPA companion: transcript index for {run.meta['id']}: {exc}", flush=True)

    def transcript_index(self, run_id: str):
        """The run's transcript index, built on first use and kept (its records mapped) until the run is deleted."""
        run = self.get(run_id)
        with self.lock:
            index = self.indexes.get(run_id)
        if index is None:
            from .transcript_index import TranscriptIndex
            index = TranscriptIndex(run.live)
            with self.lock:
                if self.runs.get(run_id) is not run:     # deleted while it was being built
                    index.close()
                    raise RunError("Unknown run", 404)
                index = self.indexes.setdefault(run_id, index)
        return index

    def _release(self, run: Run):
        """Let go of what the engine keeps of a run before its files go (on a FUSE-mounted disk a file still
        mapped survives as .fuse_hidden*, and so does its folder): its transcript index and its cached tiles."""
        with self.lock:
            index = self.indexes.pop(run.meta["id"], None)
        if index is not None:
            index.close()
        tiles.CACHE.forget(str(run.dir))

    def _remove(self, folder: Path, delays=(1, 5, 30, 120)):
        """Delete a run's folder. Whatever still held a file of it then (a query in flight, an index being built,
        proseg exiting) leaves the folder behind on a FUSE-mounted disk: removed again once nothing holds it, and else
        at the next start (leftovers.sweep)."""
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            return

        def retry():
            for delay in delays:
                time.sleep(delay)
                if self.closing or folder.name in self.runs or leftovers.remove_folder(folder, held=self.held):
                    return            # (a run of the same id again: never; ids carry their creation second)
        threading.Thread(target=retry, name=f"remove-{folder.name}", daemon=True).start()

    def _schedule_done(self, run: Run):
        """proseg completed its schedule and holds there (`live/schedule_done`, written after its "paused" ack): the run
        pauses on that state, proseg's own result (meta `schedule_done` {iter, seq}), unless a finish or an abort is on
        its way. Finish & save writes it; Resume or Step samples on (meta `continued`, control `continue 1`)."""
        try:
            done = json.loads(portable.read_text(run.live / "schedule_done"))
        except (OSError, ValueError):
            done = {}
        with run.lock:
            meta = run.meta
            meta["schedule_done"] = {"iter": done.get("iter"), "seq": done.get("seq")}
            if not meta.get("finish_requested") and not run.stop_requested and not meta.get("continued"):
                meta["paused"] = True
                if meta.get("state") == "running":
                    meta["state"] = "paused"
                self._write_control(run)
            run.save()
            run.touch()
        self.changed()

    def _guard_disk(self, run: Run):
        """Under MIN_FREE_GB free on the data disk, a run sampling pauses there, saying why. A test's own run is aborted
        instead, as its Abort would be (control), and why is kept through its stop (`stop_reason`): the test then fails
        at once, where a paused run would hold its place until the test's timeout."""
        try:
            free = shutil.disk_usage(run.dir).free / 1e9
        except OSError:
            return
        if free < MIN_FREE_GB and run.meta.get("state") == "running":
            if is_test_run(run.meta):
                why = f"only {free:.0f} GB free on the data disk"
                with run.lock:
                    if run.stop_requested:          # aborted already: proseg is on its way out
                        return
                    run.meta.update(stop_reason=why, message=why)
                try:
                    self.control(run.meta["id"], "abort")
                except RunError:                    # it ended, or went, meanwhile
                    pass
                return
            with run.lock:
                run.meta["paused"] = True
                self._write_control(run)
            self._set(run, state="paused", message=f"Paused automatically: only {free:.0f} GB free on the data disk. "
                                                   "Free some space, then resume.")

    def _outputs(self, run: Run) -> list[dict]:
        found = []
        output = run.dir / "output"
        for entry in sorted(output.iterdir()) if output.is_dir() else []:
            try:
                found.append({"name": entry.name, "path": str(entry), "bytes": _size(entry)})
            except OSError:
                continue
        return found

    def _saved_message(self, run: Run) -> str:
        last = run.frames[-1] if run.frames else None
        where = f" at iteration {last['iter']}" if last else ""
        return f"Finished{where}"

    def _failure(self, run: Run, code: int) -> str:
        try:
            lines = [l for l in (run.dir / "proseg.log").read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            lines = []
        panic = next((l for l in reversed(lines) if "panicked" in l or "error" in l.lower()), lines[-1] if lines else "")
        return f"proseg exited with code {code}" + (f": {panic.strip()[:400]}" if panic else "")

    # ---- control -------------------------------------------------------------------------------------------
    def _write_control(self, run: Run):
        # Under the run lock: a writer that read older metadata can never replace a newer pause/finish.
        with run.lock:
            meta = run.meta
            lines = [f"paused {1 if meta.get('paused') else 0}", f"step {meta.get('steps', 0)}",
                     f"min_ms {SPEEDS[meta.get('speed', 'max')]}", f"finish {1 if meta.get('finish_requested') else 0}",
                     f"continue {1 if meta.get('continued') else 0}"]
            if meta.get("view"):
                lines.append("view " + " ".join(f"{v:.2f}" for v in meta["view"]))
            module = _options_module()
            if module is not None and meta.get("live_values"):
                lines += module.live_lines(meta["live_values"])
            # The newest counts and where tokens (request_counts, request_where): proseg serves each once.
            if int(meta.get("counts_token") or 0) > 0:
                lines.append(f"counts {int(meta['counts_token'])}")
            if int(meta.get("where_token") or 0) > 0 and meta.get("where_box"):
                lines.append(f"where {int(meta['where_token'])} " + " ".join(f"{v:.2f}" for v in meta["where_box"]))
            run.live.mkdir(exist_ok=True)
            atomic_text(run.live / "control", "\n".join(lines) + "\n")

    def set_view(self, run_id: str, box) -> dict:
        """The part of the tissue on screen (µm): tiled runs keep per-layer labels for it (3-D, transcript colours)."""
        run = self.get(run_id)
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            raise RunError("view must be [x0, y0, x1, y1] in µm") from None
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            raise RunError("The view is empty")
        with run.lock:
            run.meta["view"] = [x0, y0, x1, y1]
            if run.meta.get("state") in ACTIVE and run.live.is_dir():
                self._write_control(run)
        return {"view": run.meta["view"]}

    # ---- proseg's own count: tokens, where exports, the spool --------------------------------------------------------
    @staticmethod
    def _counting(run: Run, states) -> bool:
        """A proseg run in one of `states` that keeps proseg's own count (the caller holds the run's lock)."""
        return run.meta.get("kind", "proseg") == "proseg" and run.meta.get("state") in states \
            and counts_reason(run) is None

    @staticmethod
    def _paused_on(run: Run) -> int | None:
        """The frame proseg is paused or held on, its counts line merged (the caller holds the run's lock); None while
        it samples (an ack left from an earlier pause does not count)."""
        ack = run.ack or {}
        seq = ack.get("seq")
        if ack.get("state") != "paused" or run.meta.get("state") != "paused" or not _whole(seq):
            return None
        return seq if 0 <= seq < len(run.frames) and "counts" in run.frames[seq] else None

    def _spool_refusal(self, run: Run) -> str | None:
        """Why no frame may be asked of proseg for the run now ("run_cap", "engine_cap", "low_space": CountsSpool
        can_issue; "spool_lost" for a run without its spool folder, whatever the disk's room); None when one may. A low
        disk is said once in the engine's log, until there is room again."""
        if run.spool_lost:
            return "spool_lost"
        _, why = self.spool.can_issue(run.meta["id"])
        if why == "low_space":
            if not self.spool_low:
                self.spool_low = True
                self.note(f"proseg's own count: no frame is asked of proseg while {self.spool.root} has under "
                          f"{2 * self.spool.run_cap_mb:,} MB free", run.meta["id"])
        elif self.spool_low and (why is None or not self.spool.low_space()):
            self.spool_low = False             # room again (a cap's refusal says nothing of the disk: it is looked at)
        return why

    def _usable_wheres(self, run: Run, seq=None, box=None, token=None) -> list[dict]:
        """The run's kept where exports whose frame's counts line is merged (their counts give a cell's keep), newest
        first: of frame `seq`, covering `box`, of `token`, each when given (the caller holds the run's lock)."""
        from .held import covers
        found = []
        for record in reversed(run.wheres):
            at = record["seq"]
            if (seq is not None and at != seq) or (token is not None and record["token"] != token):
                continue
            if not (at < len(run.frames) and "counts" in run.frames[at]):
                continue
            try:
                if box is not None and not covers(record, box):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            found.append(record)
        return found

    @staticmethod
    def _present(records: list[dict]) -> dict | None:
        """The first record whose file is still there (proseg keeps its WHERES_KEPT newest), as a copy."""
        for record in records:
            if os.path.isfile(record["path"]):
                return dict(record)
        return None

    def request_counts(self, run_id: str) -> int | None:
        """Ask proseg to count its next frame (the scheduler, engine/analysis.py, when a live pass starts): the counts
        token on its way (one at a time: issued and not served yet), else a new one written into the control file.
        None when the run is not active or keeps no proseg's count (counts_reason), or the spool has no room for one
        more frame (_spool_refusal; never while paused: proseg then counts nothing new for it). A token not served
        within TOKEN_STALE_S no longer holds a new one back."""
        try:
            run = self.get(run_id)
        except RunError:
            return None
        with run.lock:
            if not self._counting(run, ACTIVE):
                return None
        refused = run.meta.get("state") != "paused" and self._spool_refusal(run)     # looked at without the run's lock
        with run.lock:
            if not self._counting(run, ACTIVE):
                return None
            issued, now = int(run.meta.get("counts_token") or 0), time.monotonic()
            if issued > run.served_counts() and now - run.counts_issued_at < TOKEN_STALE_S:
                return issued
            if refused:
                return None
            run.meta["counts_token"], run.counts_issued_at = issued + 1, now
            run.served_counts()                # paused: served by the paused state at once
            self._write_control(run)
            return issued + 1

    def request_where(self, run_id: str, box, wait_s: float = 2.0) -> dict | None:
        """Proseg's pools in `box` (µm; RunError 422 for one that is not a box) for Go deeper's pink dots and a test's
        targets (engine/held.py reads them): while proseg is paused or held, the newest export of that frame covering
        the box, at once; else one asked for now (a where token, and a counts token when none is on its way), served at
        once while paused or held and with the next counted frame while sampling. -> its record (where.json's, "path"
        added) once its frame's counts line is merged, or None: none came within `wait_s` (0: ask, never wait: the
        export comes with a later frame, where_export), the run is neither sampling nor paused, keeps no proseg's count,
        or the spool has no room (as request_counts). A where token on its way holds a new one back (proseg serves only
        the newest) for up to `wait_s`, and no longer once it is TOKEN_STALE_S old."""
        box = where_box(box)
        try:
            run = self.get(run_id)
        except RunError:
            return None
        wait_s = max(0.0, float(wait_s or 0))
        deadline = time.monotonic() + wait_s
        with run.lock:
            seq = self._paused_on(run)
            kept = self._usable_wheres(run, seq=seq, box=box) if seq is not None else []
        found = self._present(kept)
        if found is not None:
            return found
        token = None
        while token is None:
            with run.lock:
                if not self._counting(run, ("running", "paused")):
                    return None
            if run.meta.get("state") != "paused" and self._spool_refusal(run):    # looked at without the run's lock
                return None
            with run.lock:
                if not self._counting(run, ("running", "paused")):
                    return None
                current, now = int(run.meta.get("where_token") or 0), time.monotonic()
                if current <= run.where_served or now - run.where_issued_at >= TOKEN_STALE_S:
                    token = current + 1
                    run.meta.update(where_token=token, where_box=box)
                    run.where_issued_at = now
                    issued = int(run.meta.get("counts_token") or 0)
                    if issued <= run.served_counts() or now - run.counts_issued_at >= TOKEN_STALE_S:
                        run.meta["counts_token"], run.counts_issued_at = issued + 1, now
                        run.served_counts()
                    self._write_control(run)
            if token is None:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(TOKEN_POLL_S)
        while wait_s > 0:
            with run.lock:
                records = self._usable_wheres(run, token=token)
                active = run.meta.get("state") in ACTIVE
            found = self._present(records)
            if found is not None:
                return found
            if not active or time.monotonic() >= deadline:
                return None
            time.sleep(TOKEN_POLL_S)
        return None

    def where_export(self, run_id: str, seq: int, box) -> dict | None:
        """The newest where export of frame `seq` that covers `box` and whose file is still there, with that frame's
        counts line merged: its record ("path" added); None otherwise."""
        try:
            box = [float(v) for v in box]
            run = self.get(run_id)
        except (RunError, TypeError, ValueError):
            return None
        if len(box) != 4:
            return None
        with run.lock:
            records = self._usable_wheres(run, seq=seq, box=box)
        return self._present(records)

    def counts_status(self, run_id: str) -> dict:
        """Where the run's proseg's count stands, for the scheduler (RunError 404 for an unknown run): {mode ("request"
        | "every" | None: info.json's), reason (counts_reason), issued, served (counts tokens), outstanding, newest (the
        newest counted frame's seq | None), newest_at (time.monotonic() when its line was merged), extract_ms and bytes
        (its line's ms.extract and bytes), spool_mb (what the run's spool folder holds), can_issue, why (None |
        "run_cap" | "engine_cap" | "low_space" | "spool_lost" | "ended" (not active, or not a proseg run) | "no_counts" |
        "old_proseg")}."""
        run = self.get(run_id)
        with run.lock:
            reason = counts_reason(run)
            active = run.meta.get("kind", "proseg") == "proseg" and run.meta.get("state") in ACTIVE
            paused = run.meta.get("state") == "paused"
        why = "ended" if not active else reason or (None if paused else self._spool_refusal(run))
        with run.lock:
            mode = ((run.info or {}).get("counts") or {}).get("mode") if isinstance(run.info, dict) else None
            issued = int(run.meta.get("counts_token") or 0)
            served = run.served_counts()
            newest = run.counted[-1] if run.counted else None
            line = run.frames[newest].get("counts") if newest is not None else None
            newest_at = run.newest_at if newest is not None else None
        line = line if isinstance(line, dict) else {}
        timing = line.get("ms") if isinstance(line.get("ms"), dict) else {}
        return {"mode": mode if mode in ("request", "every") else None, "reason": reason, "issued": issued,
                "served": served, "outstanding": issued > served, "newest": newest, "newest_at": newest_at,
                "extract_ms": timing.get("extract"), "bytes": line.get("bytes"),
                "spool_mb": self.spool.usage_mb(run.meta["id"]), "can_issue": why is None, "why": why}

    def control(self, run_id: str, action: str, speed: str | None = None, values: dict | None = None) -> dict:
        run = self.get(run_id)
        if run.meta.get("kind") == "patches":
            raise RunError("Use the training patch controls for this item", 409)
        if run.meta.get("kind") == "stain":
            return self.stain.control(run, action)
        with run.lock:
            state = run.meta.get("state")
            if action == "speed":
                if speed not in SPEEDS:
                    raise RunError("Unknown speed")
                run.meta["speed"] = speed
            elif action in ("pause", "resume", "step"):
                if state not in ("preparing", "running", "paused"):
                    raise RunError("This run is not sampling", 409)
                if action == "pause":
                    run.meta["paused"] = True
                elif action == "resume":
                    run.meta["paused"] = False
                else:
                    run.meta["paused"] = True
                    run.meta["steps"] = run.meta.get("steps", 0) + 1
                if action != "pause":
                    run.meta.pop("paused_away", None)    # carried on: the pause while no page showed it is over
                if action != "pause" and run.meta.get("schedule_done"):
                    run.meta["continued"] = True     # past the end of proseg's schedule: sample on, open-ended
                if state != "preparing":
                    run.meta["state"] = "paused" if run.meta["paused"] else "running"
            elif action == "set":
                module = _options_module()
                if module is None:
                    raise RunError("Live knobs need engine/proseg_options.py", 503)
                if state not in ACTIVE:
                    raise RunError("This run is not active", 409)
                merged = {**run.meta.get("live_values", {}), **(values or {})}
                try:
                    module.live_lines(merged)       # validates
                except ValueError as exc:
                    raise RunError(str(exc)) from None
                run.meta["live_values"] = merged
            elif action == "finish":
                if state not in ACTIVE:
                    raise RunError("This run is not active", 409)
                run.meta.pop("paused_away", None)
                if state == "preparing" and run.process is None:
                    run.stop_requested = True    # nothing to save yet
                else:
                    run.meta["finish_requested"] = True
                    run.meta["state"] = "finishing"
                    last = run.frames[-1]["iter"] if run.frames else 0
                    run.meta["message"] = f"Saving results from iteration {last}…"
            elif action == "abort":
                if state not in ACTIVE:
                    raise RunError("This run is not active", 409)
                run.meta.pop("paused_away", None)
                run.stop_requested = True
                if run.process is not None:
                    try:
                        run.process.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            else:
                raise RunError("Unknown action")
            if run.live.is_dir() or state != "preparing":
                self._write_control(run)
            run.save()
            run.touch()
        self.changed()
        return run.public()

    def _write_beside(self, run: Run, path: Path, data: bytes, durable: bool = False):
        """Cache a derived file in the run's folder unless the run was deleted while it was being built (its folder
        may still exist on a FUSE-mounted disk: a file written there now would keep it for good)."""
        with self.lock:
            alive = self.runs.get(run.meta["id"]) is run
        if not alive:
            raise RunError("Unknown run", 404)
        path.parent.mkdir(exist_ok=True)               # not parents: a run deleted meanwhile stays deleted
        atomic_bytes(path, data, durable)

    def xenium_boundaries(self, run_id: str) -> Path:
        """The run's prior shapes (cells, nuclei) around its region as JSON (cached beside the run): no route serves it
        (the page loads prior_vectors' binary); the tests check that binary against it, value for value."""
        run = self.get(run_id)
        path = run.dir / "xenium" / "boundaries.json"
        if not path.exists():
            if run.meta.get("sample_kind") == "zarr":
                value = self._prior_shapes(run)
            else:
                value = self.samples.xenium_boundaries(run.meta["sample"], run.meta["box"])
            self._write_beside(run, path, json.dumps(value).encode())
        return path

    def _prior_selection(self, run: Run) -> tuple[list, int, list]:
        """What both forms of a run's prior shapes are made of: the box (the run's, grown by PRIOR_MARGIN_UM), the
        limit per layer (by mode) and the [(layer, shapes key)] of its shapes priors. A sample forgotten since is
        opened again where the run found it, as for its image and saves."""
        x0, y0, x1, y1 = run.meta["box"]
        m = vectors.PRIOR_MARGIN_UM
        prior = run.meta.get("prior") or {}
        parts = [(part, prior[part]["key"]) for part in vectors.LAYERS
                 if (prior.get(part) or {}).get("kind") == "shapes"]
        if parts and not self.zarr.known(run.meta["sample"]):
            self.zarr.open(run.meta["sample_path"], recent=False)     # for the run: the open dialog's list stays
        limit = vectors.PRIOR_LIMITS["tiled" if run.meta.get("mode") == "tiled" else "detail"]
        return [x0 - m, y0 - m, x1 + m, y1 + m], limit, parts

    def _prior_shapes(self, run: Run) -> dict:
        """The run's prior shapes (cells, nuclei) around its region, as {cells, nuclei: [{id, poly}]} (µm)."""
        box, limit, parts = self._prior_selection(run)
        value = {"cells": [], "nuclei": [], "source": {}}
        for part, key in parts:
            value[part], value["source"][part] = self.zarr.shapes(run.meta["sample"], key, box, limit)["polygons"], key
        return value

    def prior_vectors(self, run_id: str) -> Path:
        """The run's prior shapes as one binary (engine/vectors.py: what the JSON above carries), zstd-compressed beside
        the run. Built once by whichever comes first: the build a whole-sample run starts with (_prebuild_vectors) or
        the first request; the other waits for it."""
        run = self.get(run_id)
        path = run.dir / "xenium" / vectors.FILE
        if path.exists():
            return path
        with self.lock:
            guard = self.vector_guards.setdefault(run_id, threading.Lock())
        with guard:
            if not path.exists():
                if run.meta.get("sample_kind") == "zarr":
                    layers, meta = self._prior_layers(run)
                else:   # a Xenium bundle run (made by an older version): a small region, from the JSON form
                    value = self.samples.xenium_boundaries(run.meta["sample"], run.meta["box"])
                    layers, meta = {part: vectors.from_polygons(value.get(part) or []) for part in vectors.LAYERS}, {}
                self._write_beside(run, path, _zstd(vectors.encode(layers, meta), 3), durable=True)   # served immutable
        return path

    def _prior_layers(self, run: Run) -> tuple[dict, dict]:
        """_prior_shapes as arrays (vectors.select): the same polygons, order, coordinates and ids, without a Python
        object per polygon or vertex."""
        box, limit, parts = self._prior_selection(run)
        layers, source = {}, {}
        for part, key in parts:
            layers[part], source[part] = vectors.select(self.zarr._outlines(run.meta["sample"], key), box, limit), key
        return layers, {"source": source}

    def _prebuild_vectors(self, run: Run):
        """Whole-sample runs with prior shapes: build the prior vectors while the run prepares (well under a second once
        the sample's outlines are cached), so reaching the detail level never waits for them."""
        if run.meta.get("mode") != "tiled" or not any((s or {}).get("kind") == "shapes" for s in (run.meta.get("prior") or {}).values()):
            return

        def build():
            try:
                self.prior_vectors(run.meta["id"])
            except Exception as exc:          # the first request builds them instead (and reports what fails)
                print(f"SoPA companion: prior vectors of {run.meta['id']} not prebuilt: {exc}", flush=True)
        threading.Thread(target=build, name=f"vectors-{run.meta['id']}", daemon=True).start()

    def xenium_masks(self, run_id: str) -> dict:
        """Xenium's nucleus/cell label masks over the region, cropped on first use (a few seconds)."""
        run = self.get(run_id)
        if run.meta.get("sample_kind") == "zarr":
            raise RunError("Xenium masks exist only for runs on a Xenium output bundle", 404)
        path = run.dir / "xenium" / "masks.json"
        if not path.exists():
            return self.samples.crop_xenium_masks(run.meta["sample"], run.meta["box"], run.dir / "xenium")
        return json.loads(path.read_text())

    def delete(self, run_id: str, later: bool = False):
        """Delete a run that is not active. `later`: its folder leaves the runs folder at once (into .trash/) and is
        emptied in the background, as a forced delete's is, instead of before this returns (a test's runs: a few
        hundred frames each, which take a while to remove on an NTFS disk)."""
        run = self.get(run_id)
        with run.lock:
            if run.meta.get("leaving"):
                raise RunError("Wait for this run to finish leaving, or press Stay.", 409)
        self._delete(run, later=later)

    def _delete(self, run: Run, reason=None, token=None, force=False, later=False):
        # A new stain run must not start while the old run's shared helper is being released.
        with self.creation_lock:
            self._delete_run(run, reason, token, force, later)

    def _delete_run(self, run: Run, reason=None, token=None, force=False, later=False):
        assert not force or self.closing
        run_id = run.meta["id"]
        with run.lock:
            if token is not None and run.leave_token is not token:
                return
            if token is None and run.meta.get("leaving") and not self.closing:
                raise RunError("Wait for this run to finish leaving, or press Stay.", 409)
            if not force and run.meta.get("state") in ACTIVE:
                raise RunError("Stop the run before deleting it", 409)
            if not force and self._save_open(run):
                raise RunError("Wait until the run has been saved into the sample", 409)
            if not force and run_id in self.stain.patches.exporting:  # an export works without the run's lock
                raise RunError("Wait until its training patches have been exported", 409)
            run.leave_token = None
            run.meta["leaving"] = None
            run.removed = True                  # under its lock: a save's update after this writes nothing there
            with self.lock:
                self.runs.pop(run_id, None)
                self.vector_guards.pop(run_id, None)
                if reason:
                    self.gone[run_id] = reason
                    self.departed[run_id] = deepcopy(run.public())
                    while len(self.departed) > 64:
                        self.departed.pop(next(iter(self.departed)))
            if run.meta.get("kind") in ("stain", "patches"):
                try:
                    self.stain.worker.close()  # release its run-local log/cwd before removing files
                    self.stain.patches.forget(run)
                except Exception as exc:
                    self.note(f"releasing stain run {run_id} failed: {exc}", run_id)
        self._release(run)
        self.spool.remove_run(run_id)              # its counted frames in the spool go with it, a test's runs' too
        if self.on_delete is not None:
            try:
                self.on_delete(run)
            except Exception as exc:
                self.note(f"releasing run {run_id} failed: {exc}", run_id)
        if force or later:
            try:
                folder = leftovers.trash(run.dir)
            except OSError:
                folder = run.dir
            leftovers.remove_later(folder, held=self.held)
        else:
            self._remove(run.dir)
        self.changed()

    @staticmethod
    def _finishing(run: Run) -> bool:
        """Its finish was requested (Finish & save), or proseg has finished sampling: proseg is (about to be) writing its
        results. A run held at the end of its schedule is not: it is paused, waiting for the user."""
        return bool(run.meta.get("finish_requested")) or run.meta.get("state") == "finishing"

    def _decided(self, meta) -> bool:
        """A leave or save under way, or a Finish & save (finish_and_save), is decided: never asked again, completed
        whatever the closing answer. In an API session whose kernel has gone a Finish & save is not (nothing can reach
        the notebook any more: what was not added is dropped)."""
        if meta.get("leaving") or leave_kind(meta)["kind"] == "saving":
            return True
        return finish_and_save(meta) and not getattr(self.session, "kernel_gone", False)

    def _pending_row(self, meta):
        predicate = leave_kind(meta)
        progress = meta.get("progress") or {}
        return {"id": meta["id"], "kind": meta.get("kind", "proseg"), "sample": meta.get("sample"),
                "sample_name": meta.get("sample_name"), "state": meta.get("state"),
                "iter": (meta.get("last") or meta.get("ack") or {}).get("iter"),
                "tiles": {"done": progress.get("done", 0), "of": progress.get("total", 0)},
                "can_add": predicate["can_add"], "why": predicate["why"], "decided": self._decided(meta)}

    def pending(self) -> list[dict]:
        """The registry's runs whose results are not in their sample yet (kind ask or saving), as {id, kind, sample,
        sample_name, state, iter, tiles: {done, of}, can_add, why, decided} (GET /closing, the close popup)."""
        return [self._pending_row(meta) for meta in self.list() if leave_kind(meta)["kind"] in ("ask", "saving")]

    def pause_for_away(self) -> list[str]:
        """No page shows the runs any more and none said goodbye (server/presence.py): each run still sampling pauses as
        its Pause button would, marked `paused_away` (UTC, ISO) so that the next page can say why. A run being finished,
        left or saved, a test's own run, and a run paused already, are left as they are. -> the ids paused."""
        paused = []
        for meta in self.list():
            kind, state = meta.get("kind", "proseg"), meta.get("state")
            if (meta.get("paused") or meta.get("leaving") or is_test_run(meta)
                    or (meta.get("zarr_save") or {}).get("state") == "saving"):
                continue
            if kind == "proseg":
                if state not in ("preparing", "running") or meta.get("finish_requested"):
                    continue
            elif kind != "stain" or state not in ACTIVE or meta.get("stain_stop"):
                continue
            try:
                self.control(meta["id"], "pause")
                run = self.get(meta["id"])
            except RunError:                 # it ended, or went, meanwhile
                continue
            with run.lock:
                if not run.meta.get("paused"):
                    continue
                run.meta["paused_away"] = now()
                run.save()
                run.touch()
            paused.append(meta["id"])
        if paused:
            self.changed()
        return paused

    def leave(self, run_id: str, answer: str, revision: int | None = None, page: str | None = None) -> dict:
        """POST /runs/{id}/leave {answer: "add" | "discard" | "stay"}: answers the run at once and works in a
        thread. Stay withdraws a leave under way and takes a Finish & save back (save_cancelled). `revision` (with
        the window's `page`): answers of one window are taken in their order, never an older one after a newer one (a
        late Discard landing after Stay would delete the run the window stays on); an older answer
        is answered with the run as it is and nothing done."""
        if self.closing:
            raise RunError("The engine is shutting down", 503)
        if answer not in ("add", "discard", "stay"):
            raise RunError("Choose Save to the sample, Discard or Stay")
        run = self.get(run_id)
        with run.lock:
            if revision is not None:
                if not isinstance(revision, int) or isinstance(revision, bool):
                    raise RunError("revision must be a whole number")
                last = run.leave_revisions.get(page)
                if last is not None and revision < last:
                    return run.public()
                run.leave_revisions.pop(page, None)
                run.leave_revisions[page] = revision
                while len(run.leave_revisions) > 16:
                    run.leave_revisions.pop(next(iter(run.leave_revisions)))
            if answer == "stay":
                run.leave_token = None
                self._set(run, leaving=None, save_cancelled=True, leave_error=None)
                return run.public()
            leaving = run.meta.get("leaving")
            if leaving:
                if leaving["answer"] == answer:
                    return run.public()
                raise RunError("It is being saved already: wait, or press Stay." if leaving["answer"] == "add" else
                               "It is being discarded already.", 409)
            predicate = leave_kind(run.public())
            if answer == "add" and predicate["kind"] == "ask" and not predicate["can_add"]:
                raise RunError("Your filters leave no cells, so there is nothing to save." if predicate["why"] == "filtered"
                               else "This run cannot be saved into its sample. Stay or discard it.", 409)
            token = run.leave_token = object()
            self._leave_progress(run, answer, "writing" if predicate["kind"] == "saving" else "finishing",
                                 "Saving to the sample…" if predicate["kind"] == "saving" else
                                 "Finishing the run…" if answer == "add" else "Discarding the run…")
            public = run.public()
            worker = run.leave_worker = threading.Thread(target=self._leave, args=(run, answer, token),
                                                         name=f"leave-{run_id}", daemon=True)
            worker.start()
            return public

    def _leave_progress(self, run, answer, stage, message):
        """A leave under way moves on (`leaving`); the last one's failure (`leave_error`: {answer, message, at}, the
        page's "Couldn't save" or "Couldn't discard") goes once another starts."""
        self._set(run, leaving={"answer": answer, "stage": stage, "message": message}, leave_error=None)

    @staticmethod
    def _wait_until(done, timeout, message):
        deadline = time.monotonic() + timeout
        while not done():
            if time.monotonic() >= deadline:
                raise RunError(message, 409)
            time.sleep(0.02)

    def _save_open(self, run) -> bool:
        """A save of the run that is not over: its helper still writing, or (API mode) a result the notebook still has
        to take, which while the engine closes means one it has not received yet (the kernel applies it by itself).
        A record that says "saving" with no save thread of this engine to write it (none, or one that ended without
        saying: a run.json written by hand, a bug) is nothing to wait for: closing never waits an hour on it."""
        saver = run.saver
        saved = run.meta.get("zarr_save") or {}
        if saver is not None and (saver.is_alive() or saver.ident is None and saved.get("state") == "saving"):
            return True                                # writing, or about to start (_start_save)
        if saved.get("state") == "saving":
            return False
        if saved.get("state") == "queued" and self.closing and saved.get("received") is not None:
            return False
        return leave_kind(run.public())["kind"] == "saving"

    def _wait_save(self, run, timeout: float | None = None):
        self._wait_until(lambda: not self._save_open(run), SAVE_WAIT_S if timeout is None else timeout,
                         "The save has not finished yet.")

    def _stop(self, run, timeout=30):
        if run.meta.get("state") in ACTIVE:
            self.control(run.meta["id"], "abort")
        elif run.meta.get("kind") == "patches" and run.process is not None:
            try:
                portable.signal_group(run.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if run.worker is not None and run.worker is not threading.current_thread():
            run.worker.join(timeout)
        process = run.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if run.worker is not None and run.worker.is_alive():
            run.worker.join(3)
            if run.worker.is_alive():
                raise RunError("The run has not stopped yet.", 409)
        if run.meta.get("state") in ACTIVE:
            self._set(run, state="stopped", paused=False)

    def _leave(self, run, answer, token, finish_limit=3600):
        """Carry out a leave once. Stay cancels deletion; any finish or save already started completes. A failure keeps
        the run with `leave_error` {answer, message, at}; an add that never reached the sample is a failed save too."""
        with run.leave_lock:
            def cancelled():
                return run.leave_token is not token
            try:
                if cancelled():
                    return
                saved_before = (run.meta.get("zarr_save") or {}).get("state") == "saving"
                self._wait_save(run)
                if cancelled():
                    return
                if (run.meta.get("zarr_save") or {}).get("state") == "saving":     # nobody writes it (_save_open):
                    self._save_update(run, state="failed", at=now(),            # an interrupted save, never one
                                      message="Interrupted: the save did not finish. Save again.")   # to pass by
                if answer == "add" and saved_before and (run.meta.get("zarr_save") or {}).get("state") == "failed":
                    raise RunError(run.meta["zarr_save"].get("message", "The save failed."), 409)
                predicate = leave_kind(run.public())
                if answer == "add" and predicate["kind"] == "ask":
                    if not predicate["can_add"]:
                        raise RunError(NO_CELLS if predicate["why"] == "filtered" else
                                       "This run cannot be saved into its sample.", 409)
                    stain = run.meta.get("kind") == "stain"
                    if run.meta.get("state") in ACTIVE:
                        self._leave_progress(run, answer, "finishing", "Finishing the run…")
                        if run.meta.get("state") != "finishing":
                            self.control(run.meta["id"], "stop" if stain else "finish")
                        self._wait_until(lambda: run.meta.get("state") not in ACTIVE, finish_limit,
                                         "The run has not finished yet.")
                        if run.worker is not None:
                            run.worker.join(finish_limit)
                    if cancelled():
                        return
                    if stain and not (run.meta.get("progress") or {}).get("resolved"):
                        self._leave_progress(run, answer, "joining", "Joining the finished tiles…")
                        self.stain.join(run)
                        self._wait_until(lambda: run.meta.get("state") not in ACTIVE, finish_limit,
                                         "The finished tiles have not been joined yet.")
                        if run.worker is not None:
                            run.worker.join(finish_limit)
                    if cancelled():
                        return
                    if (not stain and run.meta.get("state") != "done" or
                            stain and not (run.meta.get("progress") or {}).get("resolved")):
                        raise RunError(run.meta.get("message") or "The run could not finish.", 409)
                    self._leave_progress(run, answer, "writing", "Saving to the sample…")
                    self._save_with_retry(run)
                if cancelled():
                    return
                self._stop(run)
                self._wait_until(lambda: run.meta["id"] not in self.stain.patches.exporting, 3600,
                                 "The training patches have not finished saving yet.")
                self._delete(run, reason="left", token=token)
            except Exception as exc:
                if cancelled():
                    return
                self.note(f"leaving run {run.meta['id']} failed: {exc}", run.meta["id"])
                run.leave_token = None
                saved = run.meta.get("zarr_save") or {}
                # `id`: this failure's own, which tells it from an earlier one a page may still read (main.js
                # updateLeaving) however close together the two came (two within a millisecond share `at`).
                at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                failure = {"leaving": None, "leave_error": {"answer": answer, "message": str(exc), "at": at,
                                                            "id": uuid.uuid4().hex}}
                if answer == "add" and saved.get("state") not in ("saved", "queued"):
                    # The add never reached the sample: a failed save (Try again; closing keeps it as "save failed").
                    # A discard's failure, or one after the save (stopping or removing the run), keeps the save's record:
                    # a saved run never asks to be saved again.
                    failure["zarr_save"] = {**saved, "state": "failed", "message": str(exc)}
                self._set(run, **failure)

    @staticmethod
    def _answers(policy):
        """The closing answer as a function of a run's id: one answer for every run ("add" | "discard"; "save" and True
        mean add; None or False: discard), or {run id: answer}, the page's answer bound to the runs it was
        given for, any other run taking "discard" (a run started after the answer was never shown)."""
        def one(value):
            if isinstance(value, bool):
                return "add" if value else "discard"
            if value in (None, "discard"):
                return "discard"
            if value in ("add", "save"):
                return "add"
            raise ValueError("Choose add or discard")
        if isinstance(policy, Mapping):
            answers = {str(run_id): one(value) for run_id, value in policy.items()}
            return lambda run_id: answers.get(run_id, "discard")
        answer = one(policy)
        return lambda run_id: answer

    def close(self, policy=None, timeout: float = 120, finish_limit: float = 3600, *, save=None):
        """Closing is leaving, then closing. `policy`: "add" | "discard" | None (nobody answered: discard) for
        every run, or {run id: "add" | "discard"} for the runs an answer was given for, any other run discarded
        (_answers). A decided run (a leave or save under way, a Finish & save: _decided) is waited for and completed
        whatever the answer; every other run leaves with its answer (_leave). Then every folder goes, except a run whose
        save failed while closing (kept with `closed: "save failed"` and the failure in its zarr_save: the next start
        removes it and its notice says why) and one whose save helper still uses it after the hour cap
        (`closed: "save running"`: never deleted under it). In an API session nothing is kept (its folders go with the
        session). The older runs: the row under way finishes, the rest stay on disk. A test engine finishes into its
        run folders as before and never writes into a sample here. `save` is the older callers' True/False."""
        if save is not None:
            policy = "add" if save else "discard"
        answer_of = self._answers(policy)
        if self.keep_unsaved:
            add = any(self._answers(value)("") == "add" for value in policy.values()) if isinstance(policy, Mapping) \
                else answer_of("") == "add"
            return self._close_test(add, timeout, finish_limit)
        with self.lock:                    # under the lock: no run registers after this (_room)
            self.closing = True
            runs = list(self.runs.values())
        self.recheck_wake.set()
        self._stop_older(finish_limit + SAVE_WAIT_S + 60)
        self._leave_all(runs, answer_of, timeout, finish_limit, keep_failed=self.session is None)
        self.followers_done.set()
        self.stain.worker.close()
        self.spool.close()                 # the spool's folder goes with the engine (every run has left)

    def _leave_all(self, runs, answer_of, timeout: float, finish_limit: float, keep_failed: bool) -> list[dict]:
        """Every run leaves while the engine closes (close, close_session): [{run, answer, had: what leaving would have
        lost (leave_kind), left: gone from the registry, message}]. An add of a run that cannot be saved is carried out
        as a discard, `answer` still "add" and `message` saying why (an API session's summary: not added, and why)."""
        outcomes = []
        for run in runs:
            run_id = run.meta["id"]
            had = leave_kind(run.public())
            worker = run.leave_worker
            answer = carried = (run.meta.get("leaving") or {}).get("answer")
            refused = None
            if worker is not None and worker.is_alive():
                worker.join(finish_limit + SAVE_WAIT_S)       # a leave under way is decided: waited for
            if run_id in self.runs and (worker is None or not worker.is_alive()):
                answer = carried = "add" if self._decided(run.public()) else answer_of(run_id)
                predicate = leave_kind(run.public())
                if answer == "add" and predicate["kind"] == "ask" and not predicate["can_add"]:
                    # Nothing it holds can go into the sample (its filters leave no cells; a Xenium folder): discarded,
                    # never kept as a failed save for the next start to report.
                    carried = "discard"
                    refused = NO_CELLS if predicate["why"] == "filtered" else "This run cannot be saved into its sample."
                token = run.leave_token = object()
                try:                                   # one run's failure never keeps the others from closing
                    self._leave(run, carried, token, finish_limit)
                except Exception as exc:               # (its record could not even be written: its folder is gone)
                    self.note(f"leaving run {run_id} while closing failed: {exc}", run_id)
            elif run_id in self.runs:
                run.leave_token = None
            message = refused or (run.meta.get("zarr_save") or {}).get("message")
            if run_id in self.runs:
                failed_add = carried == "add" and (run.meta.get("zarr_save") or {}).get("state") == "failed"
                try:
                    self._wait_save(run)
                except RunError:
                    pass
                try:
                    self._stop(run, timeout)
                except Exception as exc:
                    self.note(f"stopping run {run_id} while closing failed: {exc}", run_id)
                if keep_failed and run.saver is not None and run.saver.is_alive():
                    self._keep_closed(run, "save running")
                elif keep_failed and failed_add:
                    self._keep_closed(run, "save failed")
                else:
                    try:
                        self._delete(run, reason="left", force=True)    # held files stay in .trash until released
                    except Exception as exc:
                        self.note(f"removing run {run_id} while closing failed: {exc}", run_id)
            saved = run.meta.get("zarr_save") or {}
            outcomes.append({"run": run_id, "answer": answer, "had": had["kind"], "left": run_id not in self.runs,
                             "saved": saved.get("state") in ("saved", "queued"), "delivered": bool(saved.get("delivery")),
                             "message": message})
        return outcomes

    def _keep_closed(self, run: Run, closed: str):
        """A run left on disk as the engine closes (its save failed, or its save helper still uses its folder), out of
        the registry, with `closed` in its run.json for the next start."""
        run_id = run.meta["id"]
        with run.lock:
            run.meta.update(closed=closed, leaving=None)
            try:
                run.save()
            except OSError as exc:
                self.note(f"run {run_id} could not be marked {closed!r} while closing: {exc}", run_id)
            run.leave_token = None
        with self.lock:
            self.runs.pop(run_id, None)
            self.vector_guards.pop(run_id, None)
        self._release(run)
        self.spool.remove_run(run_id)
        print(f"SoPA companion: kept run {run_id} of {run.meta.get('sample_name') or 'a sample'} on disk while closing "
              f"({closed}): the next start removes it", flush=True)
        self.changed()

    def close_session(self, policy=None, timeout: float = 120, finish_limit: float = 3600):
        """API mode (server/api_serve.py's teardown calls it before the session's resources go,
        then Runs.close(policy="discard")): every run leaves with the session's answer, `policy` "add" | "discard" |
        None (nobody answered: discard) or {run id: answer}. Decided runs complete whatever it is: a leave or save
        under way, and a Finish & save unless the kernel has gone (then it is dropped; adds already pressed complete on
        disk). A run added here is delivered and waited for until the notebook has received it (dropped if the kernel
        goes meanwhile); so is every delivery still on its way. A run that held a result and was discarded, or whose
        add failed, is recorded for the final summary (session.note_not_added). Its folders go with the session."""
        answer_of = self._answers(policy)
        with self.lock:
            self.closing = True
            runs = list(self.runs.values())
        outcomes = self._leave_all(runs, answer_of, timeout, finish_limit, keep_failed=False)
        deadline = time.monotonic() + RECEIVE_S
        with self.lock:
            receipts = list(self.receipts.values())
        for received in receipts:
            received.wait(max(0.0, deadline - time.monotonic()))
        note = getattr(self.session, "note_not_added", None)
        for outcome in outcomes:      # a delivery is the session's own to report (Session.finish)
            # A run that held a result: unsaved (ask), or with a save under way that then failed (saving).
            if outcome["had"] not in ("ask", "saving") or outcome["saved"] or outcome["delivered"] or note is None:
                continue
            if outcome["answer"] == "add":
                note(outcome["run"], "failed", outcome["message"] or "It could not be added")
            else:
                note(outcome["run"], "discarded", "Discarded as the session closed")
        return outcomes

    def _close_test(self, save=False, timeout=120, finish_limit=3600):
        """Engine shutdown: refuse new runs; finish-and-save every run still sampling, held at the end of its schedule
        included (`save`), or abort it. A run whose finish was already requested (Finish & save) always finishes and is
        kept, whatever `save` says; a run still preparing has nothing to save and stops. Then wait for each run's
        worker to record its final state: runs being saved for as long as proseg writes their results (reported every
        minute, at most `finish_limit` s), the others at most `timeout` s; and discard everything that was not saved
        (unless `[proseg] keep_unsaved`). Anything still running after that is killed."""
        with self.lock:
            self.closing = True
        self.followers_done.set()
        for run in list(self.runs.values()):
            run.leave_token = None
            run.meta.pop("leaving", None)
            if run.meta.get("kind") == "patches" and run.process is not None:
                try:
                    portable.signal_group(run.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        active, saving, preparing = self.active(), [], 0
        for run in active:
            with run.lock:
                finishing = self._finishing(run)
                if not finishing and run.process is None:
                    run.stop_requested = True          # still preparing: nothing to save
                    preparing += 1
                    continue
            stain = run.meta.get("kind") == "stain"
            if (finishing or save) and not stain:
                saving.append(run)
            if finishing:
                continue                               # proseg is already writing its results: never abort it
            try:                                       # a stain run stops where it is: its tiles stay, to resume later
                self.control(run.meta["id"], "stop" if stain else "finish" if save else "abort")
            except RunError:
                pass
        if saving:
            print(f"SoPA companion: waiting for {len(saving)} run(s) to finish saving their results", flush=True)
        if preparing:
            print(f"SoPA companion: discarding {preparing} run(s) still preparing (nothing to save yet)", flush=True)
        deadline = time.monotonic() + timeout
        for run in active:
            worker = run.worker
            if worker is not None and run not in saving:
                worker.join(timeout=max(0.1, deadline - time.monotonic()))
        for run in saving:     # a whole sample's results take a while to write: never cut short by `timeout`
            worker, waited = run.worker, 0.0
            while worker is not None and worker.is_alive() and waited < finish_limit:
                worker.join(timeout=min(60.0, finish_limit - waited))
                waited += 60.0
                if worker.is_alive():
                    print(f"SoPA companion: still saving run {run.meta['id']} ({waited / 60:.0f} min)", flush=True)
        for run in active:
            process = run.process
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        # A save into a sample zarr is never cut off: it runs in its own process under a lock, journalled, and the
        # engine waits for it (it reports every minute). If it is still going after an hour the engine stops anyway;
        # the helper finishes or rolls back by itself and the next save into that sample recovers it if needed.
        for saver in list(self.savers):
            waited = 0.0
            while saver.is_alive() and waited < 3600:
                saver.join(timeout=60)
                waited += 60
                if saver.is_alive():
                    print(f"SoPA companion: waiting for a save into a sample zarr to finish ({waited / 60:.0f} min)", flush=True)
        with self.lock:
            indexes, self.indexes = list(self.indexes.values()), {}
        for index in indexes:                  # nothing is queried any more: unmap them before anything is deleted
            index.close()
        self.stain.worker.close()
        if not self.keep_unsaved:
            with self.lock:
                unsaved = [r for r in self.runs.values() if r.meta.get("state") != "done" and r.meta.get("kind") != "stain"]
                for run in unsaved:
                    self.runs.pop(run.meta["id"], None)
            for run in unsaved:
                shutil.rmtree(run.dir, ignore_errors=True)    # what is still held goes at the next start
        self.spool.close()                 # the spool's counted frames go with the engine (the run folders keep theirs)
