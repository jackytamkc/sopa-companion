"""Test a setting here: does a setting move the mixing check's suspects out of the cells that hold them, more than
their own RNA at the same spot? One cell's numbers move by themselves between identical runs, so a setting is judged
only the way runs started with it can be: fresh runs of one area to the end of proseg's schedule, as now and with the
setting, over the same suspects fixed before any run.

QcTests serves the API (server/app.py: GET /runs/{id}/qc/test/pool, POST/GET/DELETE /runs/{id}/qc/test, GET
/runs/{id}/qc/test/view, and "qc_test" {state, progress} in the run detail and its WebSocket pushes) and drives each test
in a thread of its own:
  * when it runs: on any run that reaches the end of its schedule. engine/qc_pool.py's reasons are of three kinds:
    its refusals refuse the POST (409 {detail, status, reasons}; this module adds "busy", a test of this run on its
    way); while a wait holds (the run not at its end yet; the mixing check off, learning or scoring its end; Go
    deeper's details to learn; another run sampling or another test going, other_run: this module's), or while the
    cell's pool is still being worked out (a whole sample's can take minutes), the press makes a "waiting" test: the
    app takes its own steps once (`_meet`: a run paused before its end resumed, the check switched on, a failed sample
    or its details asked again) and the test starts once nothing holds and its pool is there (`_wait`); its
    caveats (where the measurements did not cover it) let it run, judged with the nearest measured row, and its answer
    says so;
  * the targets: the clicked cell's pool, fixed by engine/qc_pool.py at the parent's last frame (the parent waits paused
    at the end of its schedule; one sampling on past it is paused as the runs start), then found in the test's shared
    input (`shared_input`: the parent's own input, or the area's rows of it), which every test run reads;
  * the runs: 2N fresh runs of the area (N per start, engine/qc_rule.py's row: roles A1..AN as now, B1..BN with the
    setting: `roles_for`), made from that one input, all at once, THREADS threads each, with the parent's starting cells
    and settings (B with the new value from its start), made paused and each given its hold before its first iteration
    (`_hold_runs`: proseg makes the hold stronger by the input's transcripts per cell, so each run is given the value
    whose hold equals the parent's at the frame tested; only the value given to proseg changes), then to the end of
    the schedule at full speed; they end plainly, so that proseg writes its per-transcript file, which is all a test
    run writes besides its frames. A run that needs proseg's retry without factorisation has every run of the test made
    again without it (both sides alike, once).
    Clusters & UMAP are off for them and their meta names the test ("qc_test": {of, role, side, test}: the page leaves
    them out of its lists);
  * the answer: every run read (engine/qc_readout.py), then judged (engine/qc_rule.py answer, with the row and the
    caveats: `answer_caveats`), which the result carries as it is; then the runs are deleted (their folders to the trash
    at once, emptied in the background) and the test's own input removed, whatever happened. One test at a time per
    run, and its runs sample alone.
The test's corner view (GET /runs/{id}/qc/test/view: `view`): a crop around the cell (`crop_around`) of the newest
counted frame of runs A1 and B1 (a frame with proseg's own count), the cell and the neighbours its suspects come from
found there by their Xenium ids, and how much of each of the cell's suspects (the pool's, as the page saw them glow)
proseg counts for the cell there, held (engine/held.py weights: exact from the frame's where export of the view's box,
which each view asks of A1 and B1 for their next counted frame, Runs.request_where; else the frame's per-layer labels
times the cell's keep; `QcTests._run_view`; each suspect is the run's own transcript, told from others at its place by
depth, else unsure and never counted: `match`); its total, proseg's count of the cell. The views keep their runs'
spools to the newest SPOOL_KEPT counted files. Each run's view is worked out once per new frame and kept, only ever
for a newer frame while the runs sample, so the last ones outlive the runs.
A test lives in memory only (its runs are unsaved: an engine restart discards them, and a waiting one with them). One
that ended stays, for the page's readout, until the next one or until DELETE dismisses it.
"""
from __future__ import annotations

import base64
import copy
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time

import numpy as np

from ..analysis import counts_file
from . import held, leftovers, proseg_options, qc_pool, qc_readout, qc_rule
from . import qc_rules as qr
from .deeper_where import QUERY_MAX, DeeperWhere, Dense, Plane, _decode
from .qc import QcError
from .runs import ACTIVE, TERMINAL, RunError
from .tiles import Container

# The settings a test takes, in the form's order: three tried here, then six more of proseg's own options, each stock
# and run from its start in fresh runs (the page's twin: frontend/test/fixtures/qc-test-settings.json).
FLAGS = ("cell-compactness", "prior-weight", "diffusion-probability", "burnin-voxel-size", "ab-nihlo-bubble-prob",
         "no-diffusion", "diffusion-sigma-near", "density-bins", "voxel-layers")
# The short plain name each goes by on the page: never the knob's own label, whose "Prior", "diffusion" and "voxel" are
# banned words on screen.
NAMES = {"cell-compactness": "compactness", "prior-weight": "hold", "diffusion-probability": "far drift",
         "burnin-voxel-size": "starting grid", "ab-nihlo-bubble-prob": "background gaps", "no-diffusion": "drift",
         "diffusion-sigma-near": "short drift distance", "density-bins": "crowding levels", "voxel-layers": "depth layers"}
TRIED = {flag: flag in ("cell-compactness", "prior-weight") for flag in FLAGS}   # the others: not tried here yet
# How the form offers each and the POST takes it: a number, a whole number, one of the run's own choices (the starting
# grid's: `grid_choices`), or a switch (drift: 1 = off, 0 = on, as proseg's no-diffusion true or false).
KINDS = {**dict.fromkeys(FLAGS, "number"), "burnin-voxel-size": "choice", "no-diffusion": "switch", "density-bins": "whole",
         "voxel-layers": "whole"}
TEST_RANGES = {"diffusion-probability": (0.1, 0.3)}     # the values a test takes; the others: proseg_options' own bounds
GRID_STEPS = (1, 2, 3, 4)      # the starting grid's choices: these multiples of the run's voxel size (proseg's float32
GRID_MAX_UM = 4.0              # whole-ratio rule), up to this: coarser grids start fewer cells
LAYER_COST = 0.0226            # a test's time x (1 + LAYER_COST x (layers - 4)) past 4 depth layers (measured)
DRIFTING = ("diffusion-probability", "diffusion-sigma-near")    # do nothing while drift is off in the run
STARTUP = ("burnin-voxel-size", "density-bins", "voxel-layers")  # not live knobs: read from the run's start
SQUARE_UM = 500.0              # the largest area a test re-runs (qc_pool.area_of's side)
THREADS = 8                    # each test run's threads (2N runs at once)
INPUTS = "qc-tests"            # <data_dir>/qc-tests/<test id>/: a test's own input, when it has one
VIEW_ROLES = ("A1", "B1")      # the runs the corner shows: one as now, one with the setting
# Never copied from the parent: the engine's threads (THREADS), a Temperature set (from a run's start it would settle
# the outlines at once, where proseg's own schedule cools by itself: the answer says the parent's settings were not
# proseg's defaults, `parent_defaults`), the retry's flag (the test's own rule sets it for every run), the outputs (a
# test run writes only its per-transcript file), and every "output-*" flag.
LEFT_OUT = ("nthreads", "temperature", "no-factorization", "exclude-spatialdata-transcripts")
STATES = ("waiting", "preparing", "running", "comparing", "done", "failed", "cancelled")
ON_ITS_WAY = STATES[:4]        # cancelled by DELETE, "busy" to a second POST, cancelled as the engine closes
GOING = STATES[1:4]            # its runs on their way (going): no other run resumes meanwhile (server/app.py), and a
                               # waiting test waits (its parent must still resume and step: a waiting test is not going)
WAIT_S = 1.0                   # a waiting test asks again this often
PRESS_WAIT_S = 5.0             # a press waits this long for its cell's pool (the page's own wait for it); one still
                               # being worked out then (a whole sample's can take minutes) makes the test wait for it
FOLLOW_S = 0.25                # the test runs are looked at this often
PUSH_S = 1.0                   # the parent's run detail goes out with the progress at most this often
TIMEOUT_S = 600.0              # a test without an answer by then (from its POST, or from the end of its wait) failed,
                               # times the parent's schedule over the measured one where that is longer
CHECK_FAILED = "The mixing check couldn't learn this sample"            # a waiting test's failures, in its words
DETAILS_FAILED = "Go deeper's details couldn't be learned for this sample"
STOP_S = 30.0                  # how long the test's runs may take to stop, all together, before they are deleted anyway
JOIN_S = 10.0                  # DELETE (and the engine closing) waits this long for a test to clear its runs away
# The corner view:
MATCH_UM = 0.01                # a suspect is a run's own transcript of its gene within this of its x and of its y
UNSURE = -2                    # `match`: a suspect whose transcript cannot be told from another of its gene there
VIEW_PAD_UM = 12.0             # the crop: the cell's footprint grown by this on every side, made square, ...
VIEW_UM = (48.0, 140.0)        # ... at least and at most this wide, inside the test's area
SPOOL_KEPT = 2                 # a viewed run's counted files kept in its spool, the newest (Runs.spool.prune)
_WHOLE = re.compile(r"[0-9]{1,9}")
_PAST_SAMPLING = ("finishing", "done")    # a run's states once proseg finished sampling (live/sampling_done)


class _Stop(Exception):
    """The test ends without an answer: cancelled, its run ended, or the engine closes (state "cancelled")."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "cancelled")
        self.message = message


class _Failed(Exception):
    """The test ends without an answer, in words of its own (state "failed", `message` shown as it is)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class _Refused(Exception):
    """A waiting test ends on refusals (engine/qc_pool.py's, or its run ended before its end): state "failed", its
    `reasons` those, its message the first one's words."""

    def __init__(self, reasons: list[dict]):
        super().__init__(reasons[0]["words"])
        self.reasons = [dict(r) for r in reasons]


class _WithRoom:
    """The engine's runs as Runs.create sees them while a test makes its runs: `extra` more places than [proseg]
    max_active_runs (which counts every run that holds proseg, the parent waiting paused among them). Everything else
    is the engine's own Runs: the runs made are its runs."""

    def __init__(self, runs, extra: int):
        self._runs, self.max_active = runs, int(runs.max_active) + extra

    def __getattr__(self, name):
        return getattr(self._runs, name)


# ---- the test's runs -----------------------------------------------------------------------------------------------
def roles_for(n: int) -> list[str]:
    """The roles of a test with n runs per side: A1..An (as now), then B1..Bn (with the setting)."""
    return [f"{side}{i}" for side in "AB" for i in range(1, n + 1)]


def _own_ids(ids) -> bool:
    """Whether a file's transcript ids (an arrow column) can stay as they are: never null and each once. They join
    the targets to every run's per-transcript file (engine/qc_readout.py), so two transcripts may never share one."""
    import pyarrow.compute as pc
    return ids.null_count == 0 and pc.count_distinct(ids).as_py() == len(ids)


def shared_input(parent_input: Path, area: qc_pool.Area, folder: Path) -> dict:
    """The one input every run of a test reads, from the parent's own (<parent run>/input/transcripts.parquet: runs.py
    writes it before proseg starts and keeps it while the run exists; there is no other source, as a fresh prepare
    would not be the run "as now"): {path, owned (a file of the test's own, removed with it), transcripts (its rows),
    ids ("own": the parent's, unique and never null; "rows": renumbered 0..n-1 as uint64), bytes (written), seconds}.
    Re-run as it is: the parent's file itself, unless its ids must be renumbered (then a copy). A square or a crop: the
    parent's rows with x0 <= x < x1 and y0 <= y < y1 (the helper's own half-open rule, zarr_bridge/helper.py prepare;
    a lasso's are its own transcripts only, as its input holds). A file of the test's own is folder/transcripts.parquet
    (proseg reads it as parquet by that name), every column kept with its type (proseg takes the Xenium schema by type),
    zstd, written through a temporary file: a run never finds half of it. ValueError, in words shown after "The test
    could not finish: ", when the parent's input is missing or the area holds no transcript."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    began = time.perf_counter()
    parent_input = Path(parent_input)
    if not parent_input.is_file():
        raise ValueError("your run's input is missing")
    if area.rule != "as_is":
        x0, y0, x1, y1 = (float(v) for v in area.box)
        table = pq.read_table(parent_input, filters=[("x_location", ">=", x0), ("x_location", "<", x1),
                                                     ("y_location", ">=", y0), ("y_location", "<", y1)])
        if not table.num_rows:
            raise ValueError(f"the {'square' if area.rule == 'square' else 'area'} around this cell has no transcripts")
    else:
        ids = pq.read_table(parent_input, columns=["transcript_id"]).column("transcript_id")
        if _own_ids(ids):
            return {"path": str(parent_input), "owned": False, "transcripts": len(ids), "ids": "own", "bytes": 0,
                    "seconds": round(time.perf_counter() - began, 3)}
        table = pq.read_table(parent_input)
    own = _own_ids(table.column("transcript_id"))
    if not own:
        at = table.schema.get_field_index("transcript_id")
        table = table.set_column(at, table.schema.field(at).with_type(pa.uint64()),
                                 pa.array(np.arange(table.num_rows, dtype=np.uint64)))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "transcripts.parquet"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "owned": True, "transcripts": table.num_rows, "ids": "own" if own else "rows",
            "bytes": path.stat().st_size, "seconds": round(time.perf_counter() - began, 3)}


def no_factorization_of(meta: dict, info: dict | None) -> bool:
    """Whether the parent ran without factorisation (its runs.py retry, its options, or what proseg said it did): its
    test runs are made without it too."""
    return (bool(meta.get("retried_without_factorization")) or (meta.get("options") or {}).get("no-factorization") is True
            or (info or {}).get("factorization") is False)


def _schedule(info: dict | None, options: dict) -> dict:
    """{burnin, samples, hillclimb} of a run: info.json's schedule, else its options, else proseg's defaults."""
    found = (info or {}).get("schedule") if isinstance(info, dict) else None
    keys = {"burnin": "burnin-samples", "samples": "samples", "hillclimb": "hillclimb"}
    if isinstance(found, dict) and all(isinstance(found.get(key), int) for key in keys):
        return {key: int(found[key]) for key in keys}
    from . import proseg_options
    values = {**{o["flag"]: o["default"] for o in proseg_options.OPTIONS}, **(options or {})}
    return {key: int(values[flag]) for key, flag in keys.items()}


def schedule_total(info: dict | None, options: dict) -> int:
    """The iteration a run's schedule ends at: burn-in + samples + hill-climb (450 with proseg's defaults)."""
    schedule = _schedule(info, options)
    return schedule["burnin"] + schedule["samples"] + schedule["hillclimb"]


def parse_preview(cell: str | None, wait: str | None) -> tuple[int, float]:
    """GET /runs/{id}/qc/test/pool's query, as text: (cell, wait). `cell` a cell number (label - 1) in digits; `wait`
    (seconds the reply may wait for the pool) absent for 0, else a number from 0 to qc_pool.WAIT_MAX_S. QcError 422 for
    anything else, in the engine's words (as the where route's query: deeper_where.parse)."""
    cell = (cell or "").strip()
    if not _WHOLE.fullmatch(cell):
        raise QcError("cell must be a cell number (label - 1)", 422)
    seconds = 0.0
    if wait is not None:
        try:
            seconds = float(wait.strip())
        except ValueError:
            raise QcError("wait must be 0 to 10 seconds", 422) from None
        if not (math.isfinite(seconds) and 0.0 <= seconds <= qc_pool.WAIT_MAX_S):
            raise QcError("wait must be 0 to 10 seconds", 422)
    return int(cell), seconds


def parent_defaults(asked: dict, temperature=None) -> bool:
    """Whether the parent ran with proseg's default settings, as measured: every setting of `asked` (its options and
    live values as the test runs take them, `_asked`) at proseg's default (float32's precision: qc_rules.same), and no
    Temperature set (`temperature`: the parent's, `temperature_of`, which its test runs leave out). The answer's tooltip
    says so when not."""
    from . import proseg_options
    if temperature is not None:
        return False
    defaults = {o["flag"]: o["default"] for o in proseg_options.OPTIONS}
    return all(flag in defaults and qr.same(value, defaults[flag]) for flag, value in asked.items())


def temperature_of(meta: dict):
    """The Temperature the run was set to (its live value since, else its option), None when none was set."""
    live, options = meta.get("live_values") or {}, meta.get("options") or {}
    return next((v for v in (live.get("temperature"), options.get("temperature")) if v is not None), None)


def _asked(meta: dict) -> dict:
    """The parent's settings as its test runs take them: its options with the live values set since, without LEFT_OUT,
    the "output-*" flags and unset values."""
    asked = {**(meta.get("options") or {}), **(meta.get("live_values") or {})}
    return {k: v for k, v in asked.items() if k not in LEFT_OUT and not k.startswith("output-") and v is not None}


def _default(flag: str):
    from . import proseg_options
    return next(o["default"] for o in proseg_options.OPTIONS if o["flag"] == flag)


def current_of(run, flag: str):
    """The run's value of `flag` now: the one it was asked for (its options with the live values set since), else what
    its newest frame shows (proseg's default), else the option's default. Drift (a switch): 1 while it is off, else 0. A
    setting proseg reads only at its start (STARTUP): `startup_value`."""
    if flag in STARTUP:
        return startup_value(run, flag)
    asked = {**(run.meta.get("options") or {}), **(run.meta.get("live_values") or {})}
    if KINDS.get(flag) == "switch":
        if isinstance(asked.get(flag), bool):
            return int(asked[flag])
        with run.lock:
            shown = (run.frames[-1].get("live") or {}).get(flag) if run.frames else None
        return int(shown) if isinstance(shown, bool) else int(bool(_default(flag)))
    if qr.number(asked.get(flag)) is not None:
        return asked[flag]
    with run.lock:
        shown = (run.frames[-1].get("live") or {}).get(flag) if run.frames else None
    if qr.number(shown) is not None:
        return qr.clean(shown)
    return _default(flag)


def startup_value(run, flag: str):
    """What the run started with for a setting proseg reads only at its start: proseg's own record in info.json (the
    starting grid's `burnin_voxel_size`, the depth layers' `layers`: 1 on a sample without depth, whatever was asked),
    else the one it was asked for, else proseg's default (crowding levels: proseg records nothing of them)."""
    info = run.info or {}
    if flag == "burnin-voxel-size" and qr.number(info.get("burnin_voxel_size")) is not None:
        return qr.clean(info["burnin_voxel_size"])
    if flag == "voxel-layers" and isinstance(info.get("layers"), int) and info["layers"] > 0:
        return int(info["layers"])
    asked = run.meta.get("options") or {}
    return asked[flag] if qr.number(asked.get(flag)) is not None else _default(flag)


def voxel_of(run) -> float:
    """The run's final voxel size (µm): proseg's own record, else the one asked for, else proseg's default."""
    info, asked = run.info or {}, run.meta.get("options") or {}
    for value in (info.get("voxel_size"), asked.get("voxel-size"), _default("voxel-size")):
        if qr.number(value) is not None and value > 0:
            return float(value)
    return 1.0


def grid_choices(run) -> list[float]:
    """The starting grids (µm) a test of this run takes: k x its voxel size for k in GRID_STEPS, each up to GRID_MAX_UM
    that proseg's own rule accepts (the ratio to the voxel size whole in float32: proseg panics on 0.9 / 0.3), the run's
    own left out."""
    from . import proseg_options
    v, now = voxel_of(run), startup_value(run, "burnin-voxel-size")
    low = next(o["min"] for o in proseg_options.OPTIONS if o["flag"] == "burnin-voxel-size")
    out = []
    for k in GRID_STEPS:
        b = float(f"{k * v:.6g}")
        ratio = float(np.float32(b) / np.float32(v))
        if b < low or b > GRID_MAX_UM + 1e-9 or ratio < 1 or ratio - math.floor(ratio) > 1e-5 or qr.same(b, now):
            continue
        out.append(b)
    return out


def layers_max(run, box=None) -> int:
    """The most depth layers a test run of the area `box` ([x0, y0, x1, y1] µm; else the largest a test re-runs, a 500
    µm square) can hold in memory, as Runs.create counts it (runs.budget_columns), at most proseg's own 256."""
    from . import proseg_options, runs as runs_module
    high = next(o["max"] for o in proseg_options.OPTIONS if o["flag"] == "voxel-layers")
    w, h = (box[2] - box[0], box[3] - box[1]) if box else (SQUARE_UM, SQUARE_UM)
    columns, _, _, _ = runs_module.budget_columns(_asked(run.meta), w, h)
    return int(max(1, min(high, runs_module.MAX_VOXELS // max(1, columns))))


def layer_factor(layers) -> float:
    """A test's time against one at proseg's default 4 depth layers (measured on a 500 µm square)."""
    return max(1.0, 1.0 + LAYER_COST * ((qr.number(layers) or 4) - 4))


def flat(run) -> bool | None:
    """Whether the sample has no depth for proseg (one layer, whatever is asked): its info.json's depth span is 0, or
    the run asked to ignore depth. None while its info.json is not there yet."""
    if _asked(run.meta).get("ignore-z-coord") is True:
        return True
    info = run.info
    if not isinstance(info, dict) or not isinstance(info.get("z"), dict):
        return None
    span = qr.number(info["z"].get("thickness_um"))
    return None if span is None else span <= 1e-6


def proseg_value(flag: str, value):
    """A test's value as proseg_options takes it for the B runs: the switch a bool (1: drift off), a whole number an
    int, the rest as given."""
    kind = KINDS.get(flag)
    return (value == 1) if kind == "switch" else int(value) if kind == "whole" else value


def settings_tested(run, box=None, seq=None) -> dict:
    """What the form needs of every setting, from the run alone (the pool reply's `settings`): {flag: {now, min, max,
    refusal}} in FLAGS' order; `now` the value a test is against (`current_at` at the frame tested), min and max for the
    numbers (TEST_RANGES, else proseg_options' bounds; depth layers up to `layers_max` of the area `box`), the starting
    grid's `voxel` and `choices`; `refusal` null or {code, words} where nothing can be tested with that setting here."""
    from . import proseg_options
    out = {}
    drift_off = _asked(run.meta).get("no-diffusion") is True
    for flag in FLAGS:
        option = next(o for o in proseg_options.OPTIONS if o["flag"] == flag)
        entry = {"now": current_at(run, flag, seq), "refusal": None}
        if KINDS[flag] in ("number", "whole"):
            low, high = TEST_RANGES.get(flag, (option.get("min"), option.get("max")))
            entry.update(min=low, max=layers_max(run, box) if flag == "voxel-layers" else high)
        if flag == "burnin-voxel-size":
            entry.update(voxel=voxel_of(run), choices=grid_choices(run))
        if flag in DRIFTING and drift_off:
            entry["refusal"] = {"code": "drift_off", "words": f"{NAMES[flag].capitalize()} does nothing while drift is "
                                                               "switched off in this run"}
        if flag == "voxel-layers" and flat(run):
            entry["refusal"] = {"code": "flat", "words": FLAT_WORDS}
        out[flag] = entry
    return out


FLAT_WORDS = "This sample has no depth: proseg makes one layer whatever is set"


def answer_caveats(caveats, judged, k) -> list[dict]:
    """The caveats a result carries: the pool's (its own too_few left out), then too_few with the judged numbers (have:
    the suspects the pool judged, `judged`; need: the row's `k`) when it judged fewer than k, in the page's order."""
    found = [dict(c) for c in caveats if c.get("code") != "too_few"]
    if isinstance(judged, int) and not isinstance(judged, bool) and judged < int(k):
        found.append(qc_pool.reason("too_few", have=judged, need=int(k)))
    return qc_pool.ordered(found)


def _area(area: qc_pool.Area | None) -> dict | None:
    """A test's area as the API gives it: {rule, box, inner} (µm); None until its pool is taken."""
    if area is None:
        return None
    return {"rule": area.rule, "box": [float(v) for v in area.box], "inner": [float(v) for v in area.inner]}


def _in_order(reasons) -> list[dict]:
    """Reasons in the order the page lists them (engine/qc_pool.py REASON_WORDS)."""
    return qc_pool.ordered(reasons)


def _place_of(suspects) -> tuple[float, float] | None:
    """The cell's place for the corner's crop when its outline cannot be read: the middle of its suspects (the pool's,
    which lie in the cell as Go deeper placed them; µm), None without one. The pool gives no place of its own."""
    points = [(float(s["x"]), float(s["y"])) for s in suspects if isinstance(s, dict)
              and all(qr.number(s.get(axis)) is not None for axis in ("x", "y"))]
    if not points:
        return None
    return (round(sum(x for x, _ in points) / len(points), 3), round(sum(y for _, y in points) / len(points), 3))


class Test:
    """One test of one run (the parent): what the page asked for, the pool it tests (a qc_pool.Pool, given or `take`n
    once its wait is over; `matched` once found in the shared input), where it stands, and its answer. Without a pool
    it is "waiting": its cell, start, row and roles as pressed (`cell`, `cell_name`, `start`, `row`), its area and
    pool to come. For the corner view: the cell's Xenium id, its suspects and the neighbours they come from (the
    pool's), and the parent's frame the pool was taken at (`seq`)."""

    def __init__(self, parent: str, pool: qc_pool.Pool | None, flag: str, value, current, clock, *, runs_per_side: int,
                 of: int = 450, iter_marks: tuple[int, int] = (200, 400), was_running: bool = False,
                 hold: dict | None = None, no_factorization: bool = False, parent_defaults: bool = True,
                 asked: dict | None = None, cell: int | None = None, cell_name: str | None = None,
                 start: str | None = None, row: tuple[str, str] | None = None, waiting=()):
        self.id = secrets.token_hex(4)
        self.parent, self.pool, self.matched = parent, None, None
        self.cell, self.cell_name, self.start, self.area = cell, cell_name, start, None
        self.row = None if row is None else (str(row[0]), str(row[1]))      # the measured row it is judged with
        self.flag, self.name, self.value, self.current, self.tried = flag, NAMES.get(flag), value, current, TRIED.get(flag)
        self.runs_per_side, self.roles = runs_per_side, roles_for(runs_per_side)
        self.asked = dict(asked or {})              # the parent's settings, as every test run takes them (_asked)
        self.parent_defaults = parent_defaults
        self.hold, self.was_running = hold, was_running
        self.scale = None                           # the parent's hold scale, unrounded (`scale_of`), once taken
        self.no_factorization, self.reruns = no_factorization, 0
        self.factor = 1.0                           # its time against proseg's 4 depth layers (`set_factor`)
        self.estimate_s = None if self.row is None else qc_pool.ESTIMATE_S.get(self.row[0])
        self.input: dict | None = None              # shared_input's, once made
        self.state, self.message, self.result = ("preparing" if pool is not None else "waiting"), None, None
        self.waiting: list[dict] = [dict(w) for w in waiting]     # the waits now; [] once past waiting
        self.reasons: list[dict] = []               # the refusals a waiting test ended on
        self.met: set[str] = set()                  # the app's own steps taken for it (QcTests._meet: once each)
        self.resumed = False                        # its press resumed the parent (paused again on Cancel)
        self.runs: dict[str, str | None] = dict.fromkeys(self.roles)
        # The progress the run detail carries: iter (the lowest run's newest, capped at `of`), done, sides {A, B} (each
        # side's lowest), marks [burn-in end, sampling end], phase (waiting, preparing, loading, sampling, writing,
        # reading), retry (made again without factorisation); `_follow` keeps it.
        self.progress = {"iter": 0, "of": int(of), "runs": 2 * runs_per_side, "done": 0, "sides": {"A": 0, "B": 0},
                         "marks": [int(iter_marks[0]), int(iter_marks[1])], "phase": self.state, "retry": False}
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.clock = clock
        self.began = clock()
        self.marks: dict[str, float] = {}           # when each step was reached (the clock), for `seconds`
        self.pushed = -math.inf
        # The corner view (`place`: where its crop goes when the cell's outline cannot be read, `_place_of`).
        self.seq, self.place = None, None
        self.suspects: list[dict] = []
        self.sources: list[int] = []
        self.source_names: list[str | None] = []
        self.crop: list[float] | None = None        # the corner view's crop (µm), once worked out (QcTests._crop)
        self.where_box: list[float] | None = None   # the box A1 and B1 are asked to export with their counted frames:
                                                    # the crop and the suspects (QcTests._crop)
        self.views: dict[str, dict] = {}            # role -> {key, order, final, view} (QcTests._view): the newest
                                                    # frame's view worked out, the last one's once final
        self.names: dict[str, dict] = {}            # role -> that run's cells and genes by name (run_names)
        self.view_lock = threading.Lock()           # one view worked out at a time: one decode per run and frame
        if pool is not None:
            self._hold(pool)

    def _hold(self, pool: qc_pool.Pool):
        """The pool's own fields (its caveats with it): its cell, start, area and row, its frame, the corner's suspects
        and sources (its runs per side are the row's as pressed: a run's start and region never change)."""
        self.pool = pool
        self.cell, self.cell_name, self.start, self.area = pool.seed, pool.seed_name, pool.start, pool.area
        self.row = (str(pool.row[0]), str(pool.row[1]))
        self.estimate_s = qc_pool.ESTIMATE_S.get(self.row[0])
        if self.estimate_s is not None:
            self.estimate_s = round(self.estimate_s * self.factor)
        self.seq, self.place = pool.seq, _place_of(pool.view_suspects)
        self.suspects = [dict(s) for s in pool.view_suspects]
        self.sources = list(pool.view_sources)
        self.source_names = [pool.names.get(s) for s in pool.view_sources]

    def take(self, pool: qc_pool.Pool, run):
        """The pool taken (`_hold`), with the parent's settings read again as every test run takes them (`_asked`,
        whether they are proseg's defaults, its factorisation, its schedule's end and marks), the value it is against
        (`current`: `current_at`) and its hold at the frame tested (the pool's F: `hold_at`, its runs' aim). The hold is
        read at F alone (`weight_at`): a value set live since F, or sampled past it, is neither the runs' aim nor the
        value tested against, and the settings the runs take and the defaults say F's hold too. Its caller holds
        QcTests.lock."""
        meta, info = run.meta, run.info
        options = meta.get("options") or {}
        schedule, asked = _schedule(info, options), _asked(meta)
        weight = weight_at(run, pool.seq)
        if not qr.same(weight, current_of(run, "prior-weight")):
            asked["prior-weight"] = weight          # changed since F: F's, as its runs are given it
        current = weight if self.flag == "prior-weight" else current_of(run, self.flag)
        self._hold(pool)
        self.current, self.asked, self.parent_defaults = current, asked, parent_defaults(asked, temperature_of(meta))
        self.no_factorization, self.hold = no_factorization_of(meta, info), hold_at(run, pool.seq, weight)
        self.scale = scale_of(info)
        self.set_factor(run)
        self.progress.update(of=schedule_total(info, options),
                             marks=[schedule["burnin"], schedule["burnin"] + schedule["samples"]])

    def set_factor(self, run):
        """The test's time against one at proseg's 4 depth layers (`layer_factor`): the slower side's, A's the run's own
        layers, B's the value in a test of the depth layers. Its time limit and estimate take it."""
        now = startup_value(run, "voxel-layers")
        self.factor = max(layer_factor(now), layer_factor(self.value if self.flag == "voxel-layers" else now))
        if self.estimate_s is not None:
            self.estimate_s = round(qc_pool.ESTIMATE_S.get(self.row[0], self.estimate_s) * self.factor)

    @property
    def square(self) -> list[float] | None:
        """The stage's dashed square: the area's box whenever it is a window around the cell (a square or a crop), else
        None."""
        return list(self.area.box) if self.area is not None and self.area.rule != "as_is" else None

    @property
    def inner(self) -> list[float] | None:
        return None if self.area is None else list(self.area.inner)


def _codes(path: Path) -> np.ndarray:
    """A .qc.bin: one byte per cell, a zstd frame (or raw)."""
    raw = Path(path).read_bytes()
    if raw[:4] == b"\x28\xb5\x2f\xfd":
        import zstandard
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    return np.frombuffer(raw, np.uint8)


start_of = qr.start_of           # how a run's cells started: "cells" | "paired" | "nuclei" (callers read qr.start_of)


# The most hold proseg takes: prior-weight's bound in engine/proseg_options.py, which a live value is kept to.
HOLD_MAX = float(next(o["max"] for o in proseg_options.OPTIONS if o["flag"] == "prior-weight"))


def scale_of(info) -> float | None:
    """Proseg's hold scale for a run: its input's transcripts per cell over 100, at least 1, in float32 exactly as proseg
    works it out (native/proseg/src/main.rs:955-956, stock proseg's: `transcripts as f32 / cells.max(1) as f32`, then
    `(that / 100.0).max(1.0)`), from info.json's transcripts and cells; the hold proseg applies is the value it is given
    times this, from the first iteration on (apply_live, for a value set live). None without them."""
    info = info or {}
    transcripts, cells = qr.number(info.get("transcripts")), qr.number(info.get("cells"))
    if transcripts is None or cells is None:
        return None
    mean = np.float32(transcripts) / np.float32(max(cells, 1.0))
    return float(max(np.float32(1.0), np.float32(mean / np.float32(100.0))))


def hold_of(info, value) -> dict | None:
    """How firmly a run holds its starting cells: {value (its prior-weight), effective (that value as proseg applies
    it, the float32 product value x scale, to 6 significant digits: the weight the run's log prints, info.json's
    prior_weight, while the value is the one it started with), scale (`scale_of`, to 6 significant digits)}. The page
    aims a looser hold at an effective weight of about 1: value / effective. None without the run's counts (info.json:
    transcripts, cells) or a value."""
    weight, scale = qr.number(value), scale_of(info)
    if weight is None or scale is None:
        return None
    with np.errstate(over="ignore"):
        effective = float(np.float32(weight) * np.float32(scale))
    return {"value": value, "effective": float(f"{effective:.6g}"), "scale": float(f"{scale:.6g}")}


def given_hold(effective: float, scale: float) -> float:
    """The value that, given to a run of proseg scale `scale` (`scale_of`), makes its hold `effective`: the float32
    nearest effective / scale, as a float whose shortest decimal reads back as that float32, whether proseg takes it on
    its command line or as a live setting."""
    return float(np.float32(effective / scale))


def weight_at(run, seq):
    """The run's prior-weight at frame `seq`: the one proseg reported in that frame's record (its live values: the value
    it sampled that frame with, as the page gets it), else its value now (`current_of`; also for seq None). A test reads
    the hold here alone, at the frame tested (F, its schedule's end), whatever was set live since or sampled past it:
    its runs' aim (`hold_at`), the value a test of the hold is against (`current_at`), and the parent's settings."""
    with run.lock:
        frame = run.frames[seq] if isinstance(seq, int) and 0 <= seq < len(run.frames) else None
        live = frame.get("live") if isinstance(frame, dict) else None
        found = live.get("prior-weight") if isinstance(live, dict) else None
    return qr.clean(found) if qr.number(found) is not None else current_of(run, "prior-weight")


def current_at(run, flag: str, seq=None):
    """The value a test of `flag` is against (its `current`): the run's value now (`current_of`), but the hold's at the
    frame tested (`weight_at`: the value its runs as now are given the strength of, _hold_runs)."""
    return weight_at(run, seq) if flag == "prior-weight" else current_of(run, flag)


def hold_at(run, seq, weight=None) -> dict | None:
    """The run's hold at frame `seq` (`hold_of` of `weight_at`, or of `weight` when the caller read it), with the frame
    (`seq`) and what its test's runs were given (`given`: null until they are, `_hold_runs`). None as hold_of."""
    hold = hold_of(run.info, weight_at(run, seq) if weight is None else weight)
    return None if hold is None else {**hold, "seq": seq, "given": None}


def frame_tested(run):
    """The frame a test of the run reads (F, the end of its schedule: `schedule_done.seq`, the pool's frame), None
    before the run reached it."""
    done = run.meta.get("schedule_done") if isinstance(run.meta.get("schedule_done"), dict) else {}
    return done.get("seq")


def hold_tested(run) -> dict:
    """The run's hold at the frame a test reads (`frame_tested`: F; its value now while F is not known), as the pool
    reply gives it to the page's form: {value (`weight_at`: F's record's, else the value now), effective, scale
    (`hold_of`; None without the run's counts in its info.json), seq (F, or None)}. A test of the hold is against
    `value` (`current_at`: the POST's 422 #12), a hold set live since F aside."""
    seq = frame_tested(run)
    value = weight_at(run, seq)
    return {**(hold_of(run.info, value) or {"value": value, "effective": None, "scale": None}), "seq": seq}


# ---- the corner view ---------------------------------------------------------------------------------------------------
def crop_around(footprint, place, square) -> list[float]:
    """The corner view's crop [x0, y0, x1, y1] µm: the cell's footprint box (µm; None: none, then its
    `place` (x, y)) grown by VIEW_PAD_UM on every side and made square on its longer side, VIEW_UM[0] to VIEW_UM[1]
    across, centred on it; then moved inside the test's area `square` (and no wider than it)."""
    if footprint is not None:
        cx, cy = (footprint[0] + footprint[2]) / 2.0, (footprint[1] + footprint[3]) / 2.0
        side = max(footprint[2] - footprint[0], footprint[3] - footprint[1]) + 2.0 * VIEW_PAD_UM
    else:
        (cx, cy), side = place, 0.0
    sx0, sy0, sx1, sy1 = (float(v) for v in square)
    side = min(max(side, VIEW_UM[0]), VIEW_UM[1], sx1 - sx0, sy1 - sy0)
    x0, y0 = min(max(cx - side / 2.0, sx0), sx1 - side), min(max(cy - side / 2.0, sy0), sy1 - side)
    return [round(x0, 3), round(y0, 3), round(x0 + side, 3), round(y0 + side, 3)]


def run_names(run) -> dict:
    """A run's cells and genes by name: {"cells": {Xenium id: its label}, "genes": {gene name: its gene indices}}
    (cells.txt: line k names label k + 1; genes.txt: line g names gene g)."""
    ncells = int((run.info or {}).get("cells") or 0)
    cells_path, genes_path = run.live / "cells.txt", run.live / "genes.txt"
    cells, genes = {}, {}
    for k, name in enumerate(cells_path.read_text().split("\n")[:ncells] if cells_path.exists() else []):
        cells.setdefault(name, k + 1)
    for g, name in enumerate(genes_path.read_text().split("\n")[:-1] if genes_path.exists() else []):
        genes.setdefault(name, []).append(g)
    return {"cells": cells, "genes": {name: np.asarray(found, np.int64) for name, found in genes.items()}}


def frame_labels(run, frame: dict, path):
    """A run's frame as the where engine reads it (engine/deeper_where.py: a dense frame's per-layer labels, basis
    "voxel"; a tiled frame's 2-D projection, basis "column"), with its consensus labels over a window of columns
    (c0, r0, c1, r1) -> [rows, columns]: the outlines the stage draws. A dense frame is decoded once, whole."""
    info = run.info or {}
    x0, y0, voxel = float(frame["x0"]), float(frame["y0"]), float(frame["voxel"])
    if frame.get("format") == "tiles":
        container = Container(path)
        stats = DeeperWhere._cellstats(run, frame, int(info.get("cells") or 0))
        return (Plane("column", container, "consensus", x0, y0, voxel, stats),
                lambda c0, r0, c1, r1: container.compose("consensus", c0, r0, c1, r1))
    nx, ny, nz = int(frame["nx"]), int(frame["ny"]), int(frame["nz"])
    values, plane = _decode(path), nx * ny
    if values.size < (2 + nz) * plane:
        raise RunError("The frame's file is incomplete", 500)
    consensus, z = values[:plane].reshape(ny, nx), info.get("z") or {}
    labels = Dense("voxel", values[2 * plane:(2 + nz) * plane].reshape(nz, ny, nx), x0, y0, voxel,
                   float(z.get("grid_origin", 0.0)), float(z.get("grid_spacing", 1.0)))
    return labels, lambda c0, r0, c1, r1: consensus[r0:r1, c0:c1]


def newest_counted(run, looked: int = 8) -> dict | None:
    """A run's newest frame with proseg's own count (its record carries "counts_file") whose file is still there (a
    request frame's may have been let go from its spool), among its `looked` newest counted ones; None without one."""
    with run.lock:
        counted = []
        for frame in reversed(run.frames):
            if frame.get("counts_file"):
                counted.append(dict(frame))
                if len(counted) >= looked:
                    break
    for frame in counted:
        if (run.live / frame["counts_file"]).exists():
            return frame
    return None


def view_box(crop, suspects, area_box) -> list[float]:
    """The box a view's runs are asked to export (µm): the crop and every suspect with a place (1 µm around), inside
    the test's area."""
    x0, y0, x1, y1 = (float(v) for v in crop)
    for s in suspects:
        if isinstance(s, dict) and qr.number(s.get("x")) is not None and qr.number(s.get("y")) is not None:
            x0, y0 = min(x0, float(s["x"]) - 1.0), min(y0, float(s["y"]) - 1.0)
            x1, y1 = max(x1, float(s["x"]) + 1.0), max(y1, float(s["y"]) + 1.0)
    ax0, ay0, ax1, ay1 = (float(v) for v in area_box)
    return [round(max(x0, ax0), 3), round(max(y0, ay0), 3), round(min(x1, ax1), 3), round(min(y1, ay1), 3)]


def _layers_of(labels, run, frame: dict):
    """The per-layer labels approx reads a frame's shares from, as engine/held.py weights takes them: (labels [nz, h,
    w], c0, r0); a dense frame's whole; a tiled frame's window (None without one: no share is known there)."""
    if isinstance(labels, Dense):
        return labels.layers, 0, 0
    found = Container(run.live / frame["file"]).window()
    if found is None:
        return None
    info, layers = found
    return layers, int(info["i0"]), int(info["j0"])


def match(records, genes: dict, suspects: list[dict]) -> np.ndarray:
    """Each suspect's own transcript among a run's `records` (its index there; -1 for none, UNSURE when it cannot be
    told): the records of its gene (by name: the run's gene indices for it) within MATCH_UM of its x and of its y. One:
    that one. As many as the suspect's `of` (the where reply's points of its gene at its x, y): sorted by depth as the
    reply gives it (z to 0.001 µm, ties in the run's record order, the reply's own), its `rank`-th, the one the page
    ranked so. Any other number: unsure (two transcripts at one x, y in two layers can be two
    cells'; the nearest in the plane was either). The where reply gives x and y to 0.01 µm, and a run's records are the
    sample's own, so the same transcripts are found in every run."""
    found = np.full(len(suspects), -1, np.int64)
    rx, ry, rz = (records[axis].astype(np.float64) for axis in ("x", "y", "z"))
    by_gene: dict[str, list[int]] = {}
    for s, suspect in enumerate(suspects):
        by_gene.setdefault(suspect["gene"], []).append(s)
    for name, which in by_gene.items():
        ours = np.flatnonzero(np.isin(records["gene"], genes[name])) if name in genes else np.zeros(0, np.int64)
        if not ours.size:
            continue
        ours = ours[np.argsort(rx[ours], kind="stable")]
        sx = np.array([suspects[s]["x"] for s in which])
        lows = np.searchsorted(rx[ours], sx - MATCH_UM - 1e-9, "left")
        highs = np.searchsorted(rx[ours], sx + MATCH_UM + 1e-9, "right")
        for s, low, high in zip(which, lows, highs):
            suspect = suspects[s]
            close = ours[low:high][np.abs(ry[ours[low:high]] - suspect["y"]) <= MATCH_UM + 1e-9]
            if close.size == 1:
                found[s] = close[0]
            elif close.size and close.size == suspect.get("of"):
                close = np.sort(close)                  # the run's record order, in which the reply lists its points
                found[s] = close[np.argsort(np.round(rz[close], 3), kind="stable")][suspect["rank"]]
            elif close.size:
                found[s] = UNSURE
    return found


class QcTests:
    """See the module docstring. `runs`: Runs; `qc`: QcManager (describe, configure, models.ensure: a waiting test's
    own steps); `analysis`: the scheduler (configure); `where`: Go deeper's DeeperWhere (the cell's footprint for the
    corner view; one of its own without); `pools`: the pools (engine/qc_pool.py QcPools; one of its own without), read
    when used (tests replace it). Thread-safe: every test's fields change under `lock`, and nothing is called on the
    others while it is held (but a test's run, read under its own lock: `Test.take`); a test's views under its own
    `view_lock`, never taken with `lock` held (`lock` may be taken inside it); `gate` (one POST at a time, and one test
    leaving its wait at a time) is taken before `lock`, never inside it."""

    def __init__(self, runs, qc, analysis, *, clock=time.monotonic, where=None, pools=None):
        self.runs, self.qc, self.analysis, self.clock = runs, qc, analysis, clock
        self.where = where if where is not None else DeeperWhere(runs)
        self.pools = pools if pools is not None else qc_pool.QcPools(runs, qc, analysis, self.where)
        self.inputs = Path(runs.root).parent / INPUTS   # each test's own input folder, when it has one
        self.lock = threading.Lock()
        self.gate = threading.Lock()                  # one POST at a time (its checks, then the test is in place)
        self.tests: dict[str, Test] = {}              # the parent run's id -> its test (on its way, or as it ended)
        self.closed = False
        if self.inputs.is_dir():                      # an engine that stopped mid-test left its input
            for folder in self.inputs.iterdir():
                if folder.is_dir():
                    leftovers.remove_later(folder, held=lambda folders: runs.held(folders))

    # ---- the API -----------------------------------------------------------------------------------------------------
    def preview(self, run_id: str, cell: int, wait: float = 0.0) -> dict:
        """GET /runs/{id}/qc/test/pool?cell=&wait=: the cell's pool and its reasons by kind (engine/qc_pool.py
        QcPools.preview), with "busy" {why: "test"} among its refusals while a test of this run is waiting or on its way
        (offer false then), and other_run among its waits while another run samples or another test goes; and `hold`,
        the run's hold at the frame a test reads (`hold_tested`: the value a test of the hold is against, which the
        page's form checks a value with, as the POST does)."""
        reply = dict(self.pools.preview(run_id, cell, wait=wait))
        run = self.runs.get(run_id)
        why = self._busy(run)
        if why == "test":
            reply["reasons"] = _in_order([*(reply.get("reasons") or []), qc_pool.reason("busy", why="test")])
            reply["offer"] = False
        elif why == "run":
            reply["waits"] = _in_order([*(reply.get("waits") or []), qc_pool.reason("other_run")])
        reply["hold"] = hold_tested(run)
        box = (reply.get("area") or {}).get("box") if isinstance(reply.get("area"), dict) else None
        reply["settings"] = settings_tested(run, box, frame_tested(run))
        if qr.number(reply.get("estimate_s")) is not None:        # a run of many depth layers is slower to re-run
            reply["estimate_s"] = round(reply["estimate_s"] * layer_factor(startup_value(run, "voxel-layers")))
        return reply

    def start(self, run_id: str, body) -> dict:
        """POST /runs/{id}/qc/test {cell (label - 1), flag, value}: the test (202). In order: 422 for the body; 503
        while the app closes; 404 for an unknown run; 409 while a test of this run is waiting or on its way; the pool
        (engine/qc_pool.py: 409 {detail, status, reasons} with the refusals that hold, 404 for an unknown cell; a wait
        that holds, or a pool still being worked out after PRESS_WAIT_S, lets the press through); 422 for the value the
        test is against (`_current`: the hold's at the frame tested, the pool's F, also while the pool is still being
        worked out or waits: `frame_tested`, as the pool reply's `hold` says), or far drift while drift is off. Then,
        one POST at a time, the run's refusals again, and: "waiting" while a wait holds (`_waits`: the run's, the
        check's, F's scoring, other_run), the pool is still being worked out or its frame is not the run's end (taken
        once nothing holds), after the app's own steps (`_meet`); else "preparing" (a parent sampling on past its end
        paused: `_begin`)."""
        cell, flag, value = self._body(body)
        if self.closed:
            raise QcError("The app is closing", 503)
        run = self.runs.get(run_id)
        self._refuse_busy(run)
        try:
            pool = self.pools.pool(run_id, cell, timeout=PRESS_WAIT_S, check=self._open)
        except (qc_pool.NotYet, qc_pool.StillWorking):  # the cell's own refusals are known once nothing holds and its
            pool = None                                 # pool is there (the test's loop takes it: `_taken`)
        current = self._current(run, flag, frame_tested(run) if pool is None else pool.seq)
        self._refuse_value(run, flag, value, current, None if pool is None else pool.area.box)
        with self.gate:
            self._open()
            self._refuse_busy(run)
            refusals, waits = self.pools.prerequisites(run_id)
            if refusals:
                raise qc_pool.OutsideDomain(refusals)
            waits = self._waits(run, waits)
            done = run.meta.get("schedule_done") if isinstance(run.meta.get("schedule_done"), dict) else {}
            if pool is not None and (waits or pool.seq != done.get("seq")):
                pool = None
            if pool is None:
                test = self._pressed(run, cell, flag, value, current, waits)
                self._meet(test, run, waits)
                refusals, waits = self.pools.prerequisites(run_id)
                if not refusals:                      # (else its loop says them at once)
                    test.waiting = self._waits(run, waits)
            else:
                test = self._new(run, pool, flag, value, current)
                self._begin(test, run)
            with self.lock:
                self.tests[run_id] = test
            test.thread = threading.Thread(target=self._drive, args=(test,), name=f"qc-test-{run_id}", daemon=True)
            test.thread.start()
        self._push(test, force=True)
        return self.public(test)

    def get(self, run_id: str) -> dict:
        """GET /runs/{id}/qc/test: the run's test (on its way, or as it ended); 404 when it has none."""
        self.runs.get(run_id)
        with self.lock:
            test = self.tests.get(run_id)
        if test is None:
            raise QcError("No test for this run", 404)
        return self.public(test)

    def cancel(self, run_id: str) -> dict:
        """DELETE /runs/{id}/qc/test: a test on its way is cancelled (its runs are aborted all at once and deleted:
        waited for up to JOIN_S) and stays as "cancelled"; one that ended is dismissed (GET and the run detail no longer
        carry it). A waiting one cancelled pauses again a parent its press resumed that still samples before its end.
        Returns the test as it ended; 404 when the run has none."""
        self.runs.get(run_id)
        with self.lock:
            test = self.tests.get(run_id)
            if test is None:
                raise QcError("No test for this run", 404)
            waited = test.state == "waiting"
            if test.state in ON_ITS_WAY:
                self._end(test, "cancelled")
            else:
                del self.tests[run_id]
        test.cancel.set()
        if test.thread is not None and test.thread is not threading.current_thread():
            test.thread.join(JOIN_S)
        if waited:
            self._pause_back(test)
        self._push(test, force=True)
        return self.public(test)

    def forget(self, run_id: str):
        """A cancelled test's parent left; its view and status can go too."""
        with self.lock:
            self.tests.pop(run_id, None)

    def going(self) -> bool:
        """Whether any test's runs are on their way (preparing, running or comparing: never a waiting one): its runs are
        the only ones that may sample meanwhile."""
        with self.lock:
            return any(t.state in GOING for t in self.tests.values())

    def brief(self, run_id: str) -> dict | None:
        """The run detail's "qc_test": {state, progress} while the run has a test, else None."""
        with self.lock:
            test = self.tests.get(run_id)
            return None if test is None else {"state": test.state, "progress": _progress(test)}

    def detail(self, public: dict) -> dict:
        """A run as GET /runs/{id} and the WebSocket's "run" pushes give it: with "qc_test" while it has a test."""
        brief = self.brief(public.get("id"))
        return public if brief is None else {**public, "qc_test": brief}

    def public(self, test: Test) -> dict:
        """The test as the API gives it (the {test} record): while it waits, its area, square and pool null and its
        runs' ids too."""
        with self.lock:
            pool = test.matched.pool if test.matched is not None else test.pool
            counts = test.matched.counts if test.matched is not None else None
            return {"id": test.id, "state": test.state, "cell": test.cell, "cell_name": test.cell_name,
                    "flag": test.flag, "name": test.name, "value": test.value, "current": test.current,
                    "tried": test.tried, "start": test.start, "area": _area(test.area), "square": test.square,
                    "runs_per_side": test.runs_per_side, "threads": THREADS, "runs": dict(test.runs),
                    "progress": _progress(test),
                    "pool": None if pool is None else {
                        "cells": pool.members, "suspects": pool.suspects, "k": pool.k,
                        "matched": None if counts is None else counts["matched"]["ok"],
                        "uniform": None if counts is None else counts["uniform"]["ok"],
                        "unsure": None if counts is None else counts["target"]["unsure"],
                        "not_found": None if counts is None else counts["target"]["not_found"]},
                    "message": test.message, "result": test.result, "was_running": test.was_running,
                    "seconds": {k: round(v - test.began, 1) for k, v in test.marks.items()},
                    "hold": None if test.hold is None else dict(test.hold), "no_factorization": test.no_factorization,
                    "reruns": test.reruns, "estimate_s": test.estimate_s,
                    "waiting": [dict(w) for w in test.waiting], "reasons": [dict(r) for r in test.reasons],
                    "caveats": [] if pool is None else [dict(c) for c in pool.caveats],
                    "row": None if test.row is None else {"start": test.row[0], "situation": test.row[1]}}

    def view(self, run_id: str) -> dict:
        """GET /runs/{id}/qc/test/view: the test's corner view: {state, box (the crop, µm; null until worked out),
        suspects (the cell's, as the pool keeps them), runs: [`_run_view` of the newest counted frame of runs A1 and B1,
        once each has one]}. While the runs sample, each is asked (never waiting) for an export of the view's box (as it
        lies on its grid) with its next counted frame (Runs.request_where; one at a time: none while one is on its
        way), a run with a new counted frame is looked at again (one decode), and its spool keeps its newest SPOOL_KEPT
        counted files;
        otherwise, and once the runs are gone, the views kept are given (the last frames', final: a poll begun before
        never replaces them, `_view`). 404 when the run has no test."""
        self.runs.get(run_id)
        with self.lock:
            test = self.tests.get(run_id)
            if test is None:
                raise QcError("No test for this run", 404)
            sampling, ids, box = test.state in ("preparing", "running"), dict(test.runs), test.where_box
        for role in VIEW_ROLES if sampling else ():
            if ids.get(role) is None:
                continue
            try:
                run = self.runs.get(ids[role])
            except RunError:                            # deleted meanwhile: the view kept stays
                continue
            frame = newest_counted(run)
            asked = None if box is None or frame is None else qc_pool.on_grid(box, frame)
            if asked is not None:
                try:
                    self.runs.request_where(ids[role], list(asked), wait_s=0)
                except RunError:                        # ended or gone meanwhile: its counted frames are what there is
                    pass
            if frame is not None:
                self._view(test, role, run, frame)
                self._prune(ids[role])
        with self.lock:
            state, crop = test.state, test.crop
        with test.view_lock:
            runs = [test.views[role]["view"] for role in VIEW_ROLES if role in test.views]
        return {"state": state, "box": crop, "suspects": list(test.suspects), "runs": runs}

    def close(self):
        """The engine closes: every test on its way is cancelled and its runs are cleared away first (before Runs.close
        could save them as results)."""
        with self.gate:
            self.closed = True
        with self.lock:
            tests = list(self.tests.values())
            for test in tests:
                if test.state in ON_ITS_WAY:
                    self._end(test, "cancelled", "The app closed")
        for test in tests:
            test.cancel.set()
        for test in tests:
            if test.thread is not None:
                test.thread.join(JOIN_S)

    def finished_runs(self, test: Test) -> dict[str, Path]:
        """{role: its run's folder} of the test's runs (between their success and `_clear`, which deletes them)."""
        with self.lock:
            ids = {role: run_id for role, run_id in test.runs.items() if run_id is not None}
        return {role: self.runs.get(run_id).dir for role, run_id in ids.items()}

    # ---- checks ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _body(body) -> tuple[int, str, float]:
        """(cell, flag, value) of the POST's body; QcError 422 in the engine's words for anything else (`suspects` and
        `sources` are ignored: the pool's are the engine's own)."""
        from . import proseg_options
        if not isinstance(body, dict):
            raise QcError("A test is {cell, flag, value}")
        cell = body.get("cell")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise QcError("cell must be a cell number (label - 1)")
        if "changes" in body:
            raise QcError("A test changes one setting: send flag and value")
        flag, value = body.get("flag"), body.get("value")
        if flag not in FLAGS:
            raise QcError("A test changes one of the settings the test form offers")
        if qr.number(value) is None:
            raise QcError("value must be a number")
        option, kind = next(o for o in proseg_options.OPTIONS if o["flag"] == flag), KINDS[flag]
        if kind == "switch":
            if value not in (0, 1):
                raise QcError("Choose 1 (drift off) or 0 (drift on)")
            return cell, flag, int(value)
        if kind == "whole":
            if not float(value).is_integer() or not option["min"] <= value <= option["max"]:
                raise QcError(f"Choose a whole number from {option['min']} to {option['max']}")
            return cell, flag, int(value)
        if kind == "choice":            # the run's own choices are checked with the run (`_refuse_value`)
            if value < option["min"]:
                raise QcError("Choose a starting grid in µm")
            return cell, flag, value
        if flag in TEST_RANGES:
            low, high = TEST_RANGES[flag]
            if not low <= value <= high:
                raise QcError(f"Choose a value from {low:g} to {high:g}")
        else:
            try:
                proseg_options.live_lines({flag: value})
            except ValueError:
                raise QcError(f"Choose a value from {option['min']:g} to {option['max']:g}") from None
        return cell, flag, value

    def _open(self):
        """QcError 503 once the app closes (a pool still being worked out stops waiting)."""
        if self.closed:
            raise QcError("The app is closing", 503)

    def _busy(self, run) -> str | None:
        """"test" while a test of this run is waiting or on its way, "run" while another run samples or another test
        goes (`_others`), else None."""
        with self.lock:
            old = self.tests.get(run.meta["id"])
            if old is not None and (old.state in ON_ITS_WAY or (old.thread is not None and old.thread.is_alive())):
                return "test"
        return "run" if self._others(run) else None

    def _others(self, run) -> bool:
        """Whether another run samples (an active run, `run` aside, that is not paused: another test's runs among them)
        or another run's test is going (preparing, running or comparing: its runs may not be made yet)."""
        run_id = run.meta["id"]
        with self.lock:
            if any(test.state in GOING for parent, test in self.tests.items() if parent != run_id):
                return True
        return any(r is not run and r.meta.get("state") != "paused" for r in self.runs.active())

    def _refuse_busy(self, run):
        """409 while a test of this run is waiting or on its way (one test at a time per run)."""
        if self._busy(run) == "test":
            raise QcError("A test of this run is on its way: wait for its answer, or cancel it", 409)

    def _waits(self, run, waits) -> list[dict]:
        """A test's waits now: `waits` (engine/qc_pool.py's: the run's, the check's, F's scoring) and other_run while
        another run samples or another test goes (`_others`: its runs sample alone), in the page's order."""
        found = list(waits)
        if self._others(run):
            found.append(qc_pool.reason("other_run"))
        return _in_order(found)

    @staticmethod
    def _current(run, flag: str, seq=None):
        """The value a test of `flag` is against (`current_at`): the run's value now, the hold's at frame `seq` (F: the
        pool's, else the end of the run's schedule; its value now while F is not known)."""
        return current_at(run, flag, seq)

    @staticmethod
    def _refuse_value(run, flag: str, value, current, box=None):
        """422 for the value the test is against (`current`: its runs as now would be the ones with the value), in words
        that say whether the run uses it now or used it at its last iteration (a hold set live since F); for far drift
        and the short drift distance while drift is switched off in the run; for depth layers on a sample without depth;
        for a starting grid that is not one of the run's own choices (`grid_choices`); and for a value the B runs could
        not be made with (proseg_options with the run's other settings, or the memory of the area `box`: a test's
        runs are all made, or none)."""
        from . import proseg_options, runs as runs_module
        if qr.same(value, current):
            if KINDS.get(flag) == "switch":
                state, other = ("off", "on") if value == 1 else ("on", "off")
                raise QcError(f"Drift is already {state} in this run: choose {other} to test it", 422)
            if not qr.same(current, current_of(run, flag)):
                raise QcError(f"{value:g} is what this run used at its last iteration: choose another value to test",
                              422)
            raise QcError(f"{value:g} is what this run uses now: choose another value to test", 422)
        if flag in DRIFTING and _asked(run.meta).get("no-diffusion") is True:
            raise QcError(f"{NAMES[flag].capitalize()} does nothing while drift is switched off in this run", 422)
        if flag == "voxel-layers" and flat(run):
            raise QcError(FLAT_WORDS, 422)
        if flag == "burnin-voxel-size":
            choices = grid_choices(run)
            if not any(qr.same(value, c) for c in choices):
                if not choices:
                    raise QcError("No other starting grid can be tested for this run", 422)
                said = f"{choices[0]:g}" if len(choices) == 1 else \
                    ", ".join(f"{c:g}" for c in choices[:-1]) + f" or {choices[-1]:g}"
                raise QcError(f"Choose {said} µm for this run's starting grid", 422)
        values = {**_asked(run.meta), "nthreads": THREADS, flag: proseg_value(flag, value)}
        try:
            proseg_options.to_args(values)
        except ValueError:
            raise QcError("This value can't be combined with your run's other settings", 422) from None
        w, h = (box[2] - box[0], box[3] - box[1]) if box else (SQUARE_UM, SQUARE_UM)
        columns, layers, _, _ = runs_module.budget_columns(values, w, h)
        if columns > runs_module.MAX_COLUMNS or columns * layers > runs_module.MAX_VOXELS:
            raise QcError("This value needs more memory than a test run may take here", 422)

    def _new(self, run, pool: qc_pool.Pool, flag: str, value, current) -> Test:
        """The test of `run` (its parent) over `pool`, taken at once (`Test.take`): on its way ("preparing")."""
        test = self._pressed(run, pool.seed, flag, value, current, ())
        test.take(pool, run)
        test.state, test.progress["phase"] = "preparing", "preparing"
        return test

    def _pressed(self, run, cell: int, flag: str, value, current, waits) -> Test:
        """The test of `run` (its parent) as pressed, "waiting" (`waits`): its cell (its Xenium id from the run's
        cells.txt), its start, row and roles, the parent's settings and schedule's marks as they are now, and its hold
        at the frame tested as `current` reads it (`frame_tested`: F's once the run reached its end, else its value
        now); its area and pool to come (`take` reads them all at the pool's F)."""
        meta, info = run.meta, run.info
        options = meta.get("options") or {}
        row, schedule, asked = qc_pool.row_of(meta), _schedule(info, options), _asked(meta)
        test = Test(meta["id"], None, flag, value, current, self.clock,
                    runs_per_side=int(qc_rule.constants_for(*row)["runs_per_side"]),
                    of=schedule_total(info, options),
                    iter_marks=(schedule["burnin"], schedule["burnin"] + schedule["samples"]),
                    hold=hold_at(run, frame_tested(run)),   # F's, as `current` (its value now before F is known)
                    no_factorization=no_factorization_of(meta, info),
                    parent_defaults=parent_defaults(asked, temperature_of(meta)), asked=asked, cell=int(cell),
                    cell_name=self._cell_name(run, cell), start=qr.start_of(meta), row=row, waiting=waits)
        test.set_factor(run)
        return test

    @staticmethod
    def _cell_name(run, cell: int) -> str | None:
        """The cell's Xenium id: line `cell` of the run's cells.txt (None where it cannot be read)."""
        try:
            names = (run.live / "cells.txt").read_text().split("\n")
        except OSError:
            return None
        return names[cell] if 0 <= cell < len(names) and names[cell] else None

    # ---- a waiting test --------------------------------------------------------------------------------------------
    def _meet(self, test: Test, run, waits):
        """The app's own steps for the waits that hold, each once per test (`test.met`): a parent paused before its end
        resumed (not while another test goes: then once none does; `resumed`, so that Cancel, or the test's end before
        its runs start, pauses it again: `_pause_back`); the
        mixing check switched on (its own switch: QcManager.configure, remembered for new runs); a sample whose
        learning failed asked again (QcModels.ensure with retry: at the press, or when it first fails), a failure after
        that ending the test (_Failed CHECK_FAILED); Go deeper's details asked for (ensure with retry and detail), their
        failure after that ending it (_Failed DETAILS_FAILED). A step refused is said in the log; the test waits on."""
        codes = {w["code"] for w in waits}
        meta = run.meta
        if "not_at_end" in codes and "resume" not in test.met and meta.get("paused") \
                and meta.get("state") in ("preparing", "paused") and not qc_pool.reached_end(meta) and not self.going():
            test.met.add("resume")
            try:
                self.runs.control(test.parent, "resume")
                test.resumed = True
            except RunError as exc:
                print(f"SoPA companion: the test of {test.parent} could not resume its run: {exc}", flush=True)
        if self.qc is None or not codes & {"check_off", "check_wait", "details"}:
            return
        if "check_off" in codes and "check_off" not in test.met:
            test.met.add("check_off")
            self._ask(test, "switch the mixing check on", self.qc.configure, test.parent, enabled=True)
        try:
            described = self.qc.describe(test.parent)
        except Exception:                           # the check says why on its own panel: the test waits on
            return
        sample = meta.get("sample")
        if "check_wait" in codes and described.get("sample_state") == "failed":
            if "check_retry" in test.met:
                raise _Failed(CHECK_FAILED)
            test.met.add("check_retry")
            self._ask(test, "learn the sample again", self.qc.models.ensure, sample, retry=True)
        if "details" in codes:
            if "details" not in test.met:
                test.met.add("details")
                self._ask(test, "learn Go deeper's details", self.qc.models.ensure, sample, retry=True, detail=True)
            elif (described.get("detail") or {}).get("state") == "failed":
                raise _Failed(DETAILS_FAILED)

    @staticmethod
    def _ask(test: Test, what: str, call, *args, **kwargs):
        """One of the app's own steps, its refusal said in the log."""
        try:
            call(*args, **kwargs)
        except Exception as exc:
            print(f"SoPA companion: the test of {test.parent} could not {what}: {type(exc).__name__}: {exc}",
                  flush=True)

    def _wait(self, test: Test):
        """A waiting test's loop, every WAIT_S in its own thread, until it is on its way (`_go`): its run looked at
        (`_check`); engine/qc_pool.py's prerequisites asked (a refusal ends the test "failed", with its `reasons`); the
        app's own steps taken (`_meet`); its `waiting` (with other_run) kept, the run detail pushed when it changes;
        once nothing holds, its pool taken (`_go`). It has no time limit: a run left paused before its end waits until
        Cancel."""
        while True:
            self._check(test)
            try:
                run = self.runs.get(test.parent)
                refusals, waits = self.pools.prerequisites(test.parent)
            except RunError:                            # deleted meanwhile: stopped in its own words
                self._check(test)
                raise
            if refusals:
                raise _Refused(refusals)
            self._meet(test, run, waits)
            waits = self._waits(run, waits)
            with self.lock:
                changed = test.waiting != waits
                test.waiting = [dict(w) for w in waits]
            if changed:
                self._push(test, force=True)
            if not waits and self._go(test, run):
                return
            if test.cancel.wait(WAIT_S):
                raise _Stop(test.message)

    def _go(self, test: Test, run) -> bool:
        """Once nothing holds: the cell's pool (`_taken`), then, one at a time under the POST's gate, nothing holding
        again (no refusal, no wait, no other run sampling, no other test going) and the pool at the run's end: the pool
        taken (`Test.take`: the value it is against fails it, in the POST's words: the hold's at the pool's frame) and
        the test on its way ("preparing", `waited` in its seconds; a parent sampling on past its end paused: `_begin`).
        False to wait on."""
        pool = self._taken(test)
        if pool is None:
            return False
        with self.gate:
            self._open()
            refusals, waits = self.pools.prerequisites(test.parent)
            if refusals:
                raise _Refused(refusals)
            done = run.meta.get("schedule_done") if isinstance(run.meta.get("schedule_done"), dict) else {}
            if self._waits(run, waits) or pool.seq != done.get("seq"):
                return False
            try:
                self._refuse_value(run, test.flag, test.value, self._current(run, test.flag, pool.seq), pool.area.box)
            except QcError as exc:
                raise _Failed(str(exc)) from None
            with self.lock:
                if test.state != "waiting":
                    raise _Stop(test.message)
                test.take(pool, run)
                test.state, test.waiting, test.progress["phase"] = "preparing", [], "preparing"
                test.marks["waited"] = self.clock()
            self._begin(test, run)
        self._push(test, force=True)
        return True

    def _taken(self, test: Test) -> qc_pool.Pool | None:
        """The cell's pool for a test whose waits are over (pools.pool, looking at the test meanwhile): None while a
        wait came back (NotYet) or the pool is still being worked out past pool()'s wait (StillWorking: the run's
        worker goes on with the same request, asked again on the next round, however long a whole sample's takes); its
        refusals (the cell's, the pool's) end the test (_Refused)."""
        try:
            return self.pools.pool(test.parent, test.cell, check=lambda: self._check(test))
        except (qc_pool.NotYet, qc_pool.StillWorking):
            return None
        except qc_pool.OutsideDomain as refused:
            raise _Refused(refused.reasons) from None

    def _begin(self, test: Test, run):
        """As the runs start: a parent still sampling on past its end is paused (`was_running`: the page's Keep, Cancel
        and Close resume it); else `was_running` stays false."""
        meta = run.meta
        if meta.get("state") == "running" and qc_pool.reached_end(meta):
            try:
                self.runs.control(test.parent, "pause")
            except RunError as exc:
                print(f"SoPA companion: the test of {test.parent} could not pause its run: {exc}", flush=True)
                return
            with self.lock:
                test.was_running = True

    def _pause_back(self, test: Test):
        """A waiting test cancelled, or ended before its runs started (failed, refused): a parent its press resumed
        (`resumed`) that still samples before its end is paused again (the run as it was before the press)."""
        if not test.resumed:
            return
        try:
            run = self.runs.get(test.parent)
            meta = run.meta
            if not meta.get("paused") and meta.get("state") in ("preparing", "running") \
                    and not qc_pool.reached_end(meta):
                self.runs.control(test.parent, "pause")
        except RunError:                              # gone or ended meanwhile: nothing to pause
            pass

    # ---- the test's thread ---------------------------------------------------------------------------------------------
    def _drive(self, test: Test):
        """The test's thread: its wait (`_wait`, for a waiting test), input, matching, runs, crop, hold, following, then
        (comparing) the readout, the answer and the final views; finally its runs and input cleared away before its end
        is shown (a page that sees "done" finds them gone), and a run its press resumed paused again when it ended
        still waiting (`_pause_back`: the page's Close says the run goes on as it was before the test)."""
        state, values = "failed", {}
        try:
            if test.state == "waiting":
                self._wait(test)
            self._input(test)
            self._match(test)
            self._make(test)
            self._crop(test)                            # while the runs load
            self._hold_runs(test)
            self._follow(test)
            with self.lock:
                if test.state in ON_ITS_WAY:
                    test.progress["phase"] = "reading"
            self._set(test, state="comparing")
            state, values = "done", {"result": self._answer(test)}
        except _Stop as stop:
            state, values = "cancelled", {"message": stop.message}
        except _Refused as refused:
            values = {"message": refused.reasons[0]["words"], "reasons": refused.reasons}
        except _Failed as failed:
            values = {"message": failed.message}
        except Exception as exc:
            values = {"message": f"The test could not finish: {exc}"}
            print(f"SoPA companion: the test of {test.parent} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            try:
                self._clear(test)
            except Exception as exc:                    # never left "on its way" for want of a cleared run
                print(f"SoPA companion: the test runs of {test.parent} were not cleared: {exc}", flush=True)
            if test.resumed and "waited" not in test.marks and not self.closed:     # it ended still waiting
                try:
                    self._pause_back(test)
                except Exception as exc:                # its end is shown all the same
                    print(f"SoPA companion: the test of {test.parent} could not pause its run again: {exc}", flush=True)
            self._set(test, state=state, **values)

    def _check(self, test: Test):
        """Raise _Stop when the test was cancelled, the engine closes, or its run (the parent) was deleted or is being
        finished, saved or left (engine/qc_pool.py being_saved); _Refused ([ended]) when it ended before its end (no end
        frame will ever come). A parent that merely ended at or past its end keeps its test: its input, frames and
        scores stay until it is left (server/app.py cancels the test first)."""
        if test.cancel.is_set():
            raise _Stop(test.message)
        try:
            parent = self.runs.get(test.parent)
        except RunError:
            raise _Stop("Your run was deleted") from None
        meta = parent.meta
        if qc_pool.being_saved(meta):
            raise _Stop("Your run is being saved, so the test stopped")
        if meta.get("state") in TERMINAL and not qc_pool.reached_end(meta):
            raise _Refused([qc_pool.reason("ended")])

    def _input(self, test: Test):
        """The test's shared input (`shared_input`), from the parent's own: in self.inputs/<test id> when the test
        writes one."""
        self._check(test)
        parent = self.runs.get(test.parent)
        found = shared_input(parent.dir / "input" / "transcripts.parquet", test.area, self.inputs / test.id)
        with self.lock:
            test.input = found
            test.marks["input"] = self.clock()

    def _match(self, test: Test):
        """The pool's targets and controls found in the shared input (pools.match_input, with the settings every test
        run takes: the parent's input rule among them), then pooled again on what was found: `test.matched` (under k, a
        caveat). None of the clicked cell's suspects found ends the test there, before any run is made ("failed", in
        its own words)."""
        self._check(test)
        try:
            matched = self.pools.match_input(test.pool, Path(test.input["path"]), dict(test.asked),
                                             check=lambda: self._check(test))
        except qc_pool.TooFewFound as few:
            raise _Failed(str(few)) from None
        with self.lock:
            test.matched = matched
            test.marks["matched"] = self.clock()

    def _make(self, test: Test):
        """The 2N runs from the shared input, all at once, made in the order A1, B1, A2, B2, ...: the area
        (`Area.spec`), the parent's starting cells and settings (`test.asked`) with THREADS threads, without
        factorisation when the test has to, B with the setting from its start; paused (each is given its hold before
        its first iteration, then resumed: `_hold_runs`), at full speed; Clusters & UMAP off; their meta names the
        test, its role and side, and the input they share. Runs.create counts them against [proseg] max_active_runs
        with 2N more places (_WithRoom: the parent waits paused among them)."""
        parent = self.runs.get(test.parent)
        meta = parent.meta
        if meta.get("sample_kind") == "zarr" and not self.runs.zarr.known(meta["sample"]):
            # A sample forgotten since is opened again (the open dialog's list stays): Runs.create would otherwise take
            # the parent's sample for a Xenium bundle.
            self.runs.zarr.open(meta["sample_path"], recent=False)
        with self.lock:
            shared = {key: test.input[key] for key in ("path", "transcripts", "ids", "owned")}
        shared["prepared"] = {**(meta.get("prepared") or {}), "source": "parent"}
        room = _WithRoom(self.runs, len(test.roles))
        for i in range(1, test.runs_per_side + 1):
            for side in "AB":
                role = f"{side}{i}"
                self._check(test)
                options = {**test.asked, "nthreads": THREADS,
                           **({"no-factorization": True} if test.no_factorization else {}),
                           **({test.flag: proseg_value(test.flag, test.value)} if side == "B" else {})}
                # Tagged from its first moment: no list or push ever carries it without "qc_test" (the page leaves it
                # out), and it reads the shared input as it is (runs.py prepares nothing for it).
                made = type(self.runs).create(room, meta["sample"], region=copy.deepcopy(test.area.spec),
                                              options=options, prior=meta.get("prior"),
                                              extra={"qc_test": {"of": test.parent, "role": role, "side": side,
                                                                 "test": test.id},
                                                     "input": copy.deepcopy(shared), "paused": True, "speed": "max"})
                with self.lock:
                    test.runs[role] = made["id"]
                try:
                    self.analysis.configure(made["id"], enabled=False)
                except Exception:           # Clusters & UMAP unavailable (engine/analysis.py): nothing to switch off
                    pass
        with self.lock:
            test.marks["made"] = self.clock()

    def _hold_runs(self, test: Test):
        """Each run, made paused (`_make`), given its hold before its first iteration, then resumed. Proseg makes the
        hold stronger where cells hold more transcripts (`scale_of`: the input's transcripts per cell over 100), and a
        test's runs read another input than the parent (its square, its crop), so the value the parent runs with would
        hold them more or less firmly than it holds the parent (0.5 is 5.25 in a box of 1,050 transcripts a cell, 2.26
        in a square of 452). Once every run has its info.json (transcripts, cells) and is paused (the progress's phase
        "loading" meanwhile, looked at every FOLLOW_S), each is given, as a live `set prior-weight` (proseg applies it
        before every iteration, the first included), the value whose hold equals the parent's at the frame tested (the
        parent's value times the parent's scale, at the run's own scale: `given_hold`); in a test of the hold, side B
        the tested value at the parent's scale (the hold a run of the parent with that value would have). Only the
        value given to proseg changes: its command line stays as made (info.json's prior_weight is the command line's).
        A value past HOLD_MAX (proseg_options' bound, which Runs.control would refuse) is found as soon as one run has
        read its input (`_refuse_hold`), and refuses the test there in plain words, before any run is given a value or
        resumed; a run that failed or stopped fails it as `_follow` does, and so does no answer by its time limit.
        `test.hold["given"]`: {scale (A1's, 6 significant digits), as_now (A1's value), tried (B1's in a test of the
        hold, else None)}. Without the parent's hold (no counts in its info.json) the runs are resumed as made."""
        limit = time_limit(test.progress["of"]) * test.factor
        deadline = test.marks.get("waited", test.began) + limit
        with self.lock:
            if test.state in ON_ITS_WAY:
                test.progress["phase"] = "loading"
            hold, parent = test.hold, test.scale
        aims = None                                     # the hold each side's runs are given, as proseg applies it
        if hold is not None and parent is not None:
            with np.errstate(over="ignore"):
                as_now = np.float32(hold["value"]) * np.float32(parent)
                tried = np.float32(test.value) * np.float32(parent) if test.flag == "prior-weight" else as_now
            aims = {"A": float(as_now), "B": float(tried)}
        while True:
            seen = self._seen(test)
            if aims is not None:
                self._refuse_hold(test, aims, seen)
            if all(run["state"] == "paused" and run["scale"] is not None for run in seen.values()):
                break
            self._push(test)
            if self.clock() > deadline:
                raise ValueError(f"no answer within {limit / 60:g} minutes")
            if test.cancel.wait(FOLLOW_S):
                raise _Stop(test.message)
        with self.lock:
            ids = dict(test.runs)
        values = {}
        if aims is not None:
            values = {role: given_hold(aims[role[0]], seen[role]["scale"]) for role in test.roles}
            with self.lock:
                test.hold = {**test.hold, "given": {"scale": float(f"{seen['A1']['scale']:.6g}"),
                                                    "as_now": values["A1"],
                                                    "tried": values["B1"] if test.flag == "prior-weight" else None}}
        for role in test.roles:
            self._check(test)
            try:
                if role in values:
                    self.runs.control(ids[role], "set", values={"prior-weight": values[role]})
                self.runs.control(ids[role], "resume")
            except RunError:                            # it ended meanwhile: in its own words
                self._seen(test)
                raise

    @staticmethod
    def _refuse_hold(test: Test, aims: dict, seen: dict):
        """The test refused (`_Failed`, in plain words) once a run has read its input, where its runs would need a hold
        past HOLD_MAX: at that run's scale (every run's: they read one input from one start the same way), side A's
        (`aims["A"]`, the parent's own hold at the frame tested: nothing can be tested here) or side B's (the value
        tested, in a test of the hold: nothing can be tested at it here, another value can). The value is said so that
        it reads above the bound: `g`, else the float32's own shortest decimal (1000.0001, never "1000")."""
        def said(value: float) -> str:
            return f"{value:g}" if float(f"{value:g}") > HOLD_MAX else \
                np.format_float_positional(np.float32(value), trim="-")
        for run in seen.values():
            if run["scale"] is None:
                continue
            need = {side: given_hold(aim, run["scale"]) for side, aim in aims.items()}
            if need["A"] > HOLD_MAX:
                raise _Failed(f"Nothing can be tested here: matching your run's hold here would need "
                              f"{said(need['A'])}, above the most proseg takes ({HOLD_MAX:g})")
            if need["B"] > HOLD_MAX:
                raise _Failed(f"Nothing can be tested at {test.value:g} here: matching it here would need "
                              f"{said(need['B'])}, above the most proseg takes ({HOLD_MAX:g})")

    def _seen(self, test: Test) -> dict[str, dict]:
        """`_look` after `_check`, a run that failed or stopped failing the test, with its role and its words."""
        self._check(test)
        seen = self._look(test)
        for role, run in seen.items():
            if run["state"] in ("failed", "stopped"):
                said = f": {run['message']}" if run["message"] else ""
                raise ValueError(f"its {role} run {run['state']}{said}")
        return seen

    def _look(self, test: Test) -> dict[str, dict]:
        """Each run of the test as it stands, in role order: {role: {state, iter (its newest frame's; None before its
        first), retried (runs.py's own retry without factorisation), message, scale (proseg's hold scale, `scale_of`;
        None before its info.json)}}. A run that went fails the test."""
        with self.lock:
            ids = dict(test.runs)
        seen = {}
        for role in test.roles:
            try:
                run = self.runs.get(ids.get(role))
            except RunError:
                raise ValueError(f"its {role} run was deleted") from None
            with run.lock:
                meta = run.meta
                seen[role] = {"state": meta.get("state"),
                              "iter": run.frames[-1].get("iter", 0) if run.frames else None,
                              "retried": bool(meta.get("retried_without_factorization")),
                              "message": meta.get("message"), "scale": scale_of(run.info)}
        return seen

    def _follow(self, test: Test):
        """Until every run is "done" (proseg wrote its per-transcript file and ended): the progress kept (its phase:
        loading until every run has a frame, then sampling, then writing once every run is past sampling), "running"
        once every run has a frame, the parent's run detail pushed at most every PUSH_S for it. A run that failed or
        stopped (an abort, the disk guard's among them) fails the test, with its role and its words; one that needed
        proseg's retry without factorisation has every run of the test made again without it, once (`_rerun`); no
        answer by its time limit fails it (`time_limit`, times the test's depth layers' `factor`, from the POST or from
        the end of its wait)."""
        limit = time_limit(test.progress["of"]) * test.factor
        deadline = test.marks.get("waited", test.began) + limit
        while True:
            seen = self._seen(test)
            if test.reruns == 0 and any(run["retried"] for run in seen.values()):
                self._rerun(test)
                continue
            of = test.progress["of"]
            reached = {role: of if run["state"] in _PAST_SAMPLING else min(int(run["iter"] or 0), of)
                       for role, run in seen.items()}
            framed = all(run["iter"] is not None for run in seen.values())
            past = all(run["state"] in _PAST_SAMPLING for run in seen.values())
            done = sum(run["state"] == "done" for run in seen.values())
            with self.lock:
                test.progress.update(iter=min(reached.values()), done=done,
                                     sides={side: min(at for role, at in reached.items() if role[0] == side)
                                            for side in "AB"},
                                     phase="writing" if past else "sampling" if framed else "loading")
                if past and "sampled" not in test.marks:
                    test.marks["sampled"] = self.clock()
            if framed and test.state == "preparing":
                self._set(test, state="running")
            self._push(test)
            if done == len(seen):
                return
            if self.clock() > deadline:
                raise ValueError(f"no answer within {limit / 60:g} minutes")
            if test.cancel.wait(FOLLOW_S):
                raise _Stop(test.message)

    def _rerun(self, test: Test):
        """A run needed proseg's retry without factorisation (runs.py: its first start failed): every run of the test
        is made again without it, so both sides stay alike. In this order: every run stopped (one that is still active
        at the shared deadline fails the test: Runs refuses to delete an active run); the views and names of the old
        runs let go (the crop stays: views only move forward, so an old run's would stay); the test marked (no
        factorisation, one rerun more, its progress back to the start, pushed at once); the new runs made while the old
        ones still exist (Runs.create's ids carry their creation second: an old one's id is never taken again); then the
        old ones deleted, and the new ones given their hold and resumed (`_hold_runs`)."""
        with self.lock:
            old = [run_id for run_id in test.runs.values() if run_id is not None]
        if not self._stop_runs(test):
            raise ValueError("a test run did not stop")
        with test.view_lock:
            test.views.clear()
            test.names.clear()
        with self.lock:
            test.runs = dict.fromkeys(test.roles)
            test.no_factorization, test.reruns = True, test.reruns + 1
            test.progress.update(iter=0, done=0, sides={"A": 0, "B": 0}, phase="preparing", retry=True)
            if test.state == "running":                 # the new runs' first frames are still to come
                test.state = "preparing"
            for step in ("made", "running", "sampled"):     # the new runs' own, once reached
                test.marks.pop(step, None)
        self._push(test, force=True)
        try:
            self._make(test)
        finally:
            self._delete_runs(old)
        self._hold_runs(test)

    def _answer(self, test: Test) -> dict:
        """The runs read (engine/qc_readout.py read_runs: each run's own per-transcript file, joined to the targets by
        their ids in the shared input) and judged (engine/qc_rule.py answer, with the test's row and the matched pool's
        caveats): the result, v 2, as it is, its caveats those with too_few for the suspects judged (`answer_caveats`),
        and `hold` the test's (the parent's hold at the frame tested and what its runs were given: `_hold_runs`).
        Then the corner's final views, from each run's last counted frame while the runs are still there (`_view`,
        final: never replaced after), never at the cost of the answer."""
        self._check(test)
        matched = test.matched
        readouts = qc_readout.read_runs(self.finished_runs(test), matched.items, test.inner)
        sides = {side: [role for role in test.roles if role[0] == side] for side in "AB"}
        result = qc_rule.answer(matched.items, matched.order, readouts, sides, test.start,
                                setting={"flag": test.flag, "value": test.value, "current": test.current},
                                situation=test.area.rule, at_iter=min(int(r.iter) for r in readouts.values()),
                                matched_counts=matched.counts, parent_defaults=test.parent_defaults, row=test.row,
                                caveats=answer_caveats(matched.pool.caveats, None, matched.pool.k))
        if isinstance(result, dict):
            result["caveats"] = answer_caveats(result.get("caveats") or (),
                                               (result.get("pool") or {}).get("suspects"), matched.pool.k)
            with self.lock:
                result["hold"] = copy.deepcopy(test.hold)      # the parent's at the frame tested, and what was given
        for role in VIEW_ROLES:
            try:
                run = self.runs.get(test.runs[role])
                frame = newest_counted(run)
                if frame is not None:
                    self._view(test, role, run, frame, final=True)
            except Exception as exc:
                print(f"SoPA companion: the test of {test.parent}: the last view of its {role} run failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        return result

    def _stop_runs(self, test: Test) -> bool:
        """Every active run of the test aborted first, then all of them waited for together, until one deadline STOP_S
        away: 2N runs stop in about the time of the slowest, never one after the other. True once none is active (a run
        deleted meanwhile counts as stopped)."""
        with self.lock:
            ids = [run_id for run_id in test.runs.values() if run_id is not None]
        runs = []
        for run_id in ids:
            try:
                run = self.runs.get(run_id)
            except RunError:
                continue
            runs.append(run)
            if run.meta.get("state") in ACTIVE:
                try:
                    self.runs.control(run_id, "abort")
                except RunError:                        # it ended meanwhile
                    pass
        deadline = self.clock() + STOP_S
        while any(run.meta.get("state") in ACTIVE for run in runs) and self.clock() < deadline:
            time.sleep(0.05)
        return not any(run.meta.get("state") in ACTIVE for run in runs)

    def _delete_runs(self, ids):
        """Test runs deleted, later (each folder goes to the runs folder's trash at once and is emptied in the
        background: 2N runs' frames take seconds to remove on a slow disk). One still active is left in the list
        (aborted: never sampling on) and noted in the log; one gone already is skipped."""
        for run_id in ids:
            try:
                self.runs.delete(run_id, later=True)
            except RunError as exc:
                if exc.status != 404:
                    print(f"SoPA companion: the test run {run_id} could not be deleted: {exc}", flush=True)

    def _clear(self, test: Test):
        """Whatever happened: the test's runs stopped (`_stop_runs`) and deleted (`_delete_runs`), then its own input
        folder removed (never the parent's input, which a test only reads)."""
        self._stop_runs(test)
        with self.lock:
            ids = [run_id for run_id in test.runs.values() if run_id is not None]
        self._delete_runs(ids)
        folder = self.inputs / test.id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            if folder.exists():                         # a file still held: removed once nothing holds it
                leftovers.remove_later(folder, held=lambda folders: self.runs.held(folders))

    # ---- the corner view -----------------------------------------------------------------------------------------------
    def _crop(self, test: Test):
        """The corner view's crop (crop_around): around the cell's footprint in the parent's frame the pool was taken
        at (as the where engine finds it), else around its place there, inside the test's area; and the box its runs
        are asked to export (`view_box`: the crop and the suspects)."""
        footprint = None
        try:
            footprint = self.where.footprints(test.parent, test.seq, [test.cell])[test.cell]
        except Exception as exc:                        # its frame cannot be read now: its place instead
            print(f"SoPA companion: the test of {test.parent}: the cell's outline was not found ({exc}); its view is "
                  "centred on its place", flush=True)
        crop = None if footprint is None and test.place is None else crop_around(footprint, test.place, test.area.box)
        box = None if crop is None else view_box(crop, test.suspects, test.area.box)
        with self.lock:
            test.crop, test.where_box = crop, box

    def _prune(self, run_id: str):
        """The run's spool keeps its newest SPOOL_KEPT counted files (the view reads the newest), never failing a
        view."""
        try:
            spool = self.runs.spool
            files = sorted(spool.files(run_id), key=lambda path: Path(path).name)
            if len(files) > SPOOL_KEPT:
                spool.prune(run_id, set(files[-SPOOL_KEPT:]))
        except (OSError, RunError) as exc:
            print(f"SoPA companion: the test run {run_id}'s counts were not let go: {exc}", flush=True)

    def _view(self, test: Test, role: str, run, frame: dict, final: bool = False) -> dict | None:
        """The run's view of `frame` (`_run_view`), worked out unless it is the one kept, then kept: one decode per run
        and frame, however many ask at once. Only a later frame's view (by iteration, then seq) replaces the one kept,
        and never a final one: the last frames' (`final`, from the answer) are marked so under the lock (a poll that
        took an older frame and came late left the corner there for good, an iteration the answer never counted). Only
        the test's own run of the role is kept (one made again since, `_rerun`, is not: a poll that took the old run's
        frame and came late would leave it there). A run deleted meanwhile (its test ended) leaves the one kept (None:
        none)."""
        key = (run.meta["id"], frame.get("seq"), frame.get("iter"), frame.get("t"))
        order = (int(frame.get("iter") or 0), int(frame.get("seq") or 0))
        with test.view_lock:
            kept = test.views.get(role)
            with self.lock:
                current = test.runs.get(role) == key[0]
            if kept is not None and kept["key"] == key:
                kept["final"] = kept["final"] or final
            elif not current:
                pass
            elif kept is None or final or (not kept["final"] and order > kept["order"]):
                try:
                    kept = test.views[role] = {"key": key, "order": order, "final": final,
                                               "view": self._run_view(test, role, run, frame)}
                except (RunError, OSError, counts_file.CountsError):     # its files went with it
                    pass
            return None if kept is None else kept["view"]

    def _run_view(self, test: Test, role: str, run, frame: dict) -> dict:
        """One run's view of a counted `frame`: {role, side, seq, iter, voxel, x0, y0 (the crop's first column's corner,
        µm), w, h, labels (the crop's consensus labels: base64 of little-endian u32, w x h, row-major, y down), cell
        and sources (their labels in this run, found by their Xenium ids; null where absent), held (per suspect, how
        much of it proseg counts for the cell in this frame, 0 to 1 to 3 decimals: engine/held.py weights on the
        suspect's own transcript in this run, `match`; null where its transcript is not found or cannot be told from
        another, or without the cell), held_by ("exact": the frame's where export of the view's box; else "approx":
        the frame's per-layer labels times the cell's keep), unsure (1 where its transcript cannot be told from another
        of its gene at its place, `match`: never counted), cell_total (proseg's count of the cell in this frame,
        rounded; null without the cell), parents (flat pairs [label, the label of the cell of that name in the parent,
        ...] for the crop's labels, 0 where the parent has none: `_parent_labels`), ms (the time it took)}."""
        began = time.perf_counter()
        run_id = run.meta["id"]
        names = test.names.get(role)
        if names is None or names["run"] != run_id:
            names = test.names[role] = {"run": run_id, **run_names(run)}
        if "parents" not in names:
            names["parents"] = self._parent_labels(test, names["cells"])
        seq = int(frame["seq"])
        labels, plane = frame_labels(run, frame, self.runs.frame_path(run_id, seq))
        cell = names["cells"].get(test.cell_name)
        v, x0, y0 = labels.voxel, labels.x0, labels.y0
        box = test.crop or [x0, y0, x0, y0]
        c0, r0 = max(0, math.floor((box[0] - x0) / v)), max(0, math.floor((box[1] - y0) / v))
        c1 = max(c0, min(labels.nx, math.ceil((box[2] - x0) / v)))
        r1 = max(r0, min(labels.ny, math.ceil((box[3] - y0) / v)))
        info = run.info or {}
        ncells, z = int(info.get("cells") or 0), info.get("z") or {}
        z_origin, z_spacing = float(z.get("grid_origin", 0.0)), float(z.get("grid_spacing", 1.0))
        found_counts = counts_file.read(run.live / frame["counts_file"], ncells=ncells)
        held_values: list = [None] * len(test.suspects)
        unsure = np.zeros(len(test.suspects), bool)
        held_by, total = "approx", None
        if cell is not None:
            row = counts_file.rows(found_counts, [cell - 1], ("e", "n"))
            e, n = row["e"][0].astype(np.float64), row["n"][0].astype(np.float64)
            q = np.divide(e, n, out=np.zeros_like(e), where=n > 0)
            lo, hi = int(found_counts.indptr[cell - 1]), int(found_counts.indptr[cell])
            total = int(math.floor(float(found_counts.e[lo:hi].astype(np.float64).sum()) + 0.5))
        if test.suspects:
            asked_box = None if test.where_box is None else qc_pool.on_grid(test.where_box, frame)
            export = None if asked_box is None else self.runs.where_export(run_id, seq, list(asked_box))
            sx, sy = np.array([s["x"] for s in test.suspects]), np.array([s["y"] for s in test.suspects])
            reach = 2 * MATCH_UM
            asked = [sx.min() - reach, sy.min() - reach, sx.max() + reach, sy.max() + reach]
            grid = export if export is not None else {"x0": float(frame["x0"]), "y0": float(frame["y0"]),
                                                      "voxel": float(frame["voxel"]), "nx": int(frame["nx"]),
                                                      "ny": int(frame["ny"]), "nz": int(frame.get("nz") or 1)}
            # Whole buckets around the suspects (an exact share counts every transcript of a suspect's bucket).
            records, _ = self.runs.transcript_index(run_id).query(held.query_box(grid, asked), QUERY_MAX)
            found = match(records, names["genes"], test.suspects) if records is not None else np.full(sx.size, -1)
            hit, unsure = found >= 0, found == UNSURE
            if cell is not None and hit.any():
                if export is not None:
                    w, how = held.weights(records, cell - 1, q, grid, z_origin, z_spacing,
                                          export=held.read_where(export))
                    w = w[found[hit]]
                else:
                    w, how = held.weights(records[found[hit]], cell - 1, q, grid, z_origin, z_spacing,
                                          layers=_layers_of(labels, run, frame))
                held_by = "exact" if how == "exact" else "approx"
                if how != "none":
                    for k, value in zip(np.flatnonzero(hit).tolist(), w.tolist(), strict=True):
                        held_values[k] = round(float(value), 3)
        crop = plane(c0, r0, c1, r1) if c1 > c0 and r1 > r0 else np.zeros((0, 0), np.uint32)
        # Each cell of the crop as the parent's cell of the same name (the corner colours the test runs'
        # cells as the stage colours the parent's): flat pairs [label, parent label, ...], 0 where the parent has none.
        present = np.unique(crop)
        parents = [x for label in present[present > 0].tolist() for x in (label, names["parents"].get(label, 0))]
        return {"role": role, "side": role[0], "seq": seq, "iter": int(frame.get("iter") or 0),
                "voxel": v, "x0": x0 + c0 * v, "y0": y0 + r0 * v, "w": c1 - c0, "h": r1 - r0,
                "labels": base64.b64encode(np.ascontiguousarray(crop, "<u4").tobytes()).decode("ascii"),
                "cell": cell, "sources": [names["cells"].get(name) for name in test.source_names],
                "held": held_values, "held_by": held_by, "unsure": unsure.astype(int).tolist(),
                "cell_total": total, "parents": parents, "ms": round((time.perf_counter() - began) * 1e3, 1)}

    def _parent_labels(self, test: Test, cells: dict) -> dict:
        """{a test run's label: the label of the cell of the same name in the parent} for that run's cells (`cells`: its
        run_names "cells"), from the parent's cells.txt read once per test run, keeping only the names the test run has
        (a whole sample's parent names many more); {} when the parent's cells cannot be read."""
        try:
            parent = self.runs.get(test.parent)
            ncells, path = int((parent.info or {}).get("cells") or 0), parent.live / "cells.txt"
            lines = path.read_text().split("\n")[:ncells] if path.exists() else []
        except (RunError, OSError):
            return {}
        found: dict[int, int] = {}
        for k, name in enumerate(lines):
            label = cells.get(name)
            if label is not None and label not in found:
                found[label] = k + 1
        return found

    # ---- state -------------------------------------------------------------------------------------------------------
    def _set(self, test: Test, **values):
        """Change a test's fields; a test that ended keeps its state (a cancel wins over an answer that came later)."""
        with self.lock:
            state = values.pop("state", None)
            if state is not None and test.state in ON_ITS_WAY:
                if state in ON_ITS_WAY:
                    test.state = state
                    test.marks[state] = self.clock()
                else:
                    self._end(test, state, values.pop("message", None))
            elif state is not None:
                for key in ("message", "result", "reasons"):
                    values.pop(key, None)
            for key, value in values.items():
                setattr(test, key, value)
        if state is not None:
            self._push(test, force=True)

    def _end(self, test: Test, state: str, message: str | None = None):
        """Lock held: the test ended (past waiting, if it waited)."""
        test.state, test.message, test.waiting = state, message, []
        test.marks[state] = self.clock()

    def _push(self, test: Test, force: bool = False):
        """The parent's run detail goes out again (its WebSocket "run" push carries qc_test): at once for a new state,
        at most every PUSH_S for the progress."""
        now = self.clock()
        if not force and now - test.pushed < PUSH_S:
            return
        test.pushed = now
        try:
            self.runs.get(test.parent).touch()
        except RunError:
            pass


def time_limit(of: int) -> float:
    """A test's time limit (s) for runs of `of` iterations: TIMEOUT_S, times their schedule over the measured one's
    (burn-in + samples + hill-climb) where that is longer."""
    schedule = qc_rule.MEASURED["schedule"]
    measured = schedule["burnin"] + schedule["samples"] + schedule["hillclimb"]
    return TIMEOUT_S * max(1.0, float(of) / measured)


def _progress(test: Test) -> dict:
    """The test's progress as the API gives it (a copy): {iter, of, runs, done, sides {A, B}, marks, phase, retry}."""
    progress = dict(test.progress)
    progress["sides"], progress["marks"] = dict(progress["sides"]), list(progress["marks"])
    return progress
