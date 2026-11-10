"""Test a setting here: the test's targets, fixed before any run, and every reason that stops it, makes it wait, or
says it runs where it was not measured.

A test was measured (engine/qc_rule.py MEASURED and CONSTANTS) on detail runs paused at the end of proseg's default
schedule, started from cells (with nuclei or without), over a box re-run as it is (at most 500 µm across, at least as
wide as the start's minimum) or a 500 µm square around the cell, with the mixing check on and scored at that last
frame, its suspects chosen on proseg's own count for starts from cells only, on the drawn outlines for starts with
nuclei (count_basis says so: qc_rule.measured_on_proseg). It runs on any run that reached that end and kept proseg's
own counts. Each reason (`reason`: {code, words, ...data}, REASON_WORDS in the order the page lists them) is of one
KIND:
  * "refuse": nothing to test, or the test cannot run (the run is being saved, or ended before its end; it has no
    proseg count to test with, no_counts or old_proseg; the mixing check cannot run on the sample; the cell is not
    judged, nothing of it glows on either count, it takes its RNA from its own kind only, or only from neighbours at
    the area's edge; its suspects have no own RNA at the same spot to be judged against, no_controls): OutsideDomain,
    409;
  * "wait": the app meets it itself or waits for it (the run's end, the check switched on, learned and scored, Go
    deeper's details; another run sampling, engine/qc_test.py's): `pool` raises NotYet and engine/qc_test.py keeps the
    test waiting;
  * "caveat": the test runs, judged with the start's nearest measured row (qc_rule.row_for), and its answer says so.
`run_reasons` gives the run's own, from its meta and info alone; `prerequisites` adds the check's and whether F is
scored; the cell's and the pool's follow from the stored mixing-check bytes and Go deeper's answers.

The area (`area_of`): the region as it is when its box fits in 500 x 500 µm; else a 500 µm window around the cell's
centre at F, per axis no wider than the region and shifted to stay inside it ("square", or "crop" where the region is
narrower one way).

Every number about a cell's RNA is proseg's own count (analysis/counts_file.py: the averaged count e of a counted
frame, its keep q = e / n); the drawn outlines are positions only (footprints, near flags, start classes, centres). F
counted for the hold (or a pause, a finish, the end) is read on the count proseg saves (f) by the panel and Go deeper;
the test reads its averaged count there all the same (F's bytes on e, <seq>.qc.averaged.bin; its targets e / n), and
its reply says which count F is read on (`count`) and Go deeper's own line at F (`line_shown`: the seed's answer worked
out once more on f, `_cell` with count "sampled"; `line` itself on an averaged F). Only when the clicked cell's
suspects on e cannot be tested (nothing of it glows, every source is of its own kind, or no giver lies in the inner
area) are its suspects f's, Go deeper's there (that answer's: its targets and controls f / n), with the caveat
saved_suspects: where f's can be tested, or where nothing of it glows on e but its dots glow on f (then refused for
f's own reason, never no_suspects while Go deeper shows them). Every other cell's suspects, and the clicked cell's
otherwise, stay e's, as the test was measured.
Proseg's share of each transcript for a cell, w (engine/held.py), comes from one place per test area, its basis
(`QcPools._basis`), at the parent's last frame F (its schedule's end, `schedule_done.seq`):
  * exact: proseg's export of the area at F (a where export: Runs.where_export, or asked of the parent held at F,
    Runs.request_where);
  * approx: F's own per-layer labels times q (a dense frame's, or a tiled frame's window when it covers the area), when
    the parent moved on past F (caveat approx_targets);
  * a tiled parent whose F window misses the area takes it from its state now, exactly (asked of the parent paused or
    sampling; caveat targets_now): the frame work is then at that frame H instead of F, F's stored bytes and centres
    kept;
  * approx again when such a parent cannot be asked (it ended, or no export came in time): F's window over the part of
    the area it covers, a record outside it no cell's (caveat approx_targets);
  * none: nothing at all (a parent past F that cannot be asked, whose frame has no window over any of the area): no
    record has a share for any cell.
Never the flat 2-D column of a tiled frame. A cell's targets are its where reply's own (Go deeper's where engine places
the cell's box itself: exact from an export of that frame covering it, else that frame's labels times q, else none), so
a test whose clicked cell's or pooled cells' replies placed them approximately carries approx_targets as well, whatever
the area's basis; and a cell may glow where the area's basis holds no share of its own RNA (none; or a window that
misses the cell while Go deeper still holds an export of its box), so its suspects have no control: see no_controls
below.

QcPools computes a cell's pool (GET /runs/{id}/qc/test/pool previews it; engine/qc_test.py fixes it for the test) in
three tiers:
  * cheap, no frame work: the run's own reasons and the check's (on, learned, Go deeper's details, the measured
    settings, count_basis), F scored when no refusal or wait holds there, then from F's stored bytes and centres the
    cell's own (at the area's edge, not judged);
  * the frame context, once per (run, F, model, settings, H, basis): proseg's count at H through the worker's own code
    with numpy (analysis/worker.py Worker.count on H's counts file, as Go deeper's answers read it), the parent's
    records (transcripts.bin.gz) with a key of their 16 bytes, the starting cells (the prior's labels and permille),
    H's per-layer labels for the footprints; per test area its w (`_Weights`: the cells' records w > 0);
  * per cell, kept per frame context: Go deeper's answer (Worker.deeper with the engine's own block and the stored
    byte: the card the user saw), the page's scoring of it (engine/qc_scoring.py: the where request, then w, π and
    P = w x π per point of the where measurement at H, engine/deeper_where.py `measure`), its targets (its glowing
    points, P >= GLOW_MIN, any owner: a point just outside the outline is RNA proseg moved in; likely: π >= LIKELY_MIN
    on a point with w >= LIKELY_W) with their records, givers and start classes (where each sat when the cells
    started: NONE, OWN, GIVER_STRONG or GIVER_WEAK by the hold there, OTHER), its pattern and key, and its own-RNA
    controls (drawn from its records with w > 0: 3 per acting target with a placed giver, of the same giver, near
    flag and w band (W_BANDS), drawn in point order; 50 of its own RNA; seeded per cell).
A cell's acting targets are its likely ones with a placed giver; a cell with none but glowing ones with a placed giver
acts on those (weak_suspects). Then the pool: the cell (wherever it lies; at least one primary target: an acting one
whose giver's F centre lies in the inner area; refused no_suspects when nothing of it glows with a known neighbour on
either count, whatever its byte; a cell under the cut-off is tested on its glowing points with the caveat low_share),
then the mixed, judged cells of the cell's own key (its pattern, and the hold for a neighbour's) in the inner area,
nearest first (ties by cell number), until their primary targets (likely, of a mixed cell, the giver placed inside)
reach the row's k, and RESERVE_CELLS more; each computed only once reached. The other cells are chosen this way whatever
the clicked cell's own byte: a cell under the cut-off is pooled with the mixed cells of its key. A pool that never
reaches k holds them all (too_few, a caveat). While the row's rule is "excess" (qc_rule: the suspects judged against
their own RNA at the same spot, their matched controls), a pool none of whose cells (members and reserve: all a matching
can take from) has a matched control could never be judged, so it is refused (no_controls), at once when the area's
basis is none (no cell can have one); the rule "suspects" judges the suspects alone, and runs without them.
`match_input` finds the pool's items in the test's shared input (engine/qc_test.py shared_input) by gene and x, y bits
(duplicates by depth rank), as proseg reads it (min qv, excluded genes), pools again on what was found, and gives the
readout's Items (engine/qc_readout.py) and the rule's PoolOrder (engine/qc_rule.py).

A whole-sample (tiled) run's frame context reads H in its column basis for positions (`_TiledContext`): a target's
start is prior.tiles' label and permille at its column, a giver's footprint its consensus columns, and a record's key
comes from `_RecordIndex`. Such a run (or a box over the engine's max_region_um, run in tiles) is tested like any
other, in a 500 µm window around the cell inside its extent (`area_of`), with its caveat (whole_sample, tiled_box).

Threading: one daemon worker per run does the frame work, at a lower priority, yielding between cells, with numpy's
BLAS on one thread meanwhile (ONE_BLAS: BLAS's own threads would spin at the engine's priority); at most one request
waits for it and the newest replaces it; the same request is worked out once (single-flight). A frame context is let
go POOL_HOLD_S after its last use; answers and pools are kept, bounded. Shared fields change under `lock`, and no file
work happens under it."""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field, replace
import gzip
import importlib
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Callable, Iterable

import numpy as np

from ..analysis import counts_file
from . import held, qc_readout, qc_rule, qc_scoring
from .deeper_where import DeeperWhere, _decode, givers_of, near as near_footprint
from .qc import QcError
from .runs import TERMINAL, RunError, finish_and_save
from .transcript_index import RECORD

POOL_V = 1
POOL_HOLD_S = 60.0          # a run's frame context is let go this long after its last use
CELLS_KEPT = 2048           # cells' answers kept per frame context, whatever the area: a cell's targets are its own where
                            # reply's; its controls' shares are the area's, which are the same in every area holding the
                            # cell's pools whole (a pooled cell lies 25 µm inside the area, a window is centred on its
                            # clicked cell or ends where the run's transcripts end), so only a cell whose pools cross an
                            # area's edge would read otherwise in another area
POOLS_KEPT = 16             # pools kept
RESERVE_CELLS = 2           # cells of the pattern kept past the members, for those the matching or the runs lose
WAIT_MAX_S = 10.0           # the preview's `wait` at most
ESTIMATE_S = {"paired": 45.0, "cells": 50.0}    # a test's seconds, POST to answer, by its row's start (provisional)
MORPHOLOGY = "morphology-steps-per-iter"
SCHEDULE_FLAGS = {"burnin": "burnin-samples", "samples": "samples", "recorded": "recorded-samples",
                  "hillclimb": "hillclimb"}         # info.json's schedule names -> proseg's flags
CLASSES = ("NONE", "OWN", "GIVER", "OTHER")         # the start classes a pattern is decided over (GIVER: either hold)
PATTERNS = {"OWN": "own", "NONE": "none", "GIVER": "neighbour", "OTHER": "other", "MIXED": "mixed"}   # the reply's
# The patterns not measured (only suspects inside their own start were): each one's caveat.
PATTERN_REASONS = {"NONE": "pattern_none", "GIVER": "pattern_neighbour", "OTHER": "pattern_other",
                   "MIXED": "pattern_mixed"}
LINE = {"OWN": "own", "NONE": "none", "GIVER_STRONG": "neighbour", "GIVER_WEAK": "neighbour", "OTHER": "other"}
NOT_JUDGED = 255            # a stored byte: fewer than judge_min transcripts (analysis/worker.py NOT_JUDGED)
TIE_Z = 1e-3                # match_input: depths this close (either side) can't tell which transcript is which
TICK_S = 0.25               # an idle worker looks this often whether its frame context is due to go
POLL_S = 0.05               # a blocking pool() calls its check() this often while it waits
FAILED = "Couldn't count the nearby cells with the same pattern"     # the start of a failure's words
NONE_FOUND = "None of this cell's suspects could be found in the area's transcripts"     # TooFewFound's words
INPUT_COLUMNS = ["transcript_id", "feature_name", "x_location", "y_location", "z_location", "qv"]
LIKELY_W = 0.5              # a likely suspect: proseg counts at least this share of it for the cell (w): the page's own
                            # (qc_scoring.LIKELY_W, read when used: `likely_rule`), this until qc_scoring has one
W_BANDS = (0.25, 0.75)      # a matched control shares its target's band of w: under 0.25, 0.25 to 0.75, 0.75 or more
WHERE_WAIT_S = 10.0         # the frame work waits this long for proseg's export of a test area (held: served at once)
AREAS_KEPT = 4              # a frame context keeps the w of this many test areas, the newest

# The reasons, in the order the page lists them (frontend/src/qc-copy.mjs testReasonGroup, this order kept in each
# group); the page's deck holds the same words and kinds (the shared fixture
# frontend/test/fixtures/qc-test-reasons.json). Plain words: no banned stem, no look-alike words; "hold" only in the
# hold's own name. 1-16 are the run's own (run_reasons), 17-23 the check's and F's (count_basis: its row's,
# qc_rule.measured_on_proseg), 24-25 engine/qc_test.py's, 26-27 from the stored bytes and centres, 28-37 from the cell's
# own answer (saved_suspects: its suspects taken on the count proseg saves at F, where the averaged count gives none
# that can be tested), 38-39 from the pool (no_controls: also when the test area's basis is none; too_few: the row's
# rates are those of pools that reached k, qc_rule CONSTANTS); approx_targets and targets_now from the basis of the
# test's area (QcPools._basis).
REASON_WORDS: dict[str, str] = {
    "ending": "This run is being saved",
    "ended": "This run ended before its last iteration",
    "no_counts": "Made before proseg's own count was kept",
    "old_proseg": "Update proseg-live to test a setting",
    "not_at_end": "Starts once your run reaches its last iteration",
    "past_end": "Not measured yet for a run that sampled on past its end",
    "approx_targets": "Suspects placed approximately: proseg's exact shares at your run's end are no longer available",
    "targets_now": "Suspects taken where your run is now, not at its end",
    "whole_sample": "Not measured yet for runs of 20,000 cells or more",
    "tiled_box": "Not measured yet for regions over {max_um:g} µm",
    "schedule": "Not measured yet for a changed schedule",
    "nuclei_start": "Not measured yet for runs started from nuclei only",
    "lasso": "Not measured yet for lasso regions",
    "area_narrow": "Not measured yet for regions over 500 µm long and under 500 µm wide",
    "square_not_measured": "Not measured yet for cells-only starts on regions over 500 µm",
    "region_small": "Not measured yet for regions under {min_um:g} µm",
    "check_unavailable": "The mixing check can't run on this sample",
    "check_off": "Switches the mixing check on first",
    "check_wait": "Starts once the mixing check has learned this sample",
    "check_scoring": "Starts once the mixing check has scored your run's last iteration",
    "details": "Starts once Go deeper's details are learned for this sample",
    "settings": "Not measured yet with these mixing check settings",
    "count_basis": "Not measured yet with suspects from proseg's own count",
    "busy": "A test of this run is on its way",
    "other_run": "Starts once the other run pauses or ends",
    "edge": "Not measured yet for cells this close to your region's edge",
    "not_judged": "This cell has too few transcripts to be judged",
    "no_suspects": "No suspects to test in this cell: none glows, or none could be placed",
    "saved_suspects": "Suspects taken from proseg's count as saved: its averaged count gives none that can be tested",
    "low_share": "Doesn't look mixed overall: its pink dots are tested anyway",
    "weak_suspects": "Its suspects are weak: none of its pink dots looks likely",
    "same_type": "The neighbours this cell's RNA seems to come from all look like its own kind",
    "pattern_none": "Not measured yet for suspects that sit where no cell started",
    "pattern_neighbour": "Not measured yet for suspects that sit inside a neighbour's starting outline",
    "pattern_other": "Not measured yet for suspects that sit inside another cell's starting outline",
    "pattern_mixed": "Not measured yet for suspects that sit in different places",
    "no_primary": "This cell's suspects all come from neighbours at the area's edge",
    "no_controls": "No own RNA at the same spot to compare its suspects with",
    "too_few": "Its numbers are for tests with {need} suspects or more: {have} here",
}
# Each reason's kind: "refuse" (nothing to test, or it cannot run: 409), "wait" (the app meets it or waits for it: the
# test waits) or "caveat" (the test runs, and its answer says so).
KIND: dict[str, str] = {
    "ending": "refuse", "ended": "refuse", "no_counts": "refuse", "old_proseg": "refuse", "not_at_end": "wait",
    "past_end": "caveat", "approx_targets": "caveat", "targets_now": "caveat", "whole_sample": "caveat",
    "tiled_box": "caveat", "schedule": "caveat", "nuclei_start": "caveat", "lasso": "caveat", "area_narrow": "caveat",
    "square_not_measured": "caveat", "region_small": "caveat", "check_unavailable": "refuse", "check_off": "wait",
    "check_wait": "wait", "check_scoring": "wait", "details": "wait", "settings": "caveat", "count_basis": "caveat",
    "busy": "refuse", "other_run": "wait", "edge": "caveat", "not_judged": "refuse", "no_suspects": "refuse",
    "saved_suspects": "caveat", "low_share": "caveat", "weak_suspects": "caveat", "same_type": "refuse",
    "pattern_none": "caveat", "pattern_neighbour": "caveat", "pattern_other": "caveat", "pattern_mixed": "caveat",
    "no_primary": "refuse", "no_controls": "refuse", "too_few": "caveat",
}
_ORDER = {code: n for n, code in enumerate(REASON_WORDS)}


def reason(code: str, **data) -> dict:
    """A reason as the API gives it: {code, words, ...data} (tiled_box: max_um; region_small: min_um; too_few: have,
    need; busy: why; check_scoring: wait). KeyError for an unknown code."""
    return {"code": code, "words": REASON_WORDS[code].format(**data), **data}


def ordered(reasons: Iterable[dict]) -> list[dict]:
    """Reasons in the order the page lists them (REASON_WORDS)."""
    return sorted(reasons, key=lambda r: _ORDER.get(r.get("code"), len(_ORDER)))


def split(reasons: Iterable[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(refusals, waits, caveats) of these reasons, each in the page's order; a code of no known kind refuses."""
    found: dict[str, list] = {"refuse": [], "wait": [], "caveat": []}
    for found_reason in ordered(reasons):
        found[KIND.get(found_reason.get("code"), "refuse")].append(found_reason)
    return found["refuse"], found["wait"], found["caveat"]


class OutsideDomain(QcError):
    """A test refused (refusals only: nothing to test, or it cannot run): 409 {detail: the first reason's words, status,
    reasons} (server/app.py copies `extra` into the answer)."""

    def __init__(self, reasons: list[dict]):
        reasons = [dict(r) for r in reasons]
        if not reasons:
            raise ValueError("a refusal needs its reasons")
        if any(KIND.get(r.get("code"), "refuse") != "refuse" for r in reasons):
            raise ValueError("a refusal holds refusals only")
        super().__init__(reasons[0]["words"], 409)
        self.reasons = reasons
        self.extra = {"reasons": reasons}


class NotYet(QcError):
    """`pool` while a wait holds (`waits`, in the page's order): engine/qc_test.py keeps its test waiting; never an
    answer of the API."""

    def __init__(self, waits: list[dict]):
        waits = [dict(w) for w in waits]
        if not waits:
            raise ValueError("a wait needs its reasons")
        super().__init__(waits[0]["words"], 409)
        self.waits = waits


class StillWorking(QcError):
    """`pool` past its `timeout`, the cell's pool still being worked out in the run's worker (a whole sample's can take
    minutes): asked again, it is the same request (single-flight), so engine/qc_test.py keeps its test waiting for it;
    503 "try again" to any other caller."""

    def __init__(self):
        super().__init__(f"{FAILED} in time: try again", 503)


class TooFewFound(QcError):
    """`match_input`: none of the clicked cell's suspects was found in the test's input, so the test cannot run
    (engine/qc_test.py fails it, before any run, in these words: NONE_FOUND)."""

    def __init__(self):
        super().__init__(NONE_FOUND, 409)


# ---- the area --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Area:
    """What a test re-runs."""
    rule: str                                    # "as_is" | "square" | "crop" (the test's own situation)
    box: tuple[float, float, float, float]       # µm
    inner: tuple[float, float, float, float]     # the box inset by MEASURED["inset_um"], its edges included
    spec: dict                                   # Runs.create's region: the parent's meta["region"] as it is (as_is:
                                                 # a box or a lasso) or {"kind": "box", "box": list(box)}

    def public(self) -> dict:
        """As the pool reply and the {test} record give it: {rule, box, inner}."""
        return {"rule": self.rule, "box": list(self.box), "inner": list(self.inner)}


def situation_of(region: dict, side: float = 500.0) -> str:
    """How a region stands against the measurements (its caveats): "lasso" (a polygon region), else by its bounding box
    (w x h µm): "as_is" when max(w, h) <= side (re-run as it is), "square" when min(w, h) >= side (a square of that
    side around the cell), "narrow" otherwise (one side over, the other under)."""
    if region.get("kind") == "polygon":
        return "lasso"
    x0, y0, x1, y1 = (float(v) for v in region["bbox"])
    w, h = x1 - x0, y1 - y0
    if max(w, h) <= side:
        return "as_is"
    if min(w, h) >= side:
        return "square"
    return "narrow"


def fits(region: dict, side: float = 500.0) -> bool:
    """Whether a region (a box, a lasso or a whole sample) is re-run as it is: its bounding box fits in side x side."""
    x0, y0, x1, y1 = (float(v) for v in region["bbox"])
    return max(x1 - x0, y1 - y0) <= side


def area_of(region: dict, place: tuple[float, float], side: float = 500.0) -> Area:
    """The area a test of the cell at `place` (its centre at F, µm) re-runs: "as_is", the region itself (its bounding
    box fits in side x side; no place needed); else a window around the cell, per axis min(side, extent) wide, centred
    on it and shifted (never clipped) to stay inside the region's box: lo' = min(max(c - w/2, lo), hi - w), hi' = lo'
    + w; "square" when it is side x side, "crop" where the region is narrower one way. ValueError for a window around a
    cell with no place."""
    x0, y0, x1, y1 = (float(v) for v in region["bbox"])
    if fits(region, side):
        rule, box = "as_is", (x0, y0, x1, y1)
        spec = {key: (list(value) if isinstance(value, (list, tuple)) else value) for key, value in region.items()}
    else:
        cx, cy = (float(v) for v in place)
        if not (math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError("the cell has no place for an area around it")
        w, h = min(side, x1 - x0), min(side, y1 - y0)
        bx0, by0 = min(max(cx - w / 2.0, x0), x1 - w), min(max(cy - h / 2.0, y0), y1 - h)
        rule, box = ("square" if w == side and h == side else "crop"), (bx0, by0, bx0 + w, by0 + h)
        spec = {"kind": "box", "box": list(box)}
    inset = float(qc_rule.MEASURED["inset_um"])
    return Area(rule=rule, box=box, inner=(box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset), spec=spec)


# ---- the run's own reasons -------------------------------------------------------------------------------------------
def region_of(meta: dict) -> dict:
    """A run's region: meta["region"], else its box (an older run)."""
    return meta.get("region") or {"kind": "box", "bbox": meta.get("box")}


def row_of(meta: dict) -> tuple[str, str]:
    """The measured row a test of this run is judged with (qc_rule.row_for: its start, qc_rules.start_of's read when
    called, and its region's box)."""
    from . import qc_rules
    return qc_rule.row_for(qc_rules.start_of(meta), region_of(meta)["bbox"])


def reached_end(meta: dict) -> bool:
    """Whether the run reached the end of its schedule (proseg held there: `schedule_done` with its iteration)."""
    done = meta.get("schedule_done")
    return isinstance(done, dict) and done.get("iter") is not None


def being_saved(meta: dict) -> bool:
    """Whether the run is being finished, saved or left (ending): state "finishing", meta `leaving`, a Finish & save
    whose save is still to come (runs.finish_and_save), or a save under way (zarr_save "saving"). A test is refused
    then, and one waiting or on its way stops (engine/qc_test.py _check): the run is ending (a leave takes its folder; a
    finish or a save is the user's end of it), though a finish or a save keeps what a test reads (the run's input, last
    iteration and scores; a test's runs read no sample). Waiting through a finish or a save instead would take a wait of
    its own (words such as "Starts once your run is saved"), held until the save ends or its dialog is closed, a leave
    staying a refusal: one reason code more, which the page's words and the harness would share."""
    return (meta.get("state") == "finishing" or bool(meta.get("leaving")) or finish_and_save(meta)
            or (meta.get("zarr_save") or {}).get("state") == "saving")


def _schedule_changed(meta: dict, info: dict | None) -> bool:
    """Whether the run's schedule is not the measured one: info.json's schedule (else the run's options, else proseg's
    defaults) in burn-in, samples, recorded samples or hill-climb; the boundary attempts per iteration now (live
    values, else options, else proseg's default); or any Temperature set."""
    from . import qc_rules as qr
    measured = qc_rule.MEASURED["schedule"]
    options, live = meta.get("options") or {}, meta.get("live_values") or {}
    schedule = info.get("schedule") if isinstance(info, dict) else None
    if isinstance(schedule, dict):
        found = {key: schedule.get(key) for key in SCHEDULE_FLAGS}
    else:
        from . import proseg_options
        defaults = {o["flag"]: o["default"] for o in proseg_options.OPTIONS}
        found = {key: options.get(flag, defaults.get(flag)) for key, flag in SCHEDULE_FLAGS.items()}
    if not all(qr.same(found[key], measured[key]) for key in SCHEDULE_FLAGS):
        return True
    steps = next((v for v in (live.get(MORPHOLOGY), options.get(MORPHOLOGY)) if v is not None),
                 measured["morphology_steps"])
    if not qr.same(steps, measured["morphology_steps"]):
        return True
    return options.get("temperature") is not None or live.get("temperature") is not None


def counts_reason_of(meta: dict, info: dict | None) -> str | None:
    """Why a run has no proseg count to test with, on its meta and info.json: engine/runs.py counts_reason itself where
    the engine has it, else its rule: "old_proseg" when its proseg-live could not keep them (meta live_counts), else
    "no_counts" when its info.json holds no counts block (a run made before they were kept); None when it keeps them,
    or before its info.json."""
    from . import runs as runs_module
    shared = getattr(runs_module, "counts_reason", None)
    if shared is not None:
        return shared(_RunLike(meta, info))
    if meta.get("live_counts") == "old_proseg":
        return "old_proseg"
    if isinstance(info, dict) and not isinstance(info.get("counts"), dict):
        return "no_counts"
    return None


@dataclass(frozen=True, eq=False)
class _RunLike:
    """What engine/runs.py counts_reason reads of a run: its meta and info."""
    meta: dict
    info: dict | None


def run_reasons(meta: dict, info: dict | None, *, max_um: float) -> list[dict]:
    """The run's own reasons, of every kind, from its meta and info alone (pure and cheap), in the page's order: at most
    one of ending (being saved: `being_saved`), ended (ended before its end: no end frame will ever come), not_at_end
    (a wait: not at the end of its schedule yet) and past_end (sampled on past its end; F stays the end); no_counts or
    old_proseg (no proseg count to test with: `counts_reason_of`); whole_sample for a run of a whole sample's size
    (qc_constants.run_mode "whole": WHOLE_CELLS cells or more, tiled or not, the rule the mixing check reads) or
    tiled_box {max_um} for any other tiled run (a box or lasso over `max_um` across: the engine's [proseg]
    max_region_um, Runs.max_region, which no run records); schedule; nuclei_start (judged as a start from nuclei and
    cells, whose region reasons follow); lasso, area_narrow, the situation's own code where it was not measured for
    this start (qc_rule.refusal_of: square_not_measured), region_small {min_um}. The start is qc_rules.start_of's, read
    when called (tests patch it)."""
    from . import qc_constants, qc_rules
    found = []
    if being_saved(meta):
        found.append(reason("ending"))
    elif not reached_end(meta):
        found.append(reason("ended" if meta.get("state") in TERMINAL else "not_at_end"))
    elif meta.get("continued"):
        found.append(reason("past_end"))
    without = counts_reason_of(meta, info)
    if without is not None:
        found.append(reason(without))
    mode = qc_constants.run_mode(meta, info)
    if mode == "whole":
        found.append(reason("whole_sample"))
    elif mode == "tiled":
        found.append(reason("tiled_box", max_um=float(max_um)))
    if _schedule_changed(meta, info):
        found.append(reason("schedule"))
    start = qc_rules.start_of(meta)
    if start not in qc_rule.STARTS:
        found.append(reason("nuclei_start"))
        start = "paired"                                     # its row's (qc_rule.row_for)
    region = region_of(meta)
    situation = situation_of(region)
    if situation == "lasso":
        found.append(reason("lasso"))
    elif situation == "narrow":
        found.append(reason("area_narrow"))
    else:
        if not qc_rule.offered(start, situation):
            found.append(reason(qc_rule.refusal_of(start, situation)))
        x0, y0, x1, y1 = (float(v) for v in region["bbox"])
        if situation == "as_is" and min(x1 - x0, y1 - y0) < qc_rule.area_min_um(start):
            found.append(reason("region_small", min_um=qc_rule.area_min_um(start)))
    return ordered(found)


# ---- the targets -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One target or control, fixed before any run."""
    kind: str                     # "target" | "matched" | "uniform"
    cell: int                     # its cell in the parent (label - 1)
    target: int | None            # "matched": the index in Pool.items of its target
    giver: int | None             # the parent cell it came from; None for "uniform"
    gene: str
    record: int                   # its index in the parent's live/transcripts.bin.gz (file order): a control's from
                                  # the cell's records with w > 0; a target's, the lowest index of the parent record
                                  # whose 16 bytes (x, y, z, gene) are the ones Go deeper's where reply measured for it
    x: float                      # that record's float32 values (np.float32(x) gives the bits back)
    y: float
    zn: float                     # its normalised depth: only for the depth rank among records at one place
    near: bool | None             # within 2 µm of its giver (deeper_where.near)
    p: float | None               # a target's P = w x π (float32)
    primary: bool                 # a target that counts: acting (likely; a weak clicked cell's glowing), of a mixed
                                  # cell or the clicked one, its giver placed and that giver's centre at the parent's
                                  # last frame inside the inner area
    start_class: str | None       # a target's place when the cells started:
                                  # NONE | OWN | GIVER_STRONG | GIVER_WEAK | OTHER
    w: float | None = None        # proseg's share of it for its cell at the basis' frame (a target's as its where reply
                                  # gives it, 3 decimals; a control's from the test area's basis)


@dataclass(frozen=True)
class Pool:
    """A cell's pool at the parent's last frame F."""
    v: int
    run: str
    seq: int                      # F
    iter: int
    model: str                    # the mixing check's model id
    constants: str                # qc_rule.CONSTANTS_ID
    start: str                    # "paired" | "cells" | "nuclei" (the run's own: qc_rules.start_of)
    area: Area
    k: int                        # the row's
    seed: int                     # the clicked cell (label - 1)
    seed_name: str | None
    cells: tuple[int, ...]        # parent cell numbers: the members, then the reserve (<= RESERVE_CELLS), in pool order
    members: int                  # how many of `cells` are members
    distance_um: tuple[float, ...]
    names: dict                   # parent cell number -> Xenium id: the cells, the givers in `items`, the view sources
    items: tuple[Item, ...]       # `cells`' primary targets, their matched controls, then `cells`' uniform controls
    suspects: int                 # the members' primary targets, summed (before matching)
    available: int | None         # their sum over every cell of the pattern, only when the pool was not reached
    record_duplicates: int        # the targets' records the parent holds more than once (each counted once)
    this_cell: dict               # {likely, suspects, pattern, line: {likely, own, none, neighbour, other}}
    view_suspects: tuple[dict, ...]   # the seed's primary targets as the corner's view matches them:
                                      # {x, y (2 decimals), gene, p (3 decimals), rank?, of?}, at most 400
    view_sources: tuple[int, ...]     # the seed's where-request sources (parent cell numbers)
    row: tuple[str, str]              # the measured row it is judged with (qc_rule.row_for): (start, situation)
    caveats: tuple[dict, ...] = ()    # its caveats (reasons of kind "caveat"), in the page's order


@dataclass(frozen=True)
class Matched:
    """A pool found in the test's shared input."""
    pool: Pool                    # pooled again on the items found
    items: qc_readout.Items       # the items found, of pool.cells
    order: qc_rule.PoolOrder      # over items.cell_names indices
    counts: dict                  # {"target" | "matched" | "uniform": {"ok", "not_found", "unsure"}}
    input: str                    # the shared input's path
    seconds: float


def start_class(label: int, permille: int, cell: int, giver: int | None, strong: int) -> str:
    """Where a target sat when the cells started (the prior's label at its voxel, 0 none, -1 off the grid): NONE (no
    cell), OWN (its cell), GIVER_STRONG or GIVER_WEAK (its giver's, by the hold there: `strong` permille or more), OTHER
    (any other cell)."""
    if label <= 0:
        return "NONE"
    if label == cell + 1:
        return "OWN"
    if giver is not None and giver >= 0 and label == giver + 1:
        return "GIVER_STRONG" if permille >= strong else "GIVER_WEAK"
    return "OTHER"


def likely_rule() -> tuple[np.float32, float]:
    """The page's likely rule, read when used (engine/qc_scoring.py, the page's twin): (LIKELY_MIN as float32, the
    least π; LIKELY_W, the least w), LIKELY_W this module's until qc_scoring has its own."""
    return np.float32(qc_scoring.LIKELY_MIN), float(getattr(qc_scoring, "LIKELY_W", LIKELY_W))


def pattern_of(classes: list[str], all_same: bool) -> tuple[str | None, str | None, str | None]:
    """(pattern, hold, key) of a cell from its likely targets' start classes: the pattern is the class (the giver's,
    either hold) holding at least half of them, MIXED when none does or two do (exactly half each); hold, for the
    giver's pattern, STRONG unless more sat at the weak hold; the key is the pattern (GIVER_<hold> for the giver's),
    none when every source looks like the cell's own kind. All None without a likely target."""
    if not classes:
        return None, None, None
    merged = ["GIVER" if c.startswith("GIVER") else c for c in classes]
    top = [k for k in CLASSES if merged.count(k) * 2 >= len(merged)]
    pattern = top[0] if len(top) == 1 else "MIXED"
    hold = None
    if pattern == "GIVER":
        hold = "STRONG" if classes.count("GIVER_STRONG") >= classes.count("GIVER_WEAK") else "WEAK"
    key = None if all_same else (f"GIVER_{hold}" if pattern == "GIVER" else pattern)
    return pattern, hold, key


def _plain(value):
    """A worker answer as the page receives it: JSON's own types, NaN and infinities as null."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _codes(path) -> np.ndarray:
    """A stored .qc.bin: one byte per cell, a zstd frame (or raw), as the worker writes and reads it."""
    raw = Path(path).read_bytes()
    if raw[:4] == b"\x28\xb5\x2f\xfd":
        import zstandard                    # the engine's own (requirements.txt), as engine/qc_test.py reads them
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    return np.frombuffer(raw, np.uint8).copy()


def _void(records: np.ndarray) -> np.ndarray:
    """Records (RECORD) as one 16-byte value each, compared byte by byte."""
    return np.ascontiguousarray(records, RECORD).view(np.dtype((np.void, RECORD.itemsize)))


class _RecordIndex:
    """Each 16-byte record's lowest index and how many records hold those bytes, exactly as `_Context`'s
    np.unique(_void(records), return_index=True, return_counts=True) gives them, without sorting a whole sample's
    16-byte values: the records ordered by their place alone (x and y bits, `_xy`), file order kept within a place, and
    the bytes compared only among the records at the place asked. Exact: records with the same 16 bytes have the same x
    and y bits, so every copy of a record lies in its place's group; the stable order puts the lowest index first, the
    occurrence np.unique's return_index gives (its sort is stable too); and the group's matches are its copies."""

    def __init__(self, records: np.ndarray):
        xy = _xy(records["x"], records["y"])
        self.order = np.argsort(xy, kind="stable")       # by place, file order within one
        self.sorted = xy[self.order]
        self.records = records

    def find(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`_Context.find`'s answer: (the lowest index of each record's 16 bytes in the parent, how many parent records
        hold them). RuntimeError for a record the parent does not hold."""
        first, copies = np.zeros(len(records), np.int64), np.zeros(len(records), np.int64)
        if not len(records):
            return first, copies
        wanted = _void(records)
        xy = _xy(records["x"], records["y"])
        lo, hi = np.searchsorted(self.sorted, xy, "left"), np.searchsorted(self.sorted, xy, "right")
        for j in range(len(records)):
            held = self.order[lo[j]:hi[j]]                # the parent's records at its place, in file order
            same = held[_void(self.records[held]) == wanted[j]]
            if not same.size:
                raise RuntimeError("a suspect's transcript is not among your run's transcripts")
            first[j], copies[j] = same[0], same.size
        return first, copies


@dataclass(frozen=True)
class _Target:
    """One target of a cell: a glowing point of its where measurement (P = w x π >= GLOW_MIN, any owner), area
    aside."""
    point: int                    # its index among the measurement's points
    gene: str
    x: float                      # as the page has it: 2 decimals
    y: float
    record: int                   # the parent record (lowest index of its 16 bytes)
    copies: int                   # how many parent records hold those 16 bytes
    p: float                      # P, a float32 value
    likely: bool                  # π >= LIKELY_MIN (as float32) and w >= LIKELY_W
    near: bool
    giver: int | None             # the source its gene came from (parent cell number); None: unplaced
    start_class: str
    start_label: int              # the prior's label there (0 none, -1 off the grid)
    permille: int
    rank: int | None              # its depth rank among the measurement's points at its place (samePlace)
    of: int | None
    w: float = 1.0                # proseg's share of it for the cell (the where reply's, 3 decimals)
    pi: float = 0.0               # π, the chance it came from a neighbour (a float32 value)
    owner: int = 1                # who holds its place in the drawing (the where reply's owner code: 1 the cell)


@dataclass(eq=False)
class _Answer:
    """One cell's answer at F, whatever the area: Go deeper's, the page's scoring of it, its targets and controls."""
    cell: int
    code: int                     # the stored byte
    state: str | None             # Go deeper's answer's state
    request: bool                 # the page made a where request
    sources: tuple[int, ...]      # its sources (parent cell numbers)
    same: tuple[bool, ...]        # each source looks like the cell's own kind
    all_same: bool
    targets: tuple[_Target, ...]  # in point order
    pattern: str | None
    hold: str | None
    key: str | None
    likely: int
    line: dict                    # its likely targets by where they sat at the start:
                                  # {likely, own, none, neighbour, other}
    matched: dict                 # point -> its matched controls' records (likely, placed targets of a mixed cell)
    short: int                    # matched controls not found (fewer own records than asked)
    uniform: tuple[int, ...]      # the cell's own RNA (cells with a target)
    excluded: tuple[str, ...]     # the genes its controls leave out: asked as a neighbour's, or tagged as one
    seconds: float
    held: str | None = None       # how its where reply placed w: "exact" | "approx" | "none" (None: no reply)
    owned: int = 0                # its records with w > 0 in the test area it was worked out for (the controls' own)
    w: dict = field(default_factory=dict)     # each control's w (record -> 3 decimals)
    count: str = "averaged"       # the count it was worked out on: "averaged" (e, the test's) or "sampled" (f, Go
                                  # deeper's at a sampled F: QcPools._cell)

    def acting(self) -> list[_Target]:
        """Its targets that act as suspects: the likely ones with a placed giver, else (weak) every glowing one with a
        placed giver; in point order."""
        placed = [t for t in self.targets if t.giver is not None]
        return [t for t in placed if t.likely] or placed

    @property
    def weak(self) -> bool:
        """It glows with a placed giver, but none of those is likely: its glowing points act as its suspects."""
        placed = [t for t in self.targets if t.giver is not None]
        return bool(placed) and not any(t.likely for t in placed)

    def public(self, names, mixed: bool) -> dict:
        """As `answers` gives it (JSON's own types)."""
        acting = {t.point for t in self.acting()}
        return {"cell": self.cell, "name": names[self.cell], "code": self.code, "judged": self.code != NOT_JUDGED,
                "mixed": bool(mixed), "state": self.state, "request": self.request, "sources": list(self.sources),
                "same": list(self.same), "all_same": self.all_same, "pattern": self.pattern, "hold": self.hold,
                "key": self.key, "likely": self.likely, "line": dict(self.line), "weak": self.weak,
                "held": self.held, "owned": self.owned,
                "targets": [{"point": t.point, "gene": t.gene, "x": t.x, "y": t.y, "record": t.record,
                             "copies": t.copies, "P": t.p, "pi": t.pi, "w": t.w, "owner": t.owner, "likely": t.likely,
                             "near": t.near, "giver": t.giver, "start_class": t.start_class,
                             "start_label": t.start_label, "permille": t.permille, "rank": t.rank, "of": t.of,
                             "acting": t.point in acting,
                             "primary": bool(mixed and t.likely and t.giver is not None)} for t in self.targets],
                "controls": {"matched": [{"point": point, "records": list(records)}
                                         for point, records in self.matched.items()],
                             "uniform": list(self.uniform), "short": self.short, "excluded": list(self.excluded),
                             "w": {str(r): v for r, v in self.w.items()}},
                "seconds": round(self.seconds, 4)}


@dataclass(eq=False)
class _Facts:
    """What the cheap tier found: the reasons (of every kind), the row, and what the frame work needs (F, the model,
    the settings, F's stored bytes and centres, the area), and the count F is read on (`count`: "sampled", the count
    proseg saves, for F proseg counted for a pause, the hold, a finish or the end; its bytes on that count:
    `codes_path_f`)."""
    run_id: str
    start: str
    row: tuple[str, str] = ("paired", "as_is")
    seq: int | None = None
    iter: int | None = None
    frame: dict | None = None
    model: dict | None = None
    thresholds: dict | None = None
    codes: np.ndarray | None = None
    codes_path: Path | None = None
    codes_path_f: Path | None = None
    count: str = "averaged"
    cx: np.ndarray | None = None
    cy: np.ndarray | None = None
    area: Area | None = None
    name: str | None = None
    reasons: list = field(default_factory=list)
    failure: str | None = None

    def key(self) -> tuple:
        """The frame context's key: (run, F, model, settings, the constants' id, F's count: a pool worked out before F
        was counted again for a pause or the hold is worked out again after it)."""
        th = self.thresholds or {}
        return (self.run_id, self.seq, (self.model or {}).get("id"), tuple(sorted(th.items())),
                qc_rule.CONSTANTS_ID, self.count)


class _Stopped(Exception):
    """The worker stops (its run forgotten, the app closing): its waiters raise `error`."""

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error


@dataclass(eq=False)
class _Outcome:
    """A pool request worked out: its refusals (none: `pool` is offered), the reply's cell fields, its timings, and the
    cell's and the pool's own caveats (the seed's pattern, too_few; the run's and the check's are added as asked)."""
    reasons: list
    pool: Pool | None
    fields: dict
    one_time_ms: float = 0.0
    cells: int = 0
    caveats: list = field(default_factory=list)


@dataclass(eq=False)
class _Job:
    """A request to a run's worker: a pool (`cell`), or every answer of `cells` (the `answers` hook), in a test area
    (`area`: the pool's, else the one `answers` was given)."""
    key: tuple
    facts: _Facts
    cell: int | None = None
    cells: list | None = None
    area: Area | None = None
    done: threading.Event = field(default_factory=threading.Event)
    outcome: object = None
    dropped: bool = False         # replaced by a newer request before it began
    one_time_ms: float = 0.0
    computed: int = 0
    basis: _Basis | None = None   # where the area's w came from (QcPools._compute)


@dataclass(frozen=True, eq=False)
class _Basis:
    """Where proseg's share w of each transcript comes from for a test area (`QcPools._basis`): "exact" (proseg's
    export of the area: `where`, its record with "path"), "approx" (the frame's own per-layer labels: a dense frame's
    whole, or a tiled frame's `window` (description, layers) over the area, or over the part of it it holds when the
    parent cannot be asked) or "none"; at the frame `seq` (H: F, or the parent's state now), whose record is `frame`;
    with the caveat it carries (approx_targets, targets_now, or None)."""
    held: str
    seq: int
    frame: dict
    where: dict | None = None
    window: tuple | None = None
    caveat: str | None = None


def _ranges(order: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """order[lo[0]:hi[0]], order[lo[1]:hi[1]], ... in one array."""
    lengths = (hi - lo).astype(np.int64)
    total = int(lengths.sum())
    if not total:
        return np.zeros(0, np.int64)
    starts = np.repeat(lo.astype(np.int64) - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    return order[starts + np.arange(total)].astype(np.int64)


class _Weights:
    """Proseg's share w of each record for each cell over one test area (engine/held.py weights), each record's bucket
    or owner found once: exact from the area's export (a cell's candidates are the records of every bucket it holds
    pools of, whole, so |B| is the run's own), approx from the basis frame's per-layer labels (a cell's records are
    those its label holds at their observed voxel), none (no record). `of(cell, q_row)` -> (the cell's records with
    w > 0, their w)."""

    def __init__(self, held_by: str, records: np.ndarray, z_origin: float, z_spacing: float):
        self.held, self.records, self.z_origin, self.z_spacing = held_by, records, z_origin, z_spacing
        self.export = self.grid = self.layers = None

    @classmethod
    def exact(cls, records: np.ndarray, export: held.WhereExport, z_origin: float, z_spacing: float) -> "_Weights":
        self = cls("exact", records, z_origin, z_spacing)
        self.export = export
        self.grid = {"x0": export.x0, "y0": export.y0, "voxel": export.voxel, "nx": export.nx, "ny": export.ny,
                     "nz": export.nz}
        v = float(export.voxel)
        # Only the records of the span's columns and rows (one more each side for float32 places) are bucketed.
        near = _in_box(records, [export.x0 + (export.i0 - 1) * v, export.y0 + (export.j0 - 1) * v,
                                 export.x0 + (export.i1 + 2) * v, export.y0 + (export.j1 + 2) * v])
        key = held.buckets(records[near], self.grid, z_origin, z_spacing,
                           (export.i0, export.j0, export.i1, export.j1))
        keep = key >= 0
        near, key = near[keep], key[keep]
        by = np.argsort(key, kind="stable")
        self.order, self.sorted = near[by], key[by]
        nx, ny, nz = int(export.nx), int(export.ny), int(export.nz)
        entry = ((export.gene * nz + export.k) * ny + export.j) * nx + export.i
        by = np.argsort(export.dest, kind="stable")
        self.entry_key, self.entry_dest = entry[by], export.dest[by]
        return self

    @classmethod
    def approx(cls, records: np.ndarray, grid: dict, layers: tuple, z_origin: float,
               z_spacing: float) -> "_Weights":
        """`layers` = (labels [nz, h, w], c0, r0): the basis frame's per-layer labels over the columns c0.. and rows
        r0.. (a dense frame's whole: 0, 0); a record outside them is no cell's."""
        from ..analysis.worker import grid_columns
        self = cls("approx", records, z_origin, z_spacing)
        labels, c0, r0 = layers
        labels = np.asarray(labels)
        self.grid, self.layers = dict(grid), (labels, int(c0), int(r0))
        nz, h, w = labels.shape
        v = float(grid["voxel"])
        near = _in_box(records, [float(grid["x0"]) + (int(c0) - 1) * v, float(grid["y0"]) + (int(r0) - 1) * v,
                                 float(grid["x0"]) + (int(c0) + w + 1) * v, float(grid["y0"]) + (int(r0) + h + 1) * v])
        i, j, on = grid_columns(np, records["x"][near], records["y"][near], float(grid["x0"]), float(grid["y0"]), v,
                                int(grid["nx"]), int(grid["ny"]))
        i, j = i.astype(np.int64) - int(c0), j.astype(np.int64) - int(r0)
        on &= (i >= 0) & (i < w) & (j >= 0) & (j < h)
        k = np.clip(np.floor((records["z"][near][on] - float(z_origin)) / float(z_spacing)), 0, nz - 1).astype(np.int64)
        owner = labels[k, j[on], i[on]].astype(np.int64)
        held_ = near[on][owner > 0]
        owner = owner[owner > 0]
        by = np.argsort(owner, kind="stable")
        self.order, self.sorted = held_[by], owner[by]
        return self

    @classmethod
    def none(cls, records: np.ndarray, z_origin: float, z_spacing: float) -> "_Weights":
        return cls("none", records, z_origin, z_spacing)

    def of(self, cell: int, q_row) -> tuple[np.ndarray, np.ndarray]:
        """(the cell's records with w > 0, ascending; their w, float64), cell = label - 1."""
        empty = np.zeros(0, np.int64), np.zeros(0, np.float64)
        label = int(cell) + 1
        if self.held == "exact":
            lo, hi = np.searchsorted(self.entry_dest, label, "left"), np.searchsorted(self.entry_dest, label, "right")
            keys = np.unique(self.entry_key[lo:hi])
            found = _ranges(self.order, np.searchsorted(self.sorted, keys, "left"),
                            np.searchsorted(self.sorted, keys, "right"))
            if not found.size:
                return empty
            found = np.sort(found)
            w, _ = held.weights(self.records[found], cell, q_row, self.grid, self.z_origin, self.z_spacing,
                                export=self.export)
        elif self.held == "approx":
            lo, hi = np.searchsorted(self.sorted, label, "left"), np.searchsorted(self.sorted, label, "right")
            found = np.sort(self.order[lo:hi])
            if not found.size:
                return empty
            w, how = held.weights(self.records[found], cell, q_row, self.grid, self.z_origin, self.z_spacing,
                                  layers=self.layers)
            if how != "approx":
                raise RuntimeError("a cell's transcripts lie outside the labels of its frame")
        else:
            return empty
        keep = w > 0
        return found[keep], w[keep]


def _in_box(records: np.ndarray, box) -> np.ndarray:
    """The indices of the records inside [x0, y0, x1, y1] µm (half-open), ascending."""
    x, y = records["x"], records["y"]
    return np.flatnonzero((x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])).astype(np.int64)


def on_grid(box, frame: dict) -> tuple | None:
    """The part of `box` (µm) a frame's grid covers (its x0, y0, nx x ny columns of `voxel`), None where it covers
    none of it."""
    x0, y0, v = float(frame["x0"]), float(frame["y0"]), float(frame["voxel"])
    x1, y1 = x0 + int(frame["nx"]) * v, y0 + int(frame["ny"]) * v
    found = (max(float(box[0]), x0), max(float(box[1]), y0), min(float(box[2]), x1), min(float(box[3]), y1))
    return found if found[2] > found[0] and found[3] > found[1] else None


def _window_over(run, frame: dict, box, part: bool = False) -> tuple | None:
    """A tiled frame's per-layer window (its description {i0, j0, nx, ny, nz} and layers [nz, ny, nx]) when it holds
    every column and row of `box` (µm) on the frame's grid (with `part`: any column and row of it), else None (no
    window, or one elsewhere: proseg-live keeps the layers of the view it last showed only)."""
    from . import tiles
    container = tiles.Container(run.live / frame["file"])
    info = container.header.get("window")
    if not info:
        return None
    x0, y0, v = float(frame["x0"]), float(frame["y0"]), float(frame["voxel"])
    nx, ny = int(frame["nx"]), int(frame["ny"])
    c0, r0 = max(0, math.floor((box[0] - x0) / v)), max(0, math.floor((box[1] - y0) / v))
    c1, r1 = min(nx - 1, math.floor((box[2] - x0) / v)), min(ny - 1, math.floor((box[3] - y0) / v))
    i0, j0 = int(info["i0"]), int(info["j0"])
    i1, j1 = i0 + int(info["nx"]) - 1, j0 + int(info["ny"]) - 1
    if part:
        if not (i0 <= c1 and c0 <= i1 and j0 <= r1 and r0 <= j1):
            return None
    elif not (i0 <= c0 and c1 <= i1 and j0 <= r0 and r1 <= j1):
        return None
    return container.window()


def _records(live: Path, ngenes: int) -> np.ndarray:
    """The run's transcripts (live/transcripts.bin.gz, RECORD: x, y, z, gene), as proseg-live wrote them."""
    records = np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORD)
    if records.size and int(records["gene"].max()) >= ngenes:
        raise RuntimeError("the run's transcripts name more genes than it has")
    return records


class _Work:
    """One run's worker: its thread, the request it works on and the one waiting, its frame context."""

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.pending: _Job | None = None
        self.current: _Job | None = None
        self.context = None
        self.used = 0.0
        self.stop: Exception | None = None


class _Context:
    """One run's frame work for a test at F (see the module docstring), done once at the basis' frame H (F, or the
    parent's state now): proseg's count of every cell there, with what every cell's answer reads beside it, and for
    each test area asked (the newest AREAS_KEPT) its own basis (`bases`: its export, when exact) and its w (`held`)."""

    def __init__(self, key: tuple, run, facts: _Facts, basis: _Basis):
        began = time.perf_counter()
        frame = basis.frame
        if frame.get("format", "dense") != "dense":
            raise RuntimeError("only a detail run's frames can be tested")
        self._count(key, run, facts, basis)
        try:
            info, live = run.info, run.live
            # Each distinct 16-byte record: its lowest index and how many hold it (a target's record by its bytes).
            self.keys, self.first, self.copies = np.unique(_void(self.records), return_index=True, return_counts=True)
            # The starting cells: u32 label then u32 permille per voxel of the prior's grid.
            prior = info.get("prior") or {}
            if not prior.get("file") or prior.get("format", "dense") != "dense":
                raise RuntimeError("this run's starting cells are not stored per layer")
            self.pnx, self.pny, self.pnz = int(prior["nx"]), int(prior["ny"]), int(prior.get("nz", 1))
            self.px0, self.py0, self.pvoxel = float(prior["x0"]), float(prior["y0"]), float(prior["voxel"])
            values = _decode(live / prior["file"])
            size = self.pnx * self.pny * self.pnz
            if values.size != 2 * size:
                raise RuntimeError("the run's starting cells file is incomplete")
            self.start_label, self.start_permille = values[:size], values[size:]
            # H's per-layer labels, and each cell's box of columns, for the footprints (and approx's w).
            fnx, fny, fnz = int(frame["nx"]), int(frame["ny"]), int(frame["nz"])
            values = _decode(live / frame["file"])
            plane = fnx * fny
            if values.size < (2 + fnz) * plane:
                raise RuntimeError("the run's last frame is incomplete")
            self.layers = values[2 * plane:(2 + fnz) * plane].reshape(fnz, fny, fnx)
            self.boxes = _label_boxes(self.layers, self.ncells)
            self.feet: dict = {}
        except BaseException:
            self.close()
            raise
        self.ms = (time.perf_counter() - began) * 1e3

    def _count(self, key: tuple, run, facts: _Facts, basis: _Basis):
        """What either kind of frame context reads first: proseg's count at H through the worker (Worker.count on
        H's counts file, as Go deeper's answers read it; its numpy CSR or dense matrix of the averaged count, and the
        file read: e and n for each cell's keep), the run's records and names, F's stored bytes, centres and verdicts,
        and Go deeper's block as the engine sends it (engine/analysis.py) with the stored byte of F (the card the user
        saw), never a start line or the neighbours' classes as last shown."""
        from ..analysis import worker as W
        from .analysis import DEEPER_V
        info, live, frame = run.info, run.live, basis.frame
        self.key, self.facts_key, self.basis = key, facts.key(), basis
        self.run_id, self.seq, self.iter = facts.run_id, int(basis.seq), frame.get("iter")
        self.f_seq = int(facts.seq)
        ncells = self.ncells = int(info["cells"])
        if not frame.get("counts_file"):
            raise RuntimeError("this iteration has no count of proseg's own")
        z = info["z"]
        self.request = {"op": "analyze", "ops": ["deeper"], "run": facts.run_id, "seq": self.seq,
                        "iter": frame.get("iter"), "frame": str(live / frame["file"]),
                        "format": frame.get("format", "dense"), "nx": frame["nx"], "ny": frame["ny"],
                        "nz": frame["nz"], "x0": frame["x0"], "y0": frame["y0"], "voxel": frame["voxel"],
                        "z_origin": z["grid_origin"], "z_spacing": z["grid_spacing"],
                        "counts_file": str(live / frame["counts_file"]),
                        "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
                        "ncells": ncells,
                        "cells_file": str(live / frame["cells_file"]) if frame.get("cells_file") else None}
        thresholds = facts.thresholds
        self.block = {"model": str(facts.model["path"]), "model_id": facts.model["id"], "start": False,
                      "k": int(qc_rule.MEASURED["deeper_k"]), "fit_min": thresholds["fit_min"],
                      "judge_min": thresholds["judge_min"], "codes": str(facts.codes_path),
                      "cell_ids": str(live / "cells.txt"), "v": DEEPER_V}
        self.worker = W.Worker(xp=np)
        self.held: OrderedDict = OrderedDict()           # area box -> its _Weights
        self.bases: OrderedDict = OrderedDict()          # area box -> its _Basis (this context's frame and kind)
        try:
            self.rs = self.worker.state(self.request)
            self.counts, self.counted = self.worker.count(self.rs, self.request)
            self.worker.keep_codes(str(facts.codes_path), ncells, facts.codes)     # the bytes read once, here too
            self.genes = list(self.rs.genes)
            self.gene_names = np.asarray(self.genes, dtype=object)
            self.records = _records(live, len(self.genes))
            names = (live / "cells.txt").read_text().split("\n")[:ncells]
            if len(names) != ncells:
                raise RuntimeError(f"cells.txt names {len(names)} cells, the run has {ncells}")
            self.names = names
            self.z_origin, self.z_spacing = float(z["grid_origin"]), float(z["grid_spacing"])
            self.fx0, self.fy0, self.fvoxel = float(frame["x0"]), float(frame["y0"]), float(frame["voxel"])
            self.cx, self.cy = facts.cx, facts.cy
            self.codes = facts.codes
            self.judged = self.codes != NOT_JUDGED
            self.mixed = self.judged & (self.codes >= W.code_at(float(thresholds["mixed_at"])))
        except BaseException:
            self.close()
            raise

    def sampled(self):
        """H's count proseg saves (f) as the worker's matrix (Worker.count on the same counts file with the request's
        count "sampled"), made once: Go deeper's own answer at a sampled F (QcPools._cell with count "sampled")."""
        found = getattr(self, "f_counts", None)
        if found is None:
            found = self.f_counts = self.worker.matrix(self.counted, "f")
        return found

    def q(self, cell: int, which: str = "e") -> np.ndarray:
        """The cell's keep at H per gene: e / n of proseg's count (0 where it assigns none), float64 [genes]; f / n with
        `which` "f" (the count proseg saves: a sampled F's)."""
        rows = counts_file.rows(self.counted, [int(cell)], (which, "n"))
        e, n = rows[which][0].astype(np.float64), rows["n"][0].astype(np.float64)
        return np.divide(e, n, out=np.zeros_like(e), where=n > 0)

    def weights(self, box: tuple, basis: _Basis) -> _Weights:
        """The w of the test area `box` from its own basis, worked out once and kept with that basis (the newest
        AREAS_KEPT areas)."""
        found = self.held.get(box)
        if found is not None:
            self.held.move_to_end(box)
            self.bases.move_to_end(box)
            return found
        if basis.held == "exact":
            found = _Weights.exact(self.records, held.read_where(basis.where), self.z_origin, self.z_spacing)
        elif basis.held == "approx":
            found = _Weights.approx(self.records, self.grid(), self.labels_for(basis), self.z_origin,
                                    self.z_spacing)
        else:
            found = _Weights.none(self.records, self.z_origin, self.z_spacing)
        self.held[box], self.bases[box] = found, basis
        while len(self.held) > AREAS_KEPT:
            self.held.popitem(last=False)
            self.bases.popitem(last=False)
        return found

    def grid(self) -> dict:
        """H's grid {x0, y0, voxel, nx, ny, nz}, as its record has it."""
        frame = self.basis.frame
        return {"x0": float(frame["x0"]), "y0": float(frame["y0"]), "voxel": float(frame["voxel"]),
                "nx": int(frame["nx"]), "ny": int(frame["ny"]), "nz": int(frame["nz"])}

    def labels_for(self, basis: _Basis) -> tuple:
        """The per-layer labels approx reads w from: (labels [nz, h, w], c0, r0): a dense frame's whole."""
        return self.layers, 0, 0

    def close(self):
        """Let go of the worker's threads (as the old test did) and the arrays."""
        worker, self.worker = getattr(self, "worker", None), None
        if worker is not None and worker.pool is not None:
            worker.pool.shutdown(wait=False)

    def find(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the lowest index of each record's 16 bytes in the parent, how many parent records hold them).
        RuntimeError for a record the parent does not hold."""
        if not len(records):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        wanted = _void(records)
        at = np.minimum(np.searchsorted(self.keys, wanted), len(self.keys) - 1)
        if not (self.keys[at] == wanted).all():
            raise RuntimeError("a suspect's transcript is not among your run's transcripts")
        return self.first[at].astype(np.int64), self.copies[at].astype(np.int64)

    def start_at(self, x, y, z) -> tuple[np.ndarray, np.ndarray]:
        """(label, permille) of the starting cells at each point (the worker's voxel_lookup on the prior's grid):
        label 0 for no cell, -1 off the grid; permille 0 there."""
        from ..analysis.worker import voxel_lookup
        index = voxel_lookup(np, np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32),
                             self.px0, self.py0, self.pvoxel, self.pnx, self.pny, self.pnz, self.z_origin,
                             self.z_spacing)
        inside = index < self.start_label.size
        label = np.full(index.shape, -1, np.int64)
        permille = np.zeros(index.shape, np.int64)
        label[inside] = self.start_label[index[inside]]
        permille[inside] = self.start_permille[index[inside]]
        permille[label <= 0] = 0
        return label, permille

    def near(self, giver: int, x, y) -> np.ndarray:
        """Whether each point lies within 2 µm of the giver's footprint at F (the columns where it owns any layer):
        deeper_where.near, the where reply's own rule, over the giver's box of columns."""
        found = self.feet.get(giver)
        if found is None:
            r0, c0, r1, c1 = (int(v) for v in self.boxes[giver + 1])
            if r0 < 0:
                found = (np.zeros((1, 1), bool), 0, 0)
            else:
                found = ((self.layers[:, r0:r1 + 1, c0:c1 + 1] == giver + 1).any(axis=0), c0, r0)
            self.feet[giver] = found
        mask, c0, r0 = found
        return near_footprint(np.asarray(x, np.float64), np.asarray(y, np.float64), mask, c0, r0, self.fx0, self.fy0,
                              self.fvoxel)


def _label_boxes(layers: np.ndarray, ncells: int) -> np.ndarray:
    """[ncells + 1, 4]: each label's (row0, col0, row1, col1) over every layer, inclusive; -1 where it owns nothing."""
    _, ny, nx = layers.shape
    flat = layers.reshape(-1)
    hit = np.flatnonzero((flat > 0) & (flat <= ncells))
    out = np.full((ncells + 1, 4), -1, np.int64)
    if not hit.size:
        return out
    labels = flat[hit].astype(np.int64)
    plane = hit % (ny * nx)
    rows, cols = plane // nx, plane % nx
    order = np.argsort(labels, kind="stable")
    labels, rows, cols = labels[order], rows[order], cols[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    present = labels[starts]
    out[present, 0] = np.minimum.reduceat(rows, starts)
    out[present, 1] = np.minimum.reduceat(cols, starts)
    out[present, 2] = np.maximum.reduceat(rows, starts)
    out[present, 3] = np.maximum.reduceat(cols, starts)
    return out


class _TiledContext(_Context):
    """A whole-sample (tiled) run's frame work, its positions read in the column basis (deeper_where's "column", as Go
    deeper and the where reply read such a run's footprints): proseg's count, records and verdicts as in `_Context`; a
    record's key comes from `_RecordIndex`; the starting cells are prior.tiles' label and permille per column; a
    giver's footprint is its consensus columns at H, composed over its own box; approx's w reads H's per-layer window
    (the basis'). Nothing whole-grid is composed, and no transcript is given to a cell by its column.

    Two box conventions meet here, never mixed: `_label_boxes` (the detail path's) gives (row0, col0, row1, col1), rows
    first, both ends included, -1 where a label owns nothing; deeper_where's Plane.boxes gives [c0, r0, c1, r1), columns
    first, the ends excluded, None where a label owns nothing, and Plane.window(c0, r0, c1, r1) and
    deeper_where.near(px, py, mask, c0, r0, ...) take columns first too. This class uses deeper_where's alone."""

    def __init__(self, key: tuple, run, facts: _Facts, basis: _Basis):
        from . import tiles
        from .deeper_where import Plane
        began = time.perf_counter()
        frame = basis.frame
        if frame.get("format") != "tiles":
            raise RuntimeError("only a whole-sample run's frames are read by their columns")
        self._count(key, run, facts, basis)
        try:
            info, live = run.info, run.live
            self.index = _RecordIndex(self.records)
            # The starting cells: prior.tiles' planes "label" and "permille" on the prior's 2-D grid.
            prior = info.get("prior") or {}
            if not prior.get("file") or prior.get("format") != "tiles":
                raise RuntimeError("this run's starting cells are not stored per column")
            self.pnx, self.pny = int(prior["nx"]), int(prior["ny"])
            self.px0, self.py0, self.pvoxel = float(prior["x0"]), float(prior["y0"]), float(prior["voxel"])
            self.prior = tiles.Container(live / prior["file"])
            if not {"label", "permille"} <= set(self.prior.header["planes"]) or \
                    (self.prior.nx, self.prior.ny) != (self.pnx, self.pny):
                raise RuntimeError("the run's starting cells file is incomplete")
            # H's consensus plane as the where reply reads it, a cell's columns found from H's cell statistics.
            container = tiles.Container(live / frame["file"])
            if "consensus" not in container.header["planes"] or \
                    (container.nx, container.ny) != (int(frame["nx"]), int(frame["ny"])):
                raise RuntimeError("the run's last frame is incomplete")
            self.plane = Plane("column", container, "consensus", self.fx0, self.fy0, self.fvoxel,
                               DeeperWhere._cellstats(run, frame, self.ncells))
            self.feet: dict = {}
        except BaseException:
            self.close()
            raise
        self.ms = (time.perf_counter() - began) * 1e3

    def labels_for(self, basis: _Basis) -> tuple:
        """approx's labels: H's per-layer window (the basis': over the test area, or the part of it it holds; a record
        outside it is no cell's)."""
        if basis.window is None:
            raise RuntimeError("a whole-sample frame without its window has no layers to place transcripts by")
        info, layers = basis.window
        return layers, int(info["i0"]), int(info["j0"])

    def find(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`_Context.find`'s answer, from `_RecordIndex`."""
        return self.index.find(records)

    def start_at(self, x, y, z) -> tuple[np.ndarray, np.ndarray]:
        """(label, permille) of the starting cells at each point: prior.tiles at its column (the worker's grid_columns
        on the prior's grid, in float32 as its tile_lookup; the plane is 2-D, so z is not read), composed over the
        points' box of columns: label 0 for no cell, -1 off the grid; permille 0 there."""
        from ..analysis.worker import grid_columns
        i, j, inside = grid_columns(np, np.asarray(x, np.float32), np.asarray(y, np.float32), self.px0, self.py0,
                                    self.pvoxel, self.pnx, self.pny)
        label = np.full(inside.shape, -1, np.int64)
        permille = np.zeros(inside.shape, np.int64)
        if inside.any():
            i, j = i[inside].astype(np.int64), j[inside].astype(np.int64)
            c0, r0, c1, r1 = int(i.min()), int(j.min()), int(i.max()) + 1, int(j.max()) + 1
            label[inside] = self.prior.compose("label", c0, r0, c1, r1)[j - r0, i - c0]
            permille[inside] = self.prior.compose("permille", c0, r0, c1, r1)[j - r0, i - c0]
        permille[label <= 0] = 0
        return label, permille

    def near(self, giver: int, x, y) -> np.ndarray:
        """Whether each point lies within 2 µm of the giver's footprint at F (its columns of F's consensus plane):
        deeper_where.near, the where reply's own rule and footprint, over the giver's box of columns [c0, r0, c1, r1) as
        Plane.boxes finds it (exact: from F's cell statistics, else by scanning every tile)."""
        found = self.feet.get(giver)
        if found is None:
            box = self.plane.boxes([giver + 1])[giver + 1]
            if box is None:
                found = (np.zeros((1, 1), bool), 0, 0)
            else:
                c0, r0, c1, r1 = box
                found = (self.plane.window(c0, r0, c1, r1)[0] == giver + 1, c0, r0)
            self.feet[giver] = found
        mask, c0, r0 = found
        return near_footprint(np.asarray(x, np.float64), np.asarray(y, np.float64), mask, c0, r0, self.fx0, self.fy0,
                              self.fvoxel)


def _input_rule(options: dict) -> tuple[float, re.Pattern | None]:
    """proseg's input rule for a run of these options (min qv; the excluded genes' pattern, proseg's own search): the
    test runs' and the parent's (engine/runs.py passes proseg_options.input_args after the options)."""
    from . import proseg_options
    min_qv = float(options.get("min-qv", 20.0))
    if "excluded-genes" in options:
        pattern = options["excluded-genes"]
    else:
        given = proseg_options.input_args(options)
        pattern = given[given.index("--excluded-genes") + 1] if "--excluded-genes" in given else None
    return min_qv, None if pattern is None else re.compile(str(pattern))


def _xy(x, y) -> np.ndarray:
    """x and y float32 bits as one u64: a place, exactly."""
    xb = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    yb = np.asarray(y, np.float32).view(np.uint32).astype(np.uint64)
    return (xb << np.uint64(32)) | yb


def _groups(gene: np.ndarray, xy: np.ndarray, want_gene: np.ndarray, want_xy: np.ndarray):
    """For each wanted (gene, place): the indices of (gene, xy) holding it, in index order, as (order, lo, hi): the
    holders of wanted j are order[lo[j]:hi[j]]."""
    keys = np.empty(len(gene), [("g", "<i8"), ("xy", "<u8")])
    keys["g"], keys["xy"] = gene, xy
    order = np.lexsort((np.arange(len(gene)), xy, gene))
    keys = keys[order]
    wanted = np.empty(len(want_gene), keys.dtype)
    wanted["g"], wanted["xy"] = want_gene, want_xy
    return order, np.searchsorted(keys, wanted, "left"), np.searchsorted(keys, wanted, "right")


def _tied(values: np.ndarray) -> bool:
    """Whether two of these depths lie within TIE_Z of each other."""
    ordered_ = np.sort(np.asarray(values, np.float64))
    return bool(ordered_.size > 1 and (np.diff(ordered_) <= TIE_Z).any())


# ---- numpy's BLAS on one thread --------------------------------------------------------------------------------------
_BLAS_HOW = None            # how numpy's BLAS is told its threads, found once: threadpoolctl's controller, the
                            # (get, put) of numpy's own OpenBLAS, or False (it cannot be told)


def _wheel_openblas():
    """(get, put) of the thread number of the OpenBLAS numpy's own wheel brings (numpy.libs, .dylibs), found among the
    libraries already loaded (never loaded here); None without one (another BLAS, or no RTLD_NOLOAD on this system)."""
    import ctypes
    if not hasattr(os, "RTLD_NOLOAD"):
        return None
    root = Path(np.__file__).resolve().parent
    for path in sorted([*root.parent.glob("numpy.libs/*openblas*"), *root.glob(".dylibs/*openblas*")]):
        try:
            lib = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue
        for name in ("scipy_openblas", "openblas"):          # numpy 2's scipy-openblas, an older numpy's own
            for suffix in ("64_", ""):                      # its 64-bit integer build, or not
                get = getattr(lib, f"{name}_get_num_threads{suffix}", None)
                put = getattr(lib, f"{name}_set_num_threads{suffix}", None)
                if get is not None and put is not None:
                    get.restype, get.argtypes = ctypes.c_int, []
                    put.restype, put.argtypes = None, [ctypes.c_int]
                    return get, put
    return None


def _blas_one_now():
    """numpy's BLAS set to one thread now: the call that sets back what it found, None where BLAS cannot be told.
    threadpoolctl where it is installed (as analysis/worker.py one_blas_thread), else numpy's own OpenBLAS (the app's
    own venv has no threadpoolctl)."""
    global _BLAS_HOW
    if _BLAS_HOW is None:
        try:
            # Optional, so asked for by name: the engine never needs it (requirements.txt lists what it needs).
            _BLAS_HOW = importlib.import_module("threadpoolctl").ThreadpoolController()
        except Exception:                                   # not installed
            _BLAS_HOW = _wheel_openblas() or False
    if _BLAS_HOW is False:
        return None
    if isinstance(_BLAS_HOW, tuple):
        get, put = _BLAS_HOW
        before = int(get())
        put(1)
        return lambda: put(before)
    return _BLAS_HOW.limit(limits=1, user_api="blas").restore_original_limits


class _OneBlas:
    """A block in which numpy's BLAS uses one thread, shared by every run's worker: the first in sets one thread, the
    last out sets back what the first found. The recipe's matrices are small, and BLAS's own threads (started with
    numpy, at the engine's priority: the worker's lower one never reaches them) only spin on a shared computer: three
    real 500 µm pools and their matching took the same 2.3 s with one thread, and 7.6 s of CPU instead of 33 s. Nothing
    is set where BLAS cannot be told, and a failure to tell it never stops a pool."""

    def __init__(self):
        self.lock = threading.Lock()
        self.users = 0
        self.undo = None

    def __enter__(self):
        with self.lock:
            if self.users == 0:
                try:
                    self.undo = _blas_one_now()
                except Exception:                           # the threads stay as they are
                    self.undo = None
            self.users += 1
        return self

    def __exit__(self, *exc):
        with self.lock:
            self.users -= 1
            undo = self.undo if self.users == 0 else None
            if undo is not None:
                self.undo = None
                try:
                    undo()
                except Exception:
                    pass
        return False


ONE_BLAS = _OneBlas()


# ---- the engine's side -----------------------------------------------------------------------------------------------
class QcPools:
    """See the module docstring. `runs`: Runs; `qc`: QcManager (describe, qc_model, cells_path, configure); `analysis`:
    the scheduler (qc_thresholds, available); `where`: Go deeper's DeeperWhere (measure). `clock` times the frame
    contexts' hold and the replies' ms. `computed` counts the cells worked out and `contexts` the frame contexts made
    (tests)."""

    def __init__(self, runs, qc, analysis, where, *, clock=time.monotonic):
        self.runs, self.qc, self.analysis, self.clock = runs, qc, analysis, clock
        self.where = where if where is not None else DeeperWhere(runs)
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.closed = False
        self.work: dict[str, _Work] = {}
        self.answers_kept: OrderedDict = OrderedDict()     # context key + (cell,) -> _Answer
        self.pools_kept: OrderedDict = OrderedDict()       # pool key -> _Outcome
        self.asked: set = set()                            # (run, F, model) the check was asked to score
        self.names: OrderedDict = OrderedDict()            # (cells.txt, mtime) -> its names
        self.computed = 0
        self.contexts = 0

    # ---- the API -----------------------------------------------------------------------------------------------------
    def preview(self, run_id: str, cell: int, *, wait: float = 0.0) -> dict:
        """GET /runs/{id}/qc/test/pool without busy and other_run (engine/qc_test.py adds them): the pool reply,
        `reasons` its refusals (`offer` true without one, also while `waits` hold), `waits`, the `caveats` known so far
        and the `row`. The cheap reasons come at once; once no refusal or wait holds, the frame work goes on in the
        run's worker, and the reply waits for it at most `wait` seconds ("working" until then: the page asks again).
        RunError 404 for an unknown run or cell or a run that is not proseg's; QcError 503 while the app closes."""
        began = self.clock()
        run, start = self._run(run_id, cell)
        facts = self._cheap(run, cell, start)
        refusals, waits, caveats = split(facts.reasons)
        reply = self._reply(run, cell, facts, refusals, waits, caveats)
        outcome, one_time, cells = None, 0.0, 0
        if facts.failure is not None:
            reply.update(state="failed", message=f"{FAILED}: {facts.failure}")
        elif not refusals and not waits:
            key = self._pool_key(facts, cell)
            outcome = self._kept(key)
            if outcome is None:
                job = self._submit(run_id, _Job(key, facts, cell=cell))
                if wait > 0:
                    job.done.wait(min(float(wait), WAIT_MAX_S))
                if job.done.is_set() and not job.dropped:
                    outcome = self._settled(job.outcome)
                    if isinstance(outcome, _Outcome):
                        one_time, cells = outcome.one_time_ms, outcome.cells
            if outcome is None:
                reply.update(state="working")
            elif isinstance(outcome, str):
                reply.update(state="failed", message=outcome)
            else:                                # a copy: the kept outcome's dicts never reach a caller
                reply.update(copy.deepcopy(outcome.fields), offer=not outcome.reasons,
                             reasons=copy.deepcopy(outcome.reasons),
                             caveats=ordered(copy.deepcopy([*caveats, *outcome.caveats])))
        reply["ms"] = {"total": round((self.clock() - began) * 1e3, 1), "one_time": round(one_time, 1),
                       "cells": cells}
        return reply

    def pool(self, run_id: str, cell: int, *, timeout: float = 120.0, check: Callable[[], None] | None = None) -> Pool:
        """The cell's pool, worked out (blocking, single-flight with the previews), its caveats every one that holds
        (the run's and the check's as they are now, the cell's and the pool's): OutsideDomain with the refusals that
        hold; NotYet while a wait holds (and no refusal); RunError 404 for an unknown run or cell; StillWorking (503)
        past `timeout`, the pool still being worked out; QcError 503 while the app closes, 500 when it could not be
        worked out; `check()` is called while it waits and may raise to stop waiting."""
        run, start = self._run(run_id, cell)
        facts = self._cheap(run, cell, start)
        if facts.failure is not None:
            raise QcError(f"{FAILED}: {facts.failure}", 500)
        refusals, waits, caveats = split(facts.reasons)
        if refusals:
            raise OutsideDomain(refusals)
        if waits:
            raise NotYet(waits)
        key = self._pool_key(facts, cell)
        deadline = time.monotonic() + timeout
        while True:
            outcome = self._kept(key)
            if outcome is not None:
                break
            job = self._submit(run_id, _Job(key, facts, cell=cell))
            while not job.done.wait(POLL_S):
                if check is not None:
                    check()
                if time.monotonic() > deadline:
                    raise StillWorking()
            if not job.dropped:              # else replaced by a newer request before it began: asked again
                outcome = self._settled(job.outcome)
                break
        if isinstance(outcome, str):
            raise QcError(outcome, 500)
        if outcome.reasons:
            raise OutsideDomain(outcome.reasons)
        return replace(outcome.pool, caveats=tuple(ordered(copy.deepcopy([*caveats, *outcome.caveats]))))

    def prerequisites(self, run_id: str) -> tuple[list[dict], list[dict]]:
        """(refusals, waits) of the run itself, its mixing check and its last frame F's scoring (the check asked once
        per (run, F, model) to score it, as for a preview), without a cell: what a waiting test (engine/qc_test.py)
        asks every second. Reads no frame file. RunError 404 for an unknown run or one that is not proseg's; QcError
        503 while the app closes, 500 when it could not be worked out."""
        from . import qc_rules
        if self.closed:
            raise QcError("The app is closing", 503)
        run = self.runs.get(run_id)
        if run.meta.get("kind", "proseg") != "proseg":
            raise RunError("Not a proseg run", 404)
        facts = self._prerequisites(run, qc_rules.start_of(run.meta))
        if facts.failure is not None:
            raise QcError(f"{FAILED}: {facts.failure}", 500)
        refusals, waits, _ = split(facts.reasons)
        return refusals, waits

    def match_input(self, pool: Pool, input_path: Path, options: dict, *, check=None) -> Matched:
        """`pool`'s items found in the shared input (with the parent's own input rule, `options`: min qv and the
        excluded genes), pooled again on what was found (its too_few caveat said again with the suspects found, or
        dropped once they reach k): TooFewFound when none of the clicked cell's suspects is found (a pool under k is
        tested with all it has). An item is found by its gene and the float32 bits of x and y: none is "not_found",
        one is it; several (one place, several depths) are told apart by depth rank only when the parent holds as many
        there, each side's depths more than TIE_Z apart ("unsure" otherwise). A matched control goes with its target:
        one whose target was not found or is unsure is counted with it."""
        began = time.perf_counter()
        if check is not None:
            check()
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        try:
            table = pq.read_table(input_path, columns=INPUT_COLUMNS)
        except (OSError, ValueError) as exc:
            raise ValueError(f"the test's input could not be read ({exc})") from None
        if check is not None:
            check()
        gene = table.column("feature_name").combine_chunks()
        if not hasattr(gene, "dictionary"):
            gene = pc.dictionary_encode(gene)
        dictionary = [str(v) for v in gene.dictionary.to_pylist()]
        codes = gene.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        min_qv, excluded = _input_rule(options)
        qv = table.column("qv").to_numpy().astype(np.float32)
        keep = ~(qv < np.float32(min_qv))           # proseg: qv < min_qv leaves a transcript out
        if excluded is not None:
            dropped = np.array([bool(excluded.search(name)) for name in dictionary] or [False], bool)
            keep &= ~dropped[codes]
        rows = np.flatnonzero(keep)
        x = table.column("x_location").to_numpy().astype(np.float32)[rows]
        y = table.column("y_location").to_numpy().astype(np.float32)[rows]
        z = table.column("z_location").to_numpy().astype(np.float64)[rows]
        ids = table.column("transcript_id").to_numpy(zero_copy_only=False)[rows]
        # Genes by name, one number each, on both sides.
        number = {name: n for n, name in enumerate(dict.fromkeys(item.gene for item in pool.items))}
        lut = np.array([number.get(name, -1) for name in dictionary] or [-1], np.int64)
        row_gene = lut[codes[rows]]
        asked = row_gene >= 0
        rows, x, y, z, ids, row_gene = rows[asked], x[asked], y[asked], z[asked], ids[asked], row_gene[asked]
        items = pool.items
        item_gene = np.array([number[item.gene] for item in items], np.int64)
        item_xy = _xy([item.x for item in items], [item.y for item in items])
        order, lo, hi = _groups(row_gene, _xy(x, y), item_gene, item_xy)
        if check is not None:
            check()
        parent = self._parent_groups(pool, number, item_gene, item_xy)
        status, row = [], []
        for j, item in enumerate(items):
            held = order[lo[j]:hi[j]]
            if held.size == 0:
                status.append("not_found")
                row.append(-1)
                continue
            if held.size == 1:
                status.append("ok")
                row.append(int(held[0]))
                continue
            records, zn = parent[j]
            if records.size != held.size or _tied(zn) or _tied(z[held]):
                status.append("unsure")
                row.append(-1)
                continue
            rank = int(np.flatnonzero(records[np.lexsort((records, zn))] == item.record)[0])
            status.append("ok")
            row.append(int(held[np.lexsort((held, z[held]))][rank]))
        for j, item in enumerate(items):            # a control goes with its target
            if item.kind == "matched" and status[j] == "ok" and status[item.target] != "ok":
                status[j] = status[item.target]
        counts = {kind: {"ok": 0, "not_found": 0, "unsure": 0} for kind in ("target", "matched", "uniform")}
        for item, found in zip(items, status, strict=True):
            counts[item.kind][found] += 1
        return self._matched(pool, [s == "ok" for s in status], [ids[r] if r >= 0 else 0 for r in row], counts,
                             str(input_path), began)

    def answers(self, run_id: str, cells: Iterable[int], area: Area | None = None) -> dict[int, dict]:
        """Each cell's answer as a pool takes it, at the run's last frame (its schedule's end, else its newest frame),
        whatever the reasons: {cell: its targets with their records, P, π, w, givers and start classes ("acting": a
        suspect of the cell; "primary": likely, placed and of a mixed cell, the area aside), its pattern and key, its
        controls' records and w and the genes they leave out}, w from the basis of `area` (else the region as it is
        when it fits, or the window around the first cell asked). A check for development and tests, through the
        run's worker."""
        run = self.runs.get(run_id)
        facts = self._facts_at_end(run)
        cells = [int(c) for c in cells]
        if area is None:
            first = cells[0] if cells else 0
            area = area_of(region_of(run.meta), (float(facts.cx[first]), float(facts.cy[first]))
                           if 0 <= first < facts.cx.size else (math.nan, math.nan))
        while True:                                  # asked again when a newer request replaced it before it began
            job = self._submit(run_id, _Job(("answers", object()), facts, cells=cells, area=area))
            job.done.wait()
            if not job.dropped:
                break
        outcome = job.outcome
        if isinstance(outcome, _Stopped):
            raise outcome.error
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def forget(self, run_id: str) -> None:
        """A run deleted: everything kept for it goes, and its worker stops (its waiters get 404)."""
        with self.lock:
            work = self.work.pop(run_id, None)
            for kept in (self.answers_kept, self.pools_kept):
                for key in [key for key in kept if key[0] == run_id]:
                    del kept[key]
            self.asked = {key for key in self.asked if key[0] != run_id}
            if work is not None:
                self._stop(work, RunError("Unknown run", 404))
            self.wake.notify_all()

    def close(self) -> None:
        """The app closes: nothing new starts; work under way stops (its waiters get 503) and the frame contexts go."""
        with self.lock:
            self.closed = True
            works = list(self.work.values())
            self.work.clear()
            for work in works:
                self._stop(work, QcError("The app is closing", 503))
            self.answers_kept.clear()
            self.pools_kept.clear()
            self.wake.notify_all()
        for work in works:
            thread = work.thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(2.0)

    # ---- the cheap tier ----------------------------------------------------------------------------------------------
    def _run(self, run_id: str, cell: int):
        """(the run, its start) for a preview or pool of `cell`: RunError 404 unless it is a proseg run that has such
        a cell (when its count is known); QcError 503 while the app closes."""
        from . import qc_rules
        if self.closed:
            raise QcError("The app is closing", 503)
        run = self.runs.get(run_id)
        if run.meta.get("kind", "proseg") != "proseg":
            raise RunError("Not a proseg run", 404)
        cells = (run.info or {}).get("cells")
        if isinstance(cells, int) and not 0 <= cell < cells:
            raise RunError("Unknown cell", 404)
        return run, qc_rules.start_of(run.meta)

    def _thresholds(self) -> dict:
        """The mixing check's settings now (the scheduler's), the measured ones without a scheduler."""
        if self.analysis is not None and hasattr(self.analysis, "qc_thresholds"):
            return dict(self.analysis.qc_thresholds())
        return dict(qc_rule.MEASURED["thresholds"])

    def _cheap(self, run, cell: int, start: str) -> _Facts:
        """Every reason that needs no frame work (of every kind, in the page's order) and what the frame work needs:
        `_prerequisites` (the run's own, the check's, F scored); then, only once neither a refusal nor a wait holds
        there, the cell's at F: edge (a caveat: it is pooled anyway), not_judged (also a cell with no centre at F,
        which owns nothing there). A judged cell under the cut-off goes on to its answer (whether anything of it glows:
        no_suspects, else low_share). Else the area alone, where it is known."""
        facts = self._prerequisites(run, start)
        if facts.failure is not None:
            return facts
        found = list(facts.reasons)
        try:
            if all(KIND[r["code"]] == "caveat" for r in found):
                self._at_f(run, facts)
                facts.area, place = self._area(run, cell, facts)
                inner = None if facts.area is None else facts.area.inner
                if inner is None or not bool(qc_rule.inside_inner(place[0], place[1], inner)):
                    found.append(reason("edge"))
                code = int(facts.codes[cell])
                if code == NOT_JUDGED or not (math.isfinite(place[0]) and math.isfinite(place[1])):
                    found.append(reason("not_judged"))
                facts.name = self._names(run)[cell]
            else:
                facts.area, _ = self._area(run, cell, facts, quiet=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
            facts.failure = str(exc) or type(exc).__name__
        facts.reasons = ordered(found)
        return facts

    def _prerequisites(self, run, start: str) -> _Facts:
        """The run's own reasons (run_reasons), the check's and F's scoring, with no cell and no frame file read: the
        check's is check_unavailable (a refusal: the sample's check, or the run's, cannot run, or Clusters & UMAP
        cannot run here; not for a run whose own refusal, no_counts or old_proseg, says why), else check_off (a
        wait), else check_wait (not learned, or not ready for this run) and details (its model has no Go deeper's
        details), both waits; settings (a caveat) when the check's settings are not the measured ones; count_basis (a
        caveat) when the row it is judged with was measured with suspects chosen on the outlines, not measured again on
        proseg's own count (qc_rule.measured_on_proseg: a start with nuclei's rows; any row while the constants are
        the outlines', qc_rule.OUTLINE_IDS). F
        (the schedule's end) is asked to be scored only when no refusal or wait holds before it (`_scored`). The facts
        carry F, the model, the settings and the row."""
        from . import qc_rules as qr
        meta = run.meta
        facts = _Facts(run_id=meta["id"], start=start)
        done = meta.get("schedule_done") if isinstance(meta.get("schedule_done"), dict) else {}
        facts.seq, facts.iter = done.get("seq"), done.get("iter")
        facts.count = self._count_at(run, facts.seq)
        found = []
        try:
            facts.row = qc_rule.row_for(start, region_of(meta)["bbox"])
            found = run_reasons(meta, run.info, max_um=self.runs.max_region)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            facts.failure = str(exc) or type(exc).__name__
            return facts
        described = self.qc.describe(meta["id"])
        # A run without proseg's count (no_counts, old_proseg: its own refusal, found above) makes the run's check
        # unavailable, not the sample's: its refusal says it, never "can't run on this sample" beside it.
        own = described.get("state") == "unavailable" and described.get("reason") in ("no_counts", "old_proseg") and \
            any(r["code"] == described.get("reason") for r in found)
        if (described.get("sample_state") == "unavailable" or not getattr(self.analysis, "available", True)
                or (described.get("state") == "unavailable" and not own)):
            found.append(reason("check_unavailable"))
        elif own:                                        # nothing for the check to wait for: the run has no count
            pass
        elif not described.get("enabled"):
            found.append(reason("check_off"))
        else:
            if described.get("state") != "ready":
                found.append(reason("check_wait"))
            model = self.qc.qc_model(run)
            if model is None:
                if not any(r["code"] == "check_wait" for r in found):
                    found.append(reason("check_wait"))
            else:
                facts.model = {"id": model["id"], "path": model["path"], "deep": model.get("deep")}
                if not model.get("deep"):
                    found.append(reason("details"))
        facts.thresholds = self._thresholds()
        if not all(qr.same(facts.thresholds.get(k), v) for k, v in qc_rule.MEASURED["thresholds"].items()):
            found.append(reason("settings"))
        if not qc_rule.measured_on_proseg(*facts.row):      # suspects chosen on the outlines when it was measured
            found.append(reason("count_basis"))
        try:
            if all(KIND[r["code"]] == "caveat" for r in found):
                self._scored(run, facts, found)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
            facts.failure = str(exc) or type(exc).__name__
        facts.reasons = ordered(found)
        return facts

    def _scored(self, run, facts: _Facts, found: list):
        """F's stored bytes (by the model the check uses) on the averaged count, which the test reads (its eligibility
        and pool, as it was measured: <seq>.qc.bin, or <seq>.qc.averaged.bin beside a sampled F's), and on a sampled F
        its own bytes on the count proseg saves too (Go deeper's own answer there: `_shown`); else check_scoring
        {wait: true}: the check is asked once per (run, F, its count, model) to score F (a sample learned again asks its
        new model: the scheduler scores the newest frame and at most 40 earlier ones by itself, not always F; F counted
        again for the hold is scored again), and asked again on a later preview when it refused."""
        run_id = facts.run_id
        try:
            facts.codes_path = Path(self.qc.cells_path(run_id, facts.seq, model=facts.model["id"], count="averaged"))
            if facts.count == "sampled":
                facts.codes_path_f = Path(self.qc.cells_path(run_id, facts.seq, model=facts.model["id"],
                                                             count="sampled"))
        except QcError as exc:
            if exc.status != 404:
                raise
            facts.codes_path = facts.codes_path_f = None
            found.append(reason("check_scoring", wait=True))
            key = (run_id, facts.seq, facts.count, facts.model["id"])
            with self.lock:                          # marked first: a preview meanwhile does not ask a second time
                ask = key not in self.asked
                self.asked.add(key)
            if ask:
                try:
                    self.qc.configure(run_id, seq=facts.seq)
                except Exception:                    # refused (the check says why on its own panel): asked again
                    with self.lock:                  # on the next preview
                        self.asked.discard(key)

    def _at_f(self, run, facts: _Facts):
        """F's record, stored bytes and cell centres."""
        from ..analysis.worker import read_centres
        ncells = int(run.info["cells"])
        with run.lock:
            frame = dict(run.frames[facts.seq]) if facts.seq is not None and 0 <= facts.seq < len(run.frames) else None
        if frame is None:
            raise RuntimeError("your run's last iteration is missing")
        if not frame.get("counts_file"):
            raise RuntimeError("your run's last iteration has no count of proseg's own")
        facts.frame = frame
        if facts.codes_path is not None:
            codes = _codes(facts.codes_path)
            if codes.size != ncells:
                raise RuntimeError("the mixing check's numbers for the last iteration are not this run's")
            facts.codes = codes
        centres = read_centres(run.live / frame["cells_file"], ncells) if frame.get("cells_file") else None
        if centres is None:
            raise RuntimeError("your run's last iteration has no cell centres")
        facts.cx, facts.cy = np.asarray(centres["cx"], np.float32), np.asarray(centres["cy"], np.float32)

    def _area(self, run, cell: int, facts: _Facts, quiet: bool = False) -> tuple[Area | None, tuple[float, float]]:
        """(the test's area, the cell's place at F): the region as it is when it fits (area_of); else a window around
        the cell's centre at F (read here when the cheap tier stopped before it, `quiet`: nothing raised, None when not
        known), None for a cell with no place there."""
        region = region_of(run.meta)
        place = (math.nan, math.nan)
        try:
            if facts.cx is None and quiet and not fits(region):
                self._at_f(run, facts)
            if facts.cx is not None:
                place = (float(facts.cx[cell]), float(facts.cy[cell]))
            return area_of(region, place), place
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError):
            if quiet:
                return None, place
            if not fits(region):                    # a window around a cell with no place: no area
                return None, place
            raise

    def _names(self, run) -> list[str]:
        """The run's cells.txt names (line k: label k + 1), kept per file version."""
        path = run.live / "cells.txt"
        key = (str(path), path.stat().st_mtime_ns)
        with self.lock:
            found = self.names.get(key)
        if found is None:
            found = path.read_text().split("\n")[:int(run.info["cells"])]
            with self.lock:
                self.names[key] = found
                while len(self.names) > 8:
                    self.names.popitem(last=False)
        return found

    def _facts_at_end(self, run) -> _Facts:
        """`answers`' facts: F (the schedule's end, else the newest frame), the model, the settings, the row, the stored
        bytes and centres; no reasons asked."""
        from . import qc_rules
        meta = run.meta
        facts = _Facts(run_id=meta["id"], start=qc_rules.start_of(meta))
        facts.row = qc_rule.row_for(facts.start, region_of(meta)["bbox"])
        done = meta.get("schedule_done") if isinstance(meta.get("schedule_done"), dict) else {}
        with run.lock:
            last = dict(run.frames[-1]) if run.frames else {}
        facts.seq = done.get("seq") if done.get("seq") is not None else last.get("seq")
        facts.iter = done.get("iter") if done.get("seq") is not None else last.get("iter")
        facts.count = self._count_at(run, facts.seq)
        model = self.qc.qc_model(run)
        if model is None:
            raise QcError("No mixing check for this run yet", 404)
        facts.model = {"id": model["id"], "path": model["path"], "deep": model.get("deep")}
        facts.thresholds = self._thresholds()
        facts.codes_path = Path(self.qc.cells_path(meta["id"], facts.seq, model=model["id"], count="averaged"))
        self._at_f(run, facts)
        return facts

    @staticmethod
    def _count_at(run, seq) -> str:
        """The count frame `seq` (F) is read on now (counts_file.count_of of its counts line): "sampled" for a frame
        proseg counted for a pause, the hold, a finish or the end, else "averaged"."""
        with run.lock:
            line = run.frames[seq].get("counts") if isinstance(seq, int) and 0 <= seq < len(run.frames) else None
        return counts_file.count_of(line)

    def _reply(self, run, cell: int, facts: _Facts, refusals: list, waits: list, caveats: list) -> dict:
        """The pool reply with its cheap fields: F (seq, iter), the count F is read on (`count`: "sampled" for a frame
        proseg counted for a pause, the hold, a finish or the end, which Go deeper and the panel show on the count proseg
        saves, while the test reads the averaged count), the model, the start, its row and the row's constants, the area
        where known, the reasons by kind (`offer` without a refusal); nothing of the cell's pool yet (`line` and
        `line_shown`, Go deeper's own line at F, come with it)."""
        constants = qc_rule.constants_for(*facts.row)
        return {"v": POOL_V, "state": "ready", "run": run.meta.get("id"), "cell": cell, "cell_name": facts.name,
                "seq": facts.seq, "iter": facts.iter, "count": facts.count, "model": (facts.model or {}).get("id"),
                "constants": qc_rule.CONSTANTS_ID, "start": facts.start,
                "runs_per_side": constants["runs_per_side"], "k": constants["k"],
                "area": None if facts.area is None else facts.area.public(), "offer": not refusals,
                "reasons": list(refusals), "pattern": None, "line": None, "line_shown": None, "this_cell": None,
                "pool": None,
                "estimate_s": ESTIMATE_S.get(facts.row[0]), "ms": None, "message": None, "waits": list(waits),
                "caveats": list(caveats), "row": {"start": facts.row[0], "situation": facts.row[1]}}

    # ---- the worker --------------------------------------------------------------------------------------------------
    @staticmethod
    def _pool_key(facts: _Facts, cell: int) -> tuple:
        return (*facts.key(), facts.start, None if facts.area is None else facts.area.box, int(cell))

    def _kept(self, key: tuple) -> _Outcome | None:
        with self.lock:
            found = self.pools_kept.get(key)
            if found is not None:
                self.pools_kept.move_to_end(key)
            return found

    @staticmethod
    def _settled(outcome):
        """A finished job's outcome for its waiter: the _Outcome, a failure's words (str), or the stop's error
        raised."""
        if isinstance(outcome, _Stopped):
            raise outcome.error
        if isinstance(outcome, RunError) and outcome.status == 404 and str(outcome) == "Unknown run":
            raise outcome
        if isinstance(outcome, BaseException):
            return f"{FAILED}: {outcome}"
        return outcome

    def _submit(self, run_id: str, job: _Job) -> _Job:
        """Queue `job` on the run's worker (started when it has none): the same request on its way is shared; a newer
        one replaces the request waiting."""
        with self.lock:
            if self.closed:
                raise QcError("The app is closing", 503)
            work = self.work.get(run_id)
            if work is None:
                work = self.work[run_id] = _Work()
            for other in (work.current, work.pending):
                if other is not None and other.key == job.key and not other.dropped:
                    return other
            if work.pending is not None:
                work.pending.dropped = True
                work.pending.done.set()
            work.pending = job
            if work.thread is None:
                work.thread = threading.Thread(target=self._serve, args=(run_id, work), name=f"qc-pool-{run_id}",
                                               daemon=True)
                work.thread.start()
            self.wake.notify_all()
            return job

    @staticmethod
    def _stop(work: _Work, error: Exception):
        """(under the lock) The run's worker stops: its waiting request ends with `error`."""
        work.stop = error
        if work.pending is not None:
            work.pending.outcome = _Stopped(error)
            work.pending.done.set()
            work.pending = None

    def _serve(self, run_id: str, work: _Work):
        """A run's worker thread: its requests one at a time; idle, its frame context goes once unused for
        POOL_HOLD_S, and the thread ends once it holds none."""
        # This thread only, never back up: Linux alone takes a thread's id as a process's here (on macOS the same number
        # could name another process).
        if sys.platform.startswith("linux"):
            try:
                me_id = threading.get_native_id()
                os.setpriority(os.PRIO_PROCESS, me_id, max(10, os.getpriority(os.PRIO_PROCESS, me_id)))
            except (AttributeError, OSError):        # not allowed here: the same priority as the engine
                pass
        me = threading.current_thread()
        try:
            while True:
                job = release = None
                with self.lock:
                    while work.pending is None and work.stop is None and not self.closed:
                        if work.context is not None and self.clock() - work.used >= POOL_HOLD_S:
                            release, work.context = work.context, None
                            break
                        if work.context is None:
                            work.thread = None       # nothing held: a new request starts another thread
                            return
                        self.wake.wait(TICK_S)
                    if release is None:
                        if work.stop is not None or self.closed:
                            return
                        job, work.pending, work.current = work.pending, None, work.pending
                if release is not None:
                    release.close()
                    continue
                try:
                    with ONE_BLAS:                   # the recipe's BLAS on this (lower priority) thread alone
                        outcome = self._compute(run_id, work, job)
                except _Stopped as stopped:
                    outcome = stopped
                except Exception as exc:
                    outcome = exc
                    if not isinstance(exc, (RunError, QcError, OSError, ValueError, RuntimeError)):
                        print(f"SoPA companion: the test's pool of {run_id} failed: {type(exc).__name__}: {exc}",
                              flush=True)
                with self.lock:
                    work.current = None
                    work.used = self.clock()
                    job.outcome = outcome
                    if isinstance(outcome, _Outcome) and self.work.get(run_id) is work:
                        self.pools_kept[job.key] = outcome
                        while len(self.pools_kept) > POOLS_KEPT:
                            self.pools_kept.popitem(last=False)
                job.done.set()
        finally:
            context = None
            with self.lock:
                if work.thread is me:                # stopped (an idle end left already, holding nothing)
                    work.thread = None
                    context, work.context = work.context, None
                    if work.pending is not None:     # stopped with a request waiting
                        work.pending.outcome = _Stopped(work.stop or QcError("The app is closing", 503))
                        work.pending.done.set()
                        work.pending = None
            if context is not None:
                context.close()

    def _yield(self, work: _Work):
        """Between two cells: the engine's other threads get their turn; a stopped worker stops here."""
        time.sleep(0)
        with self.lock:
            error = work.stop if work.stop is not None else QcError("The app is closing", 503) if self.closed else None
        if error is not None:
            raise _Stopped(error)

    def _compute(self, run_id: str, work: _Work, job: _Job):
        run = self.runs.get(run_id)
        box = tuple(float(v) for v in (job.area if job.area is not None else job.facts.area).box)
        context = work.context
        # An area the frame context holds keeps its own basis (an export read is F's own, whatever the parent does
        # since); any other area is placed afresh.
        basis = None if context is None or context.facts_key != job.facts.key() else context.bases.get(box)
        if basis is None:
            basis = self._basis(run, job.facts, box)
            context = self._context(work, run, job, basis)
        try:
            weights = context.weights(box, basis)
        except (OSError, ValueError):
            if basis.held != "exact":
                raise
            # Its export went before it was read (proseg keeps its 4 newest; Go deeper's own asks make more): asked
            # again, once.
            basis = self._basis(run, job.facts, box)
            context = self._context(work, run, job, basis)
            weights = context.weights(box, basis)
        job.basis = basis
        if job.cells is not None:
            out = {}
            for cell in job.cells:
                out[cell] = self._answer(work, job, context, weights, cell).public(context.names, context.mixed[cell])
            return out
        return self._take(work, job, context, weights)

    def _basis(self, run, facts: _Facts, box: tuple) -> _Basis:
        """Where w comes from for the test area `box` at F (module docstring), in this order: an export of the area at
        F the engine holds (Runs.where_export); one asked of the parent while it is held or paused at F
        (Runs.request_where, waiting at most WHERE_WAIT_S); F's own per-layer labels (a dense frame; a tiled frame's
        window when it covers the area: approx_targets); a tiled parent's state now, exactly (an export asked of it
        paused or sampling: targets_now); a tiled frame's window over part of the area when none comes (the records
        outside it no cell's: approx_targets); none. Each export is asked of the area as it lies on F's grid (a region
        drawn past the tissue holds no transcript there)."""
        run_id, seq, frame = facts.run_id, int(facts.seq), facts.frame
        box = on_grid(box, frame)
        if box is None:                                 # the area holds no column of the run: nothing glows there
            return _Basis("none", seq, frame)
        found = self.runs.where_export(run_id, seq, list(box))
        if found is not None:
            return _Basis("exact", seq, frame, where=found)
        with run.lock:
            ack = dict(run.ack) if isinstance(getattr(run, "ack", None), dict) else {}
        if ack.get("state") == "paused" and ack.get("seq") == seq:
            found = self._ask_where(run_id, box)
            if found is not None and found.get("seq") == seq:
                return _Basis("exact", seq, frame, where=found)
        if frame.get("format") != "tiles":
            return _Basis("approx", seq, frame, caveat="approx_targets")
        window = _window_over(run, frame, box)
        if window is not None:
            return _Basis("approx", seq, frame, window=window, caveat="approx_targets")
        found = self._ask_where(run_id, box)
        if found is not None:
            now = int(found["seq"])
            with run.lock:
                record = dict(run.frames[now]) if 0 <= now < len(run.frames) else None
            if record is not None and record.get("counts_file"):
                return _Basis("exact", now, record, where=found, caveat="targets_now")
        # Nothing now (it ended, or no export came in time): F's window where it holds some of the area, so the cells
        # Go deeper places from it (their boxes inside it) have their controls; none where it holds nothing of it.
        window = _window_over(run, frame, box, part=True)
        if window is not None:
            return _Basis("approx", seq, frame, window=window, caveat="approx_targets")
        return _Basis("none", seq, frame)

    def _ask_where(self, run_id: str, box: tuple) -> dict | None:
        """proseg's export of the area asked of the parent (Runs.request_where, waiting at most WHERE_WAIT_S), None when
        none comes (or the parent cannot be asked)."""
        try:
            return self.runs.request_where(run_id, list(box), wait_s=WHERE_WAIT_S)
        except RunError:
            return None

    def _context(self, work: _Work, run, job: _Job, basis: _Basis) -> _Context:
        """The run's frame context for the job's F, model and settings at the basis' frame and kind, made when it holds
        another or none."""
        key = (*job.facts.key(), basis.seq, basis.held)
        context = work.context
        if context is not None and context.key == key:
            return context
        with self.lock:
            work.context = None
        if context is not None:
            context.close()
        self._yield(work)
        made = _TiledContext if basis.frame.get("format") == "tiles" else _Context
        context = made(key, run, job.facts, basis)
        with self.lock:
            work.context = context
            work.used = self.clock()
            self.contexts += 1
        job.one_time_ms += context.ms
        return context

    def _answer(self, work: _Work, job: _Job, context: _Context, weights: _Weights, cell: int) -> _Answer:
        """A cell's answer: kept, else worked out (after a yield)."""
        key = (*context.key, int(cell))
        with self.lock:
            found = self.answers_kept.get(key)
            if found is not None:
                self.answers_kept.move_to_end(key)
                return found
        self._yield(work)
        found = self._cell(context, weights, int(cell))
        with self.lock:
            if self.work.get(context.run_id) is work:
                self.answers_kept[key] = found
                while len(self.answers_kept) > CELLS_KEPT:
                    self.answers_kept.popitem(last=False)
            self.computed += 1
        job.computed += 1
        return found

    # ---- one cell ----------------------------------------------------------------------------------------------------
    def _cell(self, context: _Context, weights: _Weights, c: int, count: str = "averaged",
              codes: Path | None = None) -> _Answer:
        """Cell c's answer at H (module docstring): Go deeper's on proseg's count, the page's where request, w, π and
        P, the targets with their records and start classes, the pattern and key (of its likely targets; a weak cell's
        of its glowing ones), the controls (from its records with w > 0 in the area's basis). On the averaged count e
        (the test's), or with `count` "sampled" as Go deeper shows it at a sampled F, on the count proseg saves (f):
        the worker's deeper with the request's count "sampled" on f's matrix and F's own bytes on f (`codes`), the
        where measurement on f, the controls' w at f / n; its byte stays the stored one on e, which the test judges."""
        began = time.perf_counter()
        saved = count == "sampled"
        request = {**context.request, "count": "sampled"} if saved else context.request
        block = {**context.block, "cell": c, **({"codes": str(codes)} if saved else {})}
        result = context.worker.deeper(context.rs, {**request, "deeper": block},
                                       context.sampled() if saved else context.counts, {})
        raw = _plain(result)
        params = qc_scoring.page_params(raw, c) or {}
        req = params.get("req")
        sources, same, asked, targets, placed_by = (), (), (), (), None
        if req:
            p = req["params"]
            sources = tuple(int(s) for s in p["sources"])
            asked = tuple(str(g) for g in p["genes"])
            shown = {row.get("cell"): row.get("same") for row in params.get("rows") or [] if isinstance(row, dict)}
            kinds = {n.get("cell"): n.get("same") for n in raw.get("neighbours") or [] if isinstance(n, dict)}
            same = tuple(bool(shown[s]) if s in shown else bool(kinds.get(s)) for s in sources)
            try:
                out, records = self.where.measure(context.run_id, context.seq, c, list(sources), list(p["genes"]),
                                                  list(p["own"]), givers=givers_of(result),
                                                  **({"count": "sampled"} if saved else {}))
            except RunError as exc:
                if exc.status != 422:
                    raise
                out = None                           # the page shows nothing either (too far apart, too many)
            if out is not None:
                placed_by = out.get("held") if isinstance(out.get("held"), str) else None
                scores = qc_scoring.page_scores(raw, req, out)
                if scores:
                    targets = self._targets(context, c, out, records, scores, sources)
        # The genes the answer tags as a neighbour's (its gene rows' fifth field, 1), asked or not.
        tagged = {str(row[0]) for row in raw.get("genes") or []
                  if isinstance(row, list) and len(row) > 4 and row[4] == 1}
        all_same = bool(sources) and all(same)
        likely = [t for t in targets if t.likely]
        pattern, hold, key = pattern_of([t.start_class for t in (likely or targets)], all_same)
        line = {"likely": len(likely), "own": 0, "none": 0, "neighbour": 0, "other": 0}
        for t in likely:
            line[LINE[t.start_class]] += 1
        excluded = tuple(sorted(set(asked) | tagged))
        answer = _Answer(cell=c, code=int(context.codes[c]), state=raw.get("state"), request=bool(req),
                         sources=sources, same=same, all_same=all_same, targets=targets, pattern=pattern, hold=hold,
                         key=key, likely=len(likely), line=line, matched={}, short=0, uniform=(), excluded=excluded,
                         seconds=0.0, held=placed_by, count="sampled" if saved else "averaged")
        own, w = weights.of(c, context.q(c, "f" if saved else "e")) if targets else \
            (np.zeros(0, np.int64), np.zeros(0, np.float64))
        answer.matched, answer.short, answer.uniform, answer.w = self._controls(context, c, answer.acting(),
                                                                              set(excluded), own, w)
        answer.owned = int(own.size)
        answer.seconds = time.perf_counter() - began
        return answer

    def _targets(self, context: _Context, c: int, out: dict, records: np.ndarray, scores: dict,
                 sources: tuple) -> tuple[_Target, ...]:
        """The page's glowing points of the cell (P = w x π >= GLOW_MIN, whoever holds their place in the drawing:
        one just outside the outline is RNA proseg moved in), with the parent record of each (by its 16 bytes), its
        giver (the source its gene's numbers name: the where measurement's gene_stats) and where it sat at the start;
        likely: π >= LIKELY_MIN (as float32) on a point proseg counts at least LIKELY_W of for the cell."""
        points = [p for p in scores.get("points") or [] if isinstance(p, dict)]
        if not points:
            return ()
        stats = {s["gene"]: s for s in out.get("gene_stats") or [] if s.get("role") == "from"}
        idx = np.array([int(p["i"]) for p in points], np.int64)
        picked = np.ascontiguousarray(records[idx])
        found, copies = context.find(picked)
        names = context.gene_names[picked["gene"].astype(np.int64)]
        labels, permille = context.start_at(picked["x"], picked["y"], picked["z"])
        strong = int(qc_rule.MEASURED["strong_permille"])
        likely_min, likely_w = likely_rule()
        made = []
        for j, p in enumerate(points):
            if str(p["gene"]) != names[j]:
                raise RuntimeError("a suspect's gene is not its transcript's")
            k = (stats.get(p["gene"]) or {}).get("giver")
            giver = sources[k] if isinstance(k, int) and 0 <= k < len(sources) else None
            value, pi, w = np.float32(p["P"]), np.float32(p["pi"]), float(p["w"])
            made.append(_Target(point=int(p["i"]), gene=str(p["gene"]), x=float(p["x"]), y=float(p["y"]),
                                record=int(found[j]), copies=int(copies[j]), p=float(value),
                                likely=bool(pi >= likely_min and w >= likely_w), near=bool(p.get("near")),
                                giver=giver,
                                start_class=start_class(int(labels[j]), int(permille[j]), c, giver, strong),
                                start_label=int(labels[j]), permille=int(permille[j]), rank=p.get("rank"),
                                of=p.get("of"), w=w, pi=float(pi), owner=int(p.get("owner", 1))))
        return tuple(sorted(made, key=lambda t: t.point))

    def _shown(self, context: _Context, facts: _Facts, weights: _Weights, c: int) -> _Answer | None:
        """The seed's answer as Go deeper shows it at a sampled F (the count proseg saves: the panel and the pink dots
        read f there, the test e), worked out once more for the seed alone (`_cell` with count "sampled": Go deeper's
        answer on F's count proseg saves with F's own bytes on it, its where request, the where measurement on f, the
        page's scores, the targets and their line, the controls at f / n): the reply's `line_shown`, and the seed's
        suspects where e's cannot be tested (saved_suspects: `_take`). None where F is read on the averaged count (e is
        what is shown there), where the frame worked is not F (targets_now), or without F's own bytes on f."""
        if facts.count != "sampled" or context.seq != int(facts.seq) or facts.codes_path_f is None:
            return None
        return self._cell(context, weights, int(c), count="sampled", codes=facts.codes_path_f)

    @staticmethod
    def _controls(context: _Context, c: int, acting: list, excluded: set, own: np.ndarray,
                  w: np.ndarray) -> tuple[dict, int, tuple, dict]:
        """(matched, short, uniform, w of each control) of a cell with targets: its records with w > 0 (`own`, `w`:
        proseg counts some of each for it) whose gene is neither asked as a neighbour's nor tagged as one (the points
        with P > 0 are those genes' only); per acting target with a placed giver, in point order, MATCHED_PER_TARGET of
        those with the target's near flag to that giver and its band of w (W_BANDS, both to 3 decimals as the where
        reply gives a target's), drawn without replacement (seeded [seed, c, 0]); UNIFORM_PER_CELL of them (seeded
        [seed, c, 1]). Nothing for a cell without targets."""
        if not acting and not own.size:
            return {}, 0, (), {}
        controls = qc_rule.MEASURED["controls"]
        seed, per_target, per_cell = (int(controls[k]) for k in ("seed", "matched_per_target", "uniform_per_cell"))
        genes = context.gene_names[context.records["gene"][own].astype(np.int64)]
        keep = ~np.isin(genes, sorted(excluded)) if excluded else np.ones(own.size, bool)
        eligible, shares = own[keep], np.round(w[keep], 3)
        band = np.searchsorted(W_BANDS, shares, side="right")
        ex, ey = context.records["x"][eligible], context.records["y"][eligible]
        matched, short, given = {}, 0, {}
        rng = np.random.default_rng([seed, c, 0])
        near_of: dict = {}
        for t in acting:
            if t.giver is None:
                continue
            if t.giver not in near_of:
                near_of[t.giver] = context.near(t.giver, ex, ey)
            fits = (near_of[t.giver] == t.near) & (band == int(np.searchsorted(W_BANDS, round(t.w, 3), side="right")))
            pool = np.flatnonzero(fits)
            k = min(per_target, pool.size)
            short += per_target - k
            picked = pool[rng.choice(pool.size, size=k, replace=False)] if k else pool[:0]
            matched[t.point] = tuple(int(eligible[r]) for r in picked)
            given.update({int(eligible[r]): float(shares[r]) for r in picked})
        rng = np.random.default_rng([seed, c, 1])
        k = min(per_cell, eligible.size)
        picked = rng.choice(eligible.size, size=k, replace=False) if k else np.zeros(0, np.int64)
        uniform = tuple(int(eligible[r]) for r in picked)
        given.update({int(eligible[r]): float(shares[r]) for r in picked})
        return matched, short, uniform, given

    # ---- the pool ----------------------------------------------------------------------------------------------------
    def _take(self, work: _Work, job: _Job, context: _Context, weights: _Weights) -> _Outcome:
        """The seed's answer and reasons (refusals: nothing of it glows with a placed giver on e nor, at a sampled F, on
        f, every source of its own kind, no primary target; caveats: its basis' (approx_targets, targets_now), its
        suspects taken on f where e's cannot be tested and f's can, or nothing of it glows on e and f's glow
        (saved_suspects: Go deeper's answer there, `_shown`, whose refusals are then the ones said), its targets placed
        approximately (approx_targets: its where reply's held), a pattern not measured, under the cut-off (low_share),
        no likely suspect (weak_suspects)), then the pool of its own key (lazily, nearest first, the row's k; under k,
        every cell of that key with a primary target, and too_few, a caveat; approx_targets too when a pooled cell's
        targets were placed approximately) and its items; refused no_controls under the rule "excess" when none of the
        pool's cells has a matched control (before any other cell is worked out when the area's basis is none: no cell
        has one). The seed's primary targets are its acting ones whose giver lies inside, whatever its byte; the other
        cells' are their likely ones with a placed giver inside, of mixed cells only."""
        facts, seed = job.facts, int(job.cell)
        constants = qc_rule.constants_for(*facts.row)
        k, inner = int(constants["k"]), facts.area.inner
        # qc_rule's "excess" judges the suspects against their own RNA at the same spot (their matched controls): a
        # pool without one is never judged (its answer too_few, no_controls, after every run), so it is not offered.
        excess = constants["rule"] == "excess"
        cx, cy = context.cx.astype(np.float64), context.cy.astype(np.float64)
        inside = qc_rule.inside_inner(cx, cy, inner)

        def primary(answer: _Answer) -> list[_Target]:
            if answer.cell == seed:
                return [t for t in answer.acting() if inside[t.giver]]
            if not context.mixed[answer.cell]:
                return []
            return [t for t in answer.targets if t.likely and t.giver is not None and inside[t.giver]]

        def refused(answer: _Answer) -> str | None:
            """The seed's own refusal of this answer, the first of those said below (no_suspects, same_type,
            no_primary); None when its suspects can be tested."""
            if not answer.acting():
                return "no_suspects"
            if answer.all_same:
                return "same_type"
            return None if primary(answer) else "no_primary"

        first = self._answer(work, job, context, weights, seed)
        # At a sampled F the panel and Go deeper show f, the test reads e: Go deeper's own answer there gives the line
        # it shows, and the seed's suspects only where e's cannot be tested: where f's can, or where nothing of the cell
        # glows on e but its dots glow on f (then refused for their own reason, never no_suspects while they glow). The
        # usual case, e's, keeps the measured constants' meaning.
        shown = self._shown(context, facts, weights, seed)
        on_e, on_f = refused(first), "no_suspects" if shown is None else refused(shown)
        saved = on_e is not None and (on_f is None or (on_e == "no_suspects" and on_f != "no_suspects"))
        if saved:
            first = shown
        n_seed = len(primary(first))
        fields = {"cell_name": context.names[seed], "pattern": PATTERNS.get(first.pattern), "line": dict(first.line),
                  "line_shown": dict((shown or first).line), "this_cell": {"likely": first.likely, "suspects": n_seed}}

        caveats = [reason(job.basis.caveat)] if job.basis is not None and job.basis.caveat else []

        def placed_roughly(*found: _Answer):
            """approx_targets, once, when any of these answers' where replies placed its targets approximately."""
            if any(a.held == "approx" for a in found) and not any(c["code"] == "approx_targets" for c in caveats):
                caveats.append(reason("approx_targets"))

        def outcome(reasons, pool=None):
            return _Outcome(reasons, pool, fields, job.one_time_ms, job.computed, caveats=list(caveats))
        placed_roughly(first)
        if not first.acting():
            return outcome([reason("no_suspects")])
        if saved:                                   # taken on the count proseg saves: e gives none to test
            caveats.append(reason("saved_suspects"))
        if first.all_same:
            return outcome([reason("same_type")])
        if first.pattern in PATTERN_REASONS:        # pooled all the same, with the cells of its own key
            caveats.append(reason(PATTERN_REASONS[first.pattern]))
        if not context.mixed[seed]:                 # judged (not_judged refused it before), under the cut-off
            caveats.append(reason("low_share"))
        if first.weak:
            caveats.append(reason("weak_suspects"))
        if n_seed < 1:
            return outcome([reason("no_primary")])
        if excess and weights.held == "none":       # no record has a share for any cell: no control can be drawn
            return outcome([reason("no_controls")])
        here = context.judged & context.mixed & inside
        here[seed] = False
        others = np.flatnonzero(here)
        far = np.hypot(cx[others] - cx[seed], cy[others] - cy[seed])
        far = np.where(np.isfinite(far), far, np.inf)
        by = np.lexsort((others, far))
        members, reserve, total = [(seed, n_seed, 0.0)], [], n_seed
        answers = {seed: first}
        for cell, um in zip(others[by].tolist(), far[by].tolist(), strict=True):
            answer = self._answer(work, job, context, weights, cell)
            if answer.key != first.key:
                continue
            n = len(primary(answer))
            if n < 1:
                continue
            answers[cell] = answer
            if total < k:
                members.append((cell, n, um))
                total += n
            else:
                reserve.append((cell, n, um))
                if len(reserve) >= RESERVE_CELLS:
                    break
        placed_roughly(*answers.values())          # the pooled cells' targets (members and reserve)
        if excess and not any(answers[cell].matched.get(t.point) for cell, _, _ in members + reserve
                              for t in primary(answers[cell])):
            return outcome([reason("no_controls")])
        reached = total >= k
        fields["pool"] = {"cells": len(members), "suspects": total, "need": k, "reached": reached,
                          "available": None if reached else total,
                          "members": [{"cell": cell, "name": context.names[cell], "n": n, "um": round(um, 1)}
                                      for cell, n, um in members]}
        if not reached:                             # tested with all it has
            caveats.append(reason("too_few", have=total, need=k))
        return outcome([], self._pool(context, facts, seed, members, reserve, answers, primary, total, k, caveats))

    def _pool(self, context: _Context, facts: _Facts, seed: int, members: list, reserve: list, answers: dict,
              primary, total: int, k: int, caveats: list) -> Pool:
        """The Pool of these members and reserve: their primary targets, matched controls and own RNA as Items (each
        with its w), with the row and the cell's and the pool's own caveats."""
        cells = [cell for cell, _, _ in members + reserve]
        items: list[Item] = []
        targets: dict = {}

        def record_item(kind, cell, record, **values) -> Item:
            r = int(record)
            return Item(kind=kind, cell=cell, gene=context.genes[int(context.records["gene"][r])], record=r,
                        x=float(context.records["x"][r]), y=float(context.records["y"][r]),
                        zn=float(context.records["z"][r]), **values)
        for cell in cells:
            for t in primary(answers[cell]):
                targets[(cell, t.point)] = len(items)
                items.append(record_item("target", cell, t.record, target=None, giver=t.giver, near=t.near, p=t.p,
                                         primary=True, start_class=t.start_class, w=t.w))
        for cell in cells:
            for t in primary(answers[cell]):
                for r in answers[cell].matched.get(t.point, ()):
                    items.append(record_item("matched", cell, r, target=targets[(cell, t.point)], giver=t.giver,
                                             near=t.near, p=None, primary=False, start_class=None,
                                             w=answers[cell].w.get(int(r))))
        for cell in cells:
            for r in answers[cell].uniform:
                items.append(record_item("uniform", cell, r, target=None, giver=None, near=None, p=None,
                                         primary=False, start_class=None, w=answers[cell].w.get(int(r))))
        first = answers[seed]
        # Every cell named: the pool's, the givers of its items, and the seed's sources (a source that gives none of the
        # suspects is still drawn in the corner's views, found by its name in each test run).
        named = set(cells) | {item.giver for item in items if item.giver is not None} | set(first.sources)
        mine = primary(first)
        view = sorted(mine, key=lambda t: -qc_scoring.to_fixed(t.p, 3))[:qc_scoring.SUSPECTS_SENT]
        return Pool(v=POOL_V, run=facts.run_id, seq=int(facts.seq), iter=facts.iter, model=facts.model["id"],
                    constants=qc_rule.CONSTANTS_ID, start=facts.start, area=facts.area, k=k, seed=seed,
                    seed_name=context.names[seed], cells=tuple(cells), members=len(members),
                    distance_um=tuple(float(um) for _, _, um in members + reserve),
                    names={cell: context.names[cell] for cell in sorted(named)}, items=tuple(items), suspects=total,
                    available=None,
                    record_duplicates=len({t.record for cell in cells for t in primary(answers[cell]) if t.copies > 1}),
                    this_cell={"likely": first.likely, "suspects": len(mine), "pattern": PATTERNS.get(first.pattern),
                               "line": dict(first.line)},
                    view_suspects=tuple({"x": t.x, "y": t.y, "gene": t.gene, "p": qc_scoring.to_fixed(t.p, 3),
                                         **({"rank": t.rank, "of": t.of} if t.of is not None and t.rank is not None
                                            else {})} for t in view),
                    view_sources=tuple(first.sources), row=tuple(facts.row), caveats=tuple(ordered(caveats)))

    # ---- matching ----------------------------------------------------------------------------------------------------
    def _parent_groups(self, pool: Pool, number: dict, item_gene: np.ndarray, item_xy: np.ndarray) -> dict:
        """{item j: (the parent records at its gene and place, its own among them; their normalised depths)}: the
        parent's own records, the frame context's while it lasts, else read again."""
        records, genes = None, None
        with self.lock:
            work = self.work.get(pool.run)
            context = None if work is None else work.context
        if context is not None and context.key[:3] == (pool.run, pool.seq, pool.model):
            records, genes = context.records, context.genes
        if records is None:
            run = self.runs.get(pool.run)
            import gzip
            records = np.frombuffer(gzip.decompress((run.live / "transcripts.bin.gz").read_bytes()), RECORD)
            genes = (run.live / "genes.txt").read_text().split("\n")[:-1]
        lut = np.array([number.get(name, -1) for name in genes] or [-1], np.int64)
        gene = lut[np.minimum(records["gene"].astype(np.int64), len(lut) - 1)]
        asked = np.flatnonzero(gene >= 0)
        order, lo, hi = _groups(gene[asked], _xy(records["x"][asked], records["y"][asked]), item_gene, item_xy)
        out = {}
        for j in range(len(item_gene)):
            held = asked[order[lo[j]:hi[j]]]
            out[j] = (held, records["z"][held].astype(np.float64))
        return out

    def _matched(self, pool: Pool, ok: list[bool], ids: list, counts: dict, path: str, began: float) -> Matched:
        """The pool pooled again on its items found (qc_rule.pool_of over its cells' order, members first; under k,
        every cell with a suspect found, its too_few caveat said again with them), with the readout's Items and the
        rule's PoolOrder; TooFewFound when the clicked cell has no suspect found."""
        items = pool.items
        n_of = {cell: 0 for cell in pool.cells}
        for item, found in zip(items, ok, strict=True):
            if found and item.kind == "target":
                n_of[item.cell] += 1
        position = {cell: i for i, cell in enumerate(pool.cells)}
        order = qc_rule.PoolOrder(seed=0, cells=tuple(range(len(pool.cells))), distance_um=tuple(pool.distance_um),
                                  parent_cells=tuple(pool.cells))
        members, total = qc_rule.pool_of([n_of[cell] for cell in pool.cells], order, pool.k)
        if not members:
            raise TooFewFound()
        caveats = [found for found in pool.caveats if found.get("code") != "too_few"]
        if total < pool.k:
            caveats.append(reason("too_few", have=int(total), need=int(pool.k)))
        # Members first, then the rest in the pool's order (the cells that may still count keep their order).
        chosen = [pool.cells[i] for i in members]
        cells = chosen + [cell for cell in pool.cells if cell not in set(chosen)]
        distance = [pool.distance_um[position[cell]] for cell in cells]
        givers = sorted({item.giver for item, found in zip(items, ok, strict=True)
                         if found and item.giver is not None} - set(cells))
        names = [*cells, *givers]
        index = {cell: i for i, cell in enumerate(names)}
        genes = sorted({item.gene for item, found in zip(items, ok, strict=True) if found})
        gene_index = {name: g for g, name in enumerate(genes)}
        kept = [j for j, found in enumerate(ok) if found]
        new = {j: i for i, j in enumerate(kept)}
        kinds = {"target": qc_readout.SUSPECT, "matched": qc_readout.MATCHED, "uniform": qc_readout.UNIFORM}
        picked = [items[j] for j in kept]
        found_items = qc_readout.Items(
            kind=np.array([kinds[item.kind] for item in picked], np.int8),
            cell=np.array([index[item.cell] for item in picked], np.int32),
            giver=np.array([qc_readout.NO_GIVER if item.giver is None else index[item.giver] for item in picked],
                           np.int32),
            target=np.array([new[item.target] if item.kind == "matched" else -1 for item in picked], np.int64),
            gene=np.array([gene_index[item.gene] for item in picked], np.int32),
            transcript_id=np.array([ids[j] for j in kept], np.uint64),
            x=np.array([item.x for item in picked], np.float32), y=np.array([item.y for item in picked], np.float32),
            cell_names=tuple(str(pool.names[cell]) for cell in names), gene_names=tuple(genes))
        found_items.check()
        renumbered = tuple(replace(item, target=new[item.target]) if item.kind == "matched" else item
                           for item in picked)
        again = replace(pool, cells=tuple(cells), members=len(chosen), distance_um=tuple(distance),
                        names={cell: pool.names[cell] for cell in sorted(set(names) | set(pool.view_sources))},
                        items=renumbered, suspects=int(total), available=None, caveats=tuple(ordered(caveats)))
        return Matched(pool=again, items=found_items,
                       order=qc_rule.PoolOrder(seed=0, cells=tuple(range(len(cells))), distance_um=tuple(distance),
                                               parent_cells=tuple(cells)),
                       counts=counts, input=path, seconds=round(time.perf_counter() - began, 3))
