"""The mixing check's suggestions and trials, every comparison on the DEPTH-MATCHED change with thresholds calibrated
on repeat runs (engine/qc_constants.py: one entry per kind of run, its `mode`). Pure Python
and numpy; engine/qc.py's QcManager feeds it one live result at a time and never changes a knob: every change is the
user's click.

  * The depth-matched change (the GPU worker's `matched`, like for like on proseg's own count): for every cell judged
    now and in the frame's start block (the cells proseg started with, their RNA kept as proseg keeps it at this frame),
    both of its averaged profiles scaled to the shallower depth (no draws), both scored, per100 now minus per100 start
    over the same cells. Pure transcript loss gives no change; a boundary that sheds a neighbour's RNA lowers it. The
    raw headline stays the number on screen; arrows, "fewer"/"more", verdicts, the trial bands and the whole-tissue
    rules compare matched changes only (never a depth-band standardised share: it removes only part of a loss).
  * `change_threshold` / `sampler_threshold`: when a change counts: the size-aware sampler formula
    1.96 k 100 sqrt(p (1 - p) / n) (k = sampler_k; x sqrt 2 against another run) with the fixed value measured for the
    run's mode as a floor (matched_fixed: 1.0 in 100 for regions, 0.3 for whole samples); the model's own spread
    (matched_model: another learning could reverse a change below 2.3 in 100 in a region, 1.5 in a tiled one and 0.6
    in a whole sample) is reported beside it, never instead of it. An area uses its own formula (its own p and paired
    cells) and gets no verdict below its mode's minimum support (300 judged, 400 paired cells).
  * `evaluate`: the cards that match now, in the order the page offers them. Per confirmed area (a live, smoothed area
    the enter rule drew; never the start line's, a carried or a single snapshot's) the rules R0, U, R7, R1, R2, R3,
    R4, R5, R6 in this order, the first match per area winning; cards for the same knob merge across areas ("for areas
    A and C"); R2 and R3 are also checked once for the whole tissue; a harm card U for the whole tissue comes first.
    No rule ever suggests what the check's "never suggested for mixing" list names (`allowed`), and no card claims a
    setting cannot fix something: an area mixed from the start (R1) or one that rose without changing size (R6) says
    the cause is not known ("unknownCause") and points to its cells ("inspect": Go deeper, marker genes). R5's firmer
    hold is never offered to a run started from cells only (ctx "start", `start_of`: there it holds any neighbouring
    RNA inside the starting outlines).
  * `Watch`: one live knob change (a trial, or an edit in Live knobs): the flat band "expected without the change" of
    the mode's `trial` constants (the 50 iterations before the change, from burn-in + 20 on, >= 10 results), on the
    depth-matched change: centre the mean of the smoothed share (what the rows show), the comparison the matched change
    since then, half width max(2 SD of the history's matched changes, 1.0 in 100, the run's sampler threshold); the
    verdict (>= 5 results and >= 20 iterations, only up to 300 iterations: "less", "more", none yet, then "no_change"),
    an area its own verdict on its own band and its own matched change (never the tissue's ratio) when it holds the
    minimum support, and the harm the Undo card U reports (the whole tissue, or such an area, above its band on 5
    results in a row). Drift (rule R3) is judged like any other change: proseg's count shows the RNA drift moves back to
    its cell (UNSEEN_FLAGS, a change the count could not show, names none); several knobs changed at once are "mixed
    up" and get no Undo card (which one hurt cannot be told); the schedule's final settling stage (LAST_STAGE) stops a
    watch where it is.
  * `CardKeeper`: one card at a time, and steady: a card is shown once it has led on 2 results over 2 s, it gives its
    place to another only after that one led on 3 results over 10 s, and it goes 3 results over 10 s after its rule
    stopped matching (at once when none of its areas is drawn any more); U is shown at once (it is already 5 results).


What the page shows comes from frontend/src/qc-copy.mjs: a card here is data (rule, areas, the knob and its values,
the confidence and timing keys, the actions, the numbers its sentence needs), never words.
"""
from __future__ import annotations

import math

import numpy as np

from . import qc_constants as qk

# The trial band's calibrated values are the mode's `trial` constants (engine/qc_constants.py); the band's floor is
# also never below the run's sampler threshold for the matched change (Watch `floor`).
SETTLE_ITERS = 3           # the 3 iterations after the first frame with the new value are settling
STEADY = 2                 # results in a row a verdict needs to be given or taken back (no flicker at the band's edge)
HARM_RESULTS = 5           # rule U: above the band on this many results in a row
HARM_CLEAR = 3             # ... and the harm is gone again after this many results in a row not above it

# When a change counts: the mode's sampler_k, matched_fixed, matched_model, local_min_judged and local_min_paired
# (engine/qc_constants.py), calibrated on repeat runs. Only the matched change's values are used: every arrow and
# verdict compares depth-matched changes, never the raw headline.
SAMPLER_Z = 1.96

# The rules.
TOO_EARLY_ITER = 10        # R0: no card before iteration 10 (nor in init)
GROWTH = 0.10              # R1, R2, R4-R6: +10 % median growth past the starting outlines
COMPACT_STANDARD = 0.04    # R2: Cell compactness above this was loosened (proseg's default)
COMPACT_TIGHT = 0.02       # R4: the tighter shape the knob test measured
COMPACT_R4_MIN = 0.03      # R4 only while compactness is still at least this
TINY_MORE = 3              # R5: this many more tiny cells (< 10 µm²) than at the start ...
SHRUNK_SHARE = 0.10        # ... or this share of the area's cells lost more than half their starting area
CHURN_MOVING = 1.5         # an area's outlines "are still moving a lot" above this (its card says so): R6 says the
                           # outlines did not change, so it is not offered while they move
PRIOR_WEIGHT_MAX = 1000.0  # proseg_options' bound for Prior weight
FORMING_PHASES = ("init", "burnin")
# A change this view could not see gets no band, no verdict and no Undo card. None now: drift (R3) moves RNA back to
# the cell it came from in proseg's own assignment, which the check scores (repositioning comes before the split).
UNSEEN_FLAGS: tuple = ()
# The schedule's final settling stage ("Hill-climb": the temperature lowered so the outlines settle greedily): the line
# moves by itself there, and the band was calibrated on runs without it, so a watch stops where it is.
LAST_STAGE = ("hillclimb",)
# qc-copy.mjs RULE_COPY's confidence keys. R4 is a "Rough guess". R3 (drift) is "likely", as R2: on proseg's count,
# switching drift back on lowered mixing in every repeat run of two 500 µm squares, under both learnings that scored
# them, by about 1 in 100 (2026-10-31). Nothing says a setting is unlikely to fix an area from persistence alone: R1
# (mixed from the start, not grown) and R6 (rose without changing size) say the cause is not known ("unknownCause"); R7
# is what was seen in this run (a tried setting made it worse).
CONFIDENCE = {"U": "seen", "R7": "seen", "R1": "unknownCause", "R2": "likely", "R3": "likely", "R4": "rough",
              "R5": "rough", "R6": "unknownCause"}
SHAPE_RULES = ("R2", "R4", "R5")   # shape knobs: "New run with this setting" is the main action once cells formed
MORE_IDEAS = 3

# The card keeper (steady cards, the areas' own enter and clear rules).
CARD_ENTER = (2, 2.0)      # results, seconds a new card leads before it is shown
CARD_SWAP = (3, 10.0)      # ... before it takes the place of the card shown
CARD_GONE = (3, 10.0)      # ... a shown card's rule no longer matching before it goes


def number(value) -> float | None:
    """A finite number (never a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def same(a, b) -> bool:
    """Two setting values are the same: numbers within float32's precision (frames carry float32: 0.03999999910593033
    is 0.04), anything else exactly (None is "automatic")."""
    x, y = number(a), number(b)
    if x is not None and y is not None:
        return abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


def clean(value):
    """A setting value as the page gets it: a value proseg reported (a 32-bit float) as its shortest decimal
    (0.03999999910593033 -> 0.04); one asked for (the user's own digits, not a 32-bit float: 123.4567) exactly as it
    is; whole numbers, bools, None and words as they are."""
    if isinstance(value, (bool, int)) or number(value) is None:
        return value
    x = float(value)
    single = np.float32(x)
    return float(str(single)) if float(single) == x else x


def share100(value):
    """A share in 100 as a readout row shows it (one decimal), kept within 0-100: a row's "now" is the band's centre
    plus the matched change since the change, which a handful of cells could push past either end."""
    x = number(value)
    return None if x is None else round(min(100.0, max(0.0, x)), 1)


def exact(value, digits: int = 4):
    """A number as the summaries store it: unrounded for every decision (they are made before this), kept to `digits`
    decimals in the JSON (finer than anything shown); None for anything else."""
    x = number(value)
    return None if x is None else round(x, digits)


# ---- when a change counts ---------------------------------------------------------------------------------
def sampler_threshold(p, n, vs: str = "start", mode: str = "region") -> float | None:
    """The size-aware sampler threshold (points of per100): 1.96 k 100 sqrt(p (1 - p) / n), k the mode's sampler_k
    (0.6), x sqrt 2 when two runs are compared (vs "previous"). p: the share compared (0-1; held within [1/n, 1 - 1/n],
    so a share of 0 never makes every change count); n: its cells (judged, or paired for a matched change). None
    without both."""
    p, n = number(p), number(n)
    if p is None or n is None or n < 1:
        return None
    low = 1.0 / n
    p = min(max(p, low), 1.0 - low) if n > 2 else 0.5
    value = SAMPLER_Z * qk.value(mode, "sampler_k") * 100.0 * math.sqrt(p * (1.0 - p) / n)
    return value * math.sqrt(2.0) if vs == "previous" else value


def change_threshold(mode: str, p=None, n=None, vs: str = "start") -> float:
    """When a run's depth-matched change counts: the size-aware formula at the run's own p and paired cells, never below
    the largest gap measured between repeat runs' matched changes for its mode (matched_fixed: 1.0 in 100 for regions,
    0.3 for whole samples): a small run gets its larger, size-aware threshold."""
    fixed = qk.value(mode, "matched_fixed")
    formula = sampler_threshold(p, n, vs, mode)
    return fixed if formula is None else max(fixed, formula)


def direction_of(value, threshold) -> str | None:
    """"more" / "fewer" when |value| >= threshold (a change that counts), else "same"; None without both."""
    value, threshold = number(value), number(threshold)
    if value is None or threshold is None:
        return None
    return "more" if value >= threshold else "fewer" if value <= -threshold else "same"


def change_block(raw, matched, mode: str, p=None, paired=None, judged=None, forming: bool = False, now=None,
                 start=None) -> dict:
    """A summary's comparison with the start line (`change`): raw (the unrounded share minus the start's: what the
    headline did), matched (the depth-matched change: what is compared), threshold (change_threshold), model_threshold
    (the mode's matched_model), direction ("fewer" | "more" | "same": |matched| >= threshold one way or the other),
    meaningful (the verdict is "fewer" or "more"), enough (the mode's minimum support: >= 300 judged and >= 400 paired
    cells), per100_now and per100_start (`now`, `start`: the two numbers matched compares, the cells that look mixed per
    100 of those compared now and at the start, at the matched depth; None without them). No verdict (direction None,
    meaningful false) without the matched change, without that support, or while cells form (nothing is compared with
    the start then). A live result's verdict is then held steady (`Steady`: a held word keeps the two numbers it
    compared, matched stays the change's own)."""
    threshold = change_threshold(mode, p, paired)
    enough = supported(judged, paired, mode)
    direction = None if forming or not enough else direction_of(matched, threshold)
    out = {"raw": exact(raw), "matched": exact(matched), "threshold": exact(threshold, 3),
           "model_threshold": qk.value(mode, "matched_model"), "meaningful": direction in ("more", "fewer"),
           "direction": direction, "enough": bool(enough), "per100_now": exact(now), "per100_start": exact(start)}
    if forming:
        out["forming"] = True
    return out


def supported(judged, paired, mode: str = "region") -> bool:
    """The minimum support for a verdict: >= local_min_judged (300) judged and >= local_min_paired (400) paired cells of
    the mode, counted as whole cells (smoothed counts rounded, as the summaries show them)."""
    return round(number(judged) or 0.0) >= qk.value(mode, "local_min_judged") \
        and round(number(paired) or 0.0) >= qk.value(mode, "local_min_paired")


def local_threshold(p, paired, mode: str = "region") -> float | None:
    """An area's own sampler threshold (the size-aware formula with the AREA's own p and
    paired cells, never the tissue's): e.g. 2.1 points at 400 paired cells and p = 0.15."""
    return sampler_threshold(p, paired, "start", mode)


# ---- steady verdicts --------------------------------------------------------------------
STEADY_HOLD_S = 5.0        # a live verdict changes only once the new one has held this long (run clock) ...
STEADY_RESULTS = 2         # ... over at least this many live results ...
STEADY_MAX_S = 15.0        # ... and at the latest once the shown one has not been the change's own for this long
VERDICT_KEYS = ("change",)   # the tissue against its start; areas: "area:<L>"


class Steady:
    """The verdicts a run's live results show (the tissue's `change` and each drawn area's `local`).
    A verdict sits at a calibrated threshold, and a smoothed change hovering at it flipped "fewer" / "about the same"
    every second or two (16-32 flips a minute on a small region, read by a user as effects of a
    setting). So a shown direction changes only once the change's own direction has been the new one for STEADY_HOLD_S
    of live results (at least STEADY_RESULTS of them), and at the latest once the shown one has not been the change's own
    for STEADY_MAX_S. The threshold never moves: the change's own direction (`direction_now`) is always the calibrated
    rule's. No verdict (None: cells forming, too few cells, nothing to compare) is shown at once, and so is the first
    verdict after none: holding back is never held. Scrubbed and single results are judged alone (never passed here).
    A block whose shown direction is an earlier one, still held, says so: held true, its own verdict as direction_now
    and meaningful_now, and held_s how long the change's own has differed from it (a block without `held` shows its own:
    the summaries stay small). A held block also carries the two numbers its word compared (`numbers`: the change's
    per100_now and per100_start; Go deeper's now and start), those of the newest block whose own direction was the one
    shown, so a line that puts them beside the word never contradicts it (a held "fewer" beside a start below now); its
    own value (matched; Go deeper's points) and direction_now stay the newest block's."""

    def __init__(self, numbers=("per100_now", "per100_start")):
        self.verdicts: dict[str, dict] = {}
        self.numbers = tuple(numbers)

    def _shown(self, block: dict) -> dict:
        """A verdict shown as `block`'s own, nothing pending: its direction and the numbers it compared."""
        return {"shown": block.get("direction"), "pending": None, "since": None, "count": 0, "differs": None,
                "numbers": {k: block.get(k) for k in self.numbers}}

    def apply(self, key: str, block, now) -> dict | None:
        """A block ({direction, meaningful, ...}) with the direction to show at run clock `now` (seconds), and while an
        earlier one is held the numbers it compared; no block (nothing compared) forgets the key's verdict."""
        if not isinstance(block, dict):
            self.verdicts.pop(key, None)
            return block
        out = dict(block)
        own = block.get("direction")
        now = number(now)
        record = self.verdicts.get(key)
        if record is None or own is None or record["shown"] is None or own == record["shown"] or now is None:
            self.verdicts[key] = self._shown(block)
            return out
        if record["differs"] is None or now < record["differs"]:
            record["differs"] = now
        if record["pending"] != own or record["since"] is None or now < record["since"]:
            record.update(pending=own, since=now, count=1)
        else:
            record["count"] += 1
        settled = record["count"] >= STEADY_RESULTS and now - record["since"] >= STEADY_HOLD_S - 1e-9
        if settled or now - record["differs"] >= STEADY_MAX_S - 1e-9:
            self.verdicts[key] = self._shown(block)
            return out
        shown = record["shown"]
        out.update(direction=shown, meaningful=shown in ("more", "fewer"), held=True, direction_now=own,
                   meaningful_now=own in ("more", "fewer"), held_s=round(now - record["differs"], 1))
        out.update({k: v for k, v in record["numbers"].items() if k in out})
        return out

    def release(self, key: str, block) -> None:
        """The shown verdict of `key` becomes `block`'s own direction at once, with its numbers, nothing pending (a
        frame shown as its own numbers: one the run rests on, a pause, the hold, a finish or the end, says its own word,
        released from any word held before it); no block (nothing compared) forgets the key's verdict."""
        if not isinstance(block, dict):
            self.verdicts.pop(key, None)
            return
        self.verdicts[key] = self._shown(block)

    def keep(self, prefix: str, names) -> None:
        """Forget the verdicts under `prefix` whose name is not in `names` (areas no longer drawn)."""
        names = {f"{prefix}{n}" for n in names}
        for key in [k for k in self.verdicts if k.startswith(prefix) and k not in names]:
            del self.verdicts[key]

    def reset(self) -> None:
        self.verdicts.clear()


# ---- the trial band (build.trial, on the depth-matched change) -----------------------------------------------------------
def matched_band(entries, floor=None, level: str = "v", change: str = "mc", trial: dict | None = None) -> dict | None:
    """The band "expected without the change" over the live results before it: centre the mean of their smoothed share
    (what the rows show), expected the mean of their depth-matched change (what is compared), half width max(mult (2) x
    the SD of those changes (ddof 1), floor_per100 (1.0 in 100), `floor`); None under min_history_results results with
    both. `trial`: the mode's trial constants (the region's by default)."""
    trial = trial or qk.value("region", "trial")
    pairs = [(number(e.get(level)), number(e.get(change))) for e in entries]
    pairs = [(v, c) for v, c in pairs if v is not None and c is not None]
    if len(pairs) < trial["min_history_results"]:
        return None
    levels = np.asarray([v for v, _ in pairs], np.float64)
    changes = np.asarray([c for _, c in pairs], np.float64)
    centre, expected = float(levels.mean()), float(changes.mean())
    half = max(trial["mult"] * float(changes.std(ddof=1)), trial["floor_per100"], number(floor) or 0.0)
    return {"centre": centre, "expected": expected, "half": half, "low": centre - half, "high": centre + half,
            "n": len(pairs)}


def _area_value(value) -> dict:
    """An area's value at one result: {share, mc, judged, paired} (a bare number is its share)."""
    if isinstance(value, dict):
        return {k: number(value.get(k)) for k in ("share", "mc", "judged", "paired")}
    if isinstance(value, (list, tuple)):
        items = list(value) + [None] * 4
        return {"share": number(items[0]), "judged": number(items[1]), "mc": number(items[2]),
                "paired": number(items[3])}
    return {"share": number(value), "mc": None, "judged": None, "paired": None}


# ---- one live knob change ------------------------------------------------------------------------------------------
class Watch:
    """One live knob change, from the first frame whose live values show it: `changes` [{flag, from, to}], `history`
    the live results before it (QcManager's entries: {seq, iter, phase, v (the smoothed share), raw, mc (the smoothed
    depth-matched change against the start), tx (typical transcripts), areas {L: [share, judged, mc, paired]}}),
    `reveal` the iteration areas are shown from (burn-in + 20; the history starts there), `areas` {L: (bin ids, judged,
    paired)} the areas drawn as it started (their own bands, for the Undo card and a trial's area row and verdict),
    `floor` the run's sampler threshold for the matched change (the band is never narrower), else worked out for the
    run's `mode` from the history's own share and paired cells (its entries' mp, mn). The band, the verdict's minima,
    the horizon and an area's minimum support are the mode's constants (engine/qc_constants.py; the region's without a
    mode).

    Every comparison is on the depth-matched change: the whole tissue's, and an area's own on the bins
    it had when the change started (QcManager `_area_shares`); a row's "now" is the band's centre plus the matched change
    since the change (the share at the same depth), never a share scaled by the tissue's depth ratio."""

    def __init__(self, wid: int, changes: list, start: dict, history: list, reveal: int | None,
                 areas: dict | None = None, floor=None, mode: str | None = None):
        self.id = wid
        self.changes = [{"flag": c["flag"], "from": clean(c.get("from")), "to": clean(c.get("to"))} for c in changes]
        self.start_seq, self.start_iter = int(start["seq"]), int(start.get("iter") or 0)
        self.start_phase = start.get("phase")
        self.reveal = reveal
        self.mode = mode if mode in qk.MODES else "region"
        self.trial = qk.value(self.mode, "trial")
        self.unseen = any(c["flag"] in UNSEEN_FLAGS for c in self.changes)   # a change this view could not judge
        first = self.start_iter - self.trial["history_iterations"]
        before = [e for e in history if e["seq"] < self.start_seq and first <= e["iter"] < self.start_iter
                  and e.get("phase") not in FORMING_PHASES + LAST_STAGE and (reveal is None or e["iter"] >= reveal)]
        if floor is None and mode is not None:          # the run's own threshold before the change
            seen = [(number(e.get("mp")), number(e.get("mn"))) for e in before]
            seen = [(p, n) for p, n in seen if p is not None and n is not None]
            if seen:
                floor = change_threshold(mode, float(np.mean([p for p, _ in seen])), float(np.mean([n for _, n in seen])))
        self.floor = number(floor) or 0.0
        self.history = before                # what the bands are taken from (an area watched later gets its own)
        self.grid: dict | None = None        # the bin grid of `areas` (QcManager's tracker's)
        self.band = matched_band(before, self.floor, trial=self.trial) if self.bandable else None
        levels = [e for e in before if number(e.get("v")) is not None]
        self.history_ok = len(levels) >= self.trial["min_history_results"]   # enough results, with or without matching
        txs = [number(e.get("tx")) for e in before]
        txs = [t for t in txs if t is not None]
        self.tx_expected = float(np.mean(txs)) if txs else None
        self.area_bins: dict[str, np.ndarray] = {}
        self.area_band: dict[str, dict] = {}
        self.area_judged: dict[str, float] = {}
        self.area_paired: dict[str, float | None] = {}
        self.area_start: dict[str, float] = {}
        self.area_ok_now: dict[str, bool] = {}   # the area holds the minimum support NOW (not only at the start)
        for letter, value in (areas or {}).items():
            bins, judged, paired = (list(value) + [None, None])[:3]
            self.add_area(letter, bins, judged, before, paired=paired)
        self.count = 0                       # results after the settling iterations
        self.now: float | None = None        # the whole tissue at the same depth: band centre + matched change since
        self.effect: float | None = None     # the whole tissue's matched change since the change (now - expected)
        self.raw_now: float | None = None    # the smoothed share as the headline has it
        self.tx_now: float | None = None
        self.area_now: dict[str, float] = {}
        self.area_effect: dict[str, float | None] = {}
        self.iter_now = self.start_iter
        self.verdict: str | None = None      # "less" | "more" | None (no clear change yet) | "no_change" (after 300)
        self.final = False                   # past the horizon (or at the last stage): the verdict stays
        self.last_stage = False              # the run reached its final settling stage during the watch
        self.frozen = False                  # another change came: nothing here moves any more ...
        # ... and it was not this change put back, or several knobs changed at once (a trial's "mixed up" note: which
        # change did what cannot be told).
        self.mixed_up = len(self.changes) > 1
        self.pending: tuple[str | None, int] = (None, 0)
        self.above = self.not_above = 0
        self.harm = False
        self.area_pending: dict[str, tuple[str | None, int]] = {}
        self.area_verdict: dict[str, str | None] = {}   # an area with the minimum support: its own verdict
        self.area_above: dict[str, int] = {}
        self.area_not_above: dict[str, int] = {}
        self.area_harm: dict[str, bool] = {}
        self.dismissed = False               # its Undo card was answered, or its trial was kept or undone

    @property
    def bandable(self) -> bool:
        """Whether a band "expected without the change" can exist at all: not while cells form, not for a change this
        view could not see (UNSEEN_FLAGS: none now), not in the last settling stage."""
        return self.start_phase not in FORMING_PHASES + LAST_STAGE and not self.unseen

    def add_area(self, letter: str, bins, judged, history: list | None = None, paired=None):
        """Watch an area too (its bins as drawn when the change started, its judged and paired cells then): its own
        band from its history values (the history before the change by default), never narrower than its own sampler
        threshold."""
        letter = str(letter)
        self.area_bins[letter] = np.unique(np.asarray(list(bins), np.int64))
        self.area_judged[letter] = float(judged or 0)
        self.area_paired[letter] = number(paired)
        values = [_area_value(e["areas"][letter]) for e in (self.history if history is None else history)
                  if letter in (e.get("areas") or {})]
        shares = [v["share"] for v in values if v["share"] is not None]
        floor = local_threshold(np.mean(shares) / 100.0 if shares else None, self.area_paired[letter], self.mode)
        band = matched_band([{"v": v["share"], "mc": v["mc"]} for v in values], floor, trial=self.trial) \
            if self.bandable else None
        if band is not None:
            self.area_band[letter] = band

    def freeze(self, mixed_up: bool):
        """Another change came: the readout stays as it is now ("mixed up" unless it put this change back)."""
        self.frozen = True
        self.mixed_up = self.mixed_up or mixed_up
        self.harm, self.area_harm = False, {}

    @property
    def flag(self) -> str:
        return self.changes[0]["flag"]

    @property
    def forming(self) -> bool:
        """Changed while cells were still forming (init, burn-in, or before burn-in + 20: the line rises by itself, so
        there is no band)."""
        return self.start_phase in FORMING_PHASES or (self.reveal is not None and self.start_iter < self.reveal)

    @property
    def no_history(self) -> bool:
        """Changed once cells had formed, but too soon after for a band (fewer than 10 results from burn-in + 20 in the
        50 iterations before it): the readout says so, not "still forming" (nor for a change this view could not see
        or the last stage, which have no band anyway)."""
        return self.band is None and not self.forming and self.bandable and not self.history_ok

    @property
    def no_matched(self) -> bool:
        """Enough results before the change, but without the depth-matched change (an older GPU worker): no verdict,
        since the plain share alone falls whenever cells lose transcripts."""
        return self.band is None and not self.forming and self.bandable and self.history_ok

    @property
    def horizon_iter(self) -> int:
        return self.start_iter + self.trial["horizon_iterations"]

    @property
    def judging(self) -> bool:
        """Whether more results can still give it a verdict or an Undo card: it has a band (the whole tissue's, or an
        area's own) and is not past its horizon or the last settling stage, not frozen by another change and not
        answered. A band is made only when the change starts (an area's when it is first watched), so a change made
        while cells form or in the last settling stage, one too soon after cells formed, or one on a run without the
        like-for-like change (no start line) is never judged, however long it is watched."""
        return (self.band is not None or bool(self.area_band)) and not (self.final or self.frozen or self.dismissed)

    def supported(self, letter) -> bool:
        """An area holding the minimum support for a verdict of its own: >= 300 judged and >= 400 paired cells as the
        change started, and still in its newest result (an area whose cells moved away or stopped being
        judged has no comparison of its own any more, however many it had at the start)."""
        return supported(self.area_judged.get(letter, 0), self.area_paired.get(letter), self.mode) \
            and self.area_ok_now.get(letter, True)

    def trusted(self, letter) -> bool:
        """An area big enough for its own verdict that has a band of its own (its own history of matched changes)."""
        return letter in self.area_band and self.supported(letter)

    def matches(self, flag: str, value) -> bool:
        return any(c["flag"] == flag and same(c["to"], value) for c in self.changes)

    def _shown(self) -> tuple:
        return (self.verdict, self.harm, tuple(sorted(L for L, h in self.area_harm.items() if h)),
                tuple(sorted((L, v) for L, v in self.area_verdict.items() if v)))

    def update(self, entry: dict, area_now: dict | None = None) -> bool:
        """A live result at or after the change (entry as in the history; `area_now` {L: {share, mc, judged, paired}} on
        the area's starting bins now). Returns whether what the readout or the Undo card shows changed."""
        if self.frozen or self.final or entry["seq"] < self.start_seq:
            return False
        before = self._shown()
        k = int(entry["iter"]) - self.start_iter
        if k > self.trial["horizon_iterations"]:
            self.final = True
            if self.verdict is None:
                self.verdict = "no_change"
            for letter in self.area_band:
                if self.trusted(letter) and self.area_verdict.get(letter) is None:
                    self.area_verdict[letter] = "no_change"
            self.harm = False
            self.area_harm = {}
            return True
        if entry.get("phase") in LAST_STAGE and self.start_phase not in LAST_STAGE:
            self.final = self.last_stage = True        # the final settling stage: the line moves by itself there
            self.harm, self.area_harm = False, {}
            return True
        self.iter_now = int(entry["iter"])
        self.raw_now = number(entry.get("v"))
        mc, band = number(entry.get("mc")), self.band
        self.effect = None if band is None or mc is None else mc - band["expected"]
        self.now = None if self.effect is None else band["centre"] + self.effect
        self.tx_now = number(entry.get("tx"))
        seen = set()
        for letter, value in (area_now or {}).items():
            value = _area_value(value)
            area_band = self.area_band.get(letter)
            seen.add(letter)
            # Its own comparison needs its support NOW too: with too few judged or paired cells left, its
            # matched change rests on a handful of cells and is not compared (no verdict, no Undo card).
            ok_now = supported(value["judged"], value["paired"], self.mode)
            self.area_ok_now[letter] = ok_now
            if area_band is not None and value["mc"] is not None and ok_now:
                self.area_effect[letter] = value["mc"] - area_band["expected"]
                self.area_now[letter] = area_band["centre"] + self.area_effect[letter]
            elif value["share"] is not None:              # no comparison of its own: its share, for the row only
                self.area_effect[letter] = None
                self.area_now[letter] = value["share"]
            if value["share"] is not None:
                self.area_start.setdefault(letter, value["share"])
        if k <= SETTLE_ITERS:
            return False
        self.count += 1
        if band is None or self.now is None or self.count < self.trial["min_results"] \
                or k < self.trial["min_iterations"]:
            return False
        position = "below" if self.now < band["low"] else "above" if self.now > band["high"] else "inside"
        last, run = self.pending
        self.pending = (position, run + 1 if position == last else 1)
        target = {"below": "less", "above": "more", "inside": None}[position]
        if target != self.verdict and self.pending[1] >= STEADY:
            self.verdict = target
        self.above, self.not_above = (self.above + 1, 0) if position == "above" else (0, self.not_above + 1)
        if self.above >= HARM_RESULTS:
            self.harm = True
        elif self.not_above >= HARM_CLEAR:
            self.harm = False
        for letter, area_band in self.area_band.items():
            if letter not in seen:                        # not measured in this result: nothing new to judge it on
                continue
            if not self.trusted(letter) or self.area_effect.get(letter) is None:
                # No comparison of its own now (its support collapsed, or no matched sums): a verdict or an Undo card
                # it had goes, and its runs of results start again.
                self.area_verdict.pop(letter, None)
                self.area_harm.pop(letter, None)
                self.area_pending[letter] = (None, 0)
                self.area_above[letter] = self.area_not_above[letter] = 0
                continue
            share = self.area_now[letter]
            where = "below" if share < area_band["low"] else "above" if share > area_band["high"] else "inside"
            last, run = self.area_pending.get(letter, (None, 0))
            self.area_pending[letter] = (where, run + 1 if where == last else 1)
            target = {"below": "less", "above": "more", "inside": None}[where]
            if target != self.area_verdict.get(letter) and self.area_pending[letter][1] >= STEADY:
                self.area_verdict[letter] = target
            if where == "above":
                self.area_above[letter] = self.area_above.get(letter, 0) + 1
                self.area_not_above[letter] = 0
            else:
                self.area_above[letter] = 0
                self.area_not_above[letter] = self.area_not_above.get(letter, 0) + 1
            if self.area_above[letter] >= HARM_RESULTS:
                self.area_harm[letter] = True
            elif self.area_not_above.get(letter, 0) >= HARM_CLEAR:
                self.area_harm[letter] = False
        return before != self._shown()

    def harm_card(self) -> dict | None:
        """The Undo card's data while the change seems to hurt (and was not answered, nor its trial kept or undone):
        {rule "U", areas, flag, from (the value now), to (the value before: what Undo puts back), changed (the change
        itself), before_per100 (expected), after_per100 (now, at the same depth)}; the whole tissue first, else the
        areas that rose. None for a change this view could not judge (UNSEEN_FLAGS), for several knobs changed at once
        (which one hurt cannot be told) and for a value Live knobs cannot put back ("automatic": e.g. the schedule's
        own temperature)."""
        if self.dismissed or self.frozen or self.final or self.unseen or len(self.changes) > 1:
            return None
        change = self.changes[0]
        if change["from"] is None:
            return None
        base = {"rule": "U", "flag": change["flag"], "from": change["to"], "to": change["from"],
                "changed": dict(change), "watch": self.id}
        if self.harm and self.band is not None and self.now is not None:
            return {**base, "areas": [], "before_per100": share100(self.band["centre"]), "after_per100": share100(self.now)}
        rose = [L for L, h in self.area_harm.items() if h]
        if rose:
            first = rose[0]
            return {**base, "areas": rose, "before_per100": share100(self.area_band[first]["centre"]),
                    "after_per100": share100(self.area_now.get(first))}
        return None

    def band_points(self) -> list | None:
        """The band as the chart draws it: [{iter, expected, low, high}] from the change to the newest result (at most
        the horizon); None without a band."""
        if self.band is None:
            return None
        end = max(self.start_iter, min(self.iter_now, self.horizon_iter))
        b = self.band
        point = {"expected": round(b["centre"], 2), "low": round(max(0.0, b["low"]), 2),
                 "high": round(min(100.0, b["high"]), 2)}
        return [{"iter": self.start_iter, **point}, {"iter": end, **point}]

    def readout(self, area: str | None = None) -> dict:
        """What a trial's readout shows: rows [{kind: area | whole | depth, area?, expected, now, change?}] ("now" at the
        same depth: the band's centre plus the matched change since the change; "change" that matched change), the
        verdict and what it was judged on (`on`: "area" for an area with the minimum support and a band of its own, its
        own band; else "whole"), the band (the whole tissue's, as the chart draws its line), the notes (forming,
        no_history, no_matched, mixed_up, small_area, no_verdict: a change this view could not judge, last_stage), the
        transcripts lost (depth_loss, whole %) and the counts. No band (forming, no_history, no_matched, no_verdict, the
        last stage) or mixed up: no verdict and no band."""
        rows = []
        if area is not None and area in self.area_bins:
            band = self.area_band.get(area)
            expected = band["centre"] if band is not None else self.area_start.get(area)
            row = {"kind": "area", "area": area, "expected": share100(expected), "now": share100(self.area_now.get(area))}
            if self.area_effect.get(area) is not None:
                row["change"] = round(self.area_effect[area], 2)
            rows.append(row)
        if self.band is not None:
            row = {"kind": "whole", "expected": share100(self.band["centre"]), "now": share100(self.now)}
            if self.effect is not None:
                row["change"] = round(self.effect, 2)
            rows.append(row)
        if self.tx_expected is not None:
            rows.append({"kind": "depth", "expected": round(self.tx_expected), "now": None if self.tx_now is None
                         else round(self.tx_now)})
        loss = None
        if self.tx_expected and self.tx_now is not None:
            lost = 100.0 * (1.0 - self.tx_now / self.tx_expected)
            loss = int(round(lost)) if lost >= 5 else None
        small = area is not None and area in self.area_bins and not self.supported(area)
        last = self.last_stage or self.start_phase in LAST_STAGE
        notes = [name for name, on in (("forming", self.forming), ("no_history", self.no_history),
                                       ("no_matched", self.no_matched), ("mixed_up", self.mixed_up),
                                       ("small_area", small), ("no_verdict", self.unseen), ("last_stage", last)) if on]
        judged = self.band is not None and not self.mixed_up
        own = judged and area is not None and self.trusted(area)
        verdict = (self.area_verdict.get(area) if own else self.verdict) if judged else None
        return {"rows": rows, "verdict": verdict, "on": "area" if own else "whole", "notes": notes,
                "forming": self.forming, "mixed_up": self.mixed_up, "small_area": small, "depth_loss": loss,
                "band": self.band_points() if judged else None, "results": self.count,
                "horizon_iter": self.horizon_iter, "final": self.final}


# ---- the rules ------------------------------------------------------------------------------------------------------
# A start line asked of a run that has none, refused in these words by how its cells started (start_of: only "cells"
# has a start line), with reason "no_start" and its start_kind: Go deeper's start line (engine/deeper.py) and the check's
# start routes (engine/qc.py) alike.
NO_START_WORDS = {"nuclei": "This run started from nuclei only: it has no start line",
                  "paired": "This run started from cells with their nuclei: it has no start line"}


def start_of(meta: dict) -> str:
    """How a run's cells started: "nuclei" (nuclei only: no starting cells),
    "cells" (starting cells without nuclei: every transcript inside one seeds it, so each spot holding one is held to
    that cell; zarr_bridge/helper.py prepare's "seeded_from", which also says so for nuclei that flag nothing there) or
    "paired" (cells that grow from their nuclei: a Xenium bundle's own, a SpatialData sample's cells with nuclei). R5
    reads it (`evaluate`'s ctx "start"), and so does the test of a setting (engine/qc_test.py)."""
    prior = meta.get("prior")
    if meta.get("sample_kind") != "zarr" or not isinstance(prior, dict):
        return "paired"
    if not prior.get("cells"):
        return "nuclei"
    seeded = (meta.get("prepared") or {}).get("seeded_from")
    if seeded is not None:
        return "cells" if seeded == "cells" else "paired"
    return "paired" if prior.get("nuclei") else "cells"


def timing_of(phase) -> str:
    """"early" while cells form (init, burn-in), else "later"."""
    return "early" if phase in FORMING_PHASES else "later"


def actions_of(rule: str, timing: str) -> tuple[list[str], str | None]:
    """The card's buttons and its main one. Shape knobs (R2, R4, R5): Try it while cells form, then New run with this
    setting first (Try it stays offered); drift (R3): Try it (it acts after burn-in, and proseg's count shows what it
    moves back, so its trial gets a band and a verdict); U: Undo it, Keep; R1: show the starting outlines, then look at
    its cells; R6 and R7 have no knob: look at the area's cells ("inspect": Go deeper on a cell there, its marker genes:
    the cause of a persistent mix is not known, so the card points to where it can be seen)."""
    if rule == "U":
        return ["undo", "keep"], "undo"
    if rule == "R1":
        return ["show_start", "inspect"], "show_start"
    if rule == "R3":
        return ["try", "why"], "try"
    if rule in SHAPE_RULES:
        return (["try", "new_run", "why"], "try") if timing == "early" else (["new_run", "try", "why"], "new_run")
    if rule in ("R6", "R7"):
        return ["inspect"], "inspect"
    return [], None


def _rose(area: dict) -> bool:
    return area.get("tag") == "rose"


def _squeezed(area: dict) -> tuple[bool, int | None]:
    """R5's evidence: (squeezed, k cells shrunk to almost nothing): >= TINY_MORE more tiny cells than at the start, or
    >= SHRUNK_SHARE of its cells lost more than half their starting area."""
    tiny, tiny_start, shrunk, cells = (number(area.get(k)) for k in ("tiny", "tiny_start", "shrunk", "cells"))
    more = None if tiny is None else tiny - (tiny_start or 0.0)
    if more is not None and more >= TINY_MORE:
        return True, int(round(more))
    if shrunk is not None and cells and shrunk >= SHRUNK_SHARE * cells:
        return True, int(round(shrunk))
    return False, None


def _area_rule(area: dict, ctx: dict) -> dict | None:
    """The first rule that matches one confirmed area (R0 already passed), as {rule, flag?, to?, ...numbers}."""
    letter, live = area.get("L"), ctx.get("live") or {}
    growth = number(area.get("growth"))
    harm = ctx.get("harm")
    if harm and letter in (harm.get("areas") or []):
        return {"rule": "U"}
    for tried in ctx.get("tried") or ():
        if tried.get("area") == letter and tried.get("verdict") in ("more", "no_change"):
            return {"rule": "R7", "tried": {k: tried.get(k) for k in ("flag", "from", "to")}}
    if area.get("tag") == "already" and growth is not None and growth <= GROWTH:
        return {"rule": "R1"}
    compact = number(live.get("cell-compactness"))
    loosened = compact is not None and compact > COMPACT_STANDARD * (1 + 1e-6)
    if loosened and (_rose(area) or (growth is not None and growth > GROWTH)):
        # Its sentence says what is true of the area: grown past the start, or got more mixed in this run.
        return {"rule": "R2", "flag": "cell-compactness", "from": compact, "to": COMPACT_STANDARD, "growth": growth,
                "rose": _rose(area)}
    if live.get("no-diffusion") is True:
        return {"rule": "R3", "flag": "no-diffusion", "from": True, "to": False}
    if not _rose(area) or growth is None:
        return None
    if growth >= GROWTH and compact is not None and compact >= COMPACT_R4_MIN * (1 - 1e-6):
        return {"rule": "R4", "flag": "cell-compactness", "from": compact, "to": COMPACT_TIGHT, "growth": growth}
    squeezed, k = _squeezed(area)
    if growth < GROWTH and squeezed:
        # Never a firmer hold for a run started from cells only: there every transcript inside
        # a starting cell holds its spot to that cell, so a firmer hold keeps each starting outline with any neighbouring
        # RNA inside it. Nor a looser one from this evidence: a looser hold lets small cells shrink further (in tests, a
        # quarter as strong made cells 14 % smaller; in a sample started from cells only, small cells were absorbed at
        # 0.1).
        if ctx.get("start") == "cells":
            return None
        # Unreachable for now: a risen area needs a start line, which only a start from cells alone has, and that start
        # is refused just above. A later base for starts with nuclei (a "since cells formed" line, DESIGN section 9 D1)
        # would reach it; the rule is kept for that.
        weight = number(live.get("prior-weight"))
        firmer = None if weight is None else min(2 * weight, PRIOR_WEIGHT_MAX)
        if weight is not None and weight > 0 and firmer > weight and not same(firmer, weight):   # at its bound: no card
            return {"rule": "R5", "flag": "prior-weight", "from": weight, "to": firmer, "k": k}
        return None
    churn = number(area.get("churn"))
    if -GROWTH <= growth <= GROWTH and not squeezed and (churn is None or churn <= CHURN_MOVING):
        return {"rule": "R6"}
    return None


def _whole_rule(rule: str, ctx: dict) -> dict | None:
    """R2 and R3 once for the whole tissue (they are about settings, not one place). R2 says cells reach further into
    their neighbours: only once they have formed (`whole` is empty before). R3's words hold whenever drift is off, and
    drift acts only after burn-in: while cells form, the start line's areas on screen count as flagged (`flagged`), so
    it can be tried in time; once cells formed, also when the whole tissue is more mixed than at the start by a change
    that counts (`whole["change"]`: the depth-matched change against its threshold, change_block)."""
    live, whole = ctx.get("live") or {}, ctx.get("whole") or {}
    if rule == "R2":
        compact, growth = number(live.get("cell-compactness")), number(whole.get("growth"))
        if compact is not None and compact > COMPACT_STANDARD * (1 + 1e-6) and \
                (whole.get("rose") or (growth is not None and growth > GROWTH)):
            return {"rule": "R2", "flag": "cell-compactness", "from": compact, "to": COMPACT_STANDARD, "growth": growth,
                    "rose": bool(whole.get("rose"))}
    elif rule == "R3" and live.get("no-diffusion") is True:
        change = whole.get("change") if isinstance(whole.get("change"), dict) else {}
        above = bool(change.get("meaningful")) and change.get("direction") == "more"
        if ctx.get("areas") or ctx.get("flagged") or above:
            return {"rule": "R3", "flag": "no-diffusion", "from": True, "to": False}
    return None


def card_id(card: dict) -> str:
    """A card's identity: its rule and the change it offers, for R7 the change that was tried (areas may join or leave
    it), for U also the change it answers (`watch`: a later change of the same knob is another card). The page keys the
    card, the idea it picked and what it set aside by it."""
    change = card.get("tried") if card["rule"] == "R7" and isinstance(card.get("tried"), dict) else card
    key = f"{card['rule']}:{change.get('flag') or ''}:{clean(change.get('to'))}"
    return f"{key}:{card.get('watch')}" if card["rule"] == "U" else key


def allowed(card: dict) -> bool:
    """Never suggested for mixing: lowering Prior weight, switching drift off, raising drift distances
    or the far-drift share, cooling, more gap proposals, any startup-only setting as a one-click change; nor a change
    that changes nothing (a knob at its bound). U puts back what the user changed, whatever it was (drift switched off
    too), but never a change this view could not judge (UNSEEN_FLAGS: none now)."""
    if card.get("rule") == "U":
        return card.get("flag") not in UNSEEN_FLAGS
    if not card.get("flag"):
        return True
    flag, old, new = card["flag"], number(card.get("from")), number(card.get("to"))
    if flag == "no-diffusion":
        return card.get("to") is False and card.get("from") is True
    if old is None or new is None or same(old, new):
        return False
    if flag == "prior-weight":
        return new > old
    if flag == "cell-compactness":                   # tighter, or back to the standard from looser: never looser
        return new < old
    if flag in ("diffusion-sigma-near", "diffusion-sigma-far", "diffusion-sigma-z", "diffusion-probability",
                "ab-nihlo-bubble-prob"):
        return new < old
    if flag == "temperature":
        return new > old
    return False                                     # anything else, startup-only settings included: never offered


def evaluate(ctx: dict) -> list[dict]:
    """The cards that match now, the one to show first. ctx: phase and iter of the result; min_judged (R0: 40 on 100 µm
    bins, 30 on 50 µm); live (the frame's knob values); areas (the confirmed areas in the order the page lists them, as
    in the summary: L, tag, judged, growth, tiny, tiny_start, shrunk, cells, churn, ...); whole {per100, start_per100,
    rose, growth, change (change_block)} (empty while cells form); flagged (while cells form: the start line's areas are on screen); tried
    (ended trials: [{area, flag, from, to, verdict}]); harm (the Undo card's data, Watch.harm_card);
    later_s (1,000 iterations of this run, in seconds); start (how the run's cells started, `start_of`: R5 offers no
    hold for "cells"). Every card: {id, rule, areas, flag?, from?, to?, confidence,
    timing, actions, main, later_s?} plus its numbers (growth and rose: R2, whose sentence also names the loosened
    value, its `from`; growth: R4; k: R5; tried: R7; changed, before_per100, after_per100 and watch: U)."""
    phase, iteration = ctx.get("phase"), number(ctx.get("iter"))
    if phase == "init" or iteration is None or iteration < TOO_EARLY_ITER:      # R0: too early
        return []
    timing = timing_of(phase)
    harm = ctx.get("harm")
    cards, by_id = [], {}

    def add(found: dict, letter: str | None):
        card = {**found, "areas": [] if letter is None else [letter]}
        key = card_id(card)
        if key in by_id:
            if letter is not None and letter not in by_id[key]["areas"]:
                by_id[key]["areas"].append(letter)
            return
        actions, main = actions_of(card["rule"], timing)
        card.update(id=key, confidence=CONFIDENCE[card["rule"]], timing=timing, actions=actions, main=main)
        if card["rule"] in ("R2", "R3", "R4", "R5") and timing == "later" and number(ctx.get("later_s")) is not None:
            card["later_s"] = round(float(ctx["later_s"]))
        for name in ("from", "to"):
            if name in card:
                card[name] = clean(card[name])
        if allowed(card):
            by_id[key] = card
            cards.append(card)

    if harm and not harm.get("areas"):               # the whole tissue rose after the user's change: first
        add({k: v for k, v in harm.items() if k != "areas"}, None)
    min_judged = number(ctx.get("min_judged")) or 40.0
    for area in ctx.get("areas") or ():
        if (number(area.get("judged")) or 0.0) < min_judged:                    # R0: too few cells to judge
            continue
        found = _area_rule(area, ctx)
        if found is None:
            continue
        if found["rule"] == "U":
            found = {k: v for k, v in harm.items() if k != "areas"}
        add(found, area.get("L"))
    for rule in ("R2", "R3"):
        if not any(c["rule"] == rule for c in cards):
            found = _whole_rule(rule, ctx)
            if found is not None:
                add(found, None)
    return cards


# ---- one card at a time, steady --------------------------------------------------------------------------------------
class CardKeeper:
    """Which card the page shows (the rest go under "More ideas"): see the module docstring. `t` is the run clock of the
    result (s), or None when frames carry no time (then each result counts as a second)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.shown: dict | None = None
        self.lead: tuple[str | None, int, float | None] = (None, 0, None)    # (card id, results, since)
        self.gone: tuple[int, float | None] = (0, None)
        self.clock = 0.0

    def _time(self, t) -> float:
        t = number(t)
        self.clock = t if t is not None else self.clock + 1.0
        return self.clock

    @staticmethod
    def _held(streak: tuple, rule: tuple, now: float) -> bool:
        results, since = streak
        return results >= rule[0] and since is not None and now - since >= rule[1] - 1e-9

    def choose(self, cards: list[dict], t=None, drawn=()) -> tuple[dict | None, list[dict]]:
        """(the card shown, the more ideas) after one live result: `cards` from evaluate, `drawn` the letters drawn."""
        now = self._time(t)
        top = cards[0] if cards else None
        key, results, since = self.lead
        self.lead = (None, 0, None) if top is None else \
            ((key, results + 1, since) if top["id"] == key else (top["id"], 1, now))
        current = None if self.shown is None else next((c for c in cards if c["id"] == self.shown["id"]), None)
        if self.shown is not None and current is None and self.shown["rule"] == "U":
            self.shown, self.gone = None, (0, None)          # the harm is gone, undone or answered "Keep": so is U
        if top is not None and top["rule"] == "U" and (self.shown is None or self.shown["id"] != top["id"]):
            self.shown, self.gone = top, (0, None)                              # the harm card: at once
        elif self.shown is not None and current is not None:
            self.shown, self.gone = current, (0, None)
            if top["id"] != current["id"] and self._held(self.lead[1:], CARD_SWAP, now):
                self.shown = top
        elif self.shown is not None:
            letters = set(self.shown.get("areas") or [])
            if letters and not letters & set(drawn):
                self.shown = None                                                # its areas are gone: so is the card
            else:
                n, since = self.gone
                self.gone = (n + 1, since if since is not None else now)
                if self._held(self.gone, CARD_GONE, now):
                    self.shown, self.gone = None, (0, None)
        if self.shown is None and top is not None and self._held(self.lead[1:], CARD_ENTER, now):
            self.shown, self.gone = top, (0, None)
        if self.shown is None:
            return None, []
        more = [c for c in cards if c["id"] != self.shown["id"]][:MORE_IDEAS]
        return self.shown, more


# ---- timing ----------------------------------------------------------------------------------------------------------
def seconds_per_iteration(frames: list[dict], last: int = 60) -> float | None:
    """This run's pace once cells formed: the run clock over the iterations of its newest formed frames (up to `last`),
    None without times."""
    formed = [f for f in frames[-max(2, last):] if f.get("phase") not in FORMING_PHASES
              and number(f.get("t")) is not None and number(f.get("iter")) is not None]
    if len(formed) < 2:
        return None
    di = float(formed[-1]["iter"]) - float(formed[0]["iter"])
    dt = float(formed[-1]["t"]) - float(formed[0]["t"])
    return dt / di if di > 0 and dt > 0 else None
