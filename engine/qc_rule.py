"""Test a setting here: the measured constants, the pool's order and rule, and the rule that answers a test (pure; numpy
only). engine/qc_test.py calls `answer` on the runs' readouts (engine/qc_readout.py) over the targets fixed before any
run (engine/qc_pool.py); the page draws what it returns as it is and never decides anything again.

What the test is: 2N fresh runs of one area from one shared input, N as now (side A: roles A1..AN) and N with the setting
(side B), each to the end of proseg's default schedule. Per run r, over the pool's suspects (T: the likely suspects of
mixed cells whose suspects sat inside their own starting outline, each from a neighbour placed inside the area, chosen
on proseg's own count at the parent's last frame: engine/qc_pool.py), their matched own-RNA controls (M: 3 per suspect,
of the same cell, giver, distance to it and band of proseg's share) and the pooled cells' own RNA (U):
  H_T,r, H_M,r, H_U,r = the mean share still in their cell (HELD); X_r = H_T,r - H_M,r.
The gate: the side A mean of H_T reaches `gate` (else the suspects come and go between identical runs: nothing to
judge). The rule (`rule`, per start): "excess" judges X, "suspects" judges H_T alone; it says "clearly fewer" when every
B run lies below every A run and the mean gap is wider than `tau`, a gap identical runs rarely exceed.
Both forms are kept for the record; only `rule`'s decides. The numbers are the measured ones, per start and per
situation (the area re-run as it is, or a 500 µm square around the cell). A test the measurements did not cover is
judged all the same with the start's nearest measured row (`row_for`) and carries caveats saying so (engine/qc_pool.py
reasons of kind "caveat", which engine/qc_test.py hands to `answer`); a pool under k is judged too. Every decision is
made on unrounded values."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .qc_readout import Items, RunReadout

# Changes whenever a number below changes; every pool reply and result echoes it.
CONSTANTS_ID = "proseg-2026-10-31"
# The constants measured with suspects chosen on the drawn outlines' counts (phase 1a), not on proseg's own count, which
# a test's suspects come from now: a test judged with one of these carries the caveat count_basis (engine/qc_pool.py),
# and so does one judged with a row of these constants that was not measured again on proseg's count (its `measured`
# false: `measured_on_proseg`).
OUTLINE_IDS = ("phase1a-2026-10-29",)
# What the measurements held fixed. Outside them a test is judged with the nearest row and says so (engine/qc_pool.py
# run_reasons, its caveats). Its suspects were chosen at the parent's last frame, held there with proseg's exact shares
# (where exports), each kept at proseg's averaged keep e / n.
MEASURED: dict = {
    "thresholds": {"mixed_at": 0.30, "judge_min": 100, "fit_min": 20},     # the mixing check's, as measured
    "schedule": {"burnin": 200, "samples": 200, "recorded": 100, "hillclimb": 50, "morphology_steps": 4000},
    "area_um": 500.0, "inset_um": 25.0, "strong_permille": 650,           # area_um: the square's side
    "controls": {"seed": 20261028, "matched_per_target": 3, "uniform_per_cell": 50},
    "likely_min": 0.5, "glow_min": 0.2, "deeper_k": 10,
    # likely's least w (engine/qc_scoring.py LIKELY_W) and the controls' bands of w (engine/qc_pool.py W_BANDS)
    "likely_w": 0.5, "w_bands": (0.25, 0.75),
}
STARTS = ("paired", "cells")          # qc_rules.start_of: nuclei and cells, or cells only ("nuclei": row_for's paired)
FORMS = ("excess", "suspects")
SITUATIONS = ("as_is", "square")      # the rows measured (a test's own area may also be a "crop": row_for)
DROP = 0.25                           # power: a drop of the suspects alone by this share, their own RNA unchanged
RESULT_V = 2                          # the result's version ({test}.result "v")
WENT = ("stay", "neighbour", "other", "background")    # where the suspects went: the readout's HELD, GIVER, OTHER, BG
# One row per start, one sub-row per situation. A sub-row's `refusal` (a reason code) is set exactly when it is not
# offered: a test there is judged with the start's offered row instead (row_for) and carries that code as a caveat.
# `measured`: the sub-row was measured again with suspects chosen on proseg's own count (2026-10-31: the parents held
# at their last frame with proseg's exact shares, the stored fresh runs of their squares and regions read again); a
# test judged with a sub-row that was not carries the caveat count_basis (engine/qc_pool.py).
# `gate`: the side A mean of H_T a test needs; `tau`: each form's gap, a high percentile of identical runs' gaps;
# `rates` (what the page's tooltip says, all from the same measurement): fp_mean and fp_worst, how often identical runs
# came out "clearly fewer" on average and in the worst of `areas` areas; power, how often a drop of the suspects alone by
# `drop` was caught; shrink_fire, how often a pure shrink (suspects and their own RNA thinned alike) still fired. The
# rates are those of pools that reached k suspects: a pool short of k is judged all the same, and its answer says the
# rates are for k suspects or more (engine/qc_pool.py too_few). `short_rates`: such short pools' own rates where they
# were measured apart (a start from cells only, regions re-run as they are: the same measurement, on `tests` test cells
# of one sample in `areas` areas; the page's tooltip adds them after too_few's words), None elsewhere.
CONSTANTS: dict[str, dict] = {
    # Starts from nuclei and cells: phase 1a's, measured with suspects chosen on the outlines (run again on proseg's
    # count, the procedure's choice of k came back in only 56 % of resamplings, under the 80 % it needs).
    "paired": {"runs_per_side": 3, "rule": "excess", "k": 160, "shrink_share": 1 / 3,
               "area_min_um": 300.0, "as_is_max_um": 500.0,
               "situations": {
                   "as_is": {"offered": True, "refusal": None, "measured": False, "gate": 0.5,
                             "tau": {"excess": 0.08938564995169271, "suspects": 0.08527235239755926},
                             "rates": {"fp_mean": 0.007883249723108996, "fp_worst": 0.018585604032131046, "areas": 2,
                                       "power": 0.9219949182357158, "drop": DROP,
                                       "shrink_fire": 0.01849957651964297},
                             "short_rates": None},
                   "square": {"offered": True, "refusal": None, "measured": False, "gate": 0.5,
                              "tau": {"excess": 0.08938564995169271, "suspects": 0.08527235239755926},
                              "rates": {"fp_mean": 0.012977015334507574, "fp_worst": 0.032999999999999995, "areas": 3,
                                        "power": 0.7450488713027039, "drop": DROP,
                                        "shrink_fire": 0.010664357070356694},
                              "short_rates": None}}},
    # Starts from cells only: measured on proseg's own count, both situations (the square's rates on pools seeded
    # within 100 µm of its centre, each square judged with the tau of the others).
    "cells": {"runs_per_side": 4, "rule": "excess", "k": 160, "shrink_share": 1 / 3,
              "area_min_um": 300.0, "as_is_max_um": 500.0,
              "situations": {
                  "as_is": {"offered": True, "refusal": None, "measured": True, "gate": 0.5,
                            "tau": {"excess": 0.0655782303402374, "suspects": 0.06116140846464757},
                            "rates": {"fp_mean": 0.005445159465384341, "fp_worst": 0.014412683161181838, "areas": 2,
                                      "power": 0.9761685877943569, "drop": DROP,
                                      "shrink_fire": 0.10115267661410085},
                            # pools short of k, measured apart (the regions re-run as they are)
                            "short_rates": {"fp_mean": 0.028426395939086288, "fp_worst": 0.037914691943127965,
                                            "power": 0.21962774957698816, "drop": DROP,
                                            "shrink_fire": 0.042064297800338414, "tests": 8, "areas": 4}},
                  "square": {"offered": True, "refusal": None, "measured": True, "gate": 0.5,
                             "tau": {"excess": 0.0655782303402374, "suspects": 0.06116140846464757},
                             "rates": {"fp_mean": 0.011264073139657632, "fp_worst": 0.013274336283185839, "areas": 3,
                                       "power": 0.8831359560583003, "drop": DROP,
                                       "shrink_fire": 0.02605210981315147},
                             "short_rates": None}}},
}


def constants_for(start: str, situation: str) -> dict:
    """What a test of a run started `start` in `situation` is judged with: {id, start, situation, offered, refusal, rule,
    k, runs_per_side, shrink_share, gate, tau (the rule's own tau there), taus {excess, suspects}, rates (None where
    not offered), short_rates (pools short of k, where measured apart; else None)}, a copy. KeyError for an unknown
    start ("nuclei") or situation ("narrow", "lasso")."""
    row = CONSTANTS[start]
    sub = row["situations"][situation]
    short = sub.get("short_rates")
    return {"id": CONSTANTS_ID, "start": start, "situation": situation, "offered": sub["offered"],
            "refusal": sub["refusal"], "rule": row["rule"], "k": row["k"], "runs_per_side": row["runs_per_side"],
            "shrink_share": row["shrink_share"], "gate": sub["gate"], "tau": sub["tau"][row["rule"]],
            "taus": dict(sub["tau"]), "rates": None if sub["rates"] is None else dict(sub["rates"]),
            "short_rates": None if short is None else dict(short)}


def offered(start: str, situation: str) -> bool:
    return bool(CONSTANTS[start]["situations"][situation]["offered"])


def measured_on_proseg(start: str, situation: str) -> bool:
    """Whether a test judged with this row rests on numbers measured with suspects chosen on proseg's own count: the
    constants are not the outlines' (OUTLINE_IDS) and the row was measured again (its `measured`). A test judged with
    one that is not carries the caveat count_basis (engine/qc_pool.py). KeyError for an unknown start or situation."""
    return bool(CONSTANTS[start]["situations"][situation]["measured"]) and CONSTANTS_ID not in OUTLINE_IDS


def refusal_of(start: str, situation: str) -> str | None:
    """The reason code of a situation that was not measured (engine/qc_pool.py REASON_WORDS, a caveat); None when it
    is offered."""
    return CONSTANTS[start]["situations"][situation]["refusal"]


def area_min_um(start: str) -> float:
    """A region re-run as it is must be at least this wide on its short side."""
    return float(CONSTANTS[start]["area_min_um"])


def as_is_max_um(start: str) -> float:
    """A region at most this wide on its long side is re-run as it is; a larger one gets a square of this side."""
    return float(CONSTANTS[start]["as_is_max_um"])


def row_for(start: str, region_bbox) -> tuple[str, str]:
    """The measured row (start, situation) a test of a run started `start` over a region of this bounding box [x0, y0,
    x1, y1] (µm) is judged with, the nearest one where the measurements did not cover it: a start from cells only, or
    from nuclei and cells, its own; from nuclei only (none measured), as one from nuclei and cells. The square row for
    a region over as_is_max_um across (one way is enough), the as-is row otherwise; a row not offered gives way to the
    start's as-is row. KeyError for any other start."""
    if start not in ("paired", "nuclei", "cells"):
        raise KeyError(start)
    start = "cells" if start == "cells" else "paired"
    x0, y0, x1, y1 = (float(v) for v in region_bbox)
    situation = "square" if max(x1 - x0, y1 - y0) > as_is_max_um(start) else "as_is"
    return (start, situation if offered(start, situation) else "as_is")


# ---- the pool ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PoolOrder:
    """The order cells join a pool in (engine/qc_pool.py fixes it before any run; `pool_of` takes from it after)."""
    seed: int                          # the clicked cell: an index into Items.cell_names
    cells: tuple[int, ...]             # the seed first, then the cells of its pattern by distance between their centres
                                       # at the parent's last frame (ties by parent cell number): members, then the
                                       # reserve; Items.cell_names indices
    distance_um: tuple[float, ...]     # each one's distance from the seed (0.0 for the seed)
    parent_cells: tuple[int, ...]      # each one's cell number in the parent (label - 1)


def pool_of(n, order: PoolOrder, k: int) -> tuple[list[int], int]:
    """(members, their suspects summed): the seed, then the cells of `order` that have a suspect (n >= 1), in that order,
    no cell added once the sum reached `k`. n[i]: the eligible suspects of Items cell i. ([], 0) when the seed has none.
    A pool that never reached `k` holds every cell of `order` that has one (judged all the same: engine/qc_test.py
    says it is short, a caveat)."""
    n = np.asarray(n)
    seed = int(order.seed)
    if not n[seed] >= 1:
        return [], 0
    members, total = [seed], int(n[seed])
    for cell in order.cells:
        cell = int(cell)
        if cell == seed or not n[cell] >= 1:
            continue
        if total >= k:
            break
        members.append(cell)
        total += int(n[cell])
    return members, total


def inside_inner(x, y, inner) -> np.ndarray:
    """Whether each point (µm) lies in the inner area [x0, y0, x1, y1]: its edges included; NaN is never inside."""
    x0, y0, x1, y1 = (float(v) for v in inner)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


# ---- the rule ----------------------------------------------------------------------------------------------------------
def judge(gate_values, a, b, g: float, tau: float | None) -> dict:
    """One form of the rule on one test: {passed (the mean of `gate_values`, side A's H_T, reaches `g`), separated (every
    b below every a), gap (mean a - mean b), fires (passed, separated and the gap wider than `tau`; None without a
    tau), why_not (why it did not fire: "apart" when b and a are not separated, "gap" when they are but the gap is no
    wider than tau; None when it fired or the gate failed, as a failed gate answers "comes and goes")}. Unrounded floats
    in, Python floats and bools out."""
    gate = np.asarray(gate_values, np.float64)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    passed = bool(float(gate.mean()) >= g)
    separated = bool(float(b.max()) < float(a.min()))
    gap = float(a.mean() - b.mean())
    fires = None if tau is None else bool(passed and separated and gap > tau)
    why_not = None if (not passed or fires) else ("apart" if not separated else "gap")
    return {"passed": passed, "separated": separated, "gap": gap, "fires": fires, "why_not": why_not}


def answer(items: Items, order: PoolOrder, readouts: Mapping[str, RunReadout], sides: Mapping[str, Sequence[str]],
           start: str, *, setting: dict, situation: str, at_iter: int,
           matched_counts: Mapping[str, Mapping[str, int]], parent_defaults: bool,
           constants: dict | None = None, row: tuple[str, str] | None = None, caveats=()) -> dict:
    """The test's result ({test}.result, v 2), complete: engine/qc_test.py stores it as it is. `start` and `situation`
    are the test's own (a start from nuclei only, a "crop" area); `row`: the measured row (start, situation) it is
    judged with (row_for's; (start, situation) unless given); `sides`: {"A": [roles], "B": [roles]}, each
    runs_per_side long (else ValueError); `readouts`: each role's RunReadout; `matched_counts`: the targets' matching on
    the shared input ({"target" | "matched" | "uniform": {"ok", "not_found", "unsure"}}); `parent_defaults`: whether
    the parent ran with proseg's default settings (engine/qc_test.py parent_defaults); `constants`: constants_for(*row)
    unless given (ValueError when that row is not offered); `caveats`: the reasons (dicts) the result carries as given,
    in their order.

    In this order: an item counts only where it was read in every run of the test (status OK in all 2N); n per cell =
    its eligible suspects, and the pool is taken again from `order` on them (pool_of: under k, every cell that has
    one); T = the members' eligible suspects, M = their matched controls read everywhere, U = the members' own RNA read
    everywhere; per run H_T, H_M, H_U (plain means) and X = H_T - H_M; the gate on side A's H_T; both forms judged,
    `rule`'s deciding. The verdict: too_few (why: cell_lost, the clicked cell has no suspect left; no_controls, the
    excess rule has no matched control), comes_and_goes (the gate failed), clearly_fewer (the rule fired) or
    not_clear. A pool under k is judged as any other (engine/qc_test.py adds its caveat). Every decision is made on
    unrounded values; the result holds them rounded, and never a NaN."""
    from . import qc_readout as qr          # (the readout imports this module's inside_inner)
    row = (start, situation) if row is None else (str(row[0]), str(row[1]))
    constants = constants_for(*row) if constants is None else constants
    if not constants.get("offered"):
        raise ValueError(f"not measured for {row[0]} starts in the {row[1]} situation")
    rule, per_side, k = constants["rule"], int(constants["runs_per_side"]), int(constants["k"])
    if rule not in FORMS:
        raise ValueError(f"unknown rule {rule!r}")
    if set(sides) != {"A", "B"} or any(len(sides[side]) != per_side for side in ("A", "B")):
        raise ValueError(f"a test has {per_side} runs on each side: {dict(sides)}")
    side_a, side_b = [str(role) for role in sides["A"]], [str(role) for role in sides["B"]]
    roles = side_a + side_b
    if len(set(roles)) != len(roles) or any(role not in readouts for role in roles):
        raise ValueError(f"every run of the test read once: {roles}")
    count, ncells = len(items), len(items.cell_names)
    reads = [readouts[role] for role in roles]
    if any(len(np.asarray(getattr(read, name))) != count for read in reads
           for name in ("status", "held", "giver", "other", "bg")):
        raise ValueError(f"a readout has one value per item ({count})")
    seed = int(order.seed)
    place = {int(c): p for p, c in enumerate(order.cells)}           # an Items cell's place in the order
    if seed not in place or not all(0 <= c < ncells for c in place):
        raise ValueError("the pool's order has its seed, and cells of the items only")
    status = np.stack([np.asarray(read.status) for read in reads])                             # [2N, items]
    shares = np.stack([np.stack([np.asarray(getattr(read, name), np.float64)
                                 for name in ("held", "giver", "other", "bg")], axis=1)
                       for read in reads])                                                       # [2N, items, 4]
    a, b = slice(0, per_side), slice(per_side, 2 * per_side)

    # The items read in every run; n per cell; the pool taken again on them; T (in pool order), M and U.
    eligible = (status == qr.OK).all(axis=0)
    kind, cell, target = (np.asarray(v).astype(np.int64) for v in (items.kind, items.cell, items.target))
    suspect = (kind == qr.SUSPECT) & eligible
    n = np.bincount(cell[suspect], minlength=ncells)
    members, total = pool_of(n, order, k)
    rank = np.full(ncells, -1, np.int64)
    rank[np.asarray(members, np.int64)] = np.arange(len(members))
    t = np.flatnonzero(suspect & (rank[cell] >= 0))
    t = t[np.argsort(rank[cell[t]], kind="stable")]
    in_t = np.zeros(count, bool)
    in_t[t] = True
    m = (kind == qr.MATCHED) & eligible
    m[m] = in_t[target[m]]
    u = (kind == qr.UNIFORM) & eligible & (rank[cell] >= 0)

    # Per run: the mean share still in the cell over T, M and U, plain means with every item weighing the same (in
    # H_M a suspect that kept one of its matched controls counts a third as much as one that kept all three).
    h_t = shares[:, t, 0].mean(axis=1) if t.size else None
    h_m = shares[:, m, 0].mean(axis=1) if m.any() else None
    h_u = shares[:, u, 0].mean(axis=1) if u.any() else None

    # The verdict, in its order.
    gate_at, taus = float(constants["gate"]), constants["taus"]
    why = "cell_lost" if not n[seed] >= 1 else "no_controls" if rule == "excess" and h_m is None else None
    gate = forms = chosen = None
    verdict = "too_few"
    if why is None:
        gate = judge(h_t[a], h_t[a], h_t[b], gate_at, None)
        forms = {"suspects": judge(h_t[a], h_t[a], h_t[b], gate_at, taus["suspects"]),
                 "excess": None if h_m is None else judge(h_t[a], (h_t - h_m)[a], (h_t - h_m)[b], gate_at,
                                                           taus["excess"])}
        chosen = forms[rule]
        verdict = "comes_and_goes" if not gate["passed"] else "clearly_fewer" if chosen["fires"] else "not_clear"
    # "Mostly by shrinking the cell": only once the suspects alone clearly dropped; then their own RNA at the same spot
    # dropped by more than shrink_share of the suspects' drop (absolute drops of the side means).
    shrinking = None
    if gate is not None and gate["passed"] and forms["suspects"]["fires"] and h_m is not None:
        drop_t, drop_m = float(h_t[a].mean() - h_t[b].mean()), float(h_m[a].mean() - h_m[b].mean())
        shrinking = bool(drop_m > float(constants["shrink_share"]) * drop_t)

    def side(values) -> dict:
        """{a, b, change}: the side means of a per-run value and (b - a) / a."""
        if values is None:
            return {"a": None, "b": None, "change": None}
        mean_a, mean_b = float(values[a].mean()), float(values[b].mean())
        return {"a": _num(mean_a, 4), "b": _num(mean_b, 4), "change": _num(_change(mean_a, mean_b), 4)}

    def form(found: dict | None, tau) -> dict | None:
        return None if found is None else {"tau": _num(tau, 4), "gap": _num(found["gap"], 4),
                                           "separated": found["separated"], "fires": found["fires"],
                                           "why_not": found["why_not"]}

    judged = verdict in ("clearly_fewer", "not_clear")
    fewer = None
    if judged:
        values = h_t if rule == "suspects" else h_t - h_m
        fewer = {"form": rule, "tau": _num(taus[rule], 4), "a": _rounded(values[a], 4), "b": _rounded(values[b], 4),
                 "gap": _num(chosen["gap"], 4), "separated": chosen["separated"], "fires": chosen["fires"],
                 "why_not": chosen["why_not"],
                 "suspects": {"a": _rounded(h_t[a], 4), "b": _rounded(h_t[b], 4),
                              "mean": {"a": _num(h_t[a].mean(), 4), "b": _num(h_t[b].mean(), 4)}},
                 "own": None if h_m is None else {"a": _rounded(h_m[a], 4), "b": _rounded(h_m[b], 4),
                                                  "mean": {"a": _num(h_m[a].mean(), 4), "b": _num(h_m[b].mean(), 4)}},
                 "forms": {name: form(forms[name], taus[name]) for name in ("suspects", "excess")}}
    cost = went = None
    if verdict != "too_few":
        tx_a = _median(read.tx_inner for read in reads[a])
        tx_b = _median(read.tx_inner for read in reads[b])
        cost = {"own_same_spot": side(h_m), "own_all": side(h_u),
                "transcripts_per_cell": {"a": _num(tx_a, 1), "b": _num(tx_b, 1),
                                         "change": _num(_change(tx_a, tx_b), 4)},
                "shrinking": shrinking}
        classes = shares[:, t, :].mean(axis=1)                                                  # [2N, 4] over T
        went = {"A": dict(zip(WENT, _rounded(classes[a].mean(axis=0), 4))),
                "B": dict(zip(WENT, _rounded(classes[b].mean(axis=0), 4)))}

    # The pool: the members, what was left out and why (each target item of the order's cells once, by its first
    # reason: unsure or not found on the shared input, then not found, failing its check, or its cell or giver missing,
    # in any run).
    in_order = np.zeros(ncells, bool)
    in_order[np.asarray(order.cells, np.int64)] = True
    seen = status[:, (kind == qr.SUSPECT) & in_order[cell]]
    lost = (seen == qr.NOT_FOUND).any(axis=0)
    crossed = ~lost & (seen == qr.CROSSCHECK_FAILED).any(axis=0)
    absent = ~lost & ~crossed & ((seen == qr.C0_ABSENT) | (seen == qr.C1_ABSENT)).any(axis=0)
    counted = (matched_counts or {}).get("target") or {}

    def parent(c: int) -> int:
        return int(order.parent_cells[place[c]])

    pool = {"cells": [{"cell": parent(c), "name": items.cell_names[c], "n": int(n[c]),
                       "um": _num(order.distance_um[place[c]], 1)} for c in members],
            "suspects": int(total), "available": int(n[np.unique(np.asarray(order.cells, np.int64))].sum()), "k": k,
            "matched": int(m.sum()), "uniform": int(u.sum()),
            "dropped": {"unsure": int(counted.get("unsure", 0)),
                        "not_found": int(counted.get("not_found", 0)) + int(lost.sum()),
                        "crosscheck_failed": int(crossed.sum()), "cell_absent": int(absent.sum())}}
    by_point = [shares[a][:, t, :].mean(axis=0), shares[b][:, t, :].mean(axis=0)] if t.size else [np.zeros((0, 4))] * 2
    points = {"x": _rounded(np.asarray(items.x)[t], 2), "y": _rounded(np.asarray(items.y)[t], 2),
              "cell": rank[cell[t]].tolist(), "A": [_rounded(row, 3) for row in by_point[0]],
              "B": [_rounded(row, 3) for row in by_point[1]]}
    # The reading: the runs' read times added up (they overlap: read_runs reads several at once), the worst identity.
    errors = [(read.checks or {}).get("identity_max_err") for read in reads]
    errors = [float(e) for e in errors if e is not None and math.isfinite(float(e))]
    reading = {"seconds": _num(sum(float(read.seconds) for read in reads), 1),
               "identity_max_err": float(f"{max(errors):.3g}") if errors else None,
               "per_run": {role: {"iter": int(read.iter), "rows": int(read.rows), "tx": _num(read.tx_inner, 1),
                                  "seconds": _num(read.seconds, 1)} for role, read in zip(roles, reads)}}
    rates, short = constants.get("rates"), constants.get("short_rates")
    given = {key: setting.get(key) for key in ("flag", "value", "current")}
    return {
        "v": RESULT_V, "verdict": verdict, "why": why, "rule": rule, "start": start, "situation": situation,
        "at_iter": int(at_iter),
        "setting": {key: value if value is None or isinstance(value, str) else _num(value)
                    for key, value in given.items()},
        "runs": {"A": side_a, "B": side_b}, "parent_defaults": bool(parent_defaults),
        "constants": {"id": constants.get("id", CONSTANTS_ID), "start": row[0], "situation": row[1], "rule": rule,
                      "k": k, "gate": gate_at, "tau": _num(taus[rule], 4), "runs_per_side": per_side,
                      "rates": None if rates is None else {key: value if key in ("areas", "drop") else _num(value, 4)
                                                          for key, value in rates.items()},
                      "short_rates": None if short is None else {key: value if key in ("areas", "drop", "tests")
                                                                 else _num(value, 4) for key, value in short.items()}},
        "pool": pool,
        "come_back": None if gate is None else {"held": _rounded(h_t[a], 4), "mean": _num(h_t[a].mean(), 4),
                                                "gate": gate_at, "passed": gate["passed"]},
        "fewer": fewer, "cost": cost, "went": went,
        "cell": {"cell": parent(seed), "name": items.cell_names[seed], "suspects": int(n[seed]), "judged": False},
        "points": points, "reading": reading, "caveats": [dict(found) for found in caveats]}


# ---- the result's numbers ----------------------------------------------------------------------------------------------
def _num(value, places: int | None = None) -> float | None:
    """A number as the result holds it: None for None, NaN or an infinity; rounded to `places` decimals when given
    (never -0.0)."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value if places is None else round(value, places) + 0.0


def _rounded(values, places: int) -> list:
    """An array's values rounded to `places` decimals, as a list (None for a value that is not a number)."""
    return [_num(value, places) for value in np.asarray(values, np.float64).tolist()]


def _change(a: float | None, b: float | None) -> float | None:
    """(b - a) / a; None without both, or when a is not above 0."""
    return None if a is None or b is None or not a > 0 else (b - a) / a


def _median(values) -> float | None:
    """The median of the numbers given, leaving out None (a run without its frame's cell statistics); None if none."""
    found = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(found)) if found else None
