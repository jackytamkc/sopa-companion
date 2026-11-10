"""Test a setting here: the measured constants, the pool's order, the rule and the result a test answers with
(engine/qc_rule.py), on the synthetic world of tests/synth_pink.py: pools planted with exact per-run shares for the rule's
edges, and Monte Carlo runs from a model with known answers for its error rates."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import synth_pink as S
from sopa_companion.engine import qc_pool, qc_rule as R
from sopa_companion.engine.config import ROOT
from sopa_companion.engine.qc_readout import C0_ABSENT, C1_ABSENT, CROSSCHECK_FAILED, MATCHED, NOT_FOUND, UNIFORM


# ---- the constants: one test per row of the table ---------------------------------------------------------------------
def withdrawn(monkeypatch, start: str, situation: str, code: str = "square_not_measured"):
    """The row as a measurement that did not offer it would leave it (as the cells-only square was before proseg's own
    count measured it): not offered, its reason code a caveat, no tau and no rates."""
    monkeypatch.setitem(R.CONSTANTS[start]["situations"], situation,
                        {"offered": False, "refusal": code, "measured": False, "gate": 0.5,
                         "tau": {"excess": None, "suspects": None}, "rates": None})


def test_the_constants_have_one_id():
    """The set measured again with suspects chosen on proseg's own count has its own id (2026-10-31); the id of the
    set measured on the outlines stays an outline id, and a row that was not measured again says so (`measured`):
    the starts from nuclei and cells, whose numbers are still phase 1a's."""
    assert R.CONSTANTS_ID == "proseg-2026-10-31"
    assert R.OUTLINE_IDS == ("phase1a-2026-10-29",) and R.CONSTANTS_ID not in R.OUTLINE_IDS
    assert set(R.CONSTANTS) == set(R.STARTS) == {"paired", "cells"}
    for start in R.STARTS:
        assert set(R.CONSTANTS[start]["situations"]) == set(R.SITUATIONS) == {"as_is", "square"}
        for situation in R.SITUATIONS:
            assert R.constants_for(start, situation)["id"] == R.CONSTANTS_ID
            assert R.measured_on_proseg(start, situation) is (start == "cells"), (start, situation)


def test_a_row_rests_on_proseg_s_count_only_where_it_was_measured_on_it(monkeypatch):
    """measured_on_proseg (count_basis's rule, engine/qc_pool.py): the row's own `measured`, and never while the
    constants' id is an outline one; KeyError for a row that does not exist."""
    assert [R.measured_on_proseg("cells", s) for s in R.SITUATIONS] == [True, True]
    assert [R.measured_on_proseg("paired", s) for s in R.SITUATIONS] == [False, False]
    monkeypatch.setattr(R, "OUTLINE_IDS", (*R.OUTLINE_IDS, R.CONSTANTS_ID))
    assert not any(R.measured_on_proseg(start, s) for start in R.STARTS for s in R.SITUATIONS)
    with pytest.raises(KeyError):
        R.measured_on_proseg("nuclei", "as_is")


def test_runs_per_side():
    assert R.CONSTANTS["paired"]["runs_per_side"] == 3 and R.CONSTANTS["cells"]["runs_per_side"] == 4


def test_the_rule_is_the_excess_for_both_starts():
    assert [R.CONSTANTS[s]["rule"] for s in R.STARTS] == ["excess", "excess"]


def test_pools_hold_160_suspects():
    assert [R.CONSTANTS[s]["k"] for s in R.STARTS] == [160, 160]


def test_the_shrink_share_is_a_third():
    assert [R.CONSTANTS[s]["shrink_share"] for s in R.STARTS] == [1 / 3, 1 / 3]


def test_the_areas_measured():
    for start in R.STARTS:
        assert (R.area_min_um(start), R.as_is_max_um(start)) == (300.0, 500.0)
    assert R.MEASURED["area_um"] == 500.0 and R.MEASURED["inset_um"] == 25.0


@pytest.mark.parametrize("start, situation", [("paired", "as_is"), ("paired", "square"), ("cells", "as_is"),
                                              ("cells", "square")])
def test_the_situations_offered(start, situation):
    c = R.constants_for(start, situation)
    assert R.offered(start, situation) and c["offered"] and c["refusal"] is None
    assert R.refusal_of(start, situation) is None
    assert c["gate"] == 0.5
    assert c["tau"] == c["taus"][c["rule"]] and all(math.isfinite(t) and 0 < t < 0.2 for t in c["taus"].values())
    assert set(c["rates"]) == {"fp_mean", "fp_worst", "areas", "power", "drop", "shrink_fire"}
    assert all(math.isfinite(v) for v in c["rates"].values()) and c["rates"]["drop"] == 0.25


def test_cells_only_squares_are_offered_once_measured_on_proseg_s_count():
    """The cells-only square, not offered while its numbers rested on suspects chosen on the outlines (identical runs
    came out "clearly fewer" far too often there), was measured on proseg's own count (2026-10-31) and is offered:
    the as-is row's gate and tau (one tau for all of a start's pools), its own rates (pools seeded near the square's
    centre); a test there is judged with it, never with the as-is row (row_for)."""
    c, as_is = R.constants_for("cells", "square"), R.constants_for("cells", "as_is")
    assert R.offered("cells", "square") and c["offered"] is True and R.refusal_of("cells", "square") is None
    assert (c["gate"], c["taus"]) == (as_is["gate"], as_is["taus"]) and c["rates"] != as_is["rates"]
    assert c["rates"]["areas"] == 3 and R.measured_on_proseg("cells", "square")
    assert R.row_for("cells", [0, 0, 1200, 1200]) == ("cells", "square")


def test_a_row_not_offered_gives_way_to_the_as_is_row_and_says_why(monkeypatch):
    """A situation a measurement does not offer (the cells-only square before proseg's own count measured it): its code
    is a caveat, it has no tau and no rates, and a test there is judged with the start's as-is row (row_for)."""
    withdrawn(monkeypatch, "cells", "square")
    c = R.constants_for("cells", "square")
    assert not R.offered("cells", "square") and c["offered"] is False
    assert R.refusal_of("cells", "square") == c["refusal"] == "square_not_measured"
    assert c["refusal"] in qc_pool.REASON_WORDS and qc_pool.KIND[c["refusal"]] == "caveat"
    assert c["taus"] == {"excess": None, "suspects": None} and c["tau"] is None and c["rates"] is None
    assert c["gate"] == 0.5                          # kept for the record
    assert R.row_for("cells", [0, 0, 1200, 1200]) == R.row_for("cells", [0, 0, 400, 1200]) == ("cells", "as_is")
    assert R.row_for("paired", [0, 0, 1200, 1200]) == ("paired", "square")      # the other start's own row


def test_the_row_a_test_is_judged_with():
    """row_for: a start from cells only, or from nuclei and cells, its own rows; from nuclei only (never measured),
    those of nuclei and cells. The square row for a region over 500 µm across (one way is enough: a narrow region too),
    the as-is row otherwise (500 µm exactly included); every row it gives is offered; another start is not known."""
    for start, row in (("paired", "paired"), ("nuclei", "paired"), ("cells", "cells")):
        for bbox, situation in (([0, 0, 300, 300], "as_is"), ([10, 20, 510, 520], "as_is"), ([0, 0, 250, 400], "as_is"),
                                ([0, 0, 500.001, 100], "square"), ([0, 0, 400, 1200], "square"),
                                ([0, 0, 1200, 1200], "square"), ([-5000, 0, 5000, 8000], "square")):
            assert R.row_for(start, bbox) == (row, situation), (start, bbox)
    assert all(R.offered(*R.row_for(start, bbox)) for start in ("paired", "nuclei", "cells")
               for bbox in ([0, 0, 300, 300], [0, 0, 1200, 1200]))
    with pytest.raises(KeyError):
        R.row_for("xenium", [0, 0, 300, 300])


# The taus and rates: a start from nuclei and cells, phase 1a's (suspects chosen on the outlines); from cells only,
# measured again on proseg's own count (2026-10-31: tau 0.0656 / 0.0612, was 0.0678 / 0.0708; false alarms 0.54 % on
# average, was 0.76 %; power 0.976, was 0.882), its square too.
@pytest.mark.parametrize("start, tau_excess, tau_suspects", [("paired", 0.08938564995169271, 0.08527235239755926),
                                                              ("cells", 0.0655782303402374, 0.06116140846464757)])
def test_the_taus_as_measured(start, tau_excess, tau_suspects):
    for situation in R.SITUATIONS:
        assert R.constants_for(start, situation)["taus"] == {"excess": tau_excess, "suspects": tau_suspects}


@pytest.mark.parametrize("start, situation, rates", [
    ("paired", "as_is", {"fp_mean": 0.00788, "fp_worst": 0.01859, "areas": 2, "power": 0.922, "shrink_fire": 0.0185}),
    ("cells", "as_is", {"fp_mean": 0.00545, "fp_worst": 0.01441, "areas": 2, "power": 0.976, "shrink_fire": 0.1012}),
    ("paired", "square", {"fp_mean": 0.01298, "fp_worst": 0.0330, "areas": 3, "power": 0.745, "shrink_fire": 0.0107}),
    ("cells", "square", {"fp_mean": 0.01126, "fp_worst": 0.01327, "areas": 3, "power": 0.883, "shrink_fire": 0.0261})])
def test_the_rates_of_each_situation(start, situation, rates):
    """The situation's own row: the page's tooltip takes every number from it."""
    got = R.constants_for(start, situation)["rates"]
    for key, value in rates.items():
        places = len(repr(value).split(".")[1]) if isinstance(value, float) else 0
        assert round(got[key], places) == value, key
    assert got == R.CONSTANTS[start]["situations"][situation]["rates"]


@pytest.mark.parametrize("start", ["paired", "cells"])
def test_square_rates_are_not_the_as_is_ones(start):
    assert R.constants_for(start, "square")["rates"] != R.constants_for(start, "as_is")["rates"]


@pytest.mark.parametrize("start, situation", [("nuclei", "as_is"), ("paired", "narrow"), ("cells", "lasso")])
def test_what_was_not_measured_is_not_known(start, situation):
    with pytest.raises(KeyError):
        R.constants_for(start, situation)


def test_constants_for_gives_a_copy():
    c = R.constants_for("paired", "as_is")
    c["rates"]["power"] = 0.0
    c["taus"]["excess"] = 1.0
    assert R.constants_for("paired", "as_is")["rates"]["power"] > 0.9
    assert R.constants_for("paired", "as_is")["taus"]["excess"] < 0.1
    short = R.constants_for("cells", "as_is")
    short["short_rates"]["power"] = 1.0
    assert R.constants_for("cells", "as_is")["short_rates"]["power"] < 0.3


def test_pools_short_of_k_have_their_own_rates_where_they_were_measured_apart():
    """The rates of pools short of k, measured apart on starts from cells only in regions re-run as they are (8 test
    cells of one sample in 4 areas), pinned on that row alone and carried by constants_for (the page's tooltip adds them
    after too_few's words); every other row None."""
    assert R.CONSTANTS["cells"]["situations"]["as_is"]["short_rates"] == {
        "fp_mean": 0.028426395939086288, "fp_worst": 0.037914691943127965, "power": 0.21962774957698816,
        "drop": R.DROP, "shrink_fire": 0.042064297800338414, "tests": 8, "areas": 4}
    for start in R.STARTS:
        for situation in R.SITUATIONS:
            found = R.constants_for(start, situation)["short_rates"]
            assert found == R.CONSTANTS[start]["situations"][situation]["short_rates"], (start, situation)
            assert (found is None) is ((start, situation) != ("cells", "as_is")), (start, situation)


# ---- the pool's order -------------------------------------------------------------------------------------------------
def test_pool_of_on_the_toy_line():
    """Cells 10-13 share a pattern at x = 0, 1, 2, 5 with n = 3, 2, 4, 1; cell 14 has another and n = 10. Each seed takes
    its nearest same-pattern cells (ties by cell number) until the sum reaches K, as the measurements' pools did."""
    x = {10: 0.0, 11: 1.0, 12: 2.0, 13: 5.0}
    n = np.array([3, 2, 4, 1, 10])                      # Items cells 0..4 are parent cells 10..14

    def pools(k):
        out = {}
        for seed in range(5):
            if seed == 4:
                order = R.PoolOrder(seed=4, cells=(4,), distance_um=(0.0,), parent_cells=(14,))
            else:
                others = sorted((c for c in range(4) if c != seed), key=lambda c: (abs(x[10 + c] - x[10 + seed]), 10 + c))
                cells = (seed, *others)
                order = R.PoolOrder(seed=seed, cells=cells, distance_um=tuple(abs(x[10 + c] - x[10 + seed]) for c in cells),
                                    parent_cells=tuple(10 + c for c in cells))
            members, total = R.pool_of(n, order, k)
            assert total == sum(int(n[c]) for c in members) and members[0] == seed
            if total >= k:
                mem = tuple(sorted(10 + c for c in members))
                out[mem] = out.get(mem, 0) + 1
        return out
    assert pools(5) == {(10, 11): 2, (11, 12): 1, (12, 13): 1, (14,): 1}
    assert pools(6) == {(10, 11, 12): 2, (11, 12): 1, (11, 12, 13): 1, (14,): 1}
    assert pools(11) == {}


def test_pool_of_takes_no_cell_without_a_suspect_and_none_once_reached():
    order = R.PoolOrder(seed=0, cells=(0, 1, 2, 3, 4), distance_um=(0.0, 1.0, 2.0, 3.0, 4.0), parent_cells=(5, 6, 7, 8, 9))
    assert R.pool_of(np.array([2, 0, 3, 5, 9]), order, 5) == ([0, 2], 5)
    assert R.pool_of(np.array([0, 4, 4, 4, 4]), order, 5) == ([], 0)       # the seed has none: no pool
    assert R.pool_of(np.array([1, 1, 0, 1, 0]), order, 5) == ([0, 1, 3], 3)  # short: every cell that has one


def test_inside_inner_includes_its_edges_and_never_nan():
    got = R.inside_inner([0.0, 10.0, 5.0, -0.001, np.nan, 5.0], [0.0, 10.0, 5.0, 5.0, 5.0, np.inf], (0, 0, 10, 10))
    assert got.tolist() == [True, True, True, False, False, False]


# ---- the rule's edges ---------------------------------------------------------------------------------------------------
def test_a_mean_equal_to_the_gate_passes():
    assert R.judge([0.25, 0.5, 0.75], [1, 1, 1], [0, 0, 0], 0.5, 0.1)["passed"] is True
    assert R.judge([0.25, 0.5, 0.74], [1, 1, 1], [0, 0, 0], 0.5, 0.1)["passed"] is False


def test_max_b_equal_to_min_a_is_not_separated():
    got = R.judge([1, 1, 1], [0.75, 0.875, 1.0], [0.5, 0.625, 0.75], 0.5, 0.0625)
    assert got["separated"] is False and got["fires"] is False and got["why_not"] == "apart"


def test_a_gap_equal_to_tau_does_not_fire():
    got = R.judge([1, 1, 1], [1.0, 1.0, 1.0], [0.75, 0.75, 0.75], 0.5, 0.25)
    assert got["gap"] == 0.25 and got["separated"] is True
    assert got["fires"] is False and got["why_not"] == "gap"
    assert R.judge([1, 1, 1], [1.0, 1.0, 1.0], [0.75, 0.75, 0.75], 0.5, 0.2499)["fires"] is True


def test_why_not():
    tau = 0.1
    assert R.judge([1] * 3, [0.9, 0.5, 0.9], [0.55, 0.3, 0.3], 0.5, tau)["why_not"] == "apart"   # gap 0.38 > tau
    assert R.judge([1] * 3, [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], 0.5, tau)["why_not"] == "apart"    # a negative gap
    assert R.judge([1] * 3, [0.5, 0.5, 0.5], [0.45, 0.45, 0.45], 0.5, tau)["why_not"] == "gap"
    assert R.judge([1] * 3, [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], 0.5, tau)["why_not"] is None       # it fires
    assert R.judge([0.4] * 3, [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], 0.5, tau)["why_not"] is None     # the gate failed
    assert R.judge([1] * 3, [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], 0.5, None)["fires"] is None
    got = R.judge(np.float32([0.6] * 3), [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], 0.5, tau)
    assert all(type(got[key]) is bool for key in ("passed", "separated", "fires")) and type(got["gap"]) is float


def exact(**kw) -> dict:
    """A planted pool whose items all hold their run's shares exactly (values a binary float holds)."""
    return S.planted(scale=0, **kw)


def test_the_excess_form_reads_x_while_the_gate_stays_on_the_suspects():
    """H_T 0.625 passes the gate although X is below 0; the excess form judges X: every B run below every A run, by
    0.25 > tau. Had the gate read X, the suspects would come and go."""
    args = exact(sizes=[80, 80], hT={"A": [0.625] * 3, "B": [0.375] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3},
                 hU={"A": [0.75] * 3, "B": [0.75] * 3})
    res = R.answer(**args)
    assert res["verdict"] == "clearly_fewer" and res["rule"] == "excess"
    assert res["come_back"] == {"held": [0.625] * 3, "mean": 0.625, "gate": 0.5, "passed": True}
    assert res["fewer"]["a"] == [-0.125] * 3 and res["fewer"]["b"] == [-0.375] * 3 and res["fewer"]["gap"] == 0.25
    assert res["fewer"]["form"] == "excess" and res["fewer"]["fires"] is True and res["fewer"]["why_not"] is None
    assert res["fewer"]["forms"]["excess"]["fires"] is True and res["fewer"]["forms"]["suspects"]["fires"] is True


def test_only_the_rule_decides():
    """The suspects alone drop clearly while their own RNA drops as much: the excess rule says not clear (the gap of X
    is 0), the suspects form fires for the record, and the shrink flag says why."""
    args = exact(sizes=[80, 80], hT={"A": [0.625] * 3, "B": [0.375] * 3}, hM={"A": [0.75] * 3, "B": [0.5] * 3})
    res = R.answer(**args)
    assert res["verdict"] == "not_clear" and res["fewer"]["why_not"] == "apart"
    assert res["fewer"]["forms"]["suspects"] == {"tau": 0.0853, "gap": 0.25, "separated": True, "fires": True,
                                                 "why_not": None}
    assert res["cost"]["shrinking"] is True
    suspects = R.answer(**{**args, "constants": {**args["constants"], "rule": "suspects",
                                                  "tau": args["constants"]["taus"]["suspects"]}})
    assert suspects["verdict"] == "clearly_fewer" and suspects["rule"] == suspects["fewer"]["form"] == "suspects"
    assert suspects["fewer"]["a"] == [0.625] * 3 and suspects["constants"]["tau"] == 0.0853


def test_the_verdict_comes_and_goes_below_the_gate():
    args = exact(sizes=[80, 80], hT={"A": [0.375, 0.5, 0.5], "B": [0.25] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    res = R.answer(**args)
    assert res["verdict"] == "comes_and_goes" and res["why"] is None
    assert res["come_back"]["passed"] is False and res["fewer"] is None
    assert res["cost"]["shrinking"] is None and res["went"] is not None


# ---- the pool, once the runs are read ----------------------------------------------------------------------------------
def test_a_seed_without_a_suspect_read_everywhere_is_cell_lost():
    args = exact(sizes=[0, 90, 90], drops=[(0, C0_ABSENT, 5)], hT={"A": [0.625] * 3, "B": [0.5] * 3},
                 hM={"A": [0.75] * 3, "B": [0.75] * 3})
    res = R.answer(**args)
    assert (res["verdict"], res["why"]) == ("too_few", "cell_lost")
    assert res["pool"]["cells"] == [] and res["pool"]["suspects"] == 0 and res["pool"]["available"] == 180
    assert res["pool"]["dropped"]["cell_absent"] == 5
    assert res["cell"] == {"cell": 1234, "name": args["items"].cell_names[0], "suspects": 0, "judged": False}
    assert [res[key] for key in ("come_back", "fewer", "cost", "went")] == [None] * 4
    assert res["points"] == {"x": [], "y": [], "cell": [], "A": [], "B": []}


def test_a_short_pool_is_judged_with_every_cell_it_could_take():
    """A pool under k once the runs were read is judged all the same (no "pool_short" any more), over every cell that
    has a suspect read everywhere; its caveat is engine/qc_test.py's to add (answer_caveats), none here."""
    args = exact(sizes=[30, 40, 50], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    res = R.answer(**args)
    assert (res["verdict"], res["why"]) == ("clearly_fewer", None)
    assert res["fewer"]["a"] == [-0.125] * 3 and res["fewer"]["b"] == [-0.25] * 3 and res["fewer"]["gap"] == 0.125
    assert [c["n"] for c in res["pool"]["cells"]] == [30, 40, 50]
    assert res["pool"]["suspects"] == res["pool"]["available"] == 120 and res["pool"]["k"] == 160
    assert len(res["points"]["x"]) == 120 and res["went"] is not None and res["cost"] is not None
    assert res["caveats"] == []
    # Too few to pass the gate, or for the excess rule: as with any pool.
    low = R.answer(**exact(sizes=[3, 4], hT={"A": [0.375] * 3, "B": [0.25] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3}))
    assert (low["verdict"], low["why"], low["pool"]["suspects"]) == ("comes_and_goes", None, 7)
    assert R.answer(**exact(sizes=[3, 4], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM=None))["why"] == "no_controls"


def test_the_pool_is_taken_again_on_what_every_run_read():
    """Before the runs the seed and its first neighbour reach 160 (100 + 80); once read, 30 of the neighbour's suspects
    are missing from some run, so the pool takes the next cell, and the reserve stays counted in `available`."""
    args = exact(sizes=[100, 50, 40, 30], drops=[(1, C1_ABSENT, 20), (1, NOT_FOUND, 10)],
                 hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    res = R.answer(**args)
    assert [c["n"] for c in res["pool"]["cells"]] == [100, 50, 40]
    assert (res["pool"]["suspects"], res["pool"]["available"]) == (190, 220)
    assert [c["um"] for c in res["pool"]["cells"]] == [0.0, 4.0, 7.5]
    assert res["pool"]["dropped"] == {"unsure": 0, "not_found": 10, "crosscheck_failed": 0, "cell_absent": 20}


def test_the_points_follow_the_pool_order():
    """Items need not be in the pool's order: the points (and the members) follow it, each cell's in Items order."""
    args = exact(sizes=[60, 50, 70], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    items, order = args["items"], args["order"]
    args["order"] = R.PoolOrder(seed=0, cells=(0, 2, 1), distance_um=(0.0, 3.0, 5.0),
                                parent_cells=(order.parent_cells[0], order.parent_cells[2], order.parent_cells[1]))
    res = R.answer(**args)
    assert [c["n"] for c in res["pool"]["cells"]] == [60, 70, 50]
    assert [c["name"] for c in res["pool"]["cells"]] == [items.cell_names[c] for c in (0, 2, 1)]
    assert res["points"]["cell"] == [0] * 60 + [1] * 70 + [2] * 50
    first = np.flatnonzero((items.kind == 0) & (items.cell == 2))[0]
    assert (res["points"]["x"][60], res["points"]["y"][60]) == (round(float(items.x[first]), 2),
                                                                 round(float(items.y[first]), 2))


def test_an_item_counts_only_where_every_run_read_it():
    """Suspects and controls read in all runs but one are left out of every run's means: their shares (all lost here)
    change nothing. The left-out counts come from the matching (unsure, not found) and the runs (each target once)."""
    args = exact(sizes=[90, 90], drops=[(0, NOT_FOUND, 2), (0, CROSSCHECK_FAILED, 2), (1, C0_ABSENT, 1), (1, C1_ABSENT, 2)],
                 hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3},
                 hU={"A": [0.75] * 3, "B": [0.75] * 3}, matched=500, unsure=3, not_found=1)
    items, reads = args["items"], args["readouts"]
    left = np.flatnonzero(np.stack([r.status for r in reads.values()]).any(axis=0))
    assert len(left) == 7 + 40                              # the dropped suspects, and 40 matched controls
    for read in reads.values():
        read.held[left] = np.where(np.isnan(read.held[left]), np.nan, 0.0)
    res = R.answer(**args)
    assert res["come_back"]["held"] == [0.625] * 3 and res["fewer"]["suspects"]["b"] == [0.5] * 3
    assert res["fewer"]["own"]["a"] == [0.75] * 3
    assert res["pool"]["matched"] == 500 and res["pool"]["suspects"] == 180 and len(res["points"]["x"]) == 180
    assert res["pool"]["dropped"] == {"unsure": 3, "not_found": 1 + 2, "crosscheck_failed": 2, "cell_absent": 3}
    assert res["cell"]["suspects"] == 90                    # the seed's 94 suspects but the 4 not read everywhere
    # One status in one run is enough, and a target missing from one run and failing its check in another counts once,
    # as not found.
    read_everywhere = ~np.stack([r.status for r in reads.values()]).any(axis=0)
    keep = int(np.flatnonzero((items.kind == 0) & read_everywhere)[0])
    reads["A1"].status[keep] = CROSSCHECK_FAILED
    reads["B3"].status[keep] = NOT_FOUND
    again = R.answer(**args)
    assert again["pool"]["suspects"] == 179 and again["cell"]["suspects"] == 89          # `keep` is the seed's
    assert again["pool"]["dropped"]["not_found"] == 4 and again["pool"]["dropped"]["crosscheck_failed"] == 2


def test_no_matched_control_read_everywhere():
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM=None)
    res = R.answer(**args)
    assert (res["verdict"], res["why"], res["pool"]["matched"]) == ("too_few", "no_controls", 0)
    suspects = R.answer(**{**args, "constants": {**args["constants"], "rule": "suspects"}})
    assert suspects["verdict"] == "clearly_fewer" and suspects["fewer"]["own"] is None
    assert suspects["fewer"]["forms"]["excess"] is None and suspects["cost"]["own_same_spot"] == {
        "a": None, "b": None, "change": None}
    assert suspects["cost"]["shrinking"] is None


def test_own_rna_at_the_same_spot_weighs_every_control_alike():
    """H_M is the plain mean over M, each matched control weighing the same, not a mean of each suspect's own: 30 of
    the 180 suspects keep 1 of their 3 controls (the other 2 not found in one run), held at 0.25 where the other 450
    are held at 0.75. So H_M is (450 x 0.75 + 30 x 0.25) / 480 = 0.71875 in every run, where suspect by suspect it
    would be (150 x 0.75 + 30 x 0.25) / 180 = 0.6667."""
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3},
                 hU={"A": [0.75] * 3, "B": [0.75] * 3})
    items, reads = args["items"], args["readouts"]
    controls: dict[int, list] = {}
    for i in np.flatnonzero(items.kind == MATCHED):
        controls.setdefault(int(items.target[i]), []).append(int(i))
    chosen = sorted(controls)[::6]                         # 15 suspects of each cell
    kept = [controls[t][0] for t in chosen]
    lost = [i for t in chosen for i in controls[t][1:]]
    for read in reads.values():
        read.held[kept], read.giver[kept], read.other[kept], read.bg[kept] = 0.25, 0.075, 0.225, 0.45
    for name in ("held", "giver", "other", "bg"):
        getattr(reads["B2"], name)[lost] = np.nan
    reads["B2"].status[lost] = NOT_FOUND
    res = R.answer(**args)
    h_m = (450 * 0.75 + 30 * 0.25) / 480
    assert (len(chosen), res["pool"]["matched"]) == (30, 480)
    assert res["fewer"]["own"] == {"a": [round(h_m, 4)] * 3, "b": [round(h_m, 4)] * 3,
                                   "mean": {"a": round(h_m, 4), "b": round(h_m, 4)}}
    assert res["fewer"]["a"] == [round(0.625 - h_m, 4)] * 3 and res["fewer"]["b"] == [round(0.5 - h_m, 4)] * 3
    assert res["cost"]["own_same_spot"] == {"a": round(h_m, 4), "b": round(h_m, 4), "change": 0.0}


def test_all_own_rna_is_the_members_read_in_every_run():
    """U holds the members' uniform controls read in every run: not the reserve's 50 (held at 0.25 here, against the
    members' 0.75 and 0.5), nor 10 of a member's that one run did not read (held at 0 in the others)."""
    args = exact(sizes=[90, 90], reserve=[20], hT={"A": [0.625] * 3, "B": [0.5] * 3},
                 hM={"A": [0.75] * 3, "B": [0.75] * 3}, hU={"A": [0.75] * 3, "B": [0.5] * 3})
    items, reads = args["items"], args["readouts"]
    uniform = items.kind == UNIFORM
    reserve, lost = np.flatnonzero(uniform & (items.cell == 2)), np.flatnonzero(uniform & (items.cell == 1))[:10]
    for read in reads.values():                            # the rule reads only their HELD
        read.held[reserve], read.held[lost] = 0.25, 0.0
    reads["A2"].status[lost], reads["A2"].held[lost] = NOT_FOUND, np.nan
    res = R.answer(**args)
    assert [c["n"] for c in res["pool"]["cells"]] == [90, 90] and res["pool"]["available"] == 200
    assert (len(reserve), res["pool"]["uniform"]) == (50, 90)
    assert res["cost"]["own_all"] == {"a": 0.75, "b": 0.5, "change": round(-0.25 / 0.75, 4)}


# ---- what it cost, and where the suspects went --------------------------------------------------------------------------
def test_the_cost_is_side_means_and_their_change():
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.625] * 3},
                 hU={"A": [0.5, 0.75, 1.0], "B": [0.5] * 3}, tx={"A": [100.0, None, 300.0], "B": [None, None, None]})
    res = R.answer(**args)
    assert res["cost"]["own_same_spot"] == {"a": 0.75, "b": 0.625, "change": round((0.625 - 0.75) / 0.75, 4)}
    assert res["cost"]["own_all"] == {"a": 0.75, "b": 0.5, "change": round(-0.25 / 0.75, 4)}
    # Transcripts per cell: each side's median over its runs, a run without its frame's cell statistics left out.
    assert res["cost"]["transcripts_per_cell"] == {"a": 200.0, "b": None, "change": None}
    assert res["reading"]["per_run"]["A2"]["tx"] is None and res["reading"]["per_run"]["A3"]["tx"] == 300.0
    res = R.answer(**exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.625] * 3},
                           tx={"A": [100.0, 120.0, 300.0], "B": [90.0, 95.0, 400.0]}))
    assert res["cost"]["transcripts_per_cell"] == {"a": 120.0, "b": 95.0, "change": round(-25 / 120, 4)}


def test_shrinking_compares_absolute_drops():
    """Suspects 0.30 -> 0.20 and their own RNA at the same spot 0.90 -> 0.85: 0.05 > 0.10 / 3, so mostly shrinking,
    where relative drops would say no (0.056 < 0.333 x 0.333). The flag needs the suspects alone to fire (a lower gate
    here, as only a pool held this little could show it)."""
    args = exact(sizes=[90, 90], hT={"A": [0.29, 0.30, 0.31], "B": [0.19, 0.20, 0.21]},
                 hM={"A": [0.89, 0.90, 0.91], "B": [0.84, 0.85, 0.86]})
    args["constants"] = {**args["constants"], "gate": 0.25}
    res = R.answer(**args)
    assert res["fewer"]["forms"]["suspects"]["fires"] is True and res["cost"]["shrinking"] is True
    drop_t, drop_m = 0.30 - 0.20, 0.90 - 0.85
    assert drop_m / 0.90 < (drop_t / 0.30) / 3
    args["readouts"] = S.planted(scale=0, sizes=[90, 90], hT={"A": [0.29, 0.30, 0.31], "B": [0.19, 0.20, 0.21]},
                                 hM={"A": [0.89, 0.90, 0.91], "B": [0.87, 0.88, 0.89]})["readouts"]
    assert R.answer(**args)["cost"]["shrinking"] is False                  # 0.02 < 0.033


def test_where_the_suspects_went_adds_up():
    for example in S.fixture_examples():
        res = example["result"]
        if res["went"] is None:
            continue
        for side in "AB":
            assert list(res["went"][side]) == list(R.WENT)
            assert sum(res["went"][side].values()) == pytest.approx(1.0, abs=2e-4)
        assert res["went"]["A"]["stay"] == res["come_back"]["mean"]
        assert all(abs(sum(row) - 1.0) <= 2e-3 for side in "AB" for row in res["points"][side])
        assert "held" not in res["cell"] and res["cell"]["judged"] is False


def test_the_result_is_json_for_every_verdict():
    found = set()
    for example in S.fixture_examples():
        res = example["result"]
        json.dumps(res, allow_nan=False)
        found.add((res["verdict"], res["why"]))
    assert found == {("not_clear", None), ("clearly_fewer", None), ("comes_and_goes", None), ("too_few", "cell_lost"),
                     ("too_few", "no_controls")}
    # Nothing read anywhere but NaN shares in the left-out items, no cell statistics: still JSON.
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3},
                 drops=[(0, C0_ABSENT, 3)], extra=[0, 0])
    for read in args["readouts"].values():
        read.seconds, read.checks = float("nan"), {}
    res = R.answer(**args)
    text = json.dumps(res, allow_nan=False)
    assert res["reading"]["seconds"] is None and res["reading"]["identity_max_err"] is None and "NaN" not in text


def numbers(value) -> list:
    """Every number in a part of a result (True and False left out)."""
    if isinstance(value, dict):
        return [v for item in value.values() for v in numbers(item)]
    if isinstance(value, list):
        return [v for item in value for v in numbers(item)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def test_the_result_rounds_as_the_page_reads_it():
    """Shares, drops, gaps and X to 4 decimals; the points' shares to 3 (some of them needing all 3), their x and y to
    2; transcripts per cell, µm and seconds to 1. Every decision was made before, on unrounded values."""
    def within(values, places):
        return all(round(v, places) == v for v in values)
    third = False
    for example in S.fixture_examples():
        res = example["result"]
        cost = dict(res["cost"] or {})
        tx = cost.pop("transcripts_per_cell", None) or {}
        assert within(numbers([res["come_back"], res["fewer"], cost, tx.get("change"), res["went"],
                               res["constants"]["tau"], res["constants"]["rates"]]), 4), example["name"]
        assert within(numbers([tx.get("a"), tx.get("b"), [c["um"] for c in res["pool"]["cells"]],
                               res["reading"]["seconds"], list(res["reading"]["per_run"].values())]), 1)
        assert within(numbers([res["points"]["x"], res["points"]["y"]]), 2)
        shares = numbers([res["points"]["A"], res["points"]["B"]])
        assert within(shares, 3)
        third = third or not within(shares, 2)
    assert third


def test_answer_refuses_what_it_cannot_judge(monkeypatch):
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    withdrawn(monkeypatch, "cells", "square")         # every row is offered now: one that is not, as one was before
    with pytest.raises(ValueError, match="not measured"):
        R.answer(**{**args, "start": "cells", "situation": "square", "constants": None})
    with pytest.raises(ValueError, match="runs on each side"):
        R.answer(**{**args, "sides": {"A": ["A1", "A2"], "B": ["B1", "B2"]}})
    with pytest.raises(ValueError, match="read once"):
        R.answer(**{**args, "sides": {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B4"]}})
    reads = dict(args["readouts"])
    reads["B1"] = S.planted(scale=0, sizes=[90], hT={"A": [0.625] * 3, "B": [0.5] * 3})["readouts"]["B1"]
    with pytest.raises(ValueError, match="one value per item"):
        R.answer(**{**args, "readouts": reads})


def test_the_result_echoes_the_test():
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3},
                 setting={"flag": "prior-weight", "value": 2.0, "current": 1.0}, parent_defaults=False)
    res = R.answer(**args)
    assert list(res) == ["v", "verdict", "why", "rule", "start", "situation", "at_iter", "setting", "runs",
                         "parent_defaults", "constants", "pool", "come_back", "fewer", "cost", "went", "cell", "points",
                         "reading", "caveats"]
    assert (res["v"], res["start"], res["situation"], res["at_iter"]) == (2, "paired", "as_is", 450)
    assert res["setting"] == {"flag": "prior-weight", "value": 2.0, "current": 1.0} and res["parent_defaults"] is False
    assert res["runs"] == {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]}
    assert res["constants"] == {"id": R.CONSTANTS_ID, "start": "paired", "situation": "as_is", "rule": "excess",
                                "k": 160, "gate": 0.5, "tau": 0.0894, "runs_per_side": 3,
                                "rates": {"fp_mean": 0.0079, "fp_worst": 0.0186, "areas": 2, "power": 0.922,
                                          "drop": 0.25, "shrink_fire": 0.0185}, "short_rates": None}
    # A row with pools short of k measured apart says their rates too (rounded as the rates are; its counts whole).
    short = R.constants_for("cells", "as_is")["short_rates"]
    given = R.answer(**{**args, "constants": {**R.constants_for(*args.get("row") or ("paired", "as_is")),
                                              "short_rates": short}})
    assert given["constants"]["short_rates"] == {"fp_mean": 0.0284, "fp_worst": 0.0379, "power": 0.2196, "drop": 0.25,
                                                 "shrink_fire": 0.0421, "tests": 8, "areas": 4}
    assert res["caveats"] == []
    assert list(res["reading"]["per_run"]) == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert res["points"]["cell"] == [0] * 90 + [1] * 90
    # The clicked cell's own suspects, not the pool's (180).
    assert res["cell"] == {"cell": 1234, "name": args["items"].cell_names[0], "suspects": 90, "judged": False}
    assert res["pool"]["suspects"] == 180


def test_a_test_outside_the_measurements_is_judged_with_its_row_and_carries_its_caveats(monkeypatch):
    """`start` and `situation` are the test's own (here a start from nuclei only, in a crop), `row` the measured row
    it is judged with: its constants (constants.start and constants.situation the row's, with its tau, rates, k and
    runs per side); the caveats given are carried as given, in their order, as copies. Without a row the test's own
    start and situation are the row; a row that is not offered (every one is now: the cells-only square as it was
    before proseg's own count measured it) is refused, and a start of no row of its own needs one."""
    withdrawn(monkeypatch, "cells", "square")
    args = exact(sizes=[90, 90], hT={"A": [0.625] * 3, "B": [0.5] * 3}, hM={"A": [0.75] * 3, "B": [0.75] * 3})
    caveats = [qc_pool.reason("tiled_box", max_um=1200.0), qc_pool.reason("nuclei_start"),
               qc_pool.reason("area_narrow")]
    res = R.answer(**{**args, "start": "nuclei", "situation": "crop", "constants": None, "row": ("paired", "square"),
                      "caveats": caveats})
    square = R.constants_for("paired", "square")
    assert (res["start"], res["situation"], res["verdict"]) == ("nuclei", "crop", "clearly_fewer")
    assert (res["constants"]["start"], res["constants"]["situation"], res["constants"]["k"]) == \
        ("paired", "square", 160)
    assert res["constants"]["rates"]["areas"] == square["rates"]["areas"] == 3
    assert res["constants"]["tau"] == round(square["tau"], 4) and res["constants"]["runs_per_side"] == 3
    assert res["caveats"] == caveats and res["caveats"][0] is not caveats[0]
    json.dumps(res, allow_nan=False)
    same = R.answer(**args)
    assert (same["constants"]["start"], same["constants"]["situation"], same["caveats"]) == ("paired", "as_is", [])
    with pytest.raises(ValueError, match="not measured for cells starts in the square situation"):
        R.answer(**{**args, "constants": None, "row": ("cells", "square")})
    with pytest.raises(KeyError):
        R.answer(**{**args, "start": "nuclei", "constants": None})


# ---- Monte Carlo: the rule's error rates on runs from the model -------------------------------------------------------
# The model's spreads, set so that its gaps and rates sit near the measured ones (tau about 0.09, power about 0.8).
SIG = dict(alpha=0.9, delta=0.4, w=0.15, u=0.35, v=0.3, e=1.0)


def reference(world, held):
    """Per experiment and run, H_T and X over the world's own T and M, written here from the protocol."""
    ht = held[:, :, world.t_rows].mean(axis=2)
    return ht, ht - held[:, :, world.m_rows].mean(axis=2)


def placebo(ht, values, n, gate_at):
    a, b = values[:, :n], values[:, n:]
    return ht[:, :n].mean(axis=1) >= gate_at, a.mean(axis=1) - b.mean(axis=1), b.max(axis=1) < a.min(axis=1)


def calibrated(world, seed: int, draws: int = 4000) -> tuple[dict, dict]:
    """As the measurements set the rule: each form's tau, the 95th percentile of |gap| over identical runs whose gate
    passed, and its power, on other draws, for the suspects thinned by a quarter (both from the reference above, not
    from answer()). Returns (constants for answer(), stated power per form)."""
    rng = np.random.default_rng(seed)
    n = world.per_side
    c = R.constants_for(world.start, world.situation)
    ht, x = reference(world, world.experiments(draws, rng))
    taus = {}
    for form, values in (("suspects", ht), ("excess", x)):
        gate, gap, _ = placebo(ht, values, n, c["gate"])
        taus[form] = float(np.quantile(np.abs(gap[gate]), 0.95, method="inverted_cdf"))
    ht, x = reference(world, world.experiments(draws, rng, effect=R.DROP))
    power = {}
    for form, values in (("suspects", ht), ("excess", x)):
        gate, gap, sep = placebo(ht, values, n, c["gate"])
        power[form] = float((gate & sep & (gap > taus[form])).sum() / gate.sum())
    return {**c, "taus": taus, "tau": taus[c["rule"]]}, power


def answered(world, constants, count, seed, **effect) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [world.answer(held, constants) for held in world.experiments(count, rng, **effect)]


def fire_rates(results) -> dict:
    """Each form's firing rate given the gate."""
    judged = [r for r in results if r["fewer"] is not None]
    assert all(r["come_back"]["passed"] for r in judged) and judged
    return {form: sum(bool(r["fewer"]["forms"][form]["fires"]) for r in judged) / len(judged)
            for form in ("suspects", "excess")}


@pytest.fixture(scope="module")
def nc_world():
    """A pool of paired starts (3 + 3 runs) whose suspects are held as their own RNA (X about 0)."""
    world = S.World(7, start="paired", sig=SIG, alpha_mean=0.6)
    constants, power = calibrated(world, 11)
    return world, constants, power


@pytest.fixture(scope="module")
def co_world():
    """A pool of cells-only starts (4 + 4 runs) whose suspects are held more than their own RNA (X about +0.08)."""
    world = S.World(7, start="cells", sig=SIG, alpha_mean=0.6, excess=0.47)
    constants, power = calibrated(world, 12)
    return world, constants, power


def test_the_worlds_are_what_they_stand_for(nc_world, co_world):
    for (world, constants, _), x_mean in ((nc_world, 0.0), (co_world, 0.08)):
        ht, x = reference(world, world.experiments(500, np.random.default_rng(3)))
        assert abs(float(x.mean()) - x_mean) < 0.015
        assert 0.5 < float(ht.mean()) < 0.7
        assert world.layout.items.cell[world.t_rows].max() < world.members
        assert all(0.03 < t < 0.15 for t in constants["taus"].values())
    res = nc_world[0].answer(nc_world[0].experiments(1, np.random.default_rng(4))[0], nc_world[1])
    assert res["pool"]["suspects"] == len(nc_world[0].t_rows) and len(res["pool"]["cells"]) == nc_world[0].members


def test_identical_sides_rarely_fire(nc_world):
    """No setting: every "clearly fewer" is a false alarm. At most 7.5 % given the gate in both forms (5 %, and the
    noise of tau and of the draws)."""
    world, constants, _ = nc_world
    rates = fire_rates(answered(world, constants, 400, 21))
    assert rates["excess"] <= 0.075 and rates["suspects"] <= 0.075


def test_a_drop_of_the_suspects_alone_is_found(nc_world):
    world, constants, power = nc_world
    results = answered(world, constants, 300, 22, effect=R.DROP)
    rates = fire_rates(results)
    for form in ("excess", "suspects"):
        assert rates[form] >= power[form] - 0.10, (form, rates[form], power[form])
    clearly = [r for r in results if r["verdict"] == "clearly_fewer"]
    assert len(clearly) == sum(bool(r["fewer"]["forms"]["excess"]["fires"]) for r in results if r["fewer"])


def test_a_pure_shrink_is_not_an_excess(nc_world):
    """Suspects and their own RNA thinned alike, in a pool whose suspects are held as their own RNA: the excess rule
    stays quiet; the suspects alone fire, and say "mostly by shrinking" every time."""
    world, constants, _ = nc_world
    results = answered(world, constants, 300, 23, effect=R.DROP, shrink=True)
    rates = fire_rates(results)
    assert rates["excess"] <= 0.075 and rates["suspects"] >= 0.5
    flags = [r["cost"]["shrinking"] for r in results if r["fewer"] and r["fewer"]["forms"]["suspects"]["fires"]]
    assert flags and all(flag is True for flag in flags)


def test_a_pure_shrink_that_fires_the_excess_says_so(co_world):
    """Where the suspects are held more than their own RNA, thinning both alike lowers X too, and the excess rule fires
    now and then (no bound here: the measurements put it near 9 in 100); nearly every such answer says "mostly by
    shrinking"."""
    world, constants, _ = co_world
    results = answered(world, constants, 1000, 24, effect=R.DROP, shrink=True)
    fired = [r for r in results if r["verdict"] == "clearly_fewer"]
    assert len(fired) >= 5
    assert sum(r["cost"]["shrinking"] is True for r in fired) >= 0.95 * len(fired)


# ---- the page's fixture -------------------------------------------------------------------------------------------------
def shape(value):
    """A value's keys and types: dicts by their keys in order, lists by their first item, numbers as one type."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    if isinstance(value, bool) or value is None:
        return type(value).__name__
    return "number" if isinstance(value, (int, float)) else type(value).__name__


def test_the_result_fixture_is_what_answer_gives():
    """frontend/test/fixtures/qc-test-result.json is answer()'s own output on the synthetic world (python
    tests/synth_pink.py <the fixture> writes it again), its caveats as engine/qc_test.py gives them (answer_caveats):
    the same examples, keys and types, what each says, and where it was not measured (a pool judged under k; a test the
    measurements did not cover, judged with the nearest measured row)."""
    committed = json.loads((ROOT / S.FIXTURE).read_text())
    made = S.fixture_examples()
    assert [e["name"] for e in committed] == [e["name"] for e in made] == list(S.EXPECT)
    for old, new in zip(committed, made):
        assert set(old) == {"name", "result"}
        assert shape(old["result"]) == shape(new["result"]), old["name"]
        assert S.said(old["result"]) == S.said(new["result"]) == S.EXPECT[old["name"]]
        for key in ("pool", "come_back", "constants", "cell"):
            assert shape(old["result"][key]) == shape(new["result"][key])
        assert old["result"]["caveats"] == new["result"]["caveats"], old["name"]
        if old["result"]["points"]["x"]:
            assert len(old["result"]["points"]["x"]) == old["result"]["pool"]["suspects"]
    situations = {(e["result"]["start"], e["result"]["situation"]) for e in committed}
    assert situations == {("paired", "square"), ("paired", "as_is"), ("cells", "as_is"), ("nuclei", "crop")}
    assert {e["result"]["rule"] for e in committed} == {"excess", "suspects"}
    assert any(e["result"]["parent_defaults"] is False for e in committed)
    for e in committed:          # the tooltip's numbers: the row it was judged with (its own where it was measured)
        r = e["result"]
        row = (r["constants"]["start"], r["constants"]["situation"])
        assert row == (("paired", "square") if e["name"] == "not_clear_unmeasured" else (r["start"], r["situation"]))
        c = R.constants_for(*row)
        assert r["constants"]["rates"] == {key: value if key in ("areas", "drop") else round(value, 4)
                                           for key, value in c["rates"].items()}
        # The caveats engine/qc_test.py gives: too_few with the suspects judged where they are fewer than k.
        short = [c for c in r["caveats"] if c["code"] == "too_few"]
        assert short == ([qc_pool.reason("too_few", have=r["pool"]["suspects"], need=r["pool"]["k"])]
                         if r["pool"]["suspects"] < r["pool"]["k"] else []), e["name"]
    caveats = {e["name"]: [c["code"] for c in e["result"]["caveats"]] for e in committed}
    assert caveats["not_clear_short_pool"] == caveats["too_few_cell_lost"] == ["too_few"]
    assert caveats["not_clear_unmeasured"] == ["whole_sample", "nuclei_start", "area_narrow"]
    assert not any(codes for name, codes in caveats.items()
                   if name not in ("not_clear_short_pool", "too_few_cell_lost", "not_clear_unmeasured"))
