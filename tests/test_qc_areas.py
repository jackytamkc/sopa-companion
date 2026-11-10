"""The mixing check's flagged areas (engine/qc_areas.py) on synthetic per-bin sums: measured constants (each mode's,
engine/qc_constants.py), smoothing per second, enter after >= 2 results spanning >= 2 s and clear after >= 3 spanning
>= 10 s, the sticky cap, the minimum drawn area, letters (kept under jitter, a released one rests 30 s), tags against
each result's own start sums (the frame's start block per bin; the start line's bins only for its own areas), single
snapshots, a late start's seed, the first reveal, the state an engine restart resumes, outlines, evidence and the
overview's per-bin share. The same code replays the measured runs result for result."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import harness_path
from sopa_companion.engine import qc_areas as A
from sopa_companion.engine import qc_constants as K

# The calibration, in the measuring harness's folder: checked against where it is (conftest).
CALIBRATION = harness_path("resolvi-prep", "phase0", "calibration.json", skip=False)
# calibration.json build.areas as the calibration wrote it.
PHASE0 = {
    "tiled": {"bin_um": 100.0, "window": 3, "alpha": 0.5, "ref_dt": 1.0, "enter_results": 2, "stay_results": 3,
              "enter_s": 2.0, "stay_s": 10.0, "enter_min_j": 40.0, "enter_min_m": 5.0, "enter_mult": 1.5,
              "enter_add": 0.04, "wilson_z": 1.645, "top_frac": 0.1, "stay_mult": 1.25, "stay_add": 0.02,
              "stay_min_j": 30.0, "cap": 8, "cap_swap_ratio": 2.0, "listed": 3, "min_area_j": 300.0,
              "letter_jaccard": 0.2, "letter_rest_s": 30.0, "rose_z": 2.0, "rose_points": 0.03, "tissue_min_j": 1.0},
    "detail": {"bin_um": 50.0, "window": 3, "alpha": 0.5, "ref_dt": 1.0, "enter_results": 2, "stay_results": 3,
               "enter_s": 2.0, "stay_s": 10.0, "enter_min_j": 30.0, "enter_min_m": 5.0, "enter_mult": 1.5,
               "enter_add": 0.04, "wilson_z": 1.645, "top_frac": 0.1, "stay_mult": 1.25, "stay_add": 0.02,
               "stay_min_j": 20.0, "cap": 4, "cap_swap_ratio": 2.0, "listed": 3, "min_area_j": 120.0,
               "letter_jaccard": 0.2, "letter_rest_s": 30.0, "rose_z": 2.0, "rose_points": 0.03, "tissue_min_j": 1.0},
}
GRID = {"x0": 1000.0, "y0": 2000.0, "bin": 50.0, "nx": 16, "ny": 16}
DETAIL = K.value("region", "areas")       # a dense region's area constants
TILED = K.value("tiled", "areas")         # a tiled region's (and a whole sample's)


def tissue(hot=(), J=20.0, M=2.0, hot_m=12.0, grid=GRID):
    """Dense sums: J judged and M mixed in every bin (10 %), hot_m mixed in the `hot` bins [(row, col), ...]."""
    shape = (grid["ny"], grid["nx"])
    out = {"J": np.full(shape, J), "M": np.full(shape, M)}
    for r, c in hot:
        out["M"][r, c] = hot_m
    return out


def patch(r0, c0, size=3):
    return [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]


def drawn(found):
    return [a for a in found if a.letter]


def test_the_area_constants_are_phase_0s():
    assert A.FROM_ITERS == 20 and A.OUTLINE_POINTS == 64
    # One table holds them (engine/qc_constants.py "areas"): a dense region's are the detail ones, a tiled region's and
    # a whole sample's the tiled ones (100 µm bins), until each mode's own repeats re-measure them. No copy of them is
    # kept here (a calibration that fills the table is read by every Tracker made after it).
    assert K.value("region", "areas") == DETAIL == PHASE0["detail"]
    assert K.value("tiled", "areas") == K.value("whole", "areas") == TILED == PHASE0["tiled"]
    assert not hasattr(A, "PARAMS") and not hasattr(A, "params_for")
    if CALIBRATION is not None:                            # the harness's own file: the build constants
        build = json.loads(CALIBRATION.read_text())["build"]["areas"]
        assert {k: build[k] for k in ("tiled", "detail")} == PHASE0
        assert "burn-in + 20" in build["draw_from"]


def test_an_area_enters_after_two_results_and_two_seconds_and_clears_after_ten():
    tracker = A.Tracker(GRID, DETAIL)
    shown = []
    for k in range(8):                                     # a result every 0.5 s: the 2 s rule decides, not the count
        found, _, _ = tracker.update(tissue(patch(6, 6)), 0.5 * k)
        shown.append(bool(drawn(found)))
    assert shown == [False] * 4 + [True] * 4                # from t = 2.0 s
    (area,) = drawn(found)
    assert area.letter == "A" and {6 * 16 + 6, 8 * 16 + 8} <= set(area.bins.tolist()) and area.judged >= 120
    assert area.per100 == pytest.approx(100 * (9 * 12 + 36 * 2) / 900) and area.extra > 0 and len(area.windows) == 21
    shown = {}
    for t in (10.0, 10.5, 15.0, 19.5, 20.0):               # cold again after a pause (the smoothing follows at once):
        found, _, _ = tracker.update(tissue(), t)           # it fails the stay test from 10.0 on and clears 10 s later
        shown[t] = bool(drawn(found))
    assert shown == {10.0: True, 10.5: True, 15.0: True, 19.5: True, 20.0: False} and not tracker.flag.any()
    other = A.Tracker(GRID, DETAIL)                         # and only on its third failing result
    for k in range(8):
        other.update(tissue(patch(6, 6)), 0.5 * k)
    seen = [bool(drawn(other.update(tissue(), t)[0])) for t in (10.0, 22.0, 22.5)]
    assert seen == [True, True, False]


def test_smoothing_weighs_the_newest_result_by_seconds_on_the_run_clock():
    tracker = A.Tracker(GRID, DETAIL)
    tracker.update(tissue(), 0.0)
    tracker.update(tissue(J=40.0), 0.1)                     # 0.1 s later: weight 1 - 0.5 ** 0.1
    assert tracker.sm["J"][0, 0] == pytest.approx(20 + 20 * (1 - 0.5 ** 0.1))
    tracker.update(tissue(J=40.0), 10.1)                   # after a 10 s pause: almost all new
    assert tracker.sm["J"][0, 0] == pytest.approx(40, abs=0.02)


def test_a_sum_a_result_lacks_keeps_its_smoothed_value():
    """The one live frame scored before the start line has no paired sums: the tags skip it, nothing decays."""
    tracker = A.Tracker(GRID, DETAIL)
    tracker.update(tissue(), 0.0)
    assert "n10" not in tracker.sm
    with_pairs = {**tissue(), "n10": np.full((16, 16), 4.0), "n01": np.zeros((16, 16))}
    tracker.update(with_pairs, 1.0)
    assert tracker.sm["n10"][0, 0] == 4.0                   # first seen: taken as it is
    tracker.update(tissue(), 2.0)
    assert tracker.sm["n10"][0, 0] == 4.0 and tracker.sm["J"][0, 0] == 20.0


def area(letter, bins, extra):
    return A.Area(letter=letter, bins=np.asarray(bins, np.int64), windows=np.zeros(0, np.int64), extra=extra,
                  judged=500.0)


def test_the_sticky_cap_and_resting_letters():
    tracker = A.Tracker(GRID, DETAIL)                      # cap 4
    tracker.areas = [area(L, [10 * i], e) for i, (L, e) in enumerate((("A", 10), ("B", 8), ("C", 6), ("D", 4)))]
    found = [area("", [10 * i], e) for i, e in enumerate((10, 8, 6, 4))] + [area("", [200], 7.9)]
    top, rest = tracker._select(found, 100.0)
    assert [a.letter for a in top] == ["A", "B", "C", "D"] and rest[0].letter == ""   # 7.9 < 2 x 4: D keeps its place
    found = [area("", [10 * i], e) for i, e in enumerate((10, 8, 6, 4))] + [area("", [200], 9.0)]
    top, rest = tracker._select(found, 101.0)
    assert [a.letter for a in top] == ["A", "E", "B", "C"]   # 9 >= 2 x 4: D goes and rests; the newcomer takes E
    assert tracker.rest == {"D": 131.0} and rest[0].letter == ""
    found = [area("", [0], 10), area("", [10], 8), area("", [250], 5.0)]
    top, _ = tracker._select(found, 110.0)                  # E and C are gone (they rest); D still rests: F
    assert [a.letter for a in top] == ["A", "B", "F"] and tracker.rest == {"D": 131.0, "E": 140.0, "C": 140.0}
    top, _ = tracker._select([area("", [0], 10), area("", [10], 8), area("", [250], 5.0), area("", [300], 2.0)], 135.0)
    assert [a.letter for a in top] == ["A", "B", "F", "D"]   # D may name an area again after 30 s; C, E not yet


def test_letters_stay_under_jitter_and_small_areas_are_not_drawn():
    tracker = A.Tracker(GRID, DETAIL)
    for k in range(6):
        found, _, _ = tracker.update(tissue(patch(6, 6)), float(k))
    assert [a.letter for a in drawn(found)] == ["A"]
    for k in range(6, 12):                                  # the patch moves one bin: same area, same letter
        found, _, _ = tracker.update(tissue(patch(6, 7)), float(k))
    assert [a.letter for a in drawn(found)] == ["A"]
    tiled = A.Tracker(GRID, TILED)                          # >= 300 judged cells to be drawn
    for k in range(6):
        found, _, _ = tiled.update(tissue(patch(6, 6), J=6.0, M=0.6, hot_m=3.6), float(k))
    assert found and not drawn(found) and 0 < found[0].judged < 300   # a "smaller spot"
    detail = A.Tracker(GRID, DETAIL)                        # >= 120 in detail runs: the same spot is drawn there
    for k in range(6):
        found, _, _ = detail.update(tissue(patch(6, 6), J=6.0, M=0.6, hot_m=3.6), float(k))
    assert [a.letter for a in drawn(found)] == ["A"]


def test_tags_against_the_start():
    shape = (16, 16)
    start = tissue(patch(2, 2))                             # already mixed at the start, top left
    tracker = A.Tracker(GRID, DETAIL, {"J": start["J"], "M": start["M"]})
    now = tissue(patch(2, 2) + patch(10, 10))               # still there, and a new one bottom right
    pairs = {"B": np.full(shape, 20.0), "N1": now["M"].copy(), "S1": start["M"].copy(),
             "n10": np.where(now["M"] > start["M"], now["M"] - 2, 0.0), "n01": np.zeros(shape)}
    for k in range(6):
        found, _, _ = tracker.update({**now, **pairs}, float(k))
    tags = {(a.bins[0] // 16 < 8): a.tag for a in drawn(found)}
    assert tags == {True: "already", False: "rose"}
    for a in drawn(found):                                  # the start's share over the area's bins
        assert (a.start_per100 > 15) == (a.tag == "already")
    plain = A.Tracker(GRID, DETAIL, {"J": tissue()["J"], "M": tissue()["M"]})
    no_rise = {**pairs, "n10": np.zeros(shape)}             # mixed now, no paired rise, not at the start: similar
    for k in range(6):
        found, _, _ = plain.update({**tissue(patch(10, 10)), **no_rise}, float(k))
    assert [a.tag for a in drawn(found)] == ["similar"]
    found, _ = A.start_areas(GRID, DETAIL, start)           # the start line's own areas
    (own,) = drawn(found)
    assert (own.letter, own.tag) == ("A", "already") and own.start_per100 == own.per100 > 15


def test_areas_pair_with_each_results_own_start_sums():
    """A frame's bins carry its own start (J0, M0: judged and looking mixed in its start block, the cells proseg
    started with at this frame's keep): an area's start share and its "already" tag come from them, smoothed as every
    sum, whatever fixed start a tracker was given; a result with no start of its own and no fixed start tags nothing
    ("similar", no start share)."""
    shape = (16, 16)
    now = tissue(patch(2, 2) + patch(10, 10))
    begun = tissue(patch(2, 2))                             # at the frame's keep: the top left was mixed already
    pairs = {"B": np.full(shape, 20.0), "N1": now["M"].copy(), "S1": begun["M"].copy(),
             "n10": np.where(now["M"] > begun["M"], now["M"] - 2, 0.0), "n01": np.zeros(shape)}
    own = {**now, **pairs, "J0": begun["J"], "M0": begun["M"]}
    for fixed in (None, {"J": tissue()["J"], "M": tissue()["M"]}):   # a fixed start says nothing was mixed: ignored
        tracker = A.Tracker(GRID, DETAIL, fixed)
        for k in range(6):
            found, _, sm = tracker.update(own, float(k))
        assert "J0" in sm and "M0" in sm
        tags = {(a.bins[0] // 16 < 8): (a.tag, a.start_per100 > 15) for a in drawn(found)}
        assert tags == {True: ("already", True), False: ("rose", False)}, fixed
    alone = A.Tracker(GRID, DETAIL)                         # neither: nothing to pair with
    for k in range(6):
        found, _, _ = alone.update({**now, **pairs}, float(k))
    assert {a.tag for a in drawn(found)} == {"similar"} and all(math.isnan(a.start_per100) for a in drawn(found))
    snap, _, _ = A.Tracker(GRID, DETAIL).snapshot(own)      # a single snapshot pairs with its own too
    assert sorted(a.tag for a in drawn(snap)) == ["already", "rose"]
    carried = alone.carried([(a.letter, a.bins) for a in drawn(found)], own)
    assert sorted(a.tag for a in carried) == ["already", "rose"]


def test_an_areas_own_support_and_threshold_are_its_runs_modes(monkeypatch):
    """The run's mode rides in the tracker's params ("mode"): an area's minimum support and size-aware threshold are
    that mode's (engine/qc_constants.py), never another mode's."""
    shape = (4, 4)
    members = np.arange(16)
    sums = {"J": np.full(shape, 4.0), "matched_paired": np.full(shape, 5.0),
            "matched_mixed_now": np.full(shape, 1.5), "matched_mixed_start": np.full(shape, 1.0)}
    assert A.local_change(members, sums)["enough"] is False                  # 64 judged, 80 paired: too few
    monkeypatch.setitem(K.TABLE["whole"], "local_min_judged", 50)
    monkeypatch.setitem(K.TABLE["whole"], "local_min_paired", 60)
    whole = A.local_change(members, sums, "whole")
    from sopa_companion.engine import qc_rules as R
    assert whole["enough"] is True and whole["threshold"] == round(R.local_threshold(0.3, 80, "whole"), 2)
    assert A.local_change(members, sums, "tiled")["enough"] is False
    tracker = A.Tracker({"x0": 0.0, "y0": 0.0, "bin": 50.0, "nx": 4, "ny": 4}, {**DETAIL, "mode": "whole"})
    assert tracker.mode == "whole" and A.Tracker(GRID, DETAIL).mode == "region"
    area = tracker.carried([("A", members)], sums)[0]
    assert area.local["enough"] is True and area.local == whole


def test_a_rise_needs_the_areas_own_depth_matched_change_with_enough_cells():
    """With the worker's depth-matched sums an area "rose" only when its own matched change counts (its
    own size-aware threshold, >= 300 judged and >= 400 paired cells); the paired cells' plain rise alone (more
    transcripts, the same mix at the same depth) is no rise. Without those sums (an older worker) the reference's test
    alone, as before."""
    shape = (16, 16)
    start = tissue()
    now = tissue(patch(10, 10))
    pairs = {"B": np.full(shape, 20.0), "N1": now["M"].copy(), "S1": start["M"].copy(),
             "n10": np.where(now["M"] > start["M"], now["M"] - 2, 0.0), "n01": np.zeros(shape)}

    def tag(extra, paired=20.0):
        tracker = A.Tracker(GRID, DETAIL, {"J": start["J"], "M": start["M"]})
        sums = {**now, **pairs}
        if extra is not None:
            sums.update(matched_paired=np.full(shape, paired), matched_mixed_start=start["M"].copy(),
                        matched_mixed_now=start["M"] + np.where(now["M"] > start["M"], extra, 0.0))
        for k in range(6):
            found, _, _ = tracker.update(sums, float(k))
        (area,) = drawn(found)
        return area
    assert tag(None).tag == "rose" and tag(None).local["paired"] is None       # no matched sums: the reference's test
    flat = tag(0.0)                                                             # the plain rise came from depth alone
    assert flat.tag == "similar" and flat.local["matched_change"] == 0.0 and flat.local["direction"] == "same"
    risen = tag(10.0)
    assert risen.tag == "rose" and risen.local["direction"] == "more" and risen.local["enough"]
    assert risen.local["judged"] == int(round(risen.judged)) and risen.local["paired"] >= 400
    few = tag(10.0, paired=5.0)                                                 # too few paired cells to say
    assert few.tag == "similar" and few.local["enough"] is False and few.local["direction"] is None


def test_an_areas_own_comparison_and_its_threshold():
    shape = (4, 4)
    members = np.arange(16)
    sums = {"J": np.full(shape, 25.0), "matched_paired": np.full(shape, 25.0),
            "matched_mixed_now": np.full(shape, 5.0), "matched_mixed_start": np.full(shape, 3.0)}
    local = A.local_change(members, sums)                   # 400 judged, 400 paired: 20 % now, 12 % at the start
    from sopa_companion.engine import qc_rules as R
    assert local == {"judged": 400, "paired": 400, "matched_change": 8.0,
                     "threshold": round(R.local_threshold(0.2, 400), 2), "enough": True, "direction": "more"}
    assert A.local_change(members[:8], sums)["enough"] is False and A.local_change(members[:8], sums)["direction"] is None
    assert A.local_change(members, {"J": sums["J"]}) == {"judged": 400, "paired": None, "matched_change": None,
                                                          "threshold": None, "enough": False, "direction": None}
    same = {**sums, "matched_mixed_now": np.full(shape, 3.05)}
    assert A.local_change(members, same)["direction"] == "same"     # 0.2 points: under its own threshold (2.1)


def test_the_tissues_edge_is_where_its_cells_meet_empty_bins():
    """edge_mask: occupied bins next to an unoccupied one (8 neighbours), and the grid's border for a whole sample;
    a region's border is a cut (border=False). An area touching it says so (Area.edge)."""
    cells = np.full((6, 8), 10.0)
    cells[2, 4] = 0.0                                       # a hole (a vessel) inside the tissue
    cells[:, 7] = 0.0                                       # background on the right
    whole = A.edge_mask({"cells": cells}, (6, 8))
    cut = A.edge_mask({"cells": cells}, (6, 8), border=False)
    assert whole[0, 0] and whole[5, 3] and not cut[0, 0] and not cut[5, 3]
    assert cut[1, 3] and cut[3, 5] and cut[0, 6] and not cut[4, 1] and not cut[2, 4]   # round the hole and the edge
    assert not whole[3, 1] and A.edge_mask({}, (6, 8)).sum() == 0
    tracker = A.Tracker(GRID, {**DETAIL, "edge_border": False})
    hot = tissue(patch(6, 6))
    for k in range(6):
        found, _, _ = tracker.update(hot, float(k))
    assert [a.edge for a in drawn(found)] == [False]
    hot["cells"] = np.full((16, 16), 25.0)
    hot["cells"][7:9, 9:12] = 0.0                           # empty bins beside it
    for k in range(6, 12):
        found, _, _ = tracker.update(hot, float(k))
    assert [a.edge for a in drawn(found)] == [True]


def test_a_single_snapshot_changes_nothing_and_a_late_start_waits_for_the_enter_rule():
    """A snapshot judges one result alone and changes nothing. A tracker that starts late (a check switched on after
    the areas are shown) commits nothing on its first result: a window that passes the enter test on that one result
    only is never drawn (it would otherwise stay >= 10 s), and one that passes on 2 results over 2 s is, as always;
    `warm` says when the enter rule could have decided."""
    tracker = A.Tracker(GRID, DETAIL)
    found, _, _ = tracker.snapshot(tissue(patch(3, 3) + patch(11, 11, 4)))
    assert [a.letter for a in drawn(found)] == ["A", "B"] and tracker.cold and not tracker.flag.any()
    big, small = drawn(found)                               # letters by rank: the larger patch first
    assert big.extra > small.extra and big.bins.min() // 16 >= 9 and small.bins.max() // 16 <= 6
    late = A.Tracker(GRID, DETAIL)
    found, _, _ = late.update(tissue(patch(3, 3) + patch(11, 11, 4)), 50.0)   # both pass the enter test here
    assert not drawn(found) and not late.flag.any() and not late.warm
    found, _, _ = late.update(tissue(patch(3, 3)), 60.0)                     # 10 s on: the second patch no longer does
    assert late.warm and len(drawn(found)) == 1 and drawn(found)[0].bins.max() // 16 <= 6   # 2 results over 2 s
    for t in range(61, 80):                                                  # the one-result window never shows
        found, _, _ = late.update(tissue(patch(3, 3)), float(t))
        assert len(drawn(found)) == 1
    shown = [(a.letter, a.bins) for a in drawn(found)]
    again = late.carried(shown)                                              # the same areas on these sums
    assert [(a.letter, a.bins.tolist()) for a in again] == [(L, b.tolist()) for L, b in shown]
    assert again[0].per100 == pytest.approx(drawn(found)[0].per100) and late.carried([]) == []


def test_the_first_reveal_names_areas_a_b_c_by_rank():
    tracker = A.Tracker(GRID, DETAIL)
    tracker.areas = [area("D", [5], 9.0), area("K", [50], 4.0)]
    tracker.rest = {"A": 1e9}
    tracker.reveal()
    assert [a.letter for a in tracker.areas] == ["A", "B"] and tracker.rest == {} and tracker.revealed


def test_the_reveal_keeps_the_start_lines_letters_where_its_areas_still_are():
    """Until burn-in + 20 iterations the page draws the start line's own areas (dashed). When the run's own areas are
    shown, one that overlaps a start area (Jaccard >= 0.2) keeps its letter, so the same place keeps its name; a new
    place takes the first free letter; a start letter no area kept rests 30 s (never another place's name at once)."""
    tracker = A.Tracker(GRID, DETAIL)
    tracker.t_prev = 40.0
    tracker.areas = [area("Q", [10, 11, 12, 13], 9.0),       # the run's top area: where the start's B was
                     area("R", [200, 201], 5.0),             # a new place
                     area("S", [90, 91, 92], 4.0)]           # mostly where the start's A was
    start = [("A", np.array([90, 91, 92, 93, 94])), ("B", np.array([12, 13, 14])), ("C", np.array([150]))]
    tracker.reveal(start)
    assert [a.letter for a in tracker.areas] == ["B", "D", "A"] and tracker.revealed
    assert tracker.rest == {"C": 70.0}                        # C rests until t = 40 + 30
    found = [area("", [10, 11, 12, 13], 9.0), area("", [200, 201], 5.0), area("", [90, 91, 92], 4.0),
             area("", [300], 3.0)]
    top, _ = tracker._select(found, 41.0)                     # the next result: letters held, the newcomer skips C
    assert [a.letter for a in top] == ["B", "D", "A", "E"]
    lone = A.Tracker(GRID, DETAIL)                            # nothing overlaps: first free letters, start ones rest
    lone.areas = [area("Q", [250], 2.0)]
    lone.reveal([("A", np.array([0]))])
    assert [a.letter for a in lone.areas] == ["B"] and lone.rest == {"A": 30.0}


def test_outlines_get_a_point_budget():
    """Each drawn area's outline gets at most OUTLINE_BUDGET / cap points (44 on the 100 µm grid with 8 drawn, 64 on
    the 50 µm one with 4): a ragged area is thinned to that; its shape id names the point count and the outline
    version (a stored outline is never reused for another count or way of drawing)."""
    assert A.outline_points(TILED) == 44 and A.outline_points(DETAIL) == 64
    grid = {"x0": 0.0, "y0": 0.0, "bin": 100.0, "nx": 30, "ny": 30}
    ragged = np.array([r * 30 + c for r in range(2, 26) for c in range(2, 26) if not (r % 2 == 0 and c % 2 == 0)
                       or r in (2, 25) and c % 3 == 0])
    ragged = A.components(np.isin(np.arange(900), ragged).reshape(30, 30))[0]
    edge = np.array([b for b in ragged.tolist() if b // 30 in (2, 3, 24, 25) or b % 30 in (2, 3, 24, 25)])
    id64, full = A.outline(edge, grid)
    id44, thin = A.outline(edge, grid, 44)
    assert len(full) > 44 and len(thin) == 44 and id64 != id44 and len(id44) == 12
    xs, ys = [p[0] for p in thin], [p[1] for p in thin]
    assert 150 <= min(xs) and max(xs) <= 2650 and 150 <= min(ys) and max(ys) <= 2650
    assert A.shape_id(edge, 44) == id44 and A.outline(edge, grid, 44)[1] == thin             # cached


def test_an_engine_restart_resumes_the_areas(tmp_path):
    """Tracker.state is JSON (QcManager's state.json): the smoothed sums, the hysteresis that can still decide
    anything, the drawn areas and the resting letters. A tracker read back from it, given the same results, draws the
    same areas with the same letters and flags the same windows as the one that never stopped, whatever the results."""
    rng = np.random.default_rng(7)
    pairs = {"B": np.full((16, 16), 20.0), "n10": np.zeros((16, 16)), "n01": np.zeros((16, 16))}
    steady = A.Tracker(GRID, DETAIL)
    for k in range(6):
        steady.update({**tissue(patch(6, 6)), **pairs}, float(k))
    steady.reveal()
    steady.rest = {"Q": 77.0}
    saved = json.dumps(steady.state())
    assert len(saved) < 4000                                # compact: 16 x 16 bins
    resumed = A.Tracker.from_state(json.loads(saved), DETAIL)
    assert resumed.revealed and [a.letter for a in resumed.areas] == ["A"] and resumed.t_prev == 5.0
    assert resumed.rest == {"Q": 77.0} and np.array_equal(resumed.flag, steady.flag)
    for k in range(6, 60):                                  # the same results from here on: the same areas
        hot = patch(6, 6) if k < 12 else patch(2, 12) if k < 30 else patch(int(rng.integers(0, 12)), int(rng.integers(0, 12)))
        frame = tissue(hot, M=float(rng.uniform(1.5, 2.5)), hot_m=float(rng.uniform(8, 14)))
        t = k + float(rng.uniform(-0.3, 0.3))
        a, _, _ = steady.update(frame, t)
        b, _, _ = resumed.update(frame, t)
        assert [(x.letter, x.bins.tolist(), x.tag) for x in drawn(a)] == \
               [(x.letter, x.bins.tolist(), x.tag) for x in drawn(b)]
        assert np.array_equal(steady.flag, resumed.flag)
        if k == 40:                                         # and again, mid-way
            resumed = A.Tracker.from_state(json.loads(json.dumps(resumed.state())), DETAIL)
    assert A.Tracker.from_state({"grid": [0, 0, 50, 4, 4.0]}, DETAIL) is None             # incomplete: start afresh
    broken = {**json.loads(saved), "flag": [999]}
    assert A.Tracker.from_state(broken, DETAIL) is None                                    # outside the grid
    other = {**json.loads(saved), "bins": 99}
    assert A.Tracker.from_state(other, DETAIL) is None
    damaged = json.loads(saved)
    damaged["sm"]["data"] = damaged["sm"]["data"][:-12] + "AAAAAAAAAAAA"                  # a torn write
    assert A.Tracker.from_state(damaged, DETAIL) is None
    assert A.Tracker.from_state({**json.loads(saved), "sm": {"keys": ["J"], "data": "!!"}}, DETAIL) is None


def test_a_whole_samples_saved_state_stays_small():
    """A grid as large as a whole sample's (108 x 107 bins at 100 µm, ~1,500 holding cells): the saved state is tens
    of KB (a whole sample's: ~30 KB; the dense arrays it replaces were 640 KB)."""
    grid = {"x0": 0.0, "y0": 0.0, "bin": 100.0, "nx": 108, "ny": 107}
    rng = np.random.default_rng(3)
    J = np.zeros((107, 108))
    J[30:70, 20:60] = rng.uniform(5, 60, (40, 40))
    tracker = A.Tracker(grid, TILED)
    for k in range(5):
        frame = {"J": J * rng.uniform(0.95, 1.05, J.shape), "M": J * rng.uniform(0.05, 0.3, J.shape)}
        frame.update({k2: J * 0.5 for k2 in ("B", "N1", "S1", "n10", "n01", "J0", "M0")})
        tracker.update(frame, float(k))
    saved = json.dumps(tracker.state())
    assert len(saved) < 60_000
    resumed = A.Tracker.from_state(json.loads(saved), TILED)
    assert all(np.allclose(resumed.sm[k], tracker.sm[k], rtol=1e-6) for k in A.CORE)


def test_components_are_8_connected_in_scan_order():
    rng = np.random.default_rng(4)
    for _ in range(20):
        mask = rng.random((13, 17)) < 0.3
        parts = A.components(mask)
        assert sorted(np.concatenate(parts).tolist()) == np.flatnonzero(mask).tolist()
        assert [p[0] for p in parts] == sorted(p[0] for p in parts)
        label = np.zeros(mask.size, int)
        for k, p in enumerate(parts):
            label[p] = k + 1
        label = label.reshape(mask.shape)
        for y, x in zip(*np.nonzero(mask)):                 # neighbours (8 ways) share a component
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < 13 and 0 <= xx < 17 and mask[yy, xx]:
                        assert label[yy, xx] == label[y, x]
    flag = np.zeros((5, 5), bool)
    flag[2, 2] = True
    assert A.dilate(flag, 3).sum() == 9 and A.dilate(flag, 1).sum() == 1


def inside(point, polygon):
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            hit = not hit
    return hit


def centre(b, grid=GRID):
    return (grid["x0"] + grid["bin"] * (b % grid["nx"] + 0.5), grid["y0"] + grid["bin"] * (b // grid["nx"] + 0.5))


def test_outlines_follow_the_bins_in_whole_micrometres():
    one = [5 * 16 + 5]
    shape, points = A.outline(np.array(one), GRID)
    assert len(shape) == 12 and A.outline(np.array(one), GRID) == (shape, points)     # cached, the same id
    assert 4 <= len(points) <= 16 and all(isinstance(v, int) for p in points for v in p)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    assert 1250 <= min(xs) and max(xs) <= 1300 and 2250 <= min(ys) and max(ys) <= 2300 and inside(centre(one[0]), list(points))
    diagonal = [3 * 16 + 3, 4 * 16 + 4]                    # touching at a corner: one area, one outline around both
    _, points = A.outline(np.array(diagonal), GRID)
    assert all(inside(centre(b), list(points)) for b in diagonal)
    ring = [r * 16 + c for r in range(8, 13) for c in range(2, 7) if not (r == 10 and c == 4)]   # a hole: left in
    _, points = A.outline(np.array(ring), GRID)
    assert inside(centre(10 * 16 + 4), list(points)) and all(inside(centre(b), list(points)) for b in ring)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    assert (min(xs), max(xs), min(ys), max(ys)) == (1100, 1350, 2400, 2650)    # the sides stay where the bins end
    rng = np.random.default_rng(1)
    big = A.components(A.dilate(rng.random((16, 16)) < 0.35, 3))[0]       # a large ragged area
    _, points = A.outline(big, GRID)
    assert 8 <= len(points) <= 64 and len(set(points)) == len(points)
    covered = sum(inside(centre(b), list(points)) for b in big.tolist())
    assert covered >= 0.95 * len(big)


def test_outlines_leave_empty_background_out_unless_that_splits_the_area():
    grid = {"x0": 0.0, "y0": 0.0, "bin": 100.0, "nx": 6, "ny": 4}
    cells = np.ones((4, 6))
    cells[:, 4:] = 0                                        # the tissue ends at column 4
    block = np.array([r * 6 + c for r in range(4) for c in range(1, 6)])
    kept = A.drawn_bins(block, {"cells": cells}, grid)
    assert sorted(kept.tolist()) == [r * 6 + c for r in range(4) for c in range(1, 4)]
    _, points = A.outline(kept, grid)
    assert max(p[0] for p in points) <= 400                 # the outline stops where the cells do
    cells[1:3, 2] = 0                                       # a C shape: still one piece
    cells[:, 3] = 0
    assert sorted(A.drawn_bins(block, {"cells": cells}, grid).tolist()) == [1, 2, 7, 13, 19, 20]
    split = np.zeros((4, 6))
    split[:, 1] = split[:, 4] = 1                           # leaving the empty ones out cuts it in two: all kept
    assert sorted(A.drawn_bins(block, {"cells": split}, grid).tolist()) == sorted(block.tolist())
    assert A.drawn_bins(block, {"J": np.zeros((4, 6))}, grid).tolist() == block.tolist()   # nothing judged: as is
    assert A.drawn_bins(block, {}, grid).tolist() == block.tolist()


def test_evidence_and_the_overview_share():
    shape = (16, 16)
    sm = {**tissue(), "cells": np.full(shape, 25.0), "tiny": np.zeros(shape), "tiny_start": np.zeros(shape),
          "lost_half": np.zeros(shape), "tx": np.full(shape, 25.0 * 300), "tx_start": np.full(shape, 25.0 * 250),
          "changed": np.full(shape, 50.0), "assigned": np.full(shape, 1000.0), "foreground": np.full(shape, 800.0)}
    for k in range(13):
        sm[f"gh{k}"] = np.zeros(shape)
    sm["gh8"][:] = 25.0                                     # every cell 1.10-1.25 x its start
    sm["changed"][0, 0], sm["tiny"][0, 0], sm["lost_half"][0, 0], sm["foreground"][0, 0] = 200.0, 3.0, 2.0, 500.0
    run = A.totals(sm)
    out = A.evidence(np.array([0, 1]), sm, run)
    assert out["cells"] == 50 and out["tiny"] == 3 and out["tiny_start"] == 0 and out["shrunk"] == 2
    assert out["growth"] == pytest.approx(math.exp((math.log(1.1) + math.log(1.25)) / 2) - 1, abs=1e-3)
    assert out["churn"] == round((250 / 50) / (run["changed"] / run["cells"]), 2) and out["churn"] > 2
    assert out["noise"] == round(1 - 1300 / 2000, 3) and out["depth"] == 300 and out["depth_start"] == 250
    assert A.evidence(np.array([0]), tissue(), A.totals(tissue()))["growth"] is None
    code = A.share_grid({"J": np.array([[10.0, 0.2], [4.0, 0.0]]), "M": np.array([[3.0, 0.1], [4.0, 0.0]])}, (2, 2))
    assert code.tolist() == [75, 255, 250, 255]


def test_bins_tables_become_dense_sums():
    table = {"grid": np.array([0.0, 0.0, 100.0, 4, 3]), "bin": np.array([0, 5, 11], np.int32),
             "judged": np.array([10, 20, 30], np.int32), "mixed": np.array([1, 2, 3], np.int32),
             "paired": np.array([9, 9, 9], np.int32), "growth_hist": np.ones((3, 13), np.int32),
             "growth_edges": np.array(A.GROWTH_EDGES, np.float32), "share_sum": np.ones(3, np.float32)}
    grid = A.grid_of(table)
    assert grid == {"x0": 0.0, "y0": 0.0, "bin": 100.0, "nx": 4, "ny": 3}
    dense = A.dense(table, grid)
    assert dense["J"].shape == (3, 4) and dense["J"][1, 1] == 20 and dense["J"][2, 3] == 30 and dense["J"].sum() == 60
    assert dense["B"].sum() == 27 and dense["gh12"].sum() == 3 and "share_sum" not in dense
    starts = A.dense({**table, "judged_start": np.array([8, 9, 1], np.int32), "mixed_start": np.array([1, 0, 1], np.int32)},
                     grid)
    assert starts["J0"][1, 1] == 9 and starts["J0"].sum() == 18 and starts["M0"].sum() == 2   # the frame's own start
    assert {"J0", "M0"} <= set(A.CORE)                                                       # kept across a restart
    assert dense["growth_edges"] == pytest.approx(A.GROWTH_EDGES)
    assert A.grid_of({"grid": np.array([0.0, 0.0, 0.0, 4, 3])}) is None and A.grid_of({}) is None


def test_outlines_leave_the_sparse_edge_bins_out_unless_that_splits_the_area():
    """A 100 µm bin with a few cells at the tissue's edge is mostly empty background: the outline is drawn around the
    area's fuller bins (EDGE_FRACTION of its typical bin) as long as they stay one piece."""
    grid = {"x0": 0.0, "y0": 0.0, "bin": 100.0, "nx": 5, "ny": 3}
    cells = np.full((3, 5), 22.0)
    cells[:, 4] = 2.0                                       # the edge column: 2 cells a bin
    block = np.array([r * 5 + c for r in range(3) for c in range(5)])
    kept = A.drawn_bins(block, {"cells": cells}, grid)
    assert sorted(kept.tolist()) == [r * 5 + c for r in range(3) for c in range(4)]
    _, points = A.outline(kept, grid)
    assert max(p[0] for p in points) <= 400                 # not through the background beside the tissue
    bridge = np.full((3, 5), 22.0)
    bridge[:, 2] = 2.0                                      # sparse in the middle: leaving it out would cut the area
    assert sorted(A.drawn_bins(block, {"cells": bridge}, grid).tolist()) == sorted(block.tolist())
    assert A.EDGE_FRACTION == 0.25 and A.OUTLINE_VERSION == 3


def test_the_overview_grid_holds_window_shares_and_each_bins_judged_cells():
    """GET /qc/{seq}/bins (planes 2): the share of judged cells that look mixed in each bin's 3 x 3 window (sum M / sum
    J: a bin with a few cells weighs what it holds), then the bin's own judged cells (how full it is)."""
    J = np.array([[20.0, 2.0, 0.0], [20.0, 20.0, 0.0]])
    M = np.array([[2.0, 2.0, 0.0], [4.0, 2.0, 0.0]])
    out = A.overview_grid({"J": J, "M": M}, (2, 3))
    share, judged = out[:6].reshape(2, 3), out[6:].reshape(2, 3)
    assert share[0, 0] == int(np.floor(10 / 62 * 250)) and share[0, 2] == int(np.floor(4 / 22 * 250))
    assert judged.tolist() == [[20, 2, 0], [20, 20, 0]] and out.dtype == np.uint8 and A.OVERVIEW_PLANES == 2
    empty = A.overview_grid({}, (2, 3))
    assert (empty[:6] == 255).all() and (empty[6:] == 0).all()
