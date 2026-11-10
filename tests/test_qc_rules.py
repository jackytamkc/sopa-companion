"""The mixing check's suggestions and trials (engine/qc_rules.py): the measured trial constants and thresholds (the
measuring harness's calibration, where it is), one entry per mode (engine/qc_constants.py: region, tiled, whole, each
read where it is used), the size-aware sampler threshold and the comparison block every summary carries, the band on
the depth-matched change, a knob change's verdicts (>= 5 results and >= 20 iterations, steady, only up to 300
iterations) and the harm the Undo card reports, drift (R3) judged like any other change (a band, a verdict, Undo;
"Likely to help"), an area judged on its own matched change only with the minimum support (never the tissue's depth
ratio), transcript loss alone never read as a change, every rule R0-R7 and U on its pattern and only then (no card
says a setting cannot fix something), cards merged across areas, R2 and R3 for the whole tissue, the never-suggested
list, one steady card at a time, and the previous-run readout on the two runs' matched changes."""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from conftest import harness_path
from sopa_companion.engine import qc_constants as K
from sopa_companion.engine import qc_rules as R

# The calibrations, in the measuring harness's folder: checked against where they are (conftest).
CALIBRATION = harness_path("resolvi-prep", "phase0", "calibration.json", skip=False)
PHASE_A = harness_path("resolvi-prep", "go-deeper", "phaseA", "calibration.json", skip=False)


# ---- constants ------------------------------------------------------------------------------------------------------
TRIAL = {"history_iterations": 50, "settle_after_burnin_iterations": 20, "min_history_results": 10, "mult": 2.0,
         "floor_per100": 1.0, "min_results": 5, "min_iterations": 20, "horizon_iterations": 300}


def test_the_trial_constants_are_phase_0s_and_the_thresholds_measured_again_on_proseg_s_count():
    """Regions and tiled regions hold the values measured again on proseg's own count (CONSTANTS_ID, MEASURED;
    tests/test_qc_calibration.py holds every one against the calibration's): the headline smoothed more (0.2 a second
    in a region, 0.1 in a tiled one: proseg's count moves more from one result to the next) and the model's spread
    wider (2.3 and 1.5 in 100: two learnings read the change since the start with opposite signs); the trial's, the
    size-aware threshold's k, the like-for-like change's floors (1.0 in 100 for regions and tiled regions, 0.3 for
    whole samples) and the minimum support measured again and kept. Whole samples keep the values measured on outline
    counts until their own repeat runs re-measure them. The module constants they replaced are gone: every reader asks
    the mode's."""
    assert K.MODES == ("region", "tiled", "whole") and set(K.TABLE) == set(K.MODES) == set(K.MEASURED)
    assert K.CONSTANTS_ID == "proseg-2026-10-31" and K.MEASURED == {"region": True, "tiled": True, "whole": False}
    for mode in K.MODES:
        assert K.value(mode, "trial") == TRIAL, mode
        assert (K.value(mode, "sampler_k"), K.value(mode, "local_min_judged"),
                K.value(mode, "local_min_paired")) == (0.6, 300, 400), mode
        assert (K.value(mode, "headline_ref_s"), K.value(mode, "shown_margin")) == (1.0, 0.3), mode
    assert {m: K.value(m, "headline_weight") for m in K.MODES} == {"region": 0.2, "tiled": 0.1, "whole": 0.5}
    assert {m: K.value(m, "matched_model") for m in K.MODES} == {"region": 2.3, "tiled": 1.5, "whole": 0.6}
    assert {m: K.value(m, "matched_fixed") for m in K.MODES} == {"region": 1.0, "tiled": 1.0, "whole": 0.3}
    assert R.SAMPLER_Z == 1.96
    # Tighter shapes: a "Rough guess"; drift: "Likely to help", measured on proseg's count (about 1 in 100).
    assert R.CONFIDENCE["R4"] == "rough" and R.CONFIDENCE["R3"] == "likely"
    assert not any(hasattr(R, name) for name in ("TRIAL", "SAMPLER_K", "MATCHED_FIXED", "MATCHED_MODEL",
                                                 "LOCAL_MIN_JUDGED", "LOCAL_MIN_PAIRED", "TRUST_JUDGED", "repeat_mode"))
    # Only the matched change's values are used: the raw headline's own values stay in calibration.json, never in the
    # code (they were used by this test only).
    assert not hasattr(R, "RAW_FIXED") and not hasattr(R, "RAW_MODEL")
    if CALIBRATION is not None:                            # the harness's own file: the build constants
        build = json.loads(CALIBRATION.read_text())["build"]
        band, verdict = build["trial"]["band"], build["trial"]["verdict"]
        assert band["fit"] == "flat" and {k: band[k] for k in ("history_iterations", "settle_after_burnin_iterations",
                                                                 "min_history_results", "mult", "floor_per100")} == \
            {k: TRIAL[k] for k in ("history_iterations", "settle_after_burnin_iterations", "min_history_results",
                                   "mult", "floor_per100")}
        assert {k: verdict[k] for k in ("min_results", "min_iterations", "horizon_iterations")} == \
            {k: TRIAL[k] for k in ("min_results", "min_iterations", "horizon_iterations")}
        assert build["copy_updates"]["R4_confidence"] == "Rough guess"
        assert build["timing_text"]["early_phases"] == list(R.FORMING_PHASES)
    if PHASE_A is not None:                                # the calibration: the change rule and its local version
        data = json.loads(PHASE_A.read_text())
        f7 = data["F7_meaningful_change"]["thresholds"]
        rec = f7["recommended"]
        assert f7["sampler_size_aware"]["k"] == K.value("region", "sampler_k")
        assert {m: rec["matched_change_points"][m] for m in ("region", "whole")} == \
            {m: K.value(m, "matched_fixed") for m in ("region", "whole")}
        # The model's spread, one value for both modes on outline counts: still a whole sample's; a region's was
        # measured again on proseg's count, wider.
        assert rec["matched_change_model_points"]["region"] == rec["matched_change_model_points"]["whole"] == \
            K.value("whole", "matched_model") < K.value("region", "matched_model")
        support = data["F2_depth"]["local_version"]["minimum_support"]
        assert support["area_judged_cells_now_for_any_verdict"] == K.value("region", "local_min_judged")
        assert support["area_paired_cells_for_matched_change_vs_start"] == K.value("region", "local_min_paired")
        for name, example in f7["sampler_size_aware"]["examples"].items():      # "J=300,p=0.15": 2.42 / 3.43
            j, share = (float(part.split("=")[1]) for part in name.split(","))
            assert R.sampler_threshold(share, j) == pytest.approx(example["arrow_vs_start"], abs=0.006), name
            assert R.sampler_threshold(share, j, "previous") == pytest.approx(example["previous_run"], abs=0.006), name


def test_setting_values_compare_through_float32():
    assert R.same(0.03999999910593033, 0.04) and R.same(None, None) and R.same(False, False)
    assert not R.same(0.02, 0.04) and not R.same(True, 1) and not R.same(None, 0.0)
    assert R.clean(0.03999999910593033) == 0.04 and R.clean(4000) == 4000 and R.clean(True) is True
    assert R.clean(None) is None and R.clean(0.0) == 0.0


# ---- when a change counts ----------------------------------------------------------------------------------------------
def test_the_size_aware_threshold_and_its_floors():
    # 1.96 x 0.6 x 100 x sqrt(p (1 - p) / n), x sqrt 2 against another run.
    assert R.sampler_threshold(0.15, 300) == pytest.approx(1.96 * 60 * math.sqrt(0.15 * 0.85 / 300))
    assert R.sampler_threshold(0.15, 400) == pytest.approx(2.1, abs=0.01)                       # local_version's 2.1
    assert R.sampler_threshold(0.15, 300, "previous") == pytest.approx(R.sampler_threshold(0.15, 300) * math.sqrt(2))
    assert R.sampler_threshold(None, 300) is None and R.sampler_threshold(0.1, None) is None and R.sampler_threshold(0.1, 0) is None
    assert R.sampler_threshold(0.0, 400) == R.sampler_threshold(1 / 400, 400) > 0          # a share of 0: one cell's worth
    # The run's matched change: the formula, never below what repeat runs' matched changes differed by.
    assert R.change_threshold("region", 0.11, 3300) == 1.0 and R.change_threshold("whole", 0.13, 37000) == 0.3
    assert R.change_threshold("region", 0.11, 500) == pytest.approx(R.sampler_threshold(0.11, 500))
    assert R.change_threshold("region", 0.11, 500) > 1.0
    assert R.change_threshold("whole") == 0.3 and R.change_threshold("region", None, None) == 1.0
    assert R.direction_of(-1.2, 1.0) == "fewer" and R.direction_of(1.0, 1.0) == "more" and R.direction_of(0.4, 1.0) == "same"
    assert R.direction_of(None, 1.0) is None
    # The modes: a whole sample by its cells (20,000 or more), else tiled or a dense region by how it runs.
    assert K.mode_of(41190, True) == K.mode_of(41190, False) == "whole" and K.mode_of(3504, False) == "region"
    assert K.mode_of(20000, False) == K.mode_of(20000.0, True) == "whole" and K.mode_of(19999, True) == "tiled"
    assert K.mode_of(19999, False) == "region"
    assert K.mode_of(3504, True) == "tiled" and K.mode_of(None, True) == "tiled" and K.mode_of(None, False) == "region"
    assert K.mode_of(True, False) == "region" and K.mode_of(float("nan"), False) == "region"
    assert R.change_threshold("tiled", 0.11, 3300) == 1.0 and R.change_threshold("unknown") == 1.0   # a region's


def test_a_runs_mode_is_one_rule_by_its_cells_tiled_by_how_it_runs_or_its_prior():
    """run_mode: mode_of the run's cells (info.json), tiled when the run is (meta mode "tiled") or its prior is kept
    in tiles (info prior format "tiles"): the check, the settings test and the page's caveats read this one rule (a
    tiled region of 4,823 cells is "tiled", whatever its region's kind)."""
    tiled_meta = {"mode": "tiled", "region": {"kind": "full"}}
    assert K.run_mode(tiled_meta, {"cells": 4823}) == "tiled"
    assert K.run_mode(tiled_meta, {"cells": 20000}) == K.run_mode({}, {"cells": 41190}) == "whole"
    assert K.run_mode({"mode": "dense"}, {"cells": 3504, "prior": {"format": "tiles"}}) == "tiled"
    assert K.run_mode({"mode": "dense"}, {"cells": 3504, "prior": {"format": "dense"}}) == "region"
    assert K.run_mode({"region": {"kind": "full"}}, {"cells": 3504}) == "region"
    assert K.run_mode(tiled_meta, None) == "tiled" and K.run_mode({}, None) == K.run_mode({}, {"prior": 3}) == "region"


def test_the_constants_kept_in_each_mode():
    """KEPT: what each mode keeps of Go deeper's and the worker's constants, one value in every mode measured on
    regions (CALIBRATION.md): nothing in a region; the pink dots' and the where words' in a tiled one; those and the
    start band, the UMAP's fewest transcripts and Compare's like-for-like depth in a whole sample. The constants block
    every summary and a run's record carry names them (`kept`); an unknown mode is a region's."""
    assert K.KEPT == {"region": (), "tiled": ("glow", "where_words"),
                      "whole": ("glow", "where_words", "start_band", "min_counts", "like_depth")}
    assert set(K.KEPT) == set(K.MODES)
    for mode in K.MODES:
        assert K.block(mode)["kept"] == list(K.KEPT[mode]), mode
    assert K.block("lasso")["kept"] == [] and K.block("whole")["kept"] is not K.block("whole")["kept"]


def test_transcript_loss_alone_reads_about_the_same():
    """A reproduction (a whole sample, every cell binomially thinned to 50 %, no boundary moved): the raw
    headline falls 12.85 -> 9.71 in 100 (-3.15), the depth-matched change reads +0.14 (as the measuring harness's
    calibration records it): about the same, no arrow. The same raw fall with a matched change past the
    threshold is "fewer"."""
    thinned = R.change_block(9.7058 - 12.8546, 0.13589971257366193, "whole", p=0.13, paired=32384, judged=30868)
    assert thinned["direction"] == "same" and thinned["meaningful"] is False and thinned["enough"] is True
    assert thinned["raw"] == pytest.approx(-3.1488, abs=1e-4) and thinned["threshold"] == 0.3
    assert thinned["model_threshold"] == 0.6 and "forming" not in thinned
    real = R.change_block(-1.722, -1.842, "whole", p=0.13, paired=35999, judged=37219)      # W_def at 401: real
    assert real["direction"] == "fewer" and real["meaningful"]
    crop = R.change_block(-0.108, -0.946, "region", p=0.11, paired=3170, judged=3309)      # under the region's 1.0
    assert crop["direction"] == "same" and crop["threshold"] == 1.0
    # Unrounded: 12.49 against 12.51 is a 0.02-point change, never a whole point.
    tiny = R.change_block(12.51 - 12.49, 0.02, "region", p=0.12, paired=3000, judged=3300)
    assert tiny["direction"] == "same" and tiny["raw"] == pytest.approx(0.02)


def test_the_change_carries_the_two_numbers_it_compares():
    """change_block gives per100_now and per100_start beside matched: of the cells compared now and at the start (at the
    matched depth), how many per 100 look mixed on each side (kept to 4 decimals, like matched); None for either
    without them (no matched change: no start line, nothing paired), whatever the verdict. A block whose own word is
    shown keeps them as they are (Steady copies it); a held word's are the next test's."""
    block = R.change_block(-0.6, 38.123456 - 43.4, "region", p=0.38, paired=3000, judged=3300, now=38.123456,
                           start=43.4)
    assert (block["per100_now"], block["per100_start"], block["matched"]) == (38.1235, 43.4, -5.2765)
    assert block["direction"] == "fewer" and abs(block["matched"] - (block["per100_now"] - block["per100_start"])) < 1e-4
    for forming, judged in ((True, 3300), (False, 30)):            # no verdict: the numbers stay
        quiet = R.change_block(-0.6, -5.0, "region", p=0.38, paired=3000, judged=judged, forming=forming, now=38.0,
                               start=43.0)
        assert quiet["direction"] is None and (quiet["per100_now"], quiet["per100_start"]) == (38.0, 43.0)
    for now, start, kept in ((None, None, (None, None)), (38.0, None, (38.0, None)), (float("nan"), 43.0, (None, 43.0)),
                             (True, "43", (None, None))):
        none = R.change_block(-3.0, None, "region", judged=3300, now=now, start=start)
        assert (none["per100_now"], none["per100_start"]) == kept, (now, start)
    assert (R.change_block(-3.0, None, "region")["per100_now"], R.change_block(-3.0, None, "region")["per100_start"]) \
        == (None, None)
    steady = R.Steady()
    assert steady.apply("change", block, 0.0) == block
    assert steady.apply("change", {**block, "per100_now": 38.0, "matched": -5.4}, 1.0) == \
        {**block, "per100_now": 38.0, "matched": -5.4}


def test_a_held_word_keeps_the_two_numbers_it_compared():
    """A word held steady (Steady) is an earlier block's: the line puts per100_now and per100_start beside it, so a held
    block carries those of the newest block whose own word was the one shown, never the newest numbers, which may have
    the other sign ("fewer" beside 38.6 now, 38.3 at the start). Its matched, threshold and direction_now stay the
    change's own (the harness judges them together), and the word's own numbers come back once it is shown again. A
    word released to a block's own (a frame shown as its own numbers) keeps that block's numbers; an area's block, which
    has none, gets none."""
    block = lambda now, start: R.change_block(now - start, now - start, "region", p=0.38, paired=3000,  # noqa: E731
                                              judged=3300, now=now, start=start)
    fewer, same = block(38.0, 39.5), block(38.6, 38.3)
    assert (fewer["direction"], same["direction"], fewer["threshold"]) == ("fewer", "same", 1.042)
    steady = R.Steady()
    steady.apply("change", block(38.2, 39.5), 0.0)
    assert steady.apply("change", fewer, 1.0) == fewer                        # its own word: the newest numbers
    held = steady.apply("change", same, 2.0)
    assert held == {**same, "direction": "fewer", "meaningful": True, "held": True, "direction_now": "same",
                    "meaningful_now": False, "held_s": 0.0, "per100_now": 38.0, "per100_start": 39.5}
    assert held["matched"] == 0.3 and held["per100_now"] < held["per100_start"]   # the word's numbers, its own change
    later = steady.apply("change", block(38.9, 39.0), 3.0)                     # still held: still the word's numbers
    assert (later["held"], later["per100_now"], later["per100_start"]) == (True, 38.0, 39.5)
    assert steady.apply("change", block(37.9, 39.5), 4.0) == block(37.9, 39.5)  # its own word again: its own numbers
    settled = [steady.apply("change", same, t) for t in (5.0, 10.5)]           # "same" for 5 s: shown, as it is
    assert settled[0]["held"] and settled[0]["per100_now"] == 37.9 and settled[1] == same
    steady.release("change", block(37.0, 39.5))                                # a frame's own word, released
    assert steady.verdicts["change"]["numbers"] == {"per100_now": 37.0, "per100_start": 39.5}
    released = steady.apply("change", same, 11.0)
    assert (released["direction"], released["per100_now"], released["per100_start"]) == ("fewer", 37.0, 39.5)
    local = {"matched_change": -1.4, "threshold": 1.0, "direction": "fewer", "meaningful": True}
    steady.apply("area:A", local, 0.0)
    area = steady.apply("area:A", {**local, "matched_change": 0.2, "direction": "same", "meaningful": False}, 1.0)
    assert area["held"] and "per100_now" not in area and "per100_start" not in area


def test_no_verdict_without_support_the_matched_change_or_while_cells_form():
    small = R.change_block(-3.0, -3.0, "region", p=0.1, paired=250, judged=280)
    assert small["direction"] is None and not small["meaningful"] and small["enough"] is False
    plain = R.change_block(-3.0, None, "region", p=None, paired=None, judged=3300)
    assert plain["direction"] is None and plain["matched"] is None and plain["threshold"] == 1.0
    forming = R.change_block(-3.0, -3.0, "region", p=0.1, paired=3000, judged=3300, forming=True)
    assert forming["direction"] is None and forming["forming"] is True and not forming["meaningful"]


def test_each_mode_reads_its_own_constants(monkeypatch):
    """A calibration fills one entry per mode: each reader takes its run's mode's (here the whole sample's changed, the
    others not), so a whole sample's size-aware k, floor, model spread, minimum support, trial band and horizon move
    while a region's and a tiled region's stay."""
    whole = K.TABLE["whole"]
    monkeypatch.setitem(whole, "sampler_k", 1.2)
    monkeypatch.setitem(whole, "matched_fixed", 0.1)
    monkeypatch.setitem(whole, "matched_model", 0.9)
    monkeypatch.setitem(whole, "local_min_judged", 50)
    monkeypatch.setitem(whole, "local_min_paired", 60)
    monkeypatch.setitem(whole, "trial", {**TRIAL, "min_history_results": 4, "floor_per100": 0.5,
                                         "horizon_iterations": 100})
    assert R.sampler_threshold(0.15, 300, mode="whole") == pytest.approx(2 * R.sampler_threshold(0.15, 300))
    assert R.change_threshold("whole") == 0.1 and R.change_threshold("region") == R.change_threshold("tiled") == 1.0
    assert R.supported(55, 70, "whole") and not R.supported(55, 70) and not R.supported(55, 70, "tiled")
    assert R.local_threshold(0.15, 400, "whole") == pytest.approx(2 * R.local_threshold(0.15, 400))
    block = R.change_block(-10.0, -10.0, "whole", p=0.1, paired=70, judged=55)     # its threshold: 8.4 at 70 cells
    assert block["model_threshold"] == 0.9 and block["enough"] and block["direction"] == "fewer"
    assert block["threshold"] == pytest.approx(R.sampler_threshold(0.1, 70, mode="whole"), abs=1e-3)
    assert R.change_block(-10.0, -10.0, "region", p=0.1, paired=70, judged=55)["enough"] is False
    rows = [{"v": 10.0, "mc": -2.0}] * 4
    assert R.matched_band(rows) is None and R.matched_band(rows, trial=K.value("whole", "trial"))["half"] == 0.5
    changes, start = change()
    short = history(n=6, start=164)                              # 6 results in the 50 iterations before the change
    big = R.Watch(1, changes, start, short, reveal=120, mode="whole")
    small = R.Watch(2, changes, start, short, reveal=120, mode="region")
    assert big.band is not None and big.band["half"] == 0.5 and big.horizon_iter == 270 and not big.no_history
    assert small.band is None and small.no_history and small.horizon_iter == 470
    assert R.Watch(3, changes, start, short, reveal=120).mode == "region"       # no mode: a region's constants
    assert R.Watch(4, changes, start, short, reveal=120, mode="tiled").band is None


def test_a_summary_names_the_constants_it_was_judged_with():
    """`block` is what every summary (and every run, Run.public) carries ({mode, id, measured, kept}: measured false is
    the page's caveat, for whole samples alone now that regions and tiled regions are measured on proseg's count; kept,
    Go deeper's and the worker's constants kept, not measured, in that mode: their tooltips' caveat); `value` hands out
    copies, so no reader changes the table."""
    assert K.block("tiled") == {"mode": "tiled", "id": "proseg-2026-10-31", "measured": True,
                                "kept": ["glow", "where_words"]}
    assert K.block("region")["measured"] is True and K.block("whole")["measured"] is False
    assert K.block("region")["kept"] == [] and K.block("nonsense")["kept"] == []
    assert K.block("whole")["kept"] == ["glow", "where_words", "start_band", "min_counts", "like_depth"]
    kept = K.block("whole")["kept"]
    kept.append("x")                                        # a copy
    assert "x" not in K.block("whole")["kept"] and "x" not in K.KEPT["whole"]
    assert K.block("nonsense")["mode"] == "region" and K.value("nonsense", "matched_fixed") == 1.0
    trial = K.value("region", "trial")
    trial["mult"] = 99.0
    assert K.value("region", "trial")["mult"] == 2.0
    with pytest.raises(KeyError):
        K.value("region", "no_such_constant")


# ---- the band on the depth-matched change ------------------------------------------------------------------------------
def test_the_band_centres_on_the_share_and_compares_the_matched_change():
    rows = [{"v": 10.0, "mc": -2.0}] * 9
    assert R.matched_band(rows) is None                    # fewer than 10 results: no band
    steady = R.matched_band(rows + [{"v": 10.0, "mc": -2.0}])
    assert (steady["centre"], steady["expected"], steady["half"]) == (10.0, -2.0, 1.0)
    assert (steady["low"], steady["high"]) == (9.0, 11.0)
    noisy = R.matched_band([{"v": 10.0, "mc": c} for c in [-3.0, -1.0] * 5])
    sd = float(np.std([-3.0, -1.0] * 5, ddof=1))
    assert noisy["expected"] == -2.0 and noisy["half"] == pytest.approx(2 * sd) and noisy["half"] > 2.0
    assert R.matched_band([{"v": 10.0, "mc": -2.0}] * 10, floor=1.6)["half"] == 1.6      # the run's own threshold
    assert R.matched_band([{"v": 10.0, "mc": None}] * 20) is None                         # no matched change: no band


# ---- a knob change ---------------------------------------------------------------------------------------------------
START = 10.0                                               # the start line's share in these tests


def entry(seq, value, iteration=None, phase="sampling", areas=None, tx=300.0, raw=None, mc="auto"):
    """A live result as QcManager keeps it: `value` the smoothed share; `mc` its depth-matched change against the start
    (by default the share's own change: nothing depth-related happened)."""
    return {"seq": seq, "iter": seq if iteration is None else iteration, "t": 0.5 * seq, "phase": phase, "v": value,
            "raw": value if raw is None else raw, "mc": value - START if mc == "auto" else mc, "tx": tx,
            "areas": areas or {}}


def history(n=50, value=10.0, start=120, wobble=0.0, areas=None):
    return [entry(s, value + (wobble if s % 2 else -wobble), areas=areas) for s in range(start, start + n)]


def change(start=170, phase="sampling", flag="cell-compactness", old=0.04, new=0.02):
    return [{"flag": flag, "from": old, "to": new}], {"seq": start, "iter": start, "phase": phase}


def test_a_change_gets_less_mixed_only_after_5_results_20_iterations_and_twice_below_the_band():
    changes, start = change()
    watch = R.Watch(1, changes, start, history(), reveal=120)
    assert watch.band["centre"] == 10.0 and watch.band["half"] == 1.0 and not watch.forming
    seen = {}
    for s in range(170, 200):
        watch.update(entry(s, 8.0))                         # clearly below 9.0 from the change on
        seen[s] = watch.verdict
    # Settling: 170-173; results counted from 174 (5th at 178); 20 iterations from 190; twice below: 191.
    assert all(seen[s] is None for s in range(170, 191)) and seen[191] == "less"
    assert watch.readout()["rows"] == [{"kind": "whole", "expected": 10.0, "now": 8.0, "change": -2.0},
                                       {"kind": "depth", "expected": 300, "now": 300}]
    assert watch.readout()["band"] == [{"iter": 170, "expected": 10.0, "low": 9.0, "high": 11.0},
                                       {"iter": 199, "expected": 10.0, "low": 9.0, "high": 11.0}]
    for s in (200, 201):                                    # back inside the band twice: no clear change again
        watch.update(entry(s, 10.2))
    assert watch.verdict is None
    for s in (202, 203, 204):                               # a single result outside is not enough
        watch.update(entry(s, 8.0 if s == 203 else 10.0))
    assert watch.verdict is None


def test_verdicts_stop_at_300_iterations_and_no_change_is_said_then():
    changes, start = change()
    watch = R.Watch(1, changes, start, history(), reveal=120)
    for s in range(170, 471):
        watch.update(entry(s, 10.3))
    assert watch.verdict is None and not watch.final
    watch.update(entry(471, 7.0))                           # past the horizon: no verdict from it
    assert watch.final and watch.verdict == "no_change" and watch.readout()["final"]
    watch.update(entry(472, 7.0))
    assert watch.verdict == "no_change"
    kept = R.Watch(2, changes, start, history(), reveal=120)
    for s in range(170, 200):
        kept.update(entry(s, 8.0))
    kept.update(entry(480, 10.0))
    assert kept.verdict == "less"                           # a clear verdict reached in time stays


def test_no_band_while_cells_form_or_without_10_settled_results():
    changes, _ = change()
    early = R.Watch(1, changes, {"seq": 60, "iter": 60, "phase": "burnin"}, history(start=10), reveal=120)
    assert early.forming and early.band is None and early.readout()["notes"] == ["forming"]
    settling = R.Watch(2, changes, {"seq": 128, "iter": 128, "phase": "sampling"}, history(start=100, n=28), reveal=120)
    # Only 8 results from burn-in + 20: cells have formed, but there is too little before the change to judge it (not
    # "still forming": the page says so in its own words).
    assert settling.band is None and settling.no_history and not settling.forming and not settling.no_matched
    assert settling.readout()["notes"] == ["no_history"] and not settling.readout()["forming"]
    for s in range(128, 200):
        settling.update(entry(s, 5.0))
    assert settling.verdict is None and settling.readout()["verdict"] is None and settling.readout()["band"] is None
    # Changed after burn-in but before burn-in + 20: cells are still settling, so it is "forming".
    gap = R.Watch(4, changes, {"seq": 110, "iter": 110, "phase": "sampling"}, history(start=60, n=50), reveal=120)
    assert gap.forming and not gap.no_history and gap.readout()["notes"] == ["forming"]
    # The history is the 50 iterations before the change only (older and later results are not in it).
    old = history(start=60, n=50, value=30.0) + history(start=120, n=50)
    assert R.Watch(3, changes, {"seq": 170, "iter": 170, "phase": "sampling"}, old, reveal=50).band["centre"] == 10.0


def test_transcript_loss_after_a_change_is_no_verdict():
    """Cells lose transcripts after a change (tighter shapes): the raw share drops 11.2 -> 8.8 with no change of the
    depth-matched change. No verdict (the row's "now" is the share at the same depth), and it says how many
    transcripts went. Results without the matched change (an older worker) give no verdict either: the plain share
    alone would have said "less"."""
    before = [entry(s, 11.2, tx=400.0, mc=-0.5) for s in range(120, 170)]
    changes, start = change()
    watch = R.Watch(1, changes, start, before, reveal=120)
    for s in range(170, 230):                                    # the share falls, the matched change does not
        watch.update(entry(s, 8.8, tx=330.0, mc=-0.5))
    readout = watch.readout()
    assert watch.verdict is None and readout["rows"][0] == {"kind": "whole", "expected": 11.2, "now": 11.2, "change": 0.0}
    assert readout["depth_loss"] == 18 and readout["rows"][1] == {"kind": "depth", "expected": 400, "now": 330}
    assert watch.raw_now == 8.8
    plain = R.Watch(2, changes, start, [dict(e, mc=None) for e in before], reveal=120)
    for s in range(170, 230):
        plain.update(entry(s, 8.8, tx=330.0, mc=None))
    assert plain.band is None and plain.verdict is None and plain.no_matched and "no_matched" in plain.readout()["notes"]
    assert not plain.no_history


def test_harm_needs_5_results_in_a_row_above_the_band_and_goes_after_3_not_above():
    changes, start = change(flag="prior-weight", old=1.0, new=0.5)
    watch = R.Watch(1, changes, start, history(), reveal=120)
    results = [(s, 12.5) for s in range(170, 196)]
    for s, value in results:
        watch.update(entry(s, value))
        if s == 193:
            assert not watch.harm and watch.harm_card() is None  # eligible from 190: 4 results above
    assert watch.harm and watch.verdict == "more"
    card = watch.harm_card()
    assert card == {"rule": "U", "flag": "prior-weight", "from": 0.5, "to": 1.0, "areas": [], "watch": 1,
                    "changed": {"flag": "prior-weight", "from": 1.0, "to": 0.5}, "before_per100": 10.0,
                    "after_per100": 12.5}
    for s in (196, 197):
        watch.update(entry(s, 10.0))
    assert watch.harm
    watch.update(entry(198, 10.0))
    assert not watch.harm and watch.harm_card() is None
    for s in range(199, 205):
        watch.update(entry(s, 12.5))
    assert watch.harm_card() is not None
    watch.dismissed = True                                       # answered "Keep": the card goes
    assert watch.harm_card() is None
    frozen = R.Watch(2, changes, start, history(), reveal=120)
    for s in range(170, 200):
        frozen.update(entry(s, 12.5))
    frozen.freeze(mixed_up=True)                                 # another change came: nothing more to undo here
    assert frozen.harm_card() is None and frozen.readout()["notes"] == ["mixed_up"]
    frozen.update(entry(200, 5.0))
    assert frozen.readout()["rows"][0]["now"] == 12.5 and frozen.readout()["verdict"] is None


def area_value(share, judged=400, mc=None, paired=500):
    """An area's value at one result: its share, judged cells, own matched change (by default its share's own change
    from 20 in 100) and paired cells."""
    return {"share": share, "judged": judged, "mc": share - 20.0 if mc is None else mc, "paired": paired}


def test_an_area_with_the_minimum_support_rising_above_its_own_band_is_harm_too():
    areas = {"A": [20.0, 400, 0.0, 500], "B": [20.0, 150, 0.0, 180]}
    changes, start = change()
    watch = R.Watch(1, changes, start, history(areas=areas), reveal=120,
                    areas={"A": (np.arange(9), 400, 500), "B": (np.arange(9, 12), 150, 180)})
    assert set(watch.area_band) == {"A", "B"} and watch.area_band["A"]["centre"] == 20.0
    for s in range(170, 200):                                    # the tissue as before; A and B rose
        watch.update(entry(s, 10.0), {"A": area_value(24.0), "B": area_value(30.0, 150, paired=180)})
    card = watch.harm_card()
    assert card["areas"] == ["A"] and card["before_per100"] == 20.0 and card["after_per100"] == 24.0   # B too small
    readout = watch.readout("B")
    assert readout["small_area"] and "small_area" in readout["notes"]
    # Too few cells for a comparison of its own: its share for the row, never a like-for-like "now".
    assert readout["rows"][0] == {"kind": "area", "area": "B", "expected": 20.0, "now": 30.0}


def test_an_area_whose_support_collapses_after_the_change_loses_its_verdict_and_its_undo_card():
    """A reproduction: an area with 400 judged and paired cells when the change started, then results where
    one cell is left (the rest moved away or stopped being judged), whose matched change is 100 points. The trial path
    gave it "more", an Undo card and an impossible 120 in 100; its own support now is needed too."""
    areas = {"A": [20.0, 400, 0.0, 400]}
    changes, start = change()
    watch = R.Watch(1, changes, start, history(areas=areas), reveal=120, areas={"A": (np.arange(9), 400, 400)})
    for s in range(170, 200):                                    # A rises clearly on its own 400 cells: harm
        watch.update(entry(s, 10.0), {"A": area_value(26.0, 400, paired=400)})
    assert watch.area_harm.get("A") and watch.area_verdict.get("A") == "more" and watch.harm_card()["areas"] == ["A"]
    for s in range(200, 210):                                    # one judged, paired cell left
        watch.update(entry(s, 10.0), {"A": area_value(100.0, 1, mc=100.0, paired=1)})
    readout = watch.readout("A")
    assert watch.harm_card() is None and not watch.area_harm.get("A") and watch.area_verdict.get("A") is None
    assert readout["on"] == "whole" and readout["small_area"] and "small_area" in readout["notes"]
    assert readout["rows"][0]["now"] == 100.0 and "change" not in readout["rows"][0], "its share, never 120 in 100"
    # Back to 400 cells: compared again, its runs of results counted afresh.
    for s in range(210, 230):
        watch.update(entry(s, 10.0), {"A": area_value(26.0, 400, paired=400)})
    assert watch.area_verdict.get("A") == "more" and watch.readout("A")["on"] == "area"
    # A row is a share in 100: the band's centre plus a matched change never reads past 0-100.
    assert (R.share100(120.0), R.share100(-3.0), R.share100(12.34), R.share100(None)) == (100.0, 0.0, 12.3, None)


def test_a_verdict_is_held_steady_at_its_threshold_and_never_held_from_no_verdict():
    """The crop's smoothed like-for-like change sat at -0.94...-1.06 against a 1.0
    threshold and its verdict flipped 16-32 times a minute. A live verdict changes only once the new direction held for
    5 s over 2 results (at the latest after 15 s); the change's own direction is always kept as direction_now."""
    steady = R.Steady()
    block = lambda d, m=-1.0: {"matched": m, "threshold": 1.0, "direction": d, "meaningful": d in ("more", "fewer")}
    shown = []
    for k in range(40):                                          # one result a second, alternating at the threshold
        d = "fewer" if k % 2 == 0 else "same"
        out = steady.apply("change", block(d, -1.02 if d == "fewer" else -0.98), float(k))
        shown.append(out["direction"])
        if out.get("held"):                                      # an earlier verdict held: its own is said beside it
            assert out["direction_now"] == d and out["meaningful_now"] == (d == "fewer") and out["direction"] != d
        else:                                                    # its own verdict: the block as the rule gave it
            assert out["direction"] == d and "direction_now" not in out and "held" not in out
        assert out["meaningful"] == (out["direction"] == "fewer")
    flips = sum(1 for a, b in zip(shown, shown[1:]) if a != b)
    assert flips <= 2, f"{flips} flips: {shown}"                 # alternating every result never holds 5 s ...
    assert shown[0] == "fewer"                                   # ... and the first verdict shows at once
    # A new direction that holds: shown after 5 s (and at least 2 results), not before.
    s2 = R.Steady()
    s2.apply("change", block("same", -0.2), 0.0)
    assert s2.apply("change", block("fewer", -1.4), 1.0)["direction"] == "same"
    assert s2.apply("change", block("fewer", -1.4), 3.0)["direction"] == "same"
    held = s2.apply("change", block("fewer", -1.4), 5.5)
    assert held["direction"] == "same" and held["held"] and held["held_s"] == 4.5
    assert s2.apply("change", block("fewer", -1.4), 6.0)["direction"] == "fewer"
    # No verdict (too few cells, forming) shows at once, and so does the first verdict after none.
    assert s2.apply("change", block(None, -1.4), 7.0)["direction"] is None
    assert s2.apply("change", block("more", 1.4), 8.0)["direction"] == "more"
    # At the latest after 15 s of the shown direction not being the change's own, whatever the flicker between others.
    s3 = R.Steady()
    s3.apply("change", block("fewer"), 0.0)
    outs = [s3.apply("change", block("same" if k % 2 else "more", 0.0), float(k)) for k in range(1, 17)]
    assert outs[-1]["direction"] != "fewer"
    # Areas: one key per letter, forgotten once the letter is not drawn.
    s3.apply("area:A", block("fewer"), 0.0); s3.apply("area:B", block("more"), 0.0)
    s3.keep("area:", {"A"})
    assert "area:A" in s3.verdicts and "area:B" not in s3.verdicts and "change" in s3.verdicts
    assert s3.apply("prev", None, 1.0) is None and "prev" not in s3.verdicts


def test_a_held_word_is_released_to_a_blocks_own():
    """Steady.release: a frame shown as its own numbers (one the run rests on, the count proseg saves) says its own
    word at once: the record's shown direction becomes the block's own and nothing is pending, so the next live result
    is judged from it; no block forgets the key."""
    block = lambda d: {"matched": -1.4, "threshold": 1.0, "direction": d, "meaningful": d in ("more", "fewer")}
    steady = R.Steady()
    steady.apply("change", block("same"), 0.0)
    held = steady.apply("change", block("fewer"), 1.0)
    assert held["direction"] == "same" and held["held"] and steady.verdicts["change"]["pending"] == "fewer"
    steady.release("change", block("fewer"))
    assert steady.verdicts["change"] == {"shown": "fewer", "pending": None, "since": None, "count": 0, "differs": None,
                                         "numbers": {"per100_now": None, "per100_start": None}}
    assert steady.apply("change", block("fewer"), 2.0) == block("fewer")       # its own word, not held
    out = steady.apply("change", block("same"), 3.0)
    assert out["direction"] == "fewer" and out["held"]                         # a new word is held from the released one
    steady.release("change", None)
    assert "change" not in steady.verdicts


def test_an_unchanged_area_stays_unchanged_whatever_the_tissue_does():
    """A reproduction: an UNCHANGED 400-cell area at 20 % while the tissue's cells move to deeper bands (the
    tissue's raw share 15 -> 20 %, its depth-band ratio 1.333). The old readout scaled the area by the tissue's ratio
    (26.67 %, "more", Undo); judged on its own matched change it stays at 20 % with no verdict and no harm."""
    areas = {"A": [20.0, 400, 0.0, 450]}
    changes, start = change()
    before = [entry(s, 15.0, areas=areas, mc=0.0) for s in range(120, 170)]
    watch = R.Watch(1, changes, start, before, reveal=120, areas={"A": (np.arange(9), 400, 450)})
    for s in range(170, 240):                                    # the tissue's share rises, A's own numbers do not
        watch.update(entry(s, 20.0, mc=0.0), {"A": area_value(20.0, 400, mc=0.0, paired=450)})
    readout = watch.readout("A")
    assert watch.area_now["A"] == 20.0 and readout["rows"][0]["now"] == 20.0 and readout["on"] == "area"
    assert readout["verdict"] is None and watch.harm_card() is None and not watch.area_harm.get("A")
    assert watch.verdict is None                                 # nor the tissue: its matched change did not move


def test_an_area_without_enough_paired_cells_gets_no_verdict_of_its_own():
    """F2_depth.local_version: no verdict of an area's own below 300 judged or 400 paired cells, and it says so; the
    whole tissue's verdict is given instead."""
    areas = {"A": [28.0, 900, 0.0, 380]}
    changes, start = change()
    watch = R.Watch(1, changes, start, history(areas=areas), reveal=120, areas={"A": (np.arange(9), 900, 380)})
    for s in range(170, 200):
        watch.update(entry(s, 10.0), {"A": area_value(19.0, 900, mc=-9.0, paired=380)})
    readout = watch.readout("A")
    assert not watch.trusted("A") and readout["on"] == "whole" and "small_area" in readout["notes"]
    assert readout["verdict"] is None                           # the tissue: no clear change


# ---- a knob change judged with its run's mode's constants -----------------------------------------------------------
# Every mode holds today's values, so here a tiled region's (or a whole sample's) entry gets values of its own, as a
# calibration may fill it, and every reader inside Watch is checked against a dense region's Watch on the same results:
# the history window, the band's and the verdict's minima, the horizon as results arrive, the run's own floor, and an
# area's band and minimum support as the change started and now.
OWN_TRIAL = {**TRIAL, "history_iterations": 20, "min_history_results": 4, "min_results": 2, "min_iterations": 6,
             "floor_per100": 0.5, "horizon_iterations": 100}


@pytest.fixture(params=["tiled", "whole"])
def own_mode(request, monkeypatch):
    """A mode with constants of its own: a shorter history window, smaller minima, a nearer horizon, a smaller minimum
    support (50 judged, 60 paired cells), twice a region's sampler k and a lower fixed floor."""
    table = K.TABLE[request.param]
    monkeypatch.setitem(table, "trial", dict(OWN_TRIAL))
    monkeypatch.setitem(table, "local_min_judged", 50)
    monkeypatch.setitem(table, "local_min_paired", 60)
    monkeypatch.setitem(table, "sampler_k", 1.2)
    monkeypatch.setitem(table, "matched_fixed", 0.1)
    return request.param


def watches(mode, rows, **kw):
    """The same knob change (at 170) watched on the same results with `mode`'s constants and with a dense region's."""
    changes, start = change()
    return (R.Watch(1, changes, start, rows, reveal=120, mode=mode, **kw),
            R.Watch(2, changes, start, rows, reveal=120, mode="region", **kw))


def test_a_knob_changes_band_takes_its_modes_history_window(own_mode):
    own, region = watches(own_mode, history(wobble=0.2))        # 50 results before the change (120-169)
    assert own.band["n"] == 20 and region.band["n"] == 50       # the mode's 20 iterations before it, a region's 50


def test_too_little_history_before_a_knob_change_is_its_modes(own_mode):
    """6 results before the change, none with the like-for-like change: enough for the mode's band (4), but without a
    matched change ("no_matched"); too few for a region's (10: "no_history")."""
    own, region = watches(own_mode, [dict(e, mc=None) for e in history(n=6, start=164)])
    assert own.band is None and own.history_ok and own.no_matched and not own.no_history
    assert region.band is None and not region.history_ok and region.no_history and not region.no_matched


def test_a_knob_changes_verdict_takes_its_modes_minimum_results_and_iterations(own_mode):
    own, region = watches(own_mode, history(n=20, start=150))
    seen = {}
    for s in range(170, 192):                                   # clearly below both bands from the change on
        own.update(entry(s, 8.0))
        region.update(entry(s, 8.0))
        seen[s - 170] = (own.verdict, region.verdict)
    # Settling to 3 iterations after the change, results counted from 4. The mode's >= 2 results and >= 6 iterations
    # from 6, twice below: 7; a region's >= 5 results and >= 20 iterations from 20, twice below: 21.
    assert seen[6] == (None, None) and seen[7] == ("less", None)
    assert seen[20] == ("less", None) and seen[21] == ("less", "less")


def test_a_knob_change_ends_at_its_modes_horizon_as_results_arrive(own_mode):
    own, region = watches(own_mode, history(n=20, start=150))
    assert (own.horizon_iter, region.horizon_iter) == (270, 470)
    for watch in (own, region):
        watch.update(entry(270, 10.0))                          # at the mode's horizon: still judged
    assert not own.final and not region.final
    for watch in (own, region):
        watch.update(entry(271, 10.0))                          # past it: no clear change, said once and for all
    assert own.final and own.verdict == "no_change"
    assert not region.final and region.verdict is None


def test_a_knob_changes_band_is_never_narrower_than_the_runs_threshold_for_its_mode(own_mode):
    """Without a floor given, the band's floor is the run's own threshold before the change (its results' share mp and
    paired cells mn) with the mode's sampler k and fixed floor: here twice a region's."""
    own, region = watches(own_mode, [dict(e, mp=0.1, mn=200.0) for e in history(n=20, start=150)])
    assert own.floor == pytest.approx(R.change_threshold(own_mode, 0.1, 200.0))
    assert region.floor == pytest.approx(R.change_threshold("region", 0.1, 200.0))
    assert own.floor == pytest.approx(2 * region.floor)
    assert own.band["half"] == pytest.approx(own.floor) and region.band["half"] == pytest.approx(region.floor)


def test_an_areas_band_and_support_in_a_knob_change_are_its_modes(own_mode):
    """An area of 60 judged and 70 paired cells with 6 results of its own before the change: the mode's minimum support
    (50 judged, 60 paired) and band minimum (4 results) hold it, a region's (300, 400; 10) do not. Its band is never
    narrower than its own size-aware threshold with the mode's k (twice a region's), and its support now is the
    mode's too."""
    own_area = {"A": [12.0, 60, -1.0, 70]}                      # share, judged, matched change, paired
    rows = history(n=14, start=150) + history(n=6, start=164, areas=own_area)
    own, region = watches(own_mode, rows, areas={"A": (np.arange(9), 60, 70)})
    assert "A" in own.area_band and "A" not in region.area_band
    assert own.area_band["A"]["half"] == pytest.approx(R.local_threshold(0.12, 70, own_mode))
    assert own.area_band["A"]["half"] == pytest.approx(2 * R.local_threshold(0.12, 70, "region"))
    assert own.supported("A") and own.trusted("A") and not region.supported("A")
    now = {"A": area_value(12.0, 60, mc=-1.0, paired=70)}
    own.update(entry(171, 10.0), now)
    region.update(entry(171, 10.0), now)
    assert own.area_ok_now["A"] is True and region.area_ok_now["A"] is False


# ---- the rules ---------------------------------------------------------------------------------------------------------
LIVE = {"cell-compactness": 0.03999999910593033, "prior-weight": 1.0, "no-diffusion": False}


def area(letter="A", tag="rose", judged=400, growth=0.0, tiny=2, tiny_start=2, shrunk=0, cells=500, churn=1.0, **extra):
    return {"L": letter, "tag": tag, "judged": judged, "per100": 22.0, "start_per100": 12.0, "growth": growth,
            "tiny": tiny, "tiny_start": tiny_start, "shrunk": shrunk, "cells": cells, "churn": churn, **extra}


def ctx(*areas, phase="sampling", iteration=400, live=None, whole=None, tried=(), harm=None, later_s=162.0,
        min_judged=40):
    return {"phase": phase, "iter": iteration, "min_judged": min_judged, "live": {**LIVE, **(live or {})},
            "areas": list(areas), "whole": whole or {}, "tried": list(tried), "harm": harm, "later_s": later_s}


def rules(context):
    return [(c["rule"], c["areas"]) for c in R.evaluate(context)]


def test_r0_too_early_or_too_few_cells():
    grown = area(growth=0.18)
    assert rules(ctx(grown, phase="init", iteration=0)) == [] and rules(ctx(grown, phase="burnin", iteration=9)) == []
    assert rules(ctx(grown)) == [("R4", ["A"])]
    assert rules(ctx(area(judged=39, growth=0.18))) == []                        # under 40 judged cells (100 µm bins)
    assert rules(ctx(area(judged=35, growth=0.18), min_judged=30)) == [("R4", ["A"])]   # 30 on 50 µm bins


def test_r1_already_high_at_the_start_cause_not_known():
    assert rules(ctx(area(tag="already", growth=0.05))) == [("R1", ["A"])]
    assert rules(ctx(area(tag="already", growth=0.15))) == []                    # it grew: not "has not made them ..."
    assert rules(ctx(area(tag="already", growth=None))) == []                    # no evidence about its growth
    assert rules(ctx(area(tag="similar", growth=0.0))) == []                     # "about as mixed": no card
    (card,) = R.evaluate(ctx(area(tag="already", growth=0.05)))
    # Not "no setting is likely to fix this": the cause is not known; it points to the area's cells.
    assert card["actions"] == ["show_start", "inspect"] and card["main"] == "show_start"
    assert card["confidence"] == "unknownCause" and "flag" not in card


def test_r2_go_back_to_the_standard_cell_shape():
    loose = {"cell-compactness": 0.08}
    assert rules(ctx(area(growth=0.0), live=loose)) == [("R2", ["A"])]           # got more mixed
    assert rules(ctx(area(tag="similar", growth=0.2), live=loose)) == [("R2", ["A"])]   # or grew > 10 %
    assert rules(ctx(area(tag="similar", growth=0.05), live=loose)) == []
    assert rules(ctx(area(growth=0.0))) != [("R2", ["A"])]                       # the standard 0.04: not loosened
    (card,) = R.evaluate(ctx(area(growth=0.0), live=loose))
    assert (card["flag"], card["from"], card["to"], card["confidence"]) == ("cell-compactness", 0.08, 0.04, "likely")


def test_r3_turn_transcript_drift_back_on():
    """Drift off: "Turn transcript drift back on", "Likely to help" (repeat runs on proseg's count: switching drift
    back on lowered mixing in every one, about 1 in 100); Try it (drift acts on the running run, and its trial is judged
    like any other)."""
    assert rules(ctx(area(tag="similar"), live={"no-diffusion": True})) == [("R3", ["A"])]
    (card,) = R.evaluate(ctx(area(tag="similar"), live={"no-diffusion": True}))
    assert (card["from"], card["to"], card["confidence"]) == (True, False, "likely")
    assert card["actions"] == ["try", "why"] and card["main"] == "try"
    assert rules(ctx(area(tag="similar"))) == []


def test_r4_tighter_shapes_is_a_rough_guess():
    (card,) = R.evaluate(ctx(area(growth=0.18)))
    assert (card["rule"], card["from"], card["to"], card["growth"]) == ("R4", 0.04, 0.02, 0.18)
    assert card["confidence"] == "rough" and card["timing"] == "later" and card["later_s"] == 162
    assert card["actions"] == ["new_run", "try", "why"] and card["main"] == "new_run"   # after burn-in: a new run first
    assert rules(ctx(area(growth=0.18), live={"cell-compactness": 0.03})) == [("R4", ["A"])]
    assert rules(ctx(area(growth=0.18), live={"cell-compactness": 0.025})) == []        # already tight
    assert rules(ctx(area(growth=0.09))) != [("R4", ["A"])]
    early = R.evaluate(ctx(area(growth=0.18), phase="burnin", iteration=60))[0]
    assert early["timing"] == "early" and early["main"] == "try" and "later_s" not in early   # Try it while cells form


def test_r5_hold_cells_closer_to_their_nuclei():
    (card,) = R.evaluate(ctx(area(growth=0.02, tiny=9, tiny_start=4)))
    assert (card["rule"], card["flag"], card["from"], card["to"], card["k"]) == ("R5", "prior-weight", 1.0, 2.0, 5)
    assert card["confidence"] == "rough"
    assert rules(ctx(area(growth=0.02, shrunk=60, cells=500))) == [("R5", ["A"])]       # 12 % lost half their area
    assert rules(ctx(area(growth=0.02, tiny=4, tiny_start=2, shrunk=40, cells=500))) == [("R6", ["A"])]
    assert R.evaluate(ctx(area(growth=0.02, tiny=9), live={"prior-weight": 700.0}))[0]["to"] == 1000.0


def test_r5_never_holds_a_run_started_from_cells_only_firmer():
    """Started from cells only, every transcript inside a starting cell holds its spot to that
    cell, so a firmer hold keeps each starting outline with any neighbouring RNA inside it: R5 offers it only where
    cells grow from nuclei. Nor a looser one from squeezed cells (a looser hold lets them shrink further). The start is
    how the run's cells started (start_of, the test's own rule); every other rule is the same for every start."""
    squeezed = area(growth=0.02, tiny=9, tiny_start=4)
    assert rules({**ctx(squeezed), "start": "cells"}) == []
    for start in ("paired", "nuclei", None):
        (card,) = R.evaluate({**ctx(squeezed), "start": start})
        assert (card["rule"], card["flag"], card["from"], card["to"]) == ("R5", "prior-weight", 1.0, 2.0), start
    for shape, rule in ((area(growth=0.18), "R4"), (area(growth=0.02), "R6"), (area(tag="already", growth=0.05), "R1"),
                        (area(growth=0.0, tiny=4, tiny_start=2, shrunk=40), "R6")):
        assert rules({**ctx(shape), "start": "cells"}) == rules(ctx(shape)) == [(rule, ["A"])]
    kinds = zip("ABCDEFGHIJKLMNOPQRSTUVWX", itertools.product(("rose", "already"), (-0.3, 0.0, 0.05), (2, 9), (0, 80)))
    shapes = [area(L, tag=tag, growth=g, tiny=t, shrunk=s) for L, (tag, g, t, s) in kinds]
    for weight in (0.07, 0.5, 1.0, 4.0):
        cards = R.evaluate({**ctx(*shapes, live={"prior-weight": weight}), "start": "cells"})
        assert cards and not any(card.get("flag") == "prior-weight" for card in cards), weight
    zarr = {"sample_kind": "zarr", "prior": {"cells": {"kind": "shapes", "key": "cellpose_boundaries"}, "nuclei": None}}
    assert R.start_of(zarr) == "cells" and R.start_of({}) == "paired"


def test_r6_rose_without_changing_size_cause_not_known():
    assert rules(ctx(area(growth=0.02))) == [("R6", ["A"])]
    assert rules(ctx(area(growth=-0.08))) == [("R6", ["A"])]
    assert rules(ctx(area(growth=-0.2))) == []                                  # shrank, nothing squeezed: no card
    assert rules(ctx(area(growth=0.02, churn=2.0))) == []                       # outlines still moving: not "unchanged"
    (card,) = R.evaluate(ctx(area(growth=0.02)))
    assert card["actions"] == ["inspect"] and card["main"] == "inspect" and card["confidence"] == "unknownCause"


def test_r7_settings_did_not_help_here():
    tried = [{"area": "A", "flag": "cell-compactness", "from": 0.04, "to": 0.02, "verdict": "no_change"}]
    assert rules(ctx(area(growth=0.18), tried=tried)) == [("R7", ["A"])]
    assert rules(ctx(area("B", growth=0.18), tried=tried)) == [("R4", ["B"])]  # another area
    assert rules(ctx(area(growth=0.18), tried=[{**tried[0], "verdict": "less"}])) == [("R4", ["A"])]
    assert rules(ctx(area(growth=0.18), tried=[{**tried[0], "verdict": None}])) == [("R4", ["A"])]   # no fair test
    (card,) = R.evaluate(ctx(area(growth=0.18), tried=[{**tried[0], "verdict": "more"}]))
    assert card["tried"] == {"flag": "cell-compactness", "from": 0.04, "to": 0.02} and "flag" not in card
    assert card["confidence"] == "seen" and card["actions"] == ["inspect"]      # what this run showed, nothing more


def test_u_comes_first_for_its_areas_and_for_the_whole_tissue():
    harm = {"rule": "U", "flag": "prior-weight", "from": 0.5, "to": 1.0, "changed": {"flag": "prior-weight", "from": 1.0,
                                                                                    "to": 0.5}, "areas": ["B"],
            "before_per100": 20.0, "after_per100": 24.0, "watch": 3}
    cards = R.evaluate(ctx(area("A", growth=0.18), area("B", growth=0.18), harm=harm))
    assert [(c["rule"], c["areas"]) for c in cards] == [("R4", ["A"]), ("U", ["B"])]
    assert cards[1]["actions"] == ["undo", "keep"] and cards[1]["confidence"] == "seen" and cards[1]["to"] == 1.0
    whole = R.evaluate(ctx(area("A", growth=0.18), harm={**harm, "areas": []}))
    assert [(c["rule"], c["areas"]) for c in whole] == [("U", []), ("R4", ["A"])]


def test_cards_for_the_same_knob_merge_across_areas_in_rank_order():
    cards = R.evaluate(ctx(area("C", growth=0.18), area("A", tag="already", growth=0.0), area("D", growth=0.25),
                           area("E", tag="similar")))
    assert [(c["rule"], c["areas"]) for c in cards] == [("R4", ["C", "D"]), ("R1", ["A"])]
    assert cards[0]["growth"] == 0.18                                            # the first area's own numbers


def test_r2_and_r3_are_checked_once_for_the_whole_tissue():
    loose, off = {"cell-compactness": 0.08}, {"no-diffusion": True}
    assert rules(ctx(live=loose, whole={"rose": True})) == [("R2", [])]
    assert rules(ctx(live=loose, whole={"growth": 0.12})) == [("R2", [])]
    assert rules(ctx(live=loose, whole={"growth": 0.05, "rose": False})) == []
    # Every area matched by an earlier rule: R3 still comes, for the whole tissue (an area is flagged).
    assert rules(ctx(area(tag="already", growth=0.0), live=off)) == [("R1", ["A"]), ("R3", [])]
    # Above its start by a change that counts: the depth-matched change against its threshold, never the raw share.
    more = {"matched": 1.4, "threshold": 1.0, "meaningful": True, "direction": "more"}
    assert rules(ctx(live=off, whole={"per100": 13.4, "start_per100": 12.0, "change": more})) == [("R3", [])]
    same = {"matched": 0.4, "threshold": 1.0, "meaningful": False, "direction": "same"}
    assert rules(ctx(live=off, whole={"per100": 14.0, "start_per100": 12.0, "change": same})) == []
    assert rules(ctx(live=off, whole={"per100": 14.0, "start_per100": 12.0})) == []          # nothing matched: no
    assert rules(ctx(area(growth=0.0), live={**loose, **off})) == [("R2", ["A"]), ("R3", [])]
    # While cells form (no areas of the run's own yet, nothing compared with the start): only R3, while the start line's
    # areas are on screen, with Try it first (it acts after burn-in); R2's "reach further" needs formed cells.
    forming = ctx(live={**loose, **off}, phase="burnin", iteration=40)
    assert rules(forming) == [] and rules({**forming, "flagged": True}) == [("R3", [])]
    (card,) = R.evaluate({**forming, "flagged": True})
    assert card["timing"] == "early" and card["main"] == "try" and "later_s" not in card


def test_nothing_on_the_never_suggested_list_is_ever_offered():
    never = [("prior-weight", 1.0, 0.5), ("no-diffusion", False, True), ("diffusion-sigma-far", 4.0, 8.0),
             ("diffusion-sigma-near", 1.0, 2.0), ("diffusion-probability", 0.2, 0.4), ("temperature", 1.0, 0.1),
             ("ab-nihlo-bubble-prob", 0.05, 0.2), ("voxel-layers", 4, 2), ("cell-compactness", 0.04, 0.08),
             ("morphology-steps-per-iter", 4000, 8192)]
    for flag, old, new in never:
        assert not R.allowed({"rule": "R4", "flag": flag, "from": old, "to": new}), flag
    for card in ({"rule": "R4", "flag": "cell-compactness", "from": 0.04, "to": 0.02},
                 {"rule": "R2", "flag": "cell-compactness", "from": 0.08, "to": 0.04},
                 {"rule": "R5", "flag": "prior-weight", "from": 1.0, "to": 2.0},
                 {"rule": "R3", "flag": "no-diffusion", "from": True, "to": False},
                 {"rule": "U", "flag": "prior-weight", "from": 0.5, "to": 1.0}):
        assert R.allowed(card), card
    # Whatever the areas and knobs, the rules only offer allowed changes.
    lives = [{"cell-compactness": c, "prior-weight": w, "no-diffusion": d}
             for c, w, d in itertools.product((0.02, 0.03, 0.04, 0.08), (0.5, 1.0, 4.0), (False, True))]
    shapes = [area(tag=tag, growth=g, tiny=t, churn=ch) for tag, g, t, ch in
              itertools.product(("rose", "already", "similar"), (-0.3, 0.0, 0.1, 0.3, None), (2, 9), (1.0, 2.0))]
    for live in lives:
        for card in R.evaluate(ctx(*shapes[:12], live=live, whole={"rose": True, "growth": 0.2})) + \
                R.evaluate(ctx(*shapes[12:], live=live)):
            assert R.allowed(card) and card["rule"] in R.CONFIDENCE, card


# ---- one card at a time --------------------------------------------------------------------------------------------
def card(rule, areas=("A",), to=None):
    return {"id": f"{rule}:x:{to}", "rule": rule, "areas": list(areas), "to": to}


def test_a_card_is_shown_once_it_led_for_2_results_over_2_seconds():
    keeper = R.CardKeeper()
    x = card("R4")
    assert keeper.choose([x], 0.0, "A") == (None, [])
    assert keeper.choose([x], 1.0, "A") == (None, [])         # 2 results, 1 s
    assert keeper.choose([x], 2.0, "A") == (x, [])
    y = card("R5", ("B",))
    shown, more = keeper.choose([x, y], 3.0, "AB")
    assert shown == x and more == [y]


def test_a_shown_card_gives_way_only_to_one_that_led_3_results_over_10_seconds():
    keeper = R.CardKeeper()
    x, y = card("R4"), card("R1", ("B",))
    for t in (0.0, 2.0):
        keeper.choose([x], t, "AB")
    for t in (3.0, 4.0, 5.0, 12.0):                           # y leads from 3.0: 4 results but 9 s
        assert keeper.choose([y, x], t, "AB")[0] == x
    assert keeper.choose([y, x], 13.0, "AB") == (y, [x])


def test_a_card_whose_rule_stopped_matching_goes_after_10_seconds_or_at_once_with_its_areas():
    keeper = R.CardKeeper()
    x = card("R4")
    for t in (0.0, 2.0):
        keeper.choose([x], t, "A")
    for t in (3.0, 5.0, 12.0):                                # A still drawn: the card holds through a wobble
        assert keeper.choose([], t, "A")[0] == x
    assert keeper.choose([], 13.0, "A") == (None, [])
    for t in (20.0, 22.0):
        keeper.choose([x], t, "A")
    assert keeper.choose([], 23.0, "B") == (None, [])         # A is gone: so is its card
    whole = card("R3", ())
    for t in (30.0, 32.0):
        keeper.choose([whole], t, "")
    assert keeper.choose([], 33.0, "")[0] == whole            # a whole-tissue card waits for its 10 s


def test_the_undo_card_comes_and_goes_at_once():
    keeper = R.CardKeeper()
    x, u = card("R4"), card("U", ())
    for t in (0.0, 2.0):
        keeper.choose([x], t, "A")
    assert keeper.choose([u, x], 2.5, "A") == (u, [x])
    assert keeper.choose([x], 3.0, "A") == (None, [])         # gone (undone, or answered "Keep") ...
    assert keeper.choose([x], 5.0, "A") == (x, [])            # ... and the next card enters afresh
    keeper.reset()
    assert keeper.choose([x], None, "A") == (None, [])        # frames without time: a second per result
    assert keeper.choose([x], None, "A") == (None, [])
    assert keeper.choose([x], None, "A") == (x, [])


def test_at_most_three_more_ideas():
    keeper = R.CardKeeper()
    cards = [card(f"R{k}", (chr(65 + k),)) for k in range(1, 7)]
    for t in (0.0, 2.0):
        shown, more = keeper.choose(cards, t, "ABCDEFG")
    assert shown == cards[0] and more == cards[1:4]


def test_previous_run_verdicts_are_absent():
    assert not hasattr(R, "previous_readout") and not hasattr(R, "previous_change")
    assert R.VERDICT_KEYS == ("change",)
    # What only the previous-run readout used went with it: the repeat-run spreads, its rounding and its run's share.
    from sopa_companion.engine.qc import QcManager
    assert not hasattr(R, "REPEAT") and not hasattr(R, "round1") and not hasattr(QcManager, "_matched_now")


def test_the_runs_pace_once_cells_formed():
    frames = [{"iter": i, "phase": "burnin" if i < 100 else "sampling", "t": 0.1 * i if i < 100 else 10 + 0.16 * (i - 100)}
              for i in range(160)]
    assert R.seconds_per_iteration(frames) == pytest.approx(0.16)
    assert R.seconds_per_iteration(frames[:100]) is None and R.seconds_per_iteration([{"iter": 1}]) is None
    assert R.timing_of("burnin") == "early" and R.timing_of("init") == "early" and R.timing_of("recorded") == "later"


# ---- trials of an area and their verdicts ------------------------------------------------------------------
def test_an_area_trial_gets_its_own_verdict_on_its_own_band_and_a_small_one_the_tissues():
    """An area with the minimum support (300 judged and 400 paired cells) is judged on its own band (same
    rules: 5 results, 20 iterations, twice in a row, until 300 iterations); a smaller one gets the whole tissue's
    verdict and the "too small" note. Its value after the change is its own matched change, never the tissue's."""
    areas = {"A": [28.0, 400, 0.0, 450], "B": [28.0, 150, 0.0, 170]}
    changes, start = change()
    watch = R.Watch(1, changes, start, history(areas=areas), reveal=120,
                    areas={"A": (np.arange(9), 400, 450), "B": (np.arange(9, 12), 150, 170)})
    for s in range(170, 200):                                       # the tissue as before; A and B dropped
        watch.update(entry(s, 10.0), {"A": area_value(19.0, 400, mc=-9.0, paired=450),
                                      "B": area_value(19.0, 150, mc=-9.0, paired=170)})
    own, small = watch.readout("A"), watch.readout("B")
    assert own["verdict"] == "less" and own["on"] == "area" and own["rows"][0] == {"kind": "area", "area": "A",
                                                                                  "expected": 28.0, "now": 19.0,
                                                                                  "change": -9.0}
    assert small["verdict"] is None and small["on"] == "whole" and "small_area" in small["notes"]
    assert watch.readout()["verdict"] is None                       # the tissue: no clear change
    for s in range(200, 472):
        watch.update(entry(s, 10.0), {"A": area_value(28.0, 400, mc=0.0, paired=450),
                                      "B": area_value(28.0, 150, mc=0.0, paired=170)})
    assert watch.readout("A")["verdict"] == "no_change" and watch.readout()["verdict"] == "no_change"
    # Cells shrank (the area's share fell with its transcripts) but its own matched change did not move: no verdict.
    shrunk = R.Watch(2, changes, start, history(areas=areas), reveal=120, areas={"A": (np.arange(9), 400, 450)})
    for s in range(170, 200):
        shrunk.update(entry(s, 10.0), {"A": area_value(22.0, 400, mc=0.0, paired=450)})
    assert shrunk.area_now["A"] == 28.0 and shrunk.readout("A")["verdict"] is None


def test_a_watch_of_several_changes_is_mixed_up_and_offers_no_undo():
    changes = [{"flag": "cell-compactness", "from": 0.04, "to": 0.02}, {"flag": "prior-weight", "from": 1.0, "to": 0.5}]
    watch = R.Watch(1, changes, {"seq": 170, "iter": 170, "phase": "sampling"}, history(), reveal=120)
    for s in range(170, 200):
        watch.update(entry(s, 12.5))
    assert watch.harm and watch.harm_card() is None                 # which change hurt cannot be told
    readout = watch.readout()
    assert readout["mixed_up"] and "mixed_up" in readout["notes"] and readout["verdict"] is None


def test_a_drift_change_gets_a_band_a_verdict_and_an_undo_card():
    """Proseg's count shows what drift moves (the RNA it gives back to the cell it came from), so a drift change is
    judged like any other: its band, "less" or "more", and the Undo card when it hurt, which puts back what the user
    changed (here drift on again: Undo switches it off, the user's own value, never a suggestion). No "no_verdict"
    note any more: no change is one this view cannot see."""
    assert R.UNSEEN_FLAGS == ()
    back_on, start = change(flag="no-diffusion", old=True, new=False)          # drift on again, mixing falls
    watch = R.Watch(1, back_on, start, history(), reveal=120)
    assert watch.band is not None and watch.bandable and not watch.unseen
    for s in range(170, 200):
        watch.update(entry(s, 8.0))
    readout = watch.readout()
    assert watch.verdict == "less" and readout["verdict"] == "less" and readout["band"] and readout["notes"] == []
    assert not watch.harm and watch.harm_card() is None
    hurt = R.Watch(2, back_on, start, history(), reveal=120)                   # drift on again, mixing rises
    for s in range(170, 200):
        hurt.update(entry(s, 13.0))
    card = hurt.harm_card()
    assert hurt.verdict == "more" and hurt.harm and card is not None
    assert (card["rule"], card["flag"], card["from"], card["to"]) == ("U", "no-diffusion", False, True)
    assert card["changed"] == {"flag": "no-diffusion", "from": True, "to": False} and R.allowed(card)
    (offered,) = R.evaluate(ctx(harm=card))
    assert (offered["rule"], offered["actions"], offered["confidence"]) == ("U", ["undo", "keep"], "seen")
    # A drift trial reads like any other: rows, band, verdict.
    assert {r["kind"] for r in hurt.readout()["rows"]} == {"whole", "depth"} and hurt.readout()["band"]


def test_no_undo_card_for_a_value_live_knobs_cannot_put_back():
    changes, start = change(flag="temperature", old=None, new=0.5)
    watch = R.Watch(1, changes, start, history(), reveal=120)
    for s in range(170, 200):
        watch.update(entry(s, 12.5))
    assert watch.harm and watch.harm_card() is None


def test_the_last_settling_stage_stops_a_watch_where_it_is():
    """The full schedule's hill-climb moves the line by itself (and the band was calibrated without it): a watch stops
    there with its verdict as it was, gives no harm, and a change made in it gets no band."""
    changes, start = change()
    watch = R.Watch(1, changes, start, history(), reveal=120)
    for s in range(170, 195):
        watch.update(entry(s, 8.0))
    assert watch.verdict == "less"
    watch.update(entry(195, 12.5, phase="hillclimb"))
    for s in range(196, 230):
        watch.update(entry(s, 12.5, phase="hillclimb"))
    readout = watch.readout()
    assert watch.final and watch.verdict == "less" and not watch.harm and "last_stage" in readout["notes"]
    late = R.Watch(2, changes, {"seq": 400, "iter": 400, "phase": "hillclimb"},
                   history(start=350) + [entry(s, 9.0, phase="hillclimb") for s in range(390, 400)], reveal=120)
    assert late.band is None and "last_stage" in late.readout()["notes"] and not late.no_history
    plain = R.Watch(3, changes, start, history()[:40] + [entry(s, 30.0, phase="hillclimb") for s in range(160, 170)],
                    reveal=120)
    assert plain.band is None or plain.band["centre"] == 10.0       # hill-climb results are never in a band's history


def test_a_watch_is_judging_only_while_a_verdict_can_still_come():
    """Watch.judging (QcManager.watching: the scheduler scores such a run's newest frames first): a change with a band
    until its horizon, the last settling stage, another change or its Undo card answered; never a change without a
    band (made while cells form, in the last settling stage, too soon after cells formed, or without the like-for-like
    change), however long it is watched. An area's own band is enough."""
    changes, start = change()
    watch = R.Watch(1, changes, start, history(), reveal=120)
    for s in range(170, 471):
        watch.update(entry(s, 10.3))
    assert watch.judging and not watch.final
    watch.update(entry(471, 10.3))                                  # past the horizon
    assert watch.final and not watch.judging
    frozen, answered, climbing = (R.Watch(k, changes, start, history(), reveal=120) for k in (2, 3, 4))
    frozen.freeze(mixed_up=True)
    answered.dismissed = True
    climbing.update(entry(171, 10.0, phase="hillclimb"))
    assert frozen.band and answered.band and climbing.band
    assert not (frozen.judging or answered.judging or climbing.judging)
    never = [R.Watch(5, changes, {"seq": 60, "iter": 60, "phase": "burnin"}, history(start=10), reveal=120),
             R.Watch(6, changes, {"seq": 400, "iter": 400, "phase": "hillclimb"}, history(start=350), reveal=120),
             R.Watch(7, changes, {"seq": 128, "iter": 128, "phase": "sampling"}, history(start=100, n=28), reveal=120),
             R.Watch(8, changes, start, [{**e, "mc": None} for e in history()], reveal=120)]
    for w in never:
        for s in range(w.start_seq, w.start_seq + 60):
            w.update(entry(s, 5.0))
    assert all(w.band is None and not w.final and not w.judging for w in never)
    own = {"A": [12.0, 400, -1.0, 500]}                             # share, judged, matched change, paired
    area = R.Watch(9, changes, start, [{**e, "mc": None, "areas": own} for e in history()], reveal=120,
                   areas={"A": (np.arange(9), 400, 500)})
    assert area.band is None and set(area.area_band) == {"A"} and area.judging


def test_r5_at_the_prior_weight_bound_offers_nothing_and_no_card_changes_nothing():
    squeezed = area(growth=0.0, tiny=10, tiny_start=0)
    assert rules(ctx(squeezed, live={"prior-weight": 1000.0})) == []
    (card,) = R.evaluate(ctx(squeezed, live={"prior-weight": 700.0}))
    assert (card["rule"], card["from"], card["to"]) == ("R5", 700.0, 1000.0)
    for flag, old in (("prior-weight", 2.0), ("cell-compactness", 0.04), ("temperature", 0.5)):
        assert not R.allowed({"rule": "R9", "flag": flag, "from": old, "to": old})
    assert not R.allowed({"rule": "R4", "flag": "cell-compactness", "from": 0.03999999910593033, "to": 0.04})


def test_r2_says_how_the_area_changed_and_cards_are_known_by_their_change():
    loose = {"cell-compactness": 0.08}
    (grown,) = R.evaluate(ctx(area(tag="similar", growth=0.2), live=loose))
    (rose,) = R.evaluate(ctx(area(growth=0.0), live=loose))
    assert (grown["growth"], grown["rose"]) == (0.2, False) and (rose["growth"], rose["rose"]) == (0.0, True)
    whole = R.evaluate(ctx(live=loose, whole={"growth": 0.15, "rose": False, "per100": 11.0, "start_per100": 10.0}))
    assert [(c["rule"], c["areas"], c["growth"], c["rose"]) for c in whole] == [("R2", [], 0.15, False)]
    assert R.card_id({"rule": "R2", "areas": ["A"], "flag": "cell-compactness", "to": 0.04}) == \
        R.card_id({"rule": "R2", "areas": ["A", "C"], "flag": "cell-compactness", "to": 0.04})
    harm = {"rule": "U", "flag": "prior-weight", "to": 1.0}
    assert R.card_id({**harm, "watch": 3}) != R.card_id({**harm, "watch": 4})   # a later change: another card


def test_a_value_asked_for_keeps_its_digits_and_a_32_bit_one_its_shortest_decimal():
    assert R.clean(123.4567) == 123.4567 and R.clean(float(np.float32(123.4567))) == 123.4567
    assert R.clean(0.019999999552965164) == 0.02 and R.clean(0.08) == 0.08 and R.clean(0.5) == 0.5
