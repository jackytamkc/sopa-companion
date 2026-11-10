"""The calibration on proseg's own count (2026-10-31), plugged in: every value the code holds is the measurement's own
(frontend/test/fixtures/qc-calibration.json, a copy of the values in its constants file, each measured or kept there
with its reason): the mixing check's table per mode and which modes were measured (engine/qc_constants.py: regions and
tiled regions; whole samples keep the outline count's values and their caveat), the settings test's constants, ids and
rows measured on proseg's count (engine/qc_rule.py), the cell against itself at the start (the worker's band, and the
page's where a reply has none, as engine/qc_scoring.py ports it), the UMAP's and Compare's (analysis/worker.py, and
the shipped settings' min_counts), the pink dots' thresholds (engine/qc_scoring.py), the where
words' (engine/deeper_where.py) and drift's confidence (engine/qc_rules.py). The page's twins are held to the same
file by frontend/test/qc-calibration.test.mjs."""
from __future__ import annotations

import json
import tomllib

from sopa_companion.analysis import worker as W
from sopa_companion.engine import deeper_where as DW
from sopa_companion.engine import qc_constants as K
from sopa_companion.engine import qc_pool as P
from sopa_companion.engine import qc_rule as Q
from sopa_companion.engine import qc_rules as R
from sopa_companion.engine import qc_scoring as S
from sopa_companion.engine.config import ROOT

CAL = json.loads((ROOT / "frontend/test/fixtures/qc-calibration.json").read_text(encoding="utf-8"))


def test_the_checks_constants_are_the_calibrations():
    """The id, the modes re-measured on proseg's count and every value of every mode, the trial's and the areas' too;
    each summary names them (block), and says measured false for a whole sample alone (the page's caveat)."""
    cal = CAL["qc_constants"]
    assert K.CONSTANTS_ID == cal["CONSTANTS_ID"] and K.MEASURED == cal["MEASURED"]
    assert K.MODES == tuple(cal["TABLE"]) and K.TABLE == cal["TABLE"]
    assert {mode: list(names) for mode, names in K.KEPT.items()} == cal["KEPT"]
    for mode in K.MODES:
        assert {key: K.value(mode, key) for key in cal["TABLE"][mode]} == cal["TABLE"][mode], mode
        assert K.block(mode) == {"mode": mode, "id": cal["CONSTANTS_ID"], "measured": cal["MEASURED"][mode],
                                 "kept": cal["KEPT"][mode]}
    assert [mode for mode in K.MODES if not K.block(mode)["measured"]] == ["whole"]
    assert [mode for mode in K.MODES if not K.block(mode)["kept"]] == ["region"]


def test_the_settings_tests_constants_are_the_calibrations():
    """The id (the set measured on the outlines keeps its own among the outline ids), the power's drop, every start's
    row and situation (offered, its refusal, whether it was measured again on proseg's count, gate, taus and the rates
    the page's tooltip reads) and what the measurement held fixed besides: likely's least w, as the page's likely rule
    reads it, and the controls' bands of w, as the pool draws them."""
    cal = CAL["settings_test"]
    assert (Q.CONSTANTS_ID, list(Q.OUTLINE_IDS), Q.DROP) == (cal["CONSTANTS_ID"], cal["OUTLINE_IDS"], cal["DROP"])
    assert Q.CONSTANTS == cal["CONSTANTS"]
    for start, row in cal["CONSTANTS"].items():
        for situation, sub in row["situations"].items():
            assert Q.measured_on_proseg(start, situation) is sub["measured"], (start, situation)
            found = Q.constants_for(start, situation)
            assert (found["id"], found["offered"], found["tau"], found["rates"], found["short_rates"]) == \
                (cal["CONSTANTS_ID"], sub["offered"], sub["tau"][row["rule"]], sub["rates"], sub["short_rates"]), \
                (start, situation)
    short = cal["CONSTANTS"]["cells"]["situations"]["as_is"]["short_rates"]
    assert (short["tests"], short["areas"], short["drop"]) == (8, 4, cal["DROP"])
    held = cal["MEASURED"]
    assert (Q.MEASURED["likely_w"], Q.MEASURED["w_bands"]) == (held["likely_w"], tuple(held["w_bands"]))
    assert (P.likely_rule()[1], P.W_BANDS) == (held["likely_w"], tuple(held["w_bands"]))


def test_go_deepers_the_workers_and_the_pink_dots_constants_are_the_calibrations():
    """The cell against itself at the start (its band and the hold of a word shown), the UMAP's least transcripts per
    cell (the worker's default and the shipped settings' [analysis] min_counts, which every run starts from), Compare's
    like-for-like depth, the pink dots' thresholds (engine/qc_scoring.py, the page's port), the where words' (in
    proseg's count) and drift's confidence."""
    worker = CAL["worker"]
    assert (W.DEEP_START_BAND, W.DEEP_START_KEEP, W.DEFAULTS["min_counts"], W.LIKE_DEPTH) == \
        (worker["DEEP_START_BAND"], worker["DEEP_START_KEEP"], worker["DEFAULTS.min_counts"], worker["LIKE_DEPTH"])
    shipped = tomllib.loads((ROOT / "config.example.toml").read_text(encoding="utf-8"))
    assert shipped["analysis"]["min_counts"] == worker["DEFAULTS.min_counts"]
    assert {name: getattr(S, name) for name in CAL["pink_dots"]} == CAL["pink_dots"]
    assert {name: getattr(DW, name) for name in CAL["where_words"]} == CAL["where_words"]
    assert R.CONFIDENCE["R3"] == CAL["rule_R3"]["CONFIDENCE"]


def test_the_page_scorings_port_reads_a_start_change_without_a_band_as_the_worker_sends_it():
    """engine/qc_scoring.py normalise, the page's normalise in Python: a start_change without a band (the page's twin
    START_BAND, qc-deeper-copy.mjs) reads the worker's measured DEEP_START_BAND, never an older one; a reply's own band
    wins."""
    band = CAL["worker"]["DEEP_START_BAND"]
    assert S.START_BAND == band == W.DEEP_START_BAND

    def reply(change):
        return {"v": 1, "state": "ready", "seq": 12, "code": 116, "start_change": change}
    assert S.normalise(reply({"points": 4}))["start_change"]["band"] == band
    assert S.normalise(reply({"points": 4, "band": 3}))["start_change"]["band"] == 3
