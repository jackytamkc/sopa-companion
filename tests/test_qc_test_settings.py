"""The six more of proseg's own settings a test of a setting takes (starting grid, background gaps, drift, short drift
distance, crowding levels, depth layers) beside compactness, the hold and far drift: how the POST takes each kind of
value, what the run's value now is, which values a run offers, what is refused and in which words, the value its B runs
are given, the pool reply's `settings`, and the time a test of many depth layers may take."""
from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from sopa_companion.engine import proseg_options, qc_test as T
from sopa_companion.engine.qc import QcError


def run(options=None, live=None, info=None, frames=None):
    return SimpleNamespace(meta={"options": dict(options or {}), "live_values": dict(live or {})},
                           info=info, frames=list(frames or []), lock=threading.Lock())


INFO = {"layers": 4, "burnin_voxel_size": 2.0, "voxel_size": 1.0, "quad_size": 150.0, "z": {"thickness_um": 6.4}}


def body(flag, value):
    return T.QcTests._body({"cell": 3, "flag": flag, "value": value})


def test_the_nine_settings_in_the_forms_order_each_with_its_kind_and_plain_name():
    assert T.FLAGS == ("cell-compactness", "prior-weight", "diffusion-probability", "burnin-voxel-size",
                       "ab-nihlo-bubble-prob", "no-diffusion", "diffusion-sigma-near", "density-bins", "voxel-layers")
    assert set(T.NAMES) == set(T.KINDS) == set(T.TRIED) == set(T.FLAGS)
    assert [f for f in T.FLAGS if T.TRIED[f]] == ["cell-compactness", "prior-weight"]
    assert {f: k for f, k in T.KINDS.items() if k != "number"} == {"burnin-voxel-size": "choice", "no-diffusion": "switch",
                                                                   "density-bins": "whole", "voxel-layers": "whole"}
    flags = {o["flag"] for o in proseg_options.OPTIONS}
    assert set(T.FLAGS) <= flags, "every setting is one of proseg's own options"
    for name in T.NAMES.values():
        assert not any(word in name for word in ("voxel", "prior", "diffusion")), name


def test_the_post_takes_each_kind_of_value_in_its_own_form():
    assert body("no-diffusion", 1) == (3, "no-diffusion", 1) and body("no-diffusion", 0.0) == (3, "no-diffusion", 0)
    assert body("density-bins", 8.0) == (3, "density-bins", 8) and isinstance(body("density-bins", 8.0)[2], int)
    assert body("voxel-layers", 16) == (3, "voxel-layers", 16)
    assert body("burnin-voxel-size", 1.0) == (3, "burnin-voxel-size", 1.0)
    assert body("ab-nihlo-bubble-prob", 0.3) == (3, "ab-nihlo-bubble-prob", 0.3)
    assert body("diffusion-sigma-near", 2) == (3, "diffusion-sigma-near", 2)
    for flag, value, words in [
        ("no-diffusion", 2, "Choose 1 (drift off) or 0 (drift on)"), ("no-diffusion", True, "value must be a number"),
        ("density-bins", 8.5, "Choose a whole number from 1 to 16"), ("density-bins", 17, "Choose a whole number from 1 to 16"),
        ("voxel-layers", 0, "Choose a whole number from 1 to 256"), ("burnin-voxel-size", 0, "Choose a starting grid in µm"),
        ("ab-nihlo-bubble-prob", 1.5, "Choose a value from 0 to 1"), ("diffusion-sigma-near", 0, "Choose a value from 0.01 to 100"),
        ("temperature", 0.5, "A test changes one of the settings the test form offers")]:
        with pytest.raises(QcError) as caught:
            body(flag, value)
        assert (str(caught.value), caught.value.status) == (words, 422), (flag, value)


def test_the_runs_value_now_for_a_switch_and_for_settings_proseg_reads_at_its_start():
    assert T.current_of(run(), "no-diffusion") == 0
    assert T.current_of(run({"no-diffusion": True}), "no-diffusion") == 1
    assert T.current_of(run({"no-diffusion": True}, {"no-diffusion": False}), "no-diffusion") == 0
    assert T.current_of(run(frames=[{"live": {"no-diffusion": True}}]), "no-diffusion") == 1
    assert T.current_of(run(info=INFO), "burnin-voxel-size") == 2.0
    assert T.current_of(run({"burnin-voxel-size": 4.0}), "burnin-voxel-size") == 4.0, "asked, before proseg's record"
    assert T.current_of(run({"voxel-layers": 8}, info={**INFO, "layers": 1}), "voxel-layers") == 1, "proseg's own record"
    assert T.current_of(run(), "density-bins") == 5 and T.current_of(run({"density-bins": 8}), "density-bins") == 8
    assert T.current_of(run(live={"ab-nihlo-bubble-prob": 0.2}), "ab-nihlo-bubble-prob") == 0.2


def test_the_starting_grids_a_run_offers_follow_proseg_s_float32_rule_up_to_4_um():
    assert T.grid_choices(run(info=INFO)) == [1.0, 3.0, 4.0]
    assert T.grid_choices(run({"burnin-voxel-size": 1.0}, info={**INFO, "burnin_voxel_size": 1.0})) == [2.0, 3.0, 4.0]
    assert T.grid_choices(run(info={**INFO, "voxel_size": 0.5})) == [0.5, 1.0, 1.5]
    assert T.grid_choices(run(info={**INFO, "voxel_size": 2.0, "burnin_voxel_size": 2.0})) == [4.0]
    # 0.9 / 0.3 is 2.9999998 in float32: proseg panics, so 0.9 is never offered
    assert 0.9 not in T.grid_choices(run(info={**INFO, "voxel_size": 0.3, "burnin_voxel_size": 0.6}))


def test_depth_layers_up_to_what_the_area_holds_and_their_time():
    assert T.layers_max(run(info=INFO)) == 94, "a 500 µm square at 1 µm"
    assert T.layers_max(run(info=INFO), [0, 0, 300, 300]) == 197
    assert T.layers_max(run({"voxel-size": 0.5, "burnin-voxel-size": 1.0}, info=INFO)) == 23
    assert T.layer_factor(4) == T.layer_factor(2) == 1.0
    assert abs(T.layer_factor(94) - 3.034) < 1e-9


def test_a_value_is_refused_only_where_that_setting_cannot_be_tested():
    def refused(r, flag, value, box=None):
        with pytest.raises(QcError) as caught:
            T.QcTests._refuse_value(r, flag, value, T.current_of(r, flag), box)
        return str(caught.value), caught.value.status
    dry = run({"no-diffusion": True}, info=INFO)
    assert refused(dry, "diffusion-sigma-near", 2.0) == ("Short drift distance does nothing while drift is switched off in this run", 422)
    assert refused(dry, "diffusion-probability", 0.3) == ("Far drift does nothing while drift is switched off in this run", 422)
    assert refused(dry, "no-diffusion", 1) == ("Drift is already off in this run: choose on to test it", 422)
    T.QcTests._refuse_value(dry, "no-diffusion", 0, 1)                     # drift on, from a run with it off
    assert refused(run(info=INFO), "no-diffusion", 0) == ("Drift is already on in this run: choose off to test it", 422)
    flat = run(info={**INFO, "layers": 1, "z": {"thickness_um": 0.0}})
    assert refused(flat, "voxel-layers", 8) == (T.FLAT_WORDS, 422)
    assert refused(run({"ignore-z-coord": True}, info=INFO), "voxel-layers", 8) == (T.FLAT_WORDS, 422)
    assert refused(run(info=INFO), "burnin-voxel-size", 1.5) == ("Choose 1, 3 or 4 µm for this run's starting grid", 422)
    assert refused(run(info=INFO), "voxel-layers", 120) == ("This value needs more memory than a test run may take here", 422)
    for flag, value in [("burnin-voxel-size", 1.0), ("voxel-layers", 8), ("density-bins", 8), ("ab-nihlo-bubble-prob", 0.3),
                        ("diffusion-sigma-near", 2.0), ("no-diffusion", 1)]:
        T.QcTests._refuse_value(run(info=INFO), flag, value, T.current_of(run(info=INFO), flag))   # taken


def test_the_b_runs_get_each_value_as_proseg_options_takes_it():
    assert T.proseg_value("no-diffusion", 1) is True and T.proseg_value("no-diffusion", 0) is False
    assert T.proseg_value("density-bins", 8) == 8 and isinstance(T.proseg_value("voxel-layers", 8.0), int)
    for flag, value in [("no-diffusion", 1), ("density-bins", 8.0), ("voxel-layers", 8.0), ("burnin-voxel-size", 1.0)]:
        args = proseg_options.to_args({"nthreads": 8, flag: T.proseg_value(flag, value)})
        assert f"--{flag}" in args


def test_the_pool_reply_says_what_the_form_needs_of_every_setting():
    settings = T.settings_tested(run({"no-diffusion": True}, info={**INFO, "layers": 1, "z": {"thickness_um": 0.0}}))
    assert list(settings) == list(T.FLAGS)
    assert settings["burnin-voxel-size"] == {"now": 2.0, "refusal": None, "voxel": 1.0, "choices": [1.0, 3.0, 4.0]}
    assert settings["no-diffusion"] == {"now": 1, "refusal": None}
    assert settings["voxel-layers"]["max"] == 94 and settings["voxel-layers"]["now"] == 1
    assert settings["voxel-layers"]["refusal"] == {"code": "flat", "words": T.FLAT_WORDS}
    assert settings["diffusion-sigma-near"]["refusal"]["code"] == "drift_off"
    assert settings["diffusion-probability"]["min"] == 0.1 and settings["diffusion-probability"]["max"] == 0.3
    assert (settings["density-bins"]["min"], settings["density-bins"]["max"]) == (1, 16)


def test_a_test_of_many_depth_layers_has_more_time():
    test = T.Test("p", None, "voxel-layers", 32, 4, lambda: 0.0, runs_per_side=4, row=("cells", "square"))
    test.set_factor(run(info=INFO))
    assert abs(test.factor - (1 + T.LAYER_COST * 28)) < 1e-9
    other = T.Test("p", None, "density-bins", 8, 5, lambda: 0.0, runs_per_side=4, row=("cells", "square"))
    other.set_factor(run(info=INFO))
    assert other.factor == 1.0


def test_the_engine_and_the_page_take_the_same_settings_in_the_same_order():
    twins = json.loads((Path(__file__).resolve().parents[1] / "frontend/test/fixtures/qc-test-settings.json").read_text())
    assert twins["flags"] == list(T.FLAGS) and twins["tried"] == T.TRIED and twins["kinds"] == T.KINDS
    assert twins["ranges"] == {k: list(v) for k, v in T.TEST_RANGES.items()} and twins["layer_cost"] == T.LAYER_COST
