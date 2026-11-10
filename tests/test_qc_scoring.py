"""engine/qc_scoring.py, the page's Go deeper scoring in Python (the engine fixes a test's suspects with it), against
the page itself: every case of the shared golden fixture (frontend/test/fixtures/qc-scoring.json: inputs from
tests/make_qc_scoring_inputs.py, `expect` made by frontend/test/make-qc-scoring.mjs from frontend/src/qc-deeper.js and
recomputed by frontend/test/qc-scoring.test.mjs) reproduced exactly, P and pi compared as float32 bits, the sources'
support as doubles; JavaScript's Math.round and toFixed at their edges and against exact arithmetic; each function's
own rules (version 3 replies only, each point's share w, P = w x pi on any owner, the likely rule, the support that
tells a named source with none of its glow near it); that odd inputs never make it fail; and that a reply as large as
the engine sends is scored in well under a second."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import pytest

from sopa_companion.engine import qc_scoring as S

APP = Path(__file__).resolve().parents[1]
FIXTURE = APP / "frontend/test/fixtures/qc-scoring.json"
CASES = json.loads(FIXTURE.read_text())
BY_NAME = {c["name"]: c for c in CASES}


def same(a, b, path="") -> str | None:
    """None when a and b are the same JSON value, else where they differ: numbers as JSON.parse reads them (doubles: node
    writes 1.2345678901234568e20 as 123456789012345680000), never a boolean for a number, keys and lengths equal."""
    if isinstance(a, bool) or isinstance(b, bool):
        return None if isinstance(a, bool) and isinstance(b, bool) and a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return None if float(a) == float(b) else f"{path}: {a!r} != {b!r}"
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} {a!r} != {type(b).__name__} {b!r}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a)} != {sorted(b)}"
        return next((found for k in a if (found := same(a[k], b[k], f"{path}.{k}"))), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items != {len(b)}"
        return next((found for i, (x, y) in enumerate(zip(a, b, strict=True))
                     if (found := same(x, y, f"{path}[{i}]"))), None)
    return None if a == b else f"{path}: {a!r} != {b!r}"


def bits(p: float) -> int:
    return int(np.float32(p).view(np.uint32))


def kept_of(result, req, raw):
    reply = S.normalise_where(raw, req)
    return None if reply is None else S.keep_suspects(reply, S.suspect_scores(reply, S.normalise(result))[0])


# ---- the golden fixture, both sides ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BY_NAME))
def test_every_golden_case_is_the_pages_own_answer(name):
    case = BY_NAME[name]
    want = case["expect"]
    params = S.page_params(case["result"], case["cell"])
    assert same(params, want["params"], "params") is None, same(params, want["params"], "params")
    scores = S.page_scores(case["result"], params["req"], case["reply"])
    assert same(scores, want["scores"], "scores") is None, same(scores, want["scores"], "scores")
    if scores is not None:
        # P and pi are the float32 the page's Float32Arrays hold (written as their doubles): equal bits, glowing decided
        # on P, likely on pi with the point's share.
        for key in ("P", "pi"):
            assert [bits(p[key]) for p in scores["points"]] == [bits(p[key]) for p in want["scores"]["points"]]
            assert all(float(np.float32(p[key])) == p[key] for p in scores["points"])
        assert all(p["P"] >= S.GLOW_MIN and p["w"] > 0 for p in scores["points"])
        assert [p["pi"] >= S.LIKELY_MIN and p["w"] >= S.LIKELY_W for p in scores["points"]] == \
            [np.float32(p["pi"]) >= np.float32(S.LIKELY_MIN) and p["w"] >= S.LIKELY_W for p in want["scores"]["points"]]
    kept = kept_of(case["result"], params["req"], case["reply"])
    assert same(kept, want["kept"], "kept") is None, same(kept, want["kept"], "kept")


def test_the_golden_holds_every_edge_it_is_for():
    """The fixture keeps the cases the port turns on (a regenerated one may not lose them), and the page answers them
    as described here."""
    expect = {name: case["expect"] for name, case in BY_NAME.items()}
    assert len(CASES) >= 100 and FIXTURE.stat().st_size < 400_000

    def params(name):
        return expect[name]["params"]

    def points(name):
        return expect[name]["scores"]["points"]

    # Real answers and replies: the numpy worker's on a synthetic run (every state), the where engine's on it (exact
    # shares, transcripts moved elsewhere: some glow outside the cell's outline) and on hand-made frames (every word,
    # near flags, approximate and exact shares), and on the strip case.
    world = [n for n in BY_NAME if n.startswith("world_")]
    assert {params(n)["state"] for n in world} == {"ready", "too_few", "empty"}
    assert sum(expect[n]["scores"] is not None and len(points(n)) > 20 for n in world) >= 3
    assert {BY_NAME[n]["reply"]["held"] for n in world if BY_NAME[n]["reply"]} == {"exact"}
    assert any(p["owner"] != 1 for n in world if expect[n]["scores"] for p in points(n)), "glow outside its outline"
    assert len({p["w"] for n in world if expect[n]["scores"] for p in points(n)}) > 5, "shares of every size"
    detail = BY_NAME["hand_made_detail"]["reply"]
    assert detail["held"] == "approx" and BY_NAME["hand_made_whole_sample"]["reply"]["held"] == "exact"
    assert {s["word"] for s in detail["gene_stats"] if s["role"] == "from"} == {"edge", "spread", "layer", "few"}
    assert {p["P"] for p in points("hand_made_detail") if p["gene"] == "LAYER"} == {0.5}     # P exactly 0.5: likely
    assert {(p["gene"], p["near"]) for p in points("hand_made_detail")} >= {("EDGE", 1), ("SPREAD", 0)}
    assert {p["w"] for p in points("hand_made_detail") if p["gene"] == "EDGE"} == {0.75}
    assert {(p["gene"], p["owner"]) for p in points("hand_made_whole_sample")} >= {("EDGE", 1), ("EDGE", 2)}
    # The strip case (A2): 13 glowing, likely GZF along its far edge, near none of cell-f (no support);
    # beside it, all of their shares.
    far, beside = points("a2_strip_far"), points("a2_strip_beside")
    assert len(far) == len(beside) == 13 and all(p["gene"] == "GZF" and p["w"] == 0.875 for p in far + beside)
    assert expect["a2_strip_far"]["scores"]["support"] == []
    assert expect["a2_strip_beside"]["scores"]["support"] == [[0, 13 * 0.875]]
    # Chips: ties by name in UTF-16 order (an astral name before U+FFE0), the floor (with its 1e-6), the first six asked,
    # own genes neither among them nor more than three; text escaped in the key as JSON.stringify does.
    assert params("chips_tied_by_name_utf16")["chips"] == ["GENE1", "GENE2", "G" + chr(0x10000), "G" + chr(0xFFE0)]
    assert params("chips_floor")["chips"] == ["EPSILON", "HALF"]
    chips = params("chips_more_than_six_own_overlap")
    assert len(chips["chips"]) == 8 and chips["req"]["params"]["genes"] == chips["chips"][:6]
    assert chips["req"]["params"]["own"] == ["OWNA", "OWNB", "OWNC"]
    assert '\\"' in params("chips_names_escaped_in_the_key")["req"]["key"]
    assert "\\ud800" in params("chips_names_escaped_in_the_key")["req"]["key"]
    # Rows: at most five, ties in neighbour order, 0.01 exactly in, the same kind from `similar`, the cell as read.
    assert [r["cell"] for r in params("rows_more_than_five_and_ties")["rows"]] == [22, 20, 21, 23, 24]
    assert [r["cell"] for r in params("rows_share_exactly_one_percent")["rows"]] == [12, 10]
    assert [r["same"] for r in params("rows_same_kind_from_similar")["rows"]] == [True, False, True, False, False]
    assert [r["key"] for r in params("rows_cell_numbers_as_text_null_missing")["rows"]] == ["71", 1, "n2", 4.5, "51"]
    assert params("rows_none_at_code_9")["none"] and params("rows_none_at_code_9")["req"] is None
    assert not params("rows_shown_at_code_12")["none"] and params("rows_shown_at_code_12")["req"] is not None
    assert params("rows_seq_start")["req"]["key"].startswith('["start",')
    key = params("rows_numbers_written_as_javascript_does")["req"]["key"]
    assert key.startswith("[1e+21,5,[1e+21,1.0000000005838672e-7,123456789012345680000,")
    # The states, and answers the page refuses.
    assert [params(f"state_{s}")["state"] for s in ("too_few", "empty", "not_checked", "v2_refused")] == \
        ["too_few", "empty", "not_checked", None]
    # P: float32 before the cut-offs (a double under 0.2 glows, 0.49999999 is likely), toFixed's ties half up.
    fixed = {p["gene"]: p for p in points("scores_float32_and_to_fixed_ties")}
    assert "F_0.199999999" in fixed and "F_0.19999998" not in fixed and fixed["F_0.49999999"]["P"] == 0.5
    kept = {s["gene"]: s["p"] for s in expect["scores_float32_and_to_fixed_ties"]["kept"]}
    assert (kept["T_0.3125"], kept["T_0.5625"]) == (0.313, 0.563)
    # P = w x pi on any owner: a point inside the cell's outline that proseg does not count is no suspect, one it moved
    # in from a source's outline or from none is; likely needs half of it for the cell.
    each = points("scores_each_branch")
    assert {p["owner"] for p in each} == {0, 1, 2} and all(p["w"] > 0 for p in each)
    assert {(p["w"], p["P"]) for p in each if p["gene"] == "UNPLACED"} == \
        {(w, float(np.float32(w * 0.5))) for w in (1.0, 0.75, 0.5, 0.25) if w * 0.5 >= 0.2}
    rule = {(p["gene"], p["w"]): p for p in points("scores_w_and_the_likely_rule")}
    assert sorted(w for g, w in rule if g == "GAVE") == [0.25, 0.4999, 0.5, 1.0]
    assert [(g, w) for (g, w), p in rule.items() if p["pi"] >= S.LIKELY_MIN and w >= S.LIKELY_W] == \
        [("GAVE", 1.0), ("GAVE", 0.5), ("HALF", 1.0)]
    assert rule[("HALF", 0.4)]["P"] == float(np.float32(0.2))
    assert expect["scores_support_by_source"]["scores"]["support"] == [[0, 1.25]]
    # Places: ranked by depth, ties by point; none where a depth is missing; x.xx5 halves up (0.125 and 0.145 share a
    # place), -0.375 and -0.39 do not; a place past float32's range.
    ranked = {p["i"]: (p.get("rank"), p.get("of")) for p in points("places_ranked_by_depth")}
    assert ranked[0] == (1, 4) and ranked[1] == (2, 4) and ranked[4] == ranked[5] == (None, None)
    assert ranked[6] == (0, 2) and ranked[8] == ranked[9] == (None, None)
    assert any(p.get("of", 0) > 9 for p in points("places_past_float32_and_double_steps"))
    # More than 400 glowing points: 400 kept, the likeliest; gene indices past 255 wrap; long names are not sent.
    assert len(points("kept_more_than_400")) > 400 and len(expect["kept_more_than_400"]["kept"]) == 400
    assert [s["gene"] for s in expect["kept_gene_index_wraps_and_long_names"]["kept"]] == ["L" * 64, "W4", "W4"]
    # Every refusal: no scores, nothing kept; the odd replies the page takes.
    refused = [n for n in BY_NAME if n.startswith("refused_")]
    assert len(refused) >= 40 and all(expect[n]["scores"] is None and expect[n]["kept"] is None for n in refused)
    taken = [n for n in BY_NAME if n.startswith("taken_")]
    assert len(taken) >= 10 and all(expect[n]["scores"] is not None for n in taken)
    # Whole numbers past 2^53 add up as doubles: labels 2^53 and 2^53 + 1 are one row (its first place, the last
    # neighbour, asked for as 2^53 - 1), the cell at 2^53 asks for 2^53 - 1, own_near's + 1 and + 2 are lost.
    big = 2 ** 53
    assert [(r["label"], r["cell"]) for r in params("rows_labels_one_past_2_53")["rows"]] == [(big, big), (8, 7)]
    assert params("rows_labels_one_past_2_53")["req"]["params"]["sources"] == [big - 1, 7]
    assert params("rows_cell_at_2_53")["req"]["params"]["cell"] == big - 1
    assert [g["fr"] for g in expect["scores_own_near_past_2_53"]["scores"]["genes"]] == [big / 3, 4 / big]


# ---- Math.round and toFixed ------------------------------------------------------------------------------------
# Values checked against node's Math.round and Number.prototype.toFixed.
ROUNDS = [(0.5, 1), (-0.5, 0), (1.5, 2), (2.5, 3), (-2.5, -2), (-1.5, -1), (0.49999999999999994, 0),
          (-0.49999999999999994, 0), (4503599627370497.0, 4503599627370497), (2.0 ** 51 + 0.5, 2 ** 51 + 1),
          (-(2.0 ** 51) - 0.5, -(2 ** 51)), (-0.3, 0), (0.7, 1), (-0.7, -1), (12.5, 13), (-37.5, -37),
          (2.0 ** 52 - 0.5, 2 ** 52), (2.0 ** 53, 2 ** 53), (7, 7), (True, 1), (None, 0), ("2.5", 3)]
FIXED = [(0.3125, 3, 0.313), (0.5625, 3, 0.563), (-0.3125, 3, -0.313), (1.005, 2, 1.0), (1.125, 2, 1.13),
         (2.675, 2, 2.67), (1e21, 2, 1e21), (-1e21, 2, -1e21), (0.5, 0, 1.0), (1.5, 0, 2.0), (2.5, 0, 3.0),
         (-2.5, 0, -3.0), (1.45, 1, 1.4), (-0.0001, 2, 0.0), (123.456, 0, 123.0), (0.000001, 7, 0.000001),
         (5e-324, 3, 0.0), (999.9995, 3, 1000.0), (0.0625, 3, 0.063), (3707.949951171875, 2, 3707.95),
         (1e20 + 0.5, 1, 1e20), (9.995, 2, 9.99), (0.045, 2, 0.04), (100.125, 2, 100.13),
         (1.0000000000000002, 15, 1.0), (7, 2, 7.0)]


@pytest.mark.parametrize("v, want", ROUNDS)
def test_js_round_is_math_round(v, want):
    got = S.js_round(v)
    assert got == want and isinstance(got, int)


def test_js_round_rounds_halves_up_on_the_exact_value():
    """Math.round(v) = floor(v + 1/2) worked out exactly, never in floating point: for doubles of every size, the
    halves between them and their neighbours; NaN and infinities as they are."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.standard_normal(3000) * 10.0 ** rng.integers(-8, 18, 3000),
                             np.round(rng.uniform(-1e6, 1e6, 3000) * 2) / 2])
    values = np.concatenate([values, np.nextafter(values, np.inf), np.nextafter(values, -np.inf)])
    for v in values.tolist():
        assert S.js_round(v) == math.floor(Fraction(v) + Fraction(1, 2)), v
    assert math.isnan(S.js_round(math.nan)) and S.js_round(-math.inf) == -math.inf


@pytest.mark.parametrize("v, d, want", FIXED)
def test_to_fixed_is_number_to_fixed(v, d, want):
    got = S.to_fixed(v, d)
    assert got == want and isinstance(got, float)


def test_to_fixed_rounds_the_exact_decimal_half_up():
    """+v.toFixed(d): the double's exact value to d decimals, a tie to the larger magnitude, read back as the nearest
    double (exact rational arithmetic as the reference); NaN, infinities and |v| >= 1e21 as they are; d outside 0-100
    refused as JavaScript does."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.standard_normal(2000) * 10.0 ** rng.integers(-6, 16, 2000),
                             np.round(rng.uniform(-100, 100, 2000) * 16) / 16,      # binary ties at 3 and 4 decimals
                             np.round(rng.uniform(-1e4, 1e4, 2000), 3)])
    for k, v in enumerate(values.tolist()):
        d = k % 6
        exact = abs(Fraction(v)) * 10 ** d
        n = math.floor(exact + Fraction(1, 2))                                     # a tie takes the larger n
        assert S.to_fixed(v, d) == math.copysign(n / 10 ** d, v), (v, d)
    assert math.isnan(S.to_fixed(math.nan, 2)) and S.to_fixed(math.inf, 2) == math.inf
    assert S.to_fixed(1.5e21, 2) == 1.5e21 and S.to_fixed(np.float32(0.3125), 3) == 0.313
    for d in (-1, 101):
        with pytest.raises(ValueError):
            S.to_fixed(1.0, d)
    with pytest.raises(TypeError):
        S.to_fixed(True, 2)


# ---- each function's own rules -----------------------------------------------------------------------------------
def ready(**extra) -> dict:
    out = {"v": 1, "state": "ready", "seq": 12, "iter": 12, "cell": 5, "transcripts": 200, "code": 120,
           "genes": [["A", 10, 2.0, 8.0, 1, 0], ["B", 10, 5.0, 5.0, 1], ["C", 20, 20.0, 0.0, 0]],
           "own_genes": [["C", 20.0, 20]], "neighbours": [{"cell": 6, "tx": 40.0, "same": False}]}
    out.update(extra)
    return out


def reply_of(points, **extra) -> dict:
    """A v 3 where reply: points (x, y, z, gene, owner, near, w?): w by default all of it inside the cell's outline."""
    out = {"v": 3, "cell": 5, "seq": 12, "genes": ["A", "B", "C"], "roles": ["from", "from", "own"], "held": "exact",
           "gene_stats": [{"gene": "A", "role": "from", "giver": 0, "in_cell": 4, "near_giver": 3, "own_near": [2, 20]},
                          {"gene": "B", "role": "from", "giver": None, "in_cell": 3}],
           "points": {k: [p[i] for p in points] for i, k in enumerate(("x", "y", "z", "gene", "owner", "near"))}}
    out["points"]["w"] = [p[6] if len(p) > 6 else float(p[4] == 1) for p in points]
    out.update(extra)
    return out


POINTS = [(1.0, 2.0, 0.5, 0, 1, 1), (3.0, 2.0, 0.5, 0, 1, 0), (5.0, 2.0, None, 1, 1, 0), (7.0, 2.0, 0.5, 0, 2, 1),
          (9.0, 2.0, 0.5, 2, 1, 0), (1.0, 2.0, 0.25, 0, 255, 0)]


def test_a_where_reply_becomes_typed_arrays_as_the_page_holds_them():
    req = S.page_params(ready(), 5)["req"]
    reply = S.normalise_where(reply_of(POINTS), req)
    assert isinstance(reply, S.WhereReply) and reply.count == 6 and reply.v == 3 and reply.seq == 12
    assert [a.dtype for a in (reply.x, reply.y, reply.z, reply.gene, reply.owner, reply.near, reply.w)] == \
        [np.float32] * 3 + [np.uint8] * 3 + [np.float64]
    assert np.isnan(reply.z[2]) and reply.z[0] == 0.5 and reply.near.tolist() == [1, 0, 0, 1, 0, 0]
    assert reply.w.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 0.0] and reply.held == "exact"
    assert reply.stats[0]["own_near"] == [2, 20] and reply.stats[1]["giver"] is None and reply.box is None
    # Without near flags all are 0; a reply for another cell or frame is not this request's.
    flat = reply_of(POINTS)
    del flat["points"]["near"]
    assert S.normalise_where(flat, req).near.tolist() == [0] * 6
    assert S.normalise_where(reply_of(POINTS, cell=6), req) is None
    assert S.normalise_where(reply_of(POINTS, seq=13), req) is None
    assert S.normalise_where(reply_of(POINTS, seq="12"), req) is not None             # String(seq) is compared
    # Version 3 only (an older engine's reply has no shares), and every point's share from 0 to 1.
    for version in (None, 1, 2, "3", 4):
        assert S.normalise_where(reply_of(POINTS, v=version), req) is None, version
    no_shares = reply_of(POINTS)
    del no_shares["points"]["w"]
    assert S.normalise_where(no_shares, req) is None
    for w in (1.0001, -0.001, None, "1", True):
        assert S.normalise_where(reply_of([*POINTS[:5], (*POINTS[5], w)]), req) is None, w
    # What it reads leniently: held as one of its words, cell_total any number from 0.
    assert S.normalise_where(reply_of(POINTS, held="maybe"), req).held is None
    assert S.normalise_where(reply_of(POINTS, cell_total=12.375), req).cell_total == 12.375
    assert S.normalise_where(reply_of(POINTS, cell_total=-1), req).cell_total is None


def test_p_is_the_share_times_the_likelihood_on_any_owner_as_float32():
    """P = w x pi, both float32: pi the near/far likelihood of the point's gene, w proseg's share of the point for the
    cell; a point with no share is no suspect, wherever it lies; one in a source's outline that proseg moved into the
    cell is."""
    result = S.normalise(ready())
    reply = S.normalise_where(reply_of(POINTS), None)
    p, pi, genes = S.suspect_scores(reply, result)
    assert p.dtype == pi.dtype == np.float32 and p[3] == 0 and p[4] == 0 and p[5] == 0   # w 0, an own gene, w 0
    a = next(g for g in genes if g["gene"] == "A")
    fr, fg, s = 3 / 22, 0.75, 0.8
    fs = min(1.0, max(fr, (fg - (1 - s) * fr) / s))
    near = s * fs / (s * fs + (1 - s) * fr)
    assert a["giver"] == 0 and math.isclose(a["fr"], fr) and math.isclose(a["fs"], fs)
    assert p[0] == pi[0] == np.float32(near) and p[2] == pi[2] == np.float32(0.5)       # near; B: no giver, s
    moved = [(*q[:6], w) for q, w in zip(POINTS, (0.5, 0.25, 0.75, 0.625, 1.0, 0.0))]
    p, pi, _ = S.suspect_scores(S.normalise_where(reply_of(moved), None), result)
    assert p[0] == np.float32(0.5 * near) and pi[0] == np.float32(near)
    assert p[3] == np.float32(0.625 * near) and pi[3] == np.float32(near), "in the source's outline, moved in"
    assert p[2] == np.float32(0.75 * 0.5) and p[5] == 0 and pi[5] == 0
    assert S.suspect_scores(reply, None)[0].tolist() == [0.0] * 6
    assert S.suspect_scores(reply, S.normalise(ready(state="too_few")))[2] == []


def test_the_support_of_a_source_is_the_share_of_its_glow_near_it():
    """source_support: per source, w summed over the glowing points of the genes it gives that lie near it (the near
    flag); a source the glow would name with none (the strip case: its pink dots far from it) is left out."""
    result = S.normalise(ready())
    far = [(1.0 + k, 2.0, 0.5, 0, 1, 0, 0.875) for k in range(13)]                  # A's points, none near its giver
    reply = S.normalise_where(reply_of(far), None)
    p = S.suspect_scores(reply, result)[0]
    assert (p >= S.GLOW_MIN).all() and S.source_support(reply, p) == {}
    near = [(1.0 + k, 2.0, 0.5, 0, 1, int(k < 4), w) for k, w in enumerate([0.5, 0.25, 0.125, 0.75] + [1.0] * 9)]
    reply = S.normalise_where(reply_of(near), None)
    p = S.suspect_scores(reply, result)[0]
    # The third near one glows too little (0.125 x pi under 0.2): left out.
    assert float(p[2]) < S.GLOW_MIN and S.source_support(reply, p) == {0: 0.5 + 0.25 + 0.75}
    assert S.source_support(None, p) == {} and S.source_support(reply, np.zeros(13)) == {}


def test_points_sharing_a_place_are_ranked_by_depth():
    reply = S.normalise_where(reply_of(POINTS), None)
    # Point 0 and 5: gene A at (1, 2), any owner; z 0.5 and 0.25: point 5 first.
    assert S.same_place(reply, [0, 1, 2]) == {0: (1, 2)}
    assert S.same_place(reply, []) == {} and S.same_place(None, [0]) == {}


def test_the_kept_list_takes_the_cells_glowing_points_at_most_max_n():
    reply = S.normalise_where(reply_of(POINTS), None)
    p = np.array([0.9, 0.3, 0.2, 0.95, 0.19, 0.6], np.float32)                    # as if every point were scored
    # Those proseg counts for the cell (w > 0) only; P is 0 on the others anyway.
    assert [(s["x"], s["p"]) for s in S.keep_suspects(reply, p)] == [(1.0, 0.9), (3.0, 0.3), (5.0, 0.2)]
    assert [s["x"] for s in S.keep_suspects(reply, p, cell_only=False)] == [7.0, 1.0, 1.0, 3.0, 5.0]
    assert len(S.keep_suspects(reply, p, max_n=2)) == 2 and S.keep_suspects(reply, p, max_n=0) == []
    # A float64 P is first stored as float32: 0.19999999999 glows as the page's Float32Array would hold it.
    assert [s["p"] for s in S.keep_suspects(reply, np.array([0.19999999999] + [0] * 5))] == [0.2]
    assert S.keep_suspects(reply, p)[0] == {"x": 1.0, "y": 2.0, "gene": "A", "p": 0.9, "rank": 1, "of": 2}


def test_a_ready_answer_needs_a_cell_number_and_others_do_not():
    for cell in (-1, 1.5, "5", True, None):
        with pytest.raises(ValueError, match="cell must be a cell number"):
            S.page_params(ready(), cell)
    assert S.page_params(ready(state="empty"), "x")["state"] == "empty"
    assert S.page_params({"state": "ready"}, -1) == {"state": None, "none": False, "chips": [], "rows": [], "req": None}
    assert S.page_params(ready(), 5.0)["req"]["params"]["cell"] == 5


def test_the_start_change_keeps_its_two_numbers_as_the_page_does():
    """normalise keeps start_change's now and start (the cell's share from neighbours now and at the start, both at
    the matched depth: the numbers its points compare) when they are finite numbers, else null, as qc-deeper.js's
    normalise does; a start_change without points is none at all, as before."""
    sc = {"points": -4.69, "band": 5.0, "draws": 0, "now_tx": 682.0, "start_tx": 701.5, "now": 38.31, "start": 43.0,
          "direction": "fewer"}
    r = S.normalise(ready(start_change=sc))["start_change"]
    assert r == {"points": -4.69, "band": 5.0, "draws": 0, "now_tx": 682.0, "start_tx": 701.5, "now": 38.31,
                 "start": 43.0, "direction": "fewer", "held": False}
    for odd in (None, "38", True, float("nan"), float("inf"), [38.0]):
        found = S.normalise(ready(start_change={**sc, "now": odd, "start": odd}))["start_change"]
        assert (found["now"], found["start"]) == (None, None) and found["points"] == -4.69, odd
    older = S.normalise(ready(start_change={"points": 2.4, "band": 3}))["start_change"]           # a worker before
    assert (older["now"], older["start"]) == (None, None)
    assert S.normalise(ready(start_change={"now": 38.31, "start": 43.0}))["start_change"] is None


def test_a_neighbour_without_a_cell_is_javascripts_undefined():
    """normalise keeps a missing cell as UNDEFINED (not null: null + 1 is a label, undefined + 1 is not): its row is
    keyed by its place, has no label and is never a source; the page's JSON leaves the cell out."""
    r = S.normalise(ready(neighbours=[{"tx": 40.0}, {"cell": None, "tx": 30.0}]))
    assert r["neighbours"][0]["cell"] is S.UNDEFINED and not S.UNDEFINED and S.UNDEFINED is not None
    m = S.explain_core(r)
    assert [n["key"] for n in m["neighbours"]] == ["n0", 1] and math.isnan(m["neighbours"][0]["label"])
    params = S.page_params(ready(neighbours=[{"tx": 40.0}, {"cell": None, "tx": 30.0}]), 5)
    assert params["rows"][0] == {"label": None, "share": 0.2, "same": False, "key": "n0"}
    assert params["req"]["params"]["sources"] == [0]


def test_whole_numbers_past_2_53_add_up_as_doubles():
    """JavaScript adds in doubles, Python's ints keep every bit: past 2^53 a sum loses its last bits as the page's does
    (2^53 + 1 is 2^53, 2^53 + 3 rounds to the even 2^53 + 4), in a label (cell + 1), in f_r's own_near counts and in
    the genes counted; below it nothing changes."""
    big = 2 ** 53
    # Two neighbours whose labels are one double: one row, the first's place, the last neighbour.
    params = S.page_params(ready(neighbours=[{"cell": big - 1, "tx": 40.0}, {"cell": big, "tx": 30.0},
                                             {"cell": 7, "tx": 20.0}]), 5)
    assert [(r["label"], r["cell"], r["share"]) for r in params["rows"]] == [(big, big, 0.15), (8, 7, 0.1)]
    assert params["req"]["params"]["sources"] == [big - 1, 7]
    assert S.page_params(ready(), big)["req"]["params"]["cell"] == big - 1
    assert S.page_params(ready(), big - 2)["req"]["params"]["cell"] == big - 2         # exact below 2^53
    # f_r = (own_near[0] + 1) / (own_near[1] + 2), each sum a double.
    reply = S.normalise_where(reply_of(POINTS, gene_stats=[
        {"gene": "A", "role": "from", "giver": 0, "in_cell": 4, "near_giver": 3, "own_near": [big, 1]},
        {"gene": "B", "role": "from", "giver": 0, "in_cell": 4, "near_giver": 3, "own_near": [3, big - 1]}]), None)
    genes = {g["gene"]: g for g in S.suspect_scores(reply, S.normalise(ready(genes=[
        ["A", 10, 2.0, 8.0, 1, 0], ["B", 10, 2.0, 8.0, 1, 0]])))[2]}
    assert (genes["A"]["fr"], genes["B"]["fr"]) == (big / 3, 4 / big)            # not (big + 1) / 3, 4 / (big + 1)
    # The genes counted with the reply's folded rest: three rows and 2^53 more.
    assert S.normalise(ready(genes_rest=[big, 1, 1, 1]))["genes_total"] == big + 4
    assert S.normalise(ready(genes_rest=[big - 4, 1, 1, 1]))["genes_total"] == big - 1


def test_ranks_change_only_past_a_quarter_more():
    """rankOrder from rows shown before (checked against qc-deeper.js): a row moves up only when its value passes 1.25 x
    the one above, over as many passes as it takes; new rows enter where their value puts them."""
    value = {"a": 1.0, "b": 1.2, "c": 4.0, "d": 3.0, "e": 0.0, "f": 0.0, "z": 10.0}.__getitem__
    assert S.rank_order(["a", "b"], ["a", "b"], value) == ["a", "b"]
    assert S.rank_order(["a", "b"], ["a", "b"], {"a": 1.0, "b": 1.3}.__getitem__) == ["b", "a"]
    assert S.rank_order(["a"], ["a", "c", "d"], {"a": 5.0, "c": 4.0, "d": 3.0}.__getitem__) == ["a", "c", "d"]
    assert S.rank_order(["a"], ["a", "z"], value) == ["z", "a"]
    assert S.rank_order(["x", "a"], ["a"], value) == ["a"]
    assert S.rank_order(["a", "b", "c"], ["a", "b", "c"], {"a": 1, "b": 2, "c": 4}.__getitem__) == ["c", "b", "a"]
    assert S.rank_order(None, ["e", "f", "a"], value) == ["a", "e", "f"]
    model = S.explain_core(S.normalise(ready(neighbours=[{"cell": c, "tx": tx} for c, tx in ((6, 40.0), (7, 45.0))])))
    assert [r["key"] for r in S.source_rows(model)] == [8, 7]                          # rows keyed by label
    assert [r["key"] for r in S.source_rows(model, [7, 8])] == [7, 8]                  # 0.225 is under 1.25 x 0.2
    assert [r["key"] for r in S.source_rows(model, [9, 7])] == [8, 7]                  # 9 is gone; 8 enters above


def test_odd_inputs_never_make_it_fail():
    """Golden inputs with one value replaced by any JSON value, many times over: the port answers (or says the cell is
    not a cell number), as the page answers every one of them."""
    rng = random.Random(7)
    pool = [None, True, False, 0, -1, 1.5, -0.0, 1e300, 2 ** 64, "x", "", "12", [], {}, [1, "a"], {"a": 1}, [[]],
            float("nan"), float("inf")]

    def paths(v, at=()):
        yield at
        if isinstance(v, dict):
            for k, x in v.items():
                yield from paths(x, at + (k,))
        elif isinstance(v, list):
            for i, x in enumerate(v[:6]):
                yield from paths(x, at + (i,))

    def replaced(v, at, new):
        if not at:
            return new
        v = json.loads(json.dumps(v))
        node = v
        for k in at[:-1]:
            node = node[k]
        node[at[-1]] = new
        return v
    for _ in range(400):
        case = rng.choice(CASES)
        result, reply = case["result"], case["reply"]
        if rng.random() < 0.5 or reply is None:
            result = replaced(result, rng.choice(list(paths(result))), rng.choice(pool))
        else:
            reply = replaced(reply, rng.choice(list(paths(reply))), rng.choice(pool))
        params = S.page_params(result, case["cell"])
        S.page_scores(result, params["req"], reply)
        kept_of(result, params["req"], reply)
        json.dumps(params, allow_nan=False)


def test_the_constants_are_the_pages():
    """The constants as frontend/src/qc-deeper.js exports them (qc-scoring.test.mjs checks the same from the page's
    side, on the page's own exports)."""
    page = (APP / "frontend/src/qc-deeper.js").read_text()

    def exported(name):
        found = re.search(rf"\b{name} = (Object\.freeze\(\[[^\]]*\]\)|[0-9.]+)", page)
        assert found, name
        text = found.group(1)
        return tuple(json.loads(text[len("Object.freeze("):-1])) if text.startswith("Object") else json.loads(text)
    for name in ("RESULT_VERSION", "RANK_RATIO", "CHIPS", "ROWS", "ROW_MIN", "SOURCES_MIN", "SAME_KIND", "WHERE_GENES",
                 "WHERE_OWN", "WHERE_V", "GLOW_MIN", "LIKELY_MIN", "LIKELY_W", "SUSPECTS_SENT", "MATCH_UM"):
        assert getattr(S, name) == exported(name), name
    assert re.search(r"\bconst OWNER_CELL = 1\b", page)
    assert re.search(r"\bHELD = Object\.freeze\(\['exact', 'approx', 'none'\]\)", page)
    assert S.HELD == ("exact", "approx", "none")


def test_a_reply_as_large_as_the_engine_sends_is_scored_quickly():
    """30,000 points (the where reply's most) of nine genes: the page's whole work on it well under a second (the pool
    scores a cell at a time, the engine's other requests waiting)."""
    rng = np.random.default_rng(3)
    n, genes = 30_000, [f"G{k}" for k in range(9)]
    result = ready(genes=[[g, 100, 40.0, 60.0, 1, k % 3] for k, g in enumerate(genes[:6])],
                   own_genes=[[g, 99.0, 100] for g in genes[6:]], neighbours=[{"cell": 10 + k, "tx": 50.0}
                                                                         for k in range(5)])
    params = S.page_params(result, 5)
    q = params["req"]["params"]
    raw = {"v": 3, "cell": 5, "seq": 12, "genes": q["genes"] + q["own"],
           "roles": ["from"] * len(q["genes"]) + ["own"] * len(q["own"]),
           "gene_stats": [{"gene": g, "role": "from", "giver": k % 3, "in_cell": 500, "near_giver": 200,
                           "own_near": [100, 1000]} for k, g in enumerate(q["genes"])],
           "points": {"x": np.round(rng.uniform(0, 400, n), 2).tolist(), "y": np.round(rng.uniform(0, 400, n), 2).tolist(),
                      "z": np.round(rng.uniform(0, 1, n), 3).tolist(), "gene": rng.integers(0, 9, n).tolist(),
                      "owner": rng.choice([0, 1, 2, 3, 255], n).tolist(), "near": rng.integers(0, 2, n).tolist(),
                      "w": np.round(rng.choice([0.0, 0.5, 1.0], n) * rng.uniform(0.5, 1.0, n), 3).tolist()}}
    began = time.perf_counter()
    scores = S.page_scores(result, params["req"], raw)
    kept = kept_of(result, params["req"], raw)
    assert time.perf_counter() - began < 1.0
    assert len(scores["points"]) > 1000 and len(kept) == S.SUSPECTS_SENT
