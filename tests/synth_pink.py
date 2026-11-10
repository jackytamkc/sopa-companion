"""The suspect test's synthetic world, for tests/test_qc_rule.py (a helper beside the tests: pytest does not collect it).

A pool as engine/qc_pool.py fixes it before any run (Items over its cells and their givers, PoolOrder) and its test runs
as engine/qc_readout.py reads them (one RunReadout per role), made two ways:
  - World, a model with known answers, for the rule's Monte Carlo checks: as the measurements' own synthetic trees, the
    share of a transcript still in its cell in run r is
        HELD(t, r) = sigmoid(alpha_c + shift + delta_t + b_near * near_t + w(r) + u(c, r) + v(r) + e(t, r))
    with alpha_c the cell's level, delta_t the suspect's own (suspects only), near_t within 2 µm of its giver (held
    less), w(r) a run effect all the pool's items share, u(c, r) one a cell's items share, v(r) one its suspects alone
    share (suspects moving apart from their own RNA from run to run, as measured: without it the excess over matched own
    RNA cancels every run effect), e(t, r) per item and run, and `excess` added to the suspects' level (a pool whose
    suspects are held more than their own RNA: a mean X above 0). Matched controls share their suspect's cell, giver and
    near flag (not delta_t); uniform controls are near with p 0.2. Side B thins the suspects' HELD with Bernoulli(1 -
    effect) per run (a drop of the suspects alone) and, with `shrink`, every control's too (a pure shrink: suspects and
    their own RNA dropping alike). Every item is read in every run.
  - planted(): each run's pooled shares given exactly (the mean over T, M and U, the transcripts per cell, where the
    suspects went), the items spread around them with zero mean; items left out with the statuses asked. For the rule's
    edge tests, and for frontend/test/fixtures/qc-test-result.json: `fixture_examples()` builds its examples through
    answer(), their caveats then as engine/qc_test.py gives them (answer_caveats), and running this file writes them
    (python tests/synth_pink.py frontend/test/fixtures/qc-test-result.json).
Cells are named with runs of one letter (aabbbbbb-1) and genes GENE1.., as the page's fixtures name them."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))      # the folder holding the sopa_companion package (run as a script)

from sopa_companion.engine import qc_pool, qc_rule  # noqa: E402
from sopa_companion.engine.qc_readout import (  # noqa: E402
    C0_ABSENT, C1_ABSENT, CROSSCHECK_FAILED, MATCHED, NO_GIVER, NOT_FOUND, SUSPECT, UNIFORM, Items, RunReadout)
from sopa_companion.engine.qc_test import answer_caveats  # noqa: E402

LETTERS = "abcdefghijklmnop"
# Made-up Xenium ids, runs of one letter: "aabbbbbb-1", "aacccccc-1", ...
NAMES = tuple(a * k + b * (8 - k) + "-1" for k in (2, 3, 4, 5, 6) for a, b in itertools.permutations(LETTERS, 2))
GENES = tuple(f"GENE{i}" for i in range(1, 13))
INSET = qc_rule.MEASURED["inset_um"]
SPLIT = (0.15, 0.30, 0.55)     # how a share not held divides into giver, other and background, unless asked otherwise
FIXTURE = "frontend/test/fixtures/qc-test-result.json"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def roles(per_side: int) -> dict:
    return {"A": [f"A{i}" for i in range(1, per_side + 1)], "B": [f"B{i}" for i in range(1, per_side + 1)]}


# ---- the pool's layout: its cells, their items and the order ---------------------------------------------------------
class Layout:
    """The cells of `order` (seed first; members, then the reserve) and one giver cell each, and their items: per cell
    `sizes[i]` suspects, `matched_per` matched controls per suspect (same cell, giver and near flag), `uniform_per`
    uniform ones, and `extra[i]` more suspects (those planted() leaves out). Items are grouped by cell in order, each
    cell's suspects in point order. Cells sit at `distances` from the seed, inside the area's inner box."""

    def __init__(self, sizes, distances, *, box=(3000.0, 3000.0, 3500.0, 3500.0), seed_cell=1234, extra=None,
                 matched_per=3, uniform_per=50, rng=None, names=None):
        rng = np.random.default_rng(0) if rng is None else rng
        sizes = [int(v) for v in sizes]
        count = len(sizes)
        extra = [0] * count if extra is None else [int(v) for v in extra]
        self.box = tuple(float(v) for v in box)
        self.inner = (self.box[0] + INSET, self.box[1] + INSET, self.box[2] - INSET, self.box[3] - INSET)
        names = iter(NAMES if names is None else names)
        # Items.cell_names: the order's cells, then their givers (one each, never in the pool).
        self.cell_names = tuple(next(names) for _ in range(2 * count))
        self.parent_cells = tuple([seed_cell] + [seed_cell + 37 * (i + 1) + 11 * i * i for i in range(count - 1)])
        self.distances = tuple(float(v) for v in distances)
        cx0, cy0 = (self.inner[0] + self.inner[2]) / 2, (self.inner[1] + self.inner[3]) / 2
        centres = []
        for d in self.distances:
            while True:
                angle = rng.uniform(0, 2 * np.pi)
                cx, cy = cx0 + d * np.cos(angle), cy0 + d * np.sin(angle)
                if d == 0 or qc_rule.inside_inner([cx], [cy], self.inner)[0]:
                    break
            centres.append((cx, cy))
        kind, cell, giver, target, gene, x, y, near, group = [], [], [], [], [], [], [], [], []

        def add(k, c, g, tgt, at, is_near, what):
            kind.append(k)
            cell.append(c)
            giver.append(g)
            target.append(tgt)
            gene.append(int(rng.integers(len(GENES))))
            x.append(at[0])
            y.append(at[1])
            near.append(is_near)
            group.append(what)
        for c, (n, more) in enumerate(zip(sizes, extra)):
            cx, cy = centres[c]
            for what, how_many in (("t", n), ("x", more)):
                for _ in range(how_many):
                    add(SUSPECT, c, count + c, -1, (cx + rng.uniform(-4, 4), cy + rng.uniform(-4, 4)),
                        bool(rng.random() < 0.8), what)
        suspects = [i for i in range(len(kind)) if kind[i] == SUSPECT]
        for i in suspects:
            for _ in range(matched_per):
                add(MATCHED, cell[i], giver[i], i, (x[i] + rng.normal(0, 1), y[i] + rng.normal(0, 1)), near[i],
                    "m" if group[i] == "t" else "mx")
        for c in range(count):
            cx, cy = centres[c]
            for _ in range(uniform_per):
                add(UNIFORM, c, NO_GIVER, -1, (cx + rng.normal(0, 4), cy + rng.normal(0, 4)), bool(rng.random() < 0.2),
                    "u")
        n = len(kind)
        self.items = Items(kind=np.array(kind, np.int8), cell=np.array(cell, np.int32), giver=np.array(giver, np.int32),
                           target=np.array(target, np.int64), gene=np.array(gene, np.int32),
                           transcript_id=np.arange(1000, 1000 + n, dtype=np.uint64),
                           x=np.array(x, np.float32), y=np.array(y, np.float32), cell_names=self.cell_names,
                           gene_names=GENES)
        self.items.check()
        self.order = qc_rule.PoolOrder(seed=0, cells=tuple(range(count)), distance_um=self.distances,
                                       parent_cells=self.parent_cells)
        self.group = np.array(group)
        self.near = np.array(near, bool)
        self.sizes, self.extra = sizes, extra

    def rows(self, what: str) -> np.ndarray:
        """The items of one group: "t" the planned suspects, "x" the extra ones, "m"/"mx" their matched controls, "u"."""
        return np.flatnonzero(self.group == what)


def readouts(layout: Layout, held, giver, other, bg, status=None, *, tx=None, per_side: int, iteration=450,
             rows=535510, seconds=None, errors=None) -> dict:
    """One RunReadout per role from per-run arrays [2N, items] (A1..AN, then B1..BN)."""
    names = roles(per_side)["A"] + roles(per_side)["B"]
    count = len(layout.items)
    status = np.zeros((len(names), count), np.int8) if status is None else status
    out = {}
    for r, role in enumerate(names):
        out[role] = RunReadout(
            status=np.asarray(status[r], np.int8), held=held[r], giver=giver[r], other=other[r], bg=bg[r],
            tx_inner=None if tx is None else tx[r], cells_inner=40, iter=iteration, seq=iteration + 1, rows=rows,
            ids="a" * 64, seconds=0.3 if seconds is None else seconds[r],
            checks={"identity_pairs": 4000, "identity_max_err": 4e-13 if errors is None else errors[r], "f_gt_n": 0,
                    "dest_vs_assignment": 0, "rows_dest_outside": 0, "offset_residual": 0.0, "bg_gap": 0.0,
                    "crosscheck": {"gene": 0, "x": 0, "y": 0}, "duplicate_cells": 0})
    return out


# ---- the model, for the Monte Carlo checks ----------------------------------------------------------------------------
class World:
    """A pool reaching k with RESERVE more cells, from the model above (see the module docstring). sig: the spreads
    (alpha, delta, w, u, v, e) of the levels and run effects, in logits."""
    RESERVE = 2

    def __init__(self, seed: int, *, start="paired", situation="as_is", k=160, excess=0.0, sig=None, alpha_mean=0.8,
                 shift=0.2, b_near=-0.6, matched_per=3, uniform_per=0):
        rng = np.random.default_rng(seed)
        self.start, self.situation, self.k = start, situation, k
        self.per_side = qc_rule.constants_for(start, situation)["runs_per_side"]
        self.sig = dict(alpha=0.9, delta=0.4, w=0.1, u=0.35, v=0.15, e=1.0) if sig is None else dict(sig)
        sizes = []
        while sum(sizes) < k:           # cells of synth's sizes until the pool is reached, then the reserve
            sizes.append(int(np.clip(int(rng.lognormal(1.5, 0.7)), 1, 30)))
        sizes += [int(np.clip(int(rng.lognormal(1.5, 0.7)), 1, 30)) for _ in range(self.RESERVE)]
        self.members = len(sizes) - self.RESERVE
        distances = np.concatenate([[0.0], np.cumsum(rng.uniform(2.0, 6.0, len(sizes) - 1))])
        self.layout = Layout(sizes, distances, matched_per=matched_per, uniform_per=uniform_per, rng=rng)
        items = self.layout.items
        alpha = rng.normal(alpha_mean, self.sig["alpha"], len(sizes))
        suspect = items.kind == SUSPECT
        self.base = (alpha[items.cell] + shift + b_near * self.layout.near
                     + suspect * (rng.normal(0, self.sig["delta"], len(items)) + excess))
        self.suspect = suspect
        self.controls = ~suspect
        in_pool = items.cell < self.members
        self.t_rows = np.flatnonzero(suspect & in_pool)                      # T and M as the rule must take them
        self.m_rows = np.flatnonzero((items.kind == MATCHED) & in_pool)

    def held(self, runs: int, rng, *, effect=0.0, shrink=False) -> np.ndarray:
        """HELD [items, runs] for `runs` fresh runs; with `effect`, every run thinned (suspects, and controls with
        `shrink`)."""
        s = self.sig
        items = self.layout.items
        logit = (self.base[:, None] + rng.normal(0, s["w"], runs)[None, :]
                 + rng.normal(0, s["u"], (len(self.layout.sizes), runs))[items.cell]
                 + self.suspect[:, None] * rng.normal(0, s["v"], runs)[None, :]
                 + rng.normal(0, s["e"], (len(items), runs)))
        h = sigmoid(logit)
        if effect > 0:
            thinned = self.suspect | self.controls if shrink else self.suspect
            keep = rng.random(h.shape) < (1.0 - effect)
            h = np.where(thinned[:, None] & ~keep, 0.0, h)
        return h

    def experiments(self, count: int, rng, *, effect=0.0, shrink=False) -> np.ndarray:
        """HELD [count, 2N, items]: A runs as now, B runs with the effect."""
        n = self.per_side
        a = self.held(count * n, rng).T.reshape(count, n, -1)
        b = self.held(count * n, rng, effect=effect, shrink=shrink).T.reshape(count, n, -1)
        return np.concatenate([a, b], axis=1)

    def answer(self, held_runs: np.ndarray, constants: dict) -> dict:
        """answer() on one experiment's HELD [2N, items] (the rest of each share split as SPLIT)."""
        rest = 1.0 - held_runs
        kwargs = dict(per_side=self.per_side, tx=[200.0] * (2 * self.per_side))
        reads = readouts(self.layout, held_runs, rest * SPLIT[0], rest * SPLIT[1], rest * SPLIT[2], **kwargs)
        return qc_rule.answer(self.layout.items, self.layout.order, reads, roles(self.per_side), self.start,
                              setting={"flag": "cell-compactness", "value": 0.02, "current": 0.04},
                              situation=self.situation, at_iter=450, matched_counts=matched_counts(),
                              parent_defaults=True, constants=constants)


def matched_counts(unsure=0, not_found=0) -> dict:
    return {"target": {"ok": 0, "not_found": not_found, "unsure": unsure}, "matched": {"ok": 0, "not_found": 0, "unsure": 0},
            "uniform": {"ok": 0, "not_found": 0, "unsure": 0}}


# ---- planted shares, for the edge tests and the fixture ---------------------------------------------------------------
def _spread(base: np.ndarray, count: int, rng, scale: float) -> np.ndarray:
    """`count` share vectors (each adding to 1, none negative, a share of 0 kept at 0) whose mean is `base` (to the
    last bit or so), spread as Dirichlet draws around it, `scale` times (less where a share would go below 0)."""
    if count == 0:
        return np.zeros((0, 4))
    if scale == 0 or count == 1:
        return np.repeat(base[None, :], count, axis=0)
    live = base > 0
    d = np.zeros((count, 4))
    d[:, live] = rng.dirichlet(base[live] * 40.0, size=count)
    d -= d.mean(axis=0)
    low = d < 0
    room = float(np.min(np.broadcast_to(base, d.shape)[low] / -d[low])) if low.any() else np.inf
    return base[None, :] + d * min(scale, 0.9 * room)


def planted(*, start="paired", situation="as_is", row=None, caveats=(), sizes, hT, hM=None, hU=None, tx=None,
            went=None, distances=None, reserve=(), extra=None, drops=(), matched=None, unsure=0, not_found=0, rule=None,
            setting=None, parent_defaults=True, seed_cell=1234, uniform_per=50, scale=1.0, rng_seed=0, rows=535510,
            seconds=None, errors=None, names=None) -> dict:
    """answer()'s arguments for a pool of `sizes` eligible suspects per cell (seed first, then members, then the
    `reserve`), whose runs give exactly these pooled shares: hT, hM, hU {"A": [...], "B": [...]} per run (hM None: every
    matched control is missing from the last run; hU None: no uniform control), tx per run, and `went` per side ({stay,
    neighbour, other, background}, stay being the side's mean of hT). `drops`: (cell's place in the order, status,
    count) for suspects read in some runs only (each missing from one run, by its status); `extra` per cell holds them.
    `matched`: the matched controls read everywhere (default 3 per suspect). `scale`: how far items spread around their
    run's shares (0: all alike, for exact edges). `row`: the measured row a test where the measurements did not cover
    its start and situation is judged with (qc_rule.row_for), `caveats` the reasons it carries (both answer()'s, passed
    only when given)."""
    rng = np.random.default_rng(rng_seed)
    constants = qc_rule.constants_for(*(row or (start, situation)))
    if rule is not None and rule != constants["rule"]:
        constants = {**constants, "rule": rule, "tau": constants["taus"][rule]}
    per_side = constants["runs_per_side"]
    runs = 2 * per_side
    sizes = [int(v) for v in sizes] + [int(v) for v in reserve]
    if distances is None:
        distances = [0.0] + [round(4.0 + 3.5 * i, 1) for i in range(len(sizes) - 1)]
    extra = [0] * len(sizes) if extra is None else list(extra)
    for place, _, count in drops:
        extra[place] += count
    layout = Layout(sizes, distances, seed_cell=seed_cell, extra=extra, uniform_per=uniform_per,
                    matched_per=3, rng=rng, names=names)
    items = layout.items
    n_items = len(items)
    members, _ = qc_rule.pool_of(np.array(sizes), layout.order, constants["k"])
    member = np.isin(items.cell, members)
    t_rows, x_rows, u_rows = layout.rows("t"), layout.rows("x"), layout.rows("u")
    m_rows, mx_rows = layout.rows("m"), layout.rows("mx")
    # What the rule averages, each group spread around its run's shares on its own (its mean stays exact): T, the
    # members' planned suspects; M, their first `matched` matched controls (all, by default); U, their uniform ones.
    t_in = t_rows[member[t_rows]]
    m_in = m_rows[member[m_rows]]
    if hM is not None and matched is not None:
        if int(matched) > len(m_in):
            raise ValueError(f"{len(m_in)} matched controls planted, {matched} asked")
        m_in = m_in[:int(matched)]
    u_in = u_rows[member[u_rows]]
    rest_t = np.setdiff1d(np.concatenate([t_rows, x_rows]), t_in)
    rest_m = np.setdiff1d(np.concatenate([m_rows, mx_rows]), m_in)
    rest_u = np.setdiff1d(u_rows, u_in)
    held, giver, other, bg = (np.full((runs, n_items), np.nan) for _ in range(4))
    status = np.zeros((runs, n_items), np.int8)
    ht = [float(v) for v in hT["A"]] + [float(v) for v in hT["B"]]
    hm = None if hM is None else [float(v) for v in hM["A"]] + [float(v) for v in hM["B"]]
    hu = None if hU is None else [float(v) for v in hU["A"]] + [float(v) for v in hU["B"]]

    def split_of(side):
        if went is None:
            return SPLIT
        rest = 1.0 - went[side]["stay"]
        return tuple(went[side][key] / rest for key in ("neighbour", "other", "background"))

    for r in range(runs):
        nb, ot, bk = split_of("A" if r < per_side else "B")
        base_t = np.array([ht[r], (1 - ht[r]) * nb, (1 - ht[r]) * ot, (1 - ht[r]) * bk])
        h = 0.6 if hm is None else hm[r]
        base_m = np.array([h, (1 - h) * 0.1, (1 - h) * 0.3, (1 - h) * 0.6])
        h = 0.6 if hu is None else hu[r]
        base_u = np.array([h, 0.0, (1 - h) * 0.3, (1 - h) * 0.7])
        for rows_of, base in ((t_in, base_t), (rest_t, base_t), (m_in, base_m), (rest_m, base_m), (u_in, base_u),
                              (rest_u, base_u)):
            values = _spread(base, len(rows_of), rng, scale)
            held[r, rows_of], giver[r, rows_of], other[r, rows_of], bg[r, rows_of] = values.T
    # Left out: the extra suspects (each missing from one run, by its status); the members' matched controls beyond
    # `matched`, or all of them when hM is None; the uniform ones when hU is None (each missing from the last run).
    plan = []
    for place, code, count in drops:
        plan += [(place, code)] * count
    by_cell: dict[int, list] = {}
    for i in x_rows:
        by_cell.setdefault(int(items.cell[i]), []).append(int(i))
    for k, (place, code) in enumerate(plan):
        status[k % runs, by_cell[place].pop(0)] = code
    if hM is None:
        status[runs - 1, m_rows] = NOT_FOUND
    else:
        status[runs - 1, np.setdiff1d(m_rows[member[m_rows]], m_in)] = NOT_FOUND
    if hU is None:
        status[runs - 1, u_rows] = NOT_FOUND
    absent = np.isin(status, (NOT_FOUND, C0_ABSENT))
    for array in (held, giver, other, bg):
        array[absent] = np.nan
    giver[status == C1_ABSENT] = 0.0
    tx_runs = None if tx is None else list(tx["A"]) + list(tx["B"])
    reads = readouts(layout, held, giver, other, bg, status, tx=tx_runs, per_side=per_side, rows=rows,
                     seconds=seconds, errors=errors)
    args = {"items": items, "order": layout.order, "readouts": reads, "sides": roles(per_side), "start": start,
            "setting": setting or {"flag": "cell-compactness", "value": 0.02, "current": 0.04},
            "situation": situation, "at_iter": 450, "matched_counts": matched_counts(unsure, not_found),
            "parent_defaults": parent_defaults, "constants": constants}
    if row is not None:
        args["row"] = tuple(row)
    if caveats:
        args["caveats"] = [dict(c) for c in caveats]
    return args


# ---- the result fixture -----------------------------------------------------------------------------------------------
def sizes_of(head, total, count, seed, k=160, low=6):
    """`count` member sizes adding to `total`, starting with `head`, the last one taking the pool past k."""
    rng = np.random.default_rng(seed)
    last = max(low, total - (k - 1))
    middle = total - sum(head) - last
    parts = count - len(head) - 1
    cuts = np.sort(rng.choice(np.arange(1, middle), size=parts - 1, replace=False))
    sizes = list(head) + [int(v) for v in np.diff(np.concatenate([[0], cuts, [middle]]))] + [int(last)]
    assert sum(sizes) == total and len(sizes) == count and min(sizes[len(head):]) >= 1
    return sizes


def distances_of(count, first, seed, step=(3.0, 7.0)):
    rng = np.random.default_rng(seed)
    out, d = [0.0, first], first
    for _ in range(count - 2):
        d += rng.uniform(*step)
        out.append(round(d, 1))
    return out[:count]


def _times(total: float, runs: int, seed: int) -> list:
    """Per-run read times adding up to `total` (s)."""
    w = np.random.default_rng(seed).uniform(0.8, 1.2, runs)
    return list(total * w / w.sum())


def _example(name, names, *, sizes, reserve=(), first_um, seed, read_s, err, **kw) -> dict:
    """One example: planted() with the given shares, through answer(), its caveats then as engine/qc_test.py's _answer
    gives them (answer_caveats: the pool's, and too_few with the suspects judged when there are fewer than k).
    `names`: the made-up ids still unused (each example names its own cells)."""
    count = len(sizes) + len(reserve)
    distances = distances_of(count, first_um, seed)
    if reserve:          # the reserve sits a few µm beyond the last member
        distances = distances[:len(sizes)] + [round(distances[len(sizes) - 1] + 4.0 * (i + 1), 1)
                                              for i in range(len(reserve))]
    constants = qc_rule.constants_for(*kw.get("row", (kw.get("start", "paired"), kw.get("situation", "as_is"))))
    runs = 2 * constants["runs_per_side"]
    errors = list(np.random.default_rng(seed).uniform(0.6, 1.0, runs) * err)
    args = planted(sizes=sizes, reserve=reserve, distances=distances, rng_seed=seed,
                   seconds=_times(read_s, runs, seed), errors=errors, names=[next(names) for _ in range(2 * count)],
                   **kw)
    result = qc_rule.answer(**args)
    result["caveats"] = answer_caveats(result["caveats"], result["pool"]["suspects"], constants["k"])
    return {"name": name, "result": result}


def fixture_examples() -> list:
    """The canonical results the page's tests draw: one per verdict and reason, both situations, both rules, shrinking on
    and off, one whose parent ran with settings other than proseg's defaults, a pool under k judged all the same (its
    too_few caveat), and one where the measurements did not cover the test (judged with the nearest measured row, its
    caveats saying so)."""
    compactness = {"flag": "cell-compactness", "value": 0.02, "current": 0.04}
    names = iter(NAMES)
    return [
        # not_clear, why_not "apart": the headline example (paired, square).
        _example("not_clear_apart", names, start="paired", situation="square", setting=compactness,
                 sizes=sizes_of([9, 14], 187, 23, 1), reserve=[6, 8], first_um=11.2, seed=1, seed_cell=1234,
                 # B3's own RNA a little lower, so that the excess runs overlap clearly (not by a tie of the last bit).
                 hT={"A": [0.55, 0.57, 0.59], "B": [0.51, 0.52, 0.53]}, hM={"A": [0.60, 0.61, 0.62], "B": [0.57, 0.58, 0.578]},
                 hU={"A": [0.62, 0.63, 0.64], "B": [0.59, 0.60, 0.61]},
                 tx={"A": [312.5, 314.0, 316.2], "B": [266.0, 268.0, 270.4]},
                 went={"A": {"stay": 0.57, "neighbour": 0.06, "other": 0.12, "background": 0.25},
                       "B": {"stay": 0.52, "neighbour": 0.05, "other": 0.10, "background": 0.33}},
                 matched=540, unsure=2, read_s=1.9, err=4.8e-13),
        # not_clear, why_not "gap": separated, the gap no wider than tau (cells only, as is, 4 + 4; far drift).
        _example("not_clear_gap", names, start="cells", situation="as_is",
                 setting={"flag": "diffusion-probability", "value": 0.3, "current": 0.2},
                 sizes=sizes_of([12], 172, 15, 2), reserve=[7], first_um=8.4, seed=2, seed_cell=802,
                 hT={"A": [0.60, 0.61, 0.62, 0.63], "B": [0.55, 0.56, 0.57, 0.58]},
                 hM={"A": [0.66, 0.66, 0.67, 0.67], "B": [0.66, 0.67, 0.67, 0.68]},
                 hU={"A": [0.69, 0.70, 0.70, 0.71], "B": [0.68, 0.69, 0.69, 0.70]},
                 tx={"A": [205.0, 207.5, 208.0, 210.5], "B": [204.0, 205.5, 206.0, 207.0]},
                 went={"A": {"stay": 0.615, "neighbour": 0.07, "other": 0.10, "background": 0.215},
                       "B": {"stay": 0.565, "neighbour": 0.07, "other": 0.11, "background": 0.255}},
                 matched=498, unsure=1, rows=190234, read_s=1.4, err=3.1e-13),
        # not_clear with a negative gap: the setting raised the suspects' share (paired, as is; hold).
        _example("not_clear_negative_gap", names, start="paired", situation="as_is",
                 setting={"flag": "prior-weight", "value": 2.0, "current": 1.0},
                 sizes=sizes_of([10], 166, 18, 3), reserve=[5], first_um=9.6, seed=3, seed_cell=2210,
                 hT={"A": [0.58, 0.60, 0.62], "B": [0.66, 0.68, 0.70]}, hM={"A": [0.64, 0.65, 0.66], "B": [0.69, 0.70, 0.71]},
                 hU={"A": [0.65, 0.66, 0.67], "B": [0.69, 0.70, 0.71]},
                 tx={"A": [330.0, 332.4, 335.0], "B": [352.0, 355.1, 360.0]},
                 went={"A": {"stay": 0.60, "neighbour": 0.08, "other": 0.10, "background": 0.22},
                       "B": {"stay": 0.68, "neighbour": 0.06, "other": 0.08, "background": 0.18}},
                 matched=482, not_found=1, rows=402118, read_s=1.6, err=5.2e-13),
        # clearly_fewer (excess), not by shrinking; the parent ran with settings other than proseg's defaults.
        _example("clearly_fewer", names, start="paired", situation="square", parent_defaults=False,
                 setting={"flag": "cell-compactness", "value": 0.02, "current": 0.08},
                 sizes=sizes_of([11], 175, 20, 4), reserve=[9, 4], first_um=7.9, seed=4, seed_cell=1540,
                 hT={"A": [0.60, 0.62, 0.64], "B": [0.43, 0.45, 0.47]}, hM={"A": [0.69, 0.70, 0.71], "B": [0.67, 0.68, 0.69]},
                 hU={"A": [0.70, 0.71, 0.72], "B": [0.68, 0.69, 0.70]},
                 tx={"A": [318.0, 321.5, 323.0], "B": [301.0, 303.2, 306.0]},
                 went={"A": {"stay": 0.62, "neighbour": 0.07, "other": 0.09, "background": 0.22},
                       "B": {"stay": 0.45, "neighbour": 0.10, "other": 0.13, "background": 0.32}},
                 matched=508, unsure=1, rows=528902, read_s=2.0, err=4.1e-13),
        # clearly_fewer (excess) with shrinking: the card says both in one sentence (cells only, as is).
        _example("clearly_fewer_shrinking", names, start="cells", situation="as_is", setting=compactness,
                 sizes=sizes_of([8], 181, 17, 5), reserve=[5, 6], first_um=6.8, seed=5, seed_cell=611,
                 hT={"A": [0.58, 0.60, 0.61, 0.63], "B": [0.34, 0.36, 0.37, 0.39]},
                 hM={"A": [0.69, 0.70, 0.70, 0.71], "B": [0.59, 0.60, 0.60, 0.61]},
                 hU={"A": [0.71, 0.72, 0.72, 0.73], "B": [0.63, 0.64, 0.64, 0.65]},
                 tx={"A": [214.0, 216.5, 217.0, 219.0], "B": [171.0, 173.5, 174.0, 176.5]},
                 went={"A": {"stay": 0.605, "neighbour": 0.08, "other": 0.10, "background": 0.215},
                       "B": {"stay": 0.365, "neighbour": 0.12, "other": 0.15, "background": 0.365}},
                 drops=[(3, CROSSCHECK_FAILED, 1)], matched=530, rows=187455, read_s=1.5, err=2.7e-13),
        # clearly_fewer under rule "suspects" (the suspects alone), with shrinking; the seed holds 1 suspect.
        _example("clearly_fewer_suspects_rule", names, start="paired", situation="square", rule="suspects",
                 setting={"flag": "prior-weight", "value": 0.5, "current": 1.0},
                 sizes=sizes_of([1], 170, 19, 6), reserve=[7], first_um=10.3, seed=6, seed_cell=1977,
                 hT={"A": [0.58, 0.60, 0.62], "B": [0.43, 0.45, 0.47]}, hM={"A": [0.69, 0.70, 0.71], "B": [0.61, 0.62, 0.63]},
                 hU={"A": [0.69, 0.70, 0.71], "B": [0.62, 0.63, 0.64]},
                 tx={"A": [309.0, 311.0, 314.4], "B": [270.5, 272.0, 275.0]},
                 went={"A": {"stay": 0.60, "neighbour": 0.07, "other": 0.11, "background": 0.22},
                       "B": {"stay": 0.45, "neighbour": 0.09, "other": 0.14, "background": 0.32}},
                 matched=497, unsure=3, rows=531077, read_s=1.8, err=6.0e-13),
        # not_clear (excess) while the suspects-alone form fires: only the excess verdict is drawn; shrinking.
        _example("not_clear_suspects_fire", names, start="paired", situation="square",
                 setting={"flag": "diffusion-probability", "value": 0.1, "current": 0.2},
                 sizes=sizes_of([13], 178, 19, 7), reserve=[4, 5], first_um=8.8, seed=7, seed_cell=1402,
                 hT={"A": [0.60, 0.62, 0.64], "B": [0.48, 0.50, 0.52]}, hM={"A": [0.69, 0.70, 0.71], "B": [0.61, 0.62, 0.63]},
                 hU={"A": [0.70, 0.71, 0.72], "B": [0.63, 0.64, 0.65]},
                 tx={"A": [316.0, 318.0, 321.0], "B": [281.0, 284.5, 286.0]},
                 went={"A": {"stay": 0.62, "neighbour": 0.07, "other": 0.09, "background": 0.22},
                       "B": {"stay": 0.50, "neighbour": 0.08, "other": 0.12, "background": 0.30}},
                 drops=[(2, C1_ABSENT, 1), (5, C0_ABSENT, 1)], matched=521, rows=533640, read_s=1.9, err=4.4e-13),
        # comes_and_goes: the gate failed (cells only, as is).
        _example("comes_and_goes", names, start="cells", situation="as_is",
                 setting={"flag": "cell-compactness", "value": 0.06, "current": 0.04},
                 sizes=sizes_of([10], 164, 16, 8), reserve=[3], first_um=7.1, seed=8, seed_cell=345,
                 hT={"A": [0.38, 0.42, 0.45, 0.47], "B": [0.36, 0.40, 0.41, 0.45]},
                 hM={"A": [0.55, 0.57, 0.58, 0.60], "B": [0.54, 0.56, 0.57, 0.59]},
                 hU={"A": [0.61, 0.62, 0.62, 0.63], "B": [0.60, 0.61, 0.61, 0.62]},
                 tx={"A": [198.0, 200.5, 201.0, 203.0], "B": [205.0, 207.0, 208.5, 210.0]},
                 went={"A": {"stay": 0.43, "neighbour": 0.12, "other": 0.15, "background": 0.30},
                       "B": {"stay": 0.405, "neighbour": 0.12, "other": 0.155, "background": 0.32}},
                 matched=471, unsure=1, rows=192880, read_s=1.3, err=2.2e-13),
        # not_clear on a pool that never reached k once the runs were read, judged all the same, with the too_few caveat
        # engine/qc_test.py adds for the suspects judged; paired, as is.
        _example("not_clear_short_pool", names, start="paired", situation="as_is", setting=compactness,
                 sizes=[7, 9, 12, 8, 6, 11, 10, 9, 7, 8, 12, 6, 9, 7], first_um=9.9, seed=9, seed_cell=2671,
                 # B3's own RNA a little lower, so that the excess runs overlap clearly (not by a tie of the last bit).
                 hT={"A": [0.56, 0.58, 0.60], "B": [0.52, 0.53, 0.55]}, hM={"A": [0.62, 0.63, 0.64], "B": [0.60, 0.61, 0.60]},
                 hU={"A": [0.64, 0.65, 0.66], "B": [0.62, 0.63, 0.64]},
                 tx={"A": [329.0, 331.0, 333.5], "B": [301.0, 304.0, 306.5]},
                 drops=[(1, C0_ABSENT, 20), (4, C1_ABSENT, 13), (6, NOT_FOUND, 2)], matched=338, unsure=4,
                 rows=398765, read_s=1.5, err=5.0e-13),
        # too_few, cell_lost: the clicked cell's suspects were not read in every run (paired, square).
        _example("too_few_cell_lost", names, start="paired", situation="square",
                 setting={"flag": "prior-weight", "value": 2.0, "current": 1.0},
                 sizes=sizes_of([0], 188, 22, 10), first_um=12.5, seed=10, seed_cell=903,
                 hT={"A": [0.55, 0.57, 0.58], "B": [0.50, 0.52, 0.53]}, hM={"A": [0.61, 0.62, 0.63], "B": [0.58, 0.59, 0.60]},
                 hU={"A": [0.63, 0.64, 0.65], "B": [0.60, 0.61, 0.62]},
                 tx={"A": [310.0, 312.0, 315.0], "B": [281.0, 283.0, 286.0]},
                 drops=[(0, C0_ABSENT, 9)], rows=534412, read_s=1.7, err=4.6e-13),
        # too_few, no_controls: no matched own RNA was read in every run, so the excess rule cannot judge (cells only).
        _example("too_few_no_controls", names, start="cells", situation="as_is", setting=compactness,
                 sizes=sizes_of([9], 168, 16, 11), reserve=[6], first_um=8.1, seed=11, seed_cell=1088,
                 hT={"A": [0.57, 0.59, 0.60, 0.62], "B": [0.50, 0.52, 0.53, 0.55]}, hM=None,
                 hU={"A": [0.65, 0.66, 0.66, 0.67], "B": [0.61, 0.62, 0.62, 0.63]},
                 tx={"A": [210.0, 212.0, 213.5, 215.0], "B": [196.0, 198.5, 199.0, 201.0]},
                 unsure=2, rows=189932, read_s=1.2, err=3.3e-13),
        # not_clear where the measurements did not cover the test: a whole-sample run started from nuclei only, the
        # sample narrower than 500 µm one way (the area a crop), judged with the nearest measured row (paired, square:
        # qc_rule.row_for) and carrying the run's caveats (engine/qc_pool.py run_reasons, in the page's order).
        _example("not_clear_unmeasured", names, start="nuclei", situation="crop", row=("paired", "square"),
                 caveats=[qc_pool.reason(code) for code in ("whole_sample", "nuclei_start", "area_narrow")],
                 setting=compactness, sizes=sizes_of([8, 12], 176, 21, 12), reserve=[5, 7], first_um=9.4, seed=12,
                 seed_cell=1717,
                 # As not_clear_apart: B3's own RNA a little lower, so that the excess runs overlap clearly.
                 hT={"A": [0.55, 0.57, 0.59], "B": [0.51, 0.52, 0.53]}, hM={"A": [0.60, 0.61, 0.62], "B": [0.57, 0.58, 0.578]},
                 hU={"A": [0.62, 0.63, 0.64], "B": [0.59, 0.60, 0.61]},
                 tx={"A": [305.0, 307.5, 309.0], "B": [262.0, 264.5, 266.0]},
                 went={"A": {"stay": 0.57, "neighbour": 0.07, "other": 0.11, "background": 0.25},
                       "B": {"stay": 0.52, "neighbour": 0.06, "other": 0.10, "background": 0.32}},
                 matched=500, unsure=1, rows=467210, read_s=1.8, err=4.5e-13),
    ]


# What each example must say: (verdict, why, fewer.why_not, cost.shrinking).
EXPECT = {"not_clear_apart": ("not_clear", None, "apart", None), "not_clear_gap": ("not_clear", None, "gap", None),
          "not_clear_negative_gap": ("not_clear", None, "apart", None),
          "clearly_fewer": ("clearly_fewer", None, None, False),
          "clearly_fewer_shrinking": ("clearly_fewer", None, None, True),
          "clearly_fewer_suspects_rule": ("clearly_fewer", None, None, True),
          "not_clear_suspects_fire": ("not_clear", None, "gap", True),
          "comes_and_goes": ("comes_and_goes", None, None, None),
          "not_clear_short_pool": ("not_clear", None, "apart", None),
          "too_few_cell_lost": ("too_few", "cell_lost", None, None),
          "too_few_no_controls": ("too_few", "no_controls", None, None),
          "not_clear_unmeasured": ("not_clear", None, "apart", None)}


def said(result: dict) -> tuple:
    return (result["verdict"], result["why"], (result["fewer"] or {}).get("why_not"),
            (result["cost"] or {}).get("shrinking"))


def fixture_text(examples: list) -> str:
    """The fixture file's text: one example per line."""
    return "[\n" + ",\n".join("  " + json.dumps(e, ensure_ascii=False, allow_nan=False) for e in examples) + "\n]\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(f"usage: python tests/synth_pink.py {FIXTURE}", file=sys.stderr)
        return 2
    examples = fixture_examples()
    for example in examples:
        got = said(example["result"])
        if got != EXPECT[example["name"]]:
            print(f"{example['name']}: says {got}, not {EXPECT[example['name']]}", file=sys.stderr)
            return 1
    text = fixture_text(examples)
    Path(argv[0]).write_text(text, encoding="utf-8")
    print(f"{len(examples)} examples, {len(text)} bytes: {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
