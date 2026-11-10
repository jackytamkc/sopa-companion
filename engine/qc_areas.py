"""The mixing check's flagged areas, with calibrated rules (each mode's "areas" constants, engine/qc_constants.py). Pure
numpy: the engine's environment has no scipy.

Per live result, the GPU worker's per-bin sums (analysis/worker.py `qc_tables`, the frame's bins file) become dense
(ny, nx) arrays (`dense`), and `Tracker.update`:
  * smooths every per-bin sum: weight 1 - 0.5 ** (dt / 1 s) on the newest live result (dt on the run clock); a sum
    the result lacks keeps its smoothed value (the one live frame scored before the start line has no paired sums,
    and the tags skip it);
  * sums 3 x 3 bins into windows and tests each window: enter (J >= 40 (30 detail), M >= 5, share >= max(1.5 P,
    P + 4 points), Wilson 90 % lower bound > P, top 10 % of tissue windows by extra mixed cells M - P J) on >= 2
    results in a row spanning >= 2 s; clear (share < max(1.25 P, P + 2 points) or J < 30 (20)) on >= 3 results in a
    row spanning >= 10 s; P = the run's smoothed mixed / judged;
  * merges flagged windows into areas (the 8-connected union of their bins), drops areas with < 300 (120) judged
    cells ("smaller spots"), keeps at most 8 (4) drawn: the top by extra mixed cells, but a drawn area gives its place
    only to one with >= 2 x its extra (the sticky cap); letters stay while an area overlaps its previous bins
    (Jaccard >= 0.2), and a released letter rests 30 s;
  * tags each area against the start: "rose" ((n10 - n01) / sqrt(n10 + n01) >= 2 and its paired share rose >= 3
    points; and, when the bins carry the depth-matched sums, the area's OWN matched change counts: it holds >= 300
    judged and >= 400 paired cells and the change is at least its own size-aware threshold), "already"
    (a window of the area passes the enter test on the start's own bins against its level P0, and the area has not
    risen) or "similar". The start is the frame's own (its start sums J0, M0: the cells proseg started with, their RNA
    kept as proseg keeps it at this frame, smoothed as every sum), else the start line's bins a Tracker was given (the
    start line's own areas, `start_areas`); without either an area is "similar";
  * gives each area its own comparison with the start (`local_change`: its paired cells' depth-matched change, its own
    threshold, whether it has the minimum support for a verdict, and the direction) and says whether it touches the
    occupied tissue's edge (`edge_mask`: a bin with cells next to a bin without any, or the grid's border; edge windows
    look more mixed by themselves: in a test, half again as mixed as inside, at the start).
`snapshot` judges one result alone (a scrubbed or backfilled frame: no smoothing, no hysteresis; letters by rank, which
QcManager replaces with the letters the page showed there), and `start_areas` the start line's own bins. `outline`
traces an area's bins (<= 64 points, µm; all drawn areas of a result together <= OUTLINE_BUDGET points, so a summary
with 8 areas stays <= 8 KB), `evidence` sums what the card shows, `overview_grid` codes the overview layer (each bin's
3 x 3 window share and its own judged cells). `Tracker.state` / `Tracker.from_state` keep a tracker across an engine
restart, compact enough for state.json.

It reproduces a reference implementation of these rules result for result (the same windows, areas, letters and tags
on the same inputs; tests/test_qc_areas.py). Additions: `warm` (a tracker that
starts after the areas are shown, e.g. a check switched on late, decides nothing until it has had the enter rule's 2
results over 2 s: QcManager shows no areas, or the ones the page showed last, until then), `carried` (the numbers of
areas the page showed, on this result's sums), `reveal` (when the areas are first shown, they keep the letters of the
areas the page showed until then where they overlap, else take the first free letters by rank) and the sums a result
lacks (above).
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import heapq
import math
import threading

import numpy as np

from .qc_rules import direction_of, local_threshold, supported

# The calibrated area rules are the run's mode's "areas" constants (engine/qc_constants.py, read by engine/qc.py when it
# makes a Tracker: 100 µm bins for whole samples and tiled regions, 50 µm for dense regions). A Tracker's params may
# also carry the run's "mode" (its areas' own minimum support and thresholds: qc_rules.supported, local_threshold) and
# "edge_border" (edge_mask).
FROM_ITERS = 20                 # no areas until burn-in + 20 iterations (build.areas.draw_from): the start line's until then
OUTLINE_POINTS = 64             # at most this many points per outline ...
OUTLINE_BUDGET = 352            # ... and all drawn areas of a result together at most this many: 44 each with 8 drawn
                                # (100 µm grid), 64 each with 4 (50 µm grid), so a whole sample's summary stays <= 8 KB
OUTLINE_TOLERANCE = 2.0         # thinning drops a point spanning less than this x (bin / 4)² µm² with its neighbours: the
                                # outline stays within about a quarter bin of the rounded one (<= 26 µm)
OUTLINE_VERSION = 3             # part of every shape id: outlines drawn another way never reuse a stored one
EDGE_FRACTION = 0.25            # an outline leaves out an area's sparse edge bins: under this x its typical bin's cells
                                # (a bin with a few cells at the tissue's edge is 100 µm of mostly empty background)
OVERVIEW_PLANES = 2             # the overview grid: each bin's window share, then its own judged cells (overview_grid)
SHARE_SCALE, NOT_JUDGED = 250, 255     # the per-bin share's byte coding (as the per-cell bytes: floor(share x 250))
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# The worker's per-bin fields (analysis/worker.py BIN_COUNTS, BIN_SUMS) under the reference's names, and the evidence.
# J0, M0: the frame's own start, judged and looking mixed in its start block at the block's own depth (the cells proseg
# started with, their RNA kept as proseg keeps it at this frame), what each area's tags and start share pair with.
FIELDS = {"J": "judged", "M": "mixed", "B": "paired", "N1": "paired_mixed_now", "S1": "paired_mixed_start",
          "n10": "n10", "n01": "n01", "J0": "judged_start", "M0": "mixed_start"}
CORE = tuple(FIELDS)
EVIDENCE = ("cells", "tiny", "tiny_start", "lost_half", "growth_n", "tx", "tx_start", "changed", "assigned",
            "foreground")
# The worker's like-for-like per-bin sums (deterministic: no draws, each cell 0 or 1): the cells compared now and in
# the frame's start block at the matched depth, and of them those that look mixed now and at the start. An area's own
# matched change is 100 x (now - start) / paired over its bins.
MATCHED = ("matched_paired", "matched_mixed_now", "matched_mixed_start")
GROWTH_EDGES = (0.25, 0.5, 0.67, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)   # worker.GROWTH_EDGES (bins files
                                                                                     # carry their own)


def outline_points(params: dict) -> int:
    """The most points one drawn area's outline gets with these constants (OUTLINE_BUDGET over the cap)."""
    return max(4, min(OUTLINE_POINTS, OUTLINE_BUDGET // max(1, int(params["cap"]))))


def grid_of(table: dict) -> dict | None:
    """The grid {x0, y0, bin, nx, ny} of a bins file (its "grid" entry), None when it has none."""
    try:
        x0, y0, size, nx, ny = (float(v) for v in np.asarray(table["grid"], np.float64).ravel()[:5])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and size > 0 and nx >= 1 and ny >= 1):
        return None
    return {"x0": x0, "y0": y0, "bin": size, "nx": int(nx), "ny": int(ny)}


def dense(table: dict, grid: dict) -> dict:
    """A bins table (only the bins holding a cell: "bin" ids, row-major by * nx + bx) as dense (ny, nx) float64 arrays:
    the reference's J, M, B, N1, S1, n10, n01, the evidence sums and growth_hist as gh0..gh12 (only the fields the
    table has), plus "growth_edges" when the table carries them."""
    ny, nx = int(grid["ny"]), int(grid["nx"])
    ids = np.asarray(table.get("bin", ()), np.int64).ravel()
    ok = (ids >= 0) & (ids < nx * ny)
    out: dict = {}

    def put(name, values):
        values = np.asarray(values, np.float64).ravel()
        if values.shape[0] != ids.shape[0]:
            return
        flat = np.zeros(nx * ny, np.float64)
        flat[ids[ok]] = values[ok]
        out[name] = flat.reshape(ny, nx)
    for name, field in FIELDS.items():
        if field in table:
            put(name, table[field])
    for name in EVIDENCE + MATCHED:
        if name in table:
            put(name, table[name])
    hist = table.get("growth_hist")
    if hist is not None:
        hist = np.asarray(hist)
        if hist.ndim == 2 and hist.shape[0] == ids.shape[0]:
            for k in range(hist.shape[1]):
                put(f"gh{k}", hist[:, k])
    if table.get("growth_edges") is not None:
        out["growth_edges"] = tuple(float(v) for v in np.asarray(table["growth_edges"]).ravel())
    return out


# ---- the reference's maths (qcareas.py), unchanged ------------------------------------------------------------------
def wilson_lower(m, n, z):
    with np.errstate(invalid="ignore", divide="ignore"):
        p = m / n
        z2 = z * z
        centre = p + z2 / (2 * n)
        rad = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
        return np.where(n > 0, (centre - rad) / (1 + z2 / n), 0.0)


def box_sum(a, w):
    """Sum over the w x w block centred on each cell of a 2-D array (zero outside)."""
    r = w // 2
    p = np.pad(a, r)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, Wd = a.shape
    return c[w:w + H, w:w + Wd] - c[0:H, w:w + Wd] - c[w:w + H, 0:Wd] + c[0:H, 0:Wd]


def enter_test(Jw, Mw, P, p: dict, tissue):
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(Jw > 0, Mw / Jw, 0.0)
    ok = (Jw >= p["enter_min_j"]) & (Mw >= p["enter_min_m"]) & (s >= max(p["enter_mult"] * P, P + p["enter_add"]))
    ok &= wilson_lower(Mw, Jw, p["wilson_z"]) > P
    extra = Mw - P * Jw
    vals = extra[tissue]
    if vals.size:
        k = max(1, int(round(vals.size * p["top_frac"])))
        thr = np.partition(vals, vals.size - k)[vals.size - k]
        ok &= extra >= thr
    else:
        ok &= False
    return ok, s, extra


def dilate(flag: np.ndarray, w: int) -> np.ndarray:
    """Binary dilation by a w x w square (scipy.ndimage.binary_dilation with that structure; outside is False)."""
    r = w // 2
    if not r:
        return flag.copy()
    ny, nx = flag.shape
    padded = np.pad(flag, r)
    out = np.zeros_like(flag)
    for dy in range(w):
        for dx in range(w):
            out |= padded[dy:dy + ny, dx:dx + nx]
    return out


def components(mask: np.ndarray) -> list[np.ndarray]:
    """The 8-connected components of a boolean (ny, nx) array as sorted flat index arrays, in the order of their first
    bin in row-major order (scipy.ndimage.label's numbering with a 3 x 3 structure)."""
    ny, nx = mask.shape
    width = nx + 2
    padded = np.pad(mask, 1).ravel().tolist()
    offsets = (-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1)
    seen = [False] * len(padded)
    out = []
    for start in np.flatnonzero(mask).tolist():
        y, x = divmod(start, nx)
        first = (y + 1) * width + x + 1
        if seen[first]:
            continue
        seen[first] = True
        stack, members = [first], []
        while stack:
            cell = stack.pop()
            members.append(cell)
            for offset in offsets:
                other = cell + offset
                if padded[other] and not seen[other]:
                    seen[other] = True
                    stack.append(other)
        members = np.asarray(members, np.int64)
        out.append(np.sort((members // width - 1) * nx + members % width - 1))
    return out


@dataclass
class Area:
    letter: str
    bins: np.ndarray            # flat bin ids (sorted)
    windows: np.ndarray         # flat window-centre ids
    judged: float = 0.0
    per100: float = 0.0
    start_per100: float = float("nan")
    extra: float = 0.0
    tag: str = "similar"
    edge: bool = False          # touches the occupied tissue's edge (edge_mask)
    local: dict = field(default_factory=dict)   # its own comparison with the start (local_change)


def edge_mask(sm: dict, shape: tuple, border: bool = True) -> np.ndarray:
    """(ny, nx) bool: the occupied bins (holding at least half a cell: "cells", else judged) that touch an unoccupied
    bin among their 8 neighbours or, with `border`, the grid's border (outside the grid counts as unoccupied: a whole
    sample's grid ends where its tissue does). Cells there have fewer neighbours in the run, and their windows look more
    mixed by themselves. A region cut out of a sample (a box) passes border=False: its grid's border is a cut through
    tissue, not the tissue's edge (Tracker params "edge_border")."""
    cells = sm.get("cells", sm.get("J"))
    if cells is None or np.shape(cells) != tuple(shape):
        return np.zeros(shape, bool)
    occupied = np.asarray(cells, np.float64) >= 0.5
    empty = np.pad(~occupied, 1, constant_values=bool(border))
    ny, nx = occupied.shape
    near = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            near |= empty[dy:dy + ny, dx:dx + nx]
    return occupied & near


def local_change(members: np.ndarray, sm: dict, mode: str = "region") -> dict:
    """An area's own comparison with the start line, on the sums used for it (smoothed for live results), never the
    tissue's: {judged, paired, matched_change (100 x (mixed now - mixed at the start) / paired cells at the
    matched depth over its bins; the direction is decided on the unrounded value, which is kept to 2 decimals),
    threshold (its own size-aware formula: qc_rules.local_threshold at its share now and its paired cells), enough
    (>= 300 judged and >= 400 paired cells: the minimum support for a verdict of its own), direction ("fewer" | "more" |
    "same", None without enough support)}, with the constants of the run's `mode` (engine/qc_constants.py). Without
    the depth-matched sums (a run with no start line; the start line's own areas) paired, matched_change, threshold and
    direction are None."""
    idx = np.asarray(members, np.int64)

    def total(name):
        values = sm.get(name)
        return None if values is None else float(np.asarray(values).ravel()[idx].sum())
    judged = total("J") or 0.0
    out = {"judged": int(round(judged)), "paired": None, "matched_change": None, "threshold": None, "enough": False,
           "direction": None}
    paired, now, start = (total(name) for name in MATCHED)
    if paired is None or now is None or start is None:
        return out
    out["paired"] = int(round(paired))
    if paired <= 0:
        return out
    change = 100.0 * (now - start) / paired
    threshold = local_threshold(now / paired, paired, mode)
    enough = supported(judged, paired, mode)
    out.update(matched_change=round(change, 2), threshold=None if threshold is None else round(threshold, 2),
               enough=bool(enough), direction=direction_of(change, threshold) if enough else None)
    return out


def start_windows(start: dict, p: dict):
    """(start_flag, P0): the start's own windows passing the enter test against its level, no smoothing."""
    J0, M0 = box_sum(start["J"], p["window"]), box_sum(start["M"], p["window"])
    P0 = start["M"].sum() / max(1.0, start["J"].sum())
    ok, _, _ = enter_test(J0, M0, P0, p, J0 >= p["tissue_min_j"])
    return ok, P0


class Tracker:
    """One run's areas over its live results (qcareas.Tracker; module docstring). `grid` {x0, y0, bin, nx, ny},
    `params` a mode's "areas" constants (engine/qc_constants.py; with the run's "mode" and "edge_border"), `start` the
    start line's dense bins (J, M) to pair with when a result carries no start sums of its own (J0, M0), or None
    (`set_start`)."""

    def __init__(self, grid: dict, params: dict, start: dict | None = None):
        self.grid, self.p = dict(grid), dict(params)
        self.mode = str(self.p.get("mode") or "region")
        shape = (int(grid["ny"]), int(grid["nx"]))
        self.shape = shape
        self.sm: dict | None = None         # smoothed bin sums
        self.enter_streak = np.zeros(shape, np.int32)
        self.fail_streak = np.zeros(shape, np.int32)
        self.flag = np.zeros(shape, bool)
        self.enter_since = np.full(shape, np.nan)
        self.fail_since = np.full(shape, np.nan)
        self.t_prev: float | None = None
        self.t_first: float | None = None   # the run clock of the first live result
        self.n_live = 0                     # live results so far
        self.areas: list[Area] = []         # the drawn areas of the last live result, ranked
        self.rest: dict[str, float] = {}    # letter -> time it may be reused
        self.revealed = False               # the areas have been shown (QcManager: from burn-in + 20 iterations)
        self.start = self.start_flag = self.P0 = None
        self.set_start(start)

    @property
    def cold(self) -> bool:
        """No live result yet."""
        return self.sm is None

    @property
    def warm(self) -> bool:
        """The enter rule could have decided: at least enter_results live results spanning enter_s seconds (a tracker
        that started late, e.g. a check switched on after burn-in + 20, decides nothing before that: a window passing
        the enter test on one result only is never drawn)."""
        p = self.p
        return self.n_live >= p["enter_results"] and self.t_first is not None and self.t_prev is not None \
            and self.t_prev - self.t_first >= p["enter_s"] - 1e-9

    def set_start(self, start: dict | None):
        if start is not None and all(k in start and np.shape(start[k]) == self.shape for k in ("J", "M")):
            self.start = start
            self.start_flag, self.P0 = start_windows(start, self.p)
        else:
            self.start = self.start_flag = self.P0 = None

    def update(self, bins: dict, t: float, live: bool = True):
        """One result -> (every area found, the drawn ones first with their letters; P; the bin sums used). `live`:
        smoothed into the run's history (else judged alone: a single snapshot, nothing here changes). A window enters
        only by the enter rule, also on a tracker's first results (see `warm`)."""
        p = self.p
        edges = bins.get("growth_edges")
        bins = {k: v for k, v in bins.items() if isinstance(v, np.ndarray) and v.shape == self.shape}
        if self.sm is None or not live:
            sm = {k: v.copy() for k, v in bins.items()}
        else:
            a = p["alpha"]
            if p["ref_dt"] and self.t_prev is not None:
                a = 1.0 - (1.0 - p["alpha"]) ** (max(t - self.t_prev, 1e-6) / p["ref_dt"])
            sm = dict(self.sm)              # a sum this result lacks keeps its smoothed value
            for k, v in bins.items():
                old = self.sm.get(k)
                sm[k] = v.copy() if old is None else a * v + (1 - a) * old
        if edges is not None:
            sm["growth_edges"] = edges
        if live:
            self.sm = sm
            self.t_prev = t
            self.n_live += 1
            if self.t_first is None:
                self.t_first = t
        J = sm.get("J", np.zeros(self.shape))
        M = sm.get("M", np.zeros(self.shape))
        P = M.sum() / max(1.0, J.sum())
        Jw, Mw = box_sum(J, p["window"]), box_sum(M, p["window"])
        tissue = Jw >= p["tissue_min_j"]
        ok, s, extra = enter_test(Jw, Mw, P, p, tissue)
        stay_fail = (s < max(p["stay_mult"] * P, P + p["stay_add"])) | (Jw < p["stay_min_j"])
        if live:
            self.enter_streak = np.where(ok, self.enter_streak + 1, 0)
            self.fail_streak = np.where(stay_fail, self.fail_streak + 1, 0)
            self.enter_since = np.where(ok, np.where(np.isnan(self.enter_since), t, self.enter_since), np.nan)
            self.fail_since = np.where(stay_fail, np.where(np.isnan(self.fail_since), t, self.fail_since), np.nan)
            with np.errstate(invalid="ignore"):
                newly = (~self.flag) & (self.enter_streak >= p["enter_results"]) & (t - self.enter_since >= p["enter_s"])
                cleared = self.flag & (self.fail_streak >= p["stay_results"]) & (t - self.fail_since >= p["stay_s"])
            self.flag = (self.flag | newly) & ~cleared
            flag = self.flag
        else:
            flag = ok          # a single snapshot: judged without history
        return self._areas(flag, sm, P, t, live) + (sm,)

    def snapshot(self, bins: dict):
        """One result judged alone (scrubbed or backfilled frames): (areas, P, bins); nothing here changes."""
        return self.update(bins, 0.0, live=False)

    def reveal(self, previous=()):
        """The areas are shown for the first time (QcManager: at burn-in + 20 iterations). Until then the page showed
        `previous`, [(letter, bins)]: the start line's own areas. A drawn area that overlaps one of them (Jaccard of
        bins >= letter_jaccard, best pairs first, one to one) keeps its letter; the others get the first free letters
        by rank; a previous letter no area kept rests letter_rest_s (it is not reused for another place at once)."""
        t = 0.0 if self.t_prev is None else self.t_prev
        before = [(str(letter), set(np.asarray(list(bins), np.int64).tolist())) for letter, bins in previous]
        pairs = []
        for i, area in enumerate(self.areas):
            mine = set(area.bins.tolist())
            for j, (_, theirs) in enumerate(before):
                jac = len(mine & theirs) / max(1, len(mine | theirs))
                if jac >= self.p["letter_jaccard"]:
                    pairs.append((jac, -i, -j))
        pairs.sort(reverse=True)
        kept, taken = {}, set()
        for _, i, j in pairs:
            if -i in kept or -j in taken:
                continue
            kept[-i] = -j
            taken.add(-j)
        self.rest = {letter: t + self.p["letter_rest_s"] for j, (letter, _) in enumerate(before) if j not in taken}
        busy = {before[j][0] for j in taken}
        for i, area in enumerate(self.areas):
            if i in kept:
                area.letter = before[kept[i]][0]
                continue
            area.letter = next((L for L in LETTERS if L not in busy and self.rest.get(L, -1) <= t), "?")
            busy.add(area.letter)
        self.revealed = True

    def _start_of(self, sm: dict):
        """What a result's areas are paired with: (J0, M0, start_flag), flat, of its own start sums (J0, M0: the cells
        proseg started with, at this frame's keep) when `sm` carries them, else of the start line's bins this tracker
        was given (`start`), else None (no start: every area is "similar")."""
        J0, M0 = sm.get("J0"), sm.get("M0")
        if isinstance(J0, np.ndarray) and isinstance(M0, np.ndarray) and J0.shape == M0.shape == self.shape:
            flag, _ = start_windows({"J": J0, "M": M0}, self.p)
            return J0.ravel(), M0.ravel(), flag.ravel()
        if self.start is not None:
            return self.start["J"].ravel(), self.start["M"].ravel(), self.start_flag.ravel()
        return None

    def _area(self, members: np.ndarray, sm: dict, P: float, start, flat_flag=None, letter: str = "",
              edges: np.ndarray | None = None) -> Area:
        """One area over bins `members` (sorted flat ids) on the sums `sm`: its judged cells, share, extra mixed cells
        over the level P, its own comparison with the start (local_change), whether it touches the tissue's edge
        (`edges`, edge_mask of `sm`) and its tag against the start (`start`: `_start_of`). "rose" needs the paired
        cells' rise (the reference's test) and, when `sm` carries the depth-matched sums, the area's own matched change
        counting with the minimum support (local_change direction "more"): a rise that comes from depth alone, or one
        too few cells can back, is no rise."""
        p = self.p
        J, M = sm.get("J", np.zeros(self.shape)).ravel(), sm.get("M", np.zeros(self.shape)).ravel()
        Jm, Mm = J[members].sum(), M[members].sum()
        windows = members[flat_flag[members]] if flat_flag is not None else np.zeros(0, np.int64)
        area = Area(letter=letter, bins=members, windows=windows, judged=float(Jm),
                    per100=100.0 * Mm / Jm if Jm else 0.0, extra=float(Mm - P * Jm))
        if edges is None:
            edges = edge_mask(sm, self.shape, self.p.get("edge_border", True))
        area.edge = bool(edges.ravel()[members].any())
        area.local = local_change(members, sm, self.mode)
        if start is not None:
            J0, M0, start_flag = start
            J0, M0 = J0[members].sum(), M0[members].sum()
            area.start_per100 = 100.0 * M0 / J0 if J0 else float("nan")
            total = {k: (sm[k].ravel()[members].sum() if k in sm else 0.0) for k in ("n10", "n01", "B", "N1", "S1")}
            n10, n01 = total["n10"], total["n01"]
            z = (n10 - n01) / math.sqrt(n10 + n01) if n10 + n01 > 0 else 0.0
            rose = total["B"] > 0 and z >= p["rose_z"] and (total["N1"] - total["S1"]) / total["B"] >= p["rose_points"]
            if rose and area.local.get("paired") is not None:
                rose = area.local.get("direction") == "more"
            already = bool(start_flag[members].any()) and not rose
            area.tag = "rose" if rose else ("already" if already else "similar")
        return area

    def carried(self, previous, sm: dict | None = None) -> list[Area]:
        """Areas the page showed (`previous`: [(letter, bins)]) with this result's numbers and tags (on `sm`, else the
        smoothed sums): what is shown while a tracker that started late is not `warm` yet, or for a live result
        without per-bin sums, instead of blanking the areas on screen. Nothing here changes."""
        sm = self.sm if sm is None else sm
        if not previous or sm is None:
            return []
        J, M = sm.get("J"), sm.get("M")
        P = 0.0 if J is None or M is None else float(M.sum() / max(1.0, J.sum()))
        nbins = self.shape[0] * self.shape[1]
        edges = edge_mask(sm, self.shape, self.p.get("edge_border", True))
        start = self._start_of(sm)
        out = []
        for letter, bins in previous:
            members = np.unique(np.asarray(list(bins), np.int64))
            members = members[(members >= 0) & (members < nbins)]
            if members.size:
                out.append(self._area(members, sm, P, start, letter=str(letter), edges=edges))
        return out

    def _areas(self, flag, sm, P, t, live):
        p = self.p
        cover = dilate(flag, p["window"])
        flat_flag = flag.ravel()
        edges = edge_mask(sm, self.shape, self.p.get("edge_border", True))
        start = self._start_of(sm)
        found = [self._area(members, sm, P, start, flat_flag, edges=edges) for members in components(cover)]
        found.sort(key=lambda a: -a.extra)
        small = [a for a in found if a.judged < p["min_area_j"]]
        found = [a for a in found if a.judged >= p["min_area_j"]]
        if live:
            top, rest = self._select(found, t)
            rest = rest + small
        else:
            top, rest = found[:p["cap"]], found[p["cap"]:] + small
            for i, a in enumerate(top):
                a.letter = LETTERS[i % 26]
        return top + rest, P

    def _match(self, found):
        """Greedy one-to-one match of found areas to the previously drawn ones (Jaccard of bins >= letter_jaccard)."""
        prev = self.areas
        pairs = []
        for i, a in enumerate(found):
            sa = set(a.bins.tolist())
            for j, b in enumerate(prev):
                sb = set(b.bins.tolist())
                jac = len(sa & sb) / max(1, len(sa | sb))
                if jac >= self.p["letter_jaccard"]:
                    pairs.append((jac, i, j))
        pairs.sort(reverse=True)
        match, taken_j = {}, set()
        for jac, i, j in pairs:
            if i in match or j in taken_j:
                continue
            match[i] = j
            taken_j.add(j)
        return match

    def _select(self, found, t):
        """The drawn areas: the top `cap` by extra mixed cells, except that a drawn area keeps its place until an
        undrawn one has cap_swap_ratio times its extra (no churn between near-equal areas at the cap's edge)."""
        p = self.p
        match = self._match(found)
        for i, j in match.items():
            found[i].letter = self.areas[j].letter
        inc = [i for i in range(len(found)) if i in match]
        chal = [i for i in range(len(found)) if i not in match]
        shown = sorted(inc, key=lambda i: -found[i].extra)[:p["cap"]]
        dropped_inc = [i for i in inc if i not in shown]
        for i in sorted(chal, key=lambda i: -found[i].extra):
            if len(shown) < p["cap"]:
                shown.append(i)
                continue
            weakest = min(shown, key=lambda k: found[k].extra)
            if found[i].extra >= p["cap_swap_ratio"] * max(found[weakest].extra, 1e-9):
                shown.remove(weakest)
                shown.append(i)
                if weakest in inc:
                    dropped_inc.append(weakest)
        shown_set = set(shown)
        for i in range(len(found)):
            if i not in shown_set:
                if found[i].letter:
                    self.rest[found[i].letter] = t + p["letter_rest_s"]
                found[i].letter = ""
        for j, b in enumerate(self.areas):
            if j not in match.values():
                self.rest[b.letter] = t + p["letter_rest_s"]
        busy = {found[i].letter for i in shown if found[i].letter}
        for i in sorted(shown, key=lambda i: -found[i].extra):
            if not found[i].letter:
                for L in LETTERS:
                    if L not in busy and self.rest.get(L, -1) <= t:
                        found[i].letter = L
                        busy.add(L)
                        break
                else:
                    found[i].letter = "?"
        top = sorted([found[i] for i in shown], key=lambda a: -a.extra)
        rest = [found[i] for i in range(len(found)) if i not in shown_set]
        self.areas = top
        return top, rest

    # ---- persistence (QcManager's state.json "areas") -------------------------------------------------------------
    def state(self) -> dict:
        """What resumes the tracker after an engine restart, JSON-ready and compact (a whole sample's is ~30 KB): the
        smoothed core sums (float32, zstd, base64; the evidence sums start again from the next result); the windows'
        hysteresis where it can still decide anything: the unflagged windows passing the enter test now (their run
        of passing results and since when) and the flagged ones failing the stay test (theirs). A window's other count
        never matters: one is flagged only on a result that passes the enter test, which also passes the stay test,
        and cleared only on one that fails both, so the count it starts then is 0 either way. Then the drawn areas
        (letter, extra mixed cells, bins) and the resting letters."""
        flag, nbins = self.flag.ravel(), self.shape[0] * self.shape[1]
        entering = np.flatnonzero(~flag & (self.enter_streak.ravel() > 0))
        clearing = np.flatnonzero(flag & (self.fail_streak.ravel() > 0))
        sums = {} if self.sm is None else {k: self.sm[k] for k in CORE if k in self.sm}
        return {"grid": [self.grid[k] for k in ("x0", "y0", "bin", "nx", "ny")],
                "t_prev": self.t_prev, "t_first": self.t_first, "n_live": self.n_live, "revealed": self.revealed,
                "bins": nbins,
                "sm": _pack(sums), "flag": np.flatnonzero(flag).tolist(),
                "entering": [[int(i), int(self.enter_streak.flat[i]), float(self.enter_since.flat[i])] for i in entering],
                "clearing": [[int(i), int(self.fail_streak.flat[i]), float(self.fail_since.flat[i])] for i in clearing],
                "drawn": [{"L": a.letter, "extra": float(a.extra), "bins": a.bins.tolist()} for a in self.areas],
                "rest": {str(k): float(v) for k, v in self.rest.items()}}

    @classmethod
    def from_state(cls, stored: dict, params: dict, start: dict | None = None) -> "Tracker | None":
        """A tracker from `state` (None when it does not fit together)."""
        try:
            grid = grid_of({"grid": stored["grid"]})
            if grid is None or int(stored["bins"]) != grid["nx"] * grid["ny"]:
                return None
            tracker = cls(grid, params, start)
            nbins = grid["nx"] * grid["ny"]
            sums = _unpack(stored.get("sm"), nbins)
            tracker.sm = {k: v.reshape(tracker.shape) for k, v in sums.items()} or None
            flag = np.zeros(nbins, bool)
            flag[_ids(stored["flag"], nbins)] = True
            tracker.flag = flag.reshape(tracker.shape)
            for name, streak, since in (("entering", tracker.enter_streak, tracker.enter_since),
                                        ("clearing", tracker.fail_streak, tracker.fail_since)):
                for i, count, when in stored[name]:
                    if not 0 <= int(i) < nbins:
                        return None
                    streak.flat[int(i)], since.flat[int(i)] = int(count), float(when)
            t_prev = stored.get("t_prev")
            tracker.t_prev = float(t_prev) if isinstance(t_prev, (int, float)) and math.isfinite(t_prev) else None
            # A state from before `warm` (no counts): a tracker with smoothed sums had its results.
            t_first, n_live = stored.get("t_first"), stored.get("n_live")
            if isinstance(n_live, int) and not isinstance(n_live, bool) and n_live >= 0:
                tracker.n_live = n_live
                tracker.t_first = float(t_first) if isinstance(t_first, (int, float)) and math.isfinite(t_first) \
                    else None
            elif tracker.sm is not None and tracker.t_prev is not None:
                tracker.n_live, tracker.t_first = int(params["enter_results"]), tracker.t_prev - float(params["enter_s"])
            tracker.revealed = bool(stored["revealed"])
            tracker.areas = [Area(letter=str(a["L"]), bins=_ids(a["bins"], nbins), windows=np.zeros(0, np.int64),
                                  extra=float(a["extra"])) for a in stored["drawn"]]
            tracker.rest = {str(k): float(v) for k, v in (stored.get("rest") or {}).items()}
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return None
        return tracker


def _ids(values, nbins: int) -> np.ndarray:
    """Flat bin ids from JSON (sorted, within the grid; raises ValueError otherwise)."""
    ids = np.asarray(values, np.int64).ravel()
    if ids.size and (ids.min() < 0 or ids.max() >= nbins):
        raise ValueError("bin id outside the grid")
    return np.sort(ids)


def _pack(arrays: dict) -> dict:
    """Same-shaped arrays as {keys, data: base64 of zstd of their float32 values, in the keys' order}."""
    import zstandard
    keys = list(arrays)
    raw = b"".join(np.ascontiguousarray(arrays[k], "<f4").tobytes() for k in keys)
    return {"keys": keys, "data": base64.b64encode(zstandard.ZstdCompressor(level=3).compress(raw)).decode("ascii")}


def _unpack(packed, nbins: int) -> dict:
    """{key: float64 array of nbins} from _pack (an empty dict for nothing; ValueError when the sizes do not fit)."""
    if not packed or not packed.get("keys"):
        return {}
    import zstandard
    try:
        values = np.frombuffer(zstandard.ZstdDecompressor().decompress(base64.b64decode(packed["data"])), "<f4")
    except Exception as exc:                        # a damaged file (zstd's own error is no ValueError)
        raise ValueError(f"the smoothed sums cannot be read: {exc}") from None
    if values.size != nbins * len(packed["keys"]):
        raise ValueError("the smoothed sums do not fit the grid")
    return {str(k): values[i * nbins:(i + 1) * nbins].astype(np.float64) for i, k in enumerate(packed["keys"])}


def start_areas(grid: dict, params: dict, start: dict):
    """The start line's own areas (where the cells proseg starts from already look mixed): its windows passing the
    enter test against its own level, no smoothing, all tagged "already", letters by rank. (areas, P0)."""
    tracker = Tracker(grid, params, start)
    if tracker.start is None:
        return [], None
    found, P0 = tracker._areas(tracker.start_flag, start, tracker.P0, 0.0, live=False)
    for area in found:
        area.tag = "already"
        area.start_per100 = area.per100
    return found, P0


# ---- outlines -------------------------------------------------------------------------------------------------------
_OUTLINES: OrderedDict = OrderedDict()      # (grid, bins bytes) -> (shape id, outline): a drawn area keeps its bins
_OUTLINES_KEPT = 1024
_OUTLINES_LOCK = threading.Lock()


def _one_piece(ids: np.ndarray, grid: dict) -> bool:
    """Whether bins `ids` are one 8-connected piece."""
    mask = np.zeros(int(grid["ny"]) * int(grid["nx"]), bool)
    mask[ids] = True
    return len(components(mask.reshape(int(grid["ny"]), int(grid["nx"])))) == 1


def drawn_bins(bins: np.ndarray, sm: dict, grid: dict) -> np.ndarray:
    """The bins an area's outline is drawn around: its bins that hold cells (an area is the union of 3 x 3 windows,
    which reaches into empty background at the tissue's edge, and background is never flagged), and of those not the
    sparse edge ones (under EDGE_FRACTION x the cells of the area's typical bin: a 100 µm square with a few cells in a
    corner would draw the outline through empty background), as long as what is left stays one piece; otherwise the
    bins holding cells, or all of them when leaving the empty ones out would split the area."""
    bins = np.asarray(bins, np.int64)
    cells = sm.get("cells", sm.get("J"))
    if cells is None:
        return bins
    values = cells.ravel()[bins]
    held = values >= 0.5
    kept = bins[held]
    if not kept.size:
        return bins
    typical = float(np.median(values[held]))
    dense_ = bins[values >= max(0.5, EDGE_FRACTION * typical)]
    if dense_.size and dense_.size < kept.size and _one_piece(dense_, grid):
        return dense_
    if kept.size == bins.size:
        return kept
    return kept if _one_piece(kept, grid) else bins


def shape_id(bins: np.ndarray, points: int = OUTLINE_POINTS) -> str:
    """A short id for an area's bin set, outline size and the way outlines are drawn (OUTLINE_VERSION): the same id,
    the same outline (stored once per run)."""
    salt = bytes([OUTLINE_VERSION]) + int(points).to_bytes(2, "little")
    return hashlib.blake2b(np.asarray(bins, "<i4").tobytes() + salt, digest_size=6).hexdigest()


def outline(bins: np.ndarray, grid: dict, points: int = OUTLINE_POINTS) -> tuple[str, tuple]:
    """(shape id, outline) of an area: its bins' outer boundary (8-connected: bins touching at a corner are one
    shape; holes are left out), cut at the corners twice (Chaikin, at most half a bin deep) and thinned (points within
    about a quarter bin of the rounded line dropped, OUTLINE_TOLERANCE) to <= `points` points (outline_points), as
    ((x, y), ...) whole µm. Cached by bin set; the id names the bins and the point count."""
    bins = np.asarray(bins, np.int64)
    points = max(4, min(int(points), OUTLINE_POINTS))
    key = (tuple(grid[k] for k in ("x0", "y0", "bin", "nx", "ny")), points, bins.tobytes())
    with _OUTLINES_LOCK:
        found = _OUTLINES.get(key)
        if found is not None:
            _OUTLINES.move_to_end(key)
            return found
    value = (shape_id(bins, points), _outline(bins, grid, points))
    with _OUTLINES_LOCK:
        _OUTLINES[key] = value
        while len(_OUTLINES) > _OUTLINES_KEPT:
            _OUTLINES.popitem(last=False)
    return value


def _outline(bins: np.ndarray, grid: dict, points: int = OUTLINE_POINTS) -> tuple:
    nx, size = int(grid["nx"]), float(grid["bin"])
    cells = {(int(b) % nx, int(b) // nx) for b in bins.tolist()}
    if not cells:
        return ()
    loop = _outer_loop(cells)
    loop = [(grid["x0"] + size * x, grid["y0"] + size * y) for x, y in loop]
    for cut in (0.5 * size, 0.25 * size):            # two corner-cutting passes, never deeper than half a bin
        loop = _chaikin(loop, cut)
    loop = _thin(loop, points, OUTLINE_TOLERANCE * (size / 4) ** 2)
    out, last = [], None
    for x, y in loop:
        point = (int(round(x)), int(round(y)))
        if point != last:
            out.append(point)
            last = point
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return tuple(out)


# Directions in counter-clockwise order (x right, y up): E, N, W, S.
_STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _outer_loop(cells: set) -> list:
    """The outer boundary of a set of unit squares (x, y) as grid vertices without collinear points: every square's
    edges with the inside on the left, an edge shared by two squares dropped, then loops followed turning right
    first (squares touching at a corner stay in one loop); the loop enclosing the largest area."""
    edges: dict = {}
    for x, y in cells:
        if (x, y - 1) not in cells:
            edges.setdefault((x, y), []).append(0)
        if (x + 1, y) not in cells:
            edges.setdefault((x + 1, y), []).append(1)
        if (x, y + 1) not in cells:
            edges.setdefault((x + 1, y + 1), []).append(2)
        if (x - 1, y) not in cells:
            edges.setdefault((x, y + 1), []).append(3)
    used = {v: [False] * len(ds) for v, ds in edges.items()}
    loops = []
    starts = sorted(edges, key=lambda v: (v[1], v[0]))
    for single in (True, False):                     # loops start where one edge leaves (never at a pinch point)
        for vertex in starts:
            for k, direction in enumerate(edges[vertex]):
                if used[vertex][k] or (single and len(edges[vertex]) != 1):
                    continue
                used[vertex][k] = True
                points, d, v = [vertex], direction, vertex
                v = (v[0] + _STEPS[d][0], v[1] + _STEPS[d][1])
                guard = 0
                while v != vertex and guard < 4 * len(cells) + 8:
                    guard += 1
                    options = edges.get(v, [])
                    choice = None
                    for turn in (-1, 0, 1):          # right, straight, left
                        wanted = (d + turn) % 4
                        for j, od in enumerate(options):
                            if od == wanted and not used[v][j]:
                                choice = j
                                break
                        if choice is not None:
                            break
                    if choice is None:
                        break
                    used[v][choice] = True
                    points.append(v)
                    d = options[choice]
                    v = (v[0] + _STEPS[d][0], v[1] + _STEPS[d][1])
                loops.append(points)
    best, best_area = [], 0.0
    for points in loops:
        area = _signed_area(points)
        if area > best_area:
            best, best_area = points, area
    # Only corners: drop vertices where the boundary goes straight on.
    corners = []
    n = len(best)
    for i in range(n):
        a, b, c = best[i - 1], best[i], best[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            corners.append(b)
    return corners


def _signed_area(points) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2


def _chaikin(points: list, cut: float) -> list:
    """One corner-cutting pass over a closed polygon (Chaikin: each side cut at a quarter of its length from both ends),
    at most `cut` from a corner: a long straight side stays where its bins end and only the corners round off."""
    out = []
    n = len(points)
    for i in range(n):
        (x0, y0), (x1, y1) = points[i], points[(i + 1) % n]
        length = math.hypot(x1 - x0, y1 - y0)
        f = min(0.25, cut / length) if length > 0 else 0.25
        out.append(((1 - f) * x0 + f * x1, (1 - f) * y0 + f * y1))
        out.append((f * x0 + (1 - f) * x1, f * y0 + (1 - f) * y1))
    return out


def _thin(points: list, keep: int, min_area: float) -> list:
    """Visvalingam-Whyatt on a closed polygon: drop the point spanning the smallest triangle with its neighbours while
    that triangle is under `min_area` (µm²) or there are more than `keep` points; never below 3."""
    n = len(points)
    if n <= 3:
        return list(points)
    prev, nxt = [(i - 1) % n for i in range(n)], [(i + 1) % n for i in range(n)]
    alive = [True] * n

    def tri(i):
        (ax, ay), (bx, by), (cx, cy) = points[prev[i]], points[i], points[nxt[i]]
        return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
    version = [0] * n
    heap = [(tri(i), i, 0) for i in range(n)]
    heapq.heapify(heap)
    count = n
    while heap and count > 3:
        area, i, ver = heapq.heappop(heap)
        if not alive[i] or ver != version[i]:
            continue
        if area >= min_area and count <= keep:
            break
        alive[i] = False
        count -= 1
        a, c = prev[i], nxt[i]
        nxt[a], prev[c] = c, a
        for j in (a, c):
            version[j] += 1
            heapq.heappush(heap, (tri(j), j, version[j]))
    return [points[i] for i in range(n) if alive[i]]


# ---- evidence, overview grid ----------------------------------------------------------------------------------------
def totals(sm: dict) -> dict:
    """The run's sums over all bins (for the evidence's "vs the run")."""
    return {k: float(v.sum()) for k, v in sm.items() if isinstance(v, np.ndarray)}


def _median_ratio(hist: list, edges) -> float | None:
    """The median of a histogram of current / starting areas over buckets split at `edges`, interpolated within its
    bucket on a log scale (the outer buckets end at 1/16 and 16)."""
    count = sum(hist)
    if count <= 0:
        return None
    bounds = [1 / 16, *edges, 16.0]
    half, seen = count / 2, 0.0
    for k, c in enumerate(hist):
        if c > 0 and seen + c >= half:
            lo, hi = bounds[k], bounds[k + 1]
            return math.exp(math.log(lo) + (half - seen) / c * (math.log(hi) - math.log(lo)))
        seen += c
    return None


def evidence(bins: np.ndarray, sm: dict, run: dict) -> dict:
    """What the card says about an area, from the bin sums used for it (smoothed for live results):
    cells; growth (the median current / starting area - 1, a fraction); tiny cells (< 10 µm²) now and at the start;
    shrunk (cells that lost more than half their starting area); churn (boundary changes per cell here / across the
    run); noise (1 - foreground / assigned: what proseg treats as noise here); depth and depth_start (transcripts per
    cell now and of the same cells at the start). None where the bins do not carry it."""
    idx = np.asarray(bins, np.int64)

    def total(name):
        values = sm.get(name)
        return None if values is None else float(values.ravel()[idx].sum())

    def whole(value):
        return None if value is None else int(round(value))
    cells = total("cells")
    out = {"cells": whole(cells), "growth": None, "tiny": whole(total("tiny")), "tiny_start": whole(total("tiny_start")),
           "shrunk": whole(total("lost_half")), "churn": None, "noise": None, "depth": None, "depth_start": None}
    hist = [total(f"gh{k}") for k in range(len(GROWTH_EDGES) + 1)]
    if all(v is not None for v in hist):
        median = _median_ratio(hist, sm.get("growth_edges") or GROWTH_EDGES)
        out["growth"] = None if median is None else round(median - 1.0, 3)
    changed, run_changed, run_cells = total("changed"), run.get("changed"), run.get("cells")
    if changed is not None and cells and run_changed and run_cells:
        out["churn"] = round((changed / cells) / (run_changed / run_cells), 2)
    assigned, foreground = total("assigned"), total("foreground")
    if assigned and foreground is not None:
        out["noise"] = round(1.0 - foreground / assigned, 3)
    tx, tx_start = total("tx"), total("tx_start")
    if cells:
        out["depth"] = None if tx is None else int(round(tx / cells))
        out["depth_start"] = None if tx_start is None else int(round(tx_start / cells))
    return out


def share_grid(sm: dict, shape: tuple) -> np.ndarray:
    """The per-bin share mixed / judged as one byte per bin, row-major (floor(share x 250); 255 where under half a
    judged cell)."""
    J, M = sm.get("J"), sm.get("M")
    code = np.full(shape, NOT_JUDGED, np.uint8)
    if J is None or M is None:
        return code.ravel()
    ok = J >= 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        code[ok] = np.clip(np.floor(M[ok] / J[ok] * SHARE_SCALE), 0, SHARE_SCALE).astype(np.uint8)
    return code.ravel()


def overview_grid(sm: dict, shape: tuple, window: int = 3) -> np.ndarray:
    """The overview layer's grid (GET /runs/{id}/qc/{seq}/bins, OVERVIEW_PLANES planes of one byte per bin, row-major):
    first the share of judged cells that look mixed in each bin's window x window bins, the windows areas are flagged
    on (sum M / sum J: a bin with a few cells weighs what it holds; floor(share x 250), 255 where the window holds under
    half a judged cell), then the bin's own judged cells (rounded, at most 254), which tells the page how full the bin
    is (sparse bins at the tissue's edge fade out instead of lighting up background)."""
    J, M = sm.get("J"), sm.get("M")
    share = np.full(shape, NOT_JUDGED, np.uint8)
    judged = np.zeros(shape, np.uint8)
    if J is not None and M is not None and np.shape(J) == tuple(shape) and np.shape(M) == tuple(shape):
        Jw, Mw = box_sum(np.asarray(J, np.float64), window), box_sum(np.asarray(M, np.float64), window)
        ok = Jw >= 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            share[ok] = np.clip(np.floor(Mw[ok] / Jw[ok] * SHARE_SCALE), 0, SHARE_SCALE).astype(np.uint8)
        judged[:] = np.clip(np.rint(np.asarray(J, np.float64)), 0, 254).astype(np.uint8)
    return np.concatenate([share.ravel(), judged.ravel()])
