"""The inputs of the shared golden fixture of Go deeper's scoring (frontend/test/fixtures/qc-scoring.json), which
engine/qc_scoring.py (the page's scoring in Python) and frontend/src/qc-deeper.js must both reproduce exactly.

Each case is {name, result (a worker answer), cell (its cell number, label - 1), reply (a where reply, or null)}; this
script writes them without `expect`, and frontend/test/make-qc-scoring.mjs then fills in {params, scores, kept} from
the page's own code. Three kinds of case:
  - "world": the numpy worker's own Go deeper answers on the synthetic run and model of tests/test_worker_deeper.py
    (test_analysis.write_run, make_deep; every state: cell 11, gone from the frame, holds none of proseg's count), and
    the where engine's replies to the page's request on that same frame
    (engine/deeper_where.py, version 3: proseg's count of the frame and its where export of the box, a fifth of the
    transcripts moved elsewhere, so each point's share w is exact and fractional; its points capped for the fixture's
    size: the page takes any of them);
  - "hand-made": the where engine's replies on tests/test_deeper_where.py's hand-made frame (detail: w from its 3-D
    labels; whole-sample: exact from an export with transcripts moved between the cell and its sources) and on its
    strip case (pink dots near none of the neighbour that would be named, and the same strip beside it), with
    answers written to match: every gene's word, near flags worked out by the engine;
  - edge cases, written by hand: ties, cut-offs and rounding at their exact values, w scaling P on any owner, the
    likely rule, the sources' support, the states, every reply the page refuses, the values the page coerces (a cell
    number given as text, null, a missing key) and whole numbers past 2^53 (added as doubles).
Seeded and deterministic (timings fixed). Not a test: pytest does not collect it. From the app's folder, with the tests'
PYTHONPATH (the folder that holds sopa_companion), then node (it writes only with --write):
  env -u DISPLAY python tests/make_qc_scoring_inputs.py
  env -u DISPLAY node frontend/test/make-qc-scoring.mjs --write"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FIXTURE = HERE.parent / "frontend/test/fixtures/qc-scoring.json"
POINTS_KEPT = 110          # a world reply's points at most (the cell's suspects first)
SEQ, CELL = 12, 5          # the edge cases' frame and cell


# ---- the stand-in for the engine's Runs that the where engine reads a run folder through -------------------------
class _Run:
    def __init__(self, folder: Path, meta: dict):
        self.live = folder / "live"
        self.meta = meta
        self.info = json.loads((self.live / "info.json").read_text())
        self.frames = [json.loads(line) for line in (self.live / "frames.jsonl").read_text().splitlines()]
        self.ack = None                                 # proseg's acknowledgement: not paused
        self.lock = threading.Lock()


class _Runs:
    def __init__(self):
        self.runs: dict[str, _Run] = {}

    def get(self, run_id: str) -> _Run:
        return self.runs[run_id]

    def frame_path(self, run_id: str, seq: int) -> Path:
        run = self.runs[run_id]
        return run.live / run.frames[seq]["file"]

    def transcript_index(self, run_id: str):
        from sopa_companion.engine.transcript_index import TranscriptIndex
        return TranscriptIndex(self.runs[run_id].live)


def asked(where, run_id: str, result: dict, cell: int) -> dict | None:
    """The where engine's reply to the request the page makes of `result` (its params, from the port: any request
    would do as an input), None without one."""
    from sopa_companion.engine import qc_scoring
    from sopa_companion.engine.deeper_where import givers_of
    req = qc_scoring.page_params(result, cell)["req"]
    if req is None:
        return None
    q = req["params"]
    reply = json.loads(where.reply(run_id, q["seq"], q["cell"], q["sources"], q["genes"], q["own"],
                                   givers=givers_of(result)))
    reply["ms"] = 2.5
    return reply


def capped(reply: dict, keep: int, seed: int) -> dict:
    """The reply with at most `keep` of its points, in their order: the points of its "from" genes proseg counts for
    the cell (the suspects; at most two thirds of them), then others drawn at random."""
    points = reply["points"]
    n = len(points["x"])
    if n <= keep:
        return reply
    giving = {g for g, role in enumerate(reply["roles"]) if role == "from"}
    first = [i for i in range(n) if points["w"][i] > 0 and points["gene"][i] in giving][:keep * 2 // 3]
    others = sorted(set(range(n)) - set(first))
    drawn = np.random.default_rng(seed).choice(others, size=min(len(others), keep - len(first)), replace=False)
    chosen = sorted(first + [int(i) for i in drawn])
    return {**reply, "points": {k: [v[i] for i in chosen] for k, v in points.items()}}


def fixed_times(result: dict) -> dict:
    out = {k: v for k, v in result.items() if k not in ("load_ms",)}
    out.update(ms=1.25, worker_ms=1.5)
    return out


KEEPS = (0.5, 0.625, 0.75, 0.875, 1.0)     # proseg's keep e / n of a (cell, gene): steps its file holds exactly
EMPTY_LABEL = 12                           # the world's cell gone from frame 2 on (test_analysis.frame_plane): empty


# ---- the numpy worker's answers on test_worker_deeper's run, and the where engine's replies ----------------------
def world_cases(tmp: Path) -> list[dict]:
    from sopa_companion.analysis import worker as W
    from sopa_companion.engine.deeper_where import DeeperWhere
    from test_analysis import JUDGE, NCELLS, RECORD, RUN_GENES, X0, Y0, frame_plane, write_run
    from test_deeper_where import attach, destinations, export_of, frame_counts
    import test_worker_deeper as TWD
    run = write_run(tmp / "world" / "run", seed=2)
    live = run / "live"
    (live / "cells.txt").write_text("".join(c + "\n" for c in TWD.RUN_IDS))
    model = TWD.make_deep(tmp / "world" / "model", run, seed=2)
    info = json.loads((live / "info.json").read_text())
    frame = json.loads((live / "frames.jsonl").read_text().splitlines()[2])
    # Proseg's count of frame 2: each transcript's destination the owner of its observed voxel, a fifth of them moved
    # anywhere (or to no cell), never into cell 11 (label 12, gone from frame 2 on: the world's empty cell); a keep per
    # cell and gene.
    grid = {key: frame[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
    records = np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORD)
    rng = np.random.default_rng(2)
    dest = destinations(records, np.stack([frame_plane(2)] * frame["nz"]), grid, (0.0, 0.5))
    moved = rng.random(len(records)) < 0.2
    dest[moved] = rng.integers(0, NCELLS + 1, int(moved.sum()))
    dest[dest == EMPTY_LABEL] = 0
    keep = rng.choice(KEEPS, (NCELLS, len(RUN_GENES)))
    first = frame_counts(live, 0, 0, np.stack([frame_plane(0)] * frame["nz"]), records, grid, (0.0, 0.5), NCELLS,
                         len(RUN_GENES))                    # the start's (proseg's count at iteration 0)
    line = frame_counts(live, 2, 2, None, records, None, None, NCELLS, len(RUN_GENES), keep, dest=dest)
    out = tmp / "world" / "qc"
    out.mkdir()
    common = {"run": "r1", "transcripts": str(live / "transcripts.bin.gz"), "genes": str(live / "genes.txt"),
              "ncells": NCELLS, "z_origin": 0.0, "z_spacing": 0.5}
    qc = {"model": str(model), "model_id": "feedface", "mixed_at": 0.3, "judge_min": JUDGE, "fit_min": 5,
          "grid": {"x0": X0, "y0": Y0, "bin": 10.0, "nx": 4, "ny": 3}}
    start = {"op": "qc_start", **common, "seq": None, "counts_file": str(live / first["file"]),
             "prior": {**info["prior"], "file": str(live / info["prior"]["file"])},
             "qc": {**qc, "out_cells": str(out / "start.bin"), "out_start": str(out / "start.npz"),
                    "out_bins": str(out / "start.bins.npz")}}
    request = {"op": "analyze", "ops": ["qc", "deeper"], **common, "seq": 2, "iter": 2,
               "frame": str(live / frame["file"]), "format": "dense",
               **{k: frame[k] for k in ("nx", "ny", "nz", "x0", "y0", "voxel")},
               "cells_file": str(live / frame["cells_file"]), "counts_file": str(live / line["file"]),
               "qc": {**qc, "start": str(out / "start.npz"), "out_cells": str(out / "000002.qc.bin"),
                      "out_bins": str(out / "000002.bins.npz")}}
    block = {"model": str(model), "model_id": "feedface", "start": False, "k": TWD.K, "fit_min": 5,
             "judge_min": JUDGE, "codes": str(out / "000002.qc.bin"), "cell_ids": str(live / "cells.txt"), "v": 1}
    worker = W.Worker(xp=np)
    assert worker.handle(start)["ok"]
    runs = _Runs()
    runs.runs["r1"] = _Run(run, {"id": "r1", "kind": "proseg", "state": "done",
                                 "prior": {"cells": {"kind": "shapes", "key": "cells"}}})
    exports = attach(runs)
    whole = [X0, Y0, X0 + frame["nx"] * frame["voxel"], Y0 + frame["ny"] * frame["voxel"]]
    exports.keep("r1", export_of(runs.runs["r1"], whole, records, dest, seq=2, iteration=2, grid=grid))
    where = DeeperWhere(runs)
    cases = []
    for c in range(NCELLS):
        answer = worker.handle({**request, "deeper": {**block, "cell": c}})
        assert answer["ok"], answer
        result = fixed_times(answer["deeper"])
        # Every answer for its rows and request; the where replies of four cells of different places: grown into its
        # neighbour (0), beside the one it narrowed (2), among four (6), beside the empty one (7).
        reply = asked(where, "r1", result, c) if c in (0, 2, 6, 7) else None
        cases.append({"name": f"world_cell_{c}_{result['state']}", "result": result, "cell": c,
                      "reply": None if reply is None else capped(reply, POINTS_KEPT, seed=c)})
    return cases


# ---- test_deeper_where's hand-made frames, with answers written to match them -------------------------------------
def hand_made_answer(seq: int, iter_: int) -> dict:
    """Cell A (0) of the hand-made frame: B (1) gives EDGE (its giver, row 0), D (3) gives SPREAD (row 1); LAYER and FEW
    from neighbours with no giver named; C (2) under 1 % (no row)."""
    return {"v": 1, "run": "r2", "seq": seq, "iter": iter_, "model": "feedface", "cell": 0, "name": "aaaaaaaa-1",
            "state": "ready", "transcripts": 68, "outside": 0, "code": 180, "parts": [0.4, 0.58, 0.02], "rough": False,
            "genes": [["OWN1", 20, 19.5, 0.5, 0], ["EDGE", 10, 2.0, 8.0, 1, 0], ["LAYER", 10, 5.0, 5.0, 1],
                      ["OWN2", 10, 9.8, 0.2, 0], ["SPREAD", 10, 4.0, 6.0, 1, 1], ["OTHER", 5, 5.0, 0.0, 0],
                      ["FEW", 3, 1.0, 2.0, 1]],
            "genes_total": 7, "kind": ["OWN1", "OWN2"], "own_genes": [["OWN1", 19.5, 20], ["OWN2", 9.8, 10],
                                                                     ["OTHER", 5.0, 5]],
            "same_kind_share": 0.0,
            "neighbours": [{"cell": 1, "name": "bbbbbbbb-1", "um": 10.0, "tx": 20.0, "genes": [["EDGE", 8.0]],
                            "kind": ["UNASKED"], "similar": 0.2, "same": False, "known": True},
                           {"cell": 3, "name": "dddddddd-1", "um": 12.5, "tx": 8.0, "genes": [["SPREAD", 6.0]],
                            "kind": ["UNASKED"], "similar": 0.3, "same": False, "known": True},
                           {"cell": 2, "name": "cccccccc-1", "um": 14.0, "tx": 0.5, "genes": [["LAYER", 0.4]],
                            "kind": ["OTHER"], "similar": 0.95, "same": True, "known": False}],
            "learned": "partly", "gap": 0.5, "ms": 1.25, "worker_ms": 1.5}


def hand_made_cases(tmp: Path) -> list[dict]:
    """The detail frame: proseg keeps 3/4 of A's EDGE and half its SPREAD (w from the 3-D labels x the keep); the
    whole-sample frame: exact from proseg's export, with three of A's EDGE at the shared edge moved to B and two of B's
    EDGE moved into A (they glow outside A's outline)."""
    from sopa_companion.engine.deeper_where import DeeperWhere
    from test_deeper_where import CELLS, GENES, attach, destinations, export_of, layout, merge_counts, placed, write_run
    runs, cases = _Runs(), []
    exports = attach(runs)
    where = DeeperWhere(runs)
    keep = np.ones((len(CELLS), len(GENES)))
    keep[0, GENES.index("EDGE")], keep[0, GENES.index("SPREAD")] = 0.75, 0.5
    for run_id, tiled in (("r2", False), ("r3", True)):
        layers = layout()[None] if tiled else np.stack([layout()] * 4)
        folder = write_run(tmp / "hand", run_id, layers, placed(), tiled=tiled, counts=None if tiled else keep,
                           window=not tiled)
        run = runs.runs[run_id] = _Run(folder, json.loads((folder / "run.json").read_text()))
        if tiled:
            from test_deeper_where import frame_counts
            grid = {key: run.frames[0][key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz")}
            records = placed()
            dest = destinations(records, layers, grid, (0.0, 1.0))
            is_edge = records["gene"] == GENES.index("EDGE")
            edge = np.flatnonzero(is_edge & (records["x"] < 120) & (records["x"] > 119))
            into = np.flatnonzero(is_edge & (records["x"] > 125) & (records["x"] < 126))
            dest[edge[:3]], dest[into[:2]] = 2, 1
            frame_counts(run.live, 0, 40, None, records, None, None, len(CELLS), len(GENES), keep, dest=dest)
            merge_counts(run)
            exports.keep(run_id, export_of(run, [100.0, 50.0, 150.0, 90.0], records, dest, token=1, grid=grid))
        else:
            merge_counts(run)
        result = {**hand_made_answer(0, 40), "run": run_id}
        cases.append({"name": f"hand_made_{'whole_sample' if tiled else 'detail'}", "result": result, "cell": 0,
                      "reply": asked(where, run_id, result, 0)})
    # The strip case (A2): the strip of GZF far from cell-f, and the same strip beside it.
    from test_deeper_where import F_CELLS, F_GENES, strip_case, strip_case_answer
    for run_id, adjacent in (("r4", False), ("r5", True)):
        layers, records = strip_case(adjacent)
        keep = np.ones((len(F_CELLS), len(F_GENES)))
        keep[0, 0] = 0.875
        folder = write_run(tmp / "hand", run_id, layers, records, counts=keep, genes=F_GENES, cells=F_CELLS)
        run = runs.runs[run_id] = _Run(folder, json.loads((folder / "run.json").read_text()))
        merge_counts(run)
        result = {**strip_case_answer(), "run": run_id}
        cases.append({"name": f"a2_strip_{'beside' if adjacent else 'far'}", "result": result, "cell": 0,
                      "reply": asked(where, run_id, result, 0)})
    return cases


# ---- hand-written edge cases ----------------------------------------------------------------------------------------
def answer(genes=(), neighbours=(), *, own_genes=(), cell=CELL, seq=SEQ, code=120, transcripts=200, **extra) -> dict:
    """A ready v 1 answer in the worker's shape, its numbers made up: genes [name, tx, own, nb, tag, giver?]."""
    out = {"v": 1, "run": "r", "seq": seq, "iter": seq if isinstance(seq, int) else None, "model": "m", "cell": cell,
           "state": "ready", "transcripts": transcripts, "outside": 0, "code": code, "parts": [0.5, 0.48, 0.02],
           "rough": False, "genes": [list(g) for g in genes], "genes_total": len(genes), "kind": [],
           "own_genes": [list(g) for g in own_genes], "same_kind_share": 0.0,
           "neighbours": [dict(n) if isinstance(n, dict) else n for n in neighbours], "learned": "yes", "ms": 1.25}
    out.update(extra)
    return out


def nb(cell, tx, **extra) -> dict:
    """A neighbour row of an answer (extra None: that key left out)."""
    out = {"cell": cell, "um": 8.0, "tx": tx, "genes": [], "kind": [], "similar": 0.1, "same": False, "known": True,
           **extra}
    return {k: v for k, v in out.items() if k not in extra or extra[k] is not None}


def stat(gene, giver=None, in_cell=4, near_giver=None, own_near=None, role="from", **extra) -> dict:
    """A where reply's gene_stats entry."""
    return {"gene": gene, "role": role, "giver": giver, "giver_by": None if giver is None else "result",
            "in_cell": in_cell, "near_giver": near_giver, "own_near": own_near,
            "density": {"cell": 4.0, "giver": None if giver is None else 2.0, "around": 0.5}, "word": None, **extra}


def where(genes, points, *, stats=(), cell=CELL, seq=SEQ, v=3, sources=(6,), near=True, **extra) -> dict:
    """A where reply: genes [(name, role)], points [(x, y, z, gene, owner, near, w?)] (z None: no depth; w, proseg's
    share of the point for the cell, by default all of it inside the cell's outline and none elsewhere); `near` False:
    without the points' near flags."""
    w = [p[6] if len(p) > 6 else (1.0 if p[4] == 1 else 0.0) for p in points]
    out = {"v": v, "run": "r", "seq": seq, "iter": seq if isinstance(seq, int) else 0, "cell": cell,
           "sources": list(sources), "basis": "voxel", "held": "exact",
           "held_seq": seq if isinstance(seq, int) else None, "near_um": 2.0, "box": [0.0, 0.0, 60.0, 60.0],
           "genes": [g for g, _ in genes], "roles": [r for _, r in genes],
           "cell_total": float(sum(w)), "cell_area_um2": 50.0, "source_area_um2": [40.0] * len(sources),
           "gene_stats": [dict(s) if isinstance(s, dict) else s for s in stats], "into": [], "truncated": False,
           "points": {"x": [p[0] for p in points], "y": [p[1] for p in points], "z": [p[2] for p in points],
                      "gene": [p[3] for p in points], "owner": [p[4] for p in points]}, "ms": 2.5}
    if near:
        out["points"]["near"] = [p[5] for p in points]
    out["points"]["w"] = w
    out.update(extra)
    return out


def case(name, result, reply=None, cell=CELL) -> dict:
    return {"name": name, "result": result, "cell": cell, "reply": reply}


# The scoring cases' common answer: one row (cell 6, a fifth of the RNA) and the genes below.
SOURCE = nb(6, 40.0)


def grid_points(gene: int, n: int, owner: int = 1, near: int = 0, x0: float = 10.0, z: float | None = 0.5) -> list:
    """n points of one gene in a row, 0.5 µm apart (no place shared)."""
    return [(round(x0 + 0.5 * k, 2), 20.0, z, gene, owner, near) for k in range(n)]


def chip_cases() -> list[dict]:
    """Chips: ties in nb by name (UTF-16 order), the floor, more than six, own genes among them, more than three own."""
    cases = []
    tied = [["GENE2", 10, 5.0, 5.0, 1], ["GENE1", 10, 5.0, 5.0, 1], ["G" + chr(0xFFE0), 8, 4.0, 4.0, 1],
            ["G" + chr(0x10000), 8, 4.0, 4.0, 1], ["GENE3", 10, 9.0, 1.0, 0]]
    cases.append(case("chips_tied_by_name_utf16", answer(tied, [SOURCE])))
    floor = [["HALF", 2, 0.5, 1.5, 1], ["UNDER", 3, 1.4, 1.6, 1], ["EPSILON", 6, 3.0000001, 2.9999999, 1],
             ["NONE", 4, 4.0, 0.0, 1], ["ROUNDED", 5, 2.5, 2.4999999, 1]]
    cases.append(case("chips_floor", answer(floor, [SOURCE])))
    many = [[f"GENE{k}", 20, 20 - k, k, 1] for k in range(10, 18)]
    own = [["GENE15", 4.0, 20], ["OWNA", 9.0, 9], ["GENE12", 3.0, 20], ["OWNB", 7.0, 7], ["OWNC", 6.0, 6],
           ["OWND", 5.0, 5], ["OWNA", 1.0, 1]]
    cases.append(case("chips_more_than_six_own_overlap", answer(many, [SOURCE], own_genes=own)))
    odd = [['Q"UOTE', 10, 2.0, 8.0, 1], ["BACK\\SLASH", 10, 3.0, 7.0, 1], ["TAB\tLF\n\x1f", 10, 4.0, 6.0, 1],
           ["LONE" + chr(0xD800), 10, 5.0, 5.0, 1], ["LS" + chr(0x2028) + "PS" + chr(0x2029), 10, 6.0, 4.0, 1],
           ["", 10, 5.0, 5.0, 1]]
    own = [[chr(0xE9) + "t" + chr(0xE9), 3.0, 3]]
    cases.append(case("chips_names_escaped_in_the_key", answer(odd, [SOURCE], own_genes=own)))
    flags = [["ONE", 10, 2.0, 8.0, 1], ["TRUE", 10, 2.0, 7.0, True], ["ONE_POINT_O", 10, 2.0, 6.0, 1.0],
             ["TEXT", 10, 2.0, 5.0, "1"], ["TWO", 10, 2.0, 9.0, 2], ["SHORT", 10, 2.0, 9.5],
             ["NO_TX", "10", 2.0, 9.9, 1], ["NEG_TX", -4, 2.0, 9.9, 1], ["NULL_NB", 10, 2.0, None, 1],
             [5, 10, 1, 9, 1], "ROW", ["GIVER", 10, 2.0, 4.0, 1, -1]]
    cases.append(case("chips_flag_and_number_coercion", answer(flags, [SOURCE])))
    return cases


def row_cases() -> list[dict]:
    """Rows: more than five, equal shares (in neighbour order), a share of exactly 0.01, the same kind from `similar`
    without the worker's own class (which wins when given, but not as text), known false; and the cell numbers as the
    page's lookups read them."""
    genes = [["GENE1", 10, 2.0, 8.0, 1]]
    more = [nb(20, 30.0), nb(21, 30.0), nb(22, 50.0), nb(23, 20.0), nb(24, 20.0), nb(25, 10.0), nb(26, 5.0)]
    exact = [nb(10, 2.0), nb(11, 1.9999), nb(12, 2.0000001)]          # 2 of 200 is exactly 0.01
    kinds = [nb(12, 30.0, similar=0.9, same=None), nb(13, 29.0, similar=0.89999, same=None),
             nb(15, 28.0, similar=True, same=None), nb(16, 27.0, similar="0.95", same=None), nb(17, 26.0, similar=0.95)]
    classes = [nb(18, 25.0, same=True), nb(19, 24.0, same="yes", similar=0.1), nb(14, 23.0, known=False),
               nb(27, 22.0, known=None), nb(28, 21.0, known="no")]
    # A cell number given as text, null, missing, a fraction, an array (its text); then a boolean, -1 (label 0: no
    # label), an object, and neighbours that are no objects.
    texts = [nb("7", 20.0), nb(None, 19.0), {k: v for k, v in nb(0, 18.0).items() if k != "cell"}, nb(3.5, 17.0),
             nb([5], 16.0)]
    others = [nb(True, 15.0), nb(-1, 14.0), nb({"x": 1}, 13.0), "not a neighbour", [1, 2], nb([], 12.0), nb(9, 11.0)]
    return [case("rows_more_than_five_and_ties", answer(genes, more)),
            case("rows_share_exactly_one_percent", answer(genes, exact)),
            case("rows_same_kind_from_similar", answer(genes, kinds)),
            case("rows_same_kind_classes_and_known", answer(genes, classes)),
            case("rows_cell_numbers_as_text_null_missing", answer(genes, texts, transcripts=400)),
            case("rows_cell_numbers_boolean_negative_object", answer(genes, others, transcripts=400)),
            case("rows_none_at_code_9", answer(genes, [SOURCE], code=9)),
            case("rows_shown_at_code_12", answer(genes, [SOURCE], code=12)),
            case("rows_no_transcripts", answer(genes, [SOURCE], transcripts=0)),
            case("rows_seq_start", answer(genes, [SOURCE], seq="start"),
                 where([("GENE1", "from")], grid_points(0, 3), stats=[stat("GENE1", 0, 3, 3, [1, 10])], seq="start")),
            case("rows_seq_as_a_fraction_free_float", answer(genes, [SOURCE], seq=12.0)),
            # Numbers in the request's key as JavaScript writes them (1e+21, 1.0000000005838672e-7, past 2^53).
            case("rows_numbers_written_as_javascript_does",
                 answer(genes, [nb(1e21, 30.0), nb(1e-7, 29.0), nb(1.2345678901234568e20, 28.0), nb(2 ** 53 + 2, 27.0),
                                nb(2 ** 60, 26.0)], seq=10 ** 21))]


def state_cases() -> list[dict]:
    """The states the page takes and those it refuses (no request either way)."""
    base = {"v": 1, "run": "r", "seq": SEQ, "iter": SEQ, "model": "m", "cell": CELL, "transcripts": 40, "outside": 2,
            "name": "c05"}
    reply = where([("GENE1", "from")], grid_points(0, 4), stats=[stat("GENE1")])
    return [
        case("state_too_few", {**base, "state": "too_few", "genes": [["GENE1", 30], ["GENE2", 10], ["BAD"], [7, 1]]},
             reply),
        case("state_empty", {**base, "state": "empty", "transcripts": 0}),
        case("state_not_checked", {**base, "state": "not_checked", "transcripts": 300, "reason": "no_neighbours",
                                   "genes": [["GENE1", 300]]}),
        case("state_v2_refused", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], v=2), reply),
        case("state_v_one_point_o", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], v=1.0)),
        case("state_unknown_refused", {**base, "state": "learning"}),
        case("state_code_out_of_range", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], code=251)),
        case("state_code_boolean", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], code=True)),
        case("state_seq_negative", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], seq=-1)),
        case("state_seq_text", answer([["GENE1", 10, 2.0, 8.0, 1]], [SOURCE], seq="12")),
        case("state_not_an_object", None, reply),
        case("state_an_array", [1, 2, 3]),
    ]


def scoring_cases() -> list[dict]:
    """P = w x pi: each branch of suspectScores at its edges, w scaling P on any owner, float32 before the cut-offs,
    toFixed's ties, the likely rule, the sources' support."""
    cases = []
    # NEAR: near and far split; UNPLACED: no giver (s); ZERO: s = 0; ALL: s = 1; CLAMPED: f_g past 1; WHOLE: s = 1 with
    # every one near its giver (far is 0 / 0: 0); an own gene, and a "from" gene the answer does not hold (no P). Each
    # gene's points: inside the cell's outline near and far (all of it proseg's, three quarters), one inside it that
    # proseg does not count for the cell (w 0), one inside the source that proseg moved into the cell (w 0.5), one in no
    # outline (w 0.25), one in another cell's (w 0).
    genes = [["NEAR", 10, 2.0, 8.0, 1, 0], ["UNPLACED", 10, 5.0, 5.0, 1], ["ZERO", 10, 10.0, 0.0, 0],
             ["ALL", 4, 0.0, 4.0, 1, 0], ["CLAMPED", 10, 3.0, 7.0, 1, 0], ["WHOLE", 5, 0.0, 5.0, 1, 0],
             ["OWNG", 20, 20.0, 0.0, 0]]
    names = [("NEAR", "from"), ("UNPLACED", "from"), ("ZERO", "from"), ("ALL", "from"), ("CLAMPED", "from"),
             ("WHOLE", "from"), ("OWNG", "own"), ("ABSENT", "from")]
    stats = [stat("NEAR", 0, 6, 4, [3, 40]), stat("UNPLACED", None, 5), stat("ZERO", 0, 4, 2, [3, 40]),
             stat("ALL", 0, 4, 1, [10, 40]), stat("CLAMPED", 0, 4, 9, [3, 40]), stat("WHOLE", 0, 3, 3, [3, 40]),
             stat("OWNG", role="own"), stat("ABSENT", 0, 3, 3, [1, 4])]
    points = []                                         # every gene's points: the cell's near and far, then others'
    for g in range(len(names)):
        for owner, near, w in ((1, 1, 1.0), (1, 0, 0.75), (1, 1, 0.0), (2, 1, 0.5), (0, 0, 0.25), (255, 0, 0.0)):
            points.append((10.0 + 0.5 * len(points), 30.0, 0.5, g, owner, near, w))
    reply = where(names, points, stats=stats)
    cases.append(case("scores_each_branch", answer(genes, [SOURCE], own_genes=[["OWNG", 20.0, 20]]), reply))
    cases.append(case("scores_without_near", answer(genes, [SOURCE]), where(names, points, stats=stats, near=False)))
    # Likely: pi >= 0.5 on a point proseg counts at least half of for the cell. GAVE (s = 0.9, no giver: pi 0.9 for
    # each) at w 1, 0.5 (P 0.45: likely), 0.4999 (P 0.45: glows, not likely), 0.25 (P 0.225), 0.2 (P 0.18: no glow),
    # 1e-9 and 0; HALF (s = 0.5) at w 1 (P and pi 0.5: likely) and 0.4 (P 0.2: glows as float32).
    genes = [["GAVE", 10, 1.0, 9.0, 1], ["HALF", 10, 5.0, 5.0, 1]]
    points = [(1.0 + k, 3.0, 0.5, 0, 1, 0, w) for k, w in enumerate((1.0, 0.5, 0.4999, 0.25, 0.2, 1e-9, 0.0))]
    points += [(1.0 + k, 5.0, 0.5, 1, 1, 0, w) for k, w in enumerate((1.0, 0.4))]
    cases.append(case("scores_w_and_the_likely_rule", answer(genes, [SOURCE]),
                      where([("GAVE", "from"), ("HALF", "from")], points,
                            stats=[stat("GAVE", None, 9), stat("HALF", None, 2)])))
    # The sources' support: X (giver 0) glows near it (w 0.5 and 0.75) and further in; Y (giver 1) glows far from it
    # only (its near flags 0), with one point near it too faint to glow (w 0.1); Z (no giver) near nothing; a glowing
    # point near X's giver that proseg does not count (w 0: no glow). Support: source 0 only.
    genes = [["X", 10, 2.0, 8.0, 1, 0], ["Y", 10, 1.0, 9.0, 1, 1], ["Z", 10, 2.0, 8.0, 1]]
    points = [(2.0, 2.0, 0.5, 0, 1, 1, 0.5), (2.5, 2.0, 0.5, 0, 1, 1, 0.75), (3.0, 2.0, 0.5, 0, 1, 0, 1.0),
              (4.0, 2.0, 0.5, 0, 1, 1, 0.0), (20.0, 2.0, 0.5, 1, 1, 0, 1.0), (21.0, 2.0, 0.5, 1, 1, 0, 0.875),
              (22.0, 2.0, 0.5, 1, 1, 1, 0.1), (30.0, 2.0, 0.5, 2, 1, 0, 1.0)]
    stats = [stat("X", 0, 2.25, 1.25, [3, 40]), stat("Y", 1, 1.975, 0.1, [2, 40]), stat("Z", None, 1)]
    cases.append(case("scores_support_by_source", answer(genes, [nb(6, 40.0), nb(7, 30.0)]),
                      where([("X", "from"), ("Y", "from"), ("Z", "from")], points, stats=stats, sources=(6, 7))))
    # Float32 at the cut-offs: P as a double just under 0.2 glows once stored as float32; 0.49999999 is likely, 0.4999999
    # is not; 0.5 exactly is likely. toFixed's binary ties: 0.3125 -> 0.313, 0.5625 -> 0.563 (half up).
    edge = [("F_0.199999999", 0.199999999), ("F_0.19999998", 0.19999998), ("F_0.2", 0.2), ("F_0.49999999", 0.49999999),
            ("F_0.4999999", 0.4999999), ("F_0.5", 0.5), ("T_0.3125", 0.3125), ("T_0.5625", 0.5625),
            ("T_0.0625", 0.0625), ("F_1", 1.0)]
    genes = [[name, 1, 1 - s, s, 1] for name, s in edge]
    points = [(5.0 + k, 7.0, 0.25, k, 1, k % 2) for k in range(len(edge))]
    cases.append(case("scores_float32_and_to_fixed_ties", answer(genes, [SOURCE]),
                      where([(name, "from") for name, _ in edge], points)))
    # Stats as the page reads them: a giver given as 1.0, -1 or text; own_near not a pair; near_giver below 0; in_cell 0;
    # a role it does not know (read as "from"); a gene twice (the last row and the last stats win).
    genes = [["A", 10, 2.0, 8.0, 1, 0], ["B", 10, 4.0, 6.0, 1, 0], ["C", 10, 6.0, 4.0, 1, 0], ["D", 10, 3.0, 7.0, 1],
             ["E", 10, 1.0, 9.0, 1], ["F", 10, 5.0, 5.0, 1], ["A", 10, 5.0, 5.0, 1, 0], ["G", 8, 4.0, 4.0, 1]]
    names = [(g, "from") for g in "ABCDEF"] + [("G", "own"), ("H", "from")]
    stats = [stat("A", 1.0, 4, 2, [2, 20]), stat("B", -1, 4, 2, [2, 20]), stat("C", "0", 4, 2, [2, 20]),
             stat("D", 0, 4, 2, [2, 20, 1]), stat("E", 0, 4, -2, [2, 20]), stat("F", 0, 0, 0, [2, 20]),
             stat("G", 0, 4, 2, [2, 20], role="sideways"), stat("A", 0, 4, 4, [1, 20]),
             stat("H", 0, 4, 2, [2, 20], word=["edge"]), {"gene": 7}, "not a stat", None,
             {"gene": "B", "role": "own"}]
    points = [(1.0 + 0.5 * k, 2.0, 0.1 * k, k % 8, 1, (k // 8) % 2) for k in range(16)]
    cases.append(case("scores_stats_coercion", answer(genes, [nb(6, 40.0), nb(7, 30.0)]),
                      where(names, points, stats=stats, sources=(6, 7))))
    return cases


def place_cases() -> list[dict]:
    """samePlace: groups with equal depth (ranked by point), a depth missing, places at x.xx5 rounding half up, negative
    places, one place across 0; and places past float32's range (an infinite place)."""
    genes = [["G", 10, 2.0, 8.0, 1]]
    pts = [
        (1.0, 2.0, 0.5, 0, 1, 0), (1.0, 2.0, 0.5, 0, 1, 0), (1.0, 2.0, 0.2, 0, 255, 0), (1.004, 2.0, 0.9, 0, 0, 0),
        (4.0, 4.0, 0.3, 0, 1, 0), (4.0, 4.0, None, 0, 1, 0),                          # a depth missing: no rank
        (0.125, 6.0, 0.1, 0, 1, 0), (0.145, 6.0, 0.2, 0, 1, 0),                       # 12.5 -> 13 (half up): one place
        (-0.375, 8.0, 0.1, 0, 1, 0), (-0.39, 8.0, 0.2, 0, 1, 0),                      # -37.5 -> -37: two places
        (-0.004, 10.0, 0.4, 0, 1, 0), (0.004, 10.0, 0.3, 0, 1, 0),                    # across 0: one place
        (7.0, 12.135, 0.1, 0, 1, 0), (7.0, 12.125, 0.1, 0, 1, 0), (7.0, 12.115, 0.1, 0, 1, 0),
        (9.0, 14.0, 0.7, 1, 1, 0), (9.0, 14.0, 0.1, 0, 1, 0),                         # another gene, the same place
    ]
    reply = where([("G", "from"), ("OWN", "own")], pts, stats=[stat("G", None, 12)])
    cases = [case("places_ranked_by_depth", answer(genes, [SOURCE]), reply)]
    far = [(1e39, 5.0, 0.1, 0, 1, 0), (1e39, 5.0, 0.2, 0, 1, 0), (2e39, 5.0, 0.3, 0, 1, 0),
           (1e39, -1e39, 0.1, 0, 1, 0), (1e39, -1e39, 0.2, 0, 1, 0), (2.0, 3.0, 1e39, 0, 1, 0),
           (2.0, 3.0, 0.5, 0, 1, 0), (1.5e15, 3.0, 0.5, 0, 1, 0), (1.5e15, 3.0, 0.4, 0, 1, 0)]
    cases.append(case("places_past_float32_and_double_steps", answer(genes, [SOURCE]),
                      where([("G", "from")], far, stats=[stat("G", None, 9)])))
    return cases


def kept_cases() -> list[dict]:
    """More than 400 glowing points: the kept list keeps the likeliest 400 (ties in point order); a gene's index past
    255 wraps as a Uint8Array's does; a gene name past 64 UTF-16 units is never sent."""
    s = [0.25, 0.5, 0.75]
    genes = [[f"K{k}", 4, 4 * (1 - v), 4 * v, 1] for k, v in enumerate(s)]
    points = [(0.25 * (k % 40), 0.25 * (k // 40), 0.5, k % 3, 1, 0) for k in range(403)]
    cases = [case("kept_more_than_400", answer(genes, [SOURCE]),
                  where([(f"K{k}", "from") for k in range(3)], points))]
    long = "L" * 63 + chr(0x1F600)                       # 65 UTF-16 units (64 code points): never sent
    names = [long, "L" * 64] + [f"W{k}" for k in range(2, 300)]
    genes = [["W4", 10, 5.0, 5.0, 1], ["W260", 10, 1.0, 9.0, 1], [long, 10, 2.0, 8.0, 1], ["L" * 64, 10, 3.0, 7.0, 1]]
    reply = where([(n, "from") for n in names],          # point 0's gene 260 is gene 4 (W4) to the page, never W260
                  [(1.0, 1.0, 0.5, 260, 1, 0), (2.0, 1.0, 0.5, 4, 1, 0), (3.0, 1.0, 0.5, 0, 1, 0),
                   (4.0, 1.0, 0.5, 1, 1, 0)])
    cases.append(case("kept_gene_index_wraps_and_long_names", answer(genes, [SOURCE]), reply))
    return cases


def refusal_cases() -> list[dict]:
    """Every reply normaliseWhere refuses (and a few it takes that look odd)."""
    genes = [["GENE1", 10, 2.0, 8.0, 1]]
    result = answer(genes, [SOURCE])
    good = where([("GENE1", "from")], grid_points(0, 3), stats=[stat("GENE1", 0, 3, 2, [1, 10])])

    def broken(name, change=None, points=None, drop=(), drop_points=()):
        reply = json.loads(json.dumps(good))
        reply["points"].update(points or {})
        for k in drop_points:
            reply["points"].pop(k, None)
        reply.update(change or {})
        for k in drop:
            reply.pop(k, None)
        return case(name, result, reply)
    return [
        case("refused_reply_null", result, None), case("refused_reply_text", result, "where"),
        case("refused_reply_number", result, 7), case("refused_reply_array", result, [good]),
        case("refused_reply_true", result, True),
        # The version (3 only: each point's share), and the reply's cell and frame against the request's (its seq
        # compared as text).
        broken("refused_v2", {"v": 2}), broken("refused_v1", {"v": 1}), broken("refused_v4", {"v": 4}),
        broken("refused_v_text", {"v": "3"}), broken("refused_v_true", {"v": True}),
        broken("refused_v_fraction", {"v": 3.5}), broken("refused_v_null", {"v": None}),
        broken("refused_v_missing", drop=["v"]), broken("taken_v_three_point_o", {"v": 3.0}),
        broken("refused_other_cell", {"cell": CELL + 1}), broken("refused_cell_as_text", {"cell": str(CELL)}),
        broken("refused_cell_missing", drop=["cell"]), broken("refused_other_seq", {"seq": SEQ + 1}),
        broken("taken_seq_as_text", {"seq": str(SEQ)}), broken("taken_seq_as_array", {"seq": [SEQ]}),
        broken("taken_cell_as_float", {"cell": float(CELL)}),
        # Genes and roles, the box.
        broken("refused_genes_not_array", {"genes": "GENE1"}), broken("refused_roles_missing", drop=["roles"]),
        broken("refused_roles_longer", {"roles": ["from", "own"]}), broken("refused_gene_not_text", {"genes": [5]}),
        broken("refused_role_unknown", {"roles": ["to"]}),
        broken("refused_box_text", {"box": "box"}), broken("refused_box_three", {"box": [0, 0, 1]}),
        broken("refused_box_null_entry", {"box": [0, None, 1, 1]}),
        broken("refused_box_x_backwards", {"box": [5, 0, 5, 1]}), broken("refused_box_y_flat", {"box": [0, 1, 1, 1]}),
        broken("taken_box_null", {"box": None}), broken("taken_box_missing", drop=["box"]),
        # The points: their lists, then any one point that is not one.
        broken("refused_points_missing", drop=["points"]), broken("refused_points_null", {"points": None}),
        broken("refused_points_array", {"points": [[1, 2]]}), broken("refused_points_x_text", points={"x": "1,2,3"}),
        broken("refused_points_y_missing", drop_points=["y"]),
        broken("refused_points_gene_short", points={"gene": [0, 0]}),
        broken("refused_points_owner_long", points={"owner": [1, 1, 1, 1]}),
        broken("refused_near_number", points={"near": 1}), broken("refused_near_short", points={"near": [0, 0]}),
        broken("taken_near_null", points={"near": None}),
        broken("refused_x_null", points={"x": [10.0, None, 11.0]}),
        broken("refused_x_text", points={"x": [10.0, "10.5", 11.0]}),
        broken("refused_y_true", points={"y": [20.0, True, 20.0]}),
        broken("refused_gene_fraction", points={"gene": [0, 0.5, 0]}),
        broken("refused_gene_true", points={"gene": [0, True, 0]}),
        broken("refused_gene_negative", points={"gene": [0, -1, 0]}),
        broken("refused_gene_past_genes", points={"gene": [0, 1, 0]}),
        broken("refused_owner_256", points={"owner": [1, 256, 1]}),
        broken("refused_owner_negative", points={"owner": [1, -1, 1]}),
        broken("refused_owner_fraction", points={"owner": [1, 1.5, 1]}),
        broken("refused_near_2", points={"near": [0, 2, 0]}), broken("refused_near_true", points={"near": [0, True, 0]}),
        broken("refused_near_text", points={"near": [0, "1", 0]}),
        broken("taken_gene_and_owner_as_floats",
               points={"gene": [0.0, 0, 0], "owner": [1.0, 1, 255.0], "near": [1.0, 0, -0.0]}),
        broken("taken_z_missing", drop_points=["z"]), broken("taken_z_short", points={"z": [0.5]}),
        broken("taken_z_odd", points={"z": [None, "0.5", True]}),
        broken("taken_empty", points={"x": [], "y": [], "z": [], "gene": [], "owner": [], "near": [], "w": []}),
        # Each point's share: 0 to 1, a number, one per point.
        broken("refused_w_missing", drop_points=["w"]), broken("refused_w_short", points={"w": [1.0, 1.0]}),
        broken("refused_w_number", points={"w": 1.0}), broken("refused_w_null", points={"w": [1.0, None, 1.0]}),
        broken("refused_w_above_1", points={"w": [1.0, 1.5, 1.0]}),
        broken("refused_w_below_0", points={"w": [1.0, -0.5, 1.0]}),
        broken("refused_w_text", points={"w": [1.0, "0.5", 1.0]}),
        broken("refused_w_true", points={"w": [1.0, True, 1.0]}),
        broken("taken_w_whole_and_minus_zero", points={"w": [1, -0.0, 0]}),
        # What the page reads leniently: cell_total any number from 0, held one of its three words.
        broken("taken_cell_total_fraction", {"cell_total": 2.375}),
        broken("taken_cell_total_negative", {"cell_total": -1}),
        broken("taken_held_unknown", {"held": "maybe", "held_seq": "12"}),
    ]


def double_cases() -> list[dict]:
    """Whole numbers past 2^53 add up as JavaScript's doubles do (2^53 + 1 is 2^53), where Python's ints would keep
    every bit: two neighbours whose labels (cell + 1) are one double share one row (the first's place, the last
    neighbour); the cell at 2^53 asks for 2^53 - 1; own_near's counts in f_r ([2^53, 1]: 2^53 / 3; [3, 2^53 - 1]:
    4 / 2^53)."""
    big = 2 ** 53
    genes = [["GENE1", 10, 2.0, 8.0, 1, 0]]
    cases = [case("rows_labels_one_past_2_53", answer(genes, [nb(big - 1, 40.0), nb(big, 30.0), nb(7, 20.0)])),
             case("rows_cell_at_2_53", answer(genes, [SOURCE], cell=big), cell=big)]
    genes = [["BIG", 10, 2.0, 8.0, 1, 0], ["TINY", 10, 2.0, 8.0, 1, 0]]
    points = [p for g in range(2) for near in (1, 0) for p in grid_points(g, 3, near=near, x0=10.0 + 20 * g + 5 * near)]
    cases.append(case("scores_own_near_past_2_53", answer(genes, [SOURCE]),
                      where([("BIG", "from"), ("TINY", "from")], points,
                            stats=[stat("BIG", 0, 6, 4, [big, 1]), stat("TINY", 0, 6, 4, [3, big - 1])])))
    return cases


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else FIXTURE
    with tempfile.TemporaryDirectory(prefix="qc-scoring-") as tmp:
        cases = world_cases(Path(tmp)) + hand_made_cases(Path(tmp))
    cases += chip_cases() + row_cases() + state_cases() + scoring_cases() + place_cases() + kept_cases() + \
        refusal_cases() + double_cases()
    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names), "case names must be unique"
    text = "[\n" + ",\n".join(json.dumps(c, separators=(",", ":"), allow_nan=False) for c in cases) + "\n]\n"
    out.write_text(text)
    print(f"{len(cases)} cases, {len(text.encode()) / 1024:.0f} KB -> {out} (now fill in expect: node "
          f"frontend/test/make-qc-scoring.mjs --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
