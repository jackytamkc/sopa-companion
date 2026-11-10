"""Test a setting here, its targets (engine/qc_pool.py): the area rule and every reason, in the page's order, words and
kinds (a refusal refuses, a wait makes the test wait, a caveat lets it run and says so); Go deeper's recipe on proseg's
own count at the parent's last frame F through the page's scoring (engine/qc_scoring.py) and the where measurement
(engine/deeper_where.py measure), with each target's record, giver and start class; proseg's share w of each record for
a cell from the test area's basis (exact from proseg's export of the area, approx from F's labels, none), the targets
it makes (glowing points, P = w x π, any owner; likely at π >= 0.5 with w >= 0.5), the controls matched on giver, near
flag and band of w; the clicked cell's own refusals and caveats (no_suspects only when nothing glows, low_share under
the cut-off, weak_suspects); the pool rule (no_controls: no control to judge the suspects by); matching on the shared
input; the caches and the run's worker.

The parent is a small synthetic run (test_analysis.write_run: 12 cells of 8 µm on a 2 µm grid, F = its frame 2) with
test_worker_deeper.make_deep's model and its own or a hand-made start; proseg's count at F (its counts file, as
proseg-live writes it with the hold) is made from each transcript's destination (its voxel's label at F unless a test
moves it: `dest`) and a keep per gene; proseg's exports of a box are made from the same destinations (the Runs
stand-in's request_where while the parent is held at F); the mixing check is a stand-in serving F's stored bytes.
Where a test needs suspects in given places, the page's scoring is a stand-in too (Page), each point's w the world's
own (HeldWhere: the where measurement with w from the world's destinations); elsewhere it is the real port.
Pool-rule tests give the cells' answers themselves (`fabricate`)."""
from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
import gzip
import json
from pathlib import Path
import re
import sys
import threading
import time
import types

import numpy as np
import pytest

from conftest import wait_for
import counts_fixtures as CF
from sopa_companion.analysis import counts_file
from sopa_companion.analysis import worker as W
from sopa_companion.engine import held
from sopa_companion.engine import qc_pool as P
from sopa_companion.engine import qc_readout as R
from sopa_companion.engine import qc_rule as Q
from sopa_companion.engine import qc_rules
from sopa_companion.engine import qc_scoring as S
from sopa_companion.engine.deeper_where import DeeperWhere, givers_of, near
from sopa_companion.engine.qc import QcError
from sopa_companion.engine.runs import RunError
from sopa_companion.engine.transcript_index import RECORD
import test_analysis as A
import test_worker_deeper as D

ROOT = Path(__file__).resolve().parents[1]
SEQ, ITER = 2, 2                      # F: write_run's last frame
MIXED, PLAIN = 200, 50                # stored bytes: a mixed cell, a judged one that is not
MODEL = "feedfacecafe0001"
STRONG = Q.MEASURED["strong_permille"]
MAX_UM = 1200.0                       # [proseg] max_region_um: a region wider than this runs in tiles
WHOLE = {"kind": "full", "bbox": [0.0, 0.0, 4000.0, 3000.0]}   # a whole sample's region (engine/runs.py region_spec)
COUNTS = {"mode": "request", "format": "PLC1", "version": 1, "e_scale": 32, "start": False, "start_every": None,
          "spool": None, "where": 1}  # info.json's counts block: proseg-live keeping its own counts
OUTLINE_IDS = Q.OUTLINE_IDS          # as qc_rule holds them
MEASURED_ON_PROSEG = Q.measured_on_proseg     # qc_rule's own (the tests patch it away: measured_on_proseg)


@pytest.fixture(autouse=True)
def measured_on_proseg(monkeypatch):
    """Every row taken as measured on proseg's own count (no count_basis caveat, though the rows of a start with nuclei
    were not), so that each test sees its own reasons alone;
    test_count_basis_where_the_row_is_not_measured_on_proseg_s_count has it back."""
    monkeypatch.setattr(Q, "measured_on_proseg", lambda start, situation: True)


def withdrawn(monkeypatch, start: str = "cells", situation: str = "square"):
    """The row not offered, as a measurement that did not offer it would leave it (the cells-only square before
    proseg's own count measured it): a test there is judged with the start's as-is row and says square_not_measured."""
    monkeypatch.setitem(Q.CONSTANTS[start]["situations"], situation,
                        {"offered": False, "refusal": "square_not_measured", "measured": False, "gate": 0.5,
                         "tau": {"excess": None, "suspects": None}, "rates": None})


# ---- the parent and the engine's stand-ins ---------------------------------------------------------------------------
class Run:
    """What Runs.get returns: meta, info and frames (read from the folder, each counts.jsonl line merged into its
    frame's record as Run.read_new merges it: counts_file, counted_file, counts), live, dir, lock; ack (proseg's, set
    by the world: "paused" at its seq while held), wheres (the where exports kept, newest last, each with its "path"),
    `export(box, seq)` (proseg's own export of a box while paused: the world's)."""

    def __init__(self, folder: Path, meta: dict):
        self.dir, self.meta, self.lock = folder, meta, threading.RLock()
        self.info = json.loads((folder / "live" / "info.json").read_text())
        self.frames = [json.loads(line) for line in (folder / "live" / "frames.jsonl").read_text().splitlines() if line]
        self.ack, self.wheres, self.export = None, [], None
        self.read_counts()

    @property
    def live(self) -> Path:
        return self.dir / "live"

    def read_counts(self):
        path = self.live / "counts.jsonl"
        for line in (path.read_text().splitlines() if path.exists() else []):
            if line:
                found = json.loads(line)
                if 0 <= found["seq"] < len(self.frames):
                    self.frames[found["seq"]].update(counts_file=found["file"], counted_file=found["counted_file"],
                                                     counts=found)


class Runs:
    """Runs' side: get, frame_path, transcript_index, and proseg's exports (INTERFACES §5.4): where_export (the newest
    kept at that frame covering the box) and request_where (while the run is paused, an export of the box from its
    paused state, kept: `asked` records each ask; else None, as for a run that cannot answer in time). `served_at`: the
    frame an ask is served at instead of the ack's (a stale ack: the run already sampling on, its export coming with a
    later counted frame, as engine/runs.py request_where serves it)."""
    max_region = MAX_UM

    def __init__(self, *runs):
        self.runs = {run.meta["id"]: run for run in runs}
        self.indexes = {}
        self.asked = []
        self.served_at = None

    def get(self, run_id):
        if run_id not in self.runs:
            raise RunError("Unknown run", 404)
        return self.runs[run_id]

    def frame_path(self, run_id, seq):
        return self.get(run_id).live / self.get(run_id).frames[seq]["file"]

    def transcript_index(self, run_id):
        from sopa_companion.engine.transcript_index import TranscriptIndex
        if run_id not in self.indexes:
            self.indexes[run_id] = TranscriptIndex(self.get(run_id).live)
        return self.indexes[run_id]

    def where_export(self, run_id, seq, box):
        run = self.get(run_id)
        for found in reversed(run.wheres):
            if found["seq"] == seq and held.covers(found, box) and Path(found["path"]).exists():
                return dict(found)
        return None

    def request_where(self, run_id, box, wait_s=2.0):
        run = self.get(run_id)
        self.asked.append((run_id, [float(v) for v in box], wait_s))
        ack = run.ack or {}
        if ack.get("state") != "paused" or run.export is None:
            return None
        seq = ack["seq"] if self.served_at is None else self.served_at
        kept = self.where_export(run_id, seq, box)
        if kept is not None:
            return kept
        made = {**run.export(box, seq), "path": None}
        made["path"] = str(run.live / made["file"])
        run.wheres = (run.wheres + [made])[-4:]
        return dict(made)


class Qc:
    """The mixing check's side: on and learned (its run's state and its sample's), its model (with Go deeper's
    details), F's stored bytes."""

    def __init__(self, model: Path, codes: Path):
        self.model, self.codes = model, codes
        self.enabled, self.state, self.sample_state, self.model_id, self.deep = True, "ready", "ready", MODEL, True
        self.reason = None
        self.configured = []

    def describe(self, run_id):
        return {"enabled": self.enabled, "state": self.state, "reason": self.reason, "sample_state": self.sample_state}

    def qc_model(self, run):
        if self.model_id is None:
            return None
        return {"id": self.model_id, "path": str(self.model),
                "deep": str(self.model / "deep.npz") if self.deep else None}

    def cells_path(self, run_id, seq, model=None, count=None):
        """QcManager.cells_path: a frame scored on the count proseg saves keeps its bytes on the averaged count beside
        its own (<seq>.qc.averaged.bin: its presence here says the frame was scored so); `count` "averaged" names those,
        "sampled" its own only then."""
        sampled = (self.codes / f"{int(seq):06d}.qc.averaged.bin").exists()
        path = self.codes / (f"{int(seq):06d}.qc.averaged.bin" if count == "averaged" and sampled else
                             f"{int(seq):06d}.qc.bin")
        if not path.exists() or model != self.model_id or (count == "sampled" and not sampled):
            raise QcError("Not checked yet", 404)
        return path

    def configure(self, run_id, seq=None):
        self.configured.append((run_id, seq))


class Analysis:
    """The scheduler's side: the check's settings, and whether Clusters & UMAP can run here."""

    def __init__(self):
        self.thresholds = dict(Q.MEASURED["thresholds"])
        self.available = True

    def qc_thresholds(self):
        return dict(self.thresholds)


def js_fixed(v: float, d: int) -> float:
    """+v.toFixed(d) for v >= 0: the double's exact decimal value, half up."""
    return float(Decimal(v).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


class Page:
    """The page's scoring, a stand-in (installed by `world(..., page=True)`): the where request asks `sources` (default:
    the answer's neighbours with at least 1 % of its RNA, at most 5) and `genes` (default: the answer's genes with
    transcripts from neighbours, at most 6); each point of a "from" gene, whoever holds it, gets π = p(gene, near) and
    P = w x π (the where reply's w; float32, as the page's suspectScores), and glows at P >= GLOW_MIN; a row's "same"
    is the answer's own unless `same` names the source cells that look like the cell's own kind. rank/of as samePlace
    gives them."""

    def __init__(self):
        self.sources, self.genes, self.same = None, None, None
        self.p = lambda gene, near: 0.7 if near else 0.55

    def params(self, raw, cell):
        if raw.get("state") != "ready":
            return {"state": raw.get("state"), "none": False, "chips": [], "rows": [], "req": None}
        total = raw["transcripts"]
        rows = [{"label": n["cell"] + 1, "cell": n["cell"], "share": n["tx"] / total, "key": n["cell"] + 1,
                 "same": n["same"] if self.same is None else n["cell"] in self.same}
                for n in raw["neighbours"] if n["tx"] / total >= 0.01][:5]
        if self.sources is not None:
            rows = [{"label": s + 1, "cell": s, "share": 0.1, "key": s + 1,
                     "same": False if self.same is None else s in self.same} for s in self.sources]
        genes = self.genes if self.genes is not None else [g[0] for g in raw["genes"] if g[3] > 0][:6]
        own = [g[0] for g in raw["own_genes"] if g[0] not in genes][:3]
        if not rows:
            return {"state": "ready", "none": False, "chips": list(genes), "rows": [], "req": None}
        params = {"seq": raw["seq"], "cell": cell, "sources": [r["cell"] for r in rows], "genes": list(genes),
                  "own": own}
        return {"state": "ready", "none": False, "chips": list(genes), "rows": rows,
                "req": {"key": json.dumps([raw["seq"], cell, params["sources"], params["genes"], own]),
                        "params": params, "labels": [r["label"] for r in rows]}}

    def scores(self, raw, req, reply):
        pts, roles, genes = reply["points"], reply["roles"], reply["genes"]
        w = pts["w"]
        points = []
        for i in range(len(pts["x"])):
            if roles[pts["gene"][i]] != "from":
                continue
            pi64 = float(self.p(genes[pts["gene"][i]], bool(pts["near"][i])))
            p = float(np.float32(float(w[i]) * pi64))
            if p < S.GLOW_MIN:
                continue
            point = {"i": i, "x": pts["x"][i], "y": pts["y"][i], "gene": genes[pts["gene"][i]],
                     "owner": pts["owner"][i], "near": pts["near"][i], "w": w[i], "pi": float(np.float32(pi64)),
                     "P": p}
            group = [j for j in range(len(pts["x"])) if pts["gene"][j] == pts["gene"][i]
                     and abs(pts["x"][j] - pts["x"][i]) <= 0.0101 and abs(pts["y"][j] - pts["y"][i]) <= 0.0101]
            if len(group) > 1:
                group.sort(key=lambda j: (pts["z"][j], j))
                point.update(rank=group.index(i), of=len(group))
            points.append(point)
        return {"count": len(pts["x"]), "points": points, "genes": []}


class HeldWhere(DeeperWhere):
    """Go deeper's where measurement with proseg's share of each point for the cell, from the world's own truth
    (INTERFACES §8.1; what R5's where engine gives, made here the same way for any where engine): w = held.weights over
    every record of the run, exact from the pools of the world's destinations at that frame, approx from its labels or
    none, as the world says (`where_held`, or the cell's own in its `held_by`: a reply placed apart from the area's
    basis, as the where engine places each cell's box itself; `placed` says how); v 3, held, held_seq; near for a "from"
    gene's point with w > 0 or held by the cell within NEAR_UM of its giver's footprint."""

    def __init__(self, runs, world):
        super().__init__(runs)
        self.world = world

    def placed(self, run_id, seq, cell, out, records, count="averaged") -> tuple[str, np.ndarray]:
        """(held, w of each point's record) for the cell: the world's (`where_held`, or the cell's own `held_by`), on
        the count measured (q = e / n, or f / n for "sampled")."""
        how = getattr(self.world, "held_by", {}).get(int(cell), self.world.where_held)
        return how, self.world.w_of(cell, records, how, "f" if count == "sampled" else "e")

    def measure(self, run_id, seq, cell, sources, genes, own, givers=None, count="averaged"):
        out, records = super().measure(run_id, seq, cell, sources, genes, own, givers=givers, count=count)
        how, w = self.placed(run_id, seq, cell, out, records, count)
        points = out["points"]
        out.update(v=3, held=how, held_seq=seq if how == "exact" else None)
        points["w"] = np.round(w, 3).tolist()
        layers = self.world.layers_at(seq)
        index = {name: g for g, name in enumerate(out["genes"])}
        near_by = np.asarray(points["near"], np.uint8).copy()
        for stat in out.get("gene_stats") or []:
            if stat.get("role") != "from" or stat.get("giver") is None:
                continue
            foot = (layers == sources[stat["giver"]] + 1).any(axis=0)
            mine = np.flatnonzero((np.asarray(points["gene"]) == index[stat["gene"]])
                                  & ((w > 0) | (np.asarray(points["owner"]) == 1)))
            grid = self.world.grid
            close = near(records["x"][mine].astype(np.float64), records["y"][mine].astype(np.float64), foot, 0, 0,
                         grid["x0"], grid["y0"], grid["voxel"])
            near_by[mine] = close.astype(np.uint8)
        points["near"] = near_by.tolist()
        return out, records


def write_codes(folder: Path, seq: int, codes, averaged=None) -> Path:
    """The check's bytes of frame `seq` as its scoring of a frame proseg counted for the hold leaves them: on the count
    proseg saves (<seq>.qc.bin) and on the averaged count beside them (<seq>.qc.averaged.bin, which the test reads),
    `averaged` (default: the same bytes); `averaged` False: none beside (a frame scored on the averaged count)."""
    from numcodecs import Zstd
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{seq:06d}.qc.bin"
    path.write_bytes(Zstd(level=1).encode(np.asarray(codes, np.uint8).tobytes()))
    if averaged is not False:
        beside = codes if averaged is None else averaged
        (folder / f"{seq:06d}.qc.averaged.bin").write_bytes(Zstd(level=1).encode(np.asarray(beside, np.uint8).tobytes()))
    return path


def write_prior(live: Path, labels: np.ndarray, permille: np.ndarray, x0=A.X0, y0=A.Y0, voxel=A.VOXEL):
    """A hand-made start: prior.bin.zst (u32 labels then u32 permille, [nz, ny, nx]) and info.json's prior."""
    import zstandard
    raw = np.concatenate([np.asarray(labels).ravel(), np.asarray(permille).ravel()]).astype("<u4")
    (live / "prior.bin.zst").write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.tobytes()))
    info = json.loads((live / "info.json").read_text())
    nz, ny, nx = labels.shape
    info["prior"] = {"file": "prior.bin.zst", "nx": nx, "ny": ny, "nz": nz, "x0": x0, "y0": y0, "voxel": voxel}
    (live / "info.json").write_text(json.dumps(info))


def records_of(live: Path) -> np.ndarray:
    return np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORD).copy()


GRID = {"x0": A.X0, "y0": A.Y0, "voxel": A.VOXEL, "nx": A.NX, "ny": A.NY, "nz": A.NZ}
Z0, DZ = 0.0, 0.5                     # write_run's layers


class World:
    """The parent p1 paused at the end of its schedule (F = seq 2), held there (proseg's ack "paused" at F), with
    QcPools over it. F's per-layer labels: write_run's (its plane in each layer), or `layers` [NZ, NY, NX] written into
    its frame (its consensus kept). Proseg's count at F: each transcript's destination `dest` (its voxel's label at F,
    by its own layer; those of `moved` {record: label} moved there), n per (cell, gene), e = n x keep of the gene
    (`keep`: one share, or one per gene), with its sampled count; the parent answers exports of a box from those
    destinations (`export`). `where_held`: how HeldWhere places w ("exact", or "approx" once the parent moved on past
    F); `held_by` {cell: how}: a cell's own where reply placed otherwise."""

    def __init__(self, tmp_path: Path, *, codes=None, region=(0.0, 0.0, 300.0, 300.0), prior=None, records=None,
                 meta=None, clock=time.monotonic, keep=1.0, moved=None, counts=True, page=None, layers=None):
        folder = A.write_run(tmp_path / "runs" / "p1")
        self.live = folder / "live"
        (self.live / "cells.txt").write_text("".join(c + "\n" for c in D.RUN_IDS))
        if records is not None:
            (self.live / "transcripts.bin.gz").write_bytes(gzip.compress(np.asarray(records, RECORD).tobytes()))
        if prior is not None:
            write_prior(self.live, *prior)
        self.layers_f = None if layers is None else np.asarray(layers, np.uint32)
        if self.layers_f is not None:                 # F's frame (write_run's gzip: consensus, last change, layers)
            raw = np.concatenate([A.frame_plane(SEQ).ravel(), np.zeros(A.NX * A.NY), self.layers_f.ravel()])
            (self.live / "frames" / f"{SEQ:06d}.bin.gz").write_bytes(gzip.compress(raw.astype("<u4").tobytes()))
        if counts:
            info = json.loads((self.live / "info.json").read_text())
            (self.live / "info.json").write_text(json.dumps({**info, "counts": dict(COUNTS)}))
        self.records = records_of(self.live)
        self.keep = np.broadcast_to(np.asarray(keep, np.float64), (len(A.RUN_GENES),)).copy()
        self.dest = self.labels_at_f(self.records).astype(np.int64)
        for record, label in (moved or {}).items():
            self.dest[int(record)] = int(label)
        if counts:
            self.write_counts(SEQ, ITER, reason="hold")
        self.model = D.make_deep(tmp_path / "model", folder)
        self.codes = np.full(A.NCELLS, MIXED, np.uint8) if codes is None else np.asarray(codes, np.uint8)
        self.qc_dir = tmp_path / "qc"
        write_codes(self.qc_dir, SEQ, self.codes)
        x0, y0, x1, y1 = region
        self.meta = {"id": "p1", "kind": "proseg", "state": "paused", "schedule_done": {"seq": SEQ, "iter": ITER},
                     "region": {"kind": "box", "box": [x0, y0, x1, y1], "bbox": [x0, y0, x1, y1]}, "mode": "detail",
                     "options": {}, "prior": {"cells": {"kind": "shapes", "key": "cells"},
                                              "nuclei": {"kind": "shapes", "key": "nuclei"}},
                     "sample_kind": "zarr", "prepared": {"seeded_from": "nuclei"}, "live_counts": "request",
                     **(meta or {})}
        self.run = Run(folder, self.meta)
        self.run.ack, self.run.export = {"state": "paused", "seq": SEQ, "iter": ITER}, self.export
        self.runs = Runs(self.run)
        self.qc, self.analysis = Qc(self.model, self.qc_dir), Analysis()
        self.where_held, self.held_by, self.grid = "exact", {}, GRID
        self.where = HeldWhere(self.runs, self)
        self.pools = P.QcPools(self.runs, self.qc, self.analysis, self.where, clock=clock)
        self.tokens, self.kept_w = 0, {}

    def write_counts(self, seq: int, iteration: int, reason: str = "request") -> dict:
        """Proseg's count of frame `seq` from `dest` and `keep` (a counted frame as proseg-live writes it)."""
        found = CF.counts_of(self.dest, self.records["gene"], A.NCELLS, len(A.RUN_GENES))
        e = found["n"] * self.keep[found["genes"]]
        return CF.write_frame_counts(self.live, seq, iteration, ncells=A.NCELLS, ngenes=len(A.RUN_GENES),
                                     indptr=found["indptr"], genes=found["genes"], n=found["n"], e=e,
                                     f=np.floor(e).astype(np.int64), reason=reason)

    def export(self, box, seq: int) -> dict:
        """Proseg's export of `box` at `seq`: one record per transcript whose observed voxel lies in the box's span (its
        voxel, gene and destination, n = 1), as the fake proseg writes it."""
        self.tokens += 1
        i0, j0, i1, j1 = CF.where_span(box, GRID)
        i, j, k, inside = CF.observed_voxels(self.records, GRID, Z0, DZ)
        span = inside & (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
        entries = {"i": i[span], "j": j[span], "k": k[span], "gene": self.records["gene"][span].astype(np.int64),
                   "dest": self.dest[span], "n": np.ones(int(span.sum()), np.int64)}
        return CF.write_where_export(self.live, self.tokens, seq, ITER, box, GRID, entries)

    def q(self, cell: int, which: str = "e") -> np.ndarray:
        """The cell's keep at F as proseg's count stores it: e / n (or f / n: the count proseg saves)."""
        found = counts_file.read(self.live / self.run.frames[SEQ]["counts_file"])
        rows = counts_file.rows(found, [cell], (which, "n"))
        e, n = rows[which][0].astype(np.float64), rows["n"][0].astype(np.float64)
        return np.divide(e, n, out=np.zeros_like(e), where=n > 0)

    def w_all(self, cell: int, how: str = "exact", which: str = "e") -> np.ndarray:
        """Proseg's share of every record for the cell at F: exact from every destination (one pool per record), or
        approx from F's labels; 0 everywhere for "none"; at the keep of count `which`."""
        key = (cell, how, which)
        if key not in self.kept_w:
            if how == "exact":
                i, j, k, inside = CF.observed_voxels(self.records, GRID, Z0, DZ)
                export = held.WhereExport(token=0, seq=SEQ, iter=ITER, box=(0.0, 0.0, 1.0, 1.0), margin_um=0.0,
                                          x0=A.X0, y0=A.Y0, voxel=A.VOXEL, nx=A.NX, ny=A.NY, nz=A.NZ, i0=0, j0=0,
                                          i1=A.NX - 1, j1=A.NY - 1, i=i[inside], j=j[inside], k=k[inside],
                                          gene=self.records["gene"][inside].astype(np.int64),
                                          dest=self.dest[inside], n=np.ones(int(inside.sum()), np.int64))
                w, _ = held.weights(self.records, cell, self.q(cell, which), GRID, Z0, DZ, export=export)
            elif how == "approx":
                w, _ = held.weights(self.records, cell, self.q(cell, which), GRID, Z0, DZ,
                                    layers=(self.layers_at(SEQ), 0, 0))
            else:
                w = np.zeros(len(self.records))
            self.kept_w[key] = w
        return self.kept_w[key]

    def w_of(self, cell: int, records: np.ndarray, how: str = "exact", which: str = "e") -> np.ndarray:
        """w of each of these records (the run's own, found by their 16 bytes: the first holder) for the cell."""
        keys, first = np.unique(P._void(self.records), return_index=True)
        at = np.searchsorted(keys, P._void(np.ascontiguousarray(records, RECORD)))
        return self.w_all(cell, how, which)[first[at]]

    def layers_at(self, seq: int) -> np.ndarray:
        """A frame's per-layer labels [nz, ny, nx] (write_run's: the frame's plane in each layer; F's own `layers` when
        given)."""
        if seq == SEQ and self.layers_f is not None:
            return self.layers_f
        return np.stack([A.frame_plane(seq)] * A.NZ)

    def centres(self):
        frame = self.run.frames[SEQ]
        found = W.read_centres(self.live / frame["cells_file"], A.NCELLS)
        return np.asarray(found["cx"], np.float64), np.asarray(found["cy"], np.float64)

    def labels_at_f(self, records) -> np.ndarray:
        """Each record's label at F (its voxel in F's per-layer labels, by its own layer): the worker's own lookup."""
        layers = self.layers_at(SEQ).reshape(-1)
        index = W.voxel_lookup(np, records["x"], records["y"], records["z"], A.X0, A.Y0, A.VOXEL, A.NX, A.NY, A.NZ,
                               0.0, 0.5)
        return np.where(index < layers.size, layers[np.minimum(index, layers.size - 1)], 0)

    def moved_on(self, seq: int = SEQ + 3):
        """The parent sampled on past F and paused again at `seq`: no export at F is kept any more."""
        self.meta["continued"] = True
        self.run.ack = {"state": "paused", "seq": seq, "iter": seq}
        self.run.wheres = []
        self.where_held = "approx"


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    """World(...) with `page=True` installing the Page stand-in (returned as world.page)."""
    made = []

    def make(page=False, **options):
        world = World(tmp_path / f"w{len(made)}", **options)
        if page:
            world.page = Page()
            monkeypatch.setattr(S, "page_params", world.page.params)
            monkeypatch.setattr(S, "page_scores", world.page.scores)
        made.append(world)
        return world
    yield make
    for world in made:
        world.pools.close()


def run_reasons_of(meta, info=None):
    return [r["code"] for r in P.run_reasons(meta, info, max_um=MAX_UM)]


def paused(**changes):
    meta = {"state": "paused", "schedule_done": {"seq": 451, "iter": 450}, "mode": "detail", "options": {},
            "region": {"kind": "box", "box": [0, 0, 300, 300], "bbox": [0, 0, 300, 300]},
            "prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": {"kind": "shapes", "key": "nuclei"}},
            "sample_kind": "zarr", "prepared": {"seeded_from": "nuclei"}}
    meta.update(changes)
    return meta


def box(x0, y0, x1, y1):
    return {"kind": "box", "box": [x0, y0, x1, y1], "bbox": [x0, y0, x1, y1]}


def kinds_of(reply: dict) -> tuple[list, list, list]:
    """A pool reply's codes by kind: (reasons, waits, caveats), each in the page's order."""
    for key in ("reasons", "waits", "caveats"):
        assert reply[key] == P.ordered(reply[key]), key
    return tuple([r["code"] for r in reply[key]] for key in ("reasons", "waits", "caveats"))


# ---- the situation and the area --------------------------------------------------------------------------------------
def test_situations_and_their_areas():
    """A region whose box fits in 500 x 500 µm is re-run as it is (a box, a lasso, a small sample whole: no place
    needed); any other gets a window around the cell's centre, per axis min(500, its extent) wide, shifted (never
    clipped) to stay inside: "square" where it is 500 x 500, "crop" where the region is narrower one way. The region's
    own situation (its caveats) is lasso, as_is, square or narrow."""
    for region in (box(0, 0, 300, 300), box(10, 20, 460, 520), box(0, 0, 250, 400), box(0, 0, 500, 500)):
        assert P.situation_of(region) == "as_is" and P.fits(region)
        area = P.area_of(region, (float("nan"), float("nan")))       # re-run as it is: no place needed
        assert area.rule == "as_is" and area.box == tuple(float(v) for v in region["bbox"])
        assert area.inner == (area.box[0] + 25, area.box[1] + 25, area.box[2] - 25, area.box[3] - 25)
        assert area.spec == region and area.spec is not region
    # 1,200 µm: a 500 µm square centred on the cell, shifted (never clipped) to stay inside near a corner.
    big = box(2900, 2900, 4100, 4100)
    assert P.situation_of(big) == "square" and not P.fits(big)
    assert P.area_of(big, (3500.0, 3600.0)).box == (3250.0, 3350.0, 3750.0, 3850.0)
    corner = P.area_of(big, (2910.0, 4080.0))
    assert corner.box == (2900.0, 3600.0, 3400.0, 4100.0) and corner.spec == {"kind": "box", "box": list(corner.box)}
    assert corner.inner == (2925.0, 3625.0, 3375.0, 4075.0) and corner.rule == "square"
    # 500 x 800: a 500 x 500 square flush with the short side.
    tall = box(0, 0, 500, 800)
    assert P.situation_of(tall) == "square" and P.area_of(tall, (100.0, 400.0)).box == (0.0, 150.0, 500.0, 650.0)
    # One side over 500 µm, the other under: a crop, the region's own width the short way, 500 µm around the cell the
    # long way (shifted to stay inside).
    for region, place, want in ((box(0, 0, 300, 900), (100.0, 880.0), (0.0, 400.0, 300.0, 900.0)),
                                (box(0, 0, 400, 1200), (390.0, 600.0), (0.0, 350.0, 400.0, 850.0)),
                                (box(0, 0, 1200, 499), (5.0, 10.0), (0.0, 0.0, 500.0, 499.0))):
        assert P.situation_of(region) == "narrow" and not P.fits(region)
        area = P.area_of(region, place)
        assert (area.rule, area.box, area.spec) == ("crop", want, {"kind": "box", "box": list(want)}), region
        assert area.inner == (want[0] + 25, want[1] + 25, want[2] - 25, want[3] - 25)
    # A lasso, whatever its size, is its own situation: re-run as it is (the polygon itself) when its box fits, else a
    # window of its box around the cell.
    small = {"kind": "polygon", "points": [[0, 0], [100, 0], [50, 80]], "bbox": [0, 0, 100, 80]}
    wide = {"kind": "polygon", "points": [[0, 0], [2000, 0], [1000, 1800]], "bbox": [0, 0, 2000, 1800]}
    flat = {"kind": "polygon", "points": [[0, 0], [2000, 0], [1000, 300]], "bbox": [0, 0, 2000, 300]}
    assert [P.situation_of(lasso) for lasso in (small, wide, flat)] == ["lasso"] * 3
    assert P.area_of(small, (float("nan"), 0.0)) == P.Area("as_is", (0.0, 0.0, 100.0, 80.0), (25.0, 25.0, 75.0, 55.0),
                                                           small)
    assert (P.area_of(wide, (1000.0, 900.0)).rule, P.area_of(wide, (1000.0, 900.0)).box) == \
        ("square", (750.0, 650.0, 1250.0, 1150.0))
    assert (P.area_of(flat, (10.0, 100.0)).rule, P.area_of(flat, (10.0, 100.0)).box) == \
        ("crop", (0.0, 0.0, 500.0, 300.0))
    # A small sample run whole (kind "full") uses its extent.
    full = {"kind": "full", "bbox": [0.0, 0.0, 399.9, 299.8]}
    assert P.situation_of(full) == "as_is" and P.area_of(full, (0.0, 0.0)).box == (0.0, 0.0, 399.9, 299.8)
    # A window needs the cell's place.
    for region in (big, box(0, 0, 400, 1200), wide):
        with pytest.raises(ValueError):
            P.area_of(region, (float("nan"), 3000.0))
    # The inner area's edges are inside.
    inner = P.area_of(box(0, 0, 300, 300), (0.0, 0.0)).inner
    assert Q.inside_inner([25.0, 275.0, 24.999, 275.001, float("nan")], [25.0, 275.0, 100.0, 100.0, 100.0],
                          inner).tolist() == [True, True, False, False, False]


def test_every_run_level_reason_under_its_condition(monkeypatch):
    """Each of the run's own reasons alone, of its kind, in the page's order when several hold. Refused: being finished,
    saved or left (finishing, leaving, a Finish & save whose save is still to come, a save under way), ending before
    its end (no end frame will ever come: a run that ended at or past its end is tested), and no proseg count to test
    with (no_counts, old_proseg). Waited for: its end. Caveats: the rest. nuclei_start keeps the region's own (judged as
    nuclei and cells). Every start's square is measured (a cells start's since proseg's own count measured it): a
    square is square_not_measured only where its row is not offered (as the cells-only square was before)."""
    assert run_reasons_of(paused()) == []
    for meta in (paused(state="running", schedule_done=None), paused(schedule_done=None),
                 paused(schedule_done={"seq": 3, "iter": None}), paused(state="preparing", schedule_done=None)):
        assert P.run_reasons(meta, None, max_um=MAX_UM) == [P.reason("not_at_end")], meta
    assert run_reasons_of(paused(continued=True)) == run_reasons_of(paused(continued=True, state="running")) == \
        ["past_end"]
    for changes in ({"state": "finishing"}, {"state": "finishing", "schedule_done": None},
                    {"leaving": {"answer": "add"}}, {"state": "done", "finish_requested": True, "zarr_save": None},
                    {"zarr_save": {"state": "saving"}}, {"state": "done", "zarr_save": {"state": "saving"}},
                    {"continued": True, "leaving": {"answer": "discard"}}):
        assert run_reasons_of(paused(**changes)) == ["ending"], changes
    for state in ("done", "stopped", "failed"):
        assert run_reasons_of(paused(state=state, schedule_done=None)) == ["ended"], state
        assert run_reasons_of(paused(state=state)) == [], state          # at its end: its frames and input stay
    assert run_reasons_of(paused(state="stopped", continued=True)) == ["past_end"]
    assert run_reasons_of(paused(state="done", finish_requested=True, zarr_save={"state": "saved"})) == []
    assert run_reasons_of(paused(state="done", finish_requested=True, zarr_save=None, sample_kind="xenium")) == []
    assert [P.KIND[code] for code in ("ending", "ended", "not_at_end", "past_end")] == \
        ["refuse", "refuse", "wait", "caveat"]
    assert run_reasons_of(paused(mode="tiled", region=WHOLE), {"cells": 41190, "counts": COUNTS}) == ["whole_sample"]
    assert run_reasons_of(paused(mode="tiled", region=box(0, 0, 1300, 1300))) == ["tiled_box"]
    for options, live, info in (({"burnin-samples": 100}, {}, None), ({"samples": 300}, {}, None),
                                ({"recorded-samples": 50}, {}, None), ({"hillclimb": 0}, {}, None),
                                ({}, {}, {"schedule": {"burnin": 100, "samples": 200, "recorded": 100,
                                                       "hillclimb": 50}, "counts": COUNTS}),
                                ({"morphology-steps-per-iter": 2000}, {}, None),
                                ({}, {"morphology-steps-per-iter": 3000}, None), ({"temperature": 0.5}, {}, None),
                                ({}, {"temperature": 1.0}, None)):
        assert run_reasons_of(paused(options=options, live_values=live), info) == ["schedule"], (options, live, info)
    # The measured schedule, said by info.json or by the options, is no reason; nor is a live value set back.
    measured = {"burnin": 200, "samples": 200, "recorded": 100, "hillclimb": 50}
    assert run_reasons_of(paused(options={"morphology-steps-per-iter": 2000},
                                 live_values={"morphology-steps-per-iter": 4000}),
                          {"schedule": measured, "counts": COUNTS}) == []
    assert run_reasons_of(paused(options={"burnin-samples": 200, "samples": 200})) == []
    nuclei = {"prior": {"nuclei": {"kind": "shapes"}}}
    lasso_9 = {"kind": "polygon", "points": [[0, 0], [9, 0], [5, 9]], "bbox": [0, 0, 9, 9]}
    assert run_reasons_of(paused(region=lasso_9, **nuclei)) == ["nuclei_start", "lasso"]
    assert run_reasons_of(paused(region=box(0, 0, 250, 250), **nuclei)) == ["nuclei_start", "region_small"]
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200), **nuclei)) == ["nuclei_start"]
    assert run_reasons_of(paused(region={"kind": "polygon", "points": [], "bbox": [0, 0, 300, 300]})) == ["lasso"]
    assert run_reasons_of(paused(region=box(0, 0, 400, 1200))) == ["area_narrow"]
    cells = {"prepared": {"seeded_from": "cells"}}
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200), **cells)) == []        # measured on proseg's count
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200))) == []
    assert run_reasons_of(paused(region=box(0, 0, 400, 400), **cells)) == []
    assert run_reasons_of(paused(region=box(0, 0, 250, 250), **cells)) == ["region_small"]
    # Several at once, in the page's order: whole_sample before schedule, square_not_measured (a row not offered)
    # after them.
    many = paused(state="running", schedule_done=None, mode="tiled", options={"temperature": 2.0}, region=WHOLE,
                  **cells)
    whole = {"cells": 41190, "counts": COUNTS}                     # a whole sample by its cells (H11)
    assert run_reasons_of(many, whole) == ["not_at_end", "whole_sample", "schedule"]
    assert run_reasons_of(many) == ["not_at_end", "tiled_box", "schedule"]      # its cells not known yet: tiled
    withdrawn(monkeypatch)
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200), **cells)) == ["square_not_measured"]
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200))) == []                  # the other start's own row
    assert run_reasons_of(many, whole) == ["not_at_end", "whole_sample", "schedule", "square_not_measured"]
    assert [P.KIND[code] for code in run_reasons_of(many, whole)] == ["wait", "caveat", "caveat", "caveat"]
    # No proseg count to test with: a run whose proseg-live could not keep them (old_proseg, whatever its info.json
    # says), else one whose info.json was written without them (no_counts: made before they were kept); refusals,
    # beside its end's own reason. No info.json yet: nothing said of its counts.
    old = paused(live_counts="old_proseg")
    assert run_reasons_of(old) == run_reasons_of(old, {"counts": COUNTS}) == run_reasons_of(old, {"cells": 3}) == \
        ["old_proseg"]
    assert run_reasons_of(paused(), {"cells": 3}) == run_reasons_of(paused(), {}) == ["no_counts"]
    assert run_reasons_of(paused(), {"cells": 3, "counts": COUNTS}) == run_reasons_of(paused(), None) == []
    assert run_reasons_of(paused(state="failed", schedule_done=None), {"cells": 3}) == ["ended", "no_counts"]
    assert run_reasons_of(paused(continued=True), {"cells": 3}) == ["no_counts", "past_end"]       # the page's order
    assert run_reasons_of(paused(state="running", schedule_done=None, live_counts="old_proseg")) == \
        ["old_proseg", "not_at_end"]
    assert [P.KIND[code] for code in ("no_counts", "old_proseg")] == ["refuse", "refuse"]
    assert P.counts_reason_of(paused(live_counts="request"), {"cells": 3}) == "no_counts"
    assert P.counts_reason_of(paused(live_counts="request"), {"cells": 3, "counts": COUNTS}) is None
    codes = list(P.REASON_WORDS)
    assert codes.index("square_not_measured") < codes.index("region_small")
    assert [r["code"] for r in P.ordered([P.reason("region_small", min_um=300.0), P.reason("square_not_measured")])] \
        == ["square_not_measured", "region_small"]


def test_the_counts_reason_is_the_engines(monkeypatch):
    """Why a run has no proseg count to test with is engine/runs.py counts_reason's answer where the engine has it, and
    its rule where it has not yet: old_proseg (its proseg-live could not keep them), else no_counts once its info.json
    holds no counts block (none at all, or not one), else nothing (counts kept, or no info.json yet)."""
    import sopa_companion.engine.runs as engine_runs
    cases = [(paused(), None), (paused(), {}), (paused(), {"cells": 3}), (paused(), {"cells": 3, "counts": "request"}),
             (paused(), {"cells": 3, "counts": COUNTS}), (paused(live_counts="old_proseg"), {"counts": COUNTS}),
             (paused(live_counts="old_proseg"), None), (paused(live_counts="request"), {"cells": 3})]
    want = [None, "no_counts", "no_counts", "no_counts", None, "old_proseg", "old_proseg", "no_counts"]
    assert [P.counts_reason_of(meta, info) for meta, info in cases] == want
    if hasattr(engine_runs, "counts_reason"):                # the engine's own, on what it reads of a run
        assert [engine_runs.counts_reason(P._RunLike(meta, info)) for meta, info in cases] == want
        monkeypatch.delattr(engine_runs, "counts_reason")
    assert [P.counts_reason_of(meta, info) for meta, info in cases] == want     # the rule, where the engine has none


def test_a_tiled_run_is_a_whole_sample_or_a_region_over_the_size_limit(monkeypatch):
    """H11, one rule with the mixing check's (qc_constants.run_mode): a run of WHOLE_CELLS cells or more is a
    whole-sample run (whole_sample), tiled or not, whatever its region; any other tiled run (by its meta or a prior kept
    in tiles: a box or a lasso over [proseg] max_region_um across, or a "full" region of a few thousand cells, as
    a real run's 4,823) is one over that limit (tiled_box), said with the limit the engine runs with (no run records
    it), and a lasso keeps its own reason too. A dense run of fewer cells has neither, whatever its region."""
    over = box(0, 0, 1152, 1260)                        # one side over the limit: run in tiles
    lasso = {"kind": "polygon", "points": [[0, 0], [1300, 0], [650, 1250]], "bbox": [0, 0, 1300, 1250]}
    tiled_box = {"code": "tiled_box", "words": "Not measured yet for regions over 1200 µm", "max_um": 1200.0}

    def info(cells, **more):
        return {"cells": cells, "counts": COUNTS, **more}
    assert P.run_reasons(paused(mode="tiled", region=WHOLE), info(20000), max_um=MAX_UM) == [P.reason("whole_sample")]
    assert P.run_reasons(paused(mode="tiled", region=WHOLE), info(4823), max_um=MAX_UM) == [tiled_box]
    assert P.run_reasons(paused(mode="tiled", region=WHOLE), None, max_um=MAX_UM) == [tiled_box]   # kind alone: no
    assert P.run_reasons(paused(mode="tiled", region=over), info(25000), max_um=MAX_UM) == \
        [P.reason("whole_sample")]                     # a big tiled box: a whole sample by its cells
    assert P.run_reasons(paused(region=over), info(4823, prior={"format": "tiles"}), max_um=MAX_UM) == \
        [tiled_box]                                    # tiled by its prior's tiles
    assert run_reasons_of(paused(region=box(0, 0, 900, 700)), info(30000)) == ["whole_sample"]
    assert P.run_reasons(paused(mode="tiled", region=over), None, max_um=MAX_UM) == [tiled_box]
    assert P.run_reasons(paused(mode="tiled", region=over), None, max_um=1500)[0]["words"] == \
        "Not measured yet for regions over 1500 µm"
    assert run_reasons_of(paused(mode="tiled", region=lasso)) == ["tiled_box", "lasso"]
    assert run_reasons_of(paused(mode="tiled", region=over, prepared={"seeded_from": "cells"})) == ["tiled_box"]
    withdrawn(monkeypatch)                              # the cells-only square as it was before proseg's count
    assert run_reasons_of(paused(mode="tiled", region=over, prepared={"seeded_from": "cells"})) == \
        ["tiled_box", "square_not_measured"]          # started from cells only: the square around the cell, too
    for region, expected in (({"kind": "full", "bbox": [0.0, 0.0, 900.0, 700.0]}, []), (box(0, 0, 1152, 1200), []),
                             (lasso, ["lasso"])):
        assert run_reasons_of(paused(region=region)) == expected, region
    codes = list(P.REASON_WORDS)
    assert codes.index("tiled_box") == codes.index("whole_sample") + 1


def test_the_start_is_read_when_called(monkeypatch):
    monkeypatch.setattr(qc_rules, "start_of", lambda meta: "cells")
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200))) == []
    assert P.row_of(paused(region=box(0, 0, 1200, 1200))) == ("cells", "square")
    monkeypatch.setattr(qc_rules, "start_of", lambda meta: "nuclei")
    assert run_reasons_of(paused(region=box(0, 0, 1200, 1200))) == ["nuclei_start"]
    assert run_reasons_of(paused(region=box(0, 0, 250, 250))) == ["nuclei_start", "region_small"]
    assert P.row_of(paused(region=box(0, 0, 1200, 1200))) == ("paired", "square")


def test_run_and_check_reasons_start_no_frame_work(make_world, monkeypatch):
    """Through the preview: the run's and the check's reasons by kind, listed at once, and nothing worked out (no cell,
    no frame context, no worker) while a refusal or a wait holds; `offer` stays true while only waits hold. The check:
    check_unavailable (its sample's or its run's state "unavailable", or Clusters & UMAP unavailable here) refuses,
    whatever its switch; off, learning, no model yet and Go deeper's details wait; settings not measured are a caveat.
    pool() refuses with the refusals alone (OutsideDomain) and raises NotYet with the waits; the check is asked to
    score F once per (run, F, model) while F's bytes are missing (check_scoring {wait: true})."""
    world = make_world()
    pools = world.pools
    made = []
    monkeypatch.setattr(P, "_Context", lambda *a: made.append(a))

    def kinds():
        reply = pools.preview("p1", 5)
        assert reply["state"] == "ready" and reply["pool"] is None and reply["offer"] is (not reply["reasons"])
        return kinds_of(reply)
    world.meta.update(state="running", schedule_done=None)
    world.qc.enabled = False
    assert kinds() == ([], ["not_at_end", "check_off"], [])
    world.qc.enabled, world.qc.state = True, "learning"
    assert kinds() == ([], ["not_at_end", "check_wait"], [])
    world.qc.state = "failed"                                   # its learning failed: a press asks again (qc_test)
    assert kinds() == ([], ["not_at_end", "check_wait"], [])
    for change in ({"sample_state": "unavailable", "enabled": False}, {"sample_state": "unavailable"},
                   {"state": "unavailable"}):
        kept = {key: getattr(world.qc, key) for key in change}
        for key, value in change.items():
            setattr(world.qc, key, value)
        assert kinds() == (["check_unavailable"], ["not_at_end"], []), change
        for key, value in kept.items():
            setattr(world.qc, key, value)
    # A run without proseg's count (its proseg-live could not keep it, or it was made before): its own refusal says
    # why; the run's check is unavailable for that alone (the sample's is ready), so never "The mixing check can't run
    # on this sample" beside it. The sample's own check unavailable too: said as ever.
    real = P.run_reasons
    for code in ("old_proseg", "no_counts"):
        monkeypatch.setattr(P, "run_reasons", lambda meta, info, max_um=None, code=code:
                            [*real(meta, info, max_um=max_um), P.reason(code)])
        world.qc.state, world.qc.reason = "unavailable", code
        assert kinds() == ([code], ["not_at_end"], []), code
        world.qc.sample_state = "unavailable"
        assert kinds() == ([code, "check_unavailable"], ["not_at_end"], []), code
        world.qc.sample_state, world.qc.reason = "ready", None
    monkeypatch.setattr(P, "run_reasons", real)
    world.qc.state, world.qc.reason = "unavailable", "no_counts"     # the check's word alone, no refusal of the run's
    assert kinds() == (["check_unavailable"], ["not_at_end"], [])
    world.qc.reason = None
    world.qc.state, world.analysis.available = "ready", False
    assert kinds() == (["check_unavailable"], ["not_at_end"], [])
    world.analysis.available = True
    world.meta.update(state="paused", schedule_done={"seq": SEQ, "iter": ITER})
    world.qc.model_id = None
    assert kinds() == ([], ["check_wait"], [])
    world.qc.model_id, world.qc.deep = MODEL, False
    assert kinds() == ([], ["details"], [])
    with pytest.raises(P.NotYet) as waiting:
        pools.pool("p1", 5)
    assert (waiting.value.status, waiting.value.waits, str(waiting.value)) == \
        (409, [P.reason("details")], P.REASON_WORDS["details"])
    world.qc.deep = True
    world.analysis.thresholds["judge_min"] = 50
    world.meta.update(state="stopped", schedule_done=None)
    assert kinds() == (["ended"], [], ["settings"])
    with pytest.raises(P.OutsideDomain) as refused:
        pools.pool("p1", 5)
    assert refused.value.reasons == [P.reason("ended")] and str(refused.value) == P.REASON_WORDS["ended"]
    world.analysis.thresholds["judge_min"] = 100
    world.meta.update(state="paused", schedule_done={"seq": SEQ, "iter": ITER})
    (world.qc_dir / f"{SEQ:06d}.qc.bin").rename(world.qc_dir / "kept.bin")
    reply = pools.preview("p1", 5)
    assert reply["waits"] == [{"code": "check_scoring", "wait": True,
                               "words": "Starts once the mixing check has scored your run's last iteration"}]
    assert (reply["offer"], reply["reasons"], reply["caveats"]) == (True, [], [])
    pools.preview("p1", 5)
    assert world.qc.configured == [("p1", SEQ)]                  # asked once for this frame
    with pytest.raises(P.NotYet) as waiting:
        pools.pool("p1", 5)
    assert waiting.value.waits == reply["waits"]
    # A refusal and a wait at once: refused with the refusal alone (F is not asked to be scored behind a refusal).
    world.meta.update(state="finishing", mode="tiled", continued=True)
    world.runs.max_region = 1500.0                                                # the engine's own limit, said
    assert kinds() == (["ending"], [], ["tiled_box"])
    with pytest.raises(P.OutsideDomain) as refused:
        pools.pool("p1", 5)
    assert refused.value.reasons == [P.reason("ending")] and str(refused.value) == P.REASON_WORDS["ending"]
    world.meta.update(state="paused")
    reply = pools.preview("p1", 5)
    assert kinds_of(reply) == ([], ["check_scoring"], ["past_end", "tiled_box"])
    assert reply["caveats"][1] == P.reason("tiled_box", max_um=1500.0)
    assert (pools.computed, pools.contexts, made, pools.work) == (0, 0, [], {})
    # The cheap reply still says where the test would run, and the row it is judged with.
    world.meta.update(mode="detail", continued=False)
    reply = pools.preview("p1", 5)
    assert reply["area"] == {"rule": "as_is", "box": [0.0, 0.0, 300.0, 300.0], "inner": [25.0, 25.0, 275.0, 275.0]}
    assert (reply["start"], reply["runs_per_side"], reply["k"], reply["constants"], reply["row"]) == \
        ("paired", 3, 160, Q.CONSTANTS_ID, {"start": "paired", "situation": "as_is"})
    assert (reply["seq"], reply["iter"], reply["model"], reply["estimate_s"]) == (SEQ, ITER, MODEL, 45.0)


def test_prerequisites_are_the_runs_the_checks_and_fs_scoring_without_a_cell(make_world, monkeypatch):
    """prerequisites(run): the refusals and waits of the run itself, its check and F's scoring (the check asked to
    score F as for a preview, once per (run, F, model)), no caveat; no cell, no frame file read, nothing worked out.
    404 for an unknown run or one that is not proseg's; 503 while the app closes."""
    world = make_world()
    pools = world.pools
    read = []
    monkeypatch.setattr(P.QcPools, "_at_f", lambda self, run, facts: read.append(facts.seq))
    world.analysis.thresholds["judge_min"] = 50                  # a caveat: never one of them
    assert pools.prerequisites("p1") == ([], [])
    world.meta.update(state="running", schedule_done=None)
    world.qc.enabled = False
    assert pools.prerequisites("p1") == ([], [P.reason("not_at_end"), P.reason("check_off")])
    world.meta.update(state="failed")
    assert pools.prerequisites("p1") == ([P.reason("ended")], [P.reason("check_off")])
    world.meta.update(state="paused", schedule_done={"seq": SEQ, "iter": ITER})
    world.qc.enabled = True
    (world.qc_dir / f"{SEQ:06d}.qc.bin").unlink()
    assert pools.prerequisites("p1") == ([], [P.reason("check_scoring", wait=True)])
    pools.prerequisites("p1")
    assert world.qc.configured == [("p1", SEQ)] and read == [] and pools.work == {}
    for run_id, words in (("nope", "Unknown run"),):
        with pytest.raises(RunError, match=words):
            pools.prerequisites(run_id)
    world.meta["kind"] = "stain"
    with pytest.raises(RunError, match="Not a proseg run"):
        pools.prerequisites("p1")
    world.meta["kind"] = "proseg"
    pools.close()
    with pytest.raises(QcError) as closing:
        pools.prerequisites("p1")
    assert closing.value.status == 503


def test_the_check_is_asked_again_when_it_refused_to_score(make_world):
    """While F's bytes are missing, the check is asked to score F once per (run, F, model); an ask it refused (it
    says why on its own panel) is asked again on the next preview, one it took never again; a forgotten run is asked
    afresh."""
    world = make_world()
    (world.qc_dir / f"{SEQ:06d}.qc.bin").unlink()
    asked = []

    def configure(run_id, seq=None):
        asked.append((run_id, seq))
        if len(asked) == 1:
            raise QcError("Unknown frame", 404)
    world.qc.configure = configure
    for _ in range(3):
        assert kinds_of(world.pools.preview("p1", 5)) == ([], ["check_scoring"], [])
    assert asked == [("p1", SEQ)] * 2
    world.pools.forget("p1")
    world.pools.preview("p1", 5)
    assert asked == [("p1", SEQ)] * 3


def test_a_sample_learned_again_asks_its_new_model_to_score_the_runs_end(make_world):
    """The check is asked to score F once per (run, F, model): once the sample is learned again (a new model, which has
    not scored F), the new model is asked once too, whatever else it scores by itself (the newest frame and at most 40
    earlier ones, evenly spaced: not always F of a run that sampled on past its end, whose test would wait for ever); F
    scored by it, nothing holds."""
    world = make_world(meta={"continued": True})
    (world.qc_dir / f"{SEQ:06d}.qc.bin").unlink()
    scoring = ([], [P.reason("check_scoring", wait=True)])
    assert world.pools.prerequisites("p1") == scoring and world.qc.configured == [("p1", SEQ)]
    world.qc.model_id = "learned-again"
    for _ in range(3):
        assert world.pools.prerequisites("p1") == scoring
        assert kinds_of(world.pools.preview("p1", 5)) == ([], ["check_scoring"], ["past_end"])
    assert world.qc.configured == [("p1", SEQ)] * 2
    write_codes(world.qc_dir, SEQ, world.codes)
    assert world.pools.prerequisites("p1") == ([], []) and world.qc.configured == [("p1", SEQ)] * 2


def test_refusals_of_the_run_and_cell(make_world):
    world = make_world()
    pools = world.pools
    for args, status, words in ((("nope", 0), 404, "Unknown run"), (("p1", A.NCELLS), 404, "Unknown cell"),
                                (("p1", -1), 404, "Unknown cell")):
        for call in (pools.preview, pools.pool):
            with pytest.raises(RunError) as refused:
                call(*args)
            assert (refused.value.status, str(refused.value)) == (status, words)
    world.meta["kind"] = "stain"
    with pytest.raises(RunError, match="Not a proseg run"):
        pools.preview("p1", 0)
    world.meta["kind"] = "proseg"
    pools.close()
    with pytest.raises(QcError) as closing:
        pools.preview("p1", 0)
    assert closing.value.status == 503


def test_the_cells_own_reasons_come_from_the_stored_bytes_and_centres(make_world, monkeypatch):
    """edge (a caveat: its centre at F outside the inner area; it is pooled anyway) and not_judged (byte 255, or no
    centre at F: it owns nothing there): from F's stored bytes and centres alone, with no frame work. A judged cell
    under the cut-off (74, or 0) is refused nothing there: whether it glows is its answer's (no_suspects, else
    low_share); the frame work starts for every cell refused nothing."""
    codes = np.full(A.NCELLS, MIXED, np.uint8)
    codes[1], codes[2], codes[3], codes[6] = 255, 74, 75, 0
    world = make_world(codes=codes, region=(100.0, 20.0, 400.0, 320.0))      # inner: x >= 125, y >= 45
    submitted = []
    monkeypatch.setattr(P.QcPools, "_submit", lambda self, run_id, job: submitted.append(job.cell) or job)
    cx, _ = world.centres()

    def kinds(cell):
        reasons, waits, caveats = kinds_of(world.pools.preview("p1", cell))
        assert waits == []
        return reasons, caveats
    assert cx[5] < 125 and kinds(5) == ([], ["edge"])             # pooled anyway: its frame work starts (stubbed)
    assert cx[1] < 125 and kinds(1) == (["not_judged"], ["edge"])
    assert cx[2] < 125 and kinds(2) == ([], ["edge"])             # under the cut-off: to its answer
    assert cx[3] >= 125 and kinds(3) == ([], [])
    assert np.isnan(cx[11]) and kinds(11) == (["not_judged"], ["edge"])      # vanished at F: owns nothing there
    assert submitted == [5, 2, 3]
    world2 = make_world(codes=codes)
    for cell in (2, 6):
        assert kinds_of(world2.pools.preview("p1", cell))[0] == []
    assert kinds_of(world2.pools.preview("p1", 1))[0] == ["not_judged"]
    reply = world2.pools.preview("p1", 1)
    assert reply["cell_name"] == D.RUN_IDS[1] and reply["area"]["rule"] == "as_is" and reply["offer"] is False


def test_a_stored_byte_is_followed_never_scored_again(make_world, monkeypatch):
    """D14: Go deeper's answer carries the stored byte (the card the user saw), 75 as mixed, whatever the numpy
    recipe would score; and its block is the engine's own: the stored bytes' file, no start line, no classes as last
    shown, read on proseg's count at F (the request names F's counts file); the givers it measures by are the fresh
    answer's. A cell under the cut-off (74) keeps its glowing targets, none counting for another cell's pool, its
    acting ones with their controls: tested itself with the caveat low_share."""
    codes = np.full(A.NCELLS, MIXED, np.uint8)
    codes[5], codes[3] = 75, 74
    world = make_world(codes=codes, page=True)
    asked, measured = [], []
    real_deeper, real_measure = W.Worker.deeper, HeldWhere.measure

    def deeper(self, run, request, counts, kept):
        result = real_deeper(self, run, request, counts, kept)
        asked.append((dict(request), dict(request["deeper"]), dict(kept), result))
        return result

    def measure(self, *args, **kwargs):
        measured.append((args, kwargs))
        return real_measure(self, *args, **kwargs)
    monkeypatch.setattr(W.Worker, "deeper", deeper)
    monkeypatch.setattr(HeldWhere, "measure", measure)
    answer = world.pools.answers("p1", [5, 3])
    assert (answer[5]["code"], answer[5]["mixed"], answer[3]["code"], answer[3]["mixed"]) == (75, True, 74, False)
    assert answer[3]["targets"] and not any(t["primary"] for t in answer[3]["targets"])     # never another's pool's
    assert any(t["acting"] for t in answer[3]["targets"])
    assert answer[3]["controls"]["matched"] and answer[3]["controls"]["uniform"]
    assert any(t["primary"] for t in answer[5]["targets"])
    by_cell = {block["cell"]: (request, block, kept, result) for request, block, kept, result in asked}
    for cell, code in ((5, 75), (3, 74)):
        request, block, kept, result = by_cell[cell]
        assert result["code"] == code and kept == {}
        # F counted for the hold (the count proseg saves): the test reads its bytes on the averaged count.
        assert block["codes"] == str(world.qc_dir / f"{SEQ:06d}.qc.averaged.bin") and block["start"] is False
        assert request.get("count", "averaged") == "averaged"
        assert not {"same_prev", "start_line", "start_prev"} & set(block)
        assert block["model"] == str(world.model) and block["model_id"] == MODEL and block["k"] == 10
        assert (block["fit_min"], block["judge_min"]) == (20, 100)
        assert block["cell_ids"] == str(world.live / "cells.txt")
        assert request["counts_file"] == str(world.live / world.run.frames[SEQ]["counts_file"])
        assert (request["seq"], request["iter"]) == (SEQ, ITER)
    for args, kwargs in measured:
        assert args[1] == SEQ and kwargs["givers"] == givers_of(by_cell[args[2]][3])
    reply = world.pools.preview("p1", 3, wait=10)
    assert (reply["offer"], reply["reasons"]) == (True, []) and "low_share" in [c["code"] for c in reply["caveats"]]
    assert "low_share" not in [c["code"] for c in world.pools.preview("p1", 5, wait=10)["caveats"]]


# ---- one cell's answer -----------------------------------------------------------------------------------------------
def test_a_cells_targets_records_givers_and_controls_with_the_real_scoring(make_world, monkeypatch):
    """With the page's own scoring (the port) of the where engine's own reply: each target is a point of the reply
    with P = w x π >= 0.2 (any owner), its record the parent record with the same 16 bytes as measure()'s RECORD for its
    point, proseg counting some of it for the cell (w > 0), likely at π >= 0.5 (float32) with w >= 0.5, its giver one
    of the request's sources; the controls are drawn as the module says."""
    world = make_world(keep=0.75)
    world.where = DeeperWhere(world.runs)                 # the where engine's own w
    world.pools.where = world.where
    records = records_of(world.live)
    measured, asked = {}, {}
    real = DeeperWhere.measure
    seen = spy_deeper(monkeypatch)

    def measure(self, run_id, seq, cell, sources, genes, own, **kwargs):
        asked[cell] = list(genes)
        measured[cell] = real(self, run_id, seq, cell, sources, genes, own, **kwargs)
        return measured[cell]
    monkeypatch.setattr(DeeperWhere, "measure", measure)
    answers = world.pools.answers("p1", range(8))
    with_targets = [c for c in range(8) if answers[c]["targets"]]
    assert with_targets                                   # the synthetic model gives some cells suspects
    for c in with_targets:
        a = answers[c]
        assert a["state"] == "ready" and a["request"] and a["sources"] and a["code"] == MIXED and a["mixed"]
        out, held_records = measured[c]
        assert out["held"] == "exact"
        for t in a["targets"]:
            r = records[t["record"]]
            assert r.tobytes() == held_records[t["point"]].tobytes()             # the same 16 bytes
            assert t["w"] == out["points"]["w"][t["point"]] > 0 and t["P"] >= 0.2 and t["copies"] == 1
            assert (round(float(r["x"]), 2), round(float(r["y"]), 2)) == (t["x"], t["y"])
            assert A.RUN_GENES[int(r["gene"])] == t["gene"] and t["gene"] in asked[c]
            assert t["likely"] == bool(np.float32(t["pi"]) >= np.float32(0.5) and t["w"] >= 0.5)
            assert t["giver"] is None or t["giver"] in a["sources"]
        # The genes left out: those the where request asked, and those Go deeper's answer tags as a neighbour's.
        excluded = sorted(set(asked[c]) | tagged_of(seen[c]))
        assert a["controls"]["excluded"] == excluded
        check_controls(world, a, records, excluded)


def test_likely_is_the_page_s_rule_read_when_a_cell_is_worked_out(make_world, monkeypatch):
    """A target is likely by the page's rule as engine/qc_scoring holds it when the cell is worked out (its twin of
    the page's, which the calibration may change): π >= LIKELY_MIN on a point with w >= LIKELY_W. Here proseg keeps 3
    in 4 of every gene (w 0.75), π 0.7 near the giver, 0.55 away."""
    world = make_world(page=True, keep=0.75)
    genes = top_genes(world, 5, 2)
    world.page.genes, world.page.sources = genes, [4]
    monkeypatch.setattr(P, "givers_of", lambda result: dict.fromkeys(genes, 4))

    def likely():
        world.pools.forget("p1")
        targets = world.pools.answers("p1", [5])[5]["targets"]
        assert targets and {t["w"] for t in targets} == {0.75}
        return {(round(t["pi"], 2), t["likely"]) for t in targets}
    assert likely() == {(0.7, True), (0.55, True)}
    monkeypatch.setattr(S, "LIKELY_MIN", 0.6)
    assert likely() == {(0.7, True), (0.55, False)}
    monkeypatch.setattr(S, "LIKELY_W", 0.8, raising=False)
    assert likely() == {(0.7, False), (0.55, False)}
    assert P.likely_rule() == (np.float32(0.6), 0.8)


def spy_deeper(monkeypatch) -> dict:
    """{cell: Go deeper's answer as the worker gave it} for every Worker.deeper the pools ask from here on."""
    seen = {}
    real = W.Worker.deeper

    def deeper(self, run, request, counts, kept):
        seen[request["deeper"]["cell"]] = result = real(self, run, request, counts, kept)
        return result
    monkeypatch.setattr(W.Worker, "deeper", deeper)
    return seen


def tagged_of(result: dict) -> set[str]:
    """The genes a Go deeper answer tags as a neighbour's: its gene rows [gene, transcripts, its own, from neighbours,
    tag(, giver)] with tag 1 (analysis/worker.py fitted_gene_rows)."""
    return {str(row[0]) for row in result.get("genes") or [] if len(row) > 4 and row[4] == 1}


def band_of(w) -> np.ndarray:
    """The band of w a control is matched on (0: under 0.25, 1: 0.25 to 0.75, 2: 0.75 or more), w to 3 decimals."""
    return np.searchsorted([0.25, 0.75], np.round(np.asarray(w, np.float64), 3), side="right")


def check_controls(world, a, records, excluded, how="exact"):
    """The controls of answer `a` (answers()), drawn again here by the module's rule: the cell's records with w > 0 (the
    world's own w at F, `how`) whose gene is not among `excluded` (worked out by the caller, never read back from
    `a`), seeds [20261028, c, 0] (3 per acting target with a placed giver, in point order, with its near flag to its
    giver: deeper_where.near on F's footprint, and its band of w) and [20261028, c, 1] (50 of them, for a cell with
    targets); each control's w as the world has it."""
    c = a["cell"]
    w = world.w_all(c, how)
    own = np.flatnonzero(w > 0)
    names = np.asarray(A.RUN_GENES, object)[records["gene"][own].astype(np.int64)]
    eligible = own[~np.isin(names, list(excluded))] if excluded else own
    bands = band_of(w[eligible])
    layers = world.layers_at(SEQ)
    rng = np.random.default_rng([20261028, c, 0])
    matched, short = {}, 0
    for t in a["targets"]:
        if not (t["acting"] and t["giver"] is not None):
            continue
        foot = (layers == t["giver"] + 1).any(axis=0)
        close = near(records["x"][eligible].astype(np.float64), records["y"][eligible].astype(np.float64), foot, 0, 0,
                     A.X0, A.Y0, A.VOXEL)
        pool = eligible[(close == t["near"]) & (bands == band_of(t["w"]))]
        k = min(3, pool.size)
        short += 3 - k
        matched[t["point"]] = [int(r) for r in (pool[rng.choice(pool.size, size=k, replace=False)] if k else [])]
    assert {m["point"]: m["records"] for m in a["controls"]["matched"]} == matched
    assert a["controls"]["short"] == short
    rng = np.random.default_rng([20261028, c, 1])
    k = min(50, eligible.size) if a["targets"] else 0
    assert a["controls"]["uniform"] == [int(r) for r in eligible[rng.choice(eligible.size, size=k, replace=False)]] \
        if k else a["controls"]["uniform"] == []
    drawn = [r for m in matched.values() for r in m] + a["controls"]["uniform"]
    assert a["controls"]["w"] == {str(r): round(float(w[r]), 3) for r in drawn}


def top_genes(world, cell: int, n: int) -> list[str]:
    """The cell's n most frequent genes at F."""
    records = records_of(world.live)
    mine = records["gene"][world.labels_at_f(records) == cell + 1].astype(np.int64)
    return [A.RUN_GENES[g] for g in np.argsort(-np.bincount(mine, minlength=len(A.RUN_GENES)), kind="stable")[:n]]


def test_targets_take_the_lowest_record_of_their_bytes(make_world, monkeypatch):
    """A record the parent holds twice (the same 16 bytes): both of its points are targets with the lower index; the
    pool counts it once in record_duplicates, and matching finds both unsure (their depths tie)."""
    probe = make_world()
    records = records_of(probe.live)
    owner = probe.labels_at_f(records)
    gene = top_genes(probe, 5, 1)[0]
    mine = np.flatnonzero((owner == 6) & (records["gene"] == A.RUN_GENES.index(gene)))
    first = int(mine[0])
    background = np.flatnonzero(owner == 0)
    later = int(background[background > first][-1])
    copied = records.copy()
    copied[later] = copied[first]                                   # a background record becomes the same transcript
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 1)
    world = make_world(page=True, records=copied)
    world.page.genes, world.page.sources = [gene], [4]
    monkeypatch.setattr(P, "givers_of", lambda result: {gene: 4})
    answer = world.pools.answers("p1", [5])[5]
    twins = [t for t in answer["targets"] if t["record"] == first]
    assert len(twins) == 2 and all(t["copies"] == 2 for t in twins)
    assert sorted(t["record"] for t in answer["targets"]) == sorted([first, *(int(r) for r in mine)])
    pool = world.pools.pool("p1", 5)
    assert sum(item.record == first for item in pool.items if item.kind == "target") == 2
    assert pool.record_duplicates == 1
    rows = [{"id": 1000 + i, "gene": A.RUN_GENES[int(r["gene"])], "x": float(r["x"]), "y": float(r["y"]),
             "z": 2.0 + 5.0 * float(r["z"])} for i, r in enumerate(copied)]
    matched = world.pools.match_input(pool, write_input(world.live.parent / "in.parquet", rows), {})
    assert matched.counts["target"]["unsure"] == 2


def test_start_classes_come_from_the_start_at_each_targets_place(make_world, monkeypatch):
    """A hand-made start (one layer, F's grid): where each target sat is its cell's (OWN), its giver's (STRONG at 650
    permille or more: 800 and 650; WEAK under: 649 and 500), another cell's (OTHER), no cell's (NONE: also off the
    start's grid, label -1)."""
    labels = np.zeros((1, A.NY, A.NX), np.uint32)
    permille = np.zeros((1, A.NY, A.NX), np.uint32)
    # Cell 5's block at F: rows 4-7, columns 4-7. Rows 4-5: column 4 its own (800), 5-7 its giver cell 4's at 800, 650
    # and 649. Rows 6-7: column 4 cell 4's at 500, 5 cell 9's (800), 6 nobody's, 7 its own at 500.
    labels[0, 4:6, 4], permille[0, 4:6, 4] = 6, 800
    labels[0, 4:6, 5], permille[0, 4:6, 5] = 5, 800
    labels[0, 4:6, 6], permille[0, 4:6, 6] = 5, 650
    labels[0, 4:6, 7], permille[0, 4:6, 7] = 5, 649
    labels[0, 6:8, 4], permille[0, 6:8, 4] = 5, 500
    labels[0, 6:8, 5], permille[0, 6:8, 5] = 10, 800
    labels[0, 6:8, 7], permille[0, 6:8, 7] = 6, 500
    world = make_world(page=True, prior=(labels, permille))
    genes = top_genes(world, 5, 3)
    givers = {genes[0]: 4, genes[1]: 4, genes[2]: 9}
    world.page.genes, world.page.sources = genes, [4, 9]
    monkeypatch.setattr(P, "givers_of", lambda result: givers)
    answer = world.pools.answers("p1", [5])[5]
    records = records_of(world.live)
    seen = set()
    for t in answer["targets"]:
        r = records[t["record"]]
        col, row = int(np.floor((r["x"] - A.X0) / A.VOXEL)), int(np.floor((r["y"] - A.Y0) / A.VOXEL))
        label, level = int(labels[0, row, col]), int(permille[0, row, col])
        giver = givers[t["gene"]]
        want = ("NONE" if label == 0 else "OWN" if label == 6 else
                ("GIVER_STRONG" if level >= STRONG else "GIVER_WEAK") if label == giver + 1 else "OTHER")
        assert (t["start_class"], t["start_label"], t["permille"], t["giver"]) == \
            (want, label, level if label else 0, giver)
        seen.add((want, level))
    assert {("GIVER_STRONG", 800), ("GIVER_STRONG", 650), ("GIVER_WEAK", 649), ("GIVER_WEAK", 500), ("OWN", 800),
            ("OWN", 500), ("NONE", 0), ("OTHER", 800)} <= seen
    assert answer["line"]["likely"] == answer["likely"] == sum(t["likely"] for t in answer["targets"])
    assert answer["line"]["neighbour"] == sum(t["likely"] and t["start_class"].startswith("GIVER")
                                              for t in answer["targets"])
    # The start's grid elsewhere: every target off it, NONE (label -1).
    away = make_world(page=True, prior=(labels, permille, 1000.0, 1000.0))
    away.page.genes, away.page.sources = genes, [4, 9]
    off = away.pools.answers("p1", [5])[5]
    assert off["targets"] and {(t["start_class"], t["start_label"], t["permille"]) for t in off["targets"]} == \
        {("NONE", -1, 0)}
    assert off["pattern"] == "NONE"


def test_patterns_and_keys():
    """The class holding at least half of the likely targets; exactly half each (or none at half): MIXED; the giver's
    hold STRONG unless more sat at the weak hold (a tie is STRONG); no key when every source looks like the cell's own
    kind; nothing without a likely target."""
    assert P.pattern_of(["OWN", "OWN", "NONE"], False) == ("OWN", None, "OWN")
    assert P.pattern_of(["OWN", "NONE"], False) == ("MIXED", None, "MIXED")            # 50 % each
    assert P.pattern_of(["OWN", "NONE", "OTHER"], False) == ("MIXED", None, "MIXED")   # none at half
    assert P.pattern_of(["OWN", "OWN", "NONE", "OTHER"], False) == ("OWN", None, "OWN")  # exactly half
    assert P.pattern_of(["GIVER_STRONG", "GIVER_WEAK", "OWN"], False) == ("GIVER", "STRONG", "GIVER_STRONG")
    assert P.pattern_of(["GIVER_WEAK", "GIVER_WEAK", "GIVER_STRONG"], False) == ("GIVER", "WEAK", "GIVER_WEAK")
    assert P.pattern_of(["NONE"] * 3, False) == ("NONE", None, "NONE")
    assert P.pattern_of(["OTHER", "OTHER", "OWN"], True) == ("OTHER", None, None)
    assert P.pattern_of(["GIVER_STRONG"], True) == ("GIVER", "STRONG", None)
    assert P.pattern_of([], False) == (None, None, None)


# ---- the seed's reasons and the pool rule, on answers given here -----------------------------------------------------
def fabricate(table: dict, monkeypatch, calls: list | None = None):
    """QcPools._cell answering from `table`: cell -> {key, pattern, n (likely targets with a placed giver), giver,
    unplaced, glow (glowing targets that are not likely: π 0.3), same, controls (False: no matched control), f (what
    changes in its answer on the count proseg saves, Go deeper's at a sampled F: `_cell` with count "sampled"; none:
    the same answer)}; every target sits at one of the cell's own records at F (those proseg counts some of for it,
    w > 0), with that record's w; each target with a placed giver has one matched control, the next of those records
    (with its w). `calls`: the cells whose answers the pool worked out (on the averaged count)."""
    def cell(self, context, weights, c, count="averaged", codes=None):
        if calls is not None and count == "averaged":
            calls.append(c)
        spec = table.get(c, {"key": None, "pattern": None, "n": 0})
        if count == "sampled":
            spec = {**spec, **spec.get("f", {})}
        own, shares = weights.of(c, context.q(c, "f" if count == "sampled" else "e"))
        targets, point = [], 0
        for kind, number in (("placed", spec.get("n", 0)), ("unplaced", spec.get("unplaced", 0)),
                             ("glow", spec.get("glow", 0))):
            for _ in range(number):
                r = int(own[point % len(own)])
                rec, w = context.records[r], round(float(shares[point % len(own)]), 3)
                pi = 0.3 if kind == "glow" else 0.75
                targets.append(P._Target(point=point, gene=context.genes[int(rec["gene"])],
                                         x=round(float(rec["x"]), 2), y=round(float(rec["y"]), 2), record=r, copies=1,
                                         p=float(np.float32(w * pi)), likely=kind != "glow", near=False,
                                         giver=spec.get("giver", 4) if kind != "unplaced" else None,
                                         start_class="OWN", start_label=c + 1, permille=800, rank=None, of=None,
                                         w=w, pi=float(np.float32(pi)), owner=1))
                point += 1
        sources = (spec.get("giver", 4),)
        same = (spec.get("same", False),)
        matched, given = {}, {}
        for t in targets if spec.get("controls", True) else ():
            if t.giver is not None:
                at = (t.point + 1) % len(own)
                matched[t.point] = (int(own[at]),)
                given[int(own[at])] = round(float(shares[at]), 3)
        return P._Answer(cell=c, code=int(context.codes[c]), state="ready", request=True, sources=sources, same=same,
                         all_same=bool(spec.get("same", False)), targets=tuple(targets), pattern=spec.get("pattern"),
                         hold=None, key=spec.get("key"), likely=sum(t.likely for t in targets),
                         line={"likely": sum(t.likely for t in targets), "own": 0, "none": 0, "neighbour": 0,
                               "other": 0},
                         matched=matched, short=0, uniform=(), excluded=(), seconds=0.0, held="exact",
                         owned=int(own.size), w=given, count=count)
    monkeypatch.setattr(P.QcPools, "_cell", cell)


def test_the_seeds_own_reasons(make_world, monkeypatch):
    """From the cell's own answer, refused (each one's code): nothing glows with a placed giver (no_suspects: no
    target, or only unplaced ones); every source of its own kind; no acting target whose giver lies in the inner area.
    A cell that glows with placed givers but has no likely suspect runs on its glowing points (weak_suspects). A pattern
    that was not measured is a caveat, and the pool is made all the same, of the cells with the seed's own key
    (pattern, and the hold for a neighbour's), nearest first; under k with every one of them (too_few)."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 3)
    world = make_world(region=(90.0, 20.0, 390.0, 320.0))        # inner x >= 115: cell 4 (x 104) is outside
    table = {5: {"key": "OWN", "pattern": "OWN", "n": 0, "unplaced": 2, "glow": 3}}
    fabricate(table, monkeypatch)
    cases = [({"n": 0, "unplaced": 2, "pattern": "OWN", "key": "OWN"}, "no_suspects"),
             ({"n": 0, "pattern": None, "key": None}, "no_suspects"),
             ({"n": 2, "pattern": "OWN", "key": None, "same": True}, "same_type"),
             ({"n": 2, "pattern": "OWN", "key": "OWN", "giver": 4}, "no_primary"),
             ({"n": 2, "pattern": "NONE", "key": "NONE", "giver": 4}, "no_primary"),
             ({"n": 0, "unplaced": 2, "glow": 3, "pattern": "OWN", "key": "OWN", "giver": 4}, "no_primary")]
    for spec, code in cases:
        world.pools.forget("p1")
        table[6] = dict(spec)
        reply = world.pools.preview("p1", 6, wait=10)
        assert reply["state"] == "ready" and [r["code"] for r in reply["reasons"]] == [code], (code, reply)
        assert reply["offer"] is False and reply["pool"] is None and reply["this_cell"]["likely"] == \
            spec["n"] + spec.get("unplaced", 0)
        with pytest.raises(P.OutsideDomain) as refused:
            world.pools.pool("p1", 6)
        assert refused.value.reasons == reply["reasons"]
    # Glowing with placed givers inside, none likely: tested on its glowing points (3: k here), with the caveat.
    world.pools.forget("p1")
    table[6] = {"n": 0, "unplaced": 1, "glow": 3, "pattern": "OWN", "key": "OWN", "giver": 10}
    reply = world.pools.preview("p1", 6, wait=10)
    assert (reply["offer"], reply["reasons"], reply["this_cell"]) == (True, [], {"likely": 1, "suspects": 3})
    assert kinds_of(reply)[2] == ["weak_suspects"]
    pool = world.pools.pool("p1", 6)
    assert [(i.kind, i.cell, i.giver) for i in pool.items if i.kind == "target"] == [("target", 6, 10)] * 3
    assert [c["code"] for c in pool.caveats] == ["weak_suspects"] and pool.this_cell["suspects"] == 3
    # Inside the inner area, nearest to cell 6 first: 2, 7 and 10 at 8 µm (by cell number), then 3. Each pattern's
    # pool takes the cells of its key alone: NONE has 7, 10 and 3; OWN has 2; the others none.
    table.update({2: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 10},
                  7: {"key": "NONE", "pattern": "NONE", "n": 2, "giver": 10},
                  10: {"key": "NONE", "pattern": "NONE", "n": 2, "giver": 10},
                  3: {"key": "NONE", "pattern": "NONE", "n": 2, "giver": 10}})
    for pattern, key, code, cells in (("OWN", "OWN", None, (6, 2)), ("NONE", "NONE", "pattern_none", (6, 7, 10, 3)),
                                      ("GIVER", "GIVER_STRONG", "pattern_neighbour", (6,)),
                                      ("OTHER", "OTHER", "pattern_other", (6,)),
                                      ("MIXED", "MIXED", "pattern_mixed", (6,))):
        world.pools.forget("p1")
        table[6] = {"n": 2, "pattern": pattern, "key": key, "giver": 10}
        reply = world.pools.preview("p1", 6, wait=10)
        assert (reply["state"], reply["offer"], reply["reasons"], reply["pattern"]) == \
            ("ready", True, [], P.PATTERNS[pattern])
        pool = world.pools.pool("p1", 6)
        short = len(cells) == 1
        want = [c for c in (code, "too_few" if short else None) if c]
        assert [c["code"] for c in reply["caveats"]] == [c["code"] for c in pool.caveats] == want, pattern
        assert pool.cells == cells and pool.members == (1 if short else 2) and pool.k == 3, pattern
        assert reply["pool"]["reached"] is not short and [m["cell"] for m in reply["pool"]["members"]] == \
            list(cells[:pool.members])
        if short:
            assert pool.caveats[-1] == P.reason("too_few", have=2, need=3)


def test_a_cell_under_the_cut_off_is_tested_on_its_pink_dots_with_cells_of_its_key(make_world, monkeypatch):
    """A1: a judged cell under the cut-off (its stored byte under mixed_at's) is refused only when nothing of it glows
    (no_suspects); glowing, it is tested with the caveat low_share, its own acting targets as its suspects, and the
    other cells are chosen as for any cell: the mixed, judged ones of its key in the inner area, nearest first (never
    another cell under the cut-off); weak as well: both caveats."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 5)
    codes = np.full(A.NCELLS, MIXED, np.uint8)
    codes[6], codes[7] = PLAIN, 0                            # judged, under the cut-off
    world = make_world(codes=codes)
    table = {6: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 10},
             7: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 10},          # nearest, but under the cut-off
             2: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 10},
             10: {"key": "NONE", "pattern": "NONE", "n": 2, "giver": 5}}
    fabricate(table, monkeypatch)
    reply = world.pools.preview("p1", 6, wait=10)
    assert (reply["offer"], reply["reasons"]) == (True, [])
    assert kinds_of(reply)[2] == ["low_share", "too_few"]
    pool = world.pools.pool("p1", 6)
    assert pool.cells == (6, 2) and [c["code"] for c in pool.caveats] == ["low_share", "too_few"]
    assert [i.cell for i in pool.items if i.kind == "target"] == [6, 6, 2, 2]
    # Nothing of it glows: refused, whatever its byte.
    for codes_at, spec in ((PLAIN, {"key": None, "pattern": None, "n": 0}), (MIXED, {"key": None, "pattern": None})):
        world.codes[6] = codes_at
        write_codes(world.qc_dir, SEQ, world.codes)
        world.pools.forget("p1")
        table[6] = spec
        assert kinds_of(world.pools.preview("p1", 6, wait=10))[0] == ["no_suspects"]
    # Weak and under the cut-off: both.
    world.codes[6] = PLAIN
    write_codes(world.qc_dir, SEQ, world.codes)
    world.pools.forget("p1")
    table[6] = {"key": "OWN", "pattern": "OWN", "glow": 2, "giver": 10}
    assert kinds_of(world.pools.preview("p1", 6, wait=10)) == ([], [], ["low_share", "weak_suspects", "too_few"])


def setup_rule(make_world, monkeypatch, k: int, table: dict, calls: list):
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", k)
    world = make_world()
    fabricate(table, monkeypatch, calls)
    return world


def test_the_pool_rule_nearest_first_same_key_and_a_reserve(make_world, monkeypatch):
    """The seed, then the mixed, judged cells in the inner area with its key and a primary target, nearest first (ties
    by cell number), until their primary targets reach k; then RESERVE_CELLS more; every cell computed only once
    reached (lazily), and the same pool as computing every cell first."""
    calls = []
    table = {5: {"key": "OWN", "pattern": "OWN", "n": 2}, 4: {"key": "OWN", "pattern": "OWN", "n": 2},
             6: {"key": "NONE", "pattern": "NONE", "n": 5}, 9: {"key": "OWN", "pattern": "OWN", "n": 0, "unplaced": 3},
             1: {"key": "OWN", "pattern": "OWN", "n": 3}, 0: {"key": "OWN", "pattern": "OWN", "n": 1},
             2: {"key": "OWN", "pattern": "OWN", "n": 2}, 8: {"key": "OWN", "pattern": "OWN", "n": 4},
             10: {"key": "OWN", "pattern": "OWN", "n": 4}, 7: {"key": "OWN", "pattern": "OWN", "n": 4},
             3: {"key": "OWN", "pattern": "OWN", "n": 4}}
    world = setup_rule(make_world, monkeypatch, 7, table, calls)
    cx, cy = world.centres()
    far = np.hypot(cx - cx[5], cy - cy[5])
    order = [c for c in np.lexsort((np.arange(A.NCELLS), far)).tolist() if c not in (5, 11)]
    assert order[:4] == [4, 6, 9, 1] and far[4] == far[6] == far[9] == 8.0      # three at 8 µm: by cell number
    pool = world.pools.pool("p1", 5)
    assert pool.cells == (5, 4, 1, 0, 2) and pool.members == 3 and pool.suspects == 7 and pool.k == 7
    assert pool.distance_um == (0.0, far[4], far[1], far[0], far[2]) and pool.available is None
    assert calls == [5, 4, 6, 9, 1, 0, 2]                          # lazily: none past the reserve
    assert [item.cell for item in pool.items if item.kind == "target"] == [5] * 2 + [4] * 2 + [1] * 3 + [0] + [2] * 2
    assert pool.this_cell == {"likely": 2, "suspects": 2, "pattern": "own", "line": {"likely": 2, "own": 0, "none": 0,
                                                                                     "neighbour": 0, "other": 0}}
    assert (pool.seed, pool.seed_name, pool.start, pool.area.rule, pool.seq, pool.iter, pool.model) == \
        (5, D.RUN_IDS[5], "paired", "as_is", SEQ, ITER, MODEL)
    assert set(pool.names) == {5, 4, 1, 0, 2} and pool.names[4] == D.RUN_IDS[4]
    reply = world.pools.preview("p1", 5)
    assert reply["offer"] is True and reply["reasons"] == [] and reply["pattern"] == "own"
    assert reply["pool"] == {"cells": 3, "suspects": 7, "need": 7, "reached": True, "available": None,
                             "members": [{"cell": 5, "name": D.RUN_IDS[5], "n": 2, "um": 0.0},
                                         {"cell": 4, "name": D.RUN_IDS[4], "n": 2, "um": 8.0},
                                         {"cell": 1, "name": D.RUN_IDS[1], "n": 3, "um": round(far[1], 1)}]}
    assert reply["this_cell"] == {"likely": 2, "suspects": 2} and reply["cell_name"] == D.RUN_IDS[5]
    # Every cell computed first: the same pool.
    world.pools.forget("p1")
    calls.clear()
    world.pools.answers("p1", range(11))
    assert world.pools.pool("p1", 5) == pool and len(calls) == 11


def test_a_pool_not_reached_is_tested_with_what_there_is(make_world, monkeypatch):
    """A pool whose primary targets never reach k is offered all the same, with every cell of the key that has one,
    its caveat too_few {have, need}: the row's numbers are for pools of k suspects or more (since the calibration on
    proseg's count, 2026-10-31, which measured a start from cells only's short pools, and weaker: no longer "not
    measured yet")."""
    calls = []
    table = {5: {"key": "OWN", "pattern": "OWN", "n": 2}, 4: {"key": "OWN", "pattern": "OWN", "n": 3},
             1: {"key": "MIXED", "pattern": "MIXED", "n": 9}}
    world = setup_rule(make_world, monkeypatch, 160, table, calls)
    reply = world.pools.preview("p1", 5, wait=10)
    assert (reply["offer"], reply["reasons"], reply["waits"]) == (True, [], [])
    assert reply["caveats"] == [{"code": "too_few", "have": 5, "need": 160,
                                 "words": "Its numbers are for tests with 160 suspects or more: 5 here"}]
    assert reply["pool"]["reached"] is False and reply["pool"]["available"] == 5 and reply["pool"]["cells"] == 2
    assert sorted(calls) == list(range(11))                         # every candidate was looked at
    pool = world.pools.pool("p1", 5)
    assert (pool.cells, pool.members, pool.suspects, pool.k, pool.available) == ((5, 4), 2, 5, 160, None)
    assert list(pool.caveats) == reply["caveats"] and pool.row == ("paired", "as_is")
    assert [item.cell for item in pool.items if item.kind == "target"] == [5] * 2 + [4] * 3


def test_a_pool_with_no_control_to_judge_its_suspects_by_is_refused(make_world, monkeypatch):
    """Under the rule "excess" (qc_rule: the suspects judged against their own RNA at the same spot, their matched
    controls), a pool none of whose cells has a matched control could only answer too_few (no_controls) after every
    run, so it is refused (no_controls, its words the page's); one matched control among its cells, the reserve's last
    one's too (a matching that loses a member's suspects takes the reserve in), and it is taken; the rule "suspects"
    (the suspects alone) takes it without any."""
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": n, "controls": False}
             for c, n in ((5, 2), (4, 2), (1, 3), (0, 1), (2, 2))}
    world = setup_rule(make_world, monkeypatch, 7, table, calls)
    reply = world.pools.preview("p1", 5, wait=10)
    assert kinds_of(reply)[0] == ["no_controls"] and (reply["offer"], reply["pool"]) == (False, None)
    assert reply["reasons"] == [{"code": "no_controls", "words": "No own RNA at the same spot to compare its suspects "
                                                                  "with"}]
    assert sorted(calls) == [0, 1, 2, 4, 5, 6, 9]            # the members and the reserve were looked at
    with pytest.raises(P.OutsideDomain) as refused:
        world.pools.pool("p1", 5)
    assert refused.value.reasons == [P.reason("no_controls")]
    world.pools.forget("p1")
    table[2] = {"key": "OWN", "pattern": "OWN", "n": 2}       # the reserve's last cell: one control per suspect
    pool = world.pools.pool("p1", 5)
    assert (pool.cells, pool.members) == ((5, 4, 1, 0, 2), 3)
    assert [(item.cell, item.target) for item in pool.items if item.kind == "matched"] == [(2, 8), (2, 9)]
    world.pools.forget("p1")
    table[2]["controls"] = False
    monkeypatch.setitem(Q.CONSTANTS["paired"], "rule", "suspects")
    pool = world.pools.pool("p1", 5)
    assert pool.cells == (5, 4, 1, 0, 2) and {item.kind for item in pool.items} == {"target"}


def test_only_mixed_judged_cells_inside_count_and_their_primary_targets(make_world, monkeypatch):
    """Candidates: mixed and judged by their stored bytes, centres in the inner area; a target counts (primary) when
    likely, placed, and its giver's centre lies inside too."""
    calls = []
    codes = np.full(A.NCELLS, MIXED, np.uint8)
    codes[6], codes[9] = PLAIN, 255
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 100)
    world = make_world(codes=codes, region=(90.0, 20.0, 390.0, 320.0))     # inner x >= 115
    cx, _ = world.centres()
    inside = [c for c in range(11) if cx[c] >= 115]
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 6} for c in range(A.NCELLS)}
    table[7] = {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 4}          # its giver at x 104: outside
    fabricate(table, monkeypatch, calls)
    reply = world.pools.preview("p1", 2, wait=10)
    assert kinds_of(reply) == ([], [], ["too_few"])
    looked = set(calls) - {2}
    assert looked == {c for c in inside if c not in (2, 6, 9)}
    members = {m["cell"]: m["n"] for m in reply["pool"]["members"]}
    assert 7 not in members and all(n == 2 for n in members.values())


def test_a_seed_taken_on_the_count_proseg_saves_is_pooled_with_the_cells_of_its_key_there(make_world, monkeypatch):
    """At F counted for the hold, a clicked cell with no suspect on e (no key there) whose answer on f, Go deeper's
    there, has two (key OWN) is pooled as any seed is, with the cells of that key nearest first (their own answers, on
    e), and says so (saved_suspects); its answer on e is worked out once, as every pooled cell's. With nothing on f
    either: no_suspects."""
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 2} for c in range(A.NCELLS)}
    table[5] = {"key": None, "pattern": None, "n": 0, "f": {"key": "OWN", "pattern": "OWN", "n": 2}}
    world = setup_rule(make_world, monkeypatch, 4, table, calls)
    reply = world.pools.preview("p1", 5, wait=10)
    assert (reply["offer"], reply["reasons"], reply["this_cell"]) == (True, [], {"likely": 2, "suspects": 2})
    assert kinds_of(reply)[2] == ["saved_suspects"] and reply["line"] == reply["line_shown"]
    pool = world.pools.pool("p1", 5)
    assert pool.cells[:3] == (5, 4, 6) and pool.members == 2 and pool.suspects == 4
    assert [c["code"] for c in pool.caveats] == ["saved_suspects"]
    assert [item.cell for item in pool.items if item.kind == "target"][:2] == [5, 5]
    assert calls.count(5) == 1
    world.pools.forget("p1")
    table[5]["f"] = {"n": 0}
    refused = world.pools.preview("p1", 5, wait=10)
    assert kinds_of(refused)[0] == ["no_suspects"] and "saved_suspects" not in kinds_of(refused)[2]


def test_a_seed_whose_suspects_on_the_averaged_count_cannot_be_tested_is_tested_on_those_go_deeper_shows(make_world,
                                                                                                          monkeypatch):
    """At F counted for the hold the test refuses only when nothing can be tested: a clicked cell whose suspects on e
    cannot be tested, every giver outside the inner area (no_primary) or every source of its own kind (same_type), while
    Go deeper's on f can be (its giver inside; a source of another kind), is tested on f's, with the caveat
    saved_suspects. Where f's cannot be tested either, e's refusal stands, without the caveat; where nothing of it glows
    on e, f's glowing suspects are taken and refused for their own reason, never no_suspects while they glow. A cell
    whose suspects on e can be tested keeps them, whatever f's."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 3)
    world = make_world(region=(90.0, 20.0, 390.0, 320.0))        # inner x >= 115: cell 4 (x 104) is outside
    table = {2: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 10}}
    fabricate(table, monkeypatch)
    own = {"n": 2, "pattern": "OWN", "key": "OWN"}

    def asked(spec: dict):
        """(cell 6's pool reply with `spec` as its answers, its pool or the refusal)."""
        world.pools.forget("p1")
        table[6] = spec
        reply = world.pools.preview("p1", 6, wait=10)
        assert reply["state"] == "ready" and reply["count"] == "sampled", reply
        try:
            return reply, world.pools.pool("p1", 6)
        except P.OutsideDomain as refused:
            return reply, refused
    for spec in ({**own, "giver": 4, "f": {"giver": 10}}, {**own, "giver": 10, "same": True, "f": {"same": False}}):
        reply, pool = asked(spec)
        assert (reply["offer"], reply["reasons"], kinds_of(reply)[2]) == (True, [], ["saved_suspects"]), spec
        assert reply["this_cell"] == {"likely": 2, "suspects": 2} and pool.cells == (6, 2)
        assert [(i.cell, i.giver) for i in pool.items if i.kind == "target"] == [(6, 10)] * 2 + [(2, 10)] * 2
        assert [c["code"] for c in pool.caveats] == ["saved_suspects"]
    for spec, code in (({**own, "giver": 4}, "no_primary"), ({**own, "giver": 10, "same": True}, "same_type"),
                       ({**own, "giver": 4, "f": {"giver": 10, "same": True}}, "no_primary"),
                       ({**own, "giver": 10, "same": True, "f": {"giver": 4, "same": False}}, "same_type"),
                       ({**own, "giver": 4, "f": {"n": 0}}, "no_primary"),
                       ({"n": 0, "pattern": None, "key": None, "f": {**own, "giver": 4}}, "no_primary"),
                       ({"n": 0, "pattern": None, "key": None, "f": {**own, "same": True}}, "same_type")):
        reply, refused = asked(spec)
        assert kinds_of(reply)[0] == [code] and isinstance(refused, P.OutsideDomain), spec
        assert ("saved_suspects" in kinds_of(reply)[2]) == (spec["n"] == 0), spec
    reply, pool = asked({**own, "giver": 10, "f": {"n": 3, "giver": 4}})
    assert (reply["offer"], kinds_of(reply)[2], reply["this_cell"]) == (True, [], {"likely": 2, "suspects": 2})
    assert not pool.caveats


def test_only_the_clicked_cell_acts_on_glowing_points_that_are_not_likely(make_world, monkeypatch):
    """A cell of the clicked one's key whose suspects all glow without being likely is never pooled (it adds no
    suspect: only the clicked cell acts on weak glow, weak_suspects); a pooled cell's weak glow never counts beside its
    likely ones."""
    calls = []
    table = {5: {"key": "OWN", "pattern": "OWN", "n": 2},
             4: {"key": "OWN", "pattern": "OWN", "glow": 5, "giver": 9},     # the nearest: weak glow only
             6: {"key": "OWN", "pattern": "OWN", "n": 1, "glow": 4}}         # one likely, four weak
    world = setup_rule(make_world, monkeypatch, 4, table, calls)
    pool = world.pools.pool("p1", 5)
    assert 4 in calls and pool.cells == (5, 6) and pool.members == 2 and pool.suspects == 3
    assert [item.cell for item in pool.items if item.kind == "target"] == [5, 5, 6]
    assert [c["code"] for c in pool.caveats] == ["too_few"]
    # Clicked, the weak cell acts on its glow, and says so.
    assert kinds_of(world.pools.preview("p1", 4, wait=10))[2] == ["weak_suspects"]


def test_where_it_was_not_measured_the_pool_is_taken_all_the_same_and_says_so(make_world, monkeypatch):
    """Every reason that used to refuse a run where the test was not measured is a caveat now: the preview offers the
    test, pool() takes the pool, and both carry the caveat (the run's as it is now, with the cell's and the pool's);
    the pool is judged with the start's nearest measured row (its k, runs per side and estimate), in the area the
    region gives (a crop of a narrow region, a square of a whole sample, the lasso itself)."""
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3, "giver": 10} for c in range(A.NCELLS)}
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 6)
    monkeypatch.setitem(Q.CONSTANTS["cells"], "k", 6)
    lasso = {"kind": "polygon", "points": [[0, 0], [300, 0], [150, 300]], "bbox": [0, 0, 300, 300]}
    whole = {"kind": "full", "bbox": [0.0, 0.0, 4000.0, 3000.0]}
    square = {"kind": "box", "box": [0.0, 0.0, 500.0, 500.0], "inner": [25.0, 25.0, 475.0, 475.0]}
    cases = [  # (meta changes, the start, its caveats, the row, the area's rule and box)
        ({"continued": True}, "paired", ["past_end"], ("paired", "as_is"), "as_is", [0.0, 0.0, 300.0, 300.0]),
        ({"options": {"samples": 300}}, "paired", ["schedule"], ("paired", "as_is"), "as_is", [0.0, 0.0, 300.0, 300.0]),
        # A whole sample by its cells (qc_constants.run_mode "whole": the world's few cells stand for them here); the
        # same tiled run of its own few cells is a tiled box (H11: one rule, by cell count).
        ({"mode": "tiled", "region": whole, "whole": True}, "paired", ["whole_sample"], ("paired", "square"), "square",
         square["box"]),
        ({"mode": "tiled", "region": whole}, "paired", ["tiled_box"], ("paired", "square"), "square", square["box"]),
        ({"mode": "tiled", "region": box(0, 0, 1300, 1300)}, "paired", ["tiled_box"], ("paired", "square"), "square",
         square["box"]),
        ({"region": lasso}, "paired", ["lasso"], ("paired", "as_is"), "as_is", [0.0, 0.0, 300.0, 300.0]),
        ({"region": box(0, 0, 1200, 300)}, "paired", ["area_narrow"], ("paired", "square"), "crop",
         [0.0, 0.0, 500.0, 300.0]),
        ({"region": box(0, 0, 250, 250)}, "paired", ["region_small"], ("paired", "as_is"), "as_is",
         [0.0, 0.0, 250.0, 250.0]),
        # The cells-only square, measured on proseg's own count: no caveat, its own row; not offered (as it was
        # before), judged with the as-is row and said so.
        ({"region": box(0, 0, 1200, 1200)}, "cells", [], ("cells", "square"), "square", square["box"]),
        ({"region": box(0, 0, 1200, 1200), "withdrawn": True}, "cells", ["square_not_measured"], ("cells", "as_is"),
         "square", square["box"]),
        ({}, "nuclei", ["nuclei_start"], ("paired", "as_is"), "as_is", [0.0, 0.0, 300.0, 300.0]),
        ({"region": box(0, 0, 250, 250)}, "nuclei", ["nuclei_start", "region_small"], ("paired", "as_is"), "as_is",
         [0.0, 0.0, 250.0, 250.0]),
        ({"settings": 50}, "paired", ["settings"], ("paired", "as_is"), "as_is", [0.0, 0.0, 300.0, 300.0])]
    from sopa_companion.engine import qc_constants
    real_mode = qc_constants.run_mode
    for n, (changes, start, caveats, row, rule, area_box) in enumerate(cases):
        world = make_world(**({"meta": {k: v for k, v in changes.items()
                                        if k not in ("settings", "withdrawn", "whole")}}))
        if changes.get("withdrawn"):
            withdrawn(monkeypatch)
        monkeypatch.setattr(qc_constants, "run_mode",
                            (lambda meta, info: "whole") if changes.get("whole") else real_mode)
        fabricate(table, monkeypatch, calls)
        monkeypatch.setattr(qc_rules, "start_of", lambda meta, start=start: start)
        world.analysis.thresholds["judge_min"] = changes.get("settings", 100)
        reply = world.pools.preview("p1", 5, wait=10)
        assert (reply["state"], reply["offer"], reply["reasons"], reply["waits"]) == ("ready", True, [], []), changes
        assert [c["code"] for c in reply["caveats"]] == caveats, changes
        constants = Q.constants_for(*row)
        assert (reply["start"], reply["row"], reply["k"], reply["runs_per_side"], reply["estimate_s"]) == \
            (start, {"start": row[0], "situation": row[1]}, 6, constants["runs_per_side"], P.ESTIMATE_S[row[0]]), \
            changes
        assert (reply["area"]["rule"], reply["area"]["box"]) == (rule, area_box), changes
        assert reply["pool"]["reached"] is True and reply["pool"]["suspects"] >= 6
        pool = world.pools.pool("p1", 5)
        assert [c["code"] for c in pool.caveats] == caveats and pool.caveats == tuple(reply["caveats"]), changes
        assert (pool.start, pool.row, pool.k, pool.area.rule, list(pool.area.box)) == (start, row, 6, rule, area_box)
        assert pool.area.spec == (world.meta["region"] if rule == "as_is" else {"kind": "box", "box": area_box})
        world.pools.close()
    # A run-level caveat follows the run as it is now, over a pool kept as it was taken.
    world = make_world()
    fabricate(table, monkeypatch, calls)
    monkeypatch.setattr(qc_rules, "start_of", lambda meta: "paired")
    assert world.pools.pool("p1", 5).caveats == ()
    computed = world.pools.computed
    world.meta["continued"] = True
    assert [c["code"] for c in world.pools.pool("p1", 5).caveats] == ["past_end"] and world.pools.computed == computed


def test_a_cell_at_the_areas_edge_is_pooled_anyway(make_world, monkeypatch):
    """The clicked cell outside the inner area (edge, a caveat) is the pool's first cell all the same; the other cells
    still need their centre in it, and every primary target its giver's; a cell whose suspects all come from givers
    outside it is still refused (no_primary)."""
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3, "giver": 10} for c in range(A.NCELLS)}
    table[0] = {"key": "OWN", "pattern": "OWN", "n": 3, "giver": 4}
    world = setup_rule(make_world, monkeypatch, 6, table, calls)
    world.meta["region"] = box(90.0, 20.0, 390.0, 320.0)                      # inner x >= 115
    cx, _ = world.centres()
    assert cx[5] < 115 and cx[4] < 115 and cx[10] >= 115
    reply = world.pools.preview("p1", 5, wait=10)
    assert kinds_of(reply) == ([], [], ["edge"]) and reply["offer"] is True
    pool = world.pools.pool("p1", 5)
    assert pool.cells[0] == 5 and all(cx[c] >= 115 for c in pool.cells[1:]) and pool.members == 2
    assert [c["code"] for c in pool.caveats] == ["edge"]
    assert {item.giver for item in pool.items if item.kind == "target"} == {10}
    reply = world.pools.preview("p1", 0, wait=10)
    assert kinds_of(reply) == (["no_primary"], [], ["edge"]) and reply["offer"] is False


# ---- the pool's items, from the real chain ---------------------------------------------------------------------------
KEEP = [0.9, 0.3, 0.75, 0.6, 1.0, 0.5, 0.8, 0.45, 0.7, 0.95, 1.0]        # a keep per gene: proseg's background split


def moved_world(make_world, **options):
    """A world whose proseg moved some transcripts after drift: every third transcript of cell 5's two columns next to
    cell 6 goes to cell 6, every fourth of cell 2's top row to cell 6 too (outside its outline: RNA proseg moved in)."""
    probe = make_world(**options)
    owner = probe.labels_at_f(probe.records)
    col = np.floor((probe.records["x"] - A.X0) / A.VOXEL).astype(int)
    row = np.floor((probe.records["y"] - A.Y0) / A.VOXEL).astype(int)
    into = np.flatnonzero((owner == 6) & (col >= 6))[::3]
    over = np.flatnonzero((owner == 3) & (row == 0))[::4]
    assert into.size and over.size
    return make_world(moved={int(r): 7 for r in np.concatenate([into, over])}, **options), into, over


def check_pool_items(world, pool, answers, how="exact"):
    """The pool's items: its cells' primary targets first (each a transcript proseg counts some of for its cell, w > 0,
    its w the world's to 3 decimals), their matched controls (same cell, giver, near flag and band of w) and the cells'
    own RNA (w > 0); every record's bytes the parent's; the corner's suspects the seed's primary targets as keepSuspects
    lists them."""
    records = records_of(world.live)
    targets = [i for i, item in enumerate(pool.items) if item.kind == "target"]
    assert targets == list(range(len(targets)))
    for item in pool.items:
        r = records[item.record]
        w = world.w_all(item.cell, how)[item.record]
        assert (item.x, item.y, item.zn) == (float(r["x"]), float(r["y"]), float(r["z"]))
        assert item.gene == A.RUN_GENES[int(r["gene"])] and w > 0 and item.w == round(float(w), 3), item
        if item.kind == "target":
            assert item.primary and item.giver is not None and item.start_class is not None and item.p >= 0.2
        elif item.kind == "matched":
            target = pool.items[item.target]
            assert target.kind == "target" and (target.cell, target.giver, target.near) == \
                (item.cell, item.giver, item.near) and item.p is None
            assert band_of(item.w) == band_of(target.w)
        else:
            assert item.giver is None and item.near is None and item.target is None
    seed = answers[pool.seed]
    mine = [t for t in seed["targets"] if t["acting"] and t["giver"] is not None]
    want = sorted(mine, key=lambda t: -js_fixed(t["P"], 3))[:S.SUSPECTS_SENT]
    assert [(s["x"], s["y"], s["gene"], s["p"]) for s in pool.view_suspects] == \
        [(t["x"], t["y"], t["gene"], js_fixed(t["P"], 3)) for t in want]
    assert pool.view_sources == tuple(seed["sources"])
    assert pool.this_cell["likely"] == seed["likely"] and pool.this_cell["line"] == seed["line"]


def test_a_pool_from_the_scoring_and_its_items(make_world, monkeypatch):
    """End to end with the page's scoring (a stand-in) and a small k, on proseg's count with a keep per gene and RNA it
    moved after drift: the pool's items as check_pool_items says; a target may lie outside its cell's outline (moved
    in: its owner another cell's), and one proseg keeps under half of is never likely."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 4)
    world, into, over = moved_world(make_world, page=True, keep=KEEP)
    answers = world.pools.answers("p1", range(11))
    seeds = [c for c, a in answers.items() if a["key"] == "OWN" and any(t["primary"] for t in a["targets"])]
    assert seeds
    pool = world.pools.pool("p1", seeds[0])
    check_pool_items(world, pool, answers)
    targets = [t for a in answers.values() for t in a["targets"]]
    assert {t["owner"] for t in targets} >= {1, 255} and any(t["w"] < 0.5 and not t["likely"] for t in targets)
    assert all(t["likely"] == (t["pi"] >= 0.5 and t["w"] >= 0.5) for t in targets)
    moved_in = [t for t in answers[6]["targets"] if t["record"] in set(over.tolist())]
    assert moved_in and all(t["owner"] == 255 for t in moved_in)


def test_a_pool_from_the_real_scoring_and_its_items(make_world, monkeypatch):
    """The real port and where engine end to end with a small k, on proseg's count keeping 3 in 4 of every gene, with
    RNA it moved after drift (where the synthetic model's page asks for a neighbour's gene): the pool of every cell that
    glows with a primary target, its items as check_pool_items says; likely as the page's rule (π >= LIKELY_MIN with
    w >= 0.5); every reply exact (the parent held at F)."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 4)
    world, _, _ = moved_world(make_world, keep=0.75)
    world.where = world.pools.where = DeeperWhere(world.runs)
    answers = world.pools.answers("p1", range(11))
    seeds = [c for c, a in answers.items() if any(t["primary"] for t in a["targets"])]
    assert seeds
    for seed in seeds:
        check_pool_items(world, world.pools.pool("p1", seed), answers)
    targets = [t for a in answers.values() for t in a["targets"]]
    assert all(t["likely"] == (np.float32(t["pi"]) >= np.float32(S.LIKELY_MIN) and t["w"] >= 0.5) for t in targets)
    assert {a["held"] for a in answers.values() if a["request"]} == {"exact"}


def test_every_source_of_the_corner_has_its_name_before_and_after_matching(make_world, monkeypatch):
    """The seed's where-request sources, which the corner's views find by name in each test run (engine/qc_test.py),
    are named in the pool and in the matched pool, a source that gives none of the suspects too (here 6 and 9: every
    gene's giver is 4; 9 is no cell of the pool)."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 1)
    world = make_world(page=True)
    genes = top_genes(world, 5, 2)
    world.page.genes, world.page.sources = genes, [4, 6, 9]
    monkeypatch.setattr(P, "givers_of", lambda result: dict.fromkeys(genes, 4))
    pool = world.pools.pool("p1", 5)
    assert pool.view_sources == (4, 6, 9) and 9 not in pool.cells
    assert {item.giver for item in pool.items if item.giver is not None} == {4}
    want = {s: D.RUN_IDS[s] for s in (4, 6, 9)}
    assert {s: pool.names.get(s) for s in pool.view_sources} == want
    records = records_of(world.live)
    rows = [{"id": 1000 + i, "gene": A.RUN_GENES[int(r["gene"])], "x": float(r["x"]), "y": float(r["y"]),
             "z": 2.0 + 5.0 * float(r["z"])} for i, r in enumerate(records)]
    matched = world.pools.match_input(pool, write_input(world.live.parent / "in.parquet", rows), {})
    assert matched.counts["target"]["ok"] >= 1 and matched.pool.view_sources == (4, 6, 9)
    assert {s: matched.pool.names.get(s) for s in matched.pool.view_sources} == want
    assert set(matched.pool.cells) <= set(matched.pool.names)


# ---- the controls ----------------------------------------------------------------------------------------------------
def test_controls_are_drawn_per_cell_whatever_the_area(make_world, monkeypatch):
    """3 per likely target with a placed giver, in point order, from the cell's own records with the same near flag to
    that giver, seeded [20261028, c, 0]; the genes asked as a neighbour's left out; 50 of the cell's own RNA seeded
    [20261028, c, 1], for cells with a target only. The same draws in another area (a 500 µm square)."""
    first = make_world(page=True)
    seen = spy_deeper(monkeypatch)
    genes = top_genes(first, 5, 2)
    first.page.genes, first.page.sources = genes, [4, 6]
    monkeypatch.setattr(P, "givers_of", lambda result: {genes[0]: 4, genes[1]: 6})
    first.page.p = lambda gene, near: 0.8 if gene == genes[0] else (0.6 if near else 0.3)
    answers = first.pools.answers("p1", [5, 11])
    records = records_of(first.live)
    a = answers[5]
    assert a["targets"] and a["controls"]["matched"] and len(a["controls"]["uniform"]) == 50
    assert a["controls"]["excluded"] == sorted(set(genes) | tagged_of(seen[5]))
    check_controls(first, a, records, sorted(set(genes) | tagged_of(seen[5])))
    assert {m["point"] for m in a["controls"]["matched"]} == \
        {t["point"] for t in a["targets"] if t["likely"] and t["giver"] is not None}
    assert any(not t["likely"] for t in a["targets"])                # 0.3: a target, no controls of its own
    drawn = {A.RUN_GENES[int(records[r]["gene"])] for m in a["controls"]["matched"] for r in m["records"]}
    drawn |= {A.RUN_GENES[int(records[r]["gene"])] for r in a["controls"]["uniform"]}
    assert not drawn & set(a["controls"]["excluded"])
    assert answers[11]["targets"] == [] and answers[11]["controls"]["uniform"] == []      # no target: none
    second = make_world(page=True, region=(-200.0, -200.0, 900.0, 900.0))
    second.page.genes, second.page.sources, second.page.p = genes, [4, 6], first.page.p
    again = second.pools.answers("p1", [5])[5]
    assert again["controls"] == a["controls"] and again["targets"] == a["targets"]


def test_controls_leave_out_the_genes_tagged_as_a_neighbours_even_unasked(make_world, monkeypatch):
    """The gene rule's other half: a gene Go deeper's answer tags as a neighbour's (its row's tag 1) is left out of the
    controls though the where request does not ask it (the page here asks only the cell's most frequent gene); the
    genes left out are worked out from the worker's own answer, and the cell holds records of the tagged ones at F."""
    world = make_world(page=True)
    seen = spy_deeper(monkeypatch)
    gene = top_genes(world, 5, 1)[0]
    world.page.genes, world.page.sources = [gene], [4]
    monkeypatch.setattr(P, "givers_of", lambda result: {gene: 4})
    a = world.pools.answers("p1", [5])[5]
    records = records_of(world.live)
    names = np.asarray(A.RUN_GENES, object)[records["gene"].astype(np.int64)]
    mine = world.labels_at_f(records) == 6
    tagged = tagged_of(seen[5]) - {gene}                      # tagged, not asked
    assert tagged and all((mine & (names == g)).any() for g in tagged)
    excluded = sorted({gene} | tagged)
    assert a["controls"]["excluded"] == excluded
    assert a["targets"] and a["controls"]["matched"] and len(a["controls"]["uniform"]) == 50
    check_controls(world, a, records, excluded)
    drawn = [r for m in a["controls"]["matched"] for r in m["records"]] + a["controls"]["uniform"]
    assert drawn and not set(names[drawn]) & set(excluded)


def test_a_shortfall_is_counted(make_world, monkeypatch):
    """Every gene asked as a neighbour's: no own record is left to draw from, and each likely, placed target's 3
    controls are counted short."""
    world = make_world(page=True)
    world.page.genes, world.page.sources = list(A.RUN_GENES), [4]
    monkeypatch.setattr(P, "givers_of", lambda result: dict.fromkeys(A.RUN_GENES, 4))
    a = world.pools.answers("p1", [5])[5]
    placed = [t for t in a["targets"] if t["likely"] and t["giver"] is not None]
    assert placed and a["controls"]["short"] == 3 * len(placed)
    assert all(m["records"] == [] for m in a["controls"]["matched"]) and a["controls"]["uniform"] == []


def test_a_gene_proseg_keeps_none_of_gives_no_control(make_world, monkeypatch):
    """A gene the cell holds whose averaged count proseg stores as 0 (a keep that rounds to nothing: here 0): the cell's
    records of it have w = 0, so none is a control, matched or of its own RNA, though the cell holds many and the gene
    is neither asked nor tagged as a neighbour's; the controls are drawn as the module says from the rest."""
    probe = make_world()
    genes = top_genes(probe, 5, 4)
    records = records_of(probe.live)
    mine = probe.labels_at_f(records) == 6
    monkeypatch.setattr(P, "givers_of", lambda result: {genes[0]: 4})
    seen = spy_deeper(monkeypatch)
    for gene in genes[1:]:                                   # the first the answer neither asks nor tags
        zero = A.RUN_GENES.index(gene)
        keep = np.ones(len(A.RUN_GENES))
        keep[zero] = 0.0
        world = make_world(page=True, keep=keep)
        world.page.genes, world.page.sources = [genes[0]], [4]
        a = world.pools.answers("p1", [5])[5]
        if gene not in a["controls"]["excluded"]:
            break
    else:
        pytest.fail("every gene of the cell is a neighbour's here")
    assert int((mine & (records["gene"] == zero)).sum()) >= 10 and (world.w_all(5)[mine & (records["gene"] == zero)]
                                                                  == 0).all()
    drawn = [r for m in a["controls"]["matched"] for r in m["records"]] + a["controls"]["uniform"]
    assert a["targets"] and drawn and not [r for r in drawn if records["gene"][r] == zero]
    assert a["owned"] == int((world.w_all(5) > 0).sum())
    check_controls(world, a, records, sorted({genes[0]} | tagged_of(seen[5])))


def test_controls_are_matched_on_the_band_of_proseg_s_share(make_world, monkeypatch):
    """With a keep per gene and RNA proseg moved: each acting target's 3 controls come from the cell's records proseg
    counts some of (w > 0, never one it counts none of), of its giver's near flag and its band of w (under 0.25, 0.25
    to 0.75, 0.75 or more, each to 3 decimals), every control's w kept; the draws as the module says."""
    world, into, over = moved_world(make_world, page=True, keep=KEEP)
    seen = spy_deeper(monkeypatch)
    answers = world.pools.answers("p1", range(11))
    records = records_of(world.live)
    bands = set()
    for c, a in answers.items():
        if not a["targets"]:
            continue
        check_controls(world, a, records, sorted(set(a["controls"]["excluded"]) | tagged_of(seen[c])))
        bands |= {int(band_of(t["w"])) for t in a["targets"] if t["acting"]}
        drawn = [r for m in a["controls"]["matched"] for r in m["records"]] + a["controls"]["uniform"]
        assert all(world.w_all(c)[r] > 0 for r in drawn) and a["owned"] == int((world.w_all(c) > 0).sum())
    assert bands == {1, 2}                                   # targets of more than one band were matched


# ---- proseg's share of each transcript: the test area's basis -------------------------------------------------------
def basis_of(world, cell: int = 5, wait=10):
    """The preview of the cell, with the test area's basis as the pool's frame context holds it."""
    reply = world.pools.preview("p1", cell, wait=wait)
    context = world.pools.work["p1"].context
    return reply, context


def own_asks(monkeypatch) -> list:
    """The boxes the pools' frame work itself asks proseg to export (QcPools._ask_where), apart from the where engine's
    own asks for a cell's box (Go deeper's measurement asks for one while the run is paused at its frame)."""
    asked, real = [], P.QcPools._ask_where

    def ask(self, run_id, box):
        asked.append([float(v) for v in box])
        return real(self, run_id, box)
    monkeypatch.setattr(P.QcPools, "_ask_where", ask)
    return asked


GRID_BOX = [A.X0, A.Y0, A.X0 + A.NX * A.VOXEL, A.Y0 + A.NY * A.VOXEL]     # F's grid, µm: the area as it lies on it


def test_held_at_f_the_area_s_shares_are_exact_from_one_export(make_world, monkeypatch):
    """The parent held at F: the frame work asks proseg once for an export of the test area (as it lies on F's grid,
    waiting for it up to WHERE_WAIT_S), and a cell's w is held.weights' exact share from it: its pools' buckets whole,
    the same as over every record of the run, for every cell, moved RNA counted for the cell it went to; the same area
    asked again uses it, without asking again. No caveat."""
    world, into, over = moved_world(make_world, page=True, keep=KEEP)
    own = own_asks(monkeypatch)
    reply, context = basis_of(world)
    assert own == [GRID_BOX] and ("p1", GRID_BOX, P.WHERE_WAIT_S) in world.runs.asked
    assert reply["offer"] is True and kinds_of(reply)[2] == []
    weights = next(iter(context.held.values()))
    assert (context.basis.held, context.basis.seq, weights.held) == ("exact", SEQ, "exact")
    assert context.key[-2:] == (SEQ, "exact")
    for c in range(A.NCELLS):
        idx, w = weights.of(c, context.q(c))
        whole = world.w_all(c, "exact")
        assert idx.tolist() == np.flatnonzero(whole > 0).tolist(), c
        assert np.allclose(w, whole[idx], rtol=0, atol=1e-12), c
    moved, genes = np.concatenate([into, over]), world.records["gene"].astype(np.int64)
    assert (world.w_all(6)[moved] > 0).all()                     # counted for the cell they went to ...
    assert (world.w_all(5)[into] < world.q(5)[genes[into]]).any()    # ... and less for the one they left
    assert not np.allclose(world.w_all(6, "exact"), world.w_all(6, "approx"))      # exact is not the outline's rule
    world.pools.preview("p1", 3, wait=10)
    assert own == [GRID_BOX] and world.pools.contexts == 1 and context.bases[tuple(world.meta["region"]["bbox"])] \
        is context.basis


def test_an_export_gone_before_it_is_read_is_asked_again(make_world, monkeypatch):
    """The area's export found, but gone before the frame work reads it (proseg keeps only its 4 newest): asked of the
    parent again, once, and read; a second loss fails the pool in plain words."""
    world = make_world(page=True)
    real, lost = held.read_where, []

    def read_where(meta, path=None):
        if len(lost) < 1:                                  # proseg let it go just before the read
            lost.append(meta["token"])
            Path(meta["path"]).unlink()
        return real(meta, path)
    monkeypatch.setattr(held, "read_where", read_where)
    reply, context = basis_of(world)
    assert reply["offer"] is True and lost == [1] and len(world.runs.asked) == 2
    assert next(iter(context.held.values())).export.token == 2
    world.pools.forget("p1")
    lost.clear()
    monkeypatch.setattr(held, "read_where", lambda meta, path=None: (_ for _ in ()).throw(ValueError("it broke")))
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["state"] == "failed" and reply["message"] == f"{P.FAILED}: it broke"


def test_a_parent_past_f_places_its_suspects_approximately_with_the_caveat(make_world, monkeypatch):
    """The parent sampled on past F and holds no export at F: nothing is asked of it for F, and w is F's per-layer
    labels times the cell's keep at F (approx), for every cell; the test runs with the caveat approx_targets, which a
    preview and the pool carry. An export of the area at F the engine still holds is exact again, without the
    caveat."""
    world, _, _ = moved_world(make_world, page=True, keep=KEEP)
    world.moved_on()
    own = own_asks(monkeypatch)
    reply, context = basis_of(world)
    assert own == [] and world.runs.asked == [] and context.key[-2:] == (SEQ, "approx")
    assert kinds_of(reply) == ([], [], ["past_end", "approx_targets"])
    weights = next(iter(context.held.values()))
    for c in range(A.NCELLS):
        idx, w = weights.of(c, context.q(c))
        whole = world.w_all(c, "approx")
        assert idx.tolist() == np.flatnonzero(whole > 0).tolist() and np.allclose(w, whole[idx]), c
    pool = world.pools.pool("p1", 5)
    assert [c["code"] for c in pool.caveats] == ["past_end", "approx_targets"]
    # An export of the area at F kept by the engine: exact, as while held.
    held_at_f, _, _ = moved_world(make_world, page=True, keep=KEEP)
    held_at_f.pools.preview("p1", 5, wait=10)
    kept = list(held_at_f.run.wheres)
    held_at_f.pools.forget("p1")
    held_at_f.moved_on()
    held_at_f.run.wheres, held_at_f.where_held = kept, "exact"
    reply, context = basis_of(held_at_f)
    assert context.key[-2:] == (SEQ, "exact") and kinds_of(reply)[2] == ["past_end"]
    assert [box for _, box, _ in held_at_f.runs.asked] == [GRID_BOX]      # the one ask while it was held


def test_past_f_a_dense_frame_places_each_transcript_by_its_own_layer(make_world, monkeypatch):
    """A dense F whose layers differ (cell 6 holds cell 5's two right columns in the top layer only; proseg's count
    follows its layers): past F, w is F's label at each record's own layer times the cell's keep, never its column's
    (the consensus, or one layer for all): a top-layer transcript there counts for cell 6 alone, a lower one for cell 5
    alone; every cell's shares are held.weights' from those layers."""
    layers = np.stack([A.frame_plane(SEQ)] * A.NZ)
    layers[1, 4:8, 6:8] = 7                                  # cell 5's (label 6) right columns, top layer: cell 6's
    world = make_world(page=True, keep=KEEP, layers=layers)
    world.moved_on()
    reply, context = basis_of(world)
    assert context.key[-2:] == (SEQ, "approx") and "approx_targets" in kinds_of(reply)[2]
    assert np.array_equal(context.layers, layers)
    weights = next(iter(context.held.values()))
    for c in range(A.NCELLS):
        idx, w = weights.of(c, context.q(c))
        whole = world.w_all(c, "approx")
        assert idx.tolist() == np.flatnonzero(whole > 0).tolist() and np.allclose(w, whole[idx]), c
    i, j, k, inside = CF.observed_voxels(world.records, GRID, Z0, DZ)
    there = inside & (j >= 4) & (j < 8) & (i >= 6) & (i < 8)
    top, low = set(np.flatnonzero(there & (k == 1)).tolist()), set(np.flatnonzero(there & (k == 0)).tolist())
    five, six = (set(weights.of(c, context.q(c))[0].tolist()) for c in (5, 6))
    assert top and low and top <= six and low <= five and not top & five and not low & six


def test_targets_placed_approximately_by_their_where_replies_say_so_whatever_the_area(make_world, monkeypatch):
    """The area's shares exact (the parent held at F: one export of the area), but a cell's own where reply placed its
    targets approximately (the where engine places the cell's box itself: a box the area's export does not cover, or
    one asked once the parent moved on): the test says approx_targets, for the clicked cell's reply (a refusal's
    reply too) and for a pooled cell's (a member's or the reserve's), and nothing else changes here (F's labels are
    proseg's assignment, so approx gives the shares exact gives); with every reply exact, it says nothing."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 100)
    world = make_world(page=True)
    exact = world.pools.pool("p1", 5)
    assert exact.caveats == () and exact.members == 2 and len(exact.cells) > 2        # the seed 5, 4; then the reserve
    member, spare = exact.cells[1], exact.cells[-1]
    for placed in ({5: "approx"}, {member: "approx"}, {spare: "approx"}):
        world.pools.forget("p1")
        world.held_by = placed
        reply, context = basis_of(world)
        assert (context.basis.held, reply["offer"]) == ("exact", True), placed
        assert kinds_of(reply)[2] == ["approx_targets"], placed
        pool = world.pools.pool("p1", 5)
        assert pool.caveats == (P.reason("approx_targets"),), placed
        assert replace(pool, caveats=()) == exact, placed
    # A refused cell's reply says how it was placed too: here nothing of it glows (every π 0.1).
    world.pools.forget("p1")
    world.held_by, world.page.p = {5: "approx"}, lambda gene, near: 0.1
    assert kinds_of(world.pools.preview("p1", 5, wait=10)) == (["no_suspects"], [], ["approx_targets"])


def test_a_seed_asked_after_the_parent_moved_on_carries_the_caveat_of_its_own_reply(make_world, monkeypatch):
    """The parent held at F: a pool is taken (the area's export asked once, its shares kept with the frame context).
    The parent then samples on past F, so the where engine places a cell's box from F's labels: a cell asked for after
    that (its answer not worked out before) is tested on the area's exact shares kept, nothing asked again, with the
    caveats past_end and approx_targets (its targets placed approximately); a cell worked out while held keeps its exact
    targets (past_end alone)."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 1)
    world = make_world(page=True)
    own = own_asks(monkeypatch)
    assert kinds_of(world.pools.preview("p1", 5, wait=10))[2] == []
    worked = {key[-1] for key in world.pools.answers_kept}
    assert 3 not in worked and 5 in worked
    world.moved_on()
    reply, context = basis_of(world, 3)
    assert (context.basis.held, world.pools.contexts, own) == ("exact", 1, [GRID_BOX])
    assert kinds_of(reply) == ([], [], ["past_end", "approx_targets"])
    assert world.pools.answers_kept[(*context.key, 3)].held == "approx"
    with world.pools.lock:                          # the seed 5's pool worked out again, from the answers kept
        world.pools.pools_kept.clear()
    assert kinds_of(world.pools.preview("p1", 5, wait=10)) == ([], [], ["past_end"])
    assert world.pools.answers_kept[(*context.key, 5)].held == "exact"


def test_a_stale_ack_never_passes_a_later_frame_s_export_off_as_f_s(make_world, monkeypatch):
    """The ack still says paused at F, but the parent samples on (a stale ack): the export asked comes with a later
    counted frame, which is not F's, so the area's shares are F's labels times q (approx_targets), never that frame's
    export read as F's."""
    world = make_world(page=True)
    world.runs.served_at = SEQ + 1
    world.where_held = "approx"
    own = own_asks(monkeypatch)
    reply, context = basis_of(world)
    assert own == [GRID_BOX] and world.run.wheres[-1]["seq"] == SEQ + 1
    assert context.key[-2:] == (SEQ, "approx") and kinds_of(reply)[2] == ["approx_targets"]
    weights = next(iter(context.held.values()))
    assert (weights.held, weights.export) == ("approx", None)


def test_each_test_area_of_a_frame_context_has_its_own_export_and_shares(make_world, monkeypatch):
    """One frame context, two test areas (as the windows around two cells of a region over 500 µm: here F's grid cut in
    two halves, each the area of a pool taken there): each area asks proseg once for its own export (its box on F's
    grid) and holds its own shares and basis, a cell of one half holding none in the other's export; an area asked again
    uses its own, without asking."""
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 1)
    world = make_world()
    # Cells 0 and 1 (columns 0-7) in the left half, each the other's giver; 3 and 2 (columns 8-15) in the right one.
    fabricate({0: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 1},
               1: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 0},
               3: {"key": "OWN", "pattern": "OWN", "n": 2, "giver": 2}}, monkeypatch)
    own = own_asks(monkeypatch)
    half = A.X0 + A.NX * A.VOXEL / 2
    left = (A.X0, A.Y0, half, A.Y0 + A.NY * A.VOXEL)
    right = (half, A.Y0, A.X0 + A.NX * A.VOXEL, A.Y0 + A.NY * A.VOXEL)

    def take(cell, box):
        facts = world.pools._facts_at_end(world.run)
        facts.area = P.Area("square", box, box, {"kind": "box", "box": list(box)})
        job = world.pools._submit("p1", P._Job(world.pools._pool_key(facts, cell), facts, cell=cell))
        assert job.done.wait(60) and not job.dropped and isinstance(job.outcome, P._Outcome), job.outcome
        assert job.outcome.reasons == [] and job.outcome.pool.cells[0] == cell
        return job
    first, second = take(0, left), take(3, right)
    context = world.pools.work["p1"].context
    assert own == [list(left), list(right)] and world.pools.contexts == 1
    assert (first.basis.where["box"], second.basis.where["box"]) == (list(left), list(right))
    assert context.bases[left] is first.basis and context.bases[right] is second.basis
    on_left, on_right = context.held[left], context.held[right]
    assert (on_left.export.token, on_right.export.token) == (first.basis.where["token"], second.basis.where["token"])
    assert (on_left.export.i0, on_left.export.i1, on_right.export.i0, on_right.export.i1) == (0, 8, 8, 15)
    for cell, mine, other in ((0, on_left, on_right), (3, on_right, on_left)):
        assert mine.of(cell, context.q(cell))[0].size > 0 and other.of(cell, context.q(cell))[0].size == 0, cell
    again = take(1, left)                                            # another cell of the left half: its own again
    assert own == [list(left), list(right)] and again.basis is first.basis


def test_an_area_with_no_column_of_the_run_has_no_shares_and_nothing_glows(make_world, monkeypatch):
    """A test area that holds no column of F's grid (here F's grid moved away from the region): its basis is none, every
    w 0, so nothing glows: no_suspects, nothing asked of proseg for the area (the where engine may still ask for the
    cell's own box)."""
    world = make_world(page=True, region=(0.0, 0.0, 90.0, 90.0))
    frame = world.run.frames[SEQ]
    frame["x0"], frame["y0"] = 500.0, 500.0
    own = own_asks(monkeypatch)
    reply, context = basis_of(world)
    assert context.key[-2:] == (SEQ, "none") and own == []
    assert not [box for _, box, _ in world.runs.asked if box == [0.0, 0.0, 90.0, 90.0]]
    assert kinds_of(reply)[0] == ["no_suspects"]


def test_a_last_frame_without_proseg_s_count_says_so(make_world):
    """F's record without a counts file (proseg's own count missing for it): the pool cannot be worked out, said in
    plain words."""
    world = make_world(page=True)
    world.run.frames[SEQ].pop("counts_file")
    reply = world.pools.preview("p1", 5, wait=10)
    assert (reply["state"], reply["message"]) == \
        ("failed", f"{P.FAILED}: your run's last iteration has no count of proseg's own")


def test_count_basis_where_the_row_is_not_measured_on_proseg_s_count(make_world, monkeypatch):
    """A test judged with a row measured with suspects chosen on the outlines says so (count_basis, a caveat: never a
    refusal or a wait): a start from nuclei and cells, whose rows were not measured again on proseg's own count
    (qc_rule.measured_on_proseg). One from cells only, as is or in a square, was (2026-10-31): none, unless the
    constants' id is an outline one (every row then)."""
    world = make_world(page=True)
    monkeypatch.setattr(Q, "measured_on_proseg", MEASURED_ON_PROSEG)
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["row"] == {"start": "paired", "situation": "as_is"}
    assert reply["offer"] is True and kinds_of(reply)[2][0] == "count_basis"
    assert "count_basis" in [c["code"] for c in world.pools.pool("p1", 5).caveats]
    assert world.pools.prerequisites("p1") == ([], [])
    world.meta.update(state="running", schedule_done=None)
    assert kinds_of(world.pools.preview("p1", 5)) == ([], ["not_at_end"], ["count_basis"])
    world.meta["region"] = box(0, 0, 1200, 1200)                       # its square row: not measured again either
    assert kinds_of(world.pools.preview("p1", 5)) == ([], ["not_at_end"], ["count_basis"])
    monkeypatch.setattr(qc_rules, "start_of", lambda meta: "cells")
    for region in (box(0, 0, 1200, 1200), box(0, 0, 300, 300)):        # cells only, square and as is: measured
        world.meta["region"] = region
        assert kinds_of(world.pools.preview("p1", 5)) == ([], ["not_at_end"], []), region
    monkeypatch.setattr(Q, "OUTLINE_IDS", (*OUTLINE_IDS, Q.CONSTANTS_ID))     # the constants under an outline id
    assert kinds_of(world.pools.preview("p1", 5)) == ([], ["not_at_end"], ["count_basis"])


def test_the_outline_ids_keep_the_outline_measured_set_and_its_rows_say_which_were_measured_again():
    """The set measured with suspects chosen on the outlines (phase 1a) keeps its id among the outline ones; the
    constants now are another set, measured on proseg's own count (2026-10-31) for starts from cells only, its rows for
    starts from nuclei and cells still phase 1a's."""
    assert OUTLINE_IDS == ("phase1a-2026-10-29",) and Q.CONSTANTS_ID not in OUTLINE_IDS
    assert {(start, situation): MEASURED_ON_PROSEG(start, situation) for start in Q.STARTS
            for situation in Q.SITUATIONS} == {("paired", "as_is"): False, ("paired", "square"): False,
                                               ("cells", "as_is"): True, ("cells", "square"): True}


# ---- matching on the shared input ------------------------------------------------------------------------------------
def write_input(path: Path, rows: list[dict]):
    """A shared input in the helper's schema (zarr_bridge/helper.py prepare): transcript_id u64, cell_id,
    overlaps_nucleus u8, feature_name, x/y/z_location f32, qv f32, and the rest."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(rows)
    table = pa.table({
        "transcript_id": pa.array([r["id"] for r in rows], pa.uint64()),
        "cell_id": pa.array(["UNASSIGNED"] * n, pa.large_string()),
        "overlaps_nucleus": pa.array(np.zeros(n, np.uint8)),
        "feature_name": pa.array([r["gene"] for r in rows], pa.large_string()),
        "x_location": pa.array(np.array([r["x"] for r in rows], np.float32)),
        "y_location": pa.array(np.array([r["y"] for r in rows], np.float32)),
        "z_location": pa.array(np.array([r.get("z", 1.0) for r in rows], np.float32)),
        "qv": pa.array(np.array([r.get("qv", 40.0) for r in rows], np.float32)),
        "fov_name": pa.array(["0"] * n, pa.large_string()),
        "nucleus_distance": pa.array(np.zeros(n, np.float32)),
        "codeword_index": pa.array(np.zeros(n, np.int32)),
        "codeword_category": pa.array(["predesigned_gene"] * n, pa.large_string()),
        "is_gene": pa.array(np.ones(n, bool)),
    })
    pq.write_table(table, path, compression="zstd")
    return path


class Parent:
    """A parent run for matching alone: live/transcripts.bin.gz (x, y, z normalised, gene) and genes.txt."""

    def __init__(self, folder: Path, records: list[tuple], genes: list[str]):
        live = folder / "live"
        live.mkdir(parents=True)
        self.records = np.array(records, RECORD)
        (live / "transcripts.bin.gz").write_bytes(gzip.compress(self.records.tobytes()))
        (live / "genes.txt").write_text("".join(g + "\n" for g in genes))
        (live / "info.json").write_text(json.dumps({"cells": 10}))
        (live / "frames.jsonl").write_text("")
        self.genes = genes
        self.run = Run(folder, {"id": "p1", "kind": "proseg"})
        self.pools = P.QcPools(Runs(self.run), None, None, DeeperWhere(Runs(self.run)))

    def item(self, kind, cell, record, **values) -> P.Item:
        r = self.records[record]
        defaults = {"target": None, "giver": 9 if kind != "uniform" else None, "near": False if kind != "uniform"
                    else None, "p": 0.7 if kind == "target" else None, "primary": kind == "target",
                    "start_class": "OWN" if kind == "target" else None}
        return P.Item(kind=kind, cell=cell, gene=self.genes[int(r["gene"])], record=int(record), x=float(r["x"]),
                      y=float(r["y"]), zn=float(r["z"]), **{**defaults, **values})


def hand_pool(items, cells, k, members=None, distance=None, caveats=()) -> P.Pool:
    names = {c: f"cell{c}-1" for c in set(cells) | {i.giver for i in items if i.giver is not None}}
    return P.Pool(v=1, run="p1", seq=SEQ, iter=ITER, model=MODEL, constants=Q.CONSTANTS_ID, start="paired",
                  area=P.area_of(box(0, 0, 300, 300), (0.0, 0.0)), k=k, seed=cells[0], seed_name=names[cells[0]],
                  cells=tuple(cells), members=members or len(cells),
                  distance_um=tuple(distance or [float(i) for i in range(len(cells))]), names=names,
                  items=tuple(items), suspects=sum(i.kind == "target" for i in items), available=None,
                  record_duplicates=0, this_cell={}, view_suspects=(), view_sources=(), row=("paired", "as_is"),
                  caveats=tuple(caveats))


def test_matching_by_exact_bits_and_the_parents_input_rule(tmp_path):
    """An item is the input row of its gene whose x and y hold the same float32 bits (a row one ulp away is not it);
    rows proseg leaves out (qv under min-qv, a gene the excluded pattern finds) are never found; the ids are the input's
    own (here renumbered rows)."""
    genes = ["AAA", "BBB", "NegControlProbe_7"]
    parent = Parent(tmp_path / "p1", [(10.25, 20.5, 0.3, 0), (11.0, 21.0, 0.4, 1), (12.0, 22.0, 0.5, 0),
                                      (13.0, 23.0, 0.6, 1), (14.0, 24.0, 0.2, 0)], genes)
    items = [parent.item("target", 1, 0), parent.item("target", 1, 1), parent.item("matched", 1, 2, target=0),
             parent.item("uniform", 1, 3), parent.item("uniform", 1, 4)]
    x1 = float(np.nextafter(np.float32(11.0), np.float32(12.0)))
    rows = [{"id": 7, "gene": "AAA", "x": 10.25, "y": 20.5, "z": 3.1},
            {"id": 8, "gene": "BBB", "x": x1, "y": 21.0, "z": 3.2},                    # one ulp off: not it
            {"id": 9, "gene": "AAA", "x": 12.0, "y": 22.0, "z": 3.3},
            {"id": 10, "gene": "BBB", "x": 13.0, "y": 23.0, "qv": 19.5},              # under min qv 20
            {"id": 11, "gene": "AAA", "x": 14.0, "y": 24.0},
            {"id": 12, "gene": "NegControlProbe_7", "x": 12.0, "y": 22.0}]
    path = write_input(tmp_path / "in.parquet", rows)
    pool = hand_pool(items, [1], k=1)
    matched = parent.pools.match_input(pool, path, {})
    assert matched.counts == {"target": {"ok": 1, "not_found": 1, "unsure": 0},
                              "matched": {"ok": 1, "not_found": 0, "unsure": 0},
                              "uniform": {"ok": 1, "not_found": 1, "unsure": 0}}
    assert matched.items.transcript_id.tolist() == [7, 9, 11]
    assert matched.items.kind.tolist() == [R.SUSPECT, R.MATCHED, R.UNIFORM]
    assert matched.items.target.tolist() == [-1, 0, -1]
    assert matched.items.x.dtype == np.float32 and matched.items.x.tolist() == [10.25, 12.0, 14.0]
    assert matched.input == str(path) and matched.pool.suspects == 1 and matched.seconds >= 0
    # Renumbered ids (an input whose own were not unique): the rows' numbers are the ids the runs carry.
    renumbered = write_input(tmp_path / "rows.parquet", [{**r, "id": n} for n, r in enumerate(rows)])
    assert parent.pools.match_input(pool, renumbered, {}).items.transcript_id.tolist() == [0, 2, 4]
    # min-qv 19 takes the low-qv row.
    assert parent.pools.match_input(pool, path, {"min-qv": 19.0}).counts["uniform"] == \
        {"ok": 2, "not_found": 0, "unsure": 0}
    # A control probe's row: left out by default (the app's excluded genes), kept with Include neg ctrls or another
    # pattern, which replaces the default.
    probe = P.Item(**{**parent.item("uniform", 1, 2).__dict__, "gene": "NegControlProbe_7"})
    controls = hand_pool([parent.item("target", 1, 0), probe], [1], k=1)
    assert parent.pools.match_input(controls, path, {}).counts["uniform"] == {"ok": 0, "not_found": 1, "unsure": 0}
    assert parent.pools.match_input(controls, path, {"include-neg-ctrls": True}).counts["uniform"]["ok"] == 1
    assert parent.pools.match_input(controls, path, {"excluded-genes": "^ZZZ"}).counts["uniform"]["ok"] == 1
    assert parent.pools.match_input(controls, path, {"excluded-genes": "Probe"}).counts["uniform"]["not_found"] == 1


def test_duplicates_by_depth_rank_else_unsure(tmp_path):
    """Records at one gene and place: told apart by depth rank (the parent's normalised z, the input's raw z) when the
    input holds as many there and no two depths on either side lie within 1e-3; else unsure. A matched control goes
    with its target."""
    genes = ["AAA"]
    parent = Parent(tmp_path / "p1", [(5.0, 5.0, 0.9, 0), (5.0, 5.0, 0.1, 0), (5.0, 5.0, 0.5, 0),
                                      (6.0, 6.0, 0.2, 0), (6.0, 6.0, 0.2004, 0), (7.0, 7.0, 0.3, 0),
                                      (7.0, 7.0, 0.6, 0), (8.0, 8.0, 0.1, 0)], genes)
    items = [parent.item("target", 2, 0), parent.item("target", 2, 1), parent.item("target", 2, 2),
             parent.item("target", 2, 3), parent.item("target", 2, 5), parent.item("matched", 2, 7, target=3)]
    rows = [{"id": 100, "gene": "AAA", "x": 5.0, "y": 5.0, "z": 12.0}, {"id": 101, "gene": "AAA", "x": 5.0, "y": 5.0,
                                                                         "z": 30.0},
            {"id": 102, "gene": "AAA", "x": 5.0, "y": 5.0, "z": 20.0},
            {"id": 103, "gene": "AAA", "x": 6.0, "y": 6.0, "z": 1.0}, {"id": 104, "gene": "AAA", "x": 6.0, "y": 6.0,
                                                                        "z": 2.0},
            {"id": 105, "gene": "AAA", "x": 7.0, "y": 7.0, "z": 1.0}, {"id": 106, "gene": "AAA", "x": 7.0, "y": 7.0,
                                                                        "z": 2.0},
            {"id": 107, "gene": "AAA", "x": 7.0, "y": 7.0, "z": 3.0},                    # three here, two there
            {"id": 108, "gene": "AAA", "x": 8.0, "y": 8.0, "z": 1.0}]
    path = write_input(tmp_path / "in.parquet", rows)
    matched = parent.pools.match_input(hand_pool(items, [2], k=1), path, {})
    # z 0.9, 0.1, 0.5 in the parent are 30, 12, 20 in the input: ranks 2, 0, 1.
    assert matched.items.transcript_id.tolist() == [101, 100, 102]
    assert matched.counts["target"] == {"ok": 3, "not_found": 0, "unsure": 2}
    assert matched.counts["matched"] == {"ok": 0, "not_found": 0, "unsure": 1}     # with its unsure target
    tied = Parent(tmp_path / "p2", [(5.0, 5.0, 0.2, 0), (5.0, 5.0, 0.7, 0)], genes)
    path2 = write_input(tmp_path / "in2.parquet", [{"id": 1, "gene": "AAA", "x": 5.0, "y": 5.0, "z": 4.0},
                                                    {"id": 2, "gene": "AAA", "x": 5.0, "y": 5.0, "z": 4.0005}])
    pool2 = hand_pool([tied.item("target", 2, 0), tied.item("target", 2, 1)], [2], k=1)
    with pytest.raises(P.TooFewFound):                                       # both unsure: none left
        tied.pools.match_input(pool2, path2, {})


def test_matching_pools_again_and_too_few(tmp_path):
    """Pooled again on the targets found (qc_rule.pool_of over the cells' order, members first, then the rest in
    order); Items over the cells and givers, the order's PoolOrder; under k once found, every cell that has one and the
    caveat too_few {have, need} said with them (and none once they reach k); the clicked cell's suspects all lost:
    TooFewFound (the test fails before any run)."""
    genes = ["AAA"]
    records = [(float(i), float(i), 0.5, 0) for i in range(12)]
    parent = Parent(tmp_path / "p1", records, genes)
    # Cells 3 (seed: targets 0, 1), 4 (2, 3), 5 (4, 5), 6 (6, 7: the reserve), each with its uniform control.
    items = [parent.item("target", c, r, giver=9) for c, r in ((3, 0), (3, 1), (4, 2), (4, 3), (5, 4), (5, 5),
                                                                 (6, 6), (6, 7))]
    items += [parent.item("matched", 4, 8, target=2), parent.item("uniform", 5, 9)]
    found = [r for r in range(10) if r not in (2, 3)]                       # cell 4 lost both targets
    rows = [{"id": 500 + r, "gene": "AAA", "x": float(r), "y": float(r)} for r in found]
    path = write_input(tmp_path / "in.parquet", rows)
    pool = hand_pool(items, [3, 4, 5, 6], k=5, members=3, distance=[0.0, 3.0, 5.0, 9.0])
    matched = parent.pools.match_input(pool, path, {})
    assert matched.pool.cells == (3, 5, 6, 4) and matched.pool.members == 3 and matched.pool.suspects == 6
    assert matched.pool.distance_um == (0.0, 5.0, 9.0, 3.0)
    assert matched.order.parent_cells == (3, 5, 6, 4) and matched.order.seed == 0
    assert matched.order.cells == (0, 1, 2, 3) and matched.order.distance_um == (0.0, 5.0, 9.0, 3.0)
    assert matched.items.cell_names == ("cell3-1", "cell5-1", "cell6-1", "cell4-1", "cell9-1")
    assert matched.counts["target"] == {"ok": 6, "not_found": 2, "unsure": 0}
    assert matched.counts["matched"] == {"ok": 0, "not_found": 1, "unsure": 0}
    assert Q.pool_of(np.bincount(matched.items.cell[matched.items.kind == R.SUSPECT], minlength=5), matched.order, 5) \
        == ([0, 1, 2], 6)
    assert [i.kind for i in matched.pool.items] == ["target"] * 6 + ["uniform"]
    matched.items.check()
    assert matched.pool.caveats == ()
    other = P.reason("schedule")
    short = parent.pools.match_input(hand_pool(items, [3, 4, 5, 6], k=7, members=4,
                                               caveats=(other, P.reason("too_few", have=8, need=7 + 3))), path, {})
    assert (short.pool.cells, short.pool.members, short.pool.suspects) == ((3, 5, 6, 4), 3, 6)
    assert short.pool.caveats == (other, {"code": "too_few", "have": 6, "need": 7,
                                          "words": "Its numbers are for tests with 7 suspects or more: 6 here"})
    reached = parent.pools.match_input(hand_pool(items, [3, 4, 5, 6], k=5, members=3,
                                                 caveats=(P.reason("too_few", have=4, need=5),)), path, {})
    assert reached.pool.caveats == () and reached.pool.suspects == 6        # reached once found: no longer said
    lost = write_input(tmp_path / "lost.parquet", [r for r in rows if r["id"] not in (500, 501)])
    with pytest.raises(P.TooFewFound) as refused:                           # the seed lost its suspects
        parent.pools.match_input(pool, lost, {})
    assert (refused.value.status, str(refused.value)) == \
        (409, "None of this cell's suspects could be found in the area's transcripts") == (409, P.NONE_FOUND)


def test_places_are_grouped_exactly():
    """match_input's lookup: for each wanted (gene, x and y bits), every holder in index order, none for a place one
    bit away; against a plain grouping, negative coordinates included."""
    rng = np.random.default_rng(5)
    for trial in range(6):
        n = 2000
        gene = rng.integers(0, 50, n)
        x = rng.choice(np.array([-5.5, 0.0, 1.25, 3000.75, 2 ** 20 + 0.5], np.float32), n)
        y = rng.choice(np.array([-1.0, 7.5, 4100.25], np.float32), n)
        if trial % 2:
            x, y = rng.uniform(-1e4, 1e4, n).astype(np.float32), rng.uniform(-1e4, 1e4, n).astype(np.float32)
        xy = P._xy(x, y)
        want = rng.integers(0, n, 300)
        off = P._xy(np.nextafter(x[want[:20]], np.float32(np.inf)), y[want[:20]])
        wanted_gene, wanted_xy = np.concatenate([gene[want], gene[want[:20]]]), np.concatenate([xy[want], off])
        order, lo, hi = P._groups(gene, xy, wanted_gene, wanted_xy)
        plain = {}
        for i, key in enumerate(zip(gene.tolist(), xy.tolist())):
            plain.setdefault(key, []).append(i)
        for j, key in enumerate(zip(wanted_gene.tolist(), wanted_xy.tolist())):
            assert order[lo[j]:hi[j]].tolist() == plain.get(key, [])


def test_matching_reads_the_input_once_and_checks_while_it_works(tmp_path):
    genes = ["AAA"]
    parent = Parent(tmp_path / "p1", [(1.0, 1.0, 0.5, 0)], genes)
    path = write_input(tmp_path / "in.parquet", [{"id": 1, "gene": "AAA", "x": 1.0, "y": 1.0}])
    pool = hand_pool([parent.item("target", 1, 0)], [1], k=1)
    checks = []
    parent.pools.match_input(pool, path, {}, check=lambda: checks.append(1))
    assert len(checks) >= 2

    def stop():
        raise QcError("The app is closing", 503)
    with pytest.raises(QcError):
        parent.pools.match_input(pool, path, {}, check=stop)
    with pytest.raises(ValueError, match="could not be read"):
        parent.pools.match_input(pool, tmp_path / "missing.parquet", {})


# ---- caches, the worker and the API ----------------------------------------------------------------------------------
def test_answers_and_pools_are_kept_and_a_new_frame_or_model_is_a_new_key(make_world, monkeypatch):
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3} for c in range(A.NCELLS)}
    world = setup_rule(make_world, monkeypatch, 6, table, calls)
    pools = world.pools
    first = pools.preview("p1", 5, wait=10)
    assert first["offer"] is True and first["ms"]["cells"] == len(calls) and first["ms"]["one_time"] > 0
    computed, contexts = pools.computed, pools.contexts
    again = pools.preview("p1", 5)                                 # the same key: kept, nothing worked out
    assert again["pool"] == first["pool"] and pools.computed == computed and again["ms"]["cells"] == 0
    assert pools.pool("p1", 5) == pools.pool("p1", 5) and pools.computed == computed
    # Another seed in the same frame: the cells' answers are reused (only new cells are worked out).
    pools.preview("p1", 4, wait=10)
    assert pools.contexts == contexts and len(calls) == len(set(calls))
    # A new frame (the parent's end moved: a counted frame held there) or another model: a new frame context, answers
    # worked out again.
    world.write_counts(1, 1, reason="hold")
    world.run.read_counts()
    world.meta["schedule_done"] = {"seq": 1, "iter": 1}
    world.run.ack = {"state": "paused", "seq": 1, "iter": 1}
    write_codes(world.qc_dir, 1, world.codes)
    before = len(calls)
    reply = pools.preview("p1", 5, wait=10)
    assert reply["seq"] == 1 and pools.contexts == contexts + 1 and len(calls) > before
    world.qc.model_id = "another-model"
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["model"] == "another-model" and pools.contexts == contexts + 2 and reply["offer"] is True


def test_forget_drops_everything_kept_and_stops_the_worker(make_world, monkeypatch):
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3} for c in range(A.NCELLS)}
    world = setup_rule(make_world, monkeypatch, 6, table, calls)
    pools = world.pools
    pools.preview("p1", 5, wait=10)
    assert pools.answers_kept and pools.pools_kept and "p1" in pools.work
    pools.forget("p1")
    assert not pools.answers_kept and not pools.pools_kept and "p1" not in pools.work
    before = len(calls)
    pools.preview("p1", 5, wait=10)
    assert len(calls) > before


def test_one_computation_per_request_and_the_newest_waiting_wins(make_world, monkeypatch):
    """The same request asked twice is worked out once; while the worker is busy, a newer request replaces the one
    waiting (that one's waiter is told at once); a blocking pool() asks again once replaced."""
    gate, started = threading.Event(), threading.Event()
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3} for c in range(A.NCELLS)}
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 3)
    world = make_world()
    fabricate(table, monkeypatch, calls)
    real = P.QcPools._cell

    def slow(self, context, weights, c, **count):
        if c == 0:
            started.set()
            gate.wait(10)
        return real(self, context, weights, c, **count)
    monkeypatch.setattr(P.QcPools, "_cell", slow)
    pools = world.pools
    first = threading.Thread(target=lambda: pools.preview("p1", 0, wait=10))
    first.start()
    assert started.wait(10)
    replies = {}
    second = threading.Thread(target=lambda: replies.__setitem__("seven", pools.preview("p1", 7, wait=10)))
    second.start()
    wait_for(lambda: pools.work["p1"].pending is not None)
    waiting = pools.work["p1"].pending
    replies["three"] = pools.preview("p1", 3, wait=0)               # newer: replaces 7's request
    assert waiting.dropped and waiting.done.is_set()
    second.join(10)
    assert replies["seven"]["state"] == "working" and replies["three"]["state"] == "working"
    same = []
    twin = [threading.Thread(target=lambda: same.append(pools.preview("p1", 0, wait=10))) for _ in range(2)]
    for thread in twin:
        thread.start()
    time.sleep(0.2)
    gate.set()
    first.join(10)
    for thread in twin:
        thread.join(10)
    assert [r["offer"] for r in same] == [True, True] and calls.count(0) == 1
    wait_for(lambda: pools.preview("p1", 3)["state"] == "ready")
    assert calls.count(3) == 1 and not [key for key in pools.pools_kept if key[-1] == 7]


def test_a_preview_answers_as_soon_as_the_pool_is_ready_and_pool_blocks(make_world, monkeypatch):
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3} for c in range(A.NCELLS)}
    world = setup_rule(make_world, monkeypatch, 6, table, calls)
    pools = world.pools
    began = time.monotonic()
    reply = pools.preview("p1", 5, wait=9)
    assert reply["state"] == "ready" and reply["offer"] and time.monotonic() - began < 8
    pools.forget("p1")
    reply = pools.preview("p1", 5)                                   # no wait: on its way
    assert reply["state"] == "working" and reply["reasons"] == [] and reply["pool"] is None
    pool = pools.pool("p1", 5)                                       # blocks until it is there
    assert isinstance(pool, P.Pool) and pool.members >= 2
    # check() is called while pool() waits, and stops the waiting when it raises.
    pools.forget("p1")
    gate = threading.Event()
    real = P.QcPools._cell
    monkeypatch.setattr(P.QcPools, "_cell",
                        lambda self, context, weights, c, **count: gate.wait(10) and real(self, context, weights, c,
                                                                                          **count))
    before = len(calls)

    def check():
        raise QcError("The app is closing", 503)
    with pytest.raises(QcError) as stopped:
        pools.pool("p1", 5, check=check)
    assert stopped.value.status == 503 and not isinstance(stopped.value, P.StillWorking)
    # Past its wait, still being worked out: StillWorking (503, "try again" to any other caller); asked again, it is
    # the same request, which the worker goes on with: no cell worked out twice.
    with pytest.raises(P.StillWorking) as late:
        pools.pool("p1", 5, timeout=0.2)
    assert late.value.status == 503 and str(late.value) == f"{P.FAILED} in time: try again"
    gate.set()
    assert pools.pool("p1", 5) == pool
    assert calls[before:] and len(calls[before:]) == len(set(calls[before:]))


def test_the_frame_context_goes_after_its_hold_and_close_stops_everything(make_world, monkeypatch):
    now = [1000.0]
    calls = []
    table = {c: {"key": "OWN", "pattern": "OWN", "n": 3} for c in range(A.NCELLS)}
    monkeypatch.setitem(Q.CONSTANTS["paired"], "k", 6)
    world = make_world(clock=lambda: now[0])
    fabricate(table, monkeypatch, calls)
    pools = world.pools
    closed = []
    real_close = P._Context.close
    monkeypatch.setattr(P._Context, "close", lambda self: closed.append(self) or real_close(self))
    pools.preview("p1", 5, wait=10)
    work = pools.work["p1"]
    assert work.context is not None and work.thread is not None
    now[0] += P.POOL_HOLD_S - 1
    time.sleep(2 * P.TICK_S)
    assert work.context is not None and not closed
    now[0] += 2
    wait_for(lambda: work.context is None and work.thread is None, timeout=10)
    assert len(closed) == 1
    # Closing: a waiter gets 503, the worker and its context go.
    gate = threading.Event()
    real = P.QcPools._cell
    monkeypatch.setattr(P.QcPools, "_cell",
                        lambda self, context, weights, c, **count: gate.wait(10) and real(self, context, weights, c,
                                                                                          **count))
    pools.forget("p1")
    errors = []

    def wait():
        try:
            pools.pool("p1", 4)
        except QcError as exc:
            errors.append(exc.status)
    waiter = threading.Thread(target=wait)
    waiter.start()
    wait_for(lambda: "p1" in pools.work and pools.work["p1"].current is not None)
    threading.Timer(0.2, gate.set).start()
    pools.close()
    waiter.join(10)
    assert errors == [503] and pools.work == {}


def test_a_failure_is_said_in_plain_words(make_world, monkeypatch):
    world = make_world()
    monkeypatch.setattr(P.QcPools, "_cell",
                        lambda self, context, weights, c, **count: (_ for _ in ()).throw(RuntimeError("it broke")))
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["state"] == "failed" and reply["message"] == f"{P.FAILED}: it broke"
    with pytest.raises(QcError) as failed:
        world.pools.pool("p1", 5)
    assert failed.value.status == 500 and str(failed.value) == f"{P.FAILED}: it broke"


def test_the_worker_takes_numpys_blas_on_one_thread_and_sets_back_the_rest(make_world, monkeypatch):
    """numpy's BLAS on one thread while a run's worker works (frame context and cells alike), shared by every worker:
    the first in sets it, the last out sets back what it found; a BLAS that cannot be told never stops a pool. Here
    through the OpenBLAS numpy's own wheel brings, where it has one."""
    told, real_now = [], P._blas_one_now
    monkeypatch.setattr(P, "_blas_one_now", lambda: told.append("one") or (lambda: told.append("back")))
    shared = P._OneBlas()
    with shared:
        with shared:                                            # a second worker meanwhile: nothing set again
            assert told == ["one"]
        assert told == ["one"]
    assert told == ["one", "back"] and shared.users == 0
    monkeypatch.setattr(P, "_blas_one_now", lambda: (_ for _ in ()).throw(OSError("no BLAS here")))
    with shared:
        pass
    assert shared.users == 0 and shared.undo is None
    monkeypatch.setattr(P, "_blas_one_now", real_now)
    # threadpoolctl where it is installed: its limit of one BLAS thread, set back by its own restore.
    asked = []

    class Controller:
        def limit(self, limits, user_api):
            asked.append((limits, user_api))
            return types.SimpleNamespace(restore_original_limits=lambda: asked.append("restored"))
    monkeypatch.setitem(sys.modules, "threadpoolctl", types.SimpleNamespace(ThreadpoolController=Controller))
    monkeypatch.setattr(P, "_BLAS_HOW", None)
    undo = P._blas_one_now()
    assert asked == [(1, "blas")]
    undo()
    assert asked == [(1, "blas"), "restored"]
    monkeypatch.delitem(sys.modules, "threadpoolctl")
    monkeypatch.setattr(P, "_BLAS_HOW", None)                  # found again: this environment's own
    # numpy's own OpenBLAS, while a real run's worker works.
    found = P._wheel_openblas()
    if found is None:
        pytest.skip("this numpy brings no OpenBLAS of its own")
    get, put = found
    wait_for(lambda: P.ONE_BLAS.users == 0)                    # no worker of an earlier test still at work
    before = get()
    put(max(before, 2))                                         # a BLAS of several threads to come back to
    several = get()
    try:
        world = make_world()
        seen = []
        real_cell, real_context = P.QcPools._cell, P._Context

        def cell(self, context, weights, c):
            seen.append(("cell", get()))
            return real_cell(self, context, weights, c)
        monkeypatch.setattr(P.QcPools, "_cell", cell)
        monkeypatch.setattr(P, "_Context", lambda *args: seen.append(("context", get())) or real_context(*args))
        world.pools.answers("p1", [5, 3])
        assert several >= 2 and seen == [("context", 1), ("cell", 1), ("cell", 1)]
        assert get() == several and P.ONE_BLAS.users == 0
    finally:
        put(before)


# ---- the words -------------------------------------------------------------------------------------------------------
def banned_stems() -> list[str]:
    """The page's banned words (frontend/src/qc-copy.mjs BANNED_WORDS), read from the page's own source."""
    source = (ROOT / "frontend" / "src" / "qc-copy.mjs").read_text()
    block = re.search(r"BANNED_WORDS = Object\.freeze\(\[(.*?)\]\)", source, re.S).group(1)
    words = re.findall(r"'([^']+)'", block)
    assert len(words) >= 20
    return words


# The reasons proseg's own count brought, and their words (INTERFACES §9.1).
NEW_REASONS = {"no_counts": ("refuse", "Made before proseg's own count was kept"),
               "old_proseg": ("refuse", "Update proseg-live to test a setting"),
               "approx_targets": ("caveat", "Suspects placed approximately: proseg's exact shares at your run's end are "
                                            "no longer available"),
               "targets_now": ("caveat", "Suspects taken where your run is now, not at its end"),
               "count_basis": ("caveat", "Not measured yet with suspects from proseg's own count"),
               "no_suspects": ("refuse", "No suspects to test in this cell: none glows, or none could be placed"),
               "saved_suspects": ("caveat", "Suspects taken from proseg's count as saved: its averaged count gives "
                                            "none that can be tested"),
               "low_share": ("caveat", "Doesn't look mixed overall: its pink dots are tested anyway"),
               "weak_suspects": ("caveat", "Its suspects are weak: none of its pink dots looks likely"),
               "no_controls": ("refuse", "No own RNA at the same spot to compare its suspects with")}


def test_the_reasons_words_and_kinds_are_the_pages_and_plain():
    """REASON_WORDS and KIND are the page's (the shared fixture: codes in order, words, kinds, data, the words shown);
    plain words; the reasons proseg's own count brought in their places and words, none refusing a cell for its byte
    under the cut-off (the list is exactly the page's). split() parts
    reasons by kind, each in the page's order (a code of no known kind refuses); OutsideDomain holds refusals only (409,
    the first one's words), NotYet waits only."""
    examples = json.loads((ROOT / "frontend" / "test" / "fixtures" / "qc-test-reasons.json").read_text())
    assert list(P.REASON_WORDS) == [
        "ending", "ended", "no_counts", "old_proseg", "not_at_end", "past_end", "approx_targets", "targets_now",
        "whole_sample", "tiled_box", "schedule", "nuclei_start", "lasso", "area_narrow", "square_not_measured",
        "region_small", "check_unavailable", "check_off", "check_wait", "check_scoring", "details", "settings",
        "count_basis", "busy", "other_run", "edge", "not_judged", "no_suspects", "saved_suspects", "low_share",
        "weak_suspects", "same_type", "pattern_none", "pattern_neighbour", "pattern_other", "pattern_mixed", "no_primary",
        "no_controls", "too_few"]
    for code, (kind, words) in NEW_REASONS.items():
        assert (P.KIND[code], P.REASON_WORDS[code]) == (kind, words), code
    assert [e["code"] for e in examples] == list(P.REASON_WORDS)
    assert {e["code"]: e["words"] for e in examples} == P.REASON_WORDS
    assert [e["code"] for e in examples] == list(P.REASON_WORDS) == list(P.KIND)
    assert {e["code"]: e["kind"] for e in examples} == P.KIND and set(P.KIND.values()) == {"refuse", "wait", "caveat"}
    for e in examples:
        assert P.reason(e["code"], **e["data"]) == {"code": e["code"], "words": e["shown"], **e["data"]}
    stems = banned_stems()
    clashes = ("look-alike", "look alike", "similar", "same kind", "like for like")
    for code, words in P.REASON_WORDS.items():
        lower = words.lower()
        assert not [s for s in stems if s.lower() in lower], code
        assert not [c for c in clashes if c in lower], code
        assert "hold" not in lower, code
    for words in (P.FAILED, P.NONE_FOUND):                     # a failure's words, shown as they are
        assert not [s for s in stems if s.lower() in words.lower()] and not re.search(r"\d|\{", words), words
    with pytest.raises(KeyError):
        P.reason("not_offered")
    every = [P.reason(e["code"], **e["data"]) for e in reversed(examples)] + [{"code": "a_new_code", "words": "x"}]
    refusals, waits, caveats = P.split(every)
    for kind, found in (("refuse", refusals), ("wait", waits), ("caveat", caveats)):
        want = [code for code in P.REASON_WORDS if P.KIND[code] == kind] + (["a_new_code"] if kind == "refuse" else [])
        assert [r["code"] for r in found] == want, kind
    refused = P.OutsideDomain([P.reason("no_suspects"), P.reason("ending")])
    assert (refused.status, str(refused)) == (409, "No suspects to test in this cell: none glows, or none could be "
                                                   "placed")
    assert refused.extra == {"reasons": refused.reasons} and refused.reasons[1]["code"] == "ending"
    for reasons in ([], [P.reason("edge")], [P.reason("ending"), P.reason("not_at_end")]):
        with pytest.raises(ValueError):
            P.OutsideDomain(reasons)
    waiting = P.NotYet([P.reason("not_at_end"), P.reason("check_off")])
    assert (waiting.status, str(waiting), [w["code"] for w in waiting.waits]) == \
        (409, P.REASON_WORDS["not_at_end"], ["not_at_end", "check_off"])
    with pytest.raises(ValueError):
        P.NotYet([])


def test_the_words_of_weak_and_unplaced_suspects_hold_in_every_case_they_are_given():
    """weak_suspects is given to a cell that glows with no likely suspect: likely is pi >= LIKELY_MIN where proseg keeps
    at least LIKELY_W of it, so a cell is weak by pi alone (proseg keeps all of its glowing points) as often as by w
    alone; its words name neither cause, only that none looks likely (as Go deeper's line then says "maybe").
    no_suspects is given when nothing glows, when nothing that glows has a placed giver, and when the area's shares
    could not be placed at all (basis none): its words say both. approx_targets is given to a run that moved on past its
    end and to one that ended there (whose end's exact shares can't be asked for once it ended): its words never say the
    run moved on."""
    weak, unplaced, approx = (P.REASON_WORDS[code] for code in ("weak_suspects", "no_suspects", "approx_targets"))
    assert "likely" in weak and not re.search(r"aside|proseg", weak), weak
    assert "glows" in unplaced and "placed" in unplaced, unplaced
    assert "moved on" not in approx and "no longer" in approx, approx


# ---- FakePools (tests/fake_pools.py) ---------------------------------------------------------------------------------
def test_fake_pools_take_the_runs_end_and_split_the_engines_reasons(tmp_path):
    """The stand-in's pool is make's at the run's end; the run's own reasons (and `extra`, the check's) are split as
    the engine splits them: refused (OutsideDomain), waited for (NotYet, prerequisites), or carried as the pool's
    caveats; the preview lists them by kind. A pool slower than pool()'s wait is StillWorking, its work going on."""
    from fake_pools import FakePools
    world_run = type("Run", (), {})()
    world_run.meta, world_run.info = paused(id="p1"), None
    runs = type("Runs", (), {"get": lambda self, run_id: world_run if run_id == "p1" else
                             (_ for _ in ()).throw(RunError("Unknown run", 404)), "max_region": MAX_UM})()
    pool = hand_pool([], [3], k=1)
    made = []
    matched = P.Matched(pool=pool, items=None, order=None, counts={}, input="", seconds=0.0)

    def make(meta, cell):
        made.append(cell)
        return P.Pool(**{**pool.__dict__, "seq": 7, "iter": 70}), matched
    fake = FakePools(runs, make)
    got = fake.pool("p1", 3)
    assert (got.seq, got.iter, got.caveats) == (451, 450, ()) and made == [3]     # the run's end, not make's frame
    stale = FakePools(runs, make, seq_from_meta=False).pool("p1", 3)
    assert (stale.seq, stale.iter) == (7, 70)
    world_run.meta.update(state="running", schedule_done=None)
    with pytest.raises(P.NotYet) as waiting:
        fake.pool("p1", 3)
    assert [r["code"] for r in waiting.value.waits] == ["not_at_end"] and made == [3, 3]
    assert fake.prerequisites("p1") == ([], waiting.value.waits)
    assert fake.preview("p1", 3) == {"v": 1, "state": "ready", "run": "p1", "cell": 3, "offer": True, "reasons": [],
                                     "waits": waiting.value.waits, "caveats": []}
    world_run.meta.update(state="stopped")
    with pytest.raises(P.OutsideDomain) as refused:
        fake.pool("p1", 3)
    assert refused.value.reasons == [P.reason("ended")] and fake.preview("p1", 3)["offer"] is False
    assert FakePools(runs, make, reasons=False).pool("p1", 3).seq == 7             # its run's reasons left out
    with pytest.raises(RunError):
        fake.pool("nope", 3)
    world_run.meta.update(state="paused", schedule_done={"seq": 451, "iter": 450}, continued=True)
    fake.extra = [P.reason("settings"), P.reason("check_wait")]
    with pytest.raises(P.NotYet):
        fake.pool("p1", 3)
    fake.extra = [P.reason("settings")]
    assert [c["code"] for c in fake.pool("p1", 3).caveats] == ["past_end", "settings"]
    world_run.meta["continued"] = False
    again = fake.match_input(got, tmp_path / "in.parquet", {})
    assert again.pool is got and again.input == str(tmp_path / "in.parquet")
    checks = []
    FakePools(runs, make, delay_s=0.12).pool("p1", 3, check=lambda: checks.append(1))
    assert len(checks) >= 2
    # Past pool()'s wait, StillWorking as the engine's: asked again, the same work goes on (it is there sooner).
    slow = FakePools(runs, make, delay_s=1.0)
    with pytest.raises(P.StillWorking):
        slow.pool("p1", 3, timeout=0.4)
    began = time.monotonic()
    assert slow.pool("p1", 3).seq == 451 and time.monotonic() - began < 0.9
    assert fake.answers("p1", [3]) == {} and fake.calls["pool"] == 6 and fake.calls["answers"] == 1
    fake.forget("p1")
    fake.close()
    assert fake.calls["forget"] == fake.calls["close"] == 1


# ---- F counted for the hold: the count proseg saves (decision A) ------------------------------------------------------
NO_LINE = {"likely": 0, "own": 0, "none": 0, "neighbour": 0, "other": 0}


def spy_counts(monkeypatch) -> tuple[list, list]:
    """([(cell, the request's count, its block's bytes, its matrix)] of every Worker.deeper, [the count of every
    measure]) the pools ask from here on."""
    deeper, measured = [], []
    real_deeper, real_measure = W.Worker.deeper, HeldWhere.measure

    def spy_deeper(self, run, request, counts, kept):
        deeper.append((request["deeper"]["cell"], request.get("count", "averaged"), request["deeper"]["codes"], counts))
        return real_deeper(self, run, request, counts, kept)

    def spy_measure(self, *args, **kwargs):
        measured.append(kwargs.get("count", "averaged"))
        return real_measure(self, *args, **kwargs)
    monkeypatch.setattr(W.Worker, "deeper", spy_deeper)
    monkeypatch.setattr(HeldWhere, "measure", spy_measure)
    return deeper, measured


def test_at_f_counted_for_the_hold_the_test_reads_fs_bytes_on_the_averaged_count(make_world):
    """F counted for the hold (with f) is read on the count proseg saves by the panel; the test reads F's bytes on the
    averaged count all the same (<seq>.qc.averaged.bin), as it was measured: a cell F's own bytes judge but its averaged
    ones do not is not judged here, and one judged on e is, whatever its byte on f. The reply names F's count."""
    world = make_world()
    on_f, on_e = world.codes.copy(), world.codes.copy()
    on_e[5], on_f[6] = P.NOT_JUDGED, P.NOT_JUDGED
    write_codes(world.qc_dir, SEQ, on_f, averaged=on_e)
    refused = world.pools.preview("p1", 5)
    assert refused["count"] == "sampled" and kinds_of(refused)[0] == ["not_judged"]
    judged = world.pools.preview("p1", 6)
    assert judged["count"] == "sampled" and "not_judged" not in kinds_of(judged)[0]
    # F's own bytes on f not written yet (the check has not scored F's hold count): asked for, the test waits.
    (world.qc_dir / f"{SEQ:06d}.qc.averaged.bin").unlink()
    assert kinds_of(world.pools.preview("p1", 6))[1] == ["check_scoring"] and world.qc.configured[-1] == ("p1", SEQ)


def test_line_shown_is_go_deepers_own_line_at_f_counted_for_the_hold(make_world, monkeypatch):
    """At F counted for the hold Go deeper shows the count proseg saves: the pool reply's line_shown is the seed's line
    worked out once more as Go deeper shows it there: Go deeper's answer on f (the request's count "sampled", the
    worker's f matrix of F's counts file, F's own bytes on f), the page's where request from that answer, the where
    measurement on f with that answer's givers, the page's scores and the targets; `line` stays the test's own, on e.
    The page asks another gene of the answer it reads on f here (none of the run's), so the two lines differ."""
    world = make_world(page=True)
    genes = top_genes(world, 5, 2)
    world.page.sources = [4, 6]
    monkeypatch.setattr(P, "givers_of", lambda result: {genes[0]: 4, genes[1]: 6})
    real_params = world.page.params

    def params(raw, cell):                                          # the page's request on each answer it reads
        world.page.genes = genes if raw.get("count") != "sampled" else ["NOT_A_GENE"]
        return real_params(raw, cell)
    monkeypatch.setattr(S, "page_params", params)
    deeper, measured = spy_counts(monkeypatch)
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["state"] == "ready" and reply["count"] == "sampled"
    assert reply["line"]["likely"] > 0 and reply["line_shown"] == NO_LINE
    seed = [(count, codes) for cell, count, codes, _ in deeper if cell == 5]
    assert seed == [("averaged", str(world.qc_dir / f"{SEQ:06d}.qc.averaged.bin")),
                    ("sampled", str(world.qc_dir / f"{SEQ:06d}.qc.bin"))]
    f_matrix = [counts for cell, count, _, counts in deeper if count == "sampled"][0]
    read = counts_file.read(world.live / world.run.frames[SEQ]["counts_file"])
    assert np.array_equal(np.asarray(f_matrix), W.Worker(xp=np).matrix(read, "f"))
    assert measured.count("sampled") == 1                           # the seed's alone
    # With the page asking the same genes on both, line_shown is the line of f's own targets (keep 1: f = e here).
    monkeypatch.setattr(S, "page_params", real_params)
    world.page.genes = genes
    world.pools.forget("p1")
    same = world.pools.preview("p1", 5, wait=10)
    assert same["line_shown"] == same["line"] and same["line"]["likely"] > 0


def test_at_an_averaged_f_line_shown_is_the_line_and_nothing_is_worked_out_twice(make_world, monkeypatch):
    """F counted for a token (the averaged count, on the panel too): line_shown is `line`, Go deeper's answer is worked
    out once for the seed, and every measurement reads e."""
    world = make_world(page=True)
    world.write_counts(SEQ, ITER, reason="request")
    world.run.read_counts()
    world.page.genes, world.page.sources = top_genes(world, 5, 2), [4, 6]
    deeper, measured = spy_counts(monkeypatch)
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["count"] == "averaged" and reply["line_shown"] == reply["line"] and reply["line"]["likely"] > 0
    assert [count for cell, count, _, _ in deeper if cell == 5] == ["averaged"] and set(measured) == {"averaged"}


def test_a_pool_kept_for_f_averaged_is_worked_out_again_once_its_hold_line_comes(make_world):
    """The frame context's key names F's count (its newest counts line): a pool worked out while F was counted for a
    token is kept while it stays so, and worked out again once proseg counts F again for the hold, its reply then
    naming the count proseg saves."""
    world = make_world(page=True)
    world.page.genes, world.page.sources = top_genes(world, 5, 2), [4, 6]
    world.write_counts(SEQ, ITER, reason="request")
    world.run.read_counts()
    first = world.pools.preview("p1", 5, wait=10)
    contexts = world.pools.contexts
    assert first["count"] == "averaged" and world.pools.preview("p1", 5)["state"] == "ready"
    assert world.pools.contexts == contexts, "kept"
    world.write_counts(SEQ, ITER, reason="hold")
    world.run.read_counts()
    again = world.pools.preview("p1", 5, wait=10)
    assert again["count"] == "sampled" and world.pools.contexts == contexts + 1
    assert P._Facts(run_id="p1", start="paired", seq=SEQ, count="sampled").key() != \
        P._Facts(run_id="p1", start="paired", seq=SEQ).key()


def asking(world, monkeypatch, genes: dict, cell: int | None = None, others: list | None = None) -> None:
    """The page's where request asks `genes[count]` of each answer it reads of `cell` (every cell's when None; count:
    the answer's, "averaged" or "sampled"), `others` of any other cell's; a gene of none of the run's ("NOT_A_GENE")
    makes nothing glow on that count."""
    real_params = world.page.params

    def params(raw, at):
        mine = cell is None or at == cell
        world.page.genes = genes["sampled" if raw.get("count") == "sampled" else "averaged"] if mine else others
        return real_params(raw, at)
    monkeypatch.setattr(S, "page_params", params)


def seed_items(pool: P.Pool) -> list[tuple]:
    """The clicked cell's items: (kind, record, giver, w) in the pool's order."""
    return [(item.kind, item.record, item.giver, item.w) for item in pool.items if item.cell == pool.seed]


def test_a_cell_whose_dots_glow_only_on_the_count_proseg_saves_is_tested_on_them(make_world, monkeypatch):
    """F counted for the hold: Go deeper and its pink dots show the count proseg saves (f), the test reads the averaged
    one (e). A cell nothing of which glows on e (the page's request there asks a gene of none of the run's) but whose
    dots glow on f is not refused: its suspects are Go deeper's on f (its targets, their records, and its controls
    drawn at f / n), with the caveat saved_suspects, and its line is f's (line_shown alike); the other cells of its key
    are pooled on e as ever. The same answer read on e (the page asking those genes on both counts: e's suspects, the
    usual case) is pooled the same, without the caveat, and its controls' shares are e / n's. Nothing glowing on either
    count: no_suspects."""
    keep = np.full(len(A.RUN_GENES), 0.8)
    probe = make_world(page=True)
    genes = top_genes(probe, 5, 2)
    keep[[A.RUN_GENES.index(g) for g in genes]] = 1.0           # the asked genes: f = e; the cell's own RNA: f != e
    monkeypatch.setattr(P, "givers_of", lambda result: {genes[0]: 4, genes[1]: 6})

    def worked(asked: dict):
        """(the world, its pool reply for cell 5, its pool or the refusal), the page asking `asked` by count."""
        world = make_world(page=True, keep=keep)
        world.page.sources = [4, 6]
        asking(world, monkeypatch, asked, cell=5, others=genes)
        reply = world.pools.preview("p1", 5, wait=10)
        try:
            return world, reply, world.pools.pool("p1", 5)
        except P.OutsideDomain as refused:
            return world, reply, refused
    on_f, saved, pool_f = worked({"averaged": ["NOT_A_GENE"], "sampled": genes})
    on_e, usual, pool_e = worked({"averaged": genes, "sampled": genes})
    assert (saved["state"], saved["count"], saved["offer"], saved["reasons"]) == ("ready", "sampled", True, [])
    assert "saved_suspects" in kinds_of(saved)[2]
    assert saved["line"] == saved["line_shown"] and saved["line"]["likely"] > 0
    assert (usual["offer"], usual["reasons"]) == (True, []) and "saved_suspects" not in kinds_of(usual)[2]
    assert [code for code in kinds_of(saved)[2] if code != "saved_suspects"] == kinds_of(usual)[2]
    assert (saved["line"], saved["this_cell"], saved["pattern"]) == \
        (usual["line_shown"], usual["this_cell"], usual["pattern"])
    assert [c for c in pool_f.caveats if c["code"] != "saved_suspects"] == list(pool_e.caveats)
    assert P.reason("saved_suspects") in pool_f.caveats and pool_f.caveats == tuple(P.ordered(pool_f.caveats))
    targets = [item[:3] for item in seed_items(pool_f) if item[0] == "target"]
    assert targets and targets == [item[:3] for item in seed_items(pool_e) if item[0] == "target"]
    assert pool_f.view_suspects == pool_e.view_suspects and pool_f.view_sources == pool_e.view_sources
    share = lambda world, r, which: round(float(world.w_of(5, world.records[[r]], "exact", which)[0]), 3)  # noqa: E731
    controls = [item for item in seed_items(pool_f) if item[0] != "target"]
    assert controls and all(w == share(on_f, r, "f") > 0 for _, r, _, w in controls)
    usual_controls = [item for item in seed_items(pool_e) if item[0] != "target"]
    assert usual_controls and all(w == share(on_e, r, "e") > 0 for _, r, _, w in usual_controls)
    assert controls != usual_controls                            # drawn at f / n, not e / n
    assert [i.cell for i in pool_f.items if i.cell != 5] == [i.cell for i in pool_e.items if i.cell != 5]
    _, refused, caught = worked({"averaged": ["NOT_A_GENE"], "sampled": ["NOT_A_GENE"]})
    assert kinds_of(refused)[0] == ["no_suspects"] and refused["line"] == refused["line_shown"] == NO_LINE
    assert isinstance(caught, P.OutsideDomain) and [r["code"] for r in caught.reasons] == ["no_suspects"]


def test_at_f_read_on_the_averaged_count_a_cell_with_no_dots_on_it_is_refused_and_f_is_never_read(make_world,
                                                                                                    monkeypatch):
    """F counted for a token (the panel and Go deeper show e there too): a cell nothing of which glows on e is refused
    (no_suspects), whatever f would show, and nothing is worked out on f."""
    world = make_world(page=True)
    world.write_counts(SEQ, ITER, reason="request")
    world.run.read_counts()
    genes = top_genes(world, 5, 2)
    world.page.sources = [4, 6]
    monkeypatch.setattr(P, "givers_of", lambda result: {genes[0]: 4, genes[1]: 6})
    asking(world, monkeypatch, {"averaged": ["NOT_A_GENE"], "sampled": genes})
    deeper, measured = spy_counts(monkeypatch)
    reply = world.pools.preview("p1", 5, wait=10)
    assert reply["count"] == "averaged" and kinds_of(reply)[0] == ["no_suspects"]
    assert {count for _, count, _, _ in deeper} == {"averaged"} and set(measured) <= {"averaged"}
