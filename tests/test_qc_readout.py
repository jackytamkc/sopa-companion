"""Test a setting here: each test run read from proseg's own per-transcript file (engine/qc_readout.py), on tiny run
folders written here as proseg-live writes them (live/info.json, cells.txt, genes.txt, frames.jsonl with a dense frame
and its cell statistics; output/transcript-metadata.parquet in proseg's schema), with every share worked out by hand."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from sopa_companion.engine import qc_readout as R
from sopa_companion.engine.config import ROOT

NAMES = ("aaaaaaaa-1", "bbbbbbbb-1", "cccccccc-1")       # labels 1, 2, 3
GENES = ("GENE1", "GENE2", "GENE3")
# A 4 x 3 x 1 grid of 1 µm voxels from (0, 0): its labels (0: no cell), rows j = 0, 1, 2.
LABELS = np.array([[[1, 1, 2, 2],
                    [1, 1, 2, 0],
                    [3, 3, 0, 0]]], np.uint32)
INNER = (0.0, 0.0, 4.0, 3.0)


def row(tid, ox, oy, gene, *, d=(0, 0, 0), to=None, bg=False, oz=0.5):
    """One transcript: observed at (ox, oy, oz), moved by d voxels, assigned `to` (a cell name; None: none) unless
    background."""
    return {"tid": tid, "ox": ox, "oy": oy, "oz": oz, "d": d, "gene": gene, "to": to, "bg": bg or to is None}


def write_run(folder: Path, rows: list[dict], *, labels=LABELS, names=NAMES, genes=GENES, iteration=450,
              centres=None, tiled=False, where="output", dictionary=True, csv=False, voxel=1.0, z=(0.0, 1.0)) -> Path:
    """A finished test run's folder: live/ as proseg-live leaves it (the last frame dense unless `tiled`, with its
    cell statistics when `centres` are given; its layers from z[0] µm, z[1] µm apart) and proseg's per-transcript file
    in `where`."""
    live = folder / "live"
    (live / "frames").mkdir(parents=True, exist_ok=True)
    nz, ny, nx = labels.shape
    (live / "info.json").write_text(json.dumps({"cells": len(names), "genes": len(genes), "layers": nz,
                                                "z": {"grid_origin": z[0], "grid_spacing": z[1]}}))
    (live / "cells.txt").write_text("".join(f"{name}\n" for name in names))
    (live / "genes.txt").write_text("".join(f"{name}\n" for name in genes))
    zstd = zstandard.ZstdCompressor(level=1)
    seq = iteration + 1
    record = {"seq": seq, "iter": iteration, "phase": "hillclimb", "nx": nx, "ny": ny, "nz": nz, "x0": 0.0, "y0": 0.0,
              "voxel": voxel}
    if tiled:
        record.update(file=f"frames/{seq:06d}.tiles", format="tiles", tile=64)
        (live / record["file"]).write_bytes(b"PLT1")
    else:
        record["file"] = f"frames/{seq:06d}.bin.zst"
        planes = np.concatenate([labels[0].reshape(-1), np.full(nx * ny, iteration, np.uint32), labels.reshape(-1)])
        (live / record["file"]).write_bytes(zstd.compress(planes.astype("<u4").tobytes()))
    if centres is not None:
        stats = np.zeros((7, len(names)), "<u4")
        stats[5] = np.array([c[0] for c in centres], "<f4").view("<u4")
        stats[6] = np.array([c[1] for c in centres], "<f4").view("<u4")
        record["cells_file"] = f"frames/{seq:06d}.cells.zst"
        (live / record["cells_file"]).write_bytes(zstd.compress(stats.tobytes()))
    first = {**record, "seq": 0, "iter": 0, "phase": "init"}
    (live / "frames.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(record) + "\n")
    f32 = np.float32
    ox = np.array([r["ox"] for r in rows], f32)
    oy = np.array([r["oy"] for r in rows], f32)
    oz = np.array([r["oz"] for r in rows], f32)
    di = np.array([r["d"] for r in rows], np.int64).reshape(-1, 3)
    columns = {
        "transcript_id": pa.array([r["tid"] for r in rows], pa.uint64()),
        "qv": pa.array(np.full(len(rows), 40, f32)),
        "x": pa.array(ox + (di[:, 0] * voxel).astype(f32)), "y": pa.array(oy + (di[:, 1] * voxel).astype(f32)),
        "z": pa.array(oz + (di[:, 2] * z[1]).astype(f32)),
        "observed_x": pa.array(ox), "observed_y": pa.array(oy), "observed_z": pa.array(oz),
        # As proseg writes it: a dictionary over the run's own genes.txt, so its codes follow that run's order.
        "gene": (pa.DictionaryArray.from_arrays(pa.array([list(genes).index(r["gene"]) for r in rows], pa.int32()),
                                                pa.array(list(genes), pa.string())) if dictionary
                 else pa.array([r["gene"] for r in rows], pa.string())),
        "assignment": pa.array([None if r["bg"] or r["to"] is None else names.index(r["to"]) for r in rows],
                               pa.uint32()),
        "background": pa.array([r["bg"] for r in rows], pa.bool_()),
    }
    out = folder / where
    out.mkdir(parents=True, exist_ok=True)
    path = out / R.METADATA
    if csv:
        path.write_text("transcript_id,x\n1,2.0\n")
    else:
        pq.write_table(pa.table(columns), path, compression="zstd")
    return folder


def items_of(rows_or_specs, *, names=NAMES, genes=GENES) -> R.Items:
    """Items from (kind, cell, giver, tid, x, y, gene) tuples; cell and giver are names (giver None: NO_GIVER)."""
    names = list(names)
    genes = list(genes)
    extra = sorted({n for spec in rows_or_specs for n in (spec[1], spec[2]) if n is not None and n not in names})
    names += extra
    kind = np.array([spec[0] for spec in rows_or_specs], np.int8)
    target = np.full(len(rows_or_specs), -1, np.int64)
    for i, spec in enumerate(rows_or_specs):
        if spec[0] == R.MATCHED:
            target[i] = spec[7]
    items = R.Items(kind=kind, cell=np.array([names.index(s[1]) for s in rows_or_specs], np.int32),
                    giver=np.array([R.NO_GIVER if s[2] is None else names.index(s[2]) for s in rows_or_specs], np.int32),
                    target=target, gene=np.array([genes.index(s[6]) for s in rows_or_specs], np.int32),
                    transcript_id=np.array([s[3] for s in rows_or_specs], np.uint64),
                    x=np.array([s[4] for s in rows_or_specs], np.float32),
                    y=np.array([s[5] for s in rows_or_specs], np.float32), cell_names=tuple(names),
                    gene_names=tuple(genes))
    items.check()
    return items


A, B, C = NAMES
# One bucket (GENE1 at voxel (0, 0)) of four rows: two stay in A's voxel (one of them called background by proseg), one
# moves two voxels right into B, one moves to (3, 1), no cell's. So d_A = 2/4, d_B = 1/4, no cell 1/4; q(A, GENE1) = 1/2
# (2 rows reach A, 1 is counted), q(B, GENE1) = 1. Plus GENE2 rows in B and C, unmoved, and C's own GENE1 row.
BUCKET = [row(11, 0.5, 0.5, "GENE1", to=A), row(12, 0.25, 0.75, "GENE1", bg=True),
          row(13, 0.75, 0.25, "GENE1", d=(2, 0, 0), to=B), row(14, 0.4, 0.4, "GENE1", d=(3, 1, 0)),
          row(21, 2.5, 0.5, "GENE2", to=B), row(22, 0.5, 2.5, "GENE2", to=C), row(23, 1.5, 2.5, "GENE1", to=C)]


def test_the_shares_of_a_bucket_are_its_destinations_times_q(tmp_path):
    """HELD = d_c0 q(c0, g), GIVER = d_c1 q(c1, g), OTHER over every other cell, BG the rest, worked out by hand."""
    run = write_run(tmp_path / "r", BUCKET)
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"),       # A's, from B
                      (R.SUSPECT, A, C, 12, 0.25, 0.75, "GENE1"),     # A's, from C: B's share is OTHER
                      (R.MATCHED, A, B, 14, 0.4, 0.4, "GENE1", 0),    # in the same bucket: the same shares
                      (R.UNIFORM, B, None, 21, 2.5, 0.5, "GENE2"),    # B's own, alone in its bucket: all held
                      (R.SUSPECT, C, A, 23, 1.5, 2.5, "GENE1")])      # C's GENE1 row: q(C, GENE1) = 1
    out = R.read_run(run, items, INNER)
    assert out.status.tolist() == [R.OK] * 5
    expect = [(0.25, 0.25, 0.0, 0.5), (0.25, 0.0, 0.25, 0.5), (0.25, 0.25, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0),
              (1.0, 0.0, 0.0, 0.0)]
    got = np.stack([out.held, out.giver, out.other, out.bg], axis=1)
    assert np.allclose(got, expect, atol=1e-12)
    assert out.checks["identity_max_err"] <= R.IDENTITY_TOL and out.checks["f_gt_n"] == 0
    assert out.checks["identity_pairs"] == 5           # (A, GENE1), (B, GENE1), (C, GENE1), (B, GENE2), (C, GENE2)
    assert out.rows == len(BUCKET) and out.iter == 450 and out.seq == 451
    assert out.checks["rows_dest_outside"] == 0 and out.checks["dest_vs_assignment"] == 0


def built_tables(monkeypatch) -> list:
    """The per (cell, gene) tables each read builds, kept here: which check refuses a run is not in its error."""
    real, seen = R._Tables, []

    def keep(*args, **kwargs):
        seen.append(real(*args, **kwargs))
        return seen[-1]
    monkeypatch.setattr(R, "_Tables", keep)
    return seen


def test_the_identity_failing_is_a_plain_readout_error(tmp_path, monkeypatch):
    """A row counted as a cell's where no row of its gene reaches that cell: f > 0 where n = 0, the one place the
    identity fails (f > n there too). The run is not read, and what the page says holds no word it never shows."""
    rows = BUCKET + [row(31, 2.5, 0.5, "GENE3", to=A)]           # GENE3 lands in B's voxel, but proseg counts it A's
    run = write_run(tmp_path / "r", rows)
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1")])
    tables = built_tables(monkeypatch)
    with pytest.raises(R.ReadoutError) as caught:
        R.read_run(run, items, INNER)
    assert type(caught.value) is R.ReadoutError
    assert (tables[0].max_err, tables[0].f_gt_n) == (1.0, 1)
    assert not banned(f"The test could not finish: {caught.value}")


def test_more_rows_counted_than_reach_a_cell_is_refused_though_the_identity_holds(tmp_path, monkeypatch):
    """f > n > 0: two GENE3 rows proseg counts as A's, one landing in A's voxel and one in B's. The identity gives f
    back wherever n > 0 (A: 1 row reaches it, q = 2, and 1 x 1 x 2 = 2 = f; B: q = 0 = f), so f <= n alone refuses
    the run: read, A's GENE3 row would be held 2.0. Counted as B's instead, the same rows are read."""
    items = items_of([(R.SUSPECT, A, B, 31, 0.5, 0.5, "GENE3")])
    tables = built_tables(monkeypatch)
    over = write_run(tmp_path / "over", BUCKET + [row(31, 0.5, 0.5, "GENE3", to=A), row(32, 2.5, 0.5, "GENE3", to=A)])
    with pytest.raises(R.ReadoutError, match="counts do not match"):
        R.read_run(over, items, INNER)
    assert (tables[0].max_err, tables[0].f_gt_n) == (0.0, 1)
    fine = write_run(tmp_path / "fine", BUCKET + [row(31, 0.5, 0.5, "GENE3", to=A), row(32, 2.5, 0.5, "GENE3", to=B)])
    out = R.read_run(fine, items, INNER)
    assert (out.checks["identity_max_err"], out.checks["f_gt_n"]) == (0.0, 0)
    assert out.status.tolist() == [R.OK] and out.held[0] == 1.0


def banned(text: str) -> list[str]:
    """The page's banned stems (frontend/src/qc-copy.mjs BANNED_WORDS) found in `text`, ignoring case."""
    source = (ROOT / "frontend" / "src" / "qc-copy.mjs").read_text()
    block = re.search(r"BANNED_WORDS = Object\.freeze\(\[(.*?)\]\)", source, re.S).group(1)
    words = re.findall(r"'([^']+)'", block)
    assert len(words) > 20 and "voxel" in words
    return [word for word in words if word.lower() in text.lower()]


def test_no_readout_error_holds_a_banned_word():
    source = (ROOT / "engine" / "qc_readout.py").read_text()
    said = re.findall(r'ReadoutError\("([^"]+)"\)', source)
    assert len(said) >= 12
    assert [s for s in said if banned(f"The test could not finish: {s}")] == []


def test_offsets_are_added_in_i_j_k_and_a_destination_past_the_edge_is_no_cells(tmp_path):
    """A row observed in the last column and moved one voxel right lies outside the grid: no cell's, never the first
    column of the next row (a flat index would say A's)."""
    rows = BUCKET + [row(41, 3.5, 0.5, "GENE2", d=(1, 0, 0)), row(42, 3.5, 1.5, "GENE3", d=(-1, 0, 0), to=B)]
    run = write_run(tmp_path / "r", rows)
    items = items_of([(R.SUSPECT, A, B, 41, 3.5, 0.5, "GENE2"), (R.SUSPECT, A, B, 42, 3.5, 1.5, "GENE3")])
    out = R.read_run(run, items, INNER)
    assert out.status.tolist() == [R.OK, R.OK]
    assert (out.held[0], out.bg[0]) == (0.0, 1.0)
    assert out.checks["rows_dest_outside"] == 1
    assert (out.held[1], out.giver[1], out.bg[1]) == (0.0, 1.0, 0.0)     # moved back into B's voxel: its giver's


def test_layers_are_found_and_moved_through_as_proseg_does(tmp_path):
    """Two layers 1.5 µm apart from z = 2, A and B changing places in the upper one. A row's layer is floor((z - 2) /
    1.5), clipped to the layers (z 1 is the lower one's, z 9 the upper's); its offset in layers is (z - observed z) /
    1.5; a destination above the top layer is no cell's. GENE1 rows at (0.5, 0.5): 82 (z 3.05, 0.7 of a layer up, so
    the lower layer's: A) and 81 (z 4, the upper's: B) stay; 83 (z 1) moves one layer up, into B; 84 (z 9) moves above
    the top. So the lower bucket goes half to A and half to B, the upper one half to B and half nowhere."""
    upper = np.array([[2, 2, 1, 1], [2, 2, 1, 0], [3, 3, 0, 0]], np.uint32)
    rows = [row(81, 0.5, 0.5, "GENE1", oz=4.0, to=B), row(82, 0.5, 0.5, "GENE1", oz=3.05, to=A),
            row(83, 0.5, 0.5, "GENE1", oz=1.0, d=(0, 0, 1), to=B), row(84, 0.5, 0.5, "GENE1", oz=9.0, d=(0, 0, 1))]
    run = write_run(tmp_path / "r", rows, labels=np.stack([LABELS[0], upper]), z=(2.0, 1.5))
    items = items_of([(R.SUSPECT, A, B, 82, 0.5, 0.5, "GENE1"), (R.SUSPECT, B, A, 81, 0.5, 0.5, "GENE1")])
    out = R.read_run(run, items, INNER)
    assert out.status.tolist() == [R.OK, R.OK]
    got = np.stack([out.held, out.giver, out.other, out.bg], axis=1)
    assert np.allclose(got, [(0.5, 0.5, 0.0, 0.0), (0.5, 0.0, 0.0, 0.5)], atol=1e-12)
    assert out.checks["rows_dest_outside"] == 1 and out.checks["offset_residual"] == 0.0


def test_a_row_without_a_cell_is_no_cells_even_when_not_called_background(tmp_path):
    """Proseg leaves a row without a cell exactly where it calls it background; one that is not called so still
    counts as no cell's (never the first cell's): B's GENE2 bucket gains a row reaching B that no cell counts, so B's
    own row there is held at half."""
    rows = BUCKET + [dict(row(71, 2.5, 0.5, "GENE2"), bg=False)]
    out = R.read_run(write_run(tmp_path / "r", rows), items_of([(R.UNIFORM, B, None, 21, 2.5, 0.5, "GENE2")]), INNER)
    assert out.status.tolist() == [R.OK] and (out.held[0], out.bg[0]) == (0.5, 0.5)


def test_statuses_and_their_order(tmp_path):
    """C is not in this run (its voxels no cell's, its rows background), nor is D, a giver of no run."""
    rows = [dict(r, to=None, bg=True) if r["to"] == C else r for r in BUCKET]
    run = write_run(tmp_path / "r", rows, names=(A, B), labels=np.where(LABELS == 3, 0, LABELS).astype(np.uint32))
    near = float(np.nextafter(np.float32(0.5), np.float32(1)))
    items = items_of([(R.SUSPECT, A, B, 99, 0.5, 0.5, "GENE1"),        # no row has its id
                      (R.SUSPECT, A, B, 11, near, 0.5, "GENE1"),       # x one bit off
                      (R.SUSPECT, A, B, 13, 0.75, 0.25, "GENE2"),      # the row's gene is GENE1
                      (R.SUSPECT, C, A, 22, 0.5, 2.5, "GENE2"),        # its cell is not in the run
                      (R.SUSPECT, A, C, 12, 0.25, 0.75, "GENE1"),      # its giver is not in the run
                      (R.SUSPECT, C, B, 98, 0.5, 2.5, "GENE2"),        # not found beats a missing cell
                      (R.SUSPECT, C, A, 23, 1.0, 2.5, "GENE1"),        # its check failing beats a missing cell
                      (R.MATCHED, C, "dddddddd-1", 22, 0.5, 2.5, "GENE2", 3)],    # cell and giver missing: c0_absent
                     names=(A, B, C))
    out = R.read_run(run, items, INNER)
    assert out.status.tolist() == [R.NOT_FOUND, R.CROSSCHECK_FAILED, R.CROSSCHECK_FAILED, R.C0_ABSENT, R.C1_ABSENT,
                                   R.NOT_FOUND, R.CROSSCHECK_FAILED, R.C0_ABSENT]
    assert out.checks["crosscheck"] == {"gene": 1, "x": 2, "y": 0}
    # Shares wherever the row is found and the cell is in the run: also when its check failed, or its giver is missing
    # (GIVER 0 then, its giver's share counted as no other cell's: C is not in the run).
    assert np.isnan([out.held[0], out.held[3], out.held[5], out.held[7]]).all()
    assert out.held[1] == pytest.approx(0.25) and out.giver[1] == pytest.approx(0.25)
    assert out.held[4] == pytest.approx(0.25) and out.giver[4] == 0.0 and out.other[4] == pytest.approx(0.25)
    assert R.STATUS_NAMES[R.CROSSCHECK_FAILED] == "crosscheck_failed"


def test_a_name_on_two_lines_of_cells_txt_is_its_first(tmp_path):
    """Cells are found by name, at the first line naming them: in a run whose cells.txt names A again on line 3 (label
    3, which holds no row of A's bucket), A's items are read against label 1."""
    rows = [dict(r, to=None, bg=True) if r["to"] == C else r for r in BUCKET]
    run = write_run(tmp_path / "r", rows, names=(A, B, A))
    out = R.read_run(run, items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1")]), INNER)
    assert out.status.tolist() == [R.OK] and out.checks["duplicate_cells"] == 1
    assert (out.held[0], out.giver[0], out.other[0], out.bg[0]) == pytest.approx((0.25, 0.25, 0.0, 0.5))


def test_a_row_without_an_id_holds_no_item(tmp_path):
    """A null transcript_id is no id: the run has none to compare (ids None), the row still counts as one of its
    transcripts, and no item is found at it, not even one whose id is 0 (what a null is read as). Runs are compared by
    their ids only where they have them."""
    rows = BUCKET + [row(None, 2.5, 0.5, "GENE2", to=B)]
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"), (R.UNIFORM, B, None, 0, 2.5, 0.5, "GENE2")])
    out = R.read_run(write_run(tmp_path / "null", rows), items, INNER)
    assert out.ids is None and out.rows == len(rows)
    assert out.status.tolist() == [R.OK, R.NOT_FOUND] and out.held[0] == pytest.approx(0.25)
    named = [dict(r) for r in rows]
    named[-1]["tid"] = 77
    runs = R.read_runs({"A1": tmp_path / "null", "A2": write_run(tmp_path / "named", named)}, items, INNER)
    assert runs["A1"].ids is None and runs["A2"].ids is not None
    assert runs["A2"].status.tolist() == [R.OK, R.NOT_FOUND]


def test_the_gene_order_and_the_kind_of_gene_column_change_nothing(tmp_path):
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"), (R.UNIFORM, B, None, 21, 2.5, 0.5, "GENE2"),
                      (R.SUSPECT, C, A, 23, 1.5, 2.5, "GENE1")])
    plain = R.read_run(write_run(tmp_path / "a", BUCKET), items, INNER)
    turned = R.read_run(write_run(tmp_path / "b", BUCKET, genes=tuple(reversed(GENES))), items, INNER)
    text = R.read_run(write_run(tmp_path / "c", BUCKET, dictionary=False), items, INNER)
    for other in (turned, text):
        assert other.status.tolist() == plain.status.tolist()
        for name in ("held", "giver", "other", "bg"):
            assert np.array_equal(getattr(other, name), getattr(plain, name))


def test_transcripts_per_cell_is_a_median_over_the_inner_cells(tmp_path):
    """Foreground rows per cell: A 1, B 2, C 2. The inner box holds A's and B's last centres, its edges included (A's
    on it); C's lies outside, and a cell without a centre never counts. With two more rows of C's, three inner cells
    hold 1, 2 and 4: their median is 2 (their mean would be 2.33)."""
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1")])
    centres = [(1.0, 1.0), (2.5, 0.5), (0.5, 2.5)]
    out = R.read_run(write_run(tmp_path / "a", BUCKET, centres=centres), items, (1.0, 0.0, 4.0, 1.0))
    assert (out.tx_inner, out.cells_inner) == (1.5, 2)
    skewed = BUCKET + [row(61, 0.5, 2.5, "GENE2", to=C), row(62, 1.5, 2.5, "GENE2", to=C)]
    out = R.read_run(write_run(tmp_path / "e", skewed, centres=centres), items, INNER)
    assert (out.tx_inner, out.cells_inner) == (2.0, 3)
    out = R.read_run(write_run(tmp_path / "b", BUCKET, centres=[(1.0, 1.0), (np.nan, np.nan), (0.5, 2.5)]), items,
                     (1.0, 0.0, 4.0, 1.0))
    assert (out.tx_inner, out.cells_inner) == (1.0, 1)
    out = R.read_run(write_run(tmp_path / "c", BUCKET, centres=centres), items, (3.5, 2.5, 4.0, 3.0))
    assert (out.tx_inner, out.cells_inner) == (None, 0)
    out = R.read_run(write_run(tmp_path / "d", BUCKET), items, INNER)            # no cell statistics
    assert (out.tx_inner, out.cells_inner) == (None, 0)


def test_read_runs_needs_the_same_transcripts_and_the_same_last_iteration(tmp_path):
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"), (R.SUSPECT, A, C, 12, 0.25, 0.75, "GENE1")])
    dirs = {role: write_run(tmp_path / role, BUCKET, centres=[(1, 1), (2, 1), (1, 2)])
            for role in ("A1", "A2", "B1", "B2")}
    together = R.read_runs(dirs, items, INNER, workers=4)
    alone = R.read_runs(dirs, items, INNER, workers=1)
    assert list(together) == ["A1", "A2", "B1", "B2"]
    for role in dirs:
        for name in ("status", "held", "giver", "other", "bg"):
            assert np.array_equal(getattr(together[role], name), getattr(alone[role], name), equal_nan=True)
        assert together[role].ids == alone[role].ids and together[role].ids is not None
    more = {**dirs, "B2": write_run(tmp_path / "more", BUCKET + [row(51, 2.5, 0.5, "GENE2", to=B)])}
    with pytest.raises(R.ReadoutError, match="did not read the same transcripts"):
        R.read_runs(more, items, INNER)
    other_id = [dict(r) for r in BUCKET]
    other_id[4]["tid"] = 77
    with pytest.raises(R.ReadoutError, match="did not read the same transcripts"):
        R.read_runs({**dirs, "B1": write_run(tmp_path / "ids", other_id)}, items, INNER)
    with pytest.raises(R.ReadoutError, match="did not end at the same iteration"):
        R.read_runs({**dirs, "A2": write_run(tmp_path / "iter", BUCKET, iteration=449)}, items, INNER)
    # One run that cannot be read: its error, once every read has ended.
    broken = write_run(tmp_path / "broken", BUCKET)
    (broken / "output" / R.METADATA).unlink()
    with pytest.raises(R.ReadoutError, match="wrote no file"):
        R.read_runs({**dirs, "A1": broken}, items, INNER)


@pytest.mark.parametrize("how", ["missing", "csv", "tiled"])
def test_a_run_that_cannot_be_read_says_so(tmp_path, how):
    run = write_run(tmp_path / "r", BUCKET, csv=how == "csv", tiled=how == "tiled")
    if how == "missing":
        (run / "output" / R.METADATA).unlink()
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1")])
    with pytest.raises(R.ReadoutError):
        R.read_run(run, items, INNER)


def test_the_metadata_may_be_named(tmp_path):
    """A run folder that keeps proseg's outputs in out/ (as the measurements' did) is read from the file named."""
    run = write_run(tmp_path / "r", BUCKET, where="out")
    items = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1")])
    with pytest.raises(R.ReadoutError, match="wrote no file"):
        R.read_run(run, items, INNER)
    out = R.read_run(run, items, INNER, metadata_path=run / "out" / R.METADATA)
    assert out.status.tolist() == [R.OK] and out.held[0] == pytest.approx(0.25)


def test_items_are_checked():
    good = items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"), (R.MATCHED, A, B, 13, 0.75, 0.25, "GENE1", 0)])
    assert len(good) == 2
    with pytest.raises(ValueError, match="no SUSPECT of its cell"):
        items_of([(R.SUSPECT, A, B, 11, 0.5, 0.5, "GENE1"), (R.MATCHED, B, A, 13, 0.75, 0.25, "GENE1", 0)])
    with pytest.raises(ValueError, match="UNIFORM item has no giver"):
        items_of([(R.UNIFORM, A, B, 11, 0.5, 0.5, "GENE1")])
    bad = R.Items(kind=good.kind, cell=good.cell, giver=good.giver, target=good.target, gene=good.gene,
                  transcript_id=good.transcript_id.astype(np.int64), x=good.x, y=good.y, cell_names=good.cell_names,
                  gene_names=good.gene_names)
    with pytest.raises(ValueError, match="uint64"):
        bad.check()
