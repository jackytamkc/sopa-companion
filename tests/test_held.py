"""Proseg's share of each transcript for one cell (engine/held.py): where exports written and read back (their bytes
decoded here by hand), refused when they do not fit their record; the box an export covers and the box to query;
each transcript's bucket by proseg's float32 grid rule; and w, exact from a hand-made export (|B| from the
transcripts), approximate from the frame's 3-D labels, or none, never from a 2-D column. On a made-up run whose
pools are all exported, the exact w add up to each cell's averaged count, gene by gene."""
from __future__ import annotations

import ast
import dataclasses
import inspect
import io
import struct
import sys

import numpy as np
import pytest
import zstandard

from sopa_companion.analysis import counts_file as C
from sopa_companion.engine import held as H
from sopa_companion.engine.transcript_index import RECORD

GRID = {"x0": 0.0, "y0": 0.0, "voxel": 1.0, "nx": 6, "ny": 4, "nz": 2}
Z = (0.0, 1.0)                          # layers from z = 0 µm, 1 µm apart


def records(rows) -> np.ndarray:
    """Transcripts as the run's index holds them, from (x, y, z, gene) rows."""
    out = np.zeros(len(rows), RECORD)
    for name, values in zip(("x", "y", "z", "gene"), zip(*rows) if rows else ([], [], [], [])):
        out[name] = values
    return out


def record_of(path, *, entries: int, span=(0, 0, 4, 3), grid=GRID, token=3, seq=40, with_path=True) -> dict:
    """A where export's record (where.json, or a counts line's "where") for the file at `path`."""
    meta = {"token": token, "seq": seq, "iter": 120, "box": [0.5, 0.5, 4.5, 3.5], "margin_um": 24.0, **grid,
            "i0": span[0], "j0": span[1], "i1": span[2], "j1": span[3], "file": path.name, "entries": entries}
    if with_path:
        meta["path"] = str(path)
    return meta


def payload_of(path) -> bytes:
    reader = zstandard.ZstdDecompressor().decompressobj()
    data = reader.decompress(path.read_bytes())
    assert reader.eof and not reader.unused_data
    return data


# A hand-made frame on GRID (span i 0..4, j 0..3): each bucket a gene at one observed voxel (i, j, k).
#   B1 gene 0 at (1, 1, 0): 4 transcripts; pools: cell 0 (label 1) 1 + 1 (two records, not merged), no cell 2
#   B2 gene 1 at (2, 1, 0): 3 transcripts; pools: cell 0 2, cell 1 1                (a bucket split between two cells)
#   B3 gene 1 at (3, 2, 1): 2 transcripts; pools: cell 0 5                          (more than |B|: d is held at 1)
#   B4 gene 0 at (5, 3, 0): 1 transcript, outside the export's span
#   B5 gene 2 at (0, 0, 1): 2 transcripts; pools: cell 1 2
#   B6 gene 2 at (4, 3, 0): 4 transcripts; pools: cell 0 1 (the rest drifted out of the scan: d is 1 of the 4, not
#      1 of the 1 exported)
# and one transcript off the grid.
HAND = records([(1.1, 1.5, 0.5, 0), (1.3, 1.5, 0.5, 0), (1.5, 1.2, 0.1, 0), (1.9, 1.8, 0.9, 0),
                (2.2, 1.2, 0.2, 1), (2.4, 1.2, 0.2, 1), (2.8, 1.9, 0.7, 1),
                (3.5, 2.5, 1.5, 1), (3.6, 2.1, 1.2, 1),
                (5.5, 3.5, 0.5, 0),
                (0.2, 0.5, 1.9, 2), (0.7, 0.5, 1.1, 2),
                (4.1, 3.2, 0.3, 2), (4.5, 3.5, 0.5, 2), (4.9, 3.9, 0.1, 2), (4.2, 3.1, 0.8, 2),
                (-0.5, 1.0, 0.5, 0)])
HAND_EXPORT = {"i": [1, 1, 1, 2, 2, 3, 0, 4], "j": [1, 1, 1, 1, 1, 2, 0, 3], "k": [0, 0, 0, 0, 0, 1, 1, 0],
               "gene": [0, 0, 0, 1, 1, 1, 2, 2], "dest": [1, 1, 0, 1, 2, 1, 2, 1], "n": [1, 1, 2, 2, 1, 5, 2, 1]}
ENTRIES = len(HAND_EXPORT["i"])
Q = {0: np.array([0.5, 0.75, 0.25]), 1: np.array([1.0, 0.4, 0.6])}       # each cell's keep e / n, per gene
HAND_W = {0: [0.25] * 4 + [0.5] * 3 + [0.75] * 2 + [0.0] + [0.0] * 2 + [0.0625] * 4 + [0.0],
          1: [0.0] * 4 + [0.4 / 3] * 3 + [0.0] * 2 + [0.0] + [0.6] * 2 + [0.0] * 4 + [0.0]}


def hand_export(tmp_path) -> H.WhereExport:
    path = tmp_path / "where-3.bin.zst"
    H.write_where(path, *(HAND_EXPORT[name] for name in ("i", "j", "k", "gene", "dest", "n")))
    return H.read_where(record_of(path, entries=ENTRIES))


# ---- where exports ------------------------------------------------------------------------------------------------
def test_a_where_export_reads_back_as_written_in_the_documented_layout(tmp_path):
    path = tmp_path / "where-3.bin.zst"
    H.write_where(path, *(HAND_EXPORT[name] for name in ("i", "j", "k", "gene", "dest", "n")))
    raw = payload_of(path)
    assert len(raw) == 4 + 21 * ENTRIES and struct.unpack_from("<I", raw)[0] == ENTRIES
    at = 4
    for name, kind, size in (("i", "<u4", 4), ("j", "<u4", 4), ("k", "u1", 1), ("gene", "<u4", 4),
                             ("dest", "<u4", 4), ("n", "<u4", 4)):
        assert np.frombuffer(raw, kind, ENTRIES, at).tolist() == HAND_EXPORT[name]
        at += size * ENTRIES
    assert not list(tmp_path.glob("*.tmp"))

    export = H.read_where(record_of(path, entries=ENTRIES))
    for name in ("i", "j", "k", "gene", "dest", "n"):
        values = getattr(export, name)
        assert values.dtype == np.int64 and values.tolist() == HAND_EXPORT[name]
    assert (export.token, export.seq, export.iter, export.margin_um) == (3, 40, 120, 24.0)
    assert export.box == (0.5, 0.5, 4.5, 3.5) and all(isinstance(v, float) for v in export.box)
    assert (export.x0, export.y0, export.voxel, export.nx, export.ny, export.nz) == (0.0, 0.0, 1.0, 6, 4, 2)
    assert (export.i0, export.j0, export.i1, export.j1) == (0, 0, 4, 3)
    assert [field.name for field in dataclasses.fields(H.WhereExport)] == [
        "token", "seq", "iter", "box", "margin_um", "x0", "y0", "voxel", "nx", "ny", "nz", "i0", "j0", "i1", "j1",
        "i", "j", "k", "gene", "dest", "n"]
    # A record without its file's path takes the one given.
    assert H.read_where(record_of(path, entries=ENTRIES, with_path=False), path).n.tolist() == HAND_EXPORT["n"]
    # A frame whose header does not give its size (as a streaming writer leaves it) reads the same.
    out = io.BytesIO()
    with zstandard.ZstdCompressor(level=1, write_content_size=False).stream_writer(out, closefd=False) as writer:
        writer.write(raw)
    path.write_bytes(out.getvalue())
    assert H.read_where(record_of(path, entries=ENTRIES)).dest.tolist() == HAND_EXPORT["dest"]


def test_an_empty_where_export(tmp_path):
    path = tmp_path / "where-9.bin.zst"
    H.write_where(path, [], [], [], [], [], [])
    export = H.read_where(record_of(path, entries=0, span=(5, 3, 4, 2)))      # a box off the grid: an empty span
    assert export.i.size == 0 and export.n.dtype == np.int64


def rewritten(path, payload: bytes):
    path.write_bytes(zstandard.ZstdCompressor(level=1).compress(payload))


def test_a_where_export_that_does_not_fit_its_record_is_refused(tmp_path):
    path = tmp_path / "where-3.bin.zst"
    columns = [HAND_EXPORT[name] for name in ("i", "j", "k", "gene", "dest", "n")]
    H.write_where(path, *columns)
    raw = payload_of(path)
    with pytest.raises(ValueError, match=f"holds {ENTRIES} records, its record says {ENTRIES - 1}"):
        H.read_where(record_of(path, entries=ENTRIES - 1))
    for broken in (raw[:-1], raw + b"\0"):
        rewritten(path, broken)
        with pytest.raises(ValueError, match="do not hold its count"):
            H.read_where(record_of(path, entries=ENTRIES))
    H.write_where(path, *columns)
    for span in ((1, 0, 4, 3), (0, 0, 2, 3), (0, 1, 4, 3), (0, 0, 4, 1)):     # i, j of a record outside the span
        with pytest.raises(ValueError, match="outside its span"):
            H.read_where(record_of(path, entries=ENTRIES, span=span))
    with pytest.raises(ValueError, match="outside its span"):                   # k past the layers
        H.read_where(record_of(path, entries=ENTRIES, grid=dict(GRID, nz=1)))
    whole = path.read_bytes()
    for broken, words in ((b"not a zstd frame", "not a zstd frame"), (whole[:len(whole) // 2], "not one whole"),
                          (whole + whole, "not one whole"), (b"", "not one whole")):
        path.write_bytes(broken)
        with pytest.raises(ValueError, match=words):
            H.read_where(record_of(path, entries=ENTRIES))
    path.write_bytes(whole)
    meta = record_of(path, entries=ENTRIES)
    del meta["voxel"]
    with pytest.raises(ValueError, match="record cannot be read"):
        H.read_where(meta)
    with pytest.raises(ValueError, match="four numbers"):
        H.read_where(dict(record_of(path, entries=ENTRIES), box=[1, 2, 3]))
    with pytest.raises(ValueError, match="no file given"):
        H.read_where(record_of(path, entries=ENTRIES, with_path=False))
    with pytest.raises(FileNotFoundError):
        H.read_where(record_of(tmp_path / "where-4.bin.zst", entries=ENTRIES))


def test_write_where_refuses_what_its_fields_cannot_hold(tmp_path):
    path = tmp_path / "where-1.bin.zst"
    good = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]
    for at, values in ((0, [0]), (2, [0, 256]), (0, [-1, 0]), (5, [1.5, 1.0]), (1, [[0, 1]])):
        columns = list(good)
        columns[at] = values
        with pytest.raises(ValueError):
            H.write_where(path, *columns)
    assert not path.exists()
    H.write_where(path, *good)
    assert path.exists()


# ---- boxes ------------------------------------------------------------------------------------------------------------
def test_an_export_covers_a_box_its_span_holds_within_a_hundredth_of_a_micron(tmp_path):
    meta = {"x0": 10.0, "y0": 20.0, "voxel": 2.0, "i0": 2, "j0": 1, "i1": 5, "j1": 3}    # x 14 to 22, y 22 to 28
    assert H.covers(meta, [14.0, 22.0, 22.0, 28.0])
    assert H.covers(meta, [13.995, 21.995, 22.009, 28.009])
    for box in ([13.98, 22.0, 22.0, 28.0], [14.0, 21.98, 22.0, 28.0], [14.0, 22.0, 22.02, 28.0],
                [14.0, 22.0, 22.0, 28.02]):
        assert not H.covers(meta, box), box
    export = hand_export(tmp_path)                       # a WhereExport too: x 0 to 5, y 0 to 4
    assert H.covers(export, [0.0, 0.0, 5.0, 4.0]) and not H.covers(export, [0.0, 0.0, 5.5, 4.0])


def test_a_span_at_the_grids_edge_covers_a_box_reaching_past_it(tmp_path):
    """Proseg clamps a where export's span to its grid (i0 = max(0, ..), i1 = min(nx - 1, ..)): a box reaching past
    the grid on a side the span ends at the grid's edge is held whole there (no voxel lies beyond), so a cell near the
    run's edge has its exact shares. A side short of the grid's edge still has to reach the box."""
    grid = {"x0": 10.0, "y0": 20.0, "voxel": 2.0, "nx": 6, "ny": 4}                 # x 10 to 22, y 20 to 28
    whole = {**grid, "i0": 0, "j0": 0, "i1": 5, "j1": 3}
    assert H.covers(whole, [0.0, 0.0, 100.0, 100.0]) and H.covers(whole, [-5.0, 21.0, 15.0, 40.0])
    for side, box in (("i0", [11.0, 21.0, 21.0, 27.0]), ("j0", [11.0, 21.0, 21.0, 27.0]),
                      ("i1", [11.0, 21.0, 21.0, 27.0]), ("j1", [11.0, 21.0, 21.0, 27.0])):
        inner = {**whole, side: {"i0": 1, "j0": 1, "i1": 4, "j1": 2}[side]}      # that side one voxel in from the edge
        assert H.covers(inner, [12.0, 22.0, 20.0, 26.0]), side
        past = {"i0": [8.0, 22.0, 20.0, 26.0], "j0": [12.0, 18.0, 20.0, 26.0], "i1": [12.0, 22.0, 30.0, 26.0],
                "j1": [12.0, 22.0, 20.0, 40.0]}[side]
        assert not H.covers(inner, past), side                                     # past it, on that side only
        assert H.covers(whole, past), side
    # The other sides still count: a span at the left edge does not hold a box past its right side.
    assert not H.covers({**grid, "i0": 0, "j0": 0, "i1": 2, "j1": 3}, [0.0, 20.0, 18.0, 28.0])
    # A box wholly off the grid (proseg's export there is empty, i0 > i1): nothing of it to hold, so covered.
    assert H.covers({**grid, "i0": 0, "j0": 0, "i1": -3, "j1": 3}, [0.0, 20.0, 5.0, 28.0])
    # A WhereExport (columns 0 to 4 of 6, every row of 4): past the grid's left, bottom and top, but not its right.
    export = hand_export(tmp_path)
    assert H.covers(export, [-50.0, -10.0, 5.0, 40.0]) and not H.covers(export, [-50.0, -10.0, 50.0, 40.0])


def test_the_query_box_holds_every_bucket_of_the_box_whole():
    grid = {"x0": 10.0, "y0": 20.0, "voxel": 2.0, "nx": 20, "ny": 20, "nz": 1}
    assert H.query_box(grid, [15.0, 27.0, 21.0, 33.0]) == [12.0, 24.0, 24.0, 36.0]     # one column and row more
    assert H.query_box(grid, [0.0, 0.0, 100.0, 100.0]) == [10.0, 20.0, 50.0, 60.0]     # within the grid
    # Transcripts on and a hair off voxel edges, in float32: every transcript of a bucket any transcript in the box
    # lies in is inside the query box, as the transcript index compares (float32 against the box's numbers).
    rng = np.random.default_rng(3)
    grid = {"x0": 0.1, "y0": -7.3, "voxel": 0.7, "nx": 40, "ny": 30, "nz": 2}
    count = 6000
    found = np.zeros(count, RECORD)
    for name, origin, size in (("x", 0.1, 40), ("y", -7.3, 30)):
        edge = origin + 0.7 * rng.integers(0, size, count)
        found[name] = edge + rng.choice([-2e-6, 0.0, 2e-6, 0.35], count)
    found["z"] = rng.uniform(-1.0, 3.0, count)
    found["gene"] = rng.integers(0, 3, count)
    keys = H.buckets(found, grid, *Z)
    x, y = found["x"], found["y"]
    for _ in range(60):
        x0, x1 = np.sort(rng.uniform(-1.0, 29.0, 2))
        y0, y1 = np.sort(rng.uniform(-8.0, 14.0, 2))
        box = [float(x0), float(y0), float(x1), float(y1)]
        q = H.query_box(grid, box)
        in_box = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3]) & (keys >= 0)
        members = np.isin(keys, keys[in_box])
        in_query = (x >= q[0]) & (x < q[2]) & (y >= q[1]) & (y < q[3])
        assert in_query[members].all(), box


# ---- buckets ----------------------------------------------------------------------------------------------------------
def test_buckets_follow_proseg_float32_grid_and_layer_rule():
    grid = {"x0": 0.1, "y0": 0.1, "voxel": 2.0, "nx": 5, "ny": 5, "nz": 3}
    edge = float(np.float32(4.1))
    found = records([(edge, edge, edge, 1),         # float32: (4.1 - 0.1) / 2 = 2.0 (column, row and layer 2)
                     (5.0, 3.0, -5.0, 2),           # below the first layer: layer 0
                     (5.0, 3.0, 100.0, 2),          # above the last: layer 2
                     (0.0, 3.0, 1.0, 0),            # left of the grid
                     (10.1, 3.0, 1.0, 0),           # right of it: column 5 of 5
                     (5.0, 10.2, 1.0, 0)])          # below it
    keys = H.buckets(found, grid, 0.1, 2.0)
    # In float64 the first would be column 1: (4.099999904632568 - 0.1) / 2 < 2.
    assert ((edge - 0.1) / 2.0) < 2.0
    key = lambda gene, k, j, i: ((gene * 3 + k) * 5 + j) * 5 + i
    assert keys.dtype == np.int64
    assert keys.tolist() == [key(1, 2, 2, 2), key(2, 0, 1, 2), key(2, 2, 1, 2), -1, -1, -1]
    # A span keeps only its columns and rows (any layer).
    spanned = H.buckets(records([(2.5, 2.5, 1.0, 0), (0.5, 2.5, 1.0, 0), (6.5, 2.5, 1.0, 0), (2.5, 6.5, 1.0, 0)]),
                        grid, 0.1, 2.0, span=(1, 1, 2, 2))
    assert spanned.tolist() == [key(0, 0, 1, 1), -1, -1, -1]


# ---- weights ----------------------------------------------------------------------------------------------------------
def test_exact_weights_share_each_pool_evenly_among_its_buckets_transcripts(tmp_path):
    export = hand_export(tmp_path)
    for cell in (0, 1):
        w, held = H.weights(HAND, cell, Q[cell], GRID, *Z, export=export)
        assert held == "exact" and w.dtype == np.float64
        assert w.tolist() == pytest.approx(HAND_W[cell], abs=1e-15)
    # Records outside the export's span get nothing, even from a pool it holds there (B6, once the span ends at 3).
    w, _ = H.weights(HAND, 0, Q[0], GRID, *Z, export=dataclasses.replace(export, i1=3))
    assert w.tolist() == pytest.approx(HAND_W[0][:12] + [0.0] * 5, abs=1e-15)
    # The export decides on its own grid (its frame's), whatever grid is passed; and over the frame's layers.
    labels = np.ones((2, 4, 6), np.uint32)
    w, held = H.weights(HAND, 0, Q[0], dict(GRID, voxel=2.0), *Z, export=export, layers=(labels, 0, 0))
    assert held == "exact" and w.tolist() == pytest.approx(HAND_W[0], abs=1e-15)
    # Shares in a bucket: every destination's, no cell's too (label 0), add up to 1 when its pools hold |B| (B3's
    # pools hold more and are held at 1; B6's hold less); nothing outside the span or off the grid.
    shares = sum(H.weights(HAND, cell, np.ones(3), GRID, *Z, export=export)[0] for cell in (-1, 0, 1))
    assert shares.tolist() == pytest.approx([1.0] * 9 + [0.0] + [1.0] * 2 + [0.25] * 4 + [0.0])


def test_approximate_weights_take_the_3d_owner_of_each_observed_voxel():
    found = records([(1.5, 1.5, 0.5, 0),     # (1, 1) layer 0: cell 0
                     (2.5, 1.5, 0.5, 1),     # (2, 1) layer 0: cell 1
                     (3.5, 2.5, 1.5, 1),     # (3, 2) layer 1: cell 0
                     (2.5, 2.5, 1.5, 2),     # (2, 2) layer 1: no cell, though layer 0 there is cell 0's
                     (-0.5, 1.5, 0.5, 0),    # off the grid
                     (2.5, 4.5, 0.5, 0)])    # off the grid
    layers = np.zeros((2, 4, 6), np.uint32)
    layers[0, 1, 1], layers[0, 1, 2], layers[1, 2, 3], layers[0, 2, 2] = 1, 2, 1, 1
    want0 = [0.5, 0.0, 0.75, 0.0, 0.0, 0.0]
    w, held = H.weights(found, 0, Q[0], GRID, *Z, layers=(layers, 0, 0))      # a dense frame's layers
    assert held == "approx" and w.tolist() == want0
    w, held = H.weights(found, 1, Q[1], GRID, *Z, layers=(layers, 0, 0))
    assert held == "approx" and w.tolist() == [0.0, 0.4, 0.0, 0.0, 0.0, 0.0]
    window = layers[:, 0:3, 1:5].copy()          # a tiled frame's window: columns 1 to 4, rows 0 to 2
    w, held = H.weights(found, 0, Q[0], GRID, *Z, layers=(window, 1, 0))
    assert held == "approx" and w.tolist() == want0


def test_without_an_export_or_layers_covering_every_transcript_nothing_is_held():
    found = records([(1.5, 1.5, 0.5, 0), (3.5, 2.5, 1.5, 1), (5.5, 1.5, 0.5, 0)])    # the last in column 5
    layers = np.zeros((2, 4, 6), np.uint32)
    layers[:, :, :] = 1                           # cell 0 everywhere, in both layers
    window = layers[:, 0:3, 1:5].copy()          # columns 1 to 4: the window misses column 5
    for given in (None, (window, 1, 0)):
        w, held = H.weights(found, 0, Q[0], GRID, *Z, layers=given)
        assert held == "none" and w.tolist() == [0.0, 0.0, 0.0]
    # There is no 2-D column to fall back on: weights takes an export or per-layer labels, nothing else.
    assert list(inspect.signature(H.weights).parameters) == ["records", "cell", "q_row", "grid", "z_origin",
                                                             "z_spacing", "export", "layers"]
    assert H.HELD == ("exact", "approx", "none")


def test_weights_refuse_inputs_that_do_not_fit():
    with pytest.raises(ValueError, match="past the cell's 2 genes"):
        H.weights(HAND, 0, [0.5, 0.75], GRID, *Z)
    with pytest.raises(ValueError, match="layers"):
        H.weights(HAND, 0, Q[0], GRID, *Z, layers=(np.zeros((3, 4, 6), np.uint32), 0, 0))
    w, held = H.weights(records([]), 0, Q[0], GRID, *Z)
    assert held == "none" and w.size == 0


def test_exact_weights_add_up_to_each_cells_averaged_count(tmp_path):
    """A made-up run whose pools in the box are all exported: per bucket the export's pools hold |B| transcripts, and
    over the box each cell's w give back its e, gene by gene (q = e / n from its counts file)."""
    rng = np.random.default_rng(7)
    grid = {"x0": 5.0, "y0": -3.0, "voxel": 1.5, "nx": 16, "ny": 12, "nz": 3}
    z_origin, z_spacing = -1.0, 2.0
    ncells, ngenes, count = 4, 7, 3000
    found = np.zeros(count, RECORD)
    found["x"] = rng.uniform(5.01, 5.0 + 16 * 1.5 - 0.01, count)
    found["y"] = rng.uniform(-2.99, -3.0 + 12 * 1.5 - 0.01, count)
    found["z"] = rng.uniform(-2.0, 6.0, count)
    found["gene"] = rng.integers(0, ngenes, count)
    # Observed voxels as proseg places them (float32 throughout), worked out here apart from the module's rule.
    i = np.floor((found["x"] - np.float32(5.0)) / np.float32(1.5)).astype(np.int64)
    j = np.floor((found["y"] - np.float32(-3.0)) / np.float32(1.5)).astype(np.int64)
    k = np.clip(np.floor((found["z"] - np.float32(-1.0)) / np.float32(2.0)), 0, 2).astype(np.int64)
    gene = found["gene"].astype(np.int64)
    # Destinations: bands of columns, no cell in the top rows, and a quarter drifted anywhere (or to no cell).
    dest = np.where(j >= 10, 0, 1 + (i // 4) % ncells)
    drift = rng.random(count) < 0.25
    dest[drift] = rng.integers(0, ncells + 1, int(drift.sum()))

    # The export: pools per (observed voxel, gene, destination), those of two or more in two records (proseg keeps a
    # pool per offset, so one bucket's transcripts to one cell can sit in several records).
    groups, sizes = np.unique(np.stack([i, j, k, gene, dest], axis=1), axis=0, return_counts=True)
    two = sizes >= 2
    rows = np.concatenate([groups, groups[two]])
    pooled = np.concatenate([np.where(two, sizes - 1, sizes), np.ones(int(two.sum()), np.int64)])
    path = tmp_path / "where-5.bin.zst"
    H.write_where(path, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], pooled)
    export = H.read_where(record_of(path, entries=len(rows), span=(0, 0, 15, 11), grid=grid))

    # The per-bucket identity: the export's pools of each bucket hold |B|, the bucket's transcripts.
    keys = H.buckets(found, grid, z_origin, z_spacing)
    assert (keys >= 0).all()
    bucket_keys, bucket_sizes = np.unique(keys, return_counts=True)
    export_keys = ((export.gene * 3 + export.k) * 12 + export.j) * 16 + export.i
    pool_keys, at = np.unique(export_keys, return_inverse=True)
    assert np.array_equal(pool_keys, bucket_keys)
    assert np.array_equal(np.bincount(at.reshape(-1), weights=export.n), bucket_sizes)

    # Each cell's counts: n per gene from its destinations, e a share of it, written and read as proseg's file.
    n = np.zeros((ncells, ngenes), np.int64)
    np.add.at(n, (dest[dest > 0] - 1, gene[dest > 0]), 1)
    keep = rng.uniform(0.2, 1.0, (ncells, ngenes))
    cells, genes = np.nonzero(n)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=ncells))])
    counts_path = tmp_path / "000005.counts.zst"
    C.write(counts_path, ncells=ncells, ngenes=ngenes, iteration=50, indptr=indptr, genes=genes, n=n[cells, genes],
            e=(n * keep)[cells, genes])
    counts = C.read(counts_path, ncells=ncells, ngenes=ngenes)
    table = C.rows(counts, range(ncells))
    shares = np.zeros(count)
    for cell in range(ncells):
        e_row, n_row = table["e"][cell].astype(np.float64), table["n"][cell].astype(np.float64)
        q = np.divide(e_row, n_row, out=np.zeros(ngenes), where=n_row > 0)
        w, held = H.weights(found, cell, q, grid, z_origin, z_spacing, export=export)
        assert held == "exact"
        assert w.sum() == pytest.approx(e_row.sum(), rel=1e-12)
        assert np.bincount(gene, weights=w, minlength=ngenes) == pytest.approx(e_row, rel=1e-12, abs=1e-12)
        shares += H.weights(found, cell, np.ones(ngenes), grid, z_origin, z_spacing, export=export)[0]
    shares += H.weights(found, -1, np.ones(ngenes), grid, z_origin, z_spacing, export=export)[0]   # no cell's
    assert shares == pytest.approx(np.ones(count))


def test_several_cells_at_once_are_each_cells_weights(tmp_path):
    """weights_many (Go deeper's cell and its sources in one pass over the records' buckets): row c is weights() for
    cells[c], exact, approximate or none alike; one keep row per cell."""
    export = hand_export(tmp_path)
    layers = np.zeros((2, 4, 6), np.uint32)
    layers[0, 1, 1], layers[0, 1, 2], layers[1, 2, 3], layers[0, 3, 4] = 1, 2, 1, 2
    q = np.stack([Q[0], Q[1], np.ones(3)])
    for given in ({"export": export}, {"layers": (layers, 0, 0)}, {"layers": (layers[:, :2, :2], 0, 0)}, {}):
        w, held = H.weights_many(HAND, [0, 1, -1], q, GRID, *Z, **given)
        assert w.shape == (3, len(HAND)) and w.dtype == np.float64
        for c, cell in enumerate((0, 1, -1)):
            one, held_one = H.weights(HAND, cell, q[c], GRID, *Z, **given)
            assert held == held_one and w[c].tolist() == one.tolist(), (given.keys(), cell)
    assert H.weights_many(HAND, [], np.zeros((0, 3)), GRID, *Z, export=export)[0].shape == (0, len(HAND))
    with pytest.raises(ValueError, match="one keep row per cell"):
        H.weights_many(HAND, [0, 1], q, GRID, *Z)


def test_a_cells_keep_is_its_averaged_count_over_its_assigned_one(tmp_path):
    """keeps(counts, cells): e / n per gene of each cell asked (any order, repeats), 0 where it holds none of a gene."""
    path = tmp_path / "000003.counts.zst"
    C.write(path, ncells=3, ngenes=4, iteration=30, indptr=[0, 2, 2, 4], genes=[0, 3, 1, 3], n=[4, 2, 8, 1],
            e=[3.0, 0.5, 2.0, 1.0])
    counts = C.read(path)
    assert H.keeps(counts, [2, 0, 1, 0]).tolist() == [[0, 0.25, 0, 1.0], [0.75, 0, 0, 0.25], [0, 0, 0, 0],
                                                      [0.75, 0, 0, 0.25]]
    assert H.keeps(counts, []).shape == (0, 4)


def test_it_imports_only_what_the_engine_has():
    """The engine's own packages (requirements.txt: numpy, zstandard) and the app's modules."""
    tree = ast.parse(open(H.__file__).read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add(node.module.split(".")[0])
    assert found - set(sys.stdlib_module_names) == {"numpy", "zstandard"}
