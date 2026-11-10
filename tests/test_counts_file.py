"""Proseg's own count files (analysis/counts_file.py): PLC1 files and their per-cell sidecars written and read back,
their bytes decoded here by hand from the layout the module documents (so the writer and the reader cannot agree on a
wrong layout between themselves), every malformed file refused with plain words, and what the workers, the engine and
the tests read from a file: dense rows, CSR arrays, the start block and the per-cell sums."""
from __future__ import annotations

import ast
import io
import math
import re
import struct
import sys

import numpy as np
import pytest
import zstandard

from sopa_companion.analysis import counts_file as C

# Four cells, six genes: rows 0 (genes 0, 2, 5), 1 (none), 2 (1, 4), 3 (3).
SMALL = {"ncells": 4, "ngenes": 6, "iteration": 12, "indptr": [0, 3, 3, 5, 6], "genes": [0, 2, 5, 1, 4, 3],
         "n": [3, 1, 7, 2, 4, 1], "e": [2.5, 0.03125, 6.0, 1.25, 3.75, 0.5], "f": [2, 0, 6, 1, 4, 0]}
INIT = {"indptr": [0, 2, 3, 4, 4], "genes": [0, 2, 0, 1]}       # the init frame's rows: four entries
START_E = [1.5, 0.25, 2.0, 0.75]


def zstd_of(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=1).compress(payload)


def payload_of(path) -> bytes:
    """A file's one zstd frame, decoded with another zstd library than the module's."""
    reader = zstandard.ZstdDecompressor().decompressobj()
    data = reader.decompress(path.read_bytes())
    assert reader.eof and not reader.unused_data
    return data


def layout(raw: bytes) -> dict:
    """A PLC1 payload's fields, decoded by hand from the layout: header, indptr, gene steps, n, e, f, start block."""
    magic = raw[:4]
    version, ncells, ngenes, nnz, flags, scale, iteration, saturated = struct.unpack_from("<8I", raw, 4)
    at = 36
    out = {"magic": magic, "version": version, "ncells": ncells, "ngenes": ngenes, "nnz": nnz, "flags": flags,
           "e_scale": scale, "iteration": iteration, "saturated": saturated}
    out["indptr"] = np.frombuffer(raw, "<u4", ncells + 1, at).tolist()
    at += 4 * (ncells + 1)
    width = 4 if flags & 2 else 2
    out["steps"] = np.frombuffer(raw, "<u4" if width == 4 else "<u2", nnz, at).tolist()
    at += width * nnz
    for name in ("n", "e") + (("f",) if flags & 1 else ()):
        out[name] = np.frombuffer(raw, "<u2", nnz, at).tolist()
        at += 2 * nnz
    if flags & 4:
        (start_nnz,) = struct.unpack_from("<I", raw, at)
        out["start_e"] = np.frombuffer(raw, "<u2", start_nnz, at + 4).tolist()
        at += 4 + 2 * start_nnz
    assert at == len(raw)
    return out


def write_small(path, *, f=True, start=False, **changes) -> dict:
    fields = dict(SMALL, **changes)
    if not f:
        fields.pop("f")
    if start:
        fields["start_e"] = START_E
    return C.write(path, **fields)


# ---- round trips ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("with_f", [True, False])
@pytest.mark.parametrize("with_start", [True, False])
def test_a_counts_file_reads_back_as_written_in_the_documented_layout(tmp_path, with_f, with_start):
    path = tmp_path / "000012.counts.zst"
    write_small(path, f=with_f, start=with_start)
    raw = layout(payload_of(path))
    assert raw["magic"] == b"PLC1" and (raw["version"], raw["e_scale"], raw["iteration"]) == (1, 32, 12)
    assert (raw["ncells"], raw["ngenes"], raw["nnz"], raw["saturated"]) == (4, 6, 6, 0)
    assert raw["flags"] == (1 if with_f else 0) | (4 if with_start else 0)
    assert raw["indptr"] == SMALL["indptr"]
    assert raw["steps"] == [0, 2, 3, 1, 3, 3]          # per row: the first gene, then the step from the one before
    assert raw["n"] == SMALL["n"]
    assert raw["e"] == [80, 1, 192, 40, 120, 16]       # e x 32
    assert raw.get("f") == (SMALL["f"] if with_f else None)
    assert raw.get("start_e") == ([48, 8, 64, 24] if with_start else None)

    counts = C.read(path, ncells=4, ngenes=6)
    assert (counts.ncells, counts.ngenes, counts.iteration, counts.saturated, counts.nnz) == (4, 6, 12, 0, 6)
    assert counts.flags == raw["flags"]
    assert counts.indptr.dtype == np.int64 and counts.indptr.tolist() == SMALL["indptr"]
    assert counts.genes.dtype == np.int32 and counts.genes.tolist() == SMALL["genes"]
    assert counts.n.dtype == np.int32 and counts.n.tolist() == SMALL["n"]
    assert counts.e.dtype == np.float32 and counts.e.tolist() == SMALL["e"]
    if with_f:
        assert counts.f.dtype == np.int32 and counts.f.tolist() == SMALL["f"]
    else:
        assert counts.f is None
    if with_start:
        assert counts.start_e.dtype == np.float32 and counts.start_e.tolist() == START_E
    else:
        assert counts.start_e is None
    assert C.read(path).ncells == 4                    # without the run's sizes: not checked against them
    assert not list(tmp_path.glob("*.tmp"))            # written atomically: no temporary file left


def test_genes_past_65535_are_kept_as_32_bit_numbers(tmp_path):
    wide, narrow = tmp_path / "wide.counts.zst", tmp_path / "narrow.counts.zst"
    genes = [3, 65_534, 65_536, 69_999, 0, 69_998]
    C.write(wide, ncells=2, ngenes=70_000, iteration=1, indptr=[0, 4, 6], genes=genes, n=[1] * 6, e=[1.0] * 6)
    raw = layout(payload_of(wide))
    assert raw["flags"] == 2 and raw["steps"] == [3, 65_531, 2, 4_463, 0, 69_998]
    counts = C.read(wide, ncells=2, ngenes=70_000)
    assert counts.genes.tolist() == genes and counts.flags == 2
    # 65,535 genes still fit 16 bits (the largest gene is 65,534).
    C.write(narrow, ncells=1, ngenes=65_535, iteration=1, indptr=[0, 2], genes=[0, 65_534], n=[1, 1], e=[1.0, 1.0])
    assert layout(payload_of(narrow))["flags"] == 0
    assert C.read(narrow).genes.tolist() == [0, 65_534]


def test_values_above_65535_are_stored_as_65535_and_counted(tmp_path):
    path = tmp_path / "big.counts.zst"
    stats = C.write(path, ncells=1, ngenes=4, iteration=3, indptr=[0, 4], genes=[0, 1, 2, 3],
                    n=[70_000, 65_535, 3, 2_100], e=[2_100.0, 2_047.96875, 1.0, 2_048.0],
                    f=[70_000, 65_535, 2, 2_000], start_e=[5_000.0, 1.0])
    raw = layout(payload_of(path))
    assert raw["n"] == [65_535, 65_535, 3, 2_100]          # 65,535 itself is kept, not clipped
    assert raw["e"] == [65_535, 65_535, 32, 65_535]        # 2,047.96875 x 32 = 65,535 exactly; 2,048 x 32 is over
    assert raw["f"] == [65_535, 65_535, 2, 2_000]
    assert raw["start_e"] == [65_535, 32]
    assert raw["saturated"] == stats["saturated"] == 5      # n 1, e 2, f 1, start 1
    counts = C.read(path)
    assert counts.saturated == 5 and counts.e.tolist() == [2_047.96875, 2_047.96875, 1.0, 2_047.96875]
    assert stats["sum_n"] == 65_535 + 65_535 + 3 + 2_100 and stats["sum_f"] == 65_535 + 65_535 + 2 + 2_000


def test_e_is_kept_in_steps_of_a_32nd_rounded_half_up(tmp_path):
    path = tmp_path / "round.counts.zst"
    e = [1 / 64, 0.015, 3 / 64, 5 / 64, 0.0, 1.0, 0.296875]
    C.write(path, ncells=1, ngenes=7, iteration=0, indptr=[0, 7], genes=list(range(7)), n=[1] * 7, e=e)
    assert layout(payload_of(path))["e"] == [1, 0, 2, 3, 0, 32, 10]     # halves go up, never to the even step
    assert C.read(path).e.tolist() == [1 / 32, 0.0, 2 / 32, 3 / 32, 0.0, 1.0, 10 / 32]


def test_empty_rows_and_a_run_without_cells(tmp_path):
    rows_path, none_path = tmp_path / "rows.counts.zst", tmp_path / "none.counts.zst"
    C.write(rows_path, ncells=5, ngenes=3, iteration=2, indptr=[0, 0, 2, 2, 3, 3], genes=[0, 2, 1], n=[1, 2, 3],
            e=[0.5, 1.0, 1.5], f=[0, 2, 1], start_e=[])
    counts = C.read(rows_path, ncells=5, ngenes=3)
    assert counts.indptr.tolist() == [0, 0, 2, 2, 3, 3] and counts.genes.tolist() == [0, 2, 1]
    assert counts.start_e.size == 0
    assert C.counted_of(counts).tolist() == [0.0, 1.5, 0.0, 1.5, 0.0]
    stats = C.write(none_path, ncells=0, ngenes=3, iteration=0, indptr=[0], genes=[], n=[], e=[], f=[])
    assert stats == {"nnz": 0, "sum_n": 0, "sum_e": 0.0, "sum_f": 0, "has_f": True, "has_start": False,
                     "saturated": 0, "bytes": none_path.stat().st_size}
    empty = C.read(none_path, ncells=0, ngenes=3)
    assert (empty.ncells, empty.nnz) == (0, 0) and empty.indptr.tolist() == [0] and empty.f.size == 0
    assert C.counted_of(empty).shape == (0,) and C.rows(empty, [])["e"].shape == (0, 3)


def test_a_file_written_as_a_stream_reads_the_same(tmp_path):
    """A zstd frame whose header does not give its size (as a streaming writer leaves it) reads like one that does."""
    path, streamed = tmp_path / "a.counts.zst", tmp_path / "streamed.counts.zst"
    write_small(path, start=True)
    out = io.BytesIO()
    with zstandard.ZstdCompressor(level=1, write_content_size=False).stream_writer(out, closefd=False) as writer:
        writer.write(payload_of(path))
    assert zstandard.get_frame_parameters(out.getvalue()).content_size == zstandard.CONTENTSIZE_UNKNOWN
    streamed.write_bytes(out.getvalue())
    a, b = C.read(path), C.read(streamed)
    for name in ("indptr", "genes", "n", "e", "f", "start_e"):
        assert getattr(a, name).tolist() == getattr(b, name).tolist()


# ---- refusals ---------------------------------------------------------------------------------------------------------
def test_a_missing_file_is_not_found_not_malformed(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.read(tmp_path / "gone.counts.zst")
    with pytest.raises(FileNotFoundError):
        C.read_counted(tmp_path / "gone.counted.zst")


def good_payload(tmp_path, **kwargs) -> bytes:
    path = tmp_path / "good.counts.zst"
    write_small(path, **kwargs)
    return payload_of(path)


def with_header(raw: bytes, **fields) -> bytes:
    names = ("version", "ncells", "ngenes", "nnz", "flags", "e_scale", "iteration", "saturated")
    values = dict(zip(names, struct.unpack_from("<8I", raw, 4)))
    values.update(fields)
    return raw[:4] + struct.pack("<8I", *(values[name] for name in names)) + raw[36:]


def with_u32(raw: bytes, at: int, value: int) -> bytes:
    return raw[:at] + struct.pack("<I", value) + raw[at + 4:]


def with_u16(raw: bytes, at: int, value: int) -> bytes:
    return raw[:at] + struct.pack("<H", value) + raw[at + 2:]


GENES_AT = 36 + 4 * 5                   # SMALL: the gene steps start after the header and five indptr values
MALFORMED = {
    "not zstd": (lambda raw: None, "not a whole zstd frame"),
    "header cut": (lambda raw: raw[:20], "cut short"),
    "magic": (lambda raw: b"PLC2" + raw[4:], "not a proseg counts file"),
    "version": (lambda raw: with_header(raw, version=2), "version 2"),
    "e_scale": (lambda raw: with_header(raw, e_scale=16), "1/16"),
    "unknown flag": (lambda raw: with_header(raw, flags=1 | 8), "flags this app does not know"),
    "32-bit genes for few genes": (lambda raw: with_header(raw, flags=1 | 2), "32-bit numbers for 6 genes"),
    "indptr not from 0": (lambda raw: with_u32(raw, 36, 1), "do not start at 0"),
    "indptr backwards": (lambda raw: with_u32(raw, 36 + 8, 2), "go backwards"),
    "indptr not to nnz": (lambda raw: with_u32(raw, 36 + 16, 5), "end at 5, not at its 6 entries"),
    "a gene past ngenes": (lambda raw: with_u16(raw, GENES_AT, 6), "past its 6"),
    "steps past ngenes": (lambda raw: with_u16(raw, GENES_AT + 4, 4), "past its 6"),      # row 0: 0, 2, 6
    "a gene twice in a row": (lambda raw: with_u16(raw, GENES_AT + 2, 0), "out of order"),
    "short": (lambda raw: raw[:-1], "cut short"),
    "trailing": (lambda raw: raw + b"\0\0", "2 bytes past its end"),
    "start block without its flag": (lambda raw: raw + struct.pack("<I", 1) + b"\1\0", "6 bytes past its end"),
    "start flag without a block": (lambda raw: with_header(raw, flags=1 | 4), "cut short"),
    "start block short": (lambda raw: with_header(raw, flags=1 | 4) + struct.pack("<I", 5) + b"\1\0", "cut short"),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_every_malformed_counts_file_is_refused_in_plain_words(tmp_path, case):
    change, words = MALFORMED[case]
    path = tmp_path / "bad.counts.zst"
    changed = change(good_payload(tmp_path))
    path.write_bytes(b"not a zstd frame at all" if changed is None else zstd_of(changed))
    with pytest.raises(C.CountsError, match=re.escape(words)) as caught:
        C.read(path)
    assert isinstance(caught.value, ValueError) and caught.value.reason == "no_counts"


def test_wide_genes_must_be_flagged_above_65535(tmp_path):
    path = tmp_path / "wide.counts.zst"
    C.write(path, ncells=1, ngenes=70_000, iteration=1, indptr=[0, 1], genes=[69_999], n=[1], e=[1.0])
    path.write_bytes(zstd_of(with_header(payload_of(path), flags=0)))
    with pytest.raises(C.CountsError, match="keeps 70000 genes as 16-bit numbers"):
        C.read(path)


def test_zstd_that_is_cut_or_empty_is_refused(tmp_path):
    path = tmp_path / "cut.counts.zst"
    write_small(path)
    whole = path.read_bytes()
    for broken in (whole[:len(whole) // 2], b""):
        path.write_bytes(broken)
        with pytest.raises(C.CountsError, match="not a whole zstd frame"):
            C.read(path)


def test_a_file_of_another_run_says_whose_sizes_differ(tmp_path):
    path = tmp_path / "a.counts.zst"
    write_small(path)
    with pytest.raises(C.CountsError, match=re.escape("the counts file holds 4 cells, the run 5")):
        C.read(path, ncells=5, ngenes=6)
    with pytest.raises(C.CountsError, match=re.escape("the counts file holds 6 genes, the run 7")):
        C.read(path, ncells=4, ngenes=7)


# ---- what the workers and the engine read -----------------------------------------------------------------------------
def dense(name: str) -> np.ndarray:
    out = np.zeros((SMALL["ncells"], SMALL["ngenes"]), np.float32)
    for cell in range(SMALL["ncells"]):
        for at in range(SMALL["indptr"][cell], SMALL["indptr"][cell + 1]):
            out[cell, SMALL["genes"][at]] = SMALL[name][at]
    return out


def test_rows_are_dense_float32_with_zeros_outside_each_cells_genes(tmp_path):
    path = tmp_path / "a.counts.zst"
    write_small(path)
    counts = C.read(path)
    got = C.rows(counts, [2, 0, 2, 1], ("e", "n", "f"))
    assert sorted(got) == ["e", "f", "n"]
    for name in ("e", "n", "f"):
        assert got[name].dtype == np.float32 and got[name].shape == (4, 6)
        assert np.array_equal(got[name], dense(name)[[2, 0, 2, 1]])
    assert sorted(C.rows(counts, np.array([3]))) == ["e", "n"]               # e and n unless asked otherwise
    assert np.array_equal(C.rows(counts, [3], "e")["e"], dense("e")[[3]])     # one name as text
    for cells in ([4], [-1]):
        with pytest.raises(ValueError, match="below 4"):
            C.rows(counts, cells)
    with pytest.raises(ValueError, match="no count named"):
        C.rows(counts, [0], ("x",))
    write_small(path, f=False)
    with pytest.raises(C.CountsError, match="no sampled count"):
        C.rows(C.read(path), [0], ("e", "f"))


@pytest.mark.parametrize("which", ["e", "n", "f"])
def test_to_csr_gives_the_matrix_of_one_count(tmp_path, which):
    path = tmp_path / "a.counts.zst"
    write_small(path)
    counts = C.read(path)
    csr = C.to_csr(np, counts, which)
    assert csr.shape == (4, 6)
    assert (csr.data.dtype, csr.indices.dtype, csr.indptr.dtype) == (np.float32, np.int32, np.int32)
    matrix = np.zeros(csr.shape, np.float32)
    matrix[np.repeat(np.arange(4), np.diff(csr.indptr)), csr.indices] = csr.data
    assert np.array_equal(matrix, dense(which))
    csr.data[:] = -1                                     # the caller's own arrays: the counts stay as read
    csr.indices[:] = 0
    assert counts.e.tolist() == SMALL["e"] and counts.genes.tolist() == SMALL["genes"]


def test_to_csr_needs_the_count_it_is_asked_for(tmp_path):
    path = tmp_path / "a.counts.zst"
    write_small(path, f=False)
    counts = C.read(path)
    assert C.to_csr(np, counts).data.tolist() == np.float32(SMALL["e"]).tolist()     # e unless asked otherwise
    with pytest.raises(C.CountsError, match="no sampled count"):
        C.to_csr(np, counts, "f")
    with pytest.raises(ValueError, match="no count named"):
        C.to_csr(np, counts, "x")


def test_the_start_block_is_read_on_the_init_frames_rows(tmp_path):
    init_path, later_path = tmp_path / "000000.counts.zst", tmp_path / "000040.counts.zst"
    C.write(init_path, ncells=4, ngenes=6, iteration=0, n=[2, 1, 3, 1], e=[1.5, 0.5, 2.0, 1.0], **INIT)
    write_small(later_path, start=True)
    init, later = C.read(init_path), C.read(later_path)
    csr = C.start_csr(np, init, later)
    assert csr.shape == (4, 6) and csr.data.dtype == np.float32
    assert csr.data.tolist() == START_E
    assert csr.indices.tolist() == INIT["genes"] and csr.indptr.tolist() == INIT["indptr"]
    assert csr.indptr.dtype == np.int32 and csr.indices.dtype == np.int32
    with pytest.raises(C.CountsError, match="no start block"):
        C.start_csr(np, init, init)
    other = tmp_path / "other.counts.zst"
    C.write(other, ncells=4, ngenes=7, iteration=0, n=[2, 1, 3, 1], e=[1.5, 0.5, 2.0, 1.0], **INIT)
    with pytest.raises(C.CountsError, match="7 genes"):
        C.start_csr(np, C.read(other), later)
    C.write(other, ncells=4, ngenes=6, iteration=0, indptr=[0, 2, 3, 3, 3], genes=[0, 2, 0], n=[2, 1, 3],
            e=[1.5, 0.5, 2.0])
    with pytest.raises(C.CountsError, match="the start block holds 4 entries, the start's counts 3"):
        C.start_csr(np, C.read(other), later)
    C.write(other, ncells=3, ngenes=6, iteration=0, indptr=[0, 2, 3, 4], genes=[0, 2, 0, 1], n=[2, 1, 3, 1],
            e=[1.5, 0.5, 2.0, 1.0])
    with pytest.raises(C.CountsError, match="3 cells"):
        C.start_csr(np, C.read(other), later)


def test_the_sidecar_holds_each_cells_e_summed_in_float64(tmp_path):
    """counted_of sums a row in float64 and rounds once to float32: a long row of large values keeps its 32nds."""
    path, sidecar = tmp_path / "a.counts.zst", tmp_path / "a.counted.zst"
    ngenes = 400
    e = [2_047.0 + (g % 31) / 32 for g in range(ngenes)] + [0.03125, 7.5]
    C.write(path, ncells=3, ngenes=ngenes, iteration=5, indptr=[0, ngenes, ngenes, ngenes + 2],
            genes=list(range(ngenes)) + [0, 399], n=[2_048] * ngenes + [1, 8], e=e)
    counts = C.read(path, ncells=3)
    values = C.counted_of(counts)
    want = [np.float32(math.fsum(e[:ngenes])), 0.0, np.float32(0.03125 + 7.5)]
    assert values.dtype == np.float32 and values.tolist() == [float(v) for v in want]
    C.write_counted(sidecar, values)
    assert np.frombuffer(payload_of(sidecar), "<f4").tolist() == values.tolist()
    back = C.read_counted(sidecar, ncells=3)
    assert back.dtype == np.float32 and back.tolist() == values.tolist()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(C.CountsError, match=re.escape("the per-cell sidecar holds 3 cells, the run 4")):
        C.read_counted(sidecar, ncells=4)
    # Wide rows of large values: a float32 sum, in any order, misses the rounded float64 sum in many of them.
    stored = np.random.default_rng(11).integers(48_000, 65_535, (40, 1_024))
    C.write(path, ncells=40, ngenes=1_024, iteration=5, indptr=np.arange(41) * 1_024,
            genes=np.tile(np.arange(1_024), 40), n=np.full(40 * 1_024, 2_048), e=stored.ravel() / 32)
    wide = C.counted_of(C.read(path))
    assert wide.tolist() == [float(np.float32(math.fsum(row / 32))) for row in stored]
    sidecar.write_bytes(zstd_of(b"\0" * 10))
    with pytest.raises(C.CountsError, match="cut short"):
        C.read_counted(sidecar)
    sidecar.write_bytes(b"not zstd")
    with pytest.raises(C.CountsError, match="not a whole zstd frame"):
        C.read_counted(sidecar)


# ---- writing ----------------------------------------------------------------------------------------------------------
def test_write_returns_what_a_counts_line_says_of_the_file(tmp_path):
    path = tmp_path / "a.counts.zst"
    stats = write_small(path, start=True)
    counts = C.read(path)
    assert stats == {"nnz": 6, "sum_n": sum(SMALL["n"]), "sum_e": math.fsum(SMALL["e"]), "sum_f": sum(SMALL["f"]),
                     "has_f": True, "has_start": True, "saturated": 0, "bytes": path.stat().st_size}
    assert stats["sum_e"] == float(np.sum(C.counted_of(counts), dtype=np.float64))
    plain = write_small(path, f=False)
    assert (plain["sum_f"], plain["has_f"], plain["has_start"]) == (None, False, False)
    # Whole numbers as floats (as numpy's floor gives them) are taken for n and f.
    floored = C.write(path, ncells=1, ngenes=2, iteration=0, indptr=[0, 2], genes=[0, 1], n=np.array([4.0, 2.0]),
                      e=[3.0, 1.5], f=np.floor(np.array([4.0, 2.0]) * 0.75))
    assert floored["sum_f"] == 4 and C.read(path).f.tolist() == [3, 1]


REFUSED = {
    "indptr length": {"indptr": [0, 3, 3, 6]},
    "indptr not from 0": {"indptr": [1, 3, 3, 5, 6]},
    "indptr backwards": {"indptr": [0, 3, 2, 5, 6]},
    "indptr not to nnz": {"indptr": [0, 3, 3, 5, 5]},
    "gene past ngenes": {"genes": [0, 2, 6, 1, 4, 3]},
    "gene below 0": {"genes": [-1, 2, 5, 1, 4, 3]},
    "genes not rising": {"genes": [0, 2, 2, 1, 4, 3]},
    "genes falling": {"genes": [0, 5, 2, 1, 4, 3]},
    "n of 0": {"n": [3, 0, 7, 2, 4, 1], "f": [2, 0, 6, 1, 4, 0]},
    "n not whole": {"n": [3, 1.5, 7, 2, 4, 1]},
    "n short": {"n": [3, 1, 7, 2, 4]},
    "e below 0": {"e": [2.5, -0.5, 6.0, 1.25, 3.75, 0.5]},
    "e not a number": {"e": [2.5, float("nan"), 6.0, 1.25, 3.75, 0.5]},
    "e short": {"e": [2.5]},
    "f above n": {"f": [2, 2, 6, 1, 4, 0]},
    "f below 0": {"f": [2, -1, 6, 1, 4, 0]},
    "f short": {"f": [2, 0]},
    "start_e below 0": {"start_e": [1.0, -1.0]},
    "start_e infinite": {"start_e": [float("inf")]},
    "ncells below 0": {"ncells": -1},
    "ngenes not whole": {"ngenes": 6.5},
    "iteration too large": {"iteration": 1 << 32},
    "ncells missing": {"ncells": None},
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_write_refuses_inputs_that_do_not_fit_together(tmp_path, case):
    path = tmp_path / "a.counts.zst"
    with pytest.raises(ValueError):
        C.write(path, **dict(SMALL, **REFUSED[case]))
    assert not path.exists()


# ---- the contract other parts read --------------------------------------------------------------------------------
def test_the_switches_and_names_other_parts_read():
    assert C.LIVE_COUNT == "averaged" and C.LIVE_ARRAY == "e"
    assert C.ARRAY == {"averaged": "e", "sampled": "f"} and C.ARRAY[C.LIVE_COUNT] == C.LIVE_ARRAY
    assert C.SAVED_REASONS == ("pause", "hold", "finish", "end")
    assert C.BASIS == "pc1" and C.E_SCALE == 32 and C.MAGIC == b"PLC1" and C.VERSION == 1
    assert (C.FLAG_F, C.FLAG_GENES32, C.FLAG_START) == (1, 2, 4)
    assert issubclass(C.CountsError, ValueError) and C.CountsError.reason == "no_counts"
    assert C.CsrArrays._fields == ("data", "indices", "indptr", "shape")


@pytest.mark.parametrize("reason", ["init", "upscale", "request", "every", "pause", "hold", "finish", "end"])
@pytest.mark.parametrize("has_f", [True, False, None, "true", 1])
def test_a_frame_is_read_on_f_only_when_proseg_counted_it_for_a_pause_the_hold_a_finish_or_the_end(reason, has_f):
    """count_of, the one rule (decision A): a counts line of reason pause, hold, finish or end that holds f (has_f
    true, nothing truthy else) is read on the count proseg saves, "sampled"; every other line on the averaged count,
    init and upscale too, which carry f as well."""
    line = {"seq": 3, "reason": reason, "has_f": has_f}
    saved = reason in ("pause", "hold", "finish", "end") and has_f is True
    assert C.count_of(line) == ("sampled" if saved else "averaged")


def test_no_line_is_the_averaged_count():
    for line in (None, {}, [], "hold", {"reason": "hold"}, {"has_f": True}, {"reason": None, "has_f": True}):
        assert C.count_of(line) == "averaged", line


def test_it_imports_nothing_but_numpy_numcodecs_and_the_standard_library():
    """The GPU worker's environment, the CPU worker's and the engine's all read these files."""
    tree = ast.parse(open(C.__file__).read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no import of the app's own modules"
            found.add(node.module.split(".")[0])
    assert found - set(sys.stdlib_module_names) == {"numpy", "numcodecs"}
