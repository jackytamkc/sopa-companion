"""Proseg's share of each transcript for one cell, w (0 to 1), for Go deeper's pink dots and the settings test's
targets: exact from proseg's own pools where it exported them (a where export), else approximate from the frame's
per-layer labels, else none. Never from a frame's flat 2-D projection: a tiled frame outside its window has no share
at all ("none").

Proseg pools a gene's transcripts per (current voxel, offset from where each was seen) and does not track which is
which, so a pool is shared evenly among the transcripts seen in its bucket B: one gene at one observed voxel (column
i, row j, layer k) of the frame's grid. For cell c and a transcript of gene g:
  - exact: w = d_B(c) x q(c, g), with d_B(c) = min(1, the pooled transcripts of B whose destination c owns / |B|): the
    numerator from proseg's where export, |B| from the run's own transcripts (so a bucket is never shared out among
    fewer than it holds);
  - approx: w = [c owns the transcript's observed voxel in the frame's per-layer labels] x q(c, g), from a dense
    frame's layers or a tiled frame's window, only when they cover every transcript asked;
  - none: w = 0.
q(c, g) = e / n of the cell in that frame's counts (analysis/counts_file.py: the averaged count over the assigned one;
0 where n = 0), or f / n on a frame read on the count proseg saves (f: one counted for a pause, the hold, a finish or the
end). Over a box holding the cell's pools the exact w add up to the cell's e (or f), gene by gene. `others` gives the
same share of each transcript for every other cell at once (the cells around: what proseg counts of the box's RNA for
them).

The grid rule is the worker's (grid_columns, and its voxel_lookup's layer), in float32 as proseg's: Python-float origins
keep numpy in float32.

where-N.bin.zst, which proseg-live writes when asked for a box (where.json, its record, is written after it): one zstd
frame, little-endian, a struct of arrays: u32 count; u32 i[count], j[count]; u8 k[count]; u32 gene[count],
dest[count], n[count]. One record per pooled entry whose observed voxel lies in [i0..i1] x [j0..j1] (any layer); dest
is the label of the cell owning its current voxel (0: none). Records are not merged: equal (i, j, k, gene, dest) add
up."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct

import numpy as np

from ..analysis.worker import grid_columns

HELD = ("exact", "approx", "none")
SLACK_UM = 0.01             # covers(): an export's span may fall this short of the box (the record's rounding)
U32_MAX = (1 << 32) - 1


@dataclass(frozen=True, eq=False)
class WhereExport:
    """One where export read whole: its record (where.json, or a counts line's "where") and its arrays (int64 each)."""
    token: int
    seq: int
    iter: int
    box: tuple
    margin_um: float
    x0: float
    y0: float
    voxel: float
    nx: int
    ny: int
    nz: int
    i0: int
    j0: int
    i1: int
    j1: int
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    gene: np.ndarray
    dest: np.ndarray
    n: np.ndarray


def read_where(meta: dict, path=None) -> WhereExport:
    """The export a record describes: its file is meta["path"] when the record carries one, else `path`. ValueError
    when the record cannot be read, or the file is not one zstd frame of `entries` records inside the record's span
    (i in [i0, i1], j in [j0, j1], k < nz); a missing file raises FileNotFoundError as it is."""
    source = meta.get("path") or path
    if not source:
        raise ValueError("no file given for this where export")
    try:
        whole = {key: int(meta[key]) for key in ("token", "seq", "iter", "nx", "ny", "nz", "i0", "j0", "i1", "j1",
                                                 "entries")}
        amounts = {key: float(meta[key]) for key in ("margin_um", "x0", "y0", "voxel")}
        box = tuple(float(v) for v in meta["box"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a where export's record cannot be read") from None
    if len(box) != 4:
        raise ValueError("a where export's box needs four numbers")
    data = _unzstd(Path(source).read_bytes())
    count = int.from_bytes(data[:4], "little") if len(data) >= 4 else -1
    if count < 0 or len(data) != 4 + 21 * count:
        raise ValueError("a where export's arrays do not hold its count")
    entries = whole.pop("entries")
    if count != entries:
        raise ValueError(f"a where export holds {count} records, its record says {entries}")
    arrays, at = {}, 4
    for name, kind, size in (("i", "<u4", 4), ("j", "<u4", 4), ("k", "u1", 1), ("gene", "<u4", 4), ("dest", "<u4", 4),
                             ("n", "<u4", 4)):
        arrays[name] = np.frombuffer(data, kind, count, at).astype(np.int64)
        at += size * count
    if count and (int(arrays["i"].min()) < whole["i0"] or int(arrays["i"].max()) > whole["i1"]
                  or int(arrays["j"].min()) < whole["j0"] or int(arrays["j"].max()) > whole["j1"]
                  or int(arrays["k"].max()) >= whole["nz"]):
        raise ValueError("a where export holds records outside its span")
    return WhereExport(box=box, **whole, **amounts, **arrays)


def write_where(path, i, j, k, gene, dest, n) -> None:
    """Write a where export's file (zstd level 1, atomically: `<path>.tmp`, then renamed): for the tests and the fake
    proseg. ValueError for arrays of different lengths or values their fields cannot hold."""
    columns = [np.asarray(values) for values in (i, j, k, gene, dest, n)]
    count = columns[0].size
    parts = [struct.pack("<I", count)]
    for name, values, top, kind in zip(("i", "j", "k", "gene", "dest", "n"), columns,
                                       (U32_MAX, U32_MAX, 255, U32_MAX, U32_MAX, U32_MAX),
                                       ("<u4", "<u4", "u1", "<u4", "<u4", "<u4")):
        if values.ndim != 1 or values.size != count:
            raise ValueError("a where export's arrays need one length")
        if values.size and (values.dtype.kind not in "iu" or int(values.min()) < 0 or int(values.max()) > top):
            raise ValueError(f"a where export's {name} must be whole numbers from 0 to {top}")
        parts.append(values.astype(kind).tobytes())
    import zstandard                    # the engine's own (requirements.txt)
    data = zstandard.ZstdCompressor(level=1).compress(b"".join(parts))
    temporary = f"{path}.tmp"
    Path(temporary).write_bytes(data)
    os.replace(temporary, path)


def covers(meta: dict, box) -> bool:
    """Whether an export's span (its record, or a WhereExport) holds the box [x0, y0, x1, y1] µm, within SLACK_UM. A
    side of the span at the grid's edge (i0 0, i1 nx - 1, j0 0, j1 ny - 1) holds the box on that side however far the
    box reaches past it: proseg clamps a span to its grid, and no voxel lies beyond (a record without nx or ny: only
    its left and bottom edges are known)."""
    x0, y0, voxel = float(_get(meta, "x0")), float(_get(meta, "y0")), float(_get(meta, "voxel"))
    i0, j0, i1, j1 = (int(_get(meta, key)) for key in ("i0", "j0", "i1", "j1"))
    nx, ny = (_get(meta, key) if _has(meta, key) else None for key in ("nx", "ny"))
    return ((i0 <= 0 or x0 + i0 * voxel <= box[0] + SLACK_UM)
            and ((nx is not None and i1 >= int(nx) - 1) or x0 + (i1 + 1) * voxel >= box[2] - SLACK_UM)
            and (j0 <= 0 or y0 + j0 * voxel <= box[1] + SLACK_UM)
            and ((ny is not None and j1 >= int(ny) - 1) or y0 + (j1 + 1) * voxel >= box[3] - SLACK_UM))


def query_box(meta_or_grid: dict, box) -> list[float]:
    """The box to ask the run's transcript index for (half-open, as TranscriptIndex.query takes it) so that every
    bucket of `box` comes whole: its columns and rows, one more on each side (float32 placement may put a transcript a
    hair across an edge), within the grid."""
    x0, y0, voxel = float(_get(meta_or_grid, "x0")), float(_get(meta_or_grid, "y0")), float(_get(meta_or_grid, "voxel"))
    nx, ny = int(_get(meta_or_grid, "nx")), int(_get(meta_or_grid, "ny"))
    c0 = max(0, math.floor((box[0] - x0) / voxel) - 1)
    c1 = min(nx - 1, math.floor((box[2] - x0) / voxel) + 1)
    r0 = max(0, math.floor((box[1] - y0) / voxel) - 1)
    r1 = min(ny - 1, math.floor((box[3] - y0) / voxel) + 1)
    return [x0 + c0 * voxel, y0 + r0 * voxel, x0 + (c1 + 1) * voxel, y0 + (r1 + 1) * voxel]


def buckets(records, grid: dict, z_origin: float, z_spacing: float, span=None) -> np.ndarray:
    """Each record's bucket (transcript_index RECORD: x, y, z, gene) on the grid {x0, y0, voxel, nx, ny, nz} of a frame
    record (or an export's record): int64 ((gene * nz + k) * ny + j) * nx + i; -1 off the grid or, with span
    (i0, j0, i1, j1), outside it."""
    i, j, k, inside = _voxels(records, grid, z_origin, z_spacing)
    if span is not None:
        i0, j0, i1, j1 = span
        inside &= (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
    key = _keys(i, j, k, np.asarray(records["gene"]).astype(np.int64), int(_get(grid, "nx")), int(_get(grid, "ny")),
                int(_get(grid, "nz")))
    return np.where(inside, key, -1)


def weights(records, cell: int, q_row, grid: dict, z_origin: float, z_spacing: float,
            export: WhereExport | None = None, layers=None) -> tuple[np.ndarray, str]:
    """(w per record, float64; held: "exact" | "approx" | "none") for cell `cell` (label - 1), given the cell's keep
    q_row (float [ngenes]: e / n in the frame's counts, 0 where n = 0).
      - exact, with `export`: d_B(cell) x q, on the export's own grid (its frame's); |B| counts the records passed in
        the export's span, so they must hold every transcript of its buckets (query_box); records outside the span: 0.
      - approx, without one, given `layers` = (labels u32 [nz, h, w], c0, r0): the window of the frame's per-layer
        labels whose first column is c0 and first row r0 (a dense frame's whole layers: 0, 0), when it covers every
        record on `grid`: [labels at the record's voxel == cell + 1] x q; records off the grid: 0.
      - none otherwise (no export, and no layers or layers that miss a record): 0 everywhere.
    Nothing is rounded here."""
    w, held = weights_many(records, [cell], np.asarray(q_row, np.float64).reshape(1, -1), grid, z_origin, z_spacing,
                           export, layers)
    return w[0], held


def weights_many(records, cells, q_rows, grid: dict, z_origin: float, z_spacing: float,
                 export: WhereExport | None = None, layers=None) -> tuple[np.ndarray, str]:
    """`weights` for several cells at once (Go deeper's cell and its sources), the records' buckets or voxels worked
    out once: (w float64 [len(cells), records], held). `q_rows`: each cell's keep, float [len(cells), ngenes]."""
    q = np.asarray(q_rows, np.float64)
    labels_of = np.asarray(cells, np.int64).reshape(-1) + 1
    if q.ndim != 2 or q.shape[0] != labels_of.size:
        raise ValueError("one keep row per cell")
    gene = np.asarray(records["gene"]).astype(np.int64)
    if gene.size and int(gene.max()) >= q.shape[1]:
        raise ValueError(f"a transcript of gene {int(gene.max())}, past the cell's {q.shape[1]} genes")
    w = np.zeros((labels_of.size, gene.size), np.float64)
    if export is not None:
        key = buckets(records, _grid_of(export), z_origin, z_spacing, (export.i0, export.j0, export.i1, export.j1))
        inside = np.flatnonzero(key >= 0)
        if inside.size and labels_of.size:
            found, of, size = np.unique(key[inside], return_inverse=True, return_counts=True)
            of = of.reshape(-1)
            for c, label in enumerate(labels_of.tolist()):
                mine = export.dest == label
                own = _keys(export.i[mine], export.j[mine], export.k[mine], export.gene[mine], export.nx, export.ny,
                            export.nz)
                at = np.minimum(np.searchsorted(found, own), found.size - 1)
                hit = found[at] == own          # pools of buckets no record passed lies in add nothing
                pooled = np.bincount(at[hit], weights=export.n[mine][hit].astype(np.float64), minlength=found.size)
                w[c, inside] = np.minimum(1.0, pooled / size)[of] * q[c, gene[inside]]
        return w, "exact"
    if layers is not None:
        labels, c0, r0 = layers
        labels = np.asarray(labels)
        if labels.ndim != 3 or labels.shape[0] != int(_get(grid, "nz")):
            raise ValueError("the layers do not have the frame's layers")
        i, j, k, on = _voxels(records, grid, z_origin, z_spacing)
        li, lj = i - int(c0), j - int(r0)
        within = (li >= 0) & (li < labels.shape[2]) & (lj >= 0) & (lj < labels.shape[1])
        if (within | ~on).all():
            sel = np.flatnonzero(on)
            at = labels[k[sel], lj[sel], li[sel]]
            for c, label in enumerate(labels_of.tolist()):
                w[c, sel] = (at == label) * q[c, gene[sel]]
            return w, "approx"
    return w, "none"


def keeps(counts, cells, which: str = "e") -> np.ndarray:
    """Each cell's keep q = which / n per gene in a frame's counts (analysis/counts_file.py Counts; `which` "e", the
    averaged count, or "f", the count proseg saves): float64 [len(cells), ngenes], 0 where the cell holds none of a
    gene. CountsError for "f" from a frame without it."""
    from ..analysis import counts_file
    table = counts_file.rows(counts, cells, (which, "n"))
    e, n = table[which].astype(np.float64), table["n"].astype(np.float64)
    return np.divide(e, n, out=np.zeros_like(e), where=n > 0)


def keep_at(counts, cells, genes, which: str = "e") -> np.ndarray:
    """The keep q = which / n of each (cell, gene) pair in a frame's counts (`cells`, `genes`: arrays of one length):
    float64, 0 where the cell holds none of the gene; only the rows of the cells named are looked at. CountsError for
    "f" from a frame without it."""
    from ..analysis import counts_file
    cells, genes = np.asarray(cells, np.int64).reshape(-1), np.asarray(genes, np.int64).reshape(-1)
    values = {"e": counts.e, "f": counts.f}[which]
    if values is None:
        raise counts_file.CountsError("this frame's counts hold no sampled count (f)")
    out = np.zeros(cells.size, np.float64)
    if not cells.size:
        return out
    named, at = np.unique(cells, return_inverse=True)
    starts = counts.indptr[named]
    lengths = counts.indptr[named + 1] - starts
    entry = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(int(lengths.sum()), dtype=np.int64)
    keyed = np.repeat(np.arange(named.size, dtype=np.int64), lengths) * counts.ngenes + counts.genes[entry]   # rising
    if not keyed.size:
        return out
    want = at.reshape(-1) * counts.ngenes + genes
    found = np.minimum(np.searchsorted(keyed, want), keyed.size - 1)
    hit = keyed[found] == want
    pick = entry[found[hit]]
    out[hit] = np.asarray(values, np.float64)[pick] / np.maximum(counts.n[pick].astype(np.float64), 1.0)
    return out


def others(records, exclude, counts, grid: dict, z_origin: float, z_spacing: float, which: str = "e",
           export: WhereExport | None = None, layers=None) -> tuple[np.ndarray, str]:
    """(proseg's share of each record held by any cell but those in `exclude` (cell numbers, label - 1): float64 per
    record; held), by `weights`' rules, every other cell at once (Go deeper's cells around), its keep q = which / n from
    the frame's counts (`counts`):
      - exact, with `export`: per record, the sum over its bucket's pools whose destination is another cell (not 0, not
        one in `exclude`, one of the frame's cells) of min(1, those pools of that cell / |B|) x q(that cell, gene);
        records outside the export's span: 0;
      - approx, without one, given `layers` covering every record on `grid`: [the label at the record's voxel is another
        cell] x q(that cell, gene);
      - none otherwise: 0 everywhere.
    Nothing is rounded here."""
    gene = np.asarray(records["gene"]).astype(np.int64)
    out = np.zeros(gene.size, np.float64)
    left_out = {int(c) + 1 for c in exclude}
    if export is not None:
        key = buckets(records, _grid_of(export), z_origin, z_spacing, (export.i0, export.j0, export.i1, export.j1))
        inside = np.flatnonzero(key >= 0)
        if not inside.size:
            return out, "exact"
        found, of, size = np.unique(key[inside], return_inverse=True, return_counts=True)
        dest = export.dest
        theirs = (dest > 0) & (dest <= counts.ncells) & ~np.isin(dest, sorted(left_out))
        pool = _keys(export.i[theirs], export.j[theirs], export.k[theirs], export.gene[theirs], export.nx, export.ny,
                     export.nz)
        at = np.minimum(np.searchsorted(found, pool), found.size - 1)
        hit = found[at] == pool                      # pools of buckets no record passed lies in add nothing
        bucket, cell = at[hit], dest[theirs][hit] - 1
        pairs, where = np.unique(bucket * (counts.ncells + 1) + cell, return_inverse=True)
        pooled = np.bincount(where.reshape(-1), weights=export.n[theirs][hit].astype(np.float64), minlength=pairs.size)
        pair_bucket, pair_cell = pairs // (counts.ncells + 1), pairs % (counts.ncells + 1)
        pair_gene = found[pair_bucket] // (int(export.nx) * int(export.ny) * int(export.nz))    # a bucket's gene
        share = np.minimum(1.0, pooled / size[pair_bucket]) * keep_at(counts, pair_cell, pair_gene, which)
        out[inside] = np.bincount(pair_bucket, weights=share, minlength=found.size)[of.reshape(-1)]
        return out, "exact"
    if layers is not None:
        labels, c0, r0 = layers
        labels = np.asarray(labels)
        if labels.ndim != 3 or labels.shape[0] != int(_get(grid, "nz")):
            raise ValueError("the layers do not have the frame's layers")
        i, j, k, on = _voxels(records, grid, z_origin, z_spacing)
        li, lj = i - int(c0), j - int(r0)
        within = (li >= 0) & (li < labels.shape[2]) & (lj >= 0) & (lj < labels.shape[1])
        if (within | ~on).all():
            sel = np.flatnonzero(on)
            at = labels[k[sel], lj[sel], li[sel]].astype(np.int64)
            theirs = (at > 0) & (at <= counts.ncells) & ~np.isin(at, sorted(left_out))
            out[sel[theirs]] = keep_at(counts, at[theirs] - 1, gene[sel[theirs]], which)
            return out, "approx"
    return out, "none"


def _voxels(records, grid, z_origin: float, z_spacing: float):
    """(i, j, k) int64 of each record's observed voxel on the grid, and whether (i, j) lies on it: grid_columns, and
    the layer clipped to the grid's layers as voxel_lookup takes it, in float32 as proseg's."""
    x0, y0, voxel = float(_get(grid, "x0")), float(_get(grid, "y0")), float(_get(grid, "voxel"))
    nx, ny, nz = int(_get(grid, "nx")), int(_get(grid, "ny")), int(_get(grid, "nz"))
    i, j, inside = grid_columns(np, np.asarray(records["x"]), np.asarray(records["y"]), x0, y0, voxel, nx, ny)
    z = np.asarray(records["z"])
    k = np.clip(np.floor((z - float(z_origin)) / float(z_spacing)), 0, nz - 1).astype(np.int64)
    return i.astype(np.int64), j.astype(np.int64), k, inside


def _keys(i, j, k, gene, nx: int, ny: int, nz: int) -> np.ndarray:
    return ((gene * nz + k) * ny + j) * nx + i


def _grid_of(export: WhereExport) -> dict:
    return {"x0": export.x0, "y0": export.y0, "voxel": export.voxel, "nx": export.nx, "ny": export.ny,
            "nz": export.nz}


def _get(meta, name: str):
    """A field of a record (a dict) or of a WhereExport."""
    return meta[name] if isinstance(meta, Mapping) else getattr(meta, name)


def _has(meta, name: str) -> bool:
    """Whether a record (a dict) or a WhereExport has the field, and it is set."""
    return (meta.get(name) if isinstance(meta, Mapping) else getattr(meta, name, None)) is not None


def _unzstd(raw: bytes) -> bytes:
    """One whole zstd frame's bytes (with its size in its header or not); ValueError otherwise."""
    import zstandard                    # the engine's own (requirements.txt)
    reader = zstandard.ZstdDecompressor().decompressobj()
    try:
        data = reader.decompress(raw)
    except zstandard.ZstdError:
        raise ValueError("a where export is not a zstd frame") from None
    if not reader.eof or reader.unused_data:
        raise ValueError("a where export is not one whole zstd frame")
    return data
