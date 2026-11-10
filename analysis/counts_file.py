"""Proseg's own count of each cell, gene by gene, as proseg-live writes it with a counted frame (PLC1 files), and the
per-cell sidecar beside it: the one reader and writer the GPU and CPU workers, the engine and the tests' fake proseg
share. numpy and the zstd codec the worker reads proseg's frames with (numcodecs, imported where it is used), nothing
else, so it loads wherever the worker or the engine runs.

Three counts per (cell, gene), named the same everywhere:
  - n, the assigned count: the transcripts whose destination after drift the cell owns;
  - f, the sampled count: proseg's draw, at that iteration, of what it keeps as the cell's RNA (the saved table holds the
    draw of the last iteration);
  - e, the averaged count: f averaged over that random background draw, given the frame's state (per entry of the cell:
    its count x lambda_cell / (lambda_cell + lambda_background) of the entry's gene, layer and density).
Every live number reads e (LIVE_COUNT is the one switch); f comes with the frames proseg stops at or is made to keep
(init, upscale, pause, hold, finish, end). A frame counted for a pause, the hold, a finish or the end (SAVED_REASONS)
with f is read on f, the count proseg saves (`count_of`: "sampled"); every other counted frame on LIVE_COUNT's.

frames/NNNNNN.counts.zst (or in the run's spool) is one zstd frame of little-endian fields, with no padding:
    "PLC1", then u32 version (1), ncells, ngenes, nnz, flags, e_scale (32), iteration, saturated
    indptr     u32 x (ncells + 1)         rows: cell = label - 1; a row holds the genes with n > 0
    gene       u16 x nnz (u32: GENES32)   per row ascending: its first gene, then each the step from the one before
    n          u16 x nnz
    e          u16 x nnz                  floor(e x 32 + 0.5), e worked out in float64: 1/32 of a transcript
    f          u16 x nnz                  with F only; f <= n
    start_nnz  u32                        with START only: the init frame's nnz
    start_e    u16 x start_nnz            with START only, stored as e: proseg's iteration-0 assignment averaged with
                                          this frame's background split, on the init frame's rows and genes
flags: F 1, GENES32 2 (set exactly when ngenes > 65,535), START 4. A value above 65,535 is stored as 65,535, and
`saturated` says how many were (n, e, f and start_e together).

frames/NNNNNN.counted.zst, the sidecar: one zstd frame of float32 x ncells, the sum of each row's stored e (summed in
float64, then float32), the page's number for a cell."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import Any, NamedTuple

import numpy as np

MAGIC = b"PLC1"
VERSION = 1
E_SCALE = 32
U16_MAX = 65535
FLAG_F, FLAG_GENES32, FLAG_START = 1, 2, 4
BASIS = "pc1"                      # results on this count live apart: qc/<model8>.pc1/, analysis/pc1/
LIVE_COUNT = "averaged"            # which count every live number reads: "averaged" (e) or "sampled" (f)
ARRAY = {"averaged": "e", "sampled": "f"}   # each count's array in a counts file
LIVE_ARRAY = ARRAY[LIVE_COUNT]
SAVED_REASONS = ("pause", "hold", "finish", "end")   # a frame proseg counts for these, with f, is read on f: the count
                                                     # proseg saves there (its table, when the run finishes there)
NAMES = ("e", "n", "f")
HEADER = struct.Struct("<4s8I")    # magic, version, ncells, ngenes, nnz, flags, e_scale, iteration, saturated
U32_MAX = (1 << 32) - 1
INT32_MAX = (1 << 31) - 1


class CountsError(ValueError):
    """A counts file that is malformed, or that does not fit the run (its cells or genes)."""
    reason = "no_counts"


@dataclass(frozen=True, eq=False)
class Counts:
    """One counted frame as read: CSR rows (cell = label - 1) over the run's genes (genes.txt order)."""
    ncells: int
    ngenes: int
    iteration: int
    saturated: int
    flags: int
    indptr: np.ndarray                  # int64 [ncells + 1]
    genes: np.ndarray                   # int32 [nnz], absolute, ascending within each row
    n: np.ndarray                       # int32 [nnz]
    e: np.ndarray                       # float32 [nnz]: stored / 32, exactly
    f: np.ndarray | None = None         # int32 [nnz]; None without F
    start_e: np.ndarray | None = None   # float32 [start_nnz]: stored / 32; None without START

    @property
    def nnz(self) -> int:
        return int(self.genes.size)


class CsrArrays(NamedTuple):
    """A cell x gene CSR matrix as three arrays on the caller's array module (numpy or cupy), for either backend to
    wrap: data float32, indices int32 (genes), indptr int32."""
    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]


def count_of(line) -> str:
    """Which count a counted frame is read on, from its counts.jsonl line (the newest of its seq; None or anything
    else: no line): "sampled" (f, the count proseg saves) when proseg counted it for a pause, the hold, a finish or the
    end (SAVED_REASONS) with f; else LIVE_COUNT ("averaged": e), also for init and upscale, which carry f too."""
    if isinstance(line, dict) and line.get("reason") in SAVED_REASONS and line.get("has_f") is True:
        return "sampled"
    return LIVE_COUNT


# ---- reading --------------------------------------------------------------------------------------------------------
def read(path, ncells: int | None = None, ngenes: int | None = None) -> Counts:
    """A counts file, checked whole. A missing file raises FileNotFoundError as it is; anything else wrong with it, or
    a file of another run (`ncells`, `ngenes` when given), raises CountsError saying what."""
    raw = _unzstd(Path(path).read_bytes(), "the counts file")
    if len(raw) < HEADER.size:
        raise CountsError("the counts file is cut short")
    magic, version, cells, gene_count, nnz, flags, scale, iteration, saturated = HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise CountsError("not a proseg counts file")
    if version != VERSION:
        raise CountsError(f"a counts file of version {version}: this app reads version {VERSION}")
    if scale != E_SCALE:
        raise CountsError(f"a counts file keeping e in steps of 1/{scale}: this app reads 1/{E_SCALE}")
    if flags & ~(FLAG_F | FLAG_GENES32 | FLAG_START):
        raise CountsError(f"a counts file with flags this app does not know ({flags})")
    wide = bool(flags & FLAG_GENES32)
    if wide and gene_count <= U16_MAX:
        raise CountsError(f"the counts file keeps its genes as 32-bit numbers for {gene_count} genes")
    if not wide and gene_count > U16_MAX:
        raise CountsError(f"the counts file keeps {gene_count} genes as 16-bit numbers")
    if ncells is not None and cells != ncells:
        raise CountsError(f"the counts file holds {cells} cells, the run {ncells}")
    if ngenes is not None and gene_count != ngenes:
        raise CountsError(f"the counts file holds {gene_count} genes, the run {ngenes}")
    if gene_count > INT32_MAX:
        raise CountsError(f"the counts file holds {gene_count} genes, more than this app reads")
    has_f, has_start = bool(flags & FLAG_F), bool(flags & FLAG_START)
    width = 4 if wide else 2
    end = HEADER.size + 4 * (cells + 1) + nnz * (width + 4 + (2 if has_f else 0))
    start_nnz = 0
    if has_start:
        if len(raw) < end + 4:
            raise CountsError("the counts file is cut short")
        start_nnz = int.from_bytes(raw[end:end + 4], "little")
        end += 4 + 2 * start_nnz
    if len(raw) < end:
        raise CountsError("the counts file is cut short")
    if len(raw) > end:
        raise CountsError(f"the counts file has {len(raw) - end} bytes past its end")

    at = HEADER.size
    indptr = np.frombuffer(raw, "<u4", cells + 1, at).astype(np.int64)
    at += 4 * (cells + 1)
    if indptr[0] != 0:
        raise CountsError("the counts file's rows do not start at 0")
    lengths = np.diff(indptr)
    if (lengths < 0).any():
        raise CountsError("the counts file's rows go backwards")
    if indptr[-1] != nnz:
        raise CountsError(f"the counts file's rows end at {int(indptr[-1])}, not at its {nnz} entries")
    steps = np.frombuffer(raw, "<u4" if wide else "<u2", nnz, at)
    at += width * nnz
    genes = _absolute(steps, indptr[:-1][lengths > 0], gene_count)
    n = np.frombuffer(raw, "<u2", nnz, at).astype(np.int32)
    at += 2 * nnz
    e = np.frombuffer(raw, "<u2", nnz, at).astype(np.float32)
    e *= np.float32(1 / E_SCALE)                                # a power of two: exact
    at += 2 * nnz
    f = start_e = None
    if has_f:
        f = np.frombuffer(raw, "<u2", nnz, at).astype(np.int32)
        at += 2 * nnz
    if has_start:
        start_e = np.frombuffer(raw, "<u2", start_nnz, at + 4).astype(np.float32)
        start_e *= np.float32(1 / E_SCALE)
    return Counts(ncells=cells, ngenes=gene_count, iteration=iteration, saturated=saturated, flags=flags,
                  indptr=indptr, genes=genes, n=n, e=e, f=f, start_e=start_e)


def _absolute(steps: np.ndarray, starts: np.ndarray, gene_count: int) -> np.ndarray:
    """Absolute genes (int32) from the stored steps: each row's first as it is, each later one the one before plus its
    step, which must be above 0 (a row's genes rise). `starts`: the first entry of each row that has one. One cumulative
    sum over the whole file, its running total set back at each row's start: once every row's last gene is checked
    below gene_count, every running total is a gene, so int32 holds it."""
    if not steps.size:
        return np.zeros(0, np.int32)
    flat = steps == 0
    flat[starts] = False                # a row may start at gene 0
    if flat.any():
        raise CountsError("a row of the counts file has its genes out of order")
    if int(steps.max()) >= gene_count:
        raise CountsError(f"the counts file names a gene past its {gene_count}")
    last = np.add.reduceat(steps, starts, dtype=np.int64)       # each row's steps summed: its last gene
    if int(last.max()) >= gene_count:
        raise CountsError(f"the counts file names a gene past its {gene_count}")
    genes = steps.astype(np.int32)
    genes[starts[1:]] -= last[:-1].astype(np.int32)
    np.cumsum(genes, dtype=np.int32, out=genes)
    return genes


def read_counted(path, ncells: int | None = None) -> np.ndarray:
    """A frame's sidecar: float32 [ncells]. FileNotFoundError when it is missing; CountsError when it is malformed or
    holds another number of cells than `ncells` (when given)."""
    raw = _unzstd(Path(path).read_bytes(), "the per-cell sidecar")
    if len(raw) % 4:
        raise CountsError("the per-cell sidecar is cut short")
    values = np.frombuffer(raw, "<f4").astype(np.float32)
    if ncells is not None and values.size != ncells:
        raise CountsError(f"the per-cell sidecar holds {values.size} cells, the run {ncells}")
    return values


# ---- using it -------------------------------------------------------------------------------------------------------
def rows(counts: Counts, cells, which=("e", "n")) -> dict[str, np.ndarray]:
    """A few cells' rows as dense float32 [len(cells), ngenes] arrays, one per count named in `which` ("e", "n", "f";
    zeros outside the cells' genes). `cells` are cell numbers (label - 1), in any order, repeats allowed. CountsError
    for "f" from a frame without it."""
    names = (which,) if isinstance(which, str) else tuple(which)
    for name in names:
        _values(counts, name)
    cells = np.asarray(cells, np.int64).reshape(-1)
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= counts.ncells):
        raise ValueError(f"cells must be cell numbers (label - 1) below {counts.ncells}")
    starts = counts.indptr[cells]
    lengths = counts.indptr[cells + 1] - starts
    row = np.repeat(np.arange(cells.size), lengths)
    entry = np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
    column = counts.genes[entry]
    out = {}
    for name in names:
        dense = np.zeros((cells.size, counts.ngenes), np.float32)
        dense[row, column] = _values(counts, name)[entry]
        out[name] = dense
    return out


def to_csr(xp, counts: Counts, which: str = "e") -> CsrArrays:
    """The cell x gene matrix of one count ("e", "n" or "f"; n and f as float32), as fresh arrays on `xp` (numpy or
    cupy) the caller may change in place. CountsError for "f" from a frame without it."""
    values = _values(counts, which)
    if counts.nnz > INT32_MAX:
        raise CountsError(f"the counts file holds {counts.nnz} entries, more than one matrix takes")
    return CsrArrays(xp.array(np.asarray(values, np.float32)), xp.array(counts.genes),
                     xp.array(counts.indptr.astype(np.int32)), (counts.ncells, counts.ngenes))


def start_csr(xp, init: Counts, counts: Counts) -> CsrArrays:
    """The start block of `counts` as a cell x gene matrix on the rows and genes of the init frame's counts (`init`):
    data = start_e. CountsError when `counts` has no start block, or it does not fit `init` (its cells, genes or
    entries)."""
    if counts.start_e is None:
        raise CountsError("this frame's counts hold no start block")
    if (init.ncells, init.ngenes) != (counts.ncells, counts.ngenes):
        raise CountsError(f"the start's counts hold {init.ncells} cells and {init.ngenes} genes, this frame's "
                          f"{counts.ncells} and {counts.ngenes}")
    if counts.start_e.size != init.nnz:
        raise CountsError(f"the start block holds {counts.start_e.size} entries, the start's counts {init.nnz}")
    return CsrArrays(xp.array(counts.start_e), xp.array(init.genes), xp.array(init.indptr.astype(np.int32)),
                     (init.ncells, init.ngenes))


def counted_of(counts: Counts) -> np.ndarray:
    """Each cell's e summed (float64, then float32): what its sidecar stores."""
    out = np.zeros(counts.ncells, np.float64)
    filled = np.flatnonzero(np.diff(counts.indptr) > 0)
    if filled.size:
        out[filled] = np.add.reduceat(counts.e.astype(np.float64), counts.indptr[filled])
    return out.astype(np.float32)


def _values(counts: Counts, name: str) -> np.ndarray:
    if name not in NAMES:
        raise ValueError(f"no count named {name!r}: e, n or f")
    if name == "f" and counts.f is None:
        raise CountsError("this frame's counts hold no sampled count (f)")
    return {"e": counts.e, "n": counts.n, "f": counts.f}[name]


# ---- writing (the tests' fake proseg and fixtures; proseg-live writes its own) -----------------------------------
def write(path, *, ncells: int, ngenes: int, iteration: int, indptr, genes, n, e, f=None, start_e=None) -> dict:
    """Write a counts file as proseg-live does, atomically (`<path>.tmp`, then renamed). `indptr` [ncells + 1] and
    absolute `genes` (ascending within each row) give the rows; n (> 0) and f (0 to n) are whole numbers, e and
    start_e transcripts (not x 32). ValueError for inputs that do not fit together. Returns what a counts.jsonl line
    says of the file: {nnz, sum_n, sum_e, sum_f, has_f, has_start, saturated, bytes}, the sums of the values as stored
    (sum_e = the stored e / 32 summed in float64, as the sidecar's; sum_f None without f)."""
    ncells, ngenes, iteration = _u32(ncells, "ncells"), _u32(ngenes, "ngenes"), _u32(iteration, "iteration")
    indptr, genes = _whole(indptr, "indptr"), _whole(genes, "genes")
    nnz = genes.size
    if nnz > U32_MAX:
        raise ValueError("more entries than a counts file holds")
    if indptr.size != ncells + 1:
        raise ValueError(f"indptr holds {indptr.size} values for {ncells} cells")
    if indptr[0] != 0 or (np.diff(indptr) < 0).any() or indptr[-1] != nnz:
        raise ValueError(f"indptr must rise from 0 to the {nnz} entries")
    if nnz and (int(genes.min()) < 0 or int(genes.max()) >= ngenes):
        raise ValueError(f"genes must lie in 0 to {ngenes - 1}")
    starts = indptr[:-1][np.diff(indptr) > 0]
    steps = np.diff(genes, prepend=0)
    steps[starts] = genes[starts]
    rising = steps > 0
    rising[starts] = True
    if not rising.all():
        raise ValueError("a row's genes must rise")
    n = _whole(n, "n")
    if n.size != nnz or (nnz and int(n.min()) < 1):
        raise ValueError("n needs one value above 0 per entry: a row holds the genes with n > 0")
    e = _amounts(e, "e")
    if e.size != nnz:
        raise ValueError("e needs one value per entry")
    stored_n, clipped = _clip(n)
    stored_e, more = _clip(np.floor(e * E_SCALE + 0.5))
    clipped += more
    flags = FLAG_GENES32 if ngenes > U16_MAX else 0
    parts = [indptr.astype("<u4").tobytes(), steps.astype("<u4" if flags else "<u2").tobytes(), stored_n.tobytes(),
             stored_e.tobytes()]
    sum_f = None
    if f is not None:
        f = _whole(f, "f")
        if f.size != nnz or (nnz and (int(f.min()) < 0 or (f > n).any())):
            raise ValueError("f needs one value per entry, from 0 to its n")
        stored_f, more = _clip(f)
        clipped += more
        flags |= FLAG_F
        parts.append(stored_f.tobytes())
        sum_f = int(stored_f.sum(dtype=np.int64))
    if start_e is not None:
        start_e = _amounts(start_e, "start_e")
        if start_e.size > U32_MAX:
            raise ValueError("more start entries than a counts file holds")
        stored_start, more = _clip(np.floor(start_e * E_SCALE + 0.5))
        clipped += more
        flags |= FLAG_START
        parts += [struct.pack("<I", stored_start.size), stored_start.tobytes()]
    head = HEADER.pack(MAGIC, VERSION, ncells, ngenes, nnz, flags, E_SCALE, iteration, clipped)
    data = _zstd(head + b"".join(parts))
    _write_atomic(path, data)
    return {"nnz": int(nnz), "sum_n": int(stored_n.sum(dtype=np.int64)),
            "sum_e": float(stored_e.sum(dtype=np.int64)) / E_SCALE, "sum_f": sum_f, "has_f": f is not None,
            "has_start": start_e is not None, "saturated": clipped, "bytes": len(data)}


def write_counted(path, values) -> None:
    """Write a frame's sidecar (float32 per cell; counted_of gives it from a frame's counts), atomically."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("the sidecar holds one value per cell")
    _write_atomic(path, _zstd(np.ascontiguousarray(values, "<f4").tobytes()))


def _u32(value, name: str) -> int:
    try:
        whole = int(value)
    except (TypeError, ValueError, OverflowError):
        whole = None
    if isinstance(value, bool) or whole is None or whole != value or not 0 <= whole <= U32_MAX:
        raise ValueError(f"{name} must be a whole number from 0 to {U32_MAX}")
    return whole


def _whole(values, name: str) -> np.ndarray:
    """A 1-D array of whole numbers as int64 (whole-valued floats too, as numpy's floor gives them)."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    if values.dtype.kind == "f" and np.isfinite(values).all() and (values == np.floor(values)).all():
        return values.astype(np.int64)
    raise ValueError(f"{name} must hold whole numbers")


def _amounts(values, name: str) -> np.ndarray:
    """A 1-D array of finite amounts >= 0, as float64."""
    try:
        values = np.asarray(values, np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must hold numbers") from None
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be a 1-D array of finite amounts, none below 0")
    return values


def _clip(values: np.ndarray) -> tuple[np.ndarray, int]:
    """u16 values as stored (above U16_MAX: U16_MAX) and how many were clipped."""
    return np.minimum(values, U16_MAX).astype("<u2"), int((values > U16_MAX).sum())


# ---- zstd and files ---------------------------------------------------------------------------------------------
def _zstd(data: bytes) -> bytes:
    from numcodecs import Zstd
    return Zstd(level=1).encode(data)


def _unzstd(raw: bytes, what: str) -> bytes:
    """One zstd frame's bytes (with its size in its header or not, as a stream writes it); CountsError when it is
    not one."""
    from numcodecs import Zstd
    try:
        return Zstd().decode(raw)
    except (RuntimeError, ValueError, TypeError):
        raise CountsError(f"{what} is not a whole zstd frame") from None


def _write_atomic(path, data: bytes) -> None:
    temporary = f"{path}.tmp"
    Path(temporary).write_bytes(data)
    os.replace(temporary, path)
