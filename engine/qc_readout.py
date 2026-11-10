"""Test a setting here: what each test run did with the targets, read from proseg's own per-transcript file
(output/transcript-metadata.parquet, which a test run writes as it ends plainly), in memory.

The targets and controls (`Items`) are fixed before any run, on the test's shared input (engine/qc_pool.py); every run
reads that same input, so an item is found in each run's file by its transcript_id, its gene and its observed x and y
bits checked there too. Per item and run: where proseg left it at the run's last frame, as the share of it still in its
own cell (HELD), with its giver (GIVER), in any other cell (OTHER) or in no cell (BG), from proseg's drift tables (the
offset each transcript was moved by, at its observed voxel) checked against the assignment proseg wrote (`checks`). An
item that cannot be read in a run (not found, its gene or place not the input's, its cell missing from the run) says so
in `status`, and engine/qc_rule.py leaves it out of every run of the test.

How the shares are worked out (the measurements' own readout, ported; only what the test uses is kept):
  - every row k of the file: its observed voxel v_k on the last frame's grid (the GPU worker's grid_columns, and the
    depth layer as its voxel_lookup takes it, in float32 as proseg's), its offset d_k (proseg writes x = observed_x +
    di * voxel), and the cell owning its destination v_k + d_k, added in (i, j, k): a destination outside the grid is no
    cell's, as proseg's own lookup says;
  - per (cell, gene): n, the rows whose destination the cell owns; f, the rows proseg counts as the cell's (assigned, not
    background); q = f / n. Proseg's destinations and its per-cell counts are exact; which bucket-mate (the rows of a
    gene at one observed voxel) went where, and which of a cell's rows it calls background, are not, so each is shared
    out evenly: HELD = d_c0 * q(c0, g), GIVER = d_c1 * q(c1, g), OTHER = the same over every other cell, BG = the rest,
    d_o being the share of the item's bucket whose destination o owns;
  - two checks over every (cell, gene) with n or f: the identity (summed over the gene's rows, d_o * q(o, g) gives
    f(o, g) back) and f <= n, each on its own. Wherever n > 0 the identity gives f back whatever f is: it fails only
    where a cell is counted rows of a gene that no row of that gene reaches (n = 0 < f, so f > n too), and f > n > 0
    passes it, with a q above 1 and so a share above 1. A run where either fails is not read (ReadoutError): its file
    and its last frame do not describe the same assignment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from ..analysis.worker import grid_columns, read_centres
from .deeper_where import _decode
from .qc_rule import inside_inner

# An item's status in one run, in precedence order (the first that holds).
OK, NOT_FOUND, C0_ABSENT, C1_ABSENT, CROSSCHECK_FAILED = 0, 1, 2, 3, 4
STATUS_NAMES = ("ok", "not_found", "c0_absent", "c1_absent", "crosscheck_failed")
SUSPECT, MATCHED, UNIFORM = 0, 1, 2   # an item's kind: a target, one of its matched controls, a cell's own RNA
NO_GIVER = -1
METADATA = "transcript-metadata.parquet"          # in <run dir>/output unless read_run is given metadata_path
IDENTITY_TOL = 1e-6                               # the drift tables against proseg's assignment, at most this apart
READ_WORKERS = 4                                  # runs read at once
# The columns of proseg's per-transcript file the readout needs (qv and the rest are not read).
COLUMNS = ("transcript_id", "x", "y", "z", "observed_x", "observed_y", "observed_z", "gene", "assignment",
           "background")
FLOATS = ("x", "y", "z", "observed_x", "observed_y", "observed_z")


@dataclass(frozen=True, eq=False)
class Items:
    """One row per target or control, built by engine/qc_pool.py once the targets are matched on the shared input."""
    kind: np.ndarray            # int8[n]: SUSPECT | MATCHED | UNIFORM
    cell: np.ndarray            # int32[n]: the item's cell, an index into cell_names
    giver: np.ndarray           # int32[n]: the neighbour it came from, an index into cell_names; NO_GIVER for UNIFORM
    target: np.ndarray          # int64[n]: a MATCHED item's suspect (its item index); -1 otherwise
    gene: np.ndarray            # int32[n]: an index into gene_names
    transcript_id: np.ndarray   # uint64[n]: the shared input's id (the join key with each run's metadata)
    x: np.ndarray               # float32[n]: the transcript's x as the input holds it (its bits are observed_x's)
    y: np.ndarray               # float32[n]
    cell_names: tuple[str, ...] # Xenium ids (the parent's cells.txt names)
    gene_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.kind)

    def check(self) -> None:
        """ValueError unless every array holds one value per item, of its kind of number (x and y float32, the ids
        uint64: their bits are compared), every kind is one of the three, every index lies in its list (a UNIFORM item
        has no giver and no item a target but a MATCHED one), and a MATCHED item's target is a SUSPECT of its cell."""
        n = len(self.kind)
        arrays = {"kind": self.kind, "cell": self.cell, "giver": self.giver, "target": self.target, "gene": self.gene,
                  "transcript_id": self.transcript_id, "x": self.x, "y": self.y}
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray) or array.ndim != 1 or len(array) != n:
                raise ValueError(f"Items.{name} must have one value per item ({n})")
        for name in ("kind", "cell", "giver", "target", "gene"):
            if arrays[name].dtype.kind not in "iu":
                raise ValueError(f"Items.{name} must be whole numbers")
        if self.transcript_id.dtype != np.uint64 or self.x.dtype != np.float32 or self.y.dtype != np.float32:
            raise ValueError("Items: transcript_id must be uint64, x and y float32")
        if not all(isinstance(name, str) for name in (*self.cell_names, *self.gene_names)):
            raise ValueError("Items: cell and gene names must be text")
        kind = self.kind.astype(np.int64)
        if not np.isin(kind, (SUSPECT, MATCHED, UNIFORM)).all():
            raise ValueError("Items.kind must be SUSPECT, MATCHED or UNIFORM")
        ncells, ngenes = len(self.cell_names), len(self.gene_names)
        cell, giver, gene = (a.astype(np.int64) for a in (self.cell, self.giver, self.gene))
        target = self.target.astype(np.int64)
        if ((cell < 0) | (cell >= ncells)).any() or ((gene < 0) | (gene >= ngenes)).any():
            raise ValueError("Items: a cell or gene index outside its names")
        uniform = kind == UNIFORM
        if (giver[uniform] != NO_GIVER).any():
            raise ValueError("Items: a UNIFORM item has no giver")
        if ((giver[~uniform] < 0) | (giver[~uniform] >= ncells)).any():
            raise ValueError("Items: a giver index outside the cell names")
        matched = kind == MATCHED
        if (target[~matched] != -1).any():
            raise ValueError("Items: only a MATCHED item has a target")
        to = target[matched]
        if ((to < 0) | (to >= n)).any():
            raise ValueError("Items: a MATCHED item's target is no item")
        if (kind[to] != SUSPECT).any() or (cell[to] != cell[matched]).any():
            raise ValueError("Items: a MATCHED item's target is no SUSPECT of its cell")


@dataclass(eq=False)
class RunReadout:
    """One test run, read: per item (Items order) its status and its four shares, which add up to 1 where computed."""
    status: np.ndarray          # int8[n]: OK, NOT_FOUND, ...
    held: np.ndarray            # float64[n]: still in its own cell; NaN unless found and its cell is in this run
    giver: np.ndarray           # float64[n]: with its giver (0 when the giver is not in this run, or it has none)
    other: np.ndarray           # float64[n]: in any other cell
    bg: np.ndarray              # float64[n]: in no cell
    tx_inner: float | None      # the median foreground transcripts of the cells whose last centre lies in the inner area
                                # (None without the frame's cell statistics)
    cells_inner: int            # how many cells that median is over
    iter: int                   # the run's last frame
    seq: int
    rows: int                   # transcripts in its metadata
    ids: str | None             # sha256 of its transcript_id column (u64 little-endian, row order); None with any null
    seconds: float
    checks: dict                # {identity_pairs, identity_max_err, f_gt_n, dest_vs_assignment, rows_dest_outside,
                                #  offset_residual, bg_gap, crosscheck: {gene, x, y}, duplicate_cells}


class ReadoutError(Exception):
    """A run that cannot be read as the test needs, in plain words: shown as "The test could not finish: <it>"."""


# ---- a run's files ------------------------------------------------------------------------------------------------------
def _lines(path: Path) -> list[str]:
    """A text file's lines, without the final empty one."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _last_frame(live: Path) -> dict:
    """The last record of frames.jsonl: the frame the run ended on."""
    try:
        records = [line for line in (live / "frames.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    except OSError:
        records = []
    if not records:
        raise ReadoutError("a test run has no frame")
    try:
        frame = json.loads(records[-1])
    except json.JSONDecodeError:
        raise ReadoutError("a test run's last frame cannot be read") from None
    if not isinstance(frame, dict):
        raise ReadoutError("a test run's last frame cannot be read")
    return frame


def _layers(live: Path, frame: dict) -> np.ndarray:
    """A whole-area frame's per-layer labels [nz, ny, nx] (after its two planes), decoded as Go deeper decodes them. A
    tiled frame is refused: the test reads whole-area runs only."""
    if frame.get("format") == "tiles" or not str(frame.get("file", "")).endswith((".bin.zst", ".bin.gz")):
        raise ReadoutError("a test run ended on a tiled frame")
    try:
        nx, ny, nz = int(frame["nx"]), int(frame["ny"]), int(frame["nz"])
        values = _decode(live / frame["file"])
    except Exception:          # a missing, cut or undecodable file: any error of the decoders means the same here
        raise ReadoutError("a test run's last frame cannot be read") from None
    plane = nx * ny
    if values.size < (2 + nz) * plane:
        raise ReadoutError("a test run's last frame is incomplete")
    return values[2 * plane:(2 + nz) * plane].reshape(nz, ny, nx)


def _gene_indices(column, genes: list[str]) -> np.ndarray:
    """Each row's gene as its index in the run's genes.txt (their order changes between runs), dictionary columns
    (proseg's) and plain text alike; -1 for a name the run does not have."""
    import pyarrow as pa
    import pyarrow.compute as pc
    value_set = pa.array(genes, type=pa.string())
    if pa.types.is_dictionary(column.type):
        parts = []
        for chunk in column.chunks:
            dictionary = chunk.dictionary
            if not pa.types.is_string(dictionary.type):
                dictionary = dictionary.cast(pa.string())
            mapping = pc.fill_null(pc.index_in(dictionary, value_set=value_set), -1).to_numpy().astype(np.int64)
            if chunk.indices.null_count:
                raise ReadoutError("a test run's file has transcripts without a gene")
            parts.append(mapping[chunk.indices.to_numpy().astype(np.int64)])
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)
    if not pa.types.is_string(column.type):
        column = column.cast(pa.string())
    if column.null_count:
        raise ReadoutError("a test run's file has transcripts without a gene")
    return pc.fill_null(pc.index_in(column, value_set=value_set), -1).to_numpy().astype(np.int64)


def _metadata(path: Path, genes: list[str]) -> dict:
    """Proseg's per-transcript file as numpy columns: ids (uint64; `valid` false where null), the six places (float32,
    as proseg writes them: their bits are compared), gene (the run's index), assignment (-1 where null), background."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    if not path.is_file():
        raise ReadoutError("a test run wrote no file of its transcripts")
    try:
        table = pq.read_table(path, columns=list(COLUMNS), use_threads=False)
    except (OSError, ValueError, pa.ArrowException):
        raise ReadoutError("a test run's file of its transcripts cannot be read") from None
    ids = table["transcript_id"]
    if not pa.types.is_integer(ids.type):
        raise ReadoutError("a test run's file of its transcripts is not as proseg writes it")
    out = {"valid": pc.is_valid(ids).to_numpy(zero_copy_only=False).astype(bool), "nulls": int(ids.null_count)}
    try:
        out["ids"] = pc.fill_null(ids.cast(pa.uint64()), 0).to_numpy().astype(np.uint64, copy=False)
    except pa.ArrowInvalid:                        # a negative id: no transcript of the input
        raise ReadoutError("a test run's file of its transcripts is not as proseg writes it") from None
    for name in FLOATS:
        column = table[name]
        if column.type != pa.float32() or column.null_count:
            raise ReadoutError("a test run's file of its transcripts is not as proseg writes it")
        out[name] = column.to_numpy()
    out["assignment"] = pc.fill_null(table["assignment"].cast(pa.int64()), -1).to_numpy().astype(np.int64)
    background = table["background"]
    if not pa.types.is_boolean(background.type) or background.null_count:
        raise ReadoutError("a test run's file of its transcripts is not as proseg writes it")
    out["background"] = background.to_numpy(zero_copy_only=False).astype(bool)
    out["gene"] = _gene_indices(table["gene"], genes)
    if (out["gene"] < 0).any():
        raise ReadoutError("a test run's file names a gene the run does not have")
    return out


# ---- the geometry -------------------------------------------------------------------------------------------------------
def observed_voxels(xs, ys, zs, frame: dict, z_origin: float, z_spacing: float):
    """(i, j, k) of each observed place on the frame's grid: i, j by the worker's grid_columns, k clipped to the layers
    as its voxel_lookup does (proseg's own rule), in float32 arithmetic as proseg's (Python-float origins keep numpy in
    float32). int64 i, j, k, and whether (i, j) lies on the grid."""
    nx, ny, nz = int(frame["nx"]), int(frame["ny"]), int(frame["nz"])
    i, j, inside = grid_columns(np, xs, ys, float(frame["x0"]), float(frame["y0"]), float(frame["voxel"]), nx, ny)
    k = np.clip(np.floor((zs - float(z_origin)) / float(z_spacing)), 0, nz - 1).astype(np.int64)
    return i.astype(np.int64), j.astype(np.int64), k, inside


def voxel_offsets(md: dict, voxel: float, z_spacing: float):
    """Each row's offset in voxels, (round((x - observed_x) / voxel), the same for y, and for z over the layers' spacing):
    proseg writes x = observed_x + di * voxel. Also how far the quotients lie from a whole number at most."""
    out, residual = [], 0.0
    for a, b, step in (("x", "observed_x", voxel), ("y", "observed_y", voxel), ("z", "observed_z", z_spacing)):
        quotient = (md[a].astype(np.float64) - md[b].astype(np.float64)) / float(step)
        rounded = np.rint(quotient)
        if quotient.size:
            residual = max(residual, float(np.abs(quotient - rounded).max()))
        out.append(rounded.astype(np.int64))
    return out[0], out[1], out[2], residual


def destination_owners(layers: np.ndarray, i, j, k, di, dj, dk):
    """The cell (label - 1) owning each destination (i + di, j + dj, k + dk), added in (i, j, k), never as flat indices;
    -1 (no cell) for label 0 and for a destination outside the grid, where proseg's lookup gives no cell either.
    Returns (owner, inside)."""
    nz, ny, nx = layers.shape
    ui, uj, uk = i + di, j + dj, k + dk
    inside = (ui >= 0) & (ui < nx) & (uj >= 0) & (uj < ny) & (uk >= 0) & (uk < nz)
    owner = np.full(ui.shape, -1, np.int64)
    owner[inside] = layers[uk[inside], uj[inside], ui[inside]].astype(np.int64) - 1
    return owner, inside


def _expand(starts: np.ndarray, ends: np.ndarray):
    """(item of each entry, position) for the ranges [starts, ends) laid end to end."""
    lengths = ends - starts
    item = np.repeat(np.arange(starts.size), lengths)
    first = np.cumsum(lengths) - lengths
    position = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(first, lengths) + np.repeat(starts, lengths)
    return item, position


def _lookup(keys: np.ndarray, values: np.ndarray, wanted: np.ndarray, default=0):
    """values at the `wanted` keys of a sorted key array (default where absent)."""
    out = np.full(wanted.shape, default, dtype=values.dtype if values.size else np.int64)
    if not keys.size or not wanted.size:
        return out
    at = np.clip(np.searchsorted(keys, wanted), 0, keys.size - 1)
    hit = keys[at] == wanted
    out[hit] = values[at[hit]]
    return out


class _Buckets:
    """The rows grouped by (gene, observed voxel), each bucket with the cells its rows' destinations belong to (its
    pairs: pair_owner, -1 for no cell) and the share of its rows each holds (pair_w)."""

    def __init__(self, gene, i, j, k, owner, ncells: int):
        imin, jmin = (int(i.min()), int(j.min())) if i.size else (0, 0)
        width, height = (int(i.max()) - imin + 1, int(j.max()) - jmin + 1) if i.size else (1, 1)
        depth = int(k.max()) + 1 if k.size else 1
        key = ((gene * depth + k) * height + (j - jmin)) * width + (i - imin)
        _, first, self.of_row, self.size = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        self.of_row = self.of_row.reshape(-1)
        self.gene = gene[first]
        stride = ncells + 1
        pair_keys, self.pair_count = np.unique(self.of_row.astype(np.int64) * stride + (owner + 1), return_counts=True)
        self.pair_bucket = pair_keys // stride
        self.pair_owner = pair_keys % stride - 1
        self.pair_w = self.pair_count / self.size[self.pair_bucket]
        nb = self.size.size
        self.start = np.searchsorted(self.pair_bucket, np.arange(nb), "left")
        self.end = np.searchsorted(self.pair_bucket, np.arange(nb), "right")


class _Tables:
    """Per (cell, gene), keyed cell * ngenes + gene: n (rows whose destination the cell owns) and f (rows proseg counts
    as the cell's); each bucket pair's q = f / n (0 where n = 0); and over every pair with n or f, the identity's
    largest error and how many pairs have f > n."""

    def __init__(self, gene, i, j, k, owner, assign, background, ncells: int, ngenes: int):
        foreground = (~background) & (assign >= 0)
        dest = owner >= 0
        n_keys, n_cnt = np.unique(owner[dest] * ngenes + gene[dest], return_counts=True)
        f_keys, f_cnt = np.unique(assign[foreground] * ngenes + gene[foreground], return_counts=True)
        self.buckets = b = _Buckets(gene, i, j, k, owner, ncells)
        placed = b.pair_owner >= 0
        p_key = np.where(placed, b.pair_owner * ngenes + b.gene[b.pair_bucket], -1)
        p_n = _lookup(n_keys, n_cnt, p_key)
        p_f = _lookup(f_keys, f_cnt, p_key)
        self.p_q = np.where(placed & (p_n > 0), p_f / np.maximum(p_n, 1), 0.0)
        # For every (cell, gene): a bucket adds |B| rows each holding d_a(B), so its pair adds |B| * d_a(B) * q(a, g);
        # summed over the gene's buckets that is n(a, g) * q(a, g), which is f(a, g) wherever n > 0, whatever f is. So
        # the identity fails only where n = 0 < f, and f <= n is a check of its own (f_gt_n): f > n > 0 passes it.
        support = np.union1d(n_keys, f_keys)
        lhs = np.zeros(support.size, np.float64)
        if placed.any():
            lhs = np.bincount(np.searchsorted(support, p_key[placed]),
                              weights=(b.size[b.pair_bucket] * b.pair_w * self.p_q)[placed], minlength=support.size)
        f_at = _lookup(f_keys, f_cnt, support)
        n_at = _lookup(n_keys, n_cnt, support)
        self.pairs = int(support.size)
        self.max_err = float(np.abs(lhs - f_at).max(initial=0.0))
        self.f_gt_n = int((f_at > n_at).sum())


# ---- one run ------------------------------------------------------------------------------------------------------------
def read_run(run_dir: Path, items: Items, inner, *, metadata_path: Path | None = None) -> RunReadout:
    """One test run's readout of `items`, its cells' transcripts counted inside `inner` [x0, y0, x1, y1] µm. The
    metadata is `metadata_path`, else run_dir/output/METADATA. ReadoutError for a run that cannot be read: a file
    missing or not as proseg writes it, a tiled last frame, or the identity or f <= n failing."""
    began = time.perf_counter()
    items.check()
    run_dir = Path(run_dir)
    live = run_dir / "live"
    try:
        info = json.loads((live / "info.json").read_text())
        ncells = int(info["cells"])
        names = _lines(live / "cells.txt")[:ncells]
        genes = _lines(live / "genes.txt")
    except (OSError, ValueError, KeyError, TypeError):
        raise ReadoutError("a test run's own files cannot be read") from None
    if len(names) != ncells:
        raise ReadoutError("a test run's cells cannot be read")
    final = _last_frame(live)
    z = info.get("z") or {}
    z_origin, z_spacing = float(z.get("grid_origin", 0.0)), float(z.get("grid_spacing", 1.0))
    layers = _layers(live, final)
    if layers.size and int(layers.max()) > ncells:
        raise ReadoutError("a test run's last frame names more cells than the run has")
    md = _metadata(Path(metadata_path) if metadata_path is not None else run_dir / "output" / METADATA, genes)
    nrows = int(md["ids"].size)
    gene, assign, background = md["gene"], md["assignment"], md["background"]
    if int(assign.max(initial=-1)) >= ncells:
        raise ReadoutError("a test run's file names more cells than the run has")

    # Every row: its observed voxel, its offset and the cell owning its destination.
    try:
        voxel = float(final["voxel"])
        i, j, k, _ = observed_voxels(md["observed_x"], md["observed_y"], md["observed_z"], final, z_origin, z_spacing)
    except (KeyError, TypeError, ValueError):
        raise ReadoutError("a test run's last frame cannot be read") from None
    di, dj, dk, residual = voxel_offsets(md, voxel, z_spacing)
    owner, dest_inside = destination_owners(layers, i, j, k, di, dj, dk)
    foreground = (~background) & (assign >= 0)

    # Per (cell, gene): n, f and q; the identity and f <= n over every pair (each checked on its own).
    tables = _Tables(gene, i, j, k, owner, assign, background, ncells, len(genes))
    if not tables.max_err <= IDENTITY_TOL or tables.f_gt_n:
        raise ReadoutError("a test run's counts do not match where its file puts its transcripts")
    buckets, p_q = tables.buckets, tables.p_q

    # The items: the row holding each one's transcript_id, checked against the input's gene and place.
    count = len(items)
    rows_valid = np.flatnonzero(md["valid"])
    order = np.argsort(md["ids"][rows_valid], kind="stable")
    sorted_ids, sorted_rows = md["ids"][rows_valid][order], rows_valid[order]
    tid = np.asarray(items.transcript_id, np.uint64)
    found = np.zeros(count, bool)
    row = np.full(count, -1, np.int64)
    if sorted_ids.size and count:
        at = np.clip(np.searchsorted(sorted_ids, tid), 0, sorted_ids.size - 1)
        found = sorted_ids[at] == tid
        row = np.where(found, sorted_rows[at], -1)
    gene_index: dict[str, int] = {}
    for g, name in enumerate(genes):
        gene_index.setdefault(name, g)
    run_gene = np.array([gene_index.get(name, -1) for name in items.gene_names], np.int64)
    fr = np.flatnonzero(found)
    r = row[fr]
    bad_gene = gene[r] != run_gene[items.gene[fr].astype(np.int64)]
    bad_x = md["observed_x"][r].view(np.uint32) != np.ascontiguousarray(items.x[fr]).view(np.uint32)
    bad_y = md["observed_y"][r].view(np.uint32) != np.ascontiguousarray(items.y[fr]).view(np.uint32)
    cross_bad = np.zeros(count, bool)
    cross_bad[fr] = bad_gene | bad_x | bad_y

    # Their cells in this run (the first line of a name: cells are never matched by index across runs); -1 for a cell
    # the run does not have, -2 for no giver.
    cell_index: dict[str, int] = {}
    for c, name in enumerate(names):
        cell_index.setdefault(name, c)
    in_run = np.array([cell_index.get(name, -1) for name in items.cell_names], np.int64)
    c0 = in_run[items.cell.astype(np.int64)]
    giver_of = items.giver.astype(np.int64)
    has_giver = giver_of >= 0
    c1 = np.full(count, -2, np.int64)
    c1[has_giver] = in_run[giver_of[has_giver]]
    status = np.full(count, OK, np.int8)
    status[has_giver & (c1 < 0)] = C1_ABSENT
    status[c0 < 0] = C0_ABSENT
    status[cross_bad] = CROSSCHECK_FAILED
    status[~found] = NOT_FOUND

    # The shares, wherever the row is found and the cell is in this run (also for an item whose check failed or whose
    # giver is missing: its GIVER is then 0 and its giver's share falls into OTHER).
    held, giver, other, bg = (np.full(count, np.nan) for _ in range(4))
    bg_gap = 0.0
    done = np.flatnonzero(found & (c0 >= 0))
    if done.size:
        b = buckets.of_row[row[done]]
        item, position = _expand(buckets.start[b], buckets.end[b])
        o, w, q = buckets.pair_owner[position], buckets.pair_w[position], p_q[position]
        own, gave = c0[done][item], c1[done][item]
        is_c0 = o == own
        is_c1 = (gave >= 0) & (o == gave) & ~is_c0
        is_none = o < 0
        is_other = ~(is_c0 | is_c1 | is_none)
        sums = {name: np.bincount(item, weights=w * q * mask, minlength=done.size)
                for name, mask in (("held", is_c0), ("giver", is_c1), ("other", is_other))}
        held[done], giver[done], other[done] = sums["held"], sums["giver"], sums["other"]
        bg[done] = 1.0 - sums["held"] - sums["giver"] - sums["other"]
        parts = (np.bincount(item, weights=w * (1 - q) * ~is_none, minlength=done.size)
                 + np.bincount(item, weights=w * is_none, minlength=done.size))
        bg_gap = float(np.abs(bg[done] - parts).max(initial=0.0))

    # Transcripts per cell: the median of the cells whose last centre lies in the inner area.
    tx_inner, cells_inner = None, 0
    if final.get("cells_file"):
        try:
            centres = read_centres(live / final["cells_file"], ncells)
        except ValueError:
            raise ReadoutError("a test run's last frame cannot be read") from None
        if centres is not None:
            inside = inside_inner(centres["cx"], centres["cy"], inner)
            per_cell = np.bincount(assign[foreground], minlength=ncells)
            cells_inner = int(inside.sum())
            tx_inner = float(np.median(per_cell[inside])) if cells_inner else None

    ids = None if md["nulls"] else hashlib.sha256(md["ids"].astype("<u8").tobytes()).hexdigest()
    checks = {"identity_pairs": tables.pairs, "identity_max_err": tables.max_err, "f_gt_n": tables.f_gt_n,
              "dest_vs_assignment": int((foreground & (owner != assign)).sum()),
              "rows_dest_outside": int((~dest_inside).sum()), "offset_residual": residual, "bg_gap": bg_gap,
              "crosscheck": {"gene": int(bad_gene.sum()), "x": int(bad_x.sum()), "y": int(bad_y.sum())},
              "duplicate_cells": len(names) - len(cell_index)}
    return RunReadout(status=status, held=held, giver=giver, other=other, bg=bg, tx_inner=tx_inner,
                      cells_inner=cells_inner, iter=int(final.get("iter", -1)), seq=int(final.get("seq", -1)),
                      rows=nrows, ids=ids, seconds=time.perf_counter() - began, checks=checks)


def read_runs(run_dirs: Mapping[str, Path], items: Items, inner, *, workers: int = READ_WORKERS
              ) -> dict[str, RunReadout]:
    """Every test run read (`workers` at once): {role: RunReadout}. ReadoutError (the first in role order, once every
    read ended) for a run that cannot be read, or when the runs did not read the same transcripts or end at the same
    iteration."""
    roles = list(run_dirs)
    if not roles:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(roles))),
                            thread_name_prefix="qc-test-read") as pool:
        futures = {role: pool.submit(read_run, run_dirs[role], items, inner) for role in roles}
    for role in roles:                            # every read has ended here: the first failure, in role order
        error = futures[role].exception()
        if error is not None:
            raise error
    out = {role: futures[role].result() for role in roles}
    first = out[roles[0]]
    ids = {readout.ids for readout in out.values() if readout.ids is not None}
    if any(readout.rows != first.rows for readout in out.values()) or len(ids) > 1:
        raise ReadoutError("the test runs did not read the same transcripts")
    if any(readout.iter != first.iter for readout in out.values()):
        raise ReadoutError("the test runs did not end at the same iteration")
    return out
