"""Proseg's own counts on disk for tests: the files proseg-live writes with --live-counts (native/proseg/src/live.rs),
made only through the app's own readers' twins (analysis/counts_file.py write/write_counted, engine/held.py
write_where), never byte by byte here. tests/fake_proseg.py writes every counted frame and where export with these;
other roles' tests import them (read only for them).

counts_of / counts_from_labels   each cell's count per gene (CSR, cell = label - 1): n, the transcripts it holds; e =
                                 n x keep, the averaged count when proseg keeps that share of every gene; f = floor(n x
                                 keep), a sampled count (f <= n; 0 where the draw kept none)
write_frame_counts               one counted frame: NNNNNN.counts.zst (PLC1) in live/frames or the spool, its per-cell
                                 sidecar frames/NNNNNN.counted.zst, then its counts.jsonl line (the commit point)
write_where_export               one where export: where-N.bin.zst, then where.json (its commit point)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REASONS = ("init", "upscale", "request", "every", "pause", "hold", "finish", "end")   # a counts line's "reason"
LINE_KEYS = ("seq", "iter", "file", "counted_file", "reason", "token", "where", "nnz", "sum_n", "sum_e", "sum_f",
             "has_f", "has_start", "saturated", "bytes", "ms")
WHERE_KEYS = ("token", "seq", "iter", "box", "margin_um", "x0", "y0", "voxel", "nx", "ny", "nz", "i0", "j0", "i1", "j1",
              "file", "entries")
MARGIN_UM = 24.0              # max(20 µm, 6 x proseg's default diffusion sigma far, 4 µm) (main.rs:42-43)


def counts_of(owner, gene, ncells: int, ngenes: int, keep: float = 1.0) -> dict:
    """Each cell's count per gene from each transcript's owner (its label: 0 none, else cell + 1; labels above ncells
    are no cell's) and gene: {"indptr" [ncells + 1], "genes" (ascending within each row), "n", "e", "f"} (int64, and e
    float64), the pattern being the (cell, gene) pairs with n > 0."""
    owner = np.asarray(owner, np.int64).reshape(-1)
    gene = np.asarray(gene, np.int64).reshape(-1)
    if owner.shape != gene.shape:
        raise ValueError(f"{owner.size} owners for {gene.size} genes")
    if gene.size and (gene.min() < 0 or gene.max() >= ngenes):
        raise ValueError(f"a gene outside the run's {ngenes}")
    if not 0 < keep <= 1:
        raise ValueError(f"keep {keep} is not a share in (0, 1]")
    held = (owner > 0) & (owner <= ncells)
    keys, n = np.unique((owner[held] - 1) * ngenes + gene[held], return_counts=True)
    indptr = np.zeros(ncells + 1, np.int64)
    np.cumsum(np.bincount(keys // max(ngenes, 1), minlength=ncells), out=indptr[1:])
    n = n.astype(np.int64)
    return {"indptr": indptr, "genes": (keys % max(ngenes, 1)).astype(np.int64), "n": n, "e": n * float(keep),
            "f": np.floor(n * float(keep)).astype(np.int64)}


def observed_voxels(records, grid: dict, z_origin: float, z_spacing: float):
    """Each record's observed voxel as the app reads it: (i, j) by the worker's grid rule (analysis/worker.py
    grid_columns: Python-float origins, so float32 arithmetic as proseg's), k by the layer rule clip(floor((z -
    z_origin) / z_spacing), 0, nz - 1); and whether (i, j) is on the grid. (i, j, k: int64; inside: bool)."""
    from sopa_companion.analysis.worker import grid_columns
    i, j, inside = grid_columns(np, np.asarray(records["x"]), np.asarray(records["y"]), float(grid["x0"]),
                                float(grid["y0"]), float(grid["voxel"]), int(grid["nx"]), int(grid["ny"]))
    k = np.clip(np.floor((np.asarray(records["z"]) - float(z_origin)) / float(z_spacing)), 0, int(grid["nz"]) - 1)
    return i.astype(np.int64), j.astype(np.int64), k.astype(np.int64), inside


def counts_from_labels(records, labels, grid: dict, z_origin: float, z_spacing: float, ncells: int, ngenes: int,
                       keep: float = 1.0) -> dict:
    """counts_of with each record (transcript_index RECORD: x, y, z, gene) owned by the label of its observed voxel in
    `labels` (flat u32 [nz * ny * nx], per layer; 0 off the grid)."""
    i, j, k, inside = observed_voxels(records, grid, z_origin, z_spacing)
    nx, ny, nz = (int(grid[key]) for key in ("nx", "ny", "nz"))
    labels = np.asarray(labels).reshape(-1)
    if labels.size != nx * ny * nz:
        raise ValueError(f"{labels.size} labels for a grid of {nx} x {ny} x {nz}")
    owner = np.zeros(len(i), np.int64)
    owner[inside] = labels[(k[inside] * ny + j[inside]) * nx + i[inside]]
    return counts_of(owner, np.asarray(records["gene"]), ncells, ngenes, keep)


def write_frame_counts(live, seq: int, iteration: int, *, ncells: int, ngenes: int, indptr, genes, n, e, f=None,
                       start_e=None, reason: str = "request", token=None, where=None, spool=None) -> dict:
    """One counted frame as proseg-live's counts writer leaves it: the PLC1 file (live/frames/NNNNNN.counts.zst, or in
    the spool folder `spool` when given: then the line names its absolute path), the sidecar frames/NNNNNN.counted.zst
    (each cell's Σ of the stored e: counts_file.counted_of of the file read back), then one line appended to
    live/counts.jsonl, with every key (ms 0.0; where = a W or None). Returns that line."""
    from sopa_companion.analysis import counts_file
    live = Path(live)
    name = f"{int(seq):06d}.counts.zst"
    if spool is None:
        path, file = live / "frames" / name, f"frames/{name}"
    else:
        path = Path(os.path.abspath(spool)) / name
        file = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = counts_file.write(path, ncells=ncells, ngenes=ngenes, iteration=iteration, indptr=indptr, genes=genes, n=n,
                              e=e, f=f, start_e=start_e)
    counted = f"frames/{int(seq):06d}.counted.zst"
    (live / "frames").mkdir(parents=True, exist_ok=True)
    counts_file.write_counted(live / counted, counts_file.counted_of(counts_file.read(path)))
    line = {"seq": int(seq), "iter": int(iteration), "file": file, "counted_file": counted, "reason": reason,
            "token": None if token is None else int(token), "where": where, "nnz": int(stats["nnz"]),
            "sum_n": int(stats["sum_n"]), "sum_e": float(stats["sum_e"]),
            "sum_f": None if stats["sum_f"] is None else int(stats["sum_f"]), "has_f": bool(stats["has_f"]),
            "has_start": bool(stats["has_start"]), "saturated": int(stats["saturated"]), "bytes": int(stats["bytes"]),
            "ms": {"extract": 0.0, "write": 0.0}}
    with open(live / "counts.jsonl", "a") as stream:
        stream.write(json.dumps(line) + "\n")
    return line


def where_span(box, grid: dict) -> tuple[int, int, int, int]:
    """The observed voxels a where export of `box` (µm) covers, (i0, j0, i1, j1), in float32 arithmetic as proseg's:
    i0 = max(0, floor((box x0 - x0) / voxel)), i1 = min(nx - 1, floor((box x1 - x0) / voxel)), j likewise."""
    bx0, by0, bx1, by1 = (np.float32(v) for v in box)
    x0, y0, voxel = (np.float32(grid[key]) for key in ("x0", "y0", "voxel"))
    nx, ny = int(grid["nx"]), int(grid["ny"])
    return (max(0, int(np.floor((bx0 - x0) / voxel))), max(0, int(np.floor((by0 - y0) / voxel))),
            min(nx - 1, int(np.floor((bx1 - x0) / voxel))), min(ny - 1, int(np.floor((by1 - y0) / voxel))))


def write_where_export(live, token: int, seq: int, iteration: int, box, grid: dict, entries: dict,
                       margin_um: float = MARGIN_UM) -> dict:
    """One where export as proseg-live leaves it: where-<token>.bin.zst (engine/held.py write_where) of `entries`
    ({"i", "j", "k", "gene", "dest", "n"}, written as given), then where.json, its commit point: W, every key, the box
    echoed as proseg parses it (float32, shortest), the span by where_span. Returns W (no "path"). Proseg's own
    pruning (the 4 newest kept) is the fake's."""
    from sopa_companion.engine import held
    live = Path(live)
    i0, j0, i1, j1 = where_span(box, grid)
    name = f"where-{int(token)}.bin.zst"
    arrays = [np.asarray(entries[key]) for key in ("i", "j", "k", "gene", "dest", "n")]
    held.write_where(live / name, *arrays)
    meta = {"token": int(token), "seq": int(seq), "iter": int(iteration),
            "box": [float(str(np.float32(v))) for v in box], "margin_um": float(margin_um),
            "x0": float(grid["x0"]), "y0": float(grid["y0"]), "voxel": float(grid["voxel"]), "nx": int(grid["nx"]),
            "ny": int(grid["ny"]), "nz": int(grid["nz"]), "i0": i0, "j0": j0, "i1": i1, "j1": j1, "file": name,
            "entries": int(arrays[0].size)}
    temporary = live / ".where.json.tmp"
    temporary.write_text(json.dumps(meta))
    os.replace(temporary, live / "where.json")
    return meta
