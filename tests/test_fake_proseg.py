"""The fake proseg-live (tests/fake_proseg.py) as the tests of a setting's test use it: proseg's per-transcript file
(native/proseg/src/schemas.rs) and counts when asked, the threads and factorisation in info.json, and its opt-in knobs:
a world of its own (FAKE_PROSEG_WORLD: cells, genes, transcripts, labels, cell statistics in every frame, and rows moved
only by the tested value), one failed start (FAKE_PROSEG_FAIL_ONCE) and a cell left out (FAKE_PROSEG_DROP_CELL). Without
a knob it writes what it always wrote.

And proseg's own count as proseg-live 3.2.0.2 keeps it (--live-counts; INTERFACES §2, §12): the files read back with the
app's own readers (analysis/counts_file.py, engine/held.py), which frames are counted and where their files go, the
tokens served running and paused, where exports, the start block, FAKE_PROSEG_KEEP, an older proseg-live
(FAKE_PROSEG_OLD), the helpers in tests/counts_fixtures.py, and the fake analysis worker's UMAP summary on that
count."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import wait_for
from sopa_companion.analysis import counts_file
from sopa_companion.tests import counts_fixtures as CF

# A short schedule (10 frames: init, 3 burn-in, the upscale, 5 more; the last at iteration 8): the fake takes 5 ms an
# iteration at least.
SHORT = ["--burnin-samples", "3", "--samples", "4", "--recorded-samples", "2", "--hillclimb", "1"]
KNOBS = ("FAKE_PROSEG_WORLD", "FAKE_PROSEG_FAIL", "FAKE_PROSEG_FAIL_ONCE", "FAKE_PROSEG_DROP_CELL", "FAKE_PROSEG_OUTPUT",
         "FAKE_PROSEG_FINISH_S", "FAKE_PROSEG_LOAD_S", "FAKE_PROSEG_KEEP", "FAKE_PROSEG_OLD")
METADATA = ["--output-transcript-metadata", "transcript-metadata.parquet", "--output-transcript-metadata-fmt", "parquet"]


def run_fake(binary: Path, folder: Path, *extra: str, env: dict | None = None) -> tuple[Path, Path, int]:
    """One fake run to the end of a short schedule (no hold) in `folder`: (its live folder, its output folder, its exit
    code). The knobs are only the ones given in `env`."""
    live, out = folder / "live", folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items() if k not in KNOBS}
    environment.update(env or {})
    done = subprocess.run([str(binary), "--xenium", str(folder / "transcripts.parquet"), *SHORT, *extra,
                           "--output-path", str(out), "--live-dir", str(live), "--live-control", str(live / "control")],
                          env=environment, capture_output=True, text=True, timeout=120)
    return live, out, done.returncode


def frames(live: Path) -> list[dict]:
    return [json.loads(line) for line in (live / "frames.jsonl").read_text().splitlines()]


def records(live: Path) -> np.ndarray:
    from sopa_companion.engine.transcript_index import RECORD
    return np.frombuffer(gzip.decompress((live / "transcripts.bin.gz").read_bytes()), RECORD)


def layers_of(live: Path, frame: dict) -> np.ndarray:
    """A dense frame's planes: consensus, last change, then each layer, as [2 + nz, ny, nx]."""
    raw = np.frombuffer(gzip.decompress((live / frame["file"]).read_bytes()), "<u4")
    return raw.reshape(2 + frame["nz"], frame["ny"], frame["nx"])


# A world of 3 cells on a 6 x 4 grid of 2 µm voxels from (10, 20), 2 layers: c-1 and d-1 side by side in the top rows
# (a column of no cell between and after them), e-1 under c-1.
LABELS = [[1, 1, 0, 2, 2, 0],
          [1, 1, 0, 2, 2, 0],
          [3, 3, 0, 0, 0, 0],
          [3, 3, 0, 0, 0, 0]]
# (transcript_id, x, y, z, gene): two in c-1 (the second in its right-hand column), one in d-1, one in e-1, one in the
# column after d-1 (no cell), one left of the grid.
TRANSCRIPTS = [[101, 11.0, 21.0, 0.2, 0], [102, 13.5, 23.0, 0.7, 1], [103, 17.0, 21.5, 0.4, 0], [104, 11.0, 25.0, 0.9, 1],
               [105, 21.0, 21.0, 0.1, 0], [106, 9.0, 21.0, 0.1, 0]]


def write_world(path: Path, **changes) -> Path:
    world = {"cells": ["c-1", "d-1", "e-1"], "genes": ["G0", "G1"], "transcripts": TRANSCRIPTS, "nx": 6, "ny": 4, "nz": 2,
             "labels": [v for row in LABELS for v in row], "x0": 10.0, "y0": 20.0, "voxel": 2.0,
             # c-1's right-hand transcript one column on (into no cell's), d-1's one column on (still d-1's)
             "moves": {"flag": "cell-compactness", "value": 0.02, "rows": [1, 2], "dx": 1}, **changes}
    path.write_text(json.dumps(world))
    return path


def metadata(out: Path, name: str = "transcript-metadata.parquet"):
    import pyarrow.parquet as pq
    return pq.read_table(out / name)


def test_the_metadata_is_written_as_proseg_writes_it(fake_binary, tmp_path):
    """--output-transcript-metadata NAME.parquet: proseg's schema (schemas.rs: the floats float32, the gene a dictionary
    over genes.txt, the assignment nullable); row i is record i of transcripts.bin.gz (its observed bits), assigned to
    the cell its voxel holds in the last frame: the tiny sample's labels move each iteration, so the last frame's."""
    live, out, code = run_fake(fake_binary, tmp_path, "--nthreads", "8", *METADATA)
    assert code == 0
    table = metadata(out)
    assert [(f.name, str(f.type), f.nullable) for f in table.schema] == [
        ("transcript_id", "uint64", True), ("qv", "float", True), ("x", "float", False), ("y", "float", False),
        ("z", "float", False), ("observed_x", "float", False), ("observed_y", "float", False),
        ("observed_z", "float", False), ("gene", "dictionary<values=string, indices=int32, ordered=0>", False),
        ("assignment", "uint32", True), ("background", "bool", False)]
    found = records(live)
    for axis in ("x", "y", "z"):
        observed = table[f"observed_{axis}"].to_numpy()
        assert observed.view(np.uint32).tolist() == np.ascontiguousarray(found[axis]).view(np.uint32).tolist()
        assert table[axis].to_numpy().tolist() == observed.tolist()          # no offsets without a world's moves
    assert table["transcript_id"].to_pylist() == [0, 1, 2] and table["qv"].to_pylist() == [40.0] * 3
    genes = (live / "genes.txt").read_text().split("\n")[:-1]
    assert table["gene"].to_pylist() == [genes[g] for g in found["gene"].tolist()]
    assert table["gene"].chunk(0).dictionary.to_pylist() == genes
    last = frames(live)[-1]
    labels = layers_of(live, last)[2:]                         # every layer alike
    z = np.clip(np.floor(found["z"] / 0.5), 0, 1).astype(int)
    held = [int(labels[k, int(y), int(x)]) for x, y, k in zip(found["x"], found["y"], z)]
    assert table["assignment"].to_pylist() == [h - 1 if h else None for h in held]
    assert table["background"].to_pylist() == [h == 0 for h in held]
    info = json.loads((live / "info.json").read_text())
    assert (info["threads"], info["factorization"]) == (8, True)
    assert sorted(path.name for path in out.iterdir()) == \
        ["cell-polygons.geojson.gz", "proseg-output.zarr", "transcript-metadata.parquet"]


def test_other_names_are_written_as_csv_gz_and_counts_only_when_asked(fake_binary, tmp_path):
    """A detail run's own outputs (engine/runs.py DETAIL_OUTPUTS): transcript-metadata.csv.gz, read as Phase 0 read
    proseg's (the floats parsed straight to float32), and counts.mtx.gz, the assignments counted per cell and gene; a
    .parquet name with another format asked is csv.gz too."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    live, out, code = run_fake(fake_binary, tmp_path, "--output-counts", "counts.mtx.gz",
                               "--output-transcript-metadata", "transcript-metadata.csv.gz", "--no-factorization")
    assert code == 0
    types = {"transcript_id": pa.uint64(), "qv": pa.float32(), **{name: pa.float32() for name in (
        "x", "y", "z", "observed_x", "observed_y", "observed_z")}, "gene": pa.string(), "assignment": pa.uint32(),
        "background": pa.bool_()}
    table = pacsv.read_csv(out / "transcript-metadata.csv.gz", convert_options=pacsv.ConvertOptions(
        column_types=types, null_values=[""], strings_can_be_null=False))
    assert table.column_names == list(types)
    found = records(live)
    assert table["observed_x"].to_numpy().view(np.uint32).tolist() == \
        np.ascontiguousarray(found["x"]).view(np.uint32).tolist()
    assignment = table["assignment"].to_pylist()
    counts = gzip.decompress((out / "counts.mtx.gz").read_bytes()).decode().splitlines()
    assert counts[0] == "%%MatrixMarket matrix coordinate integer general"
    expected = {}
    for cell, gene in zip(assignment, found["gene"].tolist()):
        if cell is not None:
            expected[(cell + 1, gene + 1)] = expected.get((cell + 1, gene + 1), 0) + 1
    assert counts[1] == f"2 2 {len(expected)}"
    assert {tuple(int(v) for v in line.split()[:2]): int(line.split()[2]) for line in counts[2:]} == expected
    assert json.loads((live / "info.json").read_text())["factorization"] is False
    # Asked as parquet by name but csv by format: csv.gz.
    _, out, _ = run_fake(fake_binary, tmp_path / "csv", "--output-transcript-metadata", "metadata.parquet",
                         "--output-transcript-metadata-fmt", "csv")
    assert gzip.decompress((out / "metadata.parquet").read_bytes()).startswith(b"transcript_id,qv,x,")


def test_no_knob_changes_nothing(fake_binary, tmp_path):
    """Without a knob the fake writes what it always wrote (the tiny sample: 2 cells, 2 genes, 3 transcripts on a 4 x 4
    grid of 1 µm whose labels move each iteration, no cell statistics), and no per-transcript file unless asked; info.json
    only gains the threads and the factorisation."""
    live, out, code = run_fake(fake_binary, tmp_path)
    assert code == 0
    assert (live / "cells.txt").read_text() == "aaa-1\nbbb-1\n" and (live / "genes.txt").read_text() == "G1\nG2\n"
    found = records(live)
    assert [tuple(float(v) for v in r)[:3] + (int(r[3]),) for r in found.tolist()] == \
        [(1.0, 1.0, pytest.approx(0.2), 0), (2.5, 1.5, pytest.approx(0.7), 1), (3.0, 3.0, pytest.approx(0.4), 0)]
    info = json.loads((live / "info.json").read_text())
    assert set(info) == {"schema", "transcripts", "cells", "genes", "layers", "hold", "schedule", "plan", "z", "threads",
                         "factorization"}
    assert (info["transcripts"], info["cells"], info["genes"], info["layers"]) == (3, 2, 2, 2)
    assert info["z"] == {"thickness_um": 10, "grid_origin": 0, "grid_spacing": 0.5} and info["threads"] >= 1
    seen = frames(live)
    assert [f["iter"] for f in seen] == [0, 1, 2, 3, 3, 4, 5, 6, 7, 8]
    for frame in seen:
        assert "cells_file" not in frame and (frame["nx"], frame["ny"], frame["x0"], frame["voxel"]) == (4, 4, 0.0, 1.0)
        assert (frame["cells"], frame["transcripts"], frame["assigned"]) == (2, 3, 2)
        planes = layers_of(live, frame)
        expected = (frame["iter"] + np.arange(16)) % 3
        assert planes[0].ravel().tolist() == planes[2].ravel().tolist() == planes[3].ravel().tolist() == \
            expected.tolist()
    assert not list(live.glob("frames/*.cells.zst"))
    assert sorted(path.name for path in out.iterdir()) == ["cell-polygons.geojson.gz", "proseg-output.zarr"]


def test_a_world_replaces_the_tiny_sample_with_its_cell_statistics_in_every_frame(fake_binary, tmp_path):
    """FAKE_PROSEG_WORLD: its cells, genes and transcripts (their ids in the per-transcript file), its labels in every
    layer from frame 0 on, on its own grid; every frame's record names its cell statistics (frames/NNNNNN.cells.zst),
    which the analysis worker's read_centres reads: each cell's centre, x0 + (mean column + 0.5) x voxel."""
    from sopa_companion.analysis.worker import read_centres
    import zstandard
    world = write_world(tmp_path / "world.json")
    live, out, code = run_fake(fake_binary, tmp_path / "run", *METADATA, env={"FAKE_PROSEG_WORLD": str(world)})
    assert code == 0
    assert (live / "cells.txt").read_text() == "c-1\nd-1\ne-1\n" and (live / "genes.txt").read_text() == "G0\nG1\n"
    found = records(live)
    assert found["x"].tolist() == [t[1] for t in TRANSCRIPTS] and found["gene"].tolist() == [t[4] for t in TRANSCRIPTS]
    info = json.loads((live / "info.json").read_text())
    assert (info["transcripts"], info["cells"], info["genes"], info["layers"]) == (6, 3, 2, 2)
    seen = frames(live)
    assert len(seen) == 10
    for frame in seen:
        assert (frame["nx"], frame["ny"], frame["nz"], frame["x0"], frame["y0"], frame["voxel"]) == \
            (6, 4, 2, 10.0, 20.0, 2.0)
        planes = layers_of(live, frame)
        for plane in (planes[0], planes[2], planes[3]):
            assert plane.tolist() == LABELS
        assert frame["cells_file"] == f"frames/{frame['seq']:06d}.cells.zst"
        centres = read_centres(live / frame["cells_file"], 3)
        assert centres["cx"].tolist() == [12.0, 18.0, 12.0] and centres["cy"].tolist() == [22.0, 22.0, 26.0]
    stats = np.frombuffer(zstandard.ZstdDecompressor().decompress((live / seen[-1]["cells_file"]).read_bytes()), "<u4")
    area, voxels, assigned, foreground, changed = stats.reshape(7, 3)[:5].tolist()
    assert (area, voxels, assigned, foreground, changed) == ([4, 4, 4], [8, 8, 8], [2, 1, 1], [2, 1, 1], [0, 0, 0])
    assert (seen[-1]["assigned"], seen[-1]["cells"], seen[-1]["transcripts"]) == (4, 3, 6)
    table = metadata(out)
    assert table["transcript_id"].to_pylist() == [101, 102, 103, 104, 105, 106]
    assert table["assignment"].to_pylist() == [0, 0, 1, 2, None, None]       # no cell after d-1, nor off the grid
    assert table["background"].to_pylist() == [False, False, False, False, True, True]
    assert table["x"].to_pylist() == [t[1] for t in TRANSCRIPTS]
    assert table["gene"].to_pylist() == ["G0", "G1", "G0", "G1", "G0", "G0"]


@pytest.mark.parametrize("flags, moved", [(["--cell-compactness", "0.02"], True), (["--cell-compactness", "0.05"], False),
                                          ([], False), (["--prior-weight", "0.02"], False)])
def test_a_worlds_moves_follow_the_tested_value_only(fake_binary, tmp_path, flags, moved):
    """`moves` acts only in a run whose command line carries the tested value itself (side B's --cell-compactness 0.02),
    never in one that carries the flag at another value (side A copies the parent's options) or another flag at that
    value: its rows are written one voxel on (x + 2 µm, observed x kept) and assigned where they land, c-1's right-hand
    transcript to no cell (background), d-1's still to d-1; the cell statistics count them there."""
    from sopa_companion.analysis.worker import read_centres
    import zstandard
    world = write_world(tmp_path / "world.json")
    live, out, code = run_fake(fake_binary, tmp_path / "run", *flags, *METADATA, env={"FAKE_PROSEG_WORLD": str(world)})
    assert code == 0
    table = metadata(out)
    xs = [t[1] for t in TRANSCRIPTS]
    assert table["observed_x"].to_pylist() == xs
    if moved:
        assert table["x"].to_pylist() == [11.0, 15.5, 19.0, 11.0, 21.0, 9.0]
        assert table["assignment"].to_pylist() == [0, None, 1, 2, None, None]
        assert table["background"].to_pylist() == [False, True, False, False, True, True]
    else:
        assert table["x"].to_pylist() == xs and table["assignment"].to_pylist() == [0, 0, 1, 2, None, None]
    stats = np.frombuffer(zstandard.ZstdDecompressor().decompress((live / frames(live)[0]["cells_file"]).read_bytes()),
                          "<u4").reshape(7, 3)
    assert stats[2].tolist() == ([1, 1, 1] if moved else [2, 1, 1])
    assert read_centres(live / frames(live)[0]["cells_file"], 3)["cx"].tolist() == [12.0, 18.0, 12.0]   # labels kept


def test_a_cell_can_be_left_out_in_every_run_or_only_with_a_flag(fake_binary, tmp_path):
    """FAKE_PROSEG_DROP_CELL=d-1: cells.txt without it, e-1 now label 2, d-1's voxels no cell's (its transcript
    background); with ",prior-weight", only in a run carrying --prior-weight."""
    world = write_world(tmp_path / "world.json")
    env = {"FAKE_PROSEG_WORLD": str(world), "FAKE_PROSEG_DROP_CELL": "d-1"}
    live, out, _ = run_fake(fake_binary, tmp_path / "every", *METADATA, env=env)
    assert (live / "cells.txt").read_text() == "c-1\ne-1\n"
    assert json.loads((live / "info.json").read_text())["cells"] == 2
    assert layers_of(live, frames(live)[-1])[0].tolist() == [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0],
                                                             [2, 2, 0, 0, 0, 0]]
    assert metadata(out)["assignment"].to_pylist() == [0, 0, None, 1, None, None]
    env["FAKE_PROSEG_DROP_CELL"] = "d-1,prior-weight"
    live, out, _ = run_fake(fake_binary, tmp_path / "without", *METADATA, env=env)
    assert (live / "cells.txt").read_text() == "c-1\nd-1\ne-1\n"
    live, out, _ = run_fake(fake_binary, tmp_path / "with", "--prior-weight", "2.0", *METADATA, env=env)
    assert (live / "cells.txt").read_text() == "c-1\ne-1\n"
    # The tiny sample too: aaa-1 left out, its voxels no cell's, bbb-1 now label 1.
    live, _, _ = run_fake(fake_binary, tmp_path / "tiny", env={"FAKE_PROSEG_DROP_CELL": "aaa-1"})
    assert (live / "cells.txt").read_text() == "bbb-1\n"
    last = frames(live)[-1]
    assert layers_of(live, last)[0].ravel().tolist() == [max(0, (last["iter"] + p) % 3 - 1) for p in range(16)]


def test_one_start_fails_once_and_never_with_no_factorization(fake_binary, tmp_path):
    """FAKE_PROSEG_FAIL_ONCE: the first run that creates the marker exits 101 before any frame (the panic in its log);
    every later run starts, and a run with --no-factorization neither fails nor takes the failure for itself."""
    marker = tmp_path / "failed-once"
    env = {"FAKE_PROSEG_FAIL_ONCE": str(marker)}
    live, _, code = run_fake(fake_binary, tmp_path / "first", env=env)
    assert code == 101 and marker.exists() and not live.exists()
    live, _, code = run_fake(fake_binary, tmp_path / "second", env=env)
    assert code == 0 and len(frames(live)) == 10
    other = tmp_path / "other-marker"
    live, _, code = run_fake(fake_binary, tmp_path / "third", "--no-factorization",
                             env={"FAKE_PROSEG_FAIL_ONCE": str(other)})
    assert code == 0 and not other.exists()
    assert json.loads((live / "info.json").read_text())["factorization"] is False


# ---- proseg's own count (--live-counts) -----------------------------------------------------------------------------
# A world for proseg's own count: LABELS' cells with several transcripts of a gene: c-1 3 of G0 and 2 of G1, d-1 1 of
# G0 and 4 of G1, e-1 none (an empty row), and 3 in no cell (the column between, the one after d-1, left of the grid).
# Its moves (with --cell-compactness 0.02): c-1's third G0 and d-1's first G1 one column on, into no cell's.
COUNTED = [[201, 11.0, 21.0, 0.2, 0], [202, 12.5, 22.5, 0.7, 0], [203, 13.0, 21.0, 0.3, 0], [204, 11.5, 23.0, 0.6, 1],
           [205, 12.0, 20.5, 0.1, 1], [206, 17.0, 21.5, 0.4, 0], [207, 18.0, 22.0, 0.2, 1], [208, 19.0, 23.0, 0.8, 1],
           [209, 16.5, 20.5, 0.5, 1], [210, 17.5, 23.5, 0.9, 1], [211, 15.0, 21.0, 0.1, 0], [212, 21.0, 21.0, 0.1, 0],
           [213, 9.0, 21.0, 0.1, 0]]
MOVES = {"flag": "cell-compactness", "value": 0.02, "rows": [2, 6], "dx": 1}
# Its count, cell by cell (c-1: G0, G1; d-1: G0, G1; e-1: none): as observed, and with the moves.
PATTERN = {"indptr": [0, 2, 4, 4], "genes": [0, 1, 0, 1]}
ASSIGNED, MOVED = [3, 2, 1, 4], [2, 2, 1, 3]
FORCED = ("init", "upscale", "pause", "hold", "finish", "end")
REQUEST, EVERY = ["--live-counts", "request"], ["--live-counts", "every"]


def counts_lines(live: Path) -> list[dict]:
    """counts.jsonl's lines ([] without the file)."""
    path = live / "counts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def read_line(live: Path, line: dict, **fits) -> counts_file.Counts:
    """A counts line's file read back (a spool file's absolute name stays absolute under the live folder)."""
    return counts_file.read(live / line["file"], **fits)


def check_line(live: Path, line: dict, spool: Path | None = None, **fits) -> counts_file.Counts:
    """A counts line (INTERFACES §2.6): every key in its order, its file where its reason puts it (the spool for
    request, every and pause frames when there is one, else live/frames) and read back, its stats the file's own, the
    sidecar its rows' Σ e, and f <= n. Returns the file's counts."""
    assert list(line) == list(CF.LINE_KEYS) and line["reason"] in CF.REASONS
    name = f"{line['seq']:06d}.counts.zst"
    assert line["file"] == (str(spool / name) if spool is not None and line["reason"] in ("request", "every", "pause")
                            else f"frames/{name}")
    assert line["counted_file"] == f"frames/{line['seq']:06d}.counted.zst"
    counts = read_line(live, line, **fits)
    assert counts.iteration == line["iter"] and line["nnz"] == counts.nnz and line["sum_n"] == int(counts.n.sum())
    assert line["sum_e"] == pytest.approx(float(np.sum(counts.e, dtype=np.float64)))
    assert line["has_f"] == (counts.f is not None) and line["has_start"] == (counts.start_e is not None)
    assert line["sum_f"] == (None if counts.f is None else int(counts.f.sum()))
    assert line["saturated"] == 0 and line["ms"] == {"extract": 0.0, "write": 0.0}
    assert line["bytes"] == (live / line["file"]).stat().st_size
    assert counts.f is None or bool((counts.f <= counts.n).all())
    sidecar = counts_file.read_counted(live / line["counted_file"], counts.ncells)
    assert sidecar.dtype == np.float32 and sidecar.tolist() == counts_file.counted_of(counts).tolist()
    return counts


def check_export(live: Path, meta: dict, frame: dict):
    """A where export (W, and its file read back by engine/held.py): every key, the frame's grid, the span of its box,
    and one record per transcript whose observed voxel lies in the span (any layer), n = 1, its dest the label `frame`
    holds there (the tiny sample moves none)."""
    from sopa_companion.engine import held
    assert list(meta) == list(CF.WHERE_KEYS) and meta["file"] == f"where-{meta['token']}.bin.zst"
    assert meta["margin_um"] == 24.0 and (meta["seq"], meta["iter"]) == (frame["seq"], frame["iter"])
    assert all(meta[key] == frame[key] for key in ("x0", "y0", "voxel", "nx", "ny", "nz"))
    assert (meta["i0"], meta["j0"], meta["i1"], meta["j1"]) == CF.where_span(meta["box"], frame)
    found = records(live)
    i, j, k, _ = CF.observed_voxels(found, frame, 0, 0.5)
    span = (i >= meta["i0"]) & (i <= meta["i1"]) & (j >= meta["j0"]) & (j <= meta["j1"])
    dest = layers_of(live, frame)[2:][k[span], j[span], i[span]]
    expected = sorted(zip(i[span].tolist(), j[span].tolist(), k[span].tolist(), found["gene"][span].tolist(),
                          dest.tolist(), [1] * int(span.sum())))
    export = held.read_where(meta, live / meta["file"])
    assert meta["entries"] == len(expected) and \
        sorted(zip(*(getattr(export, key).tolist() for key in ("i", "j", "k", "gene", "dest", "n")))) == expected


def new_files(live: Path) -> list[str]:
    """What only a proseg-live that keeps proseg's own count writes in its live folder."""
    return sorted(path.name for pattern in ("counts.jsonl", "*.counts.zst", "*.counted.zst", "where*")
                  for path in live.rglob(pattern))


@pytest.mark.parametrize("keep, spooled", [(1.0, True), (0.75, True), (0.75, False)])
def test_every_frame_holds_each_cells_count_per_gene(fake_binary, tmp_path, keep, spooled):
    """--live-counts every: each frame's count (init and upscale in live/frames, the others in the spool when there is
    one), read back as the app reads it: per cell and gene, n = the transcripts whose destination it holds, e = n x
    keep, f = floor(n x keep), an empty row for a cell holding none; the cell statistics' assigned and foreground are
    the rows' Σ n and Σ f, and the saved counts the last frame's f. The end makes the last count durable: counted again
    from the same state into live/frames when it was spooled."""
    import zstandard
    world = write_world(tmp_path / "world.json", transcripts=COUNTED, moves=MOVES)
    spool = tmp_path / "spool" if spooled else None
    live, out, code = run_fake(fake_binary, tmp_path / "run", *EVERY,
                               *(["--live-counts-dir", str(spool)] if spool else []), "--output-counts", "counts.mtx.gz",
                               env={"FAKE_PROSEG_WORLD": str(world), "FAKE_PROSEG_KEEP": str(keep)})
    assert code == 0
    assert json.loads((live / "info.json").read_text())["counts"] == {
        "mode": "every", "format": "PLC1", "version": 1, "e_scale": 32, "start": False, "start_every": None,
        "spool": str(spool) if spool else None, "where": 1}
    seen, lines = frames(live), counts_lines(live)
    assert [line["seq"] for line in lines] == list(range(10)) + ([9] if spooled else [])
    assert [line["reason"] for line in lines] == ["init", *["every"] * 3, "upscale", *["every"] * 5] + \
        (["end"] if spooled else [])
    n = np.asarray(ASSIGNED)
    f = np.floor(n * keep).astype(int)
    for line in lines:
        assert line["iter"] == seen[line["seq"]]["iter"] and line["token"] is None and line["where"] is None
        counts = check_line(live, line, spool, ncells=3, ngenes=2)
        assert counts.indptr.tolist() == PATTERN["indptr"] and counts.genes.tolist() == PATTERN["genes"]
        assert counts.n.tolist() == ASSIGNED and counts.e.tolist() == (n * keep).tolist()
        assert counts.f.tolist() == f.tolist() and counts.start_e is None
        assert counts_file.read_counted(live / line["counted_file"], 3).tolist() == [5 * keep, 5 * keep, 0.0]
    stats = np.frombuffer(zstandard.ZstdDecompressor().decompress((live / seen[-1]["cells_file"]).read_bytes()),
                          "<u4").reshape(7, 3)
    assert stats[2].tolist() == [5, 5, 0] and stats[3].tolist() == [f[0] + f[1], f[2] + f[3], 0]
    assert (seen[-1]["assigned"], seen[-1]["foreground"]) == (10, int(f.sum()))
    saved = gzip.decompress((out / "counts.mtx.gz").read_bytes()).decode().splitlines()
    last = read_line(live, lines[-1])
    rows = np.repeat(np.arange(3), np.diff(last.indptr))
    assert saved[1] == f"3 2 {int((last.f > 0).sum())}"
    assert {tuple(int(v) for v in row.split()[:2]): int(row.split()[2]) for row in saved[2:]} == \
        {(int(c) + 1, int(g) + 1): int(v) for c, g, v in zip(rows, last.genes, last.f) if v}


@pytest.mark.parametrize("tiled", [False, True])
def test_a_request_run_counts_its_init_upscale_and_end(fake_binary, tmp_path, tiled):
    """--live-counts request with no token: only the forced frames are counted, durably (live/frames, with f): init,
    upscale and, at the end, the last frame; nothing goes to the spool. Each is the count of its frame's labels as the
    app reads them (counts_fixtures.counts_from_labels on transcripts.bin.gz: the tiny sample's labels move each
    iteration). Dense and tiled alike."""
    spool = tmp_path / "spool"
    live, _, code = run_fake(fake_binary, tmp_path / "run", *REQUEST, "--live-counts-dir", str(spool),
                             *(["--live-tiled"] if tiled else []))
    assert code == 0
    seen, lines = frames(live), counts_lines(live)
    assert [(line["seq"], line["reason"]) for line in lines] == [(0, "init"), (4, "upscale"), (9, "end")]
    assert spool.is_dir() and not list(spool.iterdir()) and not list(live.glob("where*"))
    assert json.loads((live / "info.json").read_text())["counts"]["spool"] == str(spool)
    found = records(live)
    for line in lines:
        frame = seen[line["seq"]]
        counts = check_line(live, line, spool, ncells=2, ngenes=2)
        assert line["has_f"] and not line["has_start"] and line["iter"] == frame["iter"]
        plane = [(frame["iter"] + p) % 3 for p in range(frame["nx"] * frame["ny"])]
        expected = CF.counts_from_labels(found, plane * frame["nz"], frame, 0, 0.5, 2, 2)
        assert {key: getattr(counts, key).tolist() for key in ("indptr", "genes", "n", "e", "f")} == \
            {key: expected[key].tolist() for key in ("indptr", "genes", "n", "e", "f")}


BAD_TOKENS = ["counts 0", "counts -1", "counts x", "counts 1.5", "counts 18446744073709551616", "where 0 10 20 17 24",
              "where 3 17 20 10 24", "where 3 10 24 17 24", "where 3 10 20 nan 24", "where 3 10 20 inf 24",
              "where 3 10 20 1e39 24", "where 3 1_0 20 17 24", "where 3 10 20 17", "where 3 10 20 17 24 1"]


@pytest.mark.parametrize("tokens, tiled", [(["counts 5", "where 7 10.00 20.00 17.00 24.00"], False),
                                           (["counts 5", "where 7 10.00 20.00 17.00 24.00"], True),
                                           (BAD_TOKENS, False)])
def test_tokens_waiting_at_the_start_are_served_by_the_first_frame(fake_binary, tmp_path, tokens, tiled):
    """Tokens in the control file when proseg starts are served by the first frame (init: counted anyway), its line
    carrying the counts token and the where export of its state (W, also where.json): one record per transcript
    whose observed voxel (before its move) lies in the box's span, dest the label of the voxel it moved to; no frame is
    counted for them again. Lines proseg-live would not parse (INTERFACES §2.2) are ignored."""
    from sopa_companion.engine import held
    world = write_world(tmp_path / "world.json", transcripts=COUNTED, moves=MOVES)
    folder = tmp_path / "run"
    (folder / "live").mkdir(parents=True)
    (folder / "live" / "control").write_text("".join(f"{token}\n" for token in tokens))
    live, _, code = run_fake(fake_binary, folder, *REQUEST, "--cell-compactness", "0.02",
                             *(["--live-tiled"] if tiled else []), env={"FAKE_PROSEG_WORLD": str(world)})
    assert code == 0
    lines = counts_lines(live)
    assert [(line["seq"], line["reason"]) for line in lines] == [(0, "init"), (4, "upscale"), (9, "end")]
    assert all(line["token"] is None and line["where"] is None for line in lines[1:])
    assert check_line(live, lines[0], ncells=3, ngenes=2).n.tolist() == MOVED
    if tokens is BAD_TOKENS:
        assert lines[0]["token"] is None and lines[0]["where"] is None and not list(live.glob("where*"))
        return
    meta = lines[0]["where"]
    assert lines[0]["token"] == 5 and meta == json.loads((live / "where.json").read_text())
    assert meta == {"token": 7, "seq": 0, "iter": 0, "box": [10.0, 20.0, 17.0, 24.0], "margin_um": 24.0, "x0": 10.0,
                    "y0": 20.0, "voxel": 2.0, "nx": 6, "ny": 4, "nz": 2, "i0": 0, "j0": 0, "i1": 3, "j1": 2,
                    "file": "where-7.bin.zst", "entries": 9}
    expected = []
    for row, (_, x, y, z, gene) in enumerate(COUNTED):
        i, j, k = int((x - 10.0) // 2.0), int((y - 20.0) // 2.0), min(1, int(z // 0.5))
        if 0 <= i <= 3 and 0 <= j <= 2:
            to = i + (MOVES["dx"] if row in MOVES["rows"] else 0)
            expected.append((i, j, k, gene, LABELS[j][to] if to < 6 else 0, 1))
    export = held.read_where(meta, live / meta["file"])
    assert sorted(zip(*(getattr(export, key).tolist() for key in ("i", "j", "k", "gene", "dest", "n")))) == \
        sorted(expected)
    assert (2, 0, 0, 0, 0, 1) in expected and (1, 0, 0, 0, 0, 1) in expected     # the column between; c-1's moved G0


class Driven:
    """A fake run driven as the engine drives proseg (engine/runs.py _write_control: the control file rewritten whole,
    at once), paused from its start unless told otherwise: then each step is one frame."""

    def __init__(self, binary: Path, folder: Path, *extra: str, env: dict | None = None, **control):
        self.live, self.out = folder / "live", folder / "output"
        self.live.mkdir(parents=True)
        self.out.mkdir(parents=True)
        self.values = {"paused": 1, "step": 0, "min_ms": 0, "finish": 0, "continue": 0}
        self.tokens: list[str] = []
        self.set(**control)
        environment = {k: v for k, v in os.environ.items() if k not in KNOBS}
        environment.update(env or {})
        with open(folder / "proseg.log", "wb") as log:
            self.process = subprocess.Popen(
                [str(binary), "--xenium", str(folder / "transcripts.parquet"), *SHORT, *extra, "--output-path",
                 str(self.out), "--live-dir", str(self.live), "--live-control", str(self.live / "control")],
                env=environment, stdout=log, stderr=subprocess.STDOUT)

    def set(self, *tokens: str, **values):
        """The control file rewritten: `values` changed, and the token lines replaced when any are given."""
        self.values.update(values)
        if tokens:
            self.tokens = list(tokens)
        lines = [f"{key} {value}" for key, value in self.values.items()] + self.tokens
        (self.live / ".control.tmp").write_text("\n".join(lines) + "\n")
        os.replace(self.live / ".control.tmp", self.live / "control")

    def read(self, name: str) -> dict:
        try:
            return json.loads((self.live / name).read_text())
        except (OSError, ValueError):
            return {}

    def acked(self, state: str, seq: int | None = None, since: int = 0) -> dict:
        """Waits for proseg's ack of `state` (at frame `seq` when given; else at frame `since` or later, so an ack left
        from an earlier pause does not count): that ack."""
        found = []

        def match():
            ack = self.read("ack")
            if ack.get("state") == state and (ack.get("seq") == seq if seq is not None else
                                              ack.get("seq", -1) >= since):
                found.append(ack)
            return bool(found)
        wait_for(match)
        return found[0]

    def lines(self) -> list[dict]:
        return counts_lines(self.live)

    def exit_code(self) -> int:
        return self.process.wait(30)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(30)


@pytest.fixture
def drive(fake_binary, tmp_path):
    """Starts driven fake runs (Driven), each in a folder of its own; any still running is killed when the test ends."""
    runs = []

    def start(*extra, **kwargs) -> Driven:
        runs.append(Driven(fake_binary, tmp_path / f"driven-{len(runs)}", *extra, **kwargs))
        return runs[-1]
    yield start
    for run in runs:
        run.close()


def test_a_pause_counts_its_state_once_before_its_ack(drive, tmp_path):
    """settle("paused"): the state proseg pauses on gets its count (reason pause: in the spool, with f) before the
    "paused" ack names it, and once (the paused loop never counts it again); a state already counted (frame 0's init)
    gets none. A finish then makes the paused state's count durable, counted again from the same state into
    live/frames (reason finish: the same n and e), and the end adds nothing."""
    spool = tmp_path / "spool"
    run = drive(*REQUEST, "--live-counts-dir", str(spool), "--live-hold")
    run.acked("paused", 0)
    assert [line["reason"] for line in run.lines()] == ["init"]
    for seq in (1, 2):
        run.set(step=seq)
        run.acked("paused", seq)
        lines = run.lines()                     # read after the ack: its line is there already
        assert (lines[-1]["seq"], lines[-1]["reason"]) == (seq, "pause")
        check_line(run.live, lines[-1], spool, ncells=2, ngenes=2)
        assert lines[-1]["has_f"] and not lines[-1]["has_start"] and lines[-1]["token"] is None
    stamps = set()

    def loops():
        stamps.add((run.live / "ack").stat().st_mtime_ns)
        return len(stamps) >= 4
    wait_for(loops)                             # the paused loop went round a few more times
    assert [(line["seq"], line["reason"]) for line in run.lines()] == [(0, "init"), (1, "pause"), (2, "pause")]
    run.set(finish=1)
    run.acked("finishing", 2)
    assert run.exit_code() == 0 and (run.live / "sampling_done").exists()
    lines = run.lines()
    assert [(line["seq"], line["reason"]) for line in lines] == [(0, "init"), (1, "pause"), (2, "pause"), (2, "finish")]
    paused, durable = (check_line(run.live, line, spool, ncells=2, ngenes=2) for line in lines[2:])
    assert lines[3]["file"] == "frames/000002.counts.zst" and lines[3]["has_f"]
    assert durable.n.tolist() == paused.n.tolist() and durable.e.tolist() == paused.e.tolist()


def test_a_pause_on_a_frame_a_token_counted_counts_it_again_with_f(drive, tmp_path):
    """proseg-live 3.2.0.2: a counts token and a step in one write while paused make the step's frame a request frame
    (in the spool, without f, its line carrying the token); the pause that follows counts that frame again from the
    same state (reason pause: the spool's file of that name replaced, with f and the start block, the same n and e)
    before its "paused" ack names it, once (the paused loop never counts it again). A finish then counts it durably into
    live/frames (reason finish: the same n and e), and the end adds nothing."""
    spool = tmp_path / "spool"
    run = drive(*REQUEST, "--live-counts-dir", str(spool), "--live-counts-start", "4", "--live-hold")
    run.acked("paused", 0)
    run.set("counts 1", step=1)                 # one write: the step's frame is counted for the token
    run.acked("paused", 1)
    lines = run.lines()                         # read after the ack: the pause's line is there already
    assert [(line["seq"], line["reason"], line["token"]) for line in lines] == \
        [(0, "init", None), (1, "request", 1), (1, "pause", None)]
    request, pause = lines[1:]
    assert request["file"] == pause["file"] == str(spool / "000001.counts.zst")
    assert not request["has_f"] and request["sum_f"] is None and not request["has_start"]
    assert pause["has_f"] and pause["has_start"] and pause["where"] is None
    again = check_line(run.live, pause, spool, ncells=2, ngenes=2)       # the file now holds the pause's count
    assert (pause["nnz"], pause["sum_n"], pause["sum_e"]) == (request["nnz"], request["sum_n"], request["sum_e"])
    stamps = set()

    def loops():
        stamps.add((run.live / "ack").stat().st_mtime_ns)
        return len(stamps) >= 4
    wait_for(loops)                             # the paused loop went round a few more times
    assert run.lines() == lines
    run.set(finish=1)
    run.acked("finishing", 1)
    assert run.exit_code() == 0 and (run.live / "sampling_done").exists()
    last = run.lines()
    assert last[:3] == lines and [(line["seq"], line["reason"]) for line in last[3:]] == [(1, "finish")]
    durable = check_line(run.live, last[3], spool, ncells=2, ngenes=2)
    assert last[3]["file"] == "frames/000001.counts.zst" and last[3]["has_f"] and last[3]["has_start"]
    assert durable.n.tolist() == again.n.tolist() and durable.e.tolist() == again.e.tolist()
    assert durable.f.tolist() == again.f.tolist() and durable.start_e.tolist() == again.start_e.tolist()


def test_without_a_spool_a_pause_counts_a_token_frame_again_in_live_frames_and_a_finish_adds_none(drive):
    """Without --live-counts-dir a token's frame is counted into live/frames, without f; the pause counts it again
    there (reason pause: the file of that name replaced, with f), which is durable already, so a finish adds no line."""
    run = drive(*REQUEST, "--live-hold")
    run.acked("paused", 0)
    run.set("counts 1", step=1)
    run.acked("paused", 1)
    lines = run.lines()
    assert [(line["seq"], line["reason"], line["file"], line["has_f"]) for line in lines] == \
        [(0, "init", "frames/000000.counts.zst", True), (1, "request", "frames/000001.counts.zst", False),
         (1, "pause", "frames/000001.counts.zst", True)]
    assert check_line(run.live, lines[-1], ncells=2, ngenes=2).f is not None
    run.set(finish=1)
    run.acked("finishing", 1)
    assert run.exit_code() == 0 and run.lines() == lines


def test_tokens_read_while_running_count_the_next_frame(drive, tmp_path):
    """Running, a counts token not served yet makes the next frame a request frame (in the spool, without f, its line
    carrying the token); a where token too, with the export of that frame's state (its line carrying W, its counts
    token then null). Each is served once: no later frame is counted for it, and a paused read serves the old ones
    with no line."""
    spool = tmp_path / "spool"
    run = drive(*REQUEST, "--live-counts-dir", str(spool), "--burnin-samples", "200")
    run.acked("paused", 0)
    run.set("counts 1", paused=0, min_ms=20)          # resumed with the token: no paused read serves it
    wait_for(lambda: any(line["token"] == 1 for line in run.lines()))
    [first] = [line for line in run.lines() if line["token"] == 1]
    # Its file read while the run samples: a pause landing on its frame would count it again, with f (3.2.0.2).
    assert check_line(run.live, first, spool, ncells=2, ngenes=2).f is None
    run.set(paused=1)
    run.acked("paused", since=first["seq"])
    requests = [line for line in run.lines() if line["reason"] == "request"]
    assert requests == [first] and requests[0]["token"] == 1 and requests[0]["where"] is None
    assert not requests[0]["has_f"] and requests[0]["sum_f"] is None
    assert requests[0]["seq"] in {frame["seq"] for frame in frames(run.live)}
    run.set("counts 1", "where 2 0.00 0.00 2.50 4.00", paused=0)
    wait_for(lambda: any(line["where"] for line in run.lines()))
    run.set(paused=1)
    run.acked("paused", since=next(line for line in run.lines() if line["where"])["seq"])
    lines = run.lines()
    exported = [line for line in lines if line["where"]]
    assert len(exported) == 1 and exported[0]["reason"] == "request" and exported[0]["token"] is None
    assert exported[0]["where"] == run.read("where.json") and exported[0]["where"]["token"] == 2
    check_export(run.live, exported[0]["where"], frames(run.live)[exported[0]["seq"]])
    assert [line["token"] for line in lines if line["reason"] == "request"] == [1, None]
    assert all(line["reason"] in ("init", "request", "pause") for line in lines)


def test_the_hold_counts_its_state_durably_and_serves_tokens_from_it(drive, tmp_path):
    """--live-hold: the held state (the schedule's last frame, not counted before) is counted (reason hold: live/frames,
    with f) before the "paused" ack and schedule_done; tokens read while held are served from it (a where export of it
    at once, a counts token with no line); a finish then adds no line (its count is durable already), nor does the
    end."""
    spool = tmp_path / "spool"
    run = drive(*REQUEST, "--live-counts-dir", str(spool), "--live-hold", paused=0)
    wait_for(lambda: (run.live / "schedule_done").exists())
    run.acked("paused", 9)
    lines = run.lines()
    assert [(line["seq"], line["reason"]) for line in lines] == [(0, "init"), (4, "upscale"), (9, "hold")]
    check_line(run.live, lines[-1], spool, ncells=2, ngenes=2)
    run.set("counts 3", "where 4 0.00 0.00 4.00 4.00")
    wait_for(lambda: run.read("where.json").get("token") == 4)
    meta = run.read("where.json")
    assert meta["entries"] == 3
    check_export(run.live, meta, frames(run.live)[9])
    run.set(finish=1)
    run.acked("finishing", 9)
    assert run.exit_code() == 0 and run.lines() == lines


def test_the_hold_and_a_finish_count_a_spooled_state_again_durably(drive, tmp_path):
    """--live-counts every with a spool: the held frame's count went to the spool, so the hold counts it again into
    live/frames (reason hold, the same n and e). After continue the frames go on being counted, and a finish counts
    the last one again durably (reason finish)."""
    spool = tmp_path / "spool"
    run = drive(*EVERY, "--live-counts-dir", str(spool), "--live-hold", paused=0)
    wait_for(lambda: (run.live / "schedule_done").exists())
    lines = run.lines()
    assert [line["seq"] for line in lines] == [*range(10), 9]
    assert [line["reason"] for line in lines[-2:]] == ["every", "hold"]
    spooled, durable = (check_line(run.live, line, spool, ncells=2, ngenes=2) for line in lines[-2:])
    assert durable.n.tolist() == spooled.n.tolist() and durable.e.tolist() == spooled.e.tolist()
    run.set(**{"paused": 1, "continue": 1, "step": 1})          # one "continued" frame, then paused on it
    run.acked("paused", 10)
    assert frames(run.live)[10]["phase"] == "continued"
    assert [(line["seq"], line["reason"]) for line in run.lines()[len(lines):]] == [(10, "every")]
    run.set(finish=1)
    run.acked("finishing", 10)
    assert run.exit_code() == 0
    last = run.lines()[len(lines):]
    assert [(line["seq"], line["reason"]) for line in last] == [(10, "every"), (10, "finish")]
    assert check_line(run.live, last[-1], spool, ncells=2, ngenes=2).f is not None


def test_where_tokens_read_while_paused_are_served_at_once_and_four_exports_kept(drive):
    """Paused, each new where token is served at once from the paused state: where-N.bin.zst, then where.json naming
    it (seq: the paused frame), no counts line; proseg keeps the 4 newest exports. A token not above the last served
    one is not served again, and the tokens read while paused leave nothing for the frames after the resume."""
    run = drive(*REQUEST)
    run.set(step=1)
    run.acked("paused", 1)
    before = run.lines()
    for token in range(1, 6):
        run.set(f"where {token} 0.00 0.00 {token}.00 4.00")
        wait_for(lambda token=token: run.read("where.json").get("token") == token)
        check_export(run.live, run.read("where.json"), frames(run.live)[1])
    assert sorted(path.name for path in run.live.glob("where-*.bin.zst")) == \
        [f"where-{token}.bin.zst" for token in (2, 3, 4, 5)]
    last = run.read("where.json")
    run.set("where 5 0.00 0.00 2.00 2.00", "counts 1")      # the same where token, another box: served already
    stamps = set()

    def loops():
        stamps.add((run.live / "ack").stat().st_mtime_ns)
        return len(stamps) >= 4
    wait_for(loops)
    assert run.read("where.json") == last and run.lines() == before
    run.set(paused=0)
    assert run.exit_code() == 0 and run.read("where.json") == last
    assert [(line["seq"], line["reason"]) for line in run.lines()[len(before):]] == [(4, "upscale"), (9, "end")]


@pytest.mark.parametrize("mode, every", [("every", 1), ("every", 3), ("request", 2)])
def test_the_start_block_is_frame_0s_count_on_its_pattern(fake_binary, tmp_path, mode, every):
    """--live-counts-start [K] (K = 1 alone): the forced frames and every K-th counted frame carry the start block,
    frame 0's count on frame 0's pattern (at frame 0, start_e == e), as counts_file.start_csr reads it from any
    frame."""
    live, _, code = run_fake(fake_binary, tmp_path, "--live-counts", mode, "--live-counts-start",
                             *([] if every == 1 else [str(every)]), env={"FAKE_PROSEG_KEEP": "0.75"})
    assert code == 0
    info = json.loads((live / "info.json").read_text())["counts"]
    assert (info["mode"], info["start"], info["start_every"]) == (mode, True, every)
    lines = counts_lines(live)
    assert len(lines) == (10 if mode == "every" else 3)
    assert [line["has_start"] for line in lines] == \
        [line["reason"] in FORCED or (line["seq"] + 1) % every == 0 for line in lines]
    init = check_line(live, lines[0], ncells=2, ngenes=2)
    assert init.start_e.tolist() == init.e.tolist()
    for line in lines:
        counts = check_line(live, line, ncells=2, ngenes=2)
        if line["has_start"]:
            start = counts_file.start_csr(np, init, counts)
            assert [np.asarray(a).tolist() for a in (start.data, start.indices, start.indptr)] == \
                [init.e.tolist(), init.genes.tolist(), init.indptr.tolist()]


def test_an_older_proseg_live_keeps_no_count(fake_binary, old_fake_binary, tmp_path):
    """FAKE_PROSEG_OLD (conftest old_fake_binary, a proseg-live 3.2.0): its --help lacks --live-counts (the engine's
    capability check reads it there), it refuses those flags as clap refuses an argument it does not know, and a run
    without them ignores the counts and where lines: no counts.jsonl, no count or where file, no "counts" in
    info.json."""
    new = subprocess.run([str(fake_binary), "--help"], capture_output=True, text=True, timeout=60)
    old = subprocess.run([str(old_fake_binary), "--help"], capture_output=True, text=True, timeout=60)
    assert new.returncode == old.returncode == 0 and "--live-dir" in old.stdout
    assert all(flag in new.stdout for flag in ("--live-counts", "--live-counts-dir", "--live-counts-start"))
    assert "--live-counts" not in old.stdout + old.stderr
    folder = tmp_path / "old"
    (folder / "live").mkdir(parents=True)
    (folder / "live" / "control").write_text("counts 1\nwhere 2 0 0 4 4\n")
    live, _, code = run_fake(old_fake_binary, folder)
    assert code == 0 and len(frames(live)) == 10 and new_files(live) == []
    assert "counts" not in json.loads((live / "info.json").read_text())
    refused = subprocess.run([str(old_fake_binary), "--xenium", "transcripts.parquet", *REQUEST, "--live-dir",
                              str(tmp_path / "refused")], capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2 and "unexpected argument '--live-counts'" in refused.stderr
    assert not (tmp_path / "refused").exists()


def test_without_the_flags_proseg_live_keeps_no_count(fake_binary, tmp_path):
    """A run without --live-counts writes nothing new, whatever its control file says: no counts.jsonl, no count,
    sidecar or where file, no "counts" in info.json."""
    folder = tmp_path / "plain"
    (folder / "live").mkdir(parents=True)
    (folder / "live" / "control").write_text("counts 1\nwhere 2 10 20 17 24\n")
    world = write_world(tmp_path / "world.json", transcripts=COUNTED)
    live, _, code = run_fake(fake_binary, folder, env={"FAKE_PROSEG_WORLD": str(world)})
    assert code == 0 and len(frames(live)) == 10 and new_files(live) == []
    assert "counts" not in json.loads((live / "info.json").read_text())


@pytest.mark.parametrize("extra, env, words", [
    (["--live-counts-dir", "spool"], {}, "need --live-counts"),
    (["--live-counts-start"], {}, "need --live-counts"),
    ([*REQUEST, "--live-counts-start", "0"], {}, "at least 1"),
    (["--live-counts", "sometimes"], {}, "invalid choice"),
    (REQUEST, {"FAKE_PROSEG_KEEP": "0"}, "not a share"),
    (REQUEST, {"FAKE_PROSEG_KEEP": "1.5"}, "not a share"),
    (REQUEST, {"FAKE_PROSEG_KEEP": "most"}, "not a share")])
def test_what_it_refuses_before_it_starts(fake_binary, tmp_path, extra, env, words):
    """The command lines proseg-live 3.2.0.1 refuses (clap: a flag that needs --live-counts, a start every K below 1, a
    mode it does not know), and a keep that is not a share: exit 2 before any file."""
    environment = {k: v for k, v in os.environ.items() if k not in KNOBS} | env
    done = subprocess.run([str(fake_binary), "--xenium", "transcripts.parquet", *extra, "--live-dir",
                           str(tmp_path / "live")], capture_output=True, text=True, timeout=60, env=environment)
    assert done.returncode == 2 and words in done.stderr and not (tmp_path / "live").exists()


def test_the_counts_fixtures_write_what_the_apps_readers_read(tmp_path):
    """tests/counts_fixtures.py as other tests use it: counts_from_labels owns each record by the label of its observed
    voxel in its own layer (the worker's grid and layer rules; off the grid, no cell's), e and f at a keep;
    write_frame_counts leaves a PLC1 file (the spool or live/frames), its sidecar and one counts.jsonl line, read back
    by analysis/counts_file.py; write_where_export leaves where-N.bin.zst and where.json, read back by engine/held.py;
    where_span works in float32, as proseg does."""
    from sopa_companion.engine import held
    from sopa_companion.engine.transcript_index import RECORD
    grid = {"x0": 10.0, "y0": 20.0, "voxel": 2.0, "nx": 3, "ny": 2, "nz": 2}
    labels = np.array([[[1, 1, 0], [2, 2, 0]], [[1, 2, 2], [2, 0, 0]]], np.uint32).ravel()     # layer 0, layer 1
    found = np.array([(11.0, 21.0, 1.0, 0), (13.0, 21.0, 1.0, 1), (13.0, 21.0, 6.0, 1), (11.0, 23.0, 6.0, 0),
                      (15.0, 21.0, 6.0, 0), (17.0, 21.0, 1.0, 0), (9.9, 21.0, 1.0, 1)], RECORD)
    counts = CF.counts_from_labels(found, labels, grid, 0.0, 5.0, 3, 2, keep=0.75)
    assert {key: value.tolist() for key, value in counts.items()} == {
        "indptr": [0, 2, 4, 4], "genes": [0, 1, 0, 1], "n": [1, 1, 2, 1], "e": [0.75, 0.75, 1.5, 0.75],
        "f": [0, 0, 1, 0]}
    with pytest.raises(ValueError):
        CF.counts_of([1], [2], 1, 2)                # gene 2 of a run of 2
    with pytest.raises(ValueError):
        CF.counts_of([1], [0], 1, 1, keep=0.0)
    live, spool = tmp_path / "live", tmp_path / "spool"
    csr = {key: counts[key] for key in ("indptr", "genes", "n", "e")}
    first = CF.write_frame_counts(live, 5, 12, ncells=3, ngenes=2, **csr, f=counts["f"], reason="pause", spool=spool)
    assert first == {"seq": 5, "iter": 12, "file": str(spool / "000005.counts.zst"),
                     "counted_file": "frames/000005.counted.zst", "reason": "pause", "token": None, "where": None,
                     "nnz": 4, "sum_n": 5, "sum_e": 3.75, "sum_f": 1, "has_f": True, "has_start": False,
                     "saturated": 0, "bytes": (spool / "000005.counts.zst").stat().st_size,
                     "ms": {"extract": 0.0, "write": 0.0}}
    check_line(live, first, spool, ncells=3, ngenes=2)
    assert counts_file.read_counted(live / first["counted_file"], 3).tolist() == [1.5, 2.25, 0.0]
    meta = CF.write_where_export(live, 3, 6, 13, [10.5, 20.0, 13.9, 21.99], grid,
                                 {"i": [0, 1], "j": [0, 0], "k": [1, 0], "gene": [1, 0], "dest": [2, 0], "n": [1, 4]})
    assert meta == json.loads((live / "where.json").read_text()) == {
        "token": 3, "seq": 6, "iter": 13, "box": [10.5, 20.0, 13.9, 21.99], "margin_um": 24.0, "x0": 10.0, "y0": 20.0,
        "voxel": 2.0, "nx": 3, "ny": 2, "nz": 2, "i0": 0, "j0": 0, "i1": 1, "j1": 0, "file": "where-3.bin.zst",
        "entries": 2}
    export = held.read_where(meta, live / meta["file"])
    assert [getattr(export, key).tolist() for key in ("i", "j", "k", "gene", "dest", "n")] == \
        [[0, 1], [0, 0], [1, 0], [1, 0], [2, 0], [1, 4]]
    second = CF.write_frame_counts(live, 6, 13, ncells=3, ngenes=2, **csr, f=counts["f"], start_e=counts["e"],
                                   reason="hold", token=7, where=meta)
    assert (second["file"], second["token"], second["where"], second["has_start"]) == \
        ("frames/000006.counts.zst", 7, meta, True)
    assert counts_lines(live) == [first, second]
    start = counts_file.start_csr(np, read_line(live, first), check_line(live, second, ncells=3, ngenes=2))
    assert np.asarray(start.data).tolist() == [0.75, 0.75, 1.5, 0.75]
    # float32 as proseg: (0.3 - 0.1) / 0.1 is 2 in float32 (1 in float64); a box past the grid is clipped to it.
    assert CF.where_span([0.3, 0.3, 0.5, 0.5], {"x0": 0.1, "y0": 0.1, "voxel": 0.1, "nx": 10, "ny": 10}) == (2, 2, 4, 4)
    assert CF.where_span([-5.0, -5.0, 99.0, 99.0], grid) == (0, 0, 2, 1)


def test_the_fake_workers_umap_reads_proseg_count(tmp_path):
    """The fake analysis worker's UMAP summary (tests/fake_analysis_worker.py), as the real worker's: with the frame's
    counts file, basis "proseg", the live count's name and transcripts_in_cells = round(Σ e); without one, with one
    that is not there or that does not fit the run, today's summary (it never refuses a UMAP)."""
    here = Path(__file__).resolve().parent
    path = tmp_path / "000003.counts.zst"
    counts_file.write(path, ncells=3, ngenes=2, iteration=7, indptr=[0, 2, 3, 3], genes=[0, 1, 1], n=[3, 2, 4],
                      e=[2.25, 1.5, 3.0])
    env = {**os.environ, "PYTHONPATH": str(here) + os.pathsep + str(here.parents[1])}
    process = subprocess.Popen([sys.executable, "-u", str(here / "fake_analysis_worker.py")], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, env=env)
    asked = [{"counts_file": str(path)}, {}, {"counts_file": str(tmp_path / "gone.counts.zst")},
             {"counts_file": str(path), "ncells": 4}]
    try:
        assert json.loads(process.stdout.readline())["ready"]
        for seq, extra in enumerate(asked):
            process.stdin.write(json.dumps({"op": "analyze", "ops": ["umap"], "run": "r1", "seq": seq, "iter": 7,
                                            "ncells": 3, "out_cells": str(tmp_path / f"{seq}.cells.bin"),
                                            "out_json": str(tmp_path / f"{seq}.json"), **extra}) + "\n")
        process.stdin.flush()
        replies = [json.loads(process.stdout.readline()) for _ in asked]
    finally:
        process.stdin.close()
        process.wait(30)
    summary = replies[0]["summary"]
    assert (summary["basis"], summary["count"], summary["transcripts_in_cells"]) == \
        ("proseg", counts_file.LIVE_COUNT, 7)
    assert json.loads((tmp_path / "0.json").read_text()) == summary
    for reply in replies[1:]:
        assert reply["ok"] and not {"basis", "count", "transcripts_in_cells"} & set(reply["summary"])
