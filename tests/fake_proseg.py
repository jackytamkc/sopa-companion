"""A stand-in for native/bin/proseg-live that speaks the same files and control protocol (for engine tests).

It writes info.json, cells.txt, transcripts.bin.gz, one tiny frame per iteration of proseg's schedule (init, burn-in,
upscale, sampling, recorded, hill-climb), honours paused/step/min_ms/finish in the control file, and writes output
files when sampling ends. With --live-hold it holds at the end of the schedule (ack "paused", then schedule_done) until
"continue 1" (open-ended "continued" iterations, which pause and step as usual) or "finish 1". FAKE_PROSEG_FAIL=startup makes it exit 101 before
any frame unless --no-factorization is given (like upstream issue #160 on tiny crops). With --live-tiled it writes
tile containers (7 x 5 raster, 3-voxel tiles, a window for the "view" control line, window.json when the view moves
while paused). FAKE_PROSEG_OUTPUT=<proseg-output.zarr> is copied into the outputs instead of a stub.
FAKE_PROSEG_FINISH_S=<seconds> makes writing the results (after sampling_done) take that long.
FAKE_PROSEG_LOAD_S=<seconds> makes reading the input (before frame 0) take that long.

As proseg does, info.json says the threads asked for (--nthreads) and whether factorisation ran (not --no-factorization),
and the outputs asked for are written when sampling ends: --output-transcript-metadata NAME, proseg's per-transcript file
(native/proseg/src/schemas.rs: parquet when NAME ends in .parquet and --output-transcript-metadata-fmt is infer or
parquet, csv.gz otherwise), row i being record i of transcripts.bin.gz, assigned to the cell its destination voxel holds
in the last frame (none, and background, where that is no cell); --output-counts NAME, those assignments counted per
cell and gene (MatrixMarket, cells x genes).

Opt-in knobs for the tests of a setting's test (with none set, nothing above changes):
FAKE_PROSEG_WORLD=<world.json> replaces the tiny sample: {"cells": [Xenium ids], "genes": [names], "transcripts":
[[transcript_id, x, y, z, gene], ...], "nx", "ny", "nz", "labels": [nx * ny], "x0"?, "y0"?, "voxel"? (the grid's corner
and voxel side, µm: 0, 0 and 1 unless given), "moves"?: {"flag", "value", "rows": [row], "dx"}}. Every frame, from frame
0 on, holds `labels` in every layer, and gets its cell statistics as proseg-live writes them (frames/NNNNNN.cells.zst,
named by its record's "cells_file": native/proseg/src/live.rs cell_statistics). `moves` puts its rows `dx` columns away
from the voxel they were observed in, only in a run whose command line carries the tested value itself (--<flag> followed
by a number equal to `value`: a run that carries the flag at another value is not moved).
FAKE_PROSEG_FAIL_ONCE=<marker path> makes the one run that creates the marker (O_CREAT | O_EXCL) exit 101 before any
frame, unless it has --no-factorization (as FAKE_PROSEG_FAIL=startup): the engine's own retry of that run, and every
other run, go on.
FAKE_PROSEG_DROP_CELL=<Xenium id>[,<flag>] leaves that cell out of cells.txt (the labels after it one lower, its voxels
no cell's), in every run, or only in runs whose command line carries --<flag>.

Proseg's own count (proseg-live 3.2.0.2: native/proseg/src/live.rs; the files as INTERFACES §2 fixes them). Each cell's
count of a frame, per gene: n, the records whose destination it holds; e = n x keep (the averaged count); f = floor(n x
keep) (the sampled count). FAKE_PROSEG_KEEP=<share in (0, 1]> sets keep (1 by default: today's numbers); the cell
statistics' foreground and the saved counts (--output-counts) are f. With --live-counts request|every it keeps that
count as proseg-live does, through tests/counts_fixtures.py: info.json's "counts", counts.jsonl (one line per counted
frame, its commit point), frames/NNNNNN.counts.zst (PLC1; request, every and pause frames in the --live-counts-dir spool
when given) and frames/NNNNNN.counted.zst. Init and upscale frames are always counted; a pause counts its state (before
its ack) when it has no line, or again (with f, where pause counts go: the spool's file of that name replaced) when its
line has no f (a frame a token counted); the hold's, a finish's and the end's are made durable (counted again from the
same state when its line is in the spool or has no f); `every` counts every frame. A frame has at most three lines:
request, pause, then hold, finish or end. Control lines `counts N` and `where N x0 y0 x1 y1`: running, the next frame is
counted for them (with the where export); paused or held, a counts token is served by the paused state and a where token
at once from it (where-N.bin.zst, then where.json; the 4 newest kept). --live-counts-start [K] adds the start block
(frame 0's count on its pattern) to the forced frames and to every K-th counted frame. FAKE_PROSEG_OLD=1 makes it a
proseg-live 3.2.0 from before: its --help lacks --live-counts, it refuses those flags as clap does, and it ignores the
control lines.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import struct
import sys
import time

# Its numbers are tiny: numpy's maths library starts one thread here, not one per core of the shared computer (each
# start of one per core costs about 2 s of the computer's time).
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

OLD = os.environ.get("FAKE_PROSEG_OLD") == "1"
COUNT_FLAGS = ("--live-counts", "--live-counts-dir", "--live-counts-start")
if OLD:         # proseg-live 3.2.0 has none of them: clap refuses an argument it does not know
    for _token in sys.argv[1:]:
        if _token.split("=", 1)[0] in COUNT_FLAGS:
            print(f"error: unexpected argument '{_token.split('=', 1)[0]}' found", file=sys.stderr, flush=True)
            sys.exit(2)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("transcripts")
parser.add_argument("--xenium", action="store_true")
parser.add_argument("--live-dir")
parser.add_argument("--live-control")
parser.add_argument("--live-hold", action="store_true")
parser.add_argument("--live-tiled", action="store_true")
parser.add_argument("--output-path")
parser.add_argument("--burnin-samples", type=int, default=200)
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--recorded-samples", type=int, default=100)
parser.add_argument("--hillclimb", type=int, default=50)
parser.add_argument("--no-factorization", action="store_true")
parser.add_argument("--nthreads", type=int)
parser.add_argument("--cell-compactness", type=float)
parser.add_argument("--prior-weight", type=float)
parser.add_argument("--diffusion-probability", type=float)
parser.add_argument("--output-transcript-metadata")
parser.add_argument("--output-transcript-metadata-fmt", default="infer")
parser.add_argument("--output-counts")
if not OLD:
    parser.add_argument("--live-counts", choices=("request", "every"))
    parser.add_argument("--live-counts-dir")
    parser.add_argument("--live-counts-start", nargs="?", const=1, type=int)
args, _ = parser.parse_known_args()
COUNTS = None if OLD else args.live_counts         # "request" | "every" | None (proseg's own count not kept)
if not OLD:     # what clap's `requires` and the value's own check refuse
    if args.live_counts is None and (args.live_counts_dir is not None or args.live_counts_start is not None):
        parser.error("--live-counts-dir and --live-counts-start need --live-counts")
    if args.live_counts_start is not None and args.live_counts_start < 1:
        parser.error("--live-counts-start takes a whole number of at least 1")
    if COUNTS and not args.live_dir:
        parser.error("--live-counts needs --live-dir")
SPOOL = os.path.abspath(args.live_counts_dir) if COUNTS and args.live_counts_dir else None
START_EVERY = args.live_counts_start if COUNTS else None
try:
    KEEP = float(os.environ.get("FAKE_PROSEG_KEEP") or 1.0)
except ValueError:
    KEEP = float("nan")
if not 0 < KEEP <= 1:
    print(f"FAKE_PROSEG_KEEP={os.environ.get('FAKE_PROSEG_KEEP')!r} is not a share in (0, 1]", file=sys.stderr,
          flush=True)
    sys.exit(2)

PANIC = "thread 'main' panicked at src/sampler/featureselection.rs: index out of bounds"
if os.environ.get("FAKE_PROSEG_FAIL") == "startup" and not args.no_factorization:
    print(PANIC, flush=True)
    sys.exit(101)
if os.environ.get("FAKE_PROSEG_FAIL_ONCE") and not args.no_factorization:
    try:            # the first run to get here fails; the marker it leaves lets every later one through
        os.close(os.open(os.environ["FAKE_PROSEG_FAIL_ONCE"], os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        pass
    else:
        print(PANIC, flush=True)
        sys.exit(101)


def carries(flag: str, value=None) -> bool:
    """Whether this run's command line carries --flag; with `value`, followed by a number equal to it."""
    argv = sys.argv[1:]
    for n, token in enumerate(argv):
        if token == "--" + flag:
            if value is None:
                return True
            try:
                return float(argv[n + 1]) == float(value)
            except (IndexError, ValueError):
                return False
    return False


world = json.loads(Path(os.environ["FAKE_PROSEG_WORLD"]).read_text()) if os.environ.get("FAKE_PROSEG_WORLD") else None
live = Path(args.live_dir)
(live / "frames").mkdir(parents=True, exist_ok=True)
nx = ny = 4
nz = 2
TILE = 3
x0 = y0 = 0.0
voxel = 1.0
if args.live_tiled:
    nx, ny = 7, 5
if world:
    nx, ny, nz = int(world["nx"]), int(world["ny"]), int(world["nz"])
    x0, y0, voxel = float(world.get("x0", 0.0)), float(world.get("y0", 0.0)), float(world.get("voxel", 1.0))
names = list(world["cells"]) if world else ["aaa-1", "bbb-1"]
genes = list(world["genes"]) if world else ["G1", "G2"]
records = ([(float(x), float(y), float(z), int(g)) for _, x, y, z, g in world["transcripts"]] if world else
           [(1.0, 1.0, 0.2, 0), (2.5, 1.5, 0.7, 1), (3.0, 3.0, 0.4, 0)])
ids = [int(t[0]) for t in world["transcripts"]] if world else list(range(len(records)))
# FAKE_PROSEG_DROP_CELL: the dropped cell's label (None: no cell dropped in this run); the labels after it move down.
dropped = None
drop_name, _, drop_flag = os.environ.get("FAKE_PROSEG_DROP_CELL", "").partition(",")
if drop_name and drop_name in names and (not drop_flag or carries(drop_flag)):
    dropped = names.index(drop_name) + 1
    names.pop(dropped - 1)


def remap(label: int) -> int:
    return label if dropped is None or label < dropped else 0 if label == dropped else label - 1


world_labels = [remap(int(v)) for v in world["labels"]] if world else None


def labels_at(iteration: int) -> list[int]:
    """The consensus labels of a frame (and of each of its layers): the world's, else the tiny sample's moving ones."""
    if world_labels is not None:
        return world_labels
    labels = [(iteration + p) % 3 for p in range(nx * ny)]
    return labels if dropped is None else [remap(v) for v in labels]


def destinations(labels: list[int]):
    """Where each record ends, as proseg writes it: (x, y, z) = its observed place moved by its offset (whole voxels:
    the world's `moves`, in a run that carries the tested value; none otherwise), float32 as proseg's, and the label at
    that voxel in a frame of `labels` (every layer alike; 0 outside the grid). The observed voxel is the one the app
    reads (analysis/worker.py grid_columns, the depth clipped to the layers), in float32."""
    import numpy as np
    table = np.array(records, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
    ox, oy, oz = table["x"], table["y"], table["z"]
    dx = np.zeros(len(records), np.int64)
    moves = (world or {}).get("moves")
    if moves and carries(moves["flag"], moves["value"]):
        dx[np.asarray(moves["rows"], np.int64)] = int(moves["dx"])
    i = np.floor((ox - x0) / voxel).astype(np.int64) + dx
    j = np.floor((oy - y0) / voxel).astype(np.int64)
    k = np.clip(np.floor((oz - z_origin) / z_spacing), 0, nz - 1).astype(np.int64)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    plane = np.asarray(labels, np.int64)
    label = np.zeros(len(records), np.int64)
    label[inside] = plane[j[inside] * nx + i[inside]]
    x = (ox + dx.astype(np.float32) * np.float32(voxel)).astype(np.float32)
    return table, x, np.ascontiguousarray(oy), np.ascontiguousarray(oz), label


def sampled(label, gene):
    """Each cell's sampled count (proseg's foreground): Σ over its genes of f = floor(n x keep), n the records of that
    gene whose destination label (`label`) it holds."""
    import numpy as np
    n, g = len(names), len(genes)
    held = (label > 0) & (label <= n)
    pairs = np.bincount((label[held] - 1) * g + np.asarray(gene, np.int64)[held], minlength=n * g).reshape(n, g)
    return np.floor(pairs * KEEP).astype(np.int64).sum(axis=1)


def cell_statistics(labels: list[int], label, foreground) -> bytes:
    """A frame's per-cell statistics as proseg-live writes them (live.rs cell_statistics): u32 LE arrays of ncells each,
    back to back: area (2-D columns), voxels (every layer), assigned transcripts (each cell's, by `label`, every
    record's destination label), foreground (each cell's sampled count: `foreground`), changed (none: the labels never
    move), then the centres x and y as f32 bits (x0 + (mean column + 0.5) x voxel, in f32; NaN without area)."""
    import numpy as np
    n = len(names)
    plane = np.asarray(labels, np.int64)
    columns = np.arange(plane.size)
    held = (plane > 0) & (plane <= n)
    area = np.bincount(plane[held], minlength=n + 1)[1:]
    centre = []
    for along, origin in ((columns % nx, x0), (columns // nx, y0)):
        sums = np.bincount(plane[held], weights=along[held].astype(np.float64), minlength=n + 1)[1:]
        mean = (sums / np.maximum(area, 1)).astype(np.float32)
        centre.append(np.where(area > 0, np.float32(origin) + (mean + np.float32(0.5)) * np.float32(voxel),
                               np.float32("nan")))
    counted = np.bincount(np.asarray(label, np.int64), minlength=n + 1)[1:n + 1]
    return np.concatenate([area.astype("<u4"), (area * nz).astype("<u4"), counted.astype("<u4"),
                           np.asarray(foreground).astype("<u4"), np.zeros(n, "<u4"),
                           centre[0].astype("<f4").view("<u4"), centre[1].astype("<f4").view("<u4")]).tobytes()


def container(path, names, planes, window=None):
    """proseg-live's tile container (native/proseg/src/live.rs)."""
    import zstandard
    zstd = zstandard.ZstdCompressor(level=1)
    tiles_x, tiles_y = -(-nx // TILE), -(-ny // TILE)
    blobs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            for plane in planes:
                block = [plane[y * nx + x] for y in range(ty * TILE, min((ty + 1) * TILE, ny))
                         for x in range(tx * TILE, min((tx + 1) * TILE, nx))]
                blobs.append(zstd.compress(struct.pack(f"<{len(block)}I", *block)) if any(block) else b"")
    info = None
    if window:
        info, layers = window
        blobs.append(zstd.compress(struct.pack(f"<{len(layers)}I", *layers)))
    else:
        blobs.append(b"")
    header = json.dumps({"nx": nx, "ny": ny, "nz": nz, "tile": TILE, "tiles_x": tiles_x, "tiles_y": tiles_y,
                         "planes": names, "window": info}).encode()
    offset = 8 + len(header) + 12 * len(blobs)
    head = b"PLT1" + struct.pack("<I", len(header)) + header
    for blob in blobs:
        head += struct.pack("<QI", offset, len(blob))
        offset += len(blob)
    with open(path, "wb") as stream:
        stream.write(head + b"".join(blobs))


def window_of(view, labels):
    """The raster rectangle of a view (µm, voxel 1, origin 0) and its per-layer labels."""
    if not view:
        return None
    i0, j0 = max(0, int(view[0])), max(0, int(view[1]))
    i1, j1 = min(nx, int(view[2]) + 1), min(ny, int(view[3]) + 1)
    if i1 <= i0 or j1 <= j0:
        return None
    layer = [labels[y * nx + x] for y in range(j0, j1) for x in range(i0, i1)]
    return {"i0": i0, "j0": j0, "nx": i1 - i0, "ny": j1 - j0, "nz": nz}, layer * nz
plan = [{"phase": "init", "frames": 1}, {"phase": "burnin", "frames": args.burnin_samples}, {"phase": "upscale", "frames": 1},
        {"phase": "sampling", "frames": args.samples - args.recorded_samples},
        {"phase": "recorded", "frames": args.recorded_samples}]
if args.hillclimb > 0:
    plan.append({"phase": "hillclimb", "frames": args.hillclimb})
z_origin, z_spacing = 0, 1.0 / nz         # the normalised depth (0..1) in nz layers
info = {"schema": "proseg-live-1", "transcripts": len(records), "cells": len(names), "genes": len(genes), "layers": nz,
        "hold": args.live_hold,
        "schedule": {"init_iterations": 20, "burnin": args.burnin_samples, "samples": args.samples,
                     "recorded": args.recorded_samples, "hillclimb": args.hillclimb, "morphology_steps": 4000},
        "plan": plan, "z": {"thickness_um": 10, "grid_origin": z_origin, "grid_spacing": z_spacing},
        "threads": args.nthreads if args.nthreads is not None else (os.cpu_count() or 1),
        "factorization": not args.no_factorization}
if args.live_tiled:
    prior = [p % 3 for p in range(nx * ny)]
    container(live / "prior.tiles", ["label", "permille"], [prior, [800 if v else 0 for v in prior]])
    info.update(frame_format="tiles-1", tile=TILE, window_max=1600,
                prior={"file": "prior.tiles", "format": "tiles", "tile": TILE, "nx": nx, "ny": ny, "nz": 1, "x0": 0.0,
                       "y0": 0.0, "voxel": 1.0, "nucleus_prob": 0.8, "cell_prob": 0.5})
if not OLD:     # LiveWriter::new: no where export of an earlier run in this folder
    for _path in [live / "where.json", *live.glob("where-*.bin.zst")]:
        _path.unlink(missing_ok=True)
if COUNTS:
    # The counts files are written through the app's own writers (counts_file, held), from the tree this file is in
    # (whatever sopa_companion the venv's own paths would give).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sopa_companion.analysis import counts_file
    from sopa_companion.tests import counts_fixtures as fixtures
    info["counts"] = {"mode": COUNTS, "format": counts_file.MAGIC.decode(), "version": counts_file.VERSION,
                      "e_scale": counts_file.E_SCALE, "start": START_EVERY is not None, "start_every": START_EVERY,
                      "spool": SPOOL, "where": 1}
    if SPOOL:
        Path(SPOOL).mkdir(parents=True, exist_ok=True)
    (live / "counts.jsonl").write_text("")
(live / "info.json").write_text(json.dumps(info))
(live / "cells.txt").write_text("".join(f"{name}\n" for name in names))
(live / "genes.txt").write_text("".join(f"{name}\n" for name in genes))
with gzip.open(live / "transcripts.bin.gz", "wb") as stream:
    for x, y, z, g in records:
        stream.write(struct.pack("<fffI", x, y, z, g))
# A world's frames all hold the same labels: its records' destinations, its counts and its cell statistics are the same
# in every frame of this run (computed once).
world_frame = None
if world_labels is not None:
    import zstandard
    _table, _, _, _, ends = destinations(world_labels)
    _kept = sampled(ends, _table["gene"])
    world_frame = {"cells": zstandard.ZstdCompressor(level=1).compress(cell_statistics(world_labels, ends, _kept)),
                   "assigned": int((ends > 0).sum()), "foreground": int(_kept.sum()),
                   "visible": len({v for v in world_labels if v})}
index = open(live / "frames.jsonl", "a")
seq = 0
started = time.monotonic()


def frame(iteration, phase):
    global seq
    labels = labels_at(iteration)
    record = {}
    c = control() if args.live_tiled or COUNTS else None
    counts = {"assigned": 2, "foreground": 2, "transcripts": len(records), "cells": len(names), "visible": 2}
    if args.live_tiled:
        name = f"frames/{seq:06d}.tiles"
        view = c["view"]
        window = window_of(view, labels)
        container(live / name, ["consensus", "last_change"], [labels, [iteration] * nx * ny], window)
        record = {"format": "tiles", "tile": TILE, "view": view,
                  "window": {k: window[0][k] for k in ("i0", "j0", "nx", "ny")} if window else None}
    else:
        raw = struct.pack(f"<{(2 + nz) * nx * ny}I", *labels, *([iteration] * nx * ny), *(labels * nz))
        name = f"frames/{seq:06d}.bin.gz"
        with gzip.open(live / name, "wb", compresslevel=1 if world else 9) as stream:
            stream.write(raw)
    if world_frame is not None:
        stats = f"frames/{seq:06d}.cells.zst"
        (live / stats).write_bytes(world_frame["cells"])
        record["cells_file"] = stats
        counts = {"assigned": world_frame["assigned"], "foreground": world_frame["foreground"],
                  "transcripts": len(records), "cells": len(names), "visible": world_frame["visible"]}
    index.write(json.dumps({"seq": seq, "iter": iteration, "phase": phase, "file": name, "nx": nx, "ny": ny, "nz": nz,
                            "x0": x0, "y0": y0, "voxel": voxel, "temperature": 1.0, "loglik": -1.0 * iteration,
                            **counts, "changed": 1,
                            "t": time.monotonic() - started, "ms": {}, **record}) + "\n")
    index.flush()
    if COUNTS:
        count_frame(c, seq, iteration, phase)
    seq += 1


def control():
    values = {"paused": 0, "step": 0, "min_ms": 0, "finish": 0, "continue": 0, "view": None, "counts": 0, "where": None}
    try:
        for line in Path(args.live_control).read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in ("counts", "where"):
                found = token_line(parts) if COUNTS else None        # an older proseg-live ignores both
                if found is not None:
                    values[parts[0]] = found
            elif len(parts) == 2 and parts[0] in values:
                values[parts[0]] = int(parts[1])
            elif len(parts) == 5 and parts[0] == "view":
                values["view"] = [float(v) for v in parts[1:]]
    except OSError:
        pass
    return values


def token_line(parts):
    """A `counts N` or `where N x0 y0 x1 y1` line as proseg-live reads it (live.rs read_control): N a decimal u64 of at
    least 1, the box four finite numbers (µm, parsed to float32) with x1 > x0 and y1 > y0: N, or (N, box). None: the
    line is ignored."""
    size = 2 if parts[0] == "counts" else 6
    if len(parts) != size or not (parts[1].isascii() and parts[1].isdigit()) or not 0 < int(parts[1]) < 2 ** 64:
        return None
    if size == 2:
        return int(parts[1])
    import numpy as np
    try:
        values = [float(v) for v in parts[2:] if "_" not in v]
    except ValueError:
        return None
    with np.errstate(over="ignore"):
        box = [np.float32(v) for v in values]
    if len(box) != 4 or not all(np.isfinite(box)) or not (box[2] > box[0] and box[3] > box[1]):
        return None
    return int(parts[1]), [float(v) for v in box]


windows = {"view": None, "n": 0}


def refresh_window(view):
    """Paused with a moved view: the paused state's per-layer labels (window.json + window-N.bin.zst)."""
    import zstandard
    if not args.live_tiled or not view or view == windows["view"]:
        return
    windows["view"] = view
    found = window_of(view, labels_at(iteration))
    if not found:
        return
    info, layers = found
    windows["n"] += 1
    name = f"window-{windows['n']}.bin.zst"
    (live / name).write_bytes(zstandard.ZstdCompressor(level=1).compress(struct.pack(f"<{len(layers)}I", *layers)))
    (live / ".window.tmp").write_text(json.dumps({**info, "seq": seq - 1, "iter": iteration, "raster_nx": nx,
                                                  "raster_ny": ny, "x0": 0.0, "y0": 0.0, "voxel": 1.0, "file": name,
                                                  "view": view}))
    os.replace(live / ".window.tmp", live / "window.json")


# ---- proseg's own count (--live-counts) ---------------------------------------------------------------------------
FORCED = ("init", "upscale", "pause", "hold", "finish", "end")     # counted with f and (asked) the start block
SPOOLED = ("request", "every", "pause")                            # into the spool, when one is given
EXPORTS_KEPT = 4                                                   # where exports proseg leaves in the live folder
# lines: seq -> where its newest line's file is and whether it has f; counted: frames counted so far (in seq order);
# served: the last counts and where tokens served; init: frame 0's count (the start block); world: a world's count
# (the same in every frame).
counting = {"lines": {}, "counted": 0, "counts_served": 0, "where_served": 0, "init": None, "world": None}


def state_counts(at_iter: int):
    """The state at iteration `at_iter`: (its records, their destination labels, each cell's count)."""
    if counting["world"] is not None:
        return counting["world"]
    table, _, _, _, ends = destinations(labels_at(at_iter))
    found = (table, ends, fixtures.counts_of(ends, table["gene"], len(names), len(genes), KEEP))
    if world_labels is not None:
        counting["world"] = found
    return found


def count(at: int, at_iter: int, reason: str, token=None, where=None):
    """One job of proseg-live's counts writer, of the state of frame `at`: the where export a running where token asks
    for (the same state), then the frame's PLC1 file, sidecar and counts.jsonl line. f with a forced reason and in
    `every` mode; the start block with a forced reason and on every START_EVERY-th counted frame."""
    _, _, counts = state_counts(at_iter)
    if at not in counting["lines"]:
        counting["counted"] += 1
    if reason == "init":
        counting["init"] = counts
    has_f = reason in FORCED or COUNTS == "every"
    has_start = START_EVERY is not None and (reason in FORCED or counting["counted"] % START_EVERY == 0)
    exported = export_where(where, at, at_iter) if where is not None else None
    spool = SPOOL if reason in SPOOLED else None
    fixtures.write_frame_counts(live, at, at_iter, ncells=len(names), ngenes=len(genes), indptr=counts["indptr"],
                                genes=counts["genes"], n=counts["n"], e=counts["e"],
                                f=counts["f"] if has_f else None,
                                start_e=counting["init"]["e"] if has_start else None, reason=reason, token=token,
                                where=exported, spool=spool)
    counting["lines"][at] = {"spooled": spool is not None, "has_f": has_f}


def export_where(where, at: int, at_iter: int) -> dict:
    """A where export (token, box) of the state of frame `at`: one record per transcript whose observed voxel lies in
    the box's span (any layer), its destination label as dest, n = 1; then where.json. The 4 newest are kept."""
    import numpy as np
    token, box = where
    table, ends, _ = state_counts(at_iter)
    grid = {"x0": x0, "y0": y0, "voxel": voxel, "nx": nx, "ny": ny, "nz": nz}
    i0, j0, i1, j1 = fixtures.where_span(box, grid)
    i, j, k, _ = fixtures.observed_voxels(table, grid, z_origin, z_spacing)
    span = (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1)
    exported = fixtures.write_where_export(live, token, at, at_iter, box, grid, {
        "i": i[span], "j": j[span], "k": k[span], "gene": table["gene"][span].astype(np.int64), "dest": ends[span],
        "n": np.ones(int(span.sum()), np.int64)})
    found = sorted((int(path.name[6:-8]), path) for path in live.glob("where-*.bin.zst") if path.name[6:-8].isdigit())
    for _, path in found[:-EXPORTS_KEPT]:
        path.unlink(missing_ok=True)
    return exported


def pending(c) -> tuple:
    """The counts token and the (where token, box) of the control file `c` not served yet (None each)."""
    token = c["counts"] if c["counts"] > counting["counts_served"] else None
    where = c["where"] if c["where"] is not None and c["where"][0] > counting["where_served"] else None
    return token, where


def count_frame(c, at: int, at_iter: int, phase: str):
    """A new frame's count: init and upscale frames always; a counts or where token not served yet makes it a request
    frame (both then served, its line carrying them); in `every` mode every frame."""
    token, where = pending(c)
    reason = phase if phase in ("init", "upscale") else "request" if token or where else \
        "every" if COUNTS == "every" else None
    if reason is not None:
        count(at, at_iter, reason, token=token, where=where)
    if token:
        counting["counts_served"] = token
    if where:
        counting["where_served"] = where[0]


def settle(reason: str):
    """settle(state) before its ack: the state on screen (frame seq - 1) gets its counts line. A pause counts its state
    when it has no line, or again (with f, where pause counts go: the spool's file of that name replaced) when its line
    has no f (a frame a token counted: proseg-live 3.2.0.2); the hold, a finish and the end make it durable (in
    live/frames, with f), counting it again from the same state when its line is in the spool or has no f."""
    if not COUNTS or seq == 0:
        return
    line = counting["lines"].get(seq - 1)
    if line is None or (reason == "pause" and not line["has_f"]) or \
            (reason != "pause" and (line["spooled"] or not line["has_f"])):
        count(seq - 1, iteration, reason)


def serve_paused(c):
    """Tokens read while paused or held: a counts token is served by the paused state's line (settle wrote it), a
    where token at once, from the paused state. Neither is left for the frames after a resume."""
    if not COUNTS:
        return
    token, where = pending(c)
    if token:
        counting["counts_served"] = token
    if where:
        export_where(where, seq - 1, iteration)
        counting["where_served"] = where[0]


steps_used = 0


def ack(state):
    (live / ".ack.tmp").write_text(json.dumps({"state": state, "iter": iteration, "seq": seq - 1}))
    os.replace(live / ".ack.tmp", live / "ack")


def wait():
    global steps_used
    c = control()
    while c["paused"] and not c["finish"]:
        if c["step"] > steps_used:
            steps_used += 1
            break
        settle("pause")
        serve_paused(c)
        ack("paused")
        refresh_window(c["view"])
        time.sleep(0.02)
        c = control()
    if c["finish"]:
        settle("finish")
        ack("finishing")
        return False
    time.sleep(max(c["min_ms"], 5) / 1000)
    return True


time.sleep(float(os.environ.get("FAKE_PROSEG_LOAD_S", "0")))       # a whole sample's input takes a while to read
iteration = 0
frame(0, "init")
finished = False
for _ in range(args.burnin_samples):
    if not wait():
        finished = True
        break
    iteration += 1
    frame(iteration, "burnin")
if not finished:
    frame(iteration, "upscale")
for it in range(args.samples + args.hillclimb):
    if finished:
        break
    if not wait():
        finished = True
        break
    iteration += 1
    frame(iteration, "hillclimb" if it >= args.samples else "recorded" if it >= args.samples - args.recorded_samples
          else "sampling")


def hold():
    """--live-hold at the end of the schedule: paused whatever the control file says, until continue or finish; step tokens
    issued meanwhile are spent here (the token that comes with continue is left for wait: live.rs held_steps)."""
    global steps_used
    settle("hold")
    ack("paused")
    (live / ".schedule_done.tmp").write_text(json.dumps({"iter": iteration, "seq": seq - 1}))
    os.replace(live / ".schedule_done.tmp", live / "schedule_done")
    while True:
        c = control()
        if c["finish"]:
            settle("finish")
            ack("finishing")
            return False
        if c["continue"]:
            return True
        steps_used = max(steps_used, c["step"])
        serve_paused(c)
        refresh_window(c["view"])
        time.sleep(0.02)


def write_metadata(path: Path, fmt: str, last: list[int]):
    """proseg's per-transcript file (native/proseg/src/schemas.rs, output.rs write_transcript_metadata): one row per
    record of transcripts.bin.gz, in its order; its gene a dictionary over genes.txt; its assignment the cell at its
    destination in the last frame (`last`), null and background where that is no cell."""
    import numpy as np
    table, x, y, z, label = destinations(last)
    assignment = [int(v) - 1 if v > 0 else None for v in label.tolist()]
    if path.name.endswith(".parquet") and fmt in ("infer", "parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = pa.schema([("transcript_id", pa.uint64(), True), ("qv", pa.float32(), True),
                            ("x", pa.float32(), False), ("y", pa.float32(), False), ("z", pa.float32(), False),
                            ("observed_x", pa.float32(), False), ("observed_y", pa.float32(), False),
                            ("observed_z", pa.float32(), False),
                            ("gene", pa.dictionary(pa.int32(), pa.utf8()), False),
                            ("assignment", pa.uint32(), True), ("background", pa.bool_(), False)])
        columns = [pa.array(np.asarray(ids, np.uint64)), pa.array(np.full(len(records), 40.0, np.float32)),
                   pa.array(x), pa.array(y), pa.array(z),
                   *(pa.array(np.ascontiguousarray(table[axis])) for axis in ("x", "y", "z")),
                   pa.DictionaryArray.from_arrays(pa.array(table["gene"].astype(np.int32)), pa.array(genes, pa.utf8())),
                   pa.array(assignment, pa.uint32()), pa.array(label == 0)]
        pq.write_table(pa.Table.from_arrays(columns, schema=schema), path, compression="zstd")
        return
    import csv
    import io
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["transcript_id", "qv", "x", "y", "z", "observed_x", "observed_y", "observed_z", "gene",
                     "assignment", "background"])
    for row, (ox, oy, oz, g) in enumerate(table.tolist()):
        writer.writerow([ids[row], "40", *(str(np.float32(v)) for v in (x[row], y[row], z[row], ox, oy, oz)), genes[g],
                         "" if assignment[row] is None else assignment[row], "true" if label[row] == 0 else "false"])
    path.write_bytes(gzip.compress(text.getvalue().encode()))


def write_counts(path: Path, last: list[int]):
    """The cells x genes counts proseg saves (MatrixMarket coordinate, 1-based), gzip for a .gz name: its sampled count
    of the last frame, f = floor(n x keep) of the assignments (all of them at keep 1), zeros left out."""
    table, _, _, _, label = destinations(last)
    counts = {}
    for cell, g in zip(label.tolist(), table["gene"].tolist()):
        if cell > 0:
            counts[(cell, g + 1)] = counts.get((cell, g + 1), 0) + 1
    if KEEP != 1.0:
        counts = {key: int(v * KEEP) for key, v in counts.items() if int(v * KEEP) > 0}
    text = (f"%%MatrixMarket matrix coordinate integer general\n{len(names)} {len(genes)} {len(counts)}\n" +
            "".join(f"{c} {g} {v}\n" for (c, g), v in sorted(counts.items())))
    path.write_bytes(gzip.compress(text.encode()) if path.name.endswith(".gz") else text.encode())


if args.live_hold and not finished and hold():
    while wait():
        iteration += 1
        frame(iteration, "continued")
if not finished and not args.live_hold:
    settle("end")           # the run ends without a finish (with --live-hold it always ends with one)
(live / "sampling_done").write_text("1\n")
time.sleep(float(os.environ.get("FAKE_PROSEG_FINISH_S", "0")))     # writing a whole sample's results takes a while
out = Path(args.output_path)
if os.environ.get("FAKE_PROSEG_OUTPUT"):
    import shutil
    shutil.copytree(os.environ["FAKE_PROSEG_OUTPUT"], out / "proseg-output.zarr", dirs_exist_ok=True)
else:
    (out / "proseg-output.zarr").mkdir(parents=True, exist_ok=True)
    (out / "proseg-output.zarr" / "zarr.json").write_text("{}")
(out / "cell-polygons.geojson.gz").write_bytes(gzip.compress(b"{}"))
if args.output_transcript_metadata:
    write_metadata(out / args.output_transcript_metadata, args.output_transcript_metadata_fmt, labels_at(iteration))
if args.output_counts:
    write_counts(out / args.output_counts, labels_at(iteration))
print("done", flush=True)
