"""An installed proseg-live wheel, run: python native/package/smoke.py, with the Python whose environment has the wheel
installed, and numpy and pyarrow beside it (the proseg-live workflow's smoke jobs).

The program is the one in this Python's scripts folder, where the wheel puts it and where the app looks first for an
installed copy (locate.proseg_live). It says the version of proseg it was built from, offers the live options the app
passes, and runs a short live run on 4,000 transcripts in Xenium's columns, counting every frame (--live-counts every
--live-counts-start): without --live-hold it ends by itself, and a control file that is not there reads as no command.
It must finish within 120 s and write its live folder's info.json, at least one frame, and a counts line for each
frame whose count file is there, the first counted at the init frame. Everything is written in a temporary folder.
"""
from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LIVE_OPTIONS = ("--live-dir", "--live-control", "--live-hold", "--live-tiled", "--live-counts", "--live-counts-dir",
                "--live-counts-start")


def check(passed: bool, problem) -> None:
    """Stop with the problem when a check fails (asserts would be skipped under python -O)."""
    if not passed:
        raise SystemExit(f"smoke.py: {problem}")


def program() -> Path:
    path = Path(sysconfig.get_path("scripts")) / ("proseg-live.exe" if os.name == "nt" else "proseg-live")
    if not path.is_file():
        raise SystemExit(f"smoke.py: {path} is not there: install the proseg-live wheel into this Python's environment")
    return path


def transcripts(path: Path, n: int = 4000) -> None:
    """n transcripts on a 400 x 300 µm field with the columns of a Xenium bundle's transcripts.parquet, seven genes, and
    70 % of them in square cells of 40 µm (the rest unassigned)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 400, n).astype(np.float32)
    y = rng.uniform(0, 300, n).astype(np.float32)
    cells = [f"c{int(a // 40)}{int(b // 40)}-1" if rng.random() < .7 else "UNASSIGNED" for a, b in zip(x, y)]
    pq.write_table(pa.table({
        "transcript_id": pa.array(np.arange(n, dtype=np.uint64)),
        "cell_id": pa.array(cells, pa.large_string()),
        "overlaps_nucleus": pa.array(rng.integers(0, 2, n).astype(np.uint8)),
        "feature_name": pa.array([f"G{i % 7}" for i in range(n)], pa.large_string()),
        "x_location": pa.array(x), "y_location": pa.array(y),
        "z_location": pa.array(rng.uniform(5, 15, n).astype(np.float32)),
        "qv": pa.array(np.full(n, 40, np.float32)),
        "fov_name": pa.array(["A1"] * n, pa.large_string()),
        "nucleus_distance": pa.array(np.zeros(n, np.float32)),
        "codeword_index": pa.array(np.zeros(n, np.int32)),
        "codeword_category": pa.array(["predesigned_gene"] * n, pa.large_string()),
        "is_gene": pa.array(np.ones(n, bool)),
    }), path)


def main() -> int:
    proseg = program()
    expected = "proseg " + ".".join(version("proseg-live").split(".")[:3])     # 3.2.0.1 is still proseg 3.2.0
    said = subprocess.run([str(proseg), "--version"], capture_output=True, text=True, timeout=60)
    check(said.returncode == 0 and said.stdout.strip() == expected,
          f"{proseg} --version said {said.stdout.strip()!r} {said.stderr.strip()!r}, not {expected!r}")
    helped = subprocess.run([str(proseg), "--help"], capture_output=True, text=True, timeout=60)
    check(helped.returncode == 0 and all(option in helped.stdout for option in LIVE_OPTIONS),
          f"{proseg} --help does not offer {', '.join(LIVE_OPTIONS)}:\n{helped.stdout[-2000:]}{helped.stderr[-2000:]}")
    with tempfile.TemporaryDirectory(prefix="proseg-live-smoke-") as folder:
        folder = Path(folder)
        transcripts(folder / "t.parquet")
        out = folder / "out"
        out.mkdir()                     # proseg wants its output folder made first, as the app makes each run's
        run = subprocess.run([str(proseg), "--xenium", str(folder / "t.parquet"), "--burnin-samples", "4", "--samples", "4",
                              "--hillclimb", "2", "--recorded-samples", "2", "--nthreads", "2", "--no-factorization",
                              "--output-path", str(out), "--overwrite", "--output-cell-polygons",
                              "cell-polygons.geojson.gz", "--live-dir", str(out / "live"), "--live-control",
                              str(out / "live" / "control"), "--live-counts", "every", "--live-counts-start"],
                             cwd=folder, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        check(run.returncode == 0, f"the live run exited {run.returncode}:\n{run.stderr[-3000:]}")
        frames = sorted((out / "live" / "frames").glob("*.bin.zst"))
        check((out / "live" / "info.json").is_file() and bool(frames),
              f"the live run wrote no info.json or no frame: {sorted(p.name for p in (out / 'live').rglob('*'))}")
        info = json.loads((out / "live" / "info.json").read_text())
        counted = [json.loads(line) for line in (out / "live" / "counts.jsonl").read_text().splitlines()]
        check((info.get("counts") or {}).get("format") == "PLC1" and len(counted) == len(frames)
              and counted[0]["reason"] == "init" and all((out / "live" / line["file"]).is_file() for line in counted),
              f"the live run's counts: info {info.get('counts')}, {len(counted)} lines for {len(frames)} frames")
    print(f"smoke.py: {proseg}: {expected}, live options offered, a live run wrote {len(frames)} frames and their counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
