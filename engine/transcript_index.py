"""A spatial index over a run's modelled transcripts (live/transcripts.bin.gz: 16-byte records x, y, z, gene), so the
viewer can load just the transcripts on screen of a whole-sample run (tens of millions in total).

Built once per run on first use: records sorted by 100 µm cell of a grid, written uncompressed next to the original
(transcripts.sorted.bin) with the offset of every grid cell (transcripts.grid.npy)."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import threading

import numpy as np

RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("gene", "<u4")])
CELL_UM = 100.0
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


class TranscriptIndex:
    def __init__(self, live: Path):
        self.live = Path(live)
        self.sorted_path = self.live / "transcripts.sorted.bin"
        self.grid_path = self.live / "transcripts.grid.json"
        with _guard:
            lock = _locks.setdefault(str(self.live), threading.Lock())
        with lock:
            if not (self.sorted_path.exists() and self.grid_path.exists()):
                self._build()
        self.grid = json.loads(self.grid_path.read_text())
        self.offsets = np.asarray(self.grid["offsets"], np.int64).reshape(self.grid["ny"], self.grid["nx"] + 1)
        self.records = np.memmap(self.sorted_path, dtype=RECORD, mode="r")

    def close(self):
        """Let go of the mapped records (a whole sample's are ~0.5 GB): unmapped as soon as a query still running
        is done with them. A mapped file outlives its deletion on a FUSE-mounted disk (.fuse_hidden*) and keeps the
        run's folder."""
        self.records = None

    def _build(self):
        source = self.live / "transcripts.bin.gz"
        if not source.exists():
            raise FileNotFoundError(source)
        with gzip.open(source, "rb") as stream:
            data = np.frombuffer(stream.read(), dtype=RECORD)
        x, y = data["x"], data["y"]
        x0, y0 = float(np.floor(x.min())) if data.size else 0.0, float(np.floor(y.min())) if data.size else 0.0
        nx = int((float(x.max()) - x0) // CELL_UM) + 1 if data.size else 1
        ny = int((float(y.max()) - y0) // CELL_UM) + 1 if data.size else 1
        cx = np.clip(((x - x0) // CELL_UM).astype(np.int64), 0, nx - 1)
        cy = np.clip(((y - y0) // CELL_UM).astype(np.int64), 0, ny - 1)
        key = cy * nx + cx
        order = np.argsort(key, kind="stable")
        counts = np.bincount(key, minlength=nx * ny).reshape(ny, nx)
        # Row-wise cumulative offsets: offsets[r, c] = first record of cell (r, c); offsets[r, nx] = end of row r.
        starts = np.concatenate([[0], np.cumsum(counts.ravel())])
        offsets = np.empty((ny, nx + 1), np.int64)
        for r in range(ny):
            offsets[r] = starts[r * nx:r * nx + nx + 1]
        temporary = self.sorted_path.with_name(f".{self.sorted_path.name}.{os.getpid()}")
        data[order].tofile(temporary)
        os.replace(temporary, self.sorted_path)
        grid = {"x0": x0, "y0": y0, "cell": CELL_UM, "nx": nx, "ny": ny, "count": int(data.size),
                "offsets": offsets.ravel().tolist()}
        temporary = self.grid_path.with_name(f".{self.grid_path.name}.{os.getpid()}")
        temporary.write_text(json.dumps(grid))
        os.replace(temporary, self.grid_path)

    def query(self, box, limit: int) -> tuple[np.ndarray | None, int]:
        """Records inside the box (x0, y0, x1, y1), or (None, count) when more than `limit` would be returned."""
        records = self.records             # held for this query only, even if close() comes meanwhile
        if records is None:
            raise FileNotFoundError(self.sorted_path)
        g = self.grid
        bx0, by0, bx1, by1 = box
        c0 = max(0, int((bx0 - g["x0"]) // g["cell"]))
        c1 = min(g["nx"] - 1, int((bx1 - g["x0"]) // g["cell"]))
        r0 = max(0, int((by0 - g["y0"]) // g["cell"]))
        r1 = min(g["ny"] - 1, int((by1 - g["y0"]) // g["cell"]))
        if c1 < c0 or r1 < r0:
            return np.zeros(0, RECORD), 0
        ranges = [(int(self.offsets[r, c0]), int(self.offsets[r, c1 + 1])) for r in range(r0, r1 + 1)]
        found, count = [], 0
        for a, b in ranges:          # exact in-box counting, in bounded chunks; stop once over the limit
            for start in range(a, b, 1 << 21):
                chunk = np.asarray(records[start:min(b, start + (1 << 21))])
                inside = chunk[(chunk["x"] >= bx0) & (chunk["x"] < bx1) & (chunk["y"] >= by0) & (chunk["y"] < by1)]
                count += inside.size
                if count > limit:
                    return None, count       # at least this many
                if inside.size:
                    found.append(inside)
        return (np.concatenate(found) if found else np.zeros(0, RECORD)), count
