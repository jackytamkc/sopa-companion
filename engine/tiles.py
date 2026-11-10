"""proseg-live's tiled containers (frames/NNNNNN.tiles, prior.tiles; layout in native/proseg/src/live.rs): compose
any box of a 2-D plane at any stride, and read the per-layer window. Decompressed tiles are cached (LRU by bytes)
because the viewer asks for neighbouring and overlapping boxes as it pans and zooms."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import threading

import numpy as np

MAGIC = b"PLT1"


class TileCache:
    def __init__(self, limit_bytes: int = 1 << 30):
        self.limit, self.used = limit_bytes, 0
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key, value: np.ndarray):
        with self.lock:
            if key in self.items:
                return
            self.items[key] = value
            self.used += value.nbytes + 512      # bookkeeping counts too: no entry is free
            while self.used > self.limit and self.items:
                _, old = self.items.popitem(last=False)
                self.used -= old.nbytes + 512

    def forget(self, folder: str):
        """Drop the tiles of every container under `folder` (a deleted run)."""
        prefix = folder.rstrip(os.sep) + os.sep
        with self.lock:
            for key in [k for k in self.items if str(k[0]).startswith(prefix)]:
                self.used -= self.items.pop(key).nbytes + 512


CACHE = TileCache()


def _decompress(data: bytes) -> bytes:
    import zstandard
    return zstandard.ZstdDecompressor().decompressobj().decompress(data)


class Container:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as stream:
            head = stream.read(8)
            if len(head) != 8 or head[:4] != MAGIC:
                raise ValueError(f"{self.path.name} is not a proseg-live tile container")
            size = int.from_bytes(head[4:8], "little")
            self.header = json.loads(stream.read(size))
            h = self.header
            self.nplanes = len(h["planes"])
            count = h["tiles_x"] * h["tiles_y"] * self.nplanes + 1
            raw = stream.read(count * 12)
        index = np.frombuffer(raw, dtype=np.dtype([("offset", "<u8"), ("length", "<u4")]))
        if index.size != count:
            raise ValueError(f"{self.path.name} is truncated")
        self.index = index
        self.nx, self.ny, self.tile = h["nx"], h["ny"], h["tile"]
        self.key = (str(self.path), os.stat(self.path).st_mtime_ns)

    def plane_index(self, name: str) -> int:
        try:
            return self.header["planes"].index(name)
        except ValueError:
            raise KeyError(f"No plane {name}") from None

    def _blob(self, entry: int) -> bytes | None:
        offset, length = int(self.index[entry]["offset"]), int(self.index[entry]["length"])
        if length == 0:
            return None
        with open(self.path, "rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        if len(data) != length:
            raise ValueError(f"{self.path.name} is truncated")
        return _decompress(data)

    def tile_array(self, ty: int, tx: int, plane: int) -> np.ndarray | None:
        """One tile as (h, w) uint32, None when it is all zero (known from the index: nothing to cache)."""
        entry = (ty * self.header["tiles_x"] + tx) * self.nplanes + plane
        if int(self.index[entry]["length"]) == 0:
            return None
        key = (*self.key, ty, tx, plane)
        cached = CACHE.get(key)
        if cached is not None:
            return cached
        h = min(self.tile, self.ny - ty * self.tile)
        w = min(self.tile, self.nx - tx * self.tile)
        array = np.frombuffer(self._blob(entry), "<u4").reshape(h, w)
        CACHE.put(key, array)
        return array

    def compose(self, plane: str | int, c0: int, r0: int, c1: int, r1: int, stride: int = 1) -> np.ndarray:
        """Raster columns c0..c1 (exclusive) and rows r0..r1 sampled every `stride`: out[y, x] = plane[r0 + y*stride,
        c0 + x*stride]. Clipped to the raster (outside = 0)."""
        p = plane if isinstance(plane, int) else self.plane_index(plane)
        stride = max(1, int(stride))
        out_w = max(0, (c1 - c0 + stride - 1) // stride)
        out_h = max(0, (r1 - r0 + stride - 1) // stride)
        out = np.zeros((out_h, out_w), np.uint32)
        if out_w == 0 or out_h == 0:
            return out
        T = self.tile
        cc0, cc1 = max(c0, 0), min(c1, self.nx)
        rr0, rr1 = max(r0, 0), min(r1, self.ny)
        if cc0 >= cc1 or rr0 >= rr1:
            return out
        for ty in range(rr0 // T, (rr1 - 1) // T + 1):
            for tx in range(cc0 // T, (cc1 - 1) // T + 1):
                tile = self.tile_array(ty, tx, p)
                if tile is None:
                    continue
                # First sampled row/column inside this tile, in raster coordinates.
                row_lo, row_hi = max(rr0, ty * T), min(rr1, (ty + 1) * T)
                col_lo, col_hi = max(cc0, tx * T), min(cc1, (tx + 1) * T)
                first_row = row_lo + (-(row_lo - r0)) % stride
                first_col = col_lo + (-(col_lo - c0)) % stride
                if first_row >= row_hi or first_col >= col_hi:
                    continue
                part = tile[first_row - ty * T:row_hi - ty * T:stride, first_col - tx * T:col_hi - tx * T:stride]
                oy, ox = (first_row - r0) // stride, (first_col - c0) // stride
                out[oy:oy + part.shape[0], ox:ox + part.shape[1]] = part
        return out

    def full(self, plane: str | int) -> np.ndarray:
        return self.compose(plane, 0, 0, self.nx, self.ny, 1)

    def window(self):
        """(description, layers (nz, ny, nx) uint32) of the per-layer window, or None."""
        info = self.header.get("window")
        if not info:
            return None
        data = self._blob(self.index.size - 1)
        shape = (info["nz"], info["ny"], info["nx"])
        layers = np.zeros(shape, np.uint32) if data is None else np.frombuffer(data, "<u4").reshape(shape)
        return info, layers
