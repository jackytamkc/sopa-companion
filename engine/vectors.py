"""Prior vectors: the outlines of a run's prior shapes (cells, nuclei) around its region as ONE little-endian binary
that the viewer maps into typed arrays without parsing it (layout: docs/ARCHITECTURE.md, "Prior vectors").

The JSON form of the same polygons as nested lists (Runs.xenium_boundaries; once served too, now the tests' reference)
is tens to hundreds of MB for a whole sample, built by per-polygon Python while holding the GIL (engine stalls up to
1.5 s) and was parsed on the page's main thread (+100 MB of heap). This is built with numpy only (a whole sample: ~20 MB
in ~0.1 s), once per run, cached beside it zstd-compressed and sent as that file. It carries exactly what the JSON
carries: the same polygons in the same order, the same float32 coordinates (rounded to 0.01 µm as ZarrSamples.shapes
rounds them) and the same ids.
"""
from __future__ import annotations

import json

import numpy as np

MAGIC = b"PLPV"                       # Proseg Live Prior Vectors: a file format's name, kept with the app's new name
VERSION = 1
LAYERS = ("cells", "nuclei")          # in this order, always both (absent: flags 0, no sections)
FILE = f"vectors.v{VERSION}.bin.zst"  # in the run's xenium/ folder
# A run's prior shapes: the outlines whose bounds meet its box grown by this margin, at most this many per layer (by
# the run's mode). Both forms (the JSON and this binary) are cached per run and the binary is served immutable for its
# URL, so a change here reaches existing runs only with a new VERSION.
PRIOR_MARGIN_UM = 20.0
PRIOR_LIMITS = {"detail": 60_000, "tiled": 400_000}


def hits(bounds: np.ndarray, box, limit: int) -> tuple[np.ndarray, int]:
    """The polygons whose float32 bounds (parts x 4: x0, y0, x1, y1) meet `box` (all of them without one), in their
    stored order, the first `limit`; and how many met it. Every prior source selects with this: the picker's worker
    (shape_query.encode), the run's JSON (ZarrSamples.shapes) and its binary (select)."""
    if box is None:
        found = np.arange(len(bounds))
    else:
        x0, y0, x1, y1 = box
        found = np.flatnonzero((bounds[:, 2] >= x0) & (bounds[:, 0] <= x1) & (bounds[:, 3] >= y0) & (bounds[:, 1] <= y1))
    return found[:max(0, int(limit))], int(found.size)


def gather(outlines: dict, found: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The vertices of the polygons `found` (hits) in one gather, rounded to 0.01 µm in float32 (the numbers every prior
    source carries), and edges (int64, len(found) + 1): polygon k is xy[edges[k]:edges[k + 1]]."""
    offsets = outlines["offsets"]
    start = offsets[found].astype(np.int64)
    counts = offsets[found + 1].astype(np.int64) - start
    edges = np.zeros(found.size + 1, np.int64)
    np.cumsum(counts, out=edges[1:])
    xy = np.round(outlines["coords"][np.repeat(start - edges[:-1], counts) + np.arange(edges[-1])], 2)
    return edges, xy


def _bbox(xy: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """x0, y0, x1, y1 of each polygon from its own (rounded) vertices, exactly what a viewer would compute from the
    JSON; a polygon without vertices gets an inverted infinite box that meets nothing."""
    box = np.empty((len(offsets) - 1, 4), np.float32)
    box[:, :2], box[:, 2:] = np.inf, -np.inf
    full = np.flatnonzero(offsets[1:] > offsets[:-1])
    if full.size:
        # reduceat over the starts of the non-empty polygons only: an empty one between two adds no vertex.
        starts = offsets[full].astype(np.intp)
        for k, (column, reduce) in enumerate(((0, np.minimum), (1, np.minimum), (0, np.maximum), (1, np.maximum))):
            box[full, k] = reduce.reduceat(xy[:, column], starts)
    return box


def _layer(offsets, xy, ids, names: list | None) -> dict:
    """A layer as encode takes it; ids[i] indexes names, or is already numeric when names is None."""
    offsets = np.asarray(offsets)
    if offsets[-1] >= 2 ** 32:
        raise ValueError("too many vertices for u32 offsets")
    offsets = offsets.astype(np.uint32)
    xy = np.ascontiguousarray(xy, np.float32).reshape(-1, 2)
    return {"offsets": offsets, "xy": xy, "bbox": _bbox(xy, offsets), "ids": np.asarray(ids, np.uint32),
            "names": list(names) if names is not None else None}


def select(outlines: dict, box, limit: int) -> dict:
    """The outlines whose bounding box meets `box`, in their stored order, at most `limit`: the polygons, order and
    rounding of ZarrSamples.shapes (which builds the JSON), gathered without a Python loop. `outlines` is
    ZarrSamples._outlines(): bounds (parts x 4), offsets (parts + 1), owner (parts), coords (vertices x 2), ids."""
    found, _ = hits(outlines["bounds"], box, limit)
    edges, xy = gather(outlines, found)
    owners, ids = np.unique(outlines["owner"][found], return_inverse=True)
    return _layer(edges, xy, ids, [outlines["ids"][k] for k in owners.tolist()])


def from_polygons(polygons: list[dict]) -> dict:
    """A layer from the JSON form [{id, poly: [[x, y], ...]}] (Xenium bundle runs and tests: small)."""
    offsets = np.zeros(len(polygons) + 1, np.int64)
    offsets[1:] = np.cumsum([len(p["poly"]) for p in polygons], dtype=np.int64)
    xy = np.array([v for p in polygons for v in p["poly"]], np.float32).reshape(-1, 2)
    names: dict = {}
    ids = [names.setdefault(p["id"], len(names)) for p in polygons]
    return _layer(offsets, xy, ids, list(names))


def _numeric(names: list) -> bool:
    """Whether the ids are all integers a u32 holds: then the viewer's PolygonSet keeps them as its numeric ids (as
    polygons.mjs polygonSetFromLegacy does with the JSON's), otherwise its ids are null."""
    return bool(names) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer()
                               and 0 <= v < 2 ** 32 for v in names)


def encode(layers: dict, meta: dict | None = None) -> bytes:
    """The binary of {cells, nuclei: a layer (select, from_polygons) or None}; `meta` (a JSON object) is merged into
    what the viewer decodes, e.g. source: the shapes keys, as the JSON carried them."""
    described = json.dumps(meta or {}, separators=(",", ":")).encode()
    records, sections = [], []
    for name in LAYERS:
        layer = layers.get(name)
        if layer is None:
            records += [0, 0, 0, 0]
            continue
        ids, names = layer["ids"], b""
        if layer["names"] is None:
            pass                             # stain ids already refer to the whole set, including cells outside the view
        elif _numeric(layer["names"]):        # the ids themselves (flag 2), no names table
            ids = np.array([int(v) for v in layer["names"]], np.uint32)[ids]
        else:                                 # an index into the names table, as the JSON carried them
            names = json.dumps(layer["names"], separators=(",", ":")).encode()
        records += [1 | (0 if names else 2), len(ids), len(layer["xy"]), len(names)]
        # Flat byte views of the arrays: join copies each once, straight into the result.
        sections += [np.ascontiguousarray(array, dtype).reshape(-1).view(np.uint8)
                     for array, dtype in ((layer["offsets"], "<u4"), (layer["xy"], "<f4"), (layer["bbox"], "<f4"), (ids, "<u4"))]
        sections.append(names + bytes(-len(names) % 4))
    size = 20 + 16 * len(LAYERS) + len(described)
    size += -size % 4                                    # every section starts 4-byte aligned: typed-array views
    header = MAGIC + np.array([VERSION, size, len(LAYERS), *records, len(described)], "<u4").tobytes() + described
    return b"".join([header.ljust(size, b"\0"), *sections])


def decode(data) -> dict:
    """The reverse of encode (tests, tools): {**meta, cells, nuclei: {offsets, xy (V x 2), bbox (n x 4), ids, names}
    or None}, the arrays viewing `data`; names is None when ids are the numeric ids themselves (flag 2), otherwise
    names[ids[i]] is polygon i's id."""
    buffer = memoryview(data).cast("B")
    if len(buffer) < 20 or bytes(buffer[:4]) != MAGIC:
        raise ValueError("not prior vectors")
    version, size, count = (int(v) for v in np.frombuffer(buffer, "<u4", 3, 4))
    if version != VERSION or count != len(LAYERS) or size % 4 or not 20 + 16 * count <= size <= len(buffer):
        raise ValueError(f"unsupported prior vectors (version {version}, {count} layers)")
    records = np.frombuffer(buffer, "<u4", 4 * count, 16).reshape(count, 4).tolist()
    length = int(np.frombuffer(buffer, "<u4", 1, 16 + 16 * count)[0])
    value = dict(json.loads(bytes(buffer[20 + 16 * count:20 + 16 * count + length])))
    at = size

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal at
        array = np.frombuffer(buffer, dtype, count, at)
        at += array.nbytes
        return array

    for name, (flags, n, vertices, names) in zip(LAYERS, records):
        if not flags & 1:
            value[name] = None
            continue
        offsets, xy, bbox, ids = take("<u4", n + 1), take("<f4", 2 * vertices).reshape(-1, 2), take("<f4", 4 * n), take("<u4", n)
        text = bytes(take("u1", names + -names % 4)[:names])
        if offsets[0] != 0 or offsets[-1] != vertices or (np.diff(offsets.astype(np.int64)) < 0).any():
            raise ValueError(f"bad {name} offsets")
        value[name] = {"offsets": offsets, "xy": xy, "bbox": bbox.reshape(-1, 4), "ids": ids,
                       "names": None if flags & 2 else json.loads(text)}
    if at != len(buffer):
        raise ValueError("trailing bytes after the prior vectors")
    return value


def to_json(value: dict) -> dict:
    """Decoded vectors in the JSON form {**meta, cells, nuclei: [{id, poly: [[x, y], ...]}]} (an absent layer: []):
    what Runs.xenium_boundaries writes for the same run, value for value (the tests compare the two)."""
    out = {k: v for k, v in value.items() if k not in LAYERS}
    for name in LAYERS:
        layer, out[name] = value.get(name), []
        if layer is not None:
            offsets, xy, names = layer["offsets"].tolist(), layer["xy"].tolist(), layer["names"]
            out[name] = [{"id": k if names is None else names[k], "poly": xy[a:b]}
                         for k, a, b in zip(layer["ids"].tolist(), offsets, offsets[1:])]
    return out


def encode_cells(area, flags, brightness, meta=None):
    """PLCV v1, little endian; the route wraps these bytes with `packed()` like PLPV.

    Header: magic `PLCV`, u32 version=1, header bytes H, cell count N, brightness-array count C, JSON bytes M;
    UTF-8 JSON at byte 24, zero-padded to H (multiple of 4). JSON `brightness` is the ordered channel-name list;
    callers add `set`, `areas_version`, available `channels`, `default`, `dtype` and `error`.
    Sections: ids u32[N] (1..N), area f32[N], flags u8[N] (bit 0: in a removed area), zero pad to 4 bytes,
    then C brightness f32[N] arrays. No trailing bytes. NaN means unknown, kept by ranges.
    """
    area, flags = np.asarray(area, "<f4"), np.asarray(flags, "u1")
    if area.ndim != 1 or flags.shape != area.shape or len(area) >= 2 ** 32 or np.any(flags > 1):
        raise ValueError("invalid cell values")
    arrays = [np.asarray(v, "<f4") for v in brightness.values()]
    if any(v.shape != area.shape for v in arrays):
        raise ValueError("brightness does not match the cells")
    described = json.dumps({**(meta or {}), "brightness": list(brightness)}, separators=(",", ":"), allow_nan=False).encode()
    size = 24 + len(described)
    size += -size % 4
    header = b"PLCV" + np.array([1, size, len(area), len(arrays), len(described)], "<u4").tobytes() + described
    return b"".join([header.ljust(size, b"\0"), np.arange(1, len(area) + 1, dtype="<u4").tobytes(),
                     area.tobytes(), flags.tobytes(), bytes(-len(flags) % 4), *(v.tobytes() for v in arrays)])


def decode_cells(data):
    """Strict PLCV decoder for tests and tools; the returned arrays view the supplied bytes."""
    buffer = memoryview(data).cast("B")
    if len(buffer) < 24 or bytes(buffer[:4]) != b"PLCV":
        raise ValueError("not stain cell values")
    version, size, n, count, length = (int(v) for v in np.frombuffer(buffer, "<u4", 5, 4))
    expected = size + 9 * n + -n % 4 + 4 * n * count
    if version != 1 or size % 4 or not 24 + length <= size <= len(buffer) or expected != len(buffer):
        raise ValueError("invalid stain cell values")
    meta = json.loads(bytes(buffer[24:24 + length]))
    names = meta.get("brightness")
    if not isinstance(names, list) or len(names) != count or any(not isinstance(c, str) for c in names) or len(set(names)) != count:
        raise ValueError("invalid brightness channels")
    ids = np.frombuffer(buffer, "<u4", n, size)
    area = np.frombuffer(buffer, "<f4", n, size + 4 * n)
    flags = np.frombuffer(buffer, "u1", n, size + 8 * n)
    if not np.array_equal(ids, np.arange(1, n + 1, dtype=np.uint32)) or np.any(flags > 1):
        raise ValueError("invalid stain cell ids or flags")
    at = size + 9 * n + -n % 4
    return {**meta, "ids": ids, "area": area, "flags": flags,
            "brightness": {c: np.frombuffer(buffer, "<f4", n, at + 4 * n * i) for i, c in enumerate(names)}}
