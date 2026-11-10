"""Xenium output bundles: discovery, overview images and the crops a run segments."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np

PIXEL_UM = 0.2125  # Xenium morphology images; transcripts share the same micrometre origin
OVERVIEW_PX = 7000   # the whole-section DAPI thumbnail for drawing regions (level 3 for a 1 cm section)
U16_TOP = 65535      # the page draws every image as uint16 (its views, crops and contrast limits)


class SampleError(Exception):
    def __init__(self, message, status=422, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra          # more of the answer's JSON, beside its words (a helper's name clash: clash, existing)


def _short_name(folder: str) -> str:
    # output-<instrument>__<slide>__<region>__<date>__<time> -> <region>
    parts = folder.split("__")
    return parts[2] if len(parts) >= 5 else folder


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _stretch(plane: np.ndarray) -> np.ndarray:
    """uint16 -> uint8 between robust percentiles of the non-empty pixels."""
    values = plane[plane > 0]
    if values.size < 16:
        return np.zeros(plane.shape, np.uint8)
    if values.size > 2_000_000:
        values = values[:: values.size // 2_000_000]
    lo, hi = np.percentile(values, [0.5, 99.7])
    hi = max(hi, lo + 1)
    return (np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255 + .5).astype(np.uint8)


def _display_scale(plane: np.ndarray) -> float | None:
    """What a float image is multiplied by to be drawn as uint16: U16_TOP / its largest finite value (1 when it has none
    above 0), so its views, crops and contrast limits span the uint16 range (a float image in 0..1 was cast
    as it is, every value below 1 became 0, and runs and stain views were black). None for any other image: it is drawn
    as stored."""
    if plane.dtype.kind != "f":
        return None
    finite = plane[np.isfinite(plane)]
    top = float(finite.max()) if finite.size else 0.0
    return U16_TOP / top if top > 0 else 1.0


def _u16(plane: np.ndarray, scale: float | None = None) -> np.ndarray:
    """A plane as the little-endian uint16 the page draws: times `scale` (_display_scale: float images) and rounded,
    anything below 0 as 0 and above U16_TOP as U16_TOP (signed, wide or float images), NaN as 0. uint8 and uint16 come
    as they are."""
    if scale is None and plane.dtype.kind in "ui":
        if plane.dtype.kind == "i":
            plane = np.maximum(plane, 0)
        if plane.dtype.itemsize > 2:
            plane = np.minimum(plane, U16_TOP)
        return np.ascontiguousarray(plane, dtype="<u2")
    values = plane.astype(np.float64 if plane.dtype.itemsize > 4 else np.float32)     # a copy: the plane stays as read
    if scale is not None:
        values *= scale
    np.nan_to_num(values, copy=False, nan=0.0, posinf=U16_TOP, neginf=0.0)
    np.clip(np.rint(values, out=values), 0, U16_TOP, out=values)
    return np.ascontiguousarray(values, dtype="<u2")


def _channel_stats(plane: np.ndarray) -> dict:
    """Range, default contrast limits (as the 8-bit stretch) and a 256-bin histogram for napari-style controls, of a
    plane as the page draws it (_u16: the limits are in the units of its views and crops)."""
    values = plane[plane > 0]
    top = int(plane.max()) if plane.size else 0
    if values.size > 2_000_000:
        values = values[:: values.size // 2_000_000]
    lo, hi = (np.percentile(values, [0.5, 99.7]) if values.size >= 16 else (0, max(top, 1)))
    hist, _ = np.histogram(plane, bins=256, range=(0, max(top, 1)))
    return {"dtype": "uint16", "min": int(plane.min()) if plane.size else 0, "max": top,
            "p_lo": float(lo), "p_hi": float(max(hi, lo + 1)), "hist": hist.astype(int).tolist()}


def _zstd(data: bytes, level: int = 3) -> bytes:
    import zstandard
    return zstandard.ZstdCompressor(level=level, threads=-1).compress(data)


def _xenium_id(prefix: int, suffix: int) -> str:
    """Xenium cell id string: the 8 hex digits of the prefix shifted to letters a-p, then -<dataset suffix>."""
    return "".join(chr(ord("a") + int(d, 16)) for d in f"{prefix:08x}") + f"-{suffix}"


def _png(array: np.ndarray) -> bytes:
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


class Samples:
    def __init__(self, config: dict):
        self.roots = [Path(r) for r in config["paths"]["sample_roots"]]
        self.cache = Path(config["paths"]["cache_dir"]) / "samples"
        self.lock = threading.Lock()
        self.details = {}

    def list(self) -> list[dict]:
        found = []
        for root in self.roots:
            try:
                entries = sorted(root.iterdir())
            except OSError:
                continue
            for folder in entries:
                try:
                    if (folder / "transcripts.parquet").is_file():
                        found.append({"id": folder.name, "name": _short_name(folder.name), "path": str(folder)})
                except OSError:
                    continue
        return sorted(found, key=lambda s: s["name"])

    def folder(self, sample_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", sample_id or "") or sample_id.startswith("."):
            raise SampleError("Unknown sample", 404)
        for root in self.roots:
            folder = root / sample_id
            try:
                if (folder / "transcripts.parquet").is_file():
                    return folder
            except OSError:
                continue
        raise SampleError("Unknown sample", 404)

    def _image(self, folder: Path):
        import tifffile
        path = folder / "morphology_focus" / "ch0000_dapi.ome.tif"
        if not path.is_file():
            return None
        tiff = tifffile.TiffFile(path)
        series = tiff.series[0]
        xml = tiff.ome_metadata or ""
        size = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml)
        names = re.findall(r'<Channel [^>]*Name="([^"]+)"', xml)
        axes = series.axes
        shape = series.levels[0].shape
        channels = shape[axes.index("C")] if "C" in axes else 1
        return {"tiff": tiff, "series": series, "axes": axes, "pixel_um": float(size.group(1)) if size else PIXEL_UM,
                "names": (names + [f"Channel {i}" for i in range(len(names), channels)])[:channels],
                "levels": [lv.shape for lv in series.levels]}

    def detail(self, sample_id: str) -> dict:
        """Extent, transcript count and image channels; cached per sample folder."""
        folder = self.folder(sample_id)
        with self.lock:
            if sample_id in self.details:
                return self.details[sample_id]
        meta_path = self.cache / sample_id / "meta.json"
        try:
            value = json.loads(meta_path.read_text())
            if value.get("schema") == 1:
                with self.lock:
                    self.details[sample_id] = value
                return value
        except (OSError, ValueError):
            pass
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(folder / "transcripts.parquet").metadata
        experiment = {}
        try:
            experiment = json.loads((folder / "experiment.xenium").read_text())
        except (OSError, ValueError):
            pass
        image = self._image(folder)
        value = {"schema": 1, "id": sample_id, "name": experiment.get("region_name") or _short_name(sample_id),
                 "path": str(folder), "transcripts": metadata.num_rows,
                 "panel": experiment.get("panel_name"), "cells_xenium": experiment.get("num_cells")}
        if image:
            height, width = image["levels"][0][-2:]
            value.update(pixel_um=image["pixel_um"], width_um=width * image["pixel_um"], height_um=height * image["pixel_um"],
                         channels=image["names"])
            image["tiff"].close()
        else:
            # No image: take the extent from the transcripts themselves.
            table = pq.read_table(folder / "transcripts.parquet", columns=["x_location", "y_location"])
            value.update(pixel_um=None, width_um=float(np.nanmax(table.column(0).to_numpy())),
                         height_um=float(np.nanmax(table.column(1).to_numpy())), channels=[])
        _atomic_write(meta_path, json.dumps(value).encode())
        with self.lock:
            self.details[sample_id] = value
        return value

    def overview(self, sample_id: str) -> Path:
        """A DAPI thumbnail of the whole section for drawing the region (cached PNG + json)."""
        folder = self.folder(sample_id)
        path = self.cache / sample_id / "overview-v2.png"
        if path.exists() and path.with_suffix(".json").exists():
            return path
        image = self._image(folder)
        if image is None:
            raise SampleError("This sample has no morphology_focus image", 404)
        try:
            level = next((i for i, shape in enumerate(image["levels"]) if max(shape[-2:]) <= OVERVIEW_PX), len(image["levels"]) - 1)
            import zarr
            array = zarr.open(image["series"].aszarr(level=level), mode="r")
            plane = np.asarray(array[0] if array.ndim == 3 else array[:])   # DAPI only
            full_width = image["levels"][0][-1]
            meta = {"um_per_px": image["pixel_um"] * full_width / plane.shape[1], "width": plane.shape[1], "height": plane.shape[0]}
            _atomic_write(path, _png(_stretch(plane)))
            _atomic_write(path.with_suffix(".json"), json.dumps(meta).encode())
        finally:
            image["tiff"].close()
        return path

    def overview_meta(self, sample_id: str) -> dict:
        return json.loads(self.overview(sample_id).with_suffix(".json").read_text())

    def crop_transcripts(self, sample_id: str, box: list[float], destination: Path) -> int:
        """Every transcript inside the box, in the original Xenium schema, as a parquet proseg reads with --xenium."""
        import pyarrow.parquet as pq
        x0, y0, x1, y1 = box
        table = pq.read_table(self.folder(sample_id) / "transcripts.parquet",
                              filters=[("x_location", ">=", x0), ("x_location", "<", x1),
                                       ("y_location", ">=", y0), ("y_location", "<", y1)])
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")
        return table.num_rows

    def crop_images(self, sample_id: str, box: list[float], destination: Path) -> dict:
        """Each morphology channel over the box as an 8-bit PNG at full resolution, plus where it sits in micrometres."""
        import zarr
        image = self._image(self.folder(sample_id))
        if image is None:
            return {"channels": []}
        try:
            x0, y0, x1, y1 = box
            pixel = image["pixel_um"]
            level = 0   # always the native 0.2125 um pixels: the user zooms far into the masks
            shape = image["levels"][level]
            um_per_px = pixel * image["levels"][0][-1] / shape[-1]
            c0, r0 = max(0, int(x0 / um_per_px)), max(0, int(y0 / um_per_px))
            c1, r1 = min(shape[-1], math.ceil(x1 / um_per_px)), min(shape[-2], math.ceil(y1 / um_per_px))
            array = zarr.open(image["series"].aszarr(level=level), mode="r")
            channels = []
            for index, name in enumerate(image["names"]):
                plane = np.asarray(array[index, r0:r1, c0:c1] if array.ndim == 3 else array[r0:r1, c0:c1])
                file = f"c{index}.png"
                _atomic_write(destination / file, _png(_stretch(plane)))
                # Full bit depth for napari-style contrast limits in the viewer (zstd raw uint16, row-major).
                file16 = f"c{index}.u16.zst"
                _atomic_write(destination / file16, _zstd(np.ascontiguousarray(plane, dtype="<u2").tobytes(), 3))
                channels.append({"name": name, "file": file, "file16": f"c{index}.u16", **_channel_stats(plane)})
            meta = {"x0": c0 * um_per_px, "y0": r0 * um_per_px, "um_per_px": um_per_px,
                    "width": c1 - c0, "height": r1 - r0, "level": level, "channels": channels}
            _atomic_write(destination / "image.json", json.dumps(meta).encode())
            return meta
        finally:
            image["tiff"].close()


    # ---- the Xenium segmentation (proseg's prior comes from it) --------------------------------------------------
    def xenium_boundaries(self, sample_id: str, box, margin: float = 20.0) -> dict:
        """Every Xenium cell and nucleus boundary with a vertex inside the box (+ margin), as polygons in µm."""
        import pyarrow.parquet as pq
        folder = self.folder(sample_id)
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
        result = {}
        for key, name in (("cells", "cell_boundaries.parquet"), ("nuclei", "nucleus_boundaries.parquet")):
            path = folder / name
            if not path.exists():
                result[key] = []
                continue
            table = pq.read_table(path, columns=["cell_id", "vertex_x", "vertex_y"])
            ids = table.column("cell_id").combine_chunks().dictionary_encode()
            index = ids.indices.to_numpy(zero_copy_only=False)
            names = ids.dictionary.to_pylist()
            xs = table.column("vertex_x").to_numpy()
            ys = table.column("vertex_y").to_numpy()
            inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            chosen = np.isin(index, np.unique(index[inside]))
            index, xs, ys = index[chosen], xs[chosen], ys[chosen]
            if index.size == 0:        # nothing near the box: an empty layer (np.r_[True] below made one from nothing)
                result[key] = []
                continue
            starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
            ends = np.r_[starts[1:], index.size]
            result[key] = [{"id": names[index[a]], "poly": np.round(np.column_stack([xs[a:b], ys[a:b]]), 2).tolist()}
                           for a, b in zip(starts, ends)]
        return result

    def crop_xenium_masks(self, sample_id: str, box, destination: Path) -> dict:
        """Xenium's nucleus and cell label masks over the box (uint32, full image resolution) and their cell ids."""
        import zarr
        folder = self.folder(sample_id)
        path = folder / "cells.zarr.zip"
        if not path.exists():
            raise SampleError("This sample has no cells.zarr.zip (Xenium masks)", 404)
        root = zarr.open_group(zarr.storage.ZipStore(path, mode="r"), mode="r")
        try:   # maps micrometres to mask pixels: px = T @ (x, y, 0, 1)
            transform = np.asarray(root["masks/homogeneous_transform"][:], dtype=float)
        except KeyError:
            transform = np.diag([1 / PIXEL_UM, 1 / PIXEL_UM, 1, 1])
        sx, sy, tx, ty = transform[0, 0], transform[1, 1], transform[0, 3], transform[1, 3]
        x0, y0, x1, y1 = (float(v) for v in box)
        height, width = root["masks/1"].shape
        c0, r0 = max(0, int(x0 * sx + tx)), max(0, int(y0 * sy + ty))
        c1, r1 = min(width, math.ceil(x1 * sx + tx)), min(height, math.ceil(y1 * sy + ty))
        um_per_px = 1 / sx
        nuclei = np.asarray(root["masks/0"][r0:r1, c0:c1], dtype="<u4")
        cells = np.asarray(root["masks/1"][r0:r1, c0:c1], dtype="<u4")
        ids = np.asarray(root["cell_id"][:])
        labels = np.union1d(np.unique(nuclei), np.unique(cells))
        labels = labels[(labels > 0) & (labels <= len(ids))]
        names = {int(l): _xenium_id(int(ids[l - 1, 0]), int(ids[l - 1, 1])) for l in labels.tolist()}
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_write(destination / "masks.u32.zst", _zstd(nuclei.tobytes() + cells.tobytes(), 3))
        meta = {"x0": (c0 - tx) / sx, "y0": (r0 - ty) / sy, "um_per_px": um_per_px, "width": c1 - c0, "height": r1 - r0,
                "layout": "uint32 nucleus[height*width] then cell[height*width]; label = Xenium cell index + 1",
                "ids": names}
        _atomic_write(destination / "masks.json", json.dumps(meta).encode())
        return meta
