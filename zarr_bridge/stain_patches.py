"""Method-independent crops of committed stain results. Only the image reader opens the sample (mode=r)."""
from collections import OrderedDict
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import stat
import sys

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from .. import portable
from .stain import at_level, atomic_json, image_pixels, job_level, read_shapes, resolution_info, resolve_shapes
from .tissue import tissue_area
from ..engine.cellpose_settings import NORMALIZE_DEFAULT
from ..engine.zarr_samples import ZarrSamples

MAX_PATCHES = 64
MAX_PIXELS = 16_777_216
# The start of an export's file names (<stem>-01.tif ...), as the engine makes them (engine/patch_folders.py STEM).
STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype="<u4").tobytes()).hexdigest()


class SegReader(pickle.Unpickler):
    """Cellpose stores a pickled numpy dictionary; permit its arrays, never arbitrary Python objects."""
    def find_class(self, module, name):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype"),
                             ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
                             ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
                             ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}:
            return super().find_class(module, name)
        raise ValueError("This patch has an unsupported saved value")


def read_seg(path, shape):
    # Large exports contain two uint32 planes; GUI saves can also carry image-sized flows.
    limit = max(64 * 1024 * 1024, min(MAX_PIXELS, int(np.prod(shape))) * 32 + 1024 * 1024)
    # The mask's folder may be one others can write to: it is never read through a link, and it is opened without
    # waiting, so a FIFO or a device in its place is refused before a byte is read (a FIFO would hold this helper, and
    # every patch call waiting for it, until the call timed out).
    flags = os.O_RDONLY | portable.O_NOFOLLOW | portable.O_NONBLOCK | portable.O_CLOEXEC | portable.O_BINARY
    with open(os.open(path, flags), "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("This patch mask is too large or is not a plain file")
        raw = file.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("This patch mask is too large or is not a plain file")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError("This patch uses an unsupported file format")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    dims, _, dtype = reader(stream)
    if dims != () or not dtype.hasobject:
        raise ValueError("This patch is not a Cellpose mask")
    saved = SegReader(stream).load().item()
    mask = np.asarray(saved["masks"])
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != tuple(shape) or mask.dtype.kind not in "ui" or mask.min() < 0 or mask.max() > 2**32-1:
        raise ValueError("This patch mask does not match its image")
    return mask.astype(np.uint32)


def dense_mask(mask):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    result = np.zeros(mask.shape, np.uint32)
    if len(ids):
        positive = mask > 0
        result[positive] = np.searchsorted(ids, mask[positive]) + 1
    return result, len(ids)


def write_seg(path, pixels, mask, filename, settings):
    # masks_flows_to_seg-compatible, with no invented model flows. The GUI recomputes 2-D outlines on load.
    padded = np.pad(mask, 1)
    interior = ((mask == padded[:-2, 1:-1]) & (mask == padded[2:, 1:-1]) &
                (mask == padded[1:-1, :-2]) & (mask == padded[1:-1, 2:]))
    outlines = mask * ~interior
    np.save(path, {"masks": mask, "outlines": outlines, "filename": str(filename),
                  "chan_choose": [0, 0], "ismanual": np.zeros(int(mask.max()), bool),
                  "flows": [], "diameter": 0. if settings.get("diameter") in (None, "auto") else settings["diameter"],
                  "flow_threshold": settings.get("flow_threshold", .4),
                  "cellprob_threshold": settings.get("cellprob_threshold", 0.),
                  "normalize_params": {**NORMALIZE_DEFAULT, **(settings.get("normalize") or {})},
                  "model_path": settings.get("model_path") or settings.get("model", "cpsam")})


class Result:
    def __init__(self, root, meta, exclusions=(), job=None):
        self.root, self.meta = root, meta
        # A patch preview passes the job of an item not created yet; runs and items keep theirs in stain/job.json.
        self.job = json.loads((root / "stain/job.json").read_text()) if job is None else job
        self.tissue = tissue_area(self.job.get("tissue") or [])
        self.exclusions = shapely.union_all([Polygon(r) for r in exclusions])
        image = self.job["image"]
        _, level, self.factor = job_level(self.job)
        ox, oy, sx, sy = ZarrSamples._geometry(image, level)
        self.window = ZarrSamples._window(ZarrSamples, image, level, self.job["region"]["bbox"])
        wx, wy = self.window[:2]
        self.geometry = [ox + wx*sx, oy + wy*sy, sx, sy]
        self.extent = [self.window[2]-wx, self.window[3]-wy]
        region = box(*self.job["region"]["bbox"]).intersection(
            box(ox, oy, ox+sx*level["shape"][-1], oy+sy*level["shape"][-2]))
        if self.job["region"]["kind"] == "polygon":
            region = region.intersection(Polygon(self.job["region"]["points"]))
        progress = meta["progress"]
        if meta.get("kind") == "patches":
            self.shapes = []
        elif progress.get("resolved"):
            self.shapes = read_shapes(root / "output/boundaries.geojson")
        else:
            records = json.loads((root / "stain/tiles.json").read_text())[:progress["done"]]
            shapes, tiles = [], []
            for i, record in enumerate(records):
                path = root / f"stain/tile-{i:06d}.geojson"
                if path.exists():
                    part = read_shapes(path)
                else:
                    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                        part = [Polygon(data["coords"][a:b]) for a, b in zip(data["offsets"][:-1], data["offsets"][1:])]
                shapes.extend(part)
                tiles.extend([i]*len(part))
            self.shapes = resolve_shapes(shapes, tiles, records, (sx, sy))
            region = region.intersection(shapely.union_all([box(*r["box"]) for r in records]))
        covered = meta.get("covered") or {}
        if covered.get("region"):
            spec = covered["region"]
            region = region.intersection(Polygon(spec["points"]) if spec["kind"] == "polygon" else box(*spec["box"]))
        self.region = region
        rules = meta.get("exclusions") or {}
        if "size" in rules or "brightness" in rules:
            # Match the engine's f32 outline rings and values, then compare in f64 (including on numpy 2).
            from sopa_companion.engine.stain_exclusions import centres_and_areas
            from sopa_companion.engine.stain_intensity import rings
            centres, areas = centres_and_areas(*rings(self.shapes))
            values = {c: self.brightness(c, progress) for c in (rules.get("brightness") or {})}
            inside = shapely.intersects_xy(self.exclusions, centres[:, 0], centres[:, 1])
            keep = range_keep(areas, values, inside, rules)
            self.shapes = [p for p, k in zip(self.shapes, keep) if k]
            self.tree = shapely.STRtree(self.shapes)
            return
        dim = rules.get("min_brightness")
        if dim and self.shapes:   # the run's brightness rule (engine/stain_exclusions.py): dimmer cells stay out
            values = self.brightness(dim["channel"], progress)
            with np.errstate(invalid="ignore"):
                self.shapes = [p for p, v in zip(self.shapes, values) if not v < dim["value"]]
        if not self.exclusions.is_empty:
            self.shapes = [p for p in self.shapes if not shapely.intersects_xy(self.exclusions, p.centroid.x, p.centroid.y)]
        least = (meta.get("exclusions") or {}).get("min_area_um2")
        if least:   # the run's size rule (engine/stain_exclusions.py): smaller cells stay out of proposals and masks
            self.shapes = [p for p in self.shapes if p.area >= least]
        self.tree = shapely.STRtree(self.shapes)

    def brightness(self, channel, progress):
        """Each cell's brightness in `channel` (engine/stain_intensity.py). A finished run's cells are its final outline
        file's in the same order (output/boundaries.geojson is that file's geojson), so the values the engine stored
        for it are read when they are there: the engine asks for patches under the run's lock, and measuring a whole
        sample again would hold it for seconds. A paused run's cells are joined here from its tiles: measured."""
        from sopa_companion.engine import stain_intensity
        name = progress.get("resolved_dir") or (f"resolved-{progress['done']:06d}" if "done" in progress else None)
        if progress.get("resolved") and name:
            try:
                index = self.job["image"]["channels"].index(channel)
                stamp = (self.root / "stain" / name / "offsets.npy").stat().st_mtime_ns
                stored = np.load(self.root / "stain" / "brightness" /
                                 f"{name}-c{index}-{stain_intensity.CACHE_TAG}-{stamp}.npy", allow_pickle=False)
                if len(stored) == len(self.shapes):
                    return stored
            except (OSError, ValueError):
                pass
        read, frame, size = stain_intensity.job_reader(self.job, channel)
        return stain_intensity.brightness(read, frame, *stain_intensity.rings(self.shapes), size)

    def world(self, pixels):
        ox, oy, sx, sy = self.geometry
        a, b, c, d = pixels
        return [ox+a*sx, oy+b*sy, ox+c*sx, oy+d*sy]

    def covered_region(self):
        # The run window rounds outwards to whole pixels, including at rectangle corners.
        return self.region.buffer(max(self.geometry[2:]), join_style="mitre")

    def allowed_centre(self, world):
        x, y = (world[0]+world[2])/2, (world[1]+world[3])/2
        return self.allowed_centres(x, y)

    def allowed_centres(self, x, y):
        return ((self.tissue is None or shapely.intersects_xy(self.tissue, x, y)) &
                ~shapely.intersects_xy(self.exclusions, x, y))

    def grid_starts(self, size, limit):
        # Aligned boxes always pack. Bound the grid, retaining capacity for up to MAX_PATCHES even in narrow runs.
        nx, ny = [n//size for n in self.extent]
        cols = min(nx, max(limit, int(np.ceil(MAX_PATCHES/ny))))
        rows = min(ny, max(limit, int(np.ceil(MAX_PATCHES/nx))))
        return {(int(x)*size, int(y)*size) for y in np.linspace(0, ny-1, rows, dtype=int)
                for x in np.linspace(0, nx-1, cols, dtype=int)}

    def choose(self, candidates, count, grid):
        bins = [np.quantile([c[i] for c in candidates], [1/3, 2/3]) for i in (1, 2)]

        def pick(pool, chosen):
            groups = {}
            for c in pool:
                key = tuple(int(np.searchsorted(bins[i], c[i+1], side="right")) for i in (0, 1))
                groups.setdefault(key, []).append(c)
            while groups and len(chosen) < count:
                for key in sorted(list(groups)):
                    pool = [c for c in groups[key] if not any(
                        c[0][0] < b[2] and b[0] < c[0][2] and c[0][1] < b[3] and b[1] < c[0][3] for b in chosen)]
                    if pool:
                        picked = max(pool, key=lambda c: (min((np.linalg.norm(
                            (np.array(c[0][:2])-b[:2])/self.extent) for b in chosen), default=1), c[3]))
                        chosen.append(picked[0])
                        pool.remove(picked)
                    if pool:
                        groups[key] = pool
                    else:
                        del groups[key]
                    if len(chosen) == count:
                        break
            return chosen

        chosen = pick(candidates, [])
        if len(chosen) < count:
            # A diverse first choice can block several neighbours; retry from the non-overlapping grid.
            packed = pick([c for c in candidates if tuple(c[0][:2]) in grid], [])
            packed = pick(candidates, packed)
            if len(packed) > len(chosen):
                chosen = packed
        return self.validate(chosen)

    def validate(self, boxes):
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_PATCHES:
            raise ValueError("Choose between 1 and 64 patches")
        total = 0
        region = self.covered_region()
        # Numbered as the page numbers them; a patch item's region is the user's, a stain run's its finished tiles.
        inside = "the region" if self.meta.get("kind") == "patches" else "the finished part of the run"
        for n, b in enumerate(boxes, 1):
            if not isinstance(b, list) or len(b) != 4 or any(type(v) is not int for v in b):
                raise ValueError(f"Draw box {n} inside {inside}")
            x, y, xx, yy = b
            if not (16 <= xx-x <= 4096 and 16 <= yy-y <= 4096):
                raise ValueError(f"Box {n} must be 16 to 4096 pixels on each side")
            if not (0 <= x < xx <= self.extent[0] and 0 <= y < yy <= self.extent[1]) or \
                    not region.covers(box(*self.world(b))):
                raise ValueError(f"Keep box {n} inside {inside}")
            total += (xx-x)*(yy-y)
        if total > MAX_PIXELS:
            raise ValueError("Choose fewer or smaller patches")
        return boxes

    def pixels(self, b, step=1):
        x, y = self.window[:2]
        return image_pixels(self.job, [b[0]+x, b[1]+y, b[2]+x, b[3]+y], step=step)

    def masks(self, b):
        ox, oy, sx, sy = self.geometry
        xx, yy = np.meshgrid(ox+(np.arange(b[0], b[2])+.5)*sx, oy+(np.arange(b[1], b[3])+.5)*sy)
        mask = np.zeros(xx.shape, np.uint32)
        for k in sorted(self.tree.query(box(*self.world(b)), predicate="intersects")):
            mask[shapely.intersects_xy(self.shapes[k], xx, yy)] = k+1
        return dense_mask(mask)[0]

    def filter_inferred(self, mask, b):
        from scipy.ndimage import center_of_mass
        mask, count = dense_mask(mask)
        if not count or self.tissue is None and self.exclusions.is_empty:
            return mask
        # Each label has constant weight, so its mass centre is the mean of its pixel centres.
        centres = np.asarray(center_of_mass(mask, mask, np.arange(1, count+1)))
        ox, oy, sx, sy = self.geometry
        keep = self.allowed_centres(ox+(b[0]+centres[:, 1]+.5)*sx, oy+(b[1]+centres[:, 0]+.5)*sy)
        return dense_mask(np.where(np.r_[False, keep][mask], mask, 0))[0]

    def propose_image(self, count, size):
        image = self.job["image"]
        # Read one nucleus plane at a pyramid level, never many strided full-resolution crops.
        level = next((l for l in image["levels"] if max(l["shape"][-2:]) <= 2048), image["levels"][-1])
        low_job = {**self.job, "image": {**image, "levels": [level]},
                   "settings": {"method": "instanseg", "channels": [
                       self.job["settings"].get("nucleus") or self.job["settings"]["channels"][0]]}}
        window = ZarrSamples._window(ZarrSamples, image, level, self.job["region"]["bbox"])
        step = max(1, int(np.ceil(max(window[2]-window[0], window[3]-window[1])/2048)))
        pixels = self.pixels_low(low_job, window, step)
        if not pixels.size or not np.isfinite(pixels).all() or np.ptp(pixels) == 0:
            raise ValueError("No patches with image detail were found. Try another region.")
        ox, oy, sx, sy = ZarrSamples._geometry(image, level)
        if self.tissue is None:
            hist, edges = np.histogram(pixels, bins=256)
            centres = (edges[:-1]+edges[1:])/2
            weight = hist.cumsum().astype(float)
            moment = (hist*centres).cumsum()
            score = (moment[-1]*weight-moment*weight[-1])**2 / np.maximum(weight*(weight[-1]-weight), 1)
            tissue = pixels > centres[int(np.argmax(score))]
        else:
            xx = ox+(window[0]+(np.arange(pixels.shape[1])+.5)*step)*sx
            yy = oy+(window[1]+(np.arange(pixels.shape[0])+.5)*step)*sy
            tissue = shapely.intersects_xy(self.tissue, xx[None, :], yy[:, None])
        texture = np.abs(np.diff(pixels, axis=0, prepend=pixels[:1])) + np.abs(np.diff(pixels, axis=1, prepend=pixels[:, :1]))
        integrals = [np.pad(a.astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                     for a in (tissue, pixels, texture)]
        rng, candidates = np.random.default_rng(0), []
        grid = self.grid_starts(size, 64)
        starts = set(grid)
        # Include small tissue islands that fall between the bounded grid's boxes on a large sample.
        tissue_pixels = np.flatnonzero(tissue)
        for index in rng.choice(tissue_pixels, min(512, len(tissue_pixels)), replace=False):
            row, col = divmod(int(index), pixels.shape[1])
            x = (ox+(window[0]+(col+.5)*step)*sx-self.geometry[0])/self.geometry[2]-size/2
            y = (oy+(window[1]+(row+.5)*step)*sy-self.geometry[1])/self.geometry[3]-size/2
            starts.add((int(np.clip(x, 0, self.extent[0]-size)), int(np.clip(y, 0, self.extent[1]-size))))
        region = self.covered_region()
        for x, y in sorted(starts):
            b = [int(x), int(y), int(x)+size, int(y)+size]
            world = self.world(b)
            if not region.covers(box(*world)) or not self.allowed_centre(world):
                continue
            a, c = np.clip([(world[i]-ox)/sx-window[0] for i in (0, 2)], 0, window[2]-window[0])/step
            d, f = np.clip([(world[i]-oy)/sy-window[1] for i in (1, 3)], 0, window[3]-window[1])/step
            a, d, c, f = int(a), int(d), min(pixels.shape[1], int(np.ceil(c))), min(pixels.shape[0], int(np.ceil(f)))
            area = (c-a)*(f-d)
            if not area:
                continue
            coverage, intensity, detail = [(t[f,c]-t[d,c]-t[f,a]+t[d,a])/area for t in integrals]
            if (self.tissue is not None or coverage >= .1) and detail > 0:
                candidates.append((b, intensity, detail, float(rng.random())))
        if not candidates:
            raise ValueError("No patches with image detail were found. Try another region.")
        return self.choose(candidates, count, grid)

    @staticmethod
    def pixels_low(job, window, step):
        return image_pixels(job, window, step=step)[..., 0]

    def propose(self, count, size):
        if type(count) is not int or not 1 <= count <= MAX_PATCHES or type(size) is not int or not 64 <= size <= 4096:
            raise ValueError("Choose 1–64 patches and a size between 64 and 4096 pixels")
        if min(self.extent) < size:
            raise ValueError("Choose a smaller patch size for this region")
        count = min(count, MAX_PIXELS//(size*size))
        if self.meta.get("kind") == "patches":
            return self.propose_image(count, size)
        # A bounded grid plus cell-centred candidates includes small tissue islands. Seeded ties are repeatable.
        rng = np.random.default_rng(0)
        nx = min(12, max(2, int(np.ceil(self.extent[0]/(size/2)))))
        ny = min(12, max(2, int(np.ceil(self.extent[1]/(size/2)))))
        starts = {(int(x), int(y)) for y in np.linspace(0, self.extent[1]-size, ny)
                  for x in np.linspace(0, self.extent[0]-size, nx)}
        grid = self.grid_starts(size, 12)
        starts.update(grid)
        ox, oy, sx, sy = self.geometry
        for k in np.linspace(0, len(self.shapes)-1, min(64, len(self.shapes)), dtype=int):
            p = self.shapes[k].representative_point()
            starts.add((int(np.clip((p.x-ox)/sx-size/2, 0, self.extent[0]-size)),
                        int(np.clip((p.y-oy)/sy-size/2, 0, self.extent[1]-size))))
        candidates = []
        region = self.covered_region()
        for x, y in sorted(starts):
            b = [x, y, x+size, y+size]
            world = self.world(b)
            if not region.covers(box(*world)) or not self.allowed_centre(world):
                continue
            density = len(self.tree.query(box(*self.world(b)), predicate="intersects"))
            if not density:
                continue
            image = self.pixels(b, step=8)
            instanseg = self.job["settings"].get("method") == "instanseg"
            intensity = float(np.mean(image if instanseg else image[..., :2]))
            detail = np.max(np.ptp(image, axis=(0, 1))) if instanseg else np.ptp(image[..., 0])
            if not np.isfinite(image).all() or intensity <= 0 or detail == 0:
                continue
            candidates.append((b, density, intensity, float(rng.random())))
        if not candidates:
            raise ValueError("No patches with cells and image detail were found. Try another region.")
        return self.choose(candidates, count, grid)

    def export(self, manifest, staging, target, segment=None, progress=None):
        """The patches into `staging`, named <stem>-NN (the manifest's stem; "patch" before stems), each mask's file
        naming its image in `target`, the folder they are copied into. `progress(done, total)` after each patch."""
        import tifffile
        stem = manifest.get("stem") or "patch"
        if not isinstance(stem, str) or not STEM.fullmatch(stem):
            raise ValueError("The patch names could not be made")
        boxes = self.validate([p["box"] for p in manifest["patches"]])
        exported = []
        for i, b in enumerate(boxes, 1):
            name = f"{stem}-{i:02d}"
            pixels = self.pixels(b).astype(np.float32)
            if not np.isfinite(pixels).all():
                raise ValueError("The image contains values that cannot be used for training")
            masks = segment(pixels.copy()) if segment is not None else self.masks(b)
            if masks.shape != pixels.shape[:2] or masks.dtype.kind not in "ui" or masks.min() < 0 or masks.max() > 2**32-1:
                raise ValueError("The model returned a mask that does not match its image")
            masks = self.filter_inferred(masks, b) if segment is not None else dense_mask(masks)[0]
            tifffile.imwrite(staging / f"{name}.tif", pixels, photometric="rgb" if pixels.shape[-1] == 3 else "minisblack",
                             planarconfig="contig", metadata={"axes": "YXC"})
            write_seg(staging / f"{name}_seg.npy", pixels, masks, target / f"{name}.tif",
                      at_level(self.job["settings"], self.factor))
            exported.append({"id": i, "box": b, "image": f"{name}.tif", "mask": f"{name}_seg.npy",
                             "export_hash": mask_hash(masks), "cells": int(masks.max())})
            if progress is not None:
                progress(i, len(boxes))
        settings = self.job["settings"]
        channels = settings.get("channels", [settings.get("nucleus"), settings.get("cytoplasm"), None])
        if settings.get("method") != "instanseg":
            channels = [*channels, *([None] * (3-len(channels)))]
        result = {**manifest, **resolution_info(self.job), "patches": exported,
                  "sample": self.meta["sample"], "sample_path": self.job["path"],
                  "element": self.job["image"]["key"], "method": settings.get("method", "cellpose"),
                  "settings": settings, "start_from": self.meta.get("start_from", "run"),
                  "channels": channels, "cellpose_channels": (channels[:3] + [None]*3)[:3],
                  "normalization": "cellpose_default", "dtype": "float32", "scale": self.geometry[2:],
                  "run_pixel_to_sample": self.geometry, "image_pixel_window": self.window}
        atomic_json(staging / "patches.json", result)
        return result


def range_keep(area, brightness, inside, rules):
    try:
        from sopa_companion.engine.stain_rules import keep
    except ModuleNotFoundError as exc:
        if exc.name != "sopa_companion.engine.stain_rules":
            raise
    else:
        return keep(np.asarray(area, np.float32), {c: np.asarray(v, np.float32) for c, v in brightness.items()},
                    inside, rules)
    kept = ~np.asarray(inside, bool)
    for values, bounds in [(area, rules.get("size")), *[(brightness[c], b) for c, b in
                                                     (rules.get("brightness") or {}).items()]]:
        if not bounds:
            continue
        values = np.asarray(values, np.float32).astype(np.float64)
        with np.errstate(invalid="ignore"):
            if bounds.get("min") is not None:
                kept &= ~(values < bounds["min"])
            if bounds.get("max") is not None:
                kept &= ~(values > bounds["max"])
    return kept


def export_model(request):
    from contextlib import nullcontext
    from .stain import Cellpose, emit
    from .stain_instanseg import InstanSegProcess
    data = Result(Path(request["root"]), request["meta"], request.get("exclusions", []))
    data.validate([p["box"] for p in request["manifest"]["patches"]])
    if data.job["settings"].get("method") == "instanseg":
        model = InstanSegProcess(data.job)
    else:
        model = nullcontext(Cellpose(data.job, data.root))
    with model as segment:
        data.export(request["manifest"], Path(request["staging"]), Path(request["target"]), segment,
                    progress=lambda done, total: emit("progress", done=done, total=total))
    emit("done")


def inspect_patch(folder, patch):
    b = patch["box"]
    if not (folder / patch["image"]).is_file():
        raise ValueError("The patch image is missing")
    mask = read_seg(folder / patch["mask"], (b[3]-b[1], b[2]-b[0]))
    _, count = dense_mask(mask)
    return {"id": patch["id"], "state": "untouched" if mask_hash(mask) == patch["export_hash"] else "corrected",
            "cells": count, "hash": mask_hash(mask)}


def serve():
    portable.nicest()                       # niceness 19 (Windows: the idle priority class)
    cache = OrderedDict()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            root, meta, action = Path(request["root"]), request["meta"], request["action"]
            with redirect_stdout(sys.stderr):
                if action == "inspect":
                    result = []
                    for p in request["manifest"]["patches"]:
                        try:
                            result.append(inspect_patch(Path(request["folder"]), p))
                        except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, TypeError, AttributeError):
                            result.append({"id": p["id"], "state": "missing", "cells": None})
                else:
                    exclusions = request.get("exclusions", [])
                    if action == "preview":
                        # An item before it exists: the request carries its job; nothing is read from a run or kept.
                        data = Result(root, meta, exclusions, job=request["job"])
                    else:
                        key = (str(root), meta["progress"]["revision"], meta.get("exclusions", {}).get("version", 0),
                               json.dumps(exclusions, sort_keys=True))
                        if key not in cache:
                            cache[key] = Result(root, meta, exclusions)
                            while len(cache) > 2:
                                cache.popitem(last=False)
                        data = cache[key]
                    propose = action == "propose" or action == "preview" and request.get("boxes") is None
                    if action == "export":
                        result = data.export(request["manifest"], Path(request["staging"]), Path(request["target"]))
                    else:
                        boxes = data.propose(request["count"], request["size"]) if propose else data.validate(request["boxes"])
                        result = {"patches": [{"id": i, "box": b} for i, b in enumerate(boxes, 1)],
                                  "run_pixel_to_sample": data.geometry, "message": ""}
                        if propose and len(boxes) < request["count"]:
                            result["message"] = (f"Found {len(boxes)} of {request['count']} requested patches. "
                                                 "Try a smaller size or another region for more.")
            answer = {"ok": True, "result": result}
        except (ValueError, FileNotFoundError) as exc:
            answer = {"ok": False, "detail": str(exc), "status": 422}
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            answer = {"ok": False, "detail": "The patches could not be read. Try opening the run again", "status": 503}
        print(json.dumps(answer, allow_nan=False), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        from .stain import emit
        portable.nicest()
        try:
            with redirect_stdout(sys.stderr):
                export_model(json.loads(Path(sys.argv[2]).read_text()))
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc))
            sys.exit(1)
    else:
        serve()


if __name__ == "__main__":
    main()
