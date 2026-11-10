"""Image helper. Reads zarr arrays with mode=r; all work stays beside the run (a built-in model it downloads goes
into Cellpose's own model folder), never in the sample."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from .. import portable
from .devices import choose_device, empty_cache

MAX_RESCALED_PIXELS = 1024**2


def discover_models(models=None):
    """List weights without loading them or downloading anything. Paths come only from local Cellpose settings."""
    if models is None:
        from cellpose import models
    directory = Path(models.MODEL_DIR).expanduser()
    builtins = list(dict.fromkeys(models.MODEL_NAMES))
    result = [{"name": name, "kind": "builtin", "path": str((directory / name).resolve())
               if (directory / name).is_file() else None, "date": None} for name in builtins]
    listing = Path(models.MODEL_LIST_PATH).expanduser()
    entries = listing.read_text().splitlines() if listing.is_file() else []
    paths = [Path(line.strip()).expanduser() for line in entries if line.strip()]
    paths = [p if p.is_absolute() else directory / p for p in paths]
    # Cellpose normally uses extensionless weight files; also accept common saved-weight suffixes.
    paths += sorted(p for p in directory.iterdir() if p.suffix.lower() in ("", ".pt", ".pth", ".ckpt", ".bin")) if directory.is_dir() else []
    seen = {m["name"] for m in result}
    for path in paths:
        name = path.name
        if (name in seen or name.startswith(".") or any(c in name for c in "/\\\0\n\r")
                or path == listing or not path.is_file()):
            continue
        seen.add(name)
        date = None
        match = re.search(r"(?<!\d)(20\d{6})(?!\d)", name)
        if match:
            try:
                date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
            except ValueError:
                pass
        if date is None:
            date = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()
        result.append({"name": name, "kind": "custom", "path": str(path.resolve()), "date": date, "source": "gui",
                       "created": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    return [m for m in result if m["kind"] == "builtin"] + sorted(
        (m for m in result if m["kind"] == "custom"), key=lambda m: m.get("created") or m.get("date") or "", reverse=True)


def model_file(settings, out, models):
    name = settings.get("model", "cpsam")
    choice = next((m for m in discover_models(models) if m["name"] == name), None)
    if choice is None:
        raise ValueError("This model is no longer available. Choose a model and start a new run.")
    if settings.get("model_kind", choice["kind"]) != choice["kind"]:
        raise ValueError("The model has changed. Choose a model and start a new run.")
    record = {"model": name, "model_kind": choice["kind"]}
    if choice["kind"] == "custom":
        if settings.get("model_path", choice["path"]) != choice["path"]:
            raise ValueError("The model file has moved. Choose a model and start a new run.")
        path = Path(choice["path"])
        with path.open("rb") as stream:
            record["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    elif choice["path"] and Path(choice["path"]).is_file():
        path = Path(choice["path"])
    else:
        path = download_builtin(name, out, models)
    record["model_path"] = str(path.resolve())
    previous = out / "stain" / "model.json"
    if previous.is_file() and choice["kind"] == "custom" and json.loads(previous.read_text()) != record:
        raise ValueError("The model file has changed. Start a new run to use it.")
    return record


def download_builtin(name, out, models) -> Path:
    """A built-in model not on this computer yet: downloaded once into Cellpose's own model folder (MODEL_DIR, normally
    ~/.cellpose/models, where Cellpose itself keeps it: every later run, every session, Cellpose's window and sopa's own
    Cellpose find it there), else, where that folder cannot be written, into the run. A run that has its own copy
    already (an older run's Resume) keeps it. Through a staging folder beside its place (the same disk: one rename
    publishes it whole), so a failed or interrupted download never becomes a model."""
    own = out / "stain" / "models"
    if (own / name).is_file():
        return own / name
    for folder in (Path(models.MODEL_DIR).expanduser(), own):
        if (folder / name).is_file():               # another run's download, finished since discovery
            return folder / name
        try:
            folder.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".download-", dir=folder))
        except OSError:
            continue
        emit("progress", message="Downloading the model…")
        previous = models.MODEL_DIR
        try:
            models.MODEL_DIR = staging
            downloaded = Path(models.cache_model_path(name))
            if not downloaded.is_file() or not downloaded.stat().st_size:
                raise OSError("The download was empty")
            os.replace(downloaded, folder / name)
        except Exception as exc:
            raise RuntimeError("Could not download the model. Check the connection and try again.") from exc
        finally:
            models.MODEL_DIR = previous
            shutil.rmtree(staging, ignore_errors=True)
        return folder / name
    raise RuntimeError("Could not download the model: neither Cellpose's model folder nor the run's folder can be "
                       "written")


def diameter(settings):
    value = settings.get("diameter")
    if value is None or value == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 10 <= value <= 1000:
        raise ValueError("Choose a cell diameter between 10 and 1000 pixels, or Auto")
    return value


def job_level(job):
    """The chosen image level, including jobs from before the resolution setting."""
    try:
        from sopa_companion.engine.stain_levels import job_level as shared
    except ModuleNotFoundError as exc:
        if exc.name != "sopa_companion.engine.stain_levels":
            raise
    else:
        return shared(job)
    # Compatibility with engines that predate the shared resolution module.
    image = job["image"]
    k = job["settings"].get("level", 0)
    if type(k) is not int or not 0 <= k < len(image["levels"]):
        raise ValueError("Choose a resolution this image has")
    level, full = image["levels"][k], image["levels"][0]
    factor = math.sqrt(math.prod(a / b for a, b in zip(level["pixel"], full["pixel"])))
    return k, level, factor


def at_level(settings, factor):
    """Convert full-image pixel settings once, without validating the smaller diameter."""
    try:
        from sopa_companion.engine.stain_levels import at_level as shared
    except ModuleNotFoundError as exc:
        if exc.name != "sopa_companion.engine.stain_levels":
            raise
    else:
        return shared(settings, factor)
    result = {**settings, "normalize": dict(settings.get("normalize") or {})}
    if settings.get("diameter") not in (None, "auto"):
        result["diameter"] = settings["diameter"] / factor
    if settings.get("min_size", -1) != -1:
        result["min_size"] = max(0, round(settings["min_size"] / factor**2))
    for key in ("sharpen_radius", "smooth_radius", "tile_norm_blocksize"):
        if key in result["normalize"]:
            value = result["normalize"][key] / factor
            result["normalize"][key] = round(value) if key == "tile_norm_blocksize" else value
    return result


def resolution_info(job):
    from sopa_companion.engine.zarr_samples import ZarrSamples
    level_index, level, factor = job_level(job)
    _, _, sx, sy = ZarrSamples._geometry(job["image"], level)
    return dict(level=level_index, factor=factor, um_per_px=math.sqrt(sx * sy))


def check_tile(shape, settings, factor=1, model_pixel_size=.5):
    if settings.get("method") == "instanseg":
        scale = settings["pixel_size"] / model_pixel_size
    else:
        value = diameter(settings)
        scale = 30 * factor / value if value is not None else 1
    if math.prod(math.ceil(n * scale) for n in shape[:2]) > MAX_RESCALED_PIXELS:
        raise ValueError("This image tile is too large for the chosen cell size. Start a new run with smaller tiles.")


def tile_layout(job, model_pixel_size=.5):
    info = resolution_info(job)
    factor = info["factor"]
    overlap = math.ceil(job["overlap_pixels"] / factor)
    if job["settings"].get("method") == "instanseg":
        scale = info["um_per_px"] / model_pixel_size
    else:
        value = diameter(job["settings"])
        value = value / factor if value is not None else None
        scale = 30 / value if value is not None else 1
        overlap = max(overlap, math.ceil(2 * (value if value is not None else 30 / factor)))
    size = min(job["tile_pixels"], math.floor(math.sqrt(MAX_RESCALED_PIXELS) / scale))
    if size < 1:
        raise ValueError("Choose a finer resolution for this image")
    return size, min(size // 2, overlap)


def emit(ev, **values):
    print(json.dumps({"ev": ev, **values}), flush=True, file=sys.__stdout__)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value))
    os.replace(temp, path)


_last_control: dict = {}


def read_control(root) -> dict:
    """control.json; when it cannot be read now, the one this run read last (replaced on NTFS through ntfs-3g it is
    missing for a moment, which must neither end the run nor read as running), else {}."""
    try:
        control = json.loads((root / "control.json").read_text())
    except (OSError, ValueError):
        return _last_control.get(str(root), {})
    _last_control[str(root)] = control
    return control


def checkpoint(root, done):
    """A stop takes precedence over pause. Acknowledgements are only emitted between committed tiles."""
    said = None
    while True:
        control = read_control(root)
        if control.get("stop"):
            return False
        state = "paused" if control.get("paused") else "running"
        if state != said:
            emit("state", state=state, done=done)
            said = state
        if state == "running":
            return True
        time.sleep(.1)


def tile_boxes(window, size, overlap):
    x0, y0, x1, y1 = window
    step = size - overlap
    # Stop once a tile reaches the edge; don't schedule an extra strip already covered by its overlap.
    for y in range(y0, max(y0+1, y1-overlap), step):
        for x in range(x0, max(x0+1, x1-overlap), step):
            yield [x, y, min(x + size, x1), min(y + size, y1)]


def tile_sources(root, i, count):
    """The tile that found each cell in tile i's file (restored spares keep theirs); older runs: tile i for all."""
    path = root / f"tile-{i:06d}.sources.json"
    if path.exists():
        sources = json.loads(path.read_text())
        if len(sources) == count:
            return sources
    return [i] * count


def read_spares(root, i):
    """[(owner tile, cell)] that tile i kept aside for tiles that had not run yet (stitch.py)."""
    from shapely.geometry import shape
    path = root / f"spare-{i:06d}.json"
    if not path.exists():
        return []
    return [(s["owner"], shape(s["geometry"])) for s in json.loads(path.read_text())["spares"]]


def restore_spares(root, boxes, i, owned, spares, geometry):
    """Tile i's share of the spare cells (stitch.py), as soon as their owning tile has run, as (cell, tile that found
    it): tile i's spares whose owning tile is finished (or none) and whose owning tile's cells do not cover them, and
    finished neighbours' spares that tile i owns and its own cells do not cover. Spares owned by tiles still to run wait
    in spare-<i>.json. Tiles run in plan order.

    Where both tiles found the cell but disagree on its outline (the owning tile's covers at least half of the spare),
    the owning tile's wins: it sees the cell away from its edges, with more context. sopa's union of the two would add
    the edge tile's extra pixels instead; this is the usual centre-wins rule of tiled segmentation."""
    from shapely.geometry import mapping
    from .stitch import neighbours, owners, uncovered
    ox, oy, sx, sy = geometry
    near = [i, *neighbours(boxes, i)]
    centres = np.array([[(c.centroid.x-ox)/sx, (c.centroid.y-oy)/sy] for c in spares]).reshape(-1, 2)
    owner = owners(centres, boxes, near)
    restored, waiting = [], []
    for k in sorted(set(owner.tolist())):
        group = [c for c, o in zip(spares, owner) if o == k]
        if k in (-1, i):
            restored += [(c, i) for c in uncovered(owned, group)]
        elif k < i:
            restored += [(c, i) for c in uncovered(read_shapes(root / f"tile-{k:06d}.geojson"), group)]
        else:
            waiting += [{"owner": k, "geometry": mapping(c)} for c in group]
    for j in near[1:]:
        if j < i:
            mine = [c for o, c in read_spares(root, j) if o == i]
            restored += [(c, j) for c in uncovered(owned + [c for c, _ in restored], mine)]
    atomic_json(root / f"spare-{i:06d}.json", {"spares": waiting})
    return restored


def polygons_in_region(polygons, region):
    """Preserve holes and all pieces of a clipped cell; never save geometry outside a lasso."""
    import shapely
    clipped = shapely.intersection(shapely.make_valid(polygons), region)
    parts = []
    for shape in clipped:
        candidates = [shape] if shape.geom_type == "Polygon" else list(getattr(shape, "geoms", []))
        parts.extend(p for p in candidates if p.geom_type == "Polygon" and not p.is_empty and p.area > 0)
    return parts


def write_shapes(base, shapes, box, mmap=False):
    from shapely.geometry import mapping
    coords, offsets, bounds = [], [0], []
    for shape in shapes:
        ring = np.asarray(shape.exterior.coords, dtype=np.float32)
        coords.extend(ring)
        offsets.append(len(coords))
        bounds.append(shape.bounds)
    arrays = dict(offsets=np.array(offsets, np.uint32), coords=np.array(coords, np.float32).reshape(-1, 2),
                  bounds=np.array(bounds, np.float32).reshape(-1, 4), box=np.array(box))
    if mmap:
        base.mkdir(exist_ok=True)
        for key, value in arrays.items():
            with (base / f"{key}.tmp").open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
            os.replace(base / f"{key}.tmp", base / f"{key}.npy")
    else:
        temp = base.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temp, base.with_suffix(".npz"))
    atomic_json(base.with_suffix(".geojson"), {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": i + 1, "properties": {}, "geometry": mapping(shape)} for i, shape in enumerate(shapes)]})


def read_shapes(path):
    from shapely.geometry import shape
    return [shape(f["geometry"]) for f in json.loads(path.read_text())["features"]]


def summary(shapes):
    areas = np.array([p.area for p in shapes])
    return {"cells": len(shapes), "area_um2": dict(zip(("p10", "median", "p90"),
            [float(v) for v in np.percentile(areas, [10, 50, 90])])) if len(areas) else None}


def covered_area(region, clip, boxes, records):
    """Proceed's one-piece region. Planned and committed boxes are in the sample's working frame.

    Plans are in row order, including when tissue filtering skipped tiles. The next tile cuts a staircase, bounded
    by the boxes that actually ran, so a sparse first row cannot claim the full width of the sample. A null region
    means the finished part is too small (or too complicated) for proseg's region input; null coverage means complete.
    """
    import shapely
    from shapely.geometry import Polygon, box as make_box
    from sopa_companion.engine.runs import RunError, region_spec

    done, total = len(records), len(boxes)
    wanted = make_box(*region["bbox"])
    if region["kind"] == "polygon":
        wanted = wanted.intersection(Polygon(region["points"]))
    if done == total and wanted.area > 0 and clip.area >= .99 * wanted.area:
        return None
    result = dict(region=None, done=done, of=total, why="tiles" if done < total else "image", pieces=0)
    if not records:
        return result
    area = clip
    if done < total:
        finished = np.asarray([r["box"] for r in records], float)
        x0, y0 = finished[:, :2].min(axis=0)
        x1, y1 = finished[:, 2:].max(axis=0)
        next_x, next_y = boxes[done][:2]
        bands = []
        if next_y > y0:
            bands.append(make_box(x0, y0, x1, min(y1, next_y)))
        if next_x > x0:
            bands.append(make_box(x0, y0, min(x1, next_x), y1))
        area = clip.intersection(shapely.union_all(bands))
    parts = [p for p in shapely.get_parts(shapely.make_valid(area)) if p.geom_type == "Polygon" and p.area > 0]
    result["pieces"] = len(parts)
    if not parts:
        return result
    area = shapely.set_precision(max(parts, key=lambda p: p.area), .01).simplify(0, preserve_topology=True)
    if area.is_empty or area.geom_type != "Polygon" or area.interiors:
        return result
    bounds = list(area.bounds)
    if area.equals(make_box(*bounds)):
        spec = dict(kind="box", box=bounds)
    else:
        points = [list(p) for p in area.exterior.coords[:-1]]
        if len(points) > 5000:
            return result
        spec = dict(kind="polygon", points=points)
    try:
        result["region"] = region_spec(spec, region["bbox"])
    except RunError:
        pass
    return result


def resolve_shapes(shapes, patches, records, pixel):
    """Match duplicates in the full cell or, for clipped cells, in the shared tile strip."""
    import shapely
    from shapely.geometry import box as make_box

    if not shapes:
        return []
    parents = list(range(len(shapes)))
    members = [{p} for p in patches]
    boxes = [r["box"] for r in records]
    edges = []
    for i, j in shapely.STRtree(shapes).query(shapes, predicate="intersects").T:
        if i >= j or patches[i] == patches[j]:
            continue
        a, b = shapes[i], shapes[j]
        intersection = a.intersection(b).area
        if intersection <= 0:
            continue
        score = intersection / min(a.area, b.area)
        ab, bb = boxes[patches[i]], boxes[patches[j]]
        # Both predictions reach facing tile edges. A full-cell overlap fraction is misleading here:
        # a large cell can agree throughout the strip while most of each fragment lies outside it.
        clipped = any(
            ab[k] < bb[k] < ab[k+2] < bb[k+2] and
            a.bounds[k+2] >= ab[k+2]-pixel[k] and b.bounds[k] <= bb[k]+pixel[k] or
            bb[k] < ab[k] < bb[k+2] < ab[k+2] and
            b.bounds[k+2] >= bb[k+2]-pixel[k] and a.bounds[k] <= ab[k]+pixel[k]
            for k in (0, 1))
        if score < .5 and clipped:
            shared = make_box(max(ab[0], bb[0]), max(ab[1], bb[1]), min(ab[2], bb[2]), min(ab[3], bb[3]))
            area = max(a.intersection(shared).area, b.intersection(shared).area)
            if area > 0:
                score = intersection / area
        if score >= .5:
            edges.append((-score, i, j))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for _, i, j in sorted(edges):
        i, j = root(i), root(j)
        # Transitive joins at four-tile corners must not collapse distinct cells from the same tile.
        if i != j and members[i].isdisjoint(members[j]):
            parents[j] = i
            members[i].update(members[j])
    groups = {}
    for i, shape in enumerate(shapes):
        groups.setdefault(root(i), []).append(shape)
    return [shapely.union_all(parts) for parts in groups.values()]


@lru_cache(maxsize=2)
def image_array(path):
    import zarr
    return zarr.open_array(path, mode="r")


def image_pixels(job, window, array=None, step=1):
    """The same channel order, zero padding and raw intensities for segmentation and patch export."""
    from sopa_companion.engine.zarr_samples import ZarrSamples
    image, settings = job["image"], job["settings"]
    _, level, _ = job_level(job)
    if array is None:
        array = image_array(str(Path(job["path"]) / level["path"]))
    instanseg = settings.get("method") == "instanseg"
    names = settings.get("channels", [settings.get("nucleus")])
    if "channels" not in settings and settings.get("cytoplasm") is not None:
        names.append(settings["cytoplasm"])
    selected = [image["channels"].index(name) for name in names]
    x0, y0, x1, y1 = window
    pixels = np.zeros(((y1-y0+step-1)//step, (x1-x0+step-1)//step, len(selected) if instanseg else 3), np.float32)
    for c, channel in enumerate(selected):
        pixels[:, :, c] = ZarrSamples._plane(array, level, channel, slice(y0, y1, step), slice(x0, x1, step))
    return pixels


def run(job, segment=None, join_only=False):
    join_only = join_only or job.get("join_only", False)
    if join_only:
        return _run(job, None, join_only=True)
    if segment is None and job["settings"].get("method") == "instanseg":
        from .stain_instanseg import InstanSegProcess
        with InstanSegProcess(job) as segment:
            return _run(job, segment)
    return _run(job, segment)


def _run(job, segment, join_only=False):
    import zarr
    from shapely.affinity import affine_transform
    from shapely.geometry import Polygon, box as make_box
    from sopa.shapes import vectorize
    from sopa_companion.engine.zarr_samples import ZarrSamples
    from .stitch import cell_diameter, needed_overlap, own_masks, solve_conflicts, whole_cell_sizes
    from .tissue import keep_inside, tissue_area, touches

    out = Path(job["out"])
    root = out / "stain"
    settings, image = job["settings"], job["image"]
    level_index, level, factor = job_level(job)
    geometry = ZarrSamples._geometry(image, level)
    ox, oy, sx, sy = geometry
    if settings.get("method") == "instanseg":
        settings = {**settings, "pixel_size": math.sqrt(sx * sy)}
    if not join_only and hasattr(segment, "start"):
        segment.start()
    model_pixel_size = getattr(segment, "model_pixel_size", .5)
    size, overlap = tile_layout(job, model_pixel_size)
    region = job["region"]
    clip = make_box(*region["bbox"]).intersection(
        make_box(ox, oy, ox+level["shape"][-1]*sx, oy+level["shape"][-2]*sy))
    if region["kind"] == "polygon":
        polygon = Polygon(region["points"])
        if not polygon.is_valid:
            raise ValueError("The lasso crosses itself")
        clip = clip.intersection(polygon)
    window = ZarrSamples._window(ZarrSamples, image, level, region["bbox"])
    manifest, plan, tiling = root / "tiles.json", root / "plan.json", root / "tiling.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    tissue = tissue_area(job["tissue"]) if job.get("tissue") else None
    # Runs created before tile ownership keep the duplicate-matching join they started with.
    own = job.get("stitch") == "own"
    array = None
    probe = {}

    def world_box(b):
        return [ox+b[0]*sx, oy+b[1]*sy, ox+b[2]*sx, oy+b[3]*sy]

    def planned(size, overlap):
        boxes = [b for b in tile_boxes(window, size, overlap) if clip.intersects(make_box(*world_box(b)))]
        kept = [b for b in boxes if touches(tissue, world_box(b))]
        return kept, len(boxes) - len(kept)

    def pixels(window):
        nonlocal array
        if array is None:
            array = zarr.open_array(str(Path(job["path"]) / level["path"]), mode="r")
        return image_pixels(job, window, array)

    if plan.exists():
        boxes = json.loads(plan.read_text())
        layout = json.loads(tiling.read_text()) if tiling.exists() else {}
    else:
        # Runs started before adaptive tiling retain their original grid and committed tile numbers.
        if records:
            size, overlap = job["tile_pixels"], job["overlap_pixels"]
        boxes, skipped = planned(size, overlap)
        if not boxes and tissue is not None:
            raise ValueError("This region has no tissue. Untick Tissue only to use it anyway.")
        layout = {"tile_px": size, "overlap_px": overlap, "tiles_skipped": skipped,
                  "cell_diameter_px": None, "cell_px_median": None}
        if own and boxes and not records and not join_only:
            # Measure the cells on central tiles first (up to three, until 20 whole cells): the overlap must be wider
            # than the cells are (stitch.py).
            if segment is None:
                segment = Cellpose(job, out)
            centre = np.mean([world_box(b) for b in boxes], axis=0)
            order = np.argsort([np.hypot(*(np.subtract(world_box(b), centre)[:2])) for b in boxes])
            sizes = []
            for k in order[:3].tolist():
                check_tile((boxes[k][3]-boxes[k][1], boxes[k][2]-boxes[k][0]), settings, factor, model_pixel_size)
                probe[tuple(boxes[k])] = np.asarray(segment(pixels(boxes[k])), dtype=np.uint32)
                sizes.extend(whole_cell_sizes(probe[tuple(boxes[k])]))
                if len(sizes) >= 20:
                    break
            layout["cell_diameter_px"] = cell_diameter(sizes)
            layout["cell_px_median"] = float(np.median(sizes)) if len(sizes) else None
            wider = needed_overlap(layout["cell_diameter_px"], overlap, size)
            if wider > overlap:
                layout["overlap_px"] = wider
                boxes, layout["tiles_skipped"] = planned(size, wider)
                probe.clear()
        atomic_json(tiling, layout)
        atomic_json(plan, boxes)
    if not boxes:
        raise ValueError("The region is outside the image")
    # The tile manifest is the commit point. Ignore and replace any files left by a crash before it.
    shapes, patches = [], []
    for i in range(len(records)):
        part = read_shapes(root / f"tile-{i:06d}.geojson")
        shapes.extend(part)
        patches.extend(tile_sources(root, i, len(part)))
    progress = {"done": len(records), "total": len(boxes), "revision": len(records), "resolved": False, "complete": False}
    emit("plan", progress=progress, summary=summary(shapes), tiles_skipped=layout.get("tiles_skipped", 0),
         tiling=layout)
    for i in range(len(records), len(boxes)):
        if join_only or not checkpoint(root, i):
            break
        started = time.monotonic()
        x0, y0, x1, y1 = boxes[i]
        check_tile((y1-y0, x1-x0), settings, factor, model_pixel_size)
        world = [ox+x0*sx, oy+y0*sy, ox+x1*sx, oy+y1*sy]
        area = make_box(*world).intersection(clip)
        if segment is None:
            segment = Cellpose(job, out)
            started = time.monotonic()
        masks = probe.pop(tuple(boxes[i]), None)
        if masks is None:
            masks = np.asarray(segment(pixels(boxes[i])), dtype=np.uint32)
        if masks.shape != (y1-y0, x1-x0):
            raise ValueError("The model returned a mask with the wrong shape")
        base = root / f"tile-{i:06d}"
        mask_temp = base.with_suffix(".masks.npz.tmp")
        with mask_temp.open("wb") as stream:
            np.savez_compressed(stream, masks=masks, pixel_box=boxes[i], geometry=geometry, level=level_index)
        os.replace(mask_temp, base.with_suffix(".masks.npz"))

        def cells(label_image):
            part = vectorize(label_image)
            # sopa contours use pixel centres at integer coordinates; image origins describe pixel corners.
            part = [affine_transform(p, [sx, 0, 0, sy, ox+(x0+.5)*sx, oy+(y0+.5)*sy]) for p in part.geometry]
            return keep_inside(polygons_in_region(part, clip), tissue)

        if own:
            owned, spare = own_masks(masks, boxes, i)
            part = cells(owned)
            back = restore_spares(root, boxes, i, part, cells(spare), geometry)
            # Each restored spare keeps the tile that found it, so sopa's join compares it with the owning tile's cells.
            sources = [i] * len(part) + [source for _, source in back]
            part += [cell for cell, _ in back]
            atomic_json(root / f"tile-{i:06d}.sources.json", sources)
        else:
            part = cells(masks)
            sources = [i] * len(part)
        write_shapes(base, part, world)
        shapes.extend(part)
        patches.extend(sources)
        records.append({"tile": i, "box": world, "cells": len(part), "seconds": time.monotonic()-started})
        atomic_json(manifest, records)
        progress = {**progress, "done": i+1, "revision": i+1}
        emit("tile", progress=progress, tile=records[-1], summary=summary(shapes))
    # Join the original fragments, also for a partial stop; a resume repeats this over all committed tiles.
    emit("state", state="finishing", done=len(records))
    if own:
        # Spares whose owner tile never ran (a stop part-way) come back where nothing covers them; then sopa's join.
        from .stitch import uncovered
        late, late_tiles = [], []
        for i in range(len(records)):
            for owner, cell in read_spares(root, i):
                if owner >= len(records):
                    late.append(cell)
                    late_tiles.append(i)
        back = {id(c) for c in uncovered(shapes, late)}
        shapes, patches = (shapes + [c for c in late if id(c) in back],
                           patches + [t for c, t in zip(late, late_tiles) if id(c) in back])
        resolved = keep_inside(solve_conflicts(shapes, patches), tissue)
    else:
        resolved = keep_inside(resolve_shapes(shapes, patches, records, (sx, sy)), tissue)
    covered = covered_area(region, clip, [world_box(b) for b in boxes], records)
    if covered and covered["region"]:
        spec = covered["region"]
        clip = clip.intersection(Polygon(spec["points"]) if spec["kind"] == "polygon" else make_box(*spec["box"]))
    resolved = polygons_in_region(resolved, clip)
    final = f"resolved-{len(records):06d}"
    write_shapes(root / final, resolved, region["bbox"], mmap=True)
    target = out / "output" / "boundaries.geojson"
    os.replace((root / final).with_suffix(".geojson"), target)
    progress = {**progress, "done": len(records), "revision": len(records)+1, "resolved": True,
                "complete": len(records) == len(boxes), "resolved_dir": final}
    result = {"progress": progress, "summary": summary(resolved), "covered": covered}
    atomic_json(root / "result.json", result)
    emit("done", **result)


class Cellpose:
    def __init__(self, job, out):
        settings = job["settings"]
        diameter(settings)
        import torch
        from cellpose import models
        from sopa_companion.engine.cellpose_settings import eval_kwargs
        self.factor = job_level(job)[2]
        self.full_settings = settings
        self.settings = at_level({**settings, **eval_kwargs(settings)}, self.factor)
        self.torch = torch
        record = model_file(settings, out, models)
        self.models, self.path = models, record["model_path"]
        # CUDA's GPU, else Apple's (MPS) where it passes a short test (else the CPU, said as a fallback), else the CPU;
        # an error of the GPU (out of memory, an operation MPS lacks) moves the run to the CPU once (fallback). A torch
        # without OutOfMemoryError (an old or a fake one) has RuntimeError.
        self.device, self.gpu, skipped = choose_device(torch)
        if skipped:
            emit("fallback", device="cpu", reason=skipped)
        self.errors = (RuntimeError, NotImplementedError, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
        try:
            self.model = self.load()
        except self.errors as exc:
            if not self.gpu:
                raise
            self.fallback(exc)
        atomic_json(out / "stain" / "model.json", record)
        emit("hello", **record, **resolution_info(job), cellpose=importlib.metadata.version("cellpose"),
             device=self.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
             nice=portable.niceness())

    def load(self):
        if not Path(self.path).is_file():
            raise ValueError("The model file is missing. Choose a model and start a new run.")
        return self.models.CellposeModel(gpu=self.gpu, pretrained_model=self.path,
                                         device=self.torch.device(self.device))

    def fallback(self, error):
        was = self.device
        self.gpu, self.device, self.model = False, "cpu", None
        empty_cache(self.torch, was)
        emit("fallback", device="cpu", reason=str(error))
        self.model = self.load()

    def __call__(self, pixels):
        check_tile(pixels.shape, getattr(self, "full_settings", self.settings), getattr(self, "factor", 1))
        from sopa_companion.engine.cellpose_settings import eval_kwargs
        args = eval_kwargs(self.settings)        # older runs: their three numbers, Cellpose's defaults for the rest
        try:
            return self.model.eval(pixels, channel_axis=-1, **args)[0]
        except self.errors as exc:
            if not self.gpu:
                raise
            self.fallback(exc)
            return self.model.eval(pixels, channel_axis=-1, **args)[0]


def main():
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    portable.nicest()                      # niceness 19 (Windows: the idle priority class)
    try:
        with redirect_stdout(sys.stderr):  # third-party status text never enters the JSON event stream
            if sys.argv[1] == "--models":
                emit("models", models=discover_models())
            else:
                args = [a for a in sys.argv[1:] if a != "--join-only"]
                run(json.loads(Path(args[0]).read_text()), join_only="--join-only" in sys.argv[1:])
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
