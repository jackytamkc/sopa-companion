"""Committed removed areas and size/brightness ranges of a stain run.

The model's output is unchanged. Outline ids refer to the whole current set; vectors omit only removed areas,
while the page hides out-of-range cells using PLCV values. Counts, patches and saves use the same pure rules.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading

import numpy as np

from . import stain_intensity, stain_rules
from .stain_rules import centres_and_areas, inside
from .runs import RunError, atomic_text

MAX_AREAS = 200
MAX_POINTS = 5000
MAX_MIN_AREA = stain_rules.MAX_SIZE            # µm²: a size rule above this would remove every cell
MAX_BRIGHTNESS = stain_rules.MAX_BRIGHTNESS          # image units: far above any stored pixel value


def meet(a, b):
    """Whether two rings share any area or edge: a vertex of one inside the other, or two edges crossing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return False
    if (a.max(axis=0) < b.min(axis=0)).any() or (b.max(axis=0) < a.min(axis=0)).any():
        return False
    if inside(a, b).any() or inside(b, a).any():
        return True
    p, r = a, np.roll(a, -1, axis=0) - a                    # edges of a: p + t r
    q, s = b, np.roll(b, -1, axis=0) - b                    # edges of b: q + u s
    lo, hi = np.minimum(q, q + s), np.maximum(q, q + s)
    for pk, rk in zip(p, r):
        near = ~((np.maximum(pk, pk + rk) < lo).any(axis=1) | (np.minimum(pk, pk + rk) > hi).any(axis=1))
        if not near.any():
            continue
        qn, sn = q[near], s[near]
        denom = rk[0] * sn[:, 1] - rk[1] * sn[:, 0]
        d = qn - pk
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (d[:, 0] * sn[:, 1] - d[:, 1] * sn[:, 0]) / denom
            u = (d[:, 0] * rk[1] - d[:, 1] * rk[0]) / denom
        if ((denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any():
            return True
    return False


def summary(areas):
    """The helper's summary (zarr_bridge.stain.summary) from cell areas."""
    return {"cells": int(len(areas)), "area_um2": dict(zip(("p10", "median", "p90"),
            [float(v) for v in np.percentile(areas, [10, 50, 90])])) if len(areas) else None}


class StainExclusions:
    def __init__(self, stain):
        self.stain = stain
        self.lock = threading.Lock()
        self.cells: dict[tuple, tuple] = {}          # (file, mtime_ns) -> (centres, areas)
        self.errors: dict[tuple, str] = {}            # brightness failures are local to their chooser
        self.bright: dict[tuple, np.ndarray] = {}    # (file, mtime_ns, channel) -> each cell's brightness
        self.measuring = threading.Lock()            # one measurement at a time: they read the image at full size

    # ---- storage ---------------------------------------------------------------------------------------------
    @staticmethod
    def path(run):
        return run.dir / "stain" / "exclusions.json"

    def load(self, run) -> dict:
        try:
            value = json.loads(self.path(run).read_text())
            if isinstance(value, dict) and isinstance(value.get("areas"), list):
                return stain_rules.normalise(value)
        except (OSError, ValueError):
            pass
        return stain_rules.normalise({})

    @staticmethod
    def digest(state):
        return stain_rules.digest(state)

    def areas(self, run) -> list[np.ndarray]:
        """The removed areas' rings, for the helper (patches, save)."""
        return [np.asarray(a["points"], float) for a in self.load(run)["areas"]]

    # ---- the current cells -------------------------------------------------------------------------------------
    @staticmethod
    def set_id(run, progress=None):
        progress = progress if progress is not None else run.meta["progress"]
        phase = "resolved" if progress.get("resolved") else "tiles"
        return f"{run.meta['id']}:{phase}:{progress.get('done', 0)}:{progress.get('revision', 0)}"

    def layout(self, run, progress=None):
        """(file, zero-based id base, count), checked against committed offsets, including tiles outside the view."""
        root = run.dir / "stain"
        progress = progress if progress is not None else run.meta["progress"]
        done = progress.get("done", 0)
        if progress.get("resolved"):
            file = root / progress.get("resolved_dir", f"resolved-{done:06d}")
            _, (_, areas) = self.file_cells(file)
            return [(file, 0, len(areas))]
        records = json.loads((root / "tiles.json").read_text())[:done] if done else []
        if len(records) != done:
            raise RunError("The cells changed. Wait for the next tile and try again.", 409)
        result, base = [], 0
        for i, record in enumerate(records):
            file = root / f"tile-{i:06d}.npz"
            _, (_, areas) = self.file_cells(file)
            count = len(areas)
            if "cells" in record and (type(record["cells"]) is not int or record["cells"] != count):
                raise RunError("The cells do not match their tile. Reopen the run to try again.", 409)
            result.append((file, base, count))
            base += count
        return result

    def files(self, run):
        return [file for file, _, _ in self.layout(run)]

    def file_cells(self, file):
        stamp = (file / "offsets.npy" if file.is_dir() else file).stat().st_mtime_ns
        key = (str(file), stamp)
        with self.lock:
            if key in self.cells:
                return key, self.cells[key]
        if file.is_dir():
            offsets = np.load(file / "offsets.npy", mmap_mode="r", allow_pickle=False)
            coords = np.load(file / "coords.npy", mmap_mode="r", allow_pickle=False)
            value = centres_and_areas(offsets, coords)
        else:
            with np.load(file, allow_pickle=False) as data:
                value = centres_and_areas(data["offsets"], data["coords"])
        value = (value[0], value[1].astype(np.float32))
        with self.lock:
            self.cells = {k: v for k, v in self.cells.items() if k[0] != key[0]}
            self.cells[key] = value
        return key, value

    @staticmethod
    def removing(state):
        state = stain_rules.normalise(state)
        return bool(state["areas"] or state["size"] or state["brightness"])

    @staticmethod
    def small(areas, state):
        return stain_rules.outside(areas, stain_rules.normalise(state)["size"])

    def values(self, run, file, channel):
        """A failed image read is unknown, hence kept; the chooser carries the error and a later request retries."""
        key = (str(run.dir), str(file), channel)
        try:
            values = self.file_brightness(run, file, channel)
        except Exception as exc:
            _, (_, areas) = self.file_cells(file)
            message = f"The image could not be read to measure {channel}: {exc}"
            with self.lock:
                self.errors[key] = message
            return np.full(len(areas), np.nan, np.float32)
        with self.lock:
            self.errors.pop(key, None)
        return values

    def dim(self, run, file, count, state):
        removed = np.zeros(count, bool)
        for channel, limits in stain_rules.normalise(state)["brightness"].items():
            removed |= stain_rules.outside(self.values(run, file, channel), limits)
        return removed

    def area_mask(self, run, file, state=None):
        """The file's cells whose centre lies in a removed area, which the outlines' vectors leave out (the ranges are
        applied page-side, and the Save's mask is Runs._stain_keep's); None while no rule removes anything."""
        state = self.load(run) if state is None else stain_rules.normalise(state)
        if not self.removing(state):
            return None
        _, (centres, _) = self.file_cells(file)
        return stain_rules.in_area(centres, state["areas"])

    def counts(self, run, state=None):
        """Per-area counts, total removed, kept summary, size removals and the union of brightness removals."""
        state = self.load(run) if state is None else stain_rules.normalise(state)
        per_area = [0] * len(state["areas"])
        removed, small, dim, kept = 0, 0, 0, []
        for file in self.files(run):
            _, (centres, areas) = self.file_cells(file)
            mask = self.small(areas, state)
            small += int(mask.sum())
            dark = self.dim(run, file, len(areas), state)
            dim += int(dark.sum())
            mask |= dark
            for k, area in enumerate(state["areas"]):
                hit = inside(centres, area["points"])
                per_area[k] += int(hit.sum())
                mask |= hit
            removed += int(mask.sum())
            kept.append(areas[~mask])
        return per_area, removed, summary(np.concatenate(kept) if kept else np.zeros(0)), small, dim

    # ---- brightness (engine/stain_intensity.py) ------------------------------------------------------------------
    def job(self, run) -> dict:
        return json.loads((run.dir / "stain" / "job.json").read_text())

    def channels(self, run) -> list[str]:
        return list(self.job(run)["image"]["channels"])

    def file_brightness(self, run, file, channel) -> np.ndarray:
        """Each of the file's cells' brightness in `channel`: kept in memory and in <run>/stain/brightness/ (named by
        the file, the channel's index, the measure and the file's time, so new outlines are measured again), measured
        otherwise. Storing never creates the run's folder: a run deleted while it was measured stays deleted."""
        job = self.job(run)
        index = job["image"]["channels"].index(channel)
        key, (_, areas) = self.file_cells(file)
        cache = (*key, channel)
        with self.lock:
            if cache in self.bright:
                return self.bright[cache]
        folder = run.dir / "stain" / "brightness"
        stored = folder / f"{file.name}-c{index}-{stain_intensity.CACHE_TAG}-{key[1]}.npy"
        try:
            values = np.load(stored, allow_pickle=False)
            if values.shape != areas.shape or values.dtype != np.float32:
                raise ValueError("The brightness values do not match the cells")
        except (OSError, ValueError):
            with self.measuring:
                try:
                    values = np.load(stored, allow_pickle=False)     # measured meanwhile, for another ask
                    if values.shape != areas.shape or values.dtype != np.float32:
                        raise ValueError("The brightness values do not match the cells")
                except (OSError, ValueError):
                    values = np.asarray(self._measure_file(job, file, channel), np.float32)
                    if values.shape != areas.shape:
                        raise ValueError("The brightness values do not match the cells")
                    try:
                        folder.mkdir(exist_ok=True)          # never parents: stain/ gone means the run is gone
                        temp = stored.with_name(stored.name + ".tmp")
                        with temp.open("wb") as stream:
                            np.save(stream, values, allow_pickle=False)
                        os.replace(temp, stored)
                    except FileNotFoundError:
                        pass
                    for old in folder.glob(f"{file.name}-c{index}-*.npy"):
                        if old != stored:
                            old.unlink(missing_ok=True)
        with self.lock:
            self.bright = {k: v for k, v in self.bright.items() if (k[0], k[2]) != (cache[0], cache[2])}
            self.bright[cache] = values
        return values

    def _measure_file(self, job, file, channel):
        if file.is_dir():
            offsets = np.load(file / "offsets.npy", allow_pickle=False)
            coords = np.load(file / "coords.npy", allow_pickle=False)
        else:
            with np.load(file, allow_pickle=False) as data:
                offsets, coords = data["offsets"], data["coords"]
        array = self.stain.runs.zarr._array(job["path"], job["image"]["levels"][0])
        read, frame, size = stain_intensity.job_reader(job, channel, array)
        return stain_intensity.brightness(read, frame, offsets, coords, size)

    def measure(self, run, channel):
        """Every cell's brightness in this channel; unreadable files supply unknown values."""
        values = [self.values(run, file, channel) for file in self.files(run)]
        return np.concatenate(values) if values else np.zeros(0, np.float32)

    def brightness(self, run, channel=None) -> dict:
        """GET …/stain/brightness: how bright the model's cells are in `channel` (the run's nucleus channel by default),
        to choose the rule from."""
        if run.meta.get("kind") != "stain":
            raise RunError("This is not an image run", 404)
        job = self.job(run)
        names, default = list(job["image"]["channels"]), stain_intensity.default_channel(job)
        channel = channel or default
        if channel not in names:
            raise RunError(f"This image has no channel {channel!r}", 422)
        if not self.files(run):
            raise RunError("The run has no cells yet", 409)
        values = self.measure(run, channel)
        measured = values[np.isfinite(values)]
        spread = [float(v) for v in np.percentile(measured, [10, 50, 90])] if len(measured) else [None] * 3
        return {"channel": channel, "channels": names, "default": default, "cells": int(len(values)),
                "measured": int(len(measured)), **dict(zip(("p10", "median", "p90"), spread))}

    # ---- the API -------------------------------------------------------------------------------------------------
    def public(self, run, state=None, counts=None):
        state = self.load(run) if state is None else stain_rules.normalise(state)
        per_area, removed, kept, small, dim = counts if counts is not None else self.counts(run, state)
        in_areas = 0
        files = self.files(run)
        current_files = {str(file) for file in files}
        for file in files:
            _, (centres, _) = self.file_cells(file)
            in_areas += int(stain_rules.in_area(centres, state["areas"]).sum())
        with self.lock:
            errors = sorted({v for (root, file, c), v in self.errors.items()
                             if root == str(run.dir) and file in current_files and c in state["brightness"]})
        return {"version": state["version"], "areas_version": state["areas_version"], "digest": self.digest(state),
                "count": len(state["areas"]), "cells": kept["cells"] + removed, "kept": kept["cells"],
                "removed": removed, "in_areas": in_areas, "size": state["size"], "by_size": small,
                "brightness": state["brightness"], "by_brightness": dim, "error": "; ".join(errors) or None,
                "areas": [{**a, "cells": n} for a, n in zip(state["areas"], per_area)]}

    def prepare(self, run, state=None):
        state = self.load(run) if state is None else stain_rules.normalise(state)
        for file in self.files(run):
            for channel in state["brightness"]:
                self.values(run, file, channel)

    def refresh(self, run):
        """Publish only the current committed state. Image read errors belong to ranges, never the segmentation."""
        self.prepare(run)
        with run.lock:
            state = self.load(run)
            counts = self.counts(run, state)
            public = self.public(run, state, counts)
            model = run.meta.get("summary_model") or run.meta.get("summary")
            kept = counts[2] if self.removing(state) else model
            self.stain.runs._set(run, exclusions={k: v for k, v in public.items() if k != "areas"},
                                 summary=kept, summary_model=model if self.removing(state) else None)

    def cell_bytes(self, run, set=None, channel=None):
        """PLCV v1 of the whole current set; a stale set/area request is a 409, never shifted values."""
        from .vectors import encode_cells
        if run.meta.get("kind") != "stain":
            raise RunError("This is not an image run", 404)
        with run.lock:
            progress, state = dict(run.meta["progress"]), self.load(run)
            current = self.set_id(run, progress)
            if set is not None and set != current:
                raise RunError("The cells changed. Reload their values and try again.", 409)
        job = self.job(run)
        channels = list(job["image"]["channels"])
        if channel is not None and channel not in channels:
            raise RunError(f"This image has no channel {channel!r}", 422)
        area, flags, bright = [], [], []
        layout = self.layout(run, progress)
        current_files = {str(file) for file, _, _ in layout}
        for file, _, count in layout:
            _, (centres, areas) = self.file_cells(file)
            area.append(areas)
            flags.append(stain_rules.in_area(centres, state["areas"]).astype(np.uint8))
            if channel is not None:
                bright.append(self.values(run, file, channel))
        area = np.concatenate(area) if area else np.zeros(0, np.float32)
        flags = np.concatenate(flags) if flags else np.zeros(0, np.uint8)
        brightness = {channel: np.concatenate(bright) if bright else np.zeros(0, np.float32)} if channel is not None else {}
        with self.lock:
            errors = sorted({v for (root, file, c), v in self.errors.items()
                             if root == str(run.dir) and file in current_files and c == channel})
        with run.lock:
            if self.set_id(run) != current or self.load(run)["areas_version"] != state["areas_version"]:
                raise RunError("The cells changed. Reload their values and try again.", 409)
        return encode_cells(area, flags, brightness,
                            {"set": current, "areas_version": state["areas_version"], "channels": channels,
                             "default": stain_intensity.default_channel(job), "dtype": job["image"].get("dtype"),
                             "error": "; ".join(errors) or None})

    def change(self, run, body, area_id=None, size=False, brightness=False):
        """One committed edit: a lasso, or one channel/size range. The page owns Undo/Redo of these commits."""
        if run.meta.get("kind") != "stain":
            raise RunError("Removing cells is for image runs", 404)
        limits = None
        if size or brightness:
            try:
                limits = stain_rules.check(body, "size" if size else "brightness", self.channels(run) if brightness else ())
            except ValueError as exc:
                raise RunError(str(exc), 422) from None
            if brightness and limits and self.stain.reviewable(run):
                self.measure(run, body["channel"])
        with run.lock:
            # Checked under the lock: a resume cannot slip in between the check and the change.
            if not self.stain.reviewable(run):
                raise RunError("Pause the run and wait for its cells before removing any", 409)
            state = self.load(run)
            version = body.get("version")
            if type(version) is not int or version != state["version"]:
                raise RunError("The removed cells changed in another window. Try again.", 409)
            if size:
                state["size"] = limits
            elif brightness:
                if limits is None:
                    state["brightness"].pop(body["channel"], None)
                else:
                    state["brightness"][body["channel"]] = limits
            elif area_id is None:
                points = body.get("points")
                if (not isinstance(points, list) or not 3 <= len(points) <= MAX_POINTS or
                        any(not isinstance(p, list) or len(p) != 2 or
                            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                                for v in p) for p in points)):
                    raise RunError("Draw a lasso around the cells to remove")
                ring = np.asarray(points, float)
                if abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], -1)))) <= 0:
                    raise RunError("Draw a lasso around the cells to remove")
                if len(state["areas"]) >= MAX_AREAS:
                    raise RunError(f"Remove at most {MAX_AREAS} areas")
                state["areas"].append({"id": state["next"], "points": ring.round(3).tolist()})
                state["next"] += 1
            else:
                if not any(a["id"] == area_id for a in state["areas"]):
                    raise RunError("This area was already put back", 404)
                state["areas"] = [a for a in state["areas"] if a["id"] != area_id]
            if not size and not brightness:
                state["areas_version"] += 1
            state["version"] += 1
            atomic_text(self.path(run), json.dumps(state))
        self.refresh(run)
        return self.public(run, state)
