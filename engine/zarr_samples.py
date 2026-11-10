"""SpatialData samples: what the user opens, how proseg's input and prior are prepared from it, and how results go
back into it.

Nothing is preset: a sample exists once the user opens a SpatialData zarr; opened samples are remembered in
var/samples.json (newest first) so the open dialog can offer them again. Everything that needs spatialdata or sopa
runs in the sopa environment through zarr_bridge/helper.py ([zarr] python): inspection, exporting shapes, assigning
transcripts to the prior (sopa's point-in-polygon join) and saving results (sopa's _read_proseg, add_parsed_table,
set_boundaries_attrs). Image pixels are read here directly with zarr-python, which is fast and needs no helper.

All coordinates are in the working frame: the intrinsic frame of the sample's transcripts (µm for Xenium); for a sample
without transcripts, the frame the helper places its images in (such a sample opens for stain segmentation only).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback

import numpy as np

from .. import portable
from ..names import BACKUP_FOLDERS, PROVENANCE_KEYS
from .samples import SampleError, _atomic_write, _channel_stats, _display_scale, _png, _stretch, _u16, _zstd
from . import children
from .config import PACKAGE
from .shape_query import ShapeQueries, Unavailable, WorkerError, load_outlines
from .vectors import VERSION as VECTORS_VERSION, encode, gather, hits, select

OVERVIEW_PX = 7000
STATS_PX = 4096
KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}")
ELEMENT_KINDS = ("images", "labels", "points", "shapes", "tables")
ROI = "region_of_interest"  # sopa's SopaKeys.ROI: the tissue's outline (tissue()), never cells, whatever its pieces
RECENT_SHOWN = 30          # samples the open dialog lists (GET /samples), the last opened first
RECENT_KEPT = 300          # samples known by id, their detail and layers answering, once off that list
NUCLEUS_STAIN = re.compile(r"dapi|hoechst", re.IGNORECASE)   # the stain picker's nucleus channel (main.js fillStainChannels)
PIXEL_TOLERANCE = 0.01     # an image within 1 % of one frame unit per pixel: the frame is likely pixels
# What an open found, said in the picker (the sample detail's `notes`; frontend/src/sample-notes.mjs): one
# short line each, the rest in its ⓘ. Page words: plain, no word of qc-copy.mjs BANNED_WORDS.
NOTE_IMAGE_ONLY = ("This sample has no transcripts: only stain segmentation runs here.",
                   "Proseg builds its cells from transcripts, and this sample holds images only. Stain segmentation finds "
                   "cells in the images.")
NOTE_NAMES = ("Names this app can use start with a letter, a digit or _, hold only a-z, A-Z, digits, _ . and -, and "
              "are at most 100 characters long. Rename a layer in the sample to use it here.")
OUTLINES_KEPT = 4          # exported shapes elements held as arrays (two samples' cells and nuclei)
# Segmentation layers and Compare: what is listed, in this order, and named by what
# made it when the zarr says so (describe_segmentation).
SEGMENTATION_GEOMETRY = ("Polygon", "MultiPolygon")
SOURCES = ("xenium", "imported", "proseg", "cellpose", "instanseg", "other")
# InstanSeg is dormant (StainRuns.methods): the shapes it saved earlier are named as any saved stain segmentation.
SOURCE_NAMES = {"xenium": "Xenium cells", "imported": "earlier segmentation (imported)", "proseg": "proseg (this app)",
                "cellpose": "Cellpose", "instanseg": "saved stain segmentation"}
COUNTS_KEPT = 4            # counts of an element over other boxes kept beside its whole count (ZarrSamples.counts)
# Whose count Compare reads for an element (count_source; the helper's `count` and its file's `count`): a table's own,
# proseg's or Xenium's (its X), else "outlines", the transcripts inside its outlines.
COUNT_BY = ("proseg", "xenium", "outlines")
# An element whose folder did not answer (the sample share is an automount that can fail for a moment) is read again
# once after this long; still unreadable, the catalogue lists it at version UNREAD, and nothing is kept (nor pruned)
# under that: its outlines and counts are refused in plain words instead (ZarrSamples._readable_stamp).
STAMP_RETRY_S = 0.5
UNREAD = "0"


def _outline_files(key: str) -> re.Pattern:
    """The layers' outline files of an element: segmentations/<key>-<stamp>.v<n>.bin.zst (stamp "0": made before
    unreadable stamps were refused)."""
    return re.compile(re.escape(key) + r"-([0-9a-f]{12}|0)\.v[0-9]+\.bin\.zst")


def _count_files(key: str) -> re.Pattern:
    """Compare's count files of an element: counts/<key>-<rule>-<stamp>[-<box>].npz, rules named by the helper's
    count_rule (qv<quality>[-x<pattern digest>]-v<n>), or a table's own count's t<digest>-v<n> (ZarrSamples.counts).
    Groups: the rule, its version n, the stamp."""
    return re.compile(re.escape(key) + r"-([A-Za-z0-9.]+(?:-x[0-9a-f]{8})?-v([0-9]+))-([0-9a-f]{12}|0)"
                                       r"(?:-[-0-9._e+]+)?\.npz")


def describe_segmentation(key: str, made_by: dict | None) -> dict:
    """{name, source, model, table, coverage} of a shapes element from what the zarr says made it (helper
    `segmentations`): an app's stain save (its attrs on the shapes) -> "Cellpose" (InstanSeg's: "saved stain
    segmentation") with its model, and the pyramid level it segmented when not scale0 ("Cellpose · scale2",
    stain_levels.resolution_name); a table annotating it (its region) written by an app's proseg save (its uns
    record) -> "proseg (this app)", whose obs segmentation_method says "Imported Cell Segmentation" -> "earlier
    segmentation (imported)", whose methods are all Xenium's own ("Segmented by ...") -> "Xenium cells"; otherwise the
    key itself (source "other"). `table`: the annotating table's key; `coverage` {done, of}: a stain save that covers
    part of the tissue only (its tiles)."""
    made_by = made_by or {}
    shapes = next((s for s in made_by.get("shapes") or [] if s.get("key") == key), None) or {}
    tables = [t for t in made_by.get("tables") or [] if key in (t.get("region") or [])]
    # The table that says most (an app's save, or Xenium's methods) first: a copy without methods says nothing.
    tables.sort(key=lambda t: (not t.get("proseg"), not (t.get("methods") or {}).get(key), t.get("key", "")))
    table = tables[0] if tables else None
    found = {"name": key, "source": "other", "model": None, "table": table["key"] if table else None, "coverage": None}
    app = shapes.get("app") or {}
    if app.get("kind") == "stain" or app.get("method") in ("cellpose", "instanseg"):
        source = "instanseg" if app.get("method") == "instanseg" else "cellpose"
        done, total = app.get("done"), app.get("total")
        partial = app.get("complete") is False and isinstance(done, int) and isinstance(total, int) and 0 <= done < total
        resolution = app.get("resolution") or (app.get("parameters") or {}).get("resolution") or {}
        from .stain_levels import level_name, resolution_name
        level = resolution_name(resolution)                  # a save at scale0 (the image as stored) is not named
        label = f" · {level}" if level and level != level_name(0) else ""
        return {**found, "name": SOURCE_NAMES[source] + label, "source": source,
                "model": str(app["model"]) if app.get("model") else None,
                "coverage": {"done": done, "of": total} if partial else None}
    if table is None:
        return found
    methods = [str(m) for m in ((table.get("methods") or {}).get(key) or {})]
    if table.get("proseg"):
        source = "proseg"
    elif any("imported" in m.lower() for m in methods):
        source = "imported"
    elif methods and all(m.startswith("Segmented by") for m in methods):
        source = "xenium"
    else:
        return found
    return {**found, "name": SOURCE_NAMES[source], "source": source}


def count_source(key: str, made_by: dict | None) -> dict:
    """{count_by, count_table}: whose count Compare reads for a shapes element (COUNT_BY), from the tables annotating it
    (helper `segmentations`): proseg's own, the X of a table whose uns carries proseg's run record (the helper's
    `proseg_count`: proseg's own output, or a proseg save of this app, `proseg`; the first by key of several); else
    Xenium's, its cell table (describe_segmentation's "xenium": the table of Xenium's methods); else "outlines" (no
    table): a segmentation with no count of its own, Cellpose's, a stain's, an imported one's (its table is an outline
    count too), is counted by the transcripts inside its outlines."""
    tables = [t for t in (made_by or {}).get("tables") or [] if key in (t.get("region") or []) and t.get("key")]
    own = sorted(str(t["key"]) for t in tables if t.get("proseg_count") or t.get("proseg"))
    if own:
        return {"count_by": "proseg", "count_table": own[0]}
    described = describe_segmentation(key, made_by)
    if described["source"] == "xenium" and described["table"]:
        return {"count_by": "xenium", "count_table": described["table"]}
    return {"count_by": "outlines", "count_table": None}


def sample_id(path: str) -> str:
    return "z" + hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:12]


def _finite(value):
    """`value` with every number that is not finite (NaN, infinity) as None, as the routes' strict JSON needs (a
    shapes element of empty geometries has a NaN extent, and its sample's open answered 500)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _extent(value) -> list[float] | None:
    """An extent [x0, y0, x1, y1] of four finite numbers, else None (an element whose geometries are all empty has a
    NaN one)."""
    if isinstance(value, (list, tuple)) and len(value) == 4 and \
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        return [float(v) for v in value]
    return None


def _usable(shape: dict) -> bool:
    """Whether a shapes element can be offered as cells or listed as a segmentation: a name the routes serve (KEY: a
    name they refuse was offered, even suggested, then its outlines answered 404), not sopa's
    region_of_interest (the tissue), and outlines somewhere (a finite extent: an element whose geometries
    are all empty has none)."""
    key = shape.get("key")
    return isinstance(key, str) and bool(KEY.fullmatch(key)) and key != ROI and _extent(shape.get("extent")) is not None


def _rgb(image: dict) -> bool:
    """An RGB image (sopa's rule: three uint8 channels), an H&E: no stain to segment cells in, nor to show first."""
    return len(image.get("channels") or []) == 3 and image.get("dtype") == "uint8"


def _um_per_px(image: dict) -> float:
    """Frame units per pixel of an image's full resolution (along x)."""
    return abs(image["to_frame"][0][0]) * image["levels"][0]["pixel"][0]


IMAGE_METADATA = ("zarr.json", ".zattrs", ".zgroup", ".zarray")      # an image's and its levels' metadata files


def _stat_identity(path: str) -> tuple:
    found = os.stat(path)                  # through a link (a view's linked image): the element it names
    return found.st_ino, found.st_size, found.st_mtime_ns


def image_stamp(path, key: str) -> str | None:
    """An image as it is on disk now: 12 hex digits of the identity, size and mtime of its folder, its metadata and each
    level's folder and array metadata (never its chunks), which change whenever the image is written again, its
    transformation, channels, shape or type alone included. None when its folder cannot be read now. The helper's
    inspect records it on each image (zarr_bridge/helper.py image_stamp, the same rule: that one is used when there)."""
    try:
        from ..zarr_bridge.helper import image_stamp as helpers
    except ImportError:
        helpers = None
    if helpers is not None:
        return helpers(path, key)
    folder = os.path.join(str(path), "images", key)
    try:
        found = [("", *_stat_identity(folder))]
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name in IMAGE_METADATA:
                    found.append((entry.name, *_stat_identity(entry.path)))
                elif entry.is_dir():
                    found.append((entry.name, *_stat_identity(entry.path)))
                    for name in IMAGE_METADATA:
                        try:
                            found.append((f"{entry.name}/{name}", *_stat_identity(os.path.join(entry.path, name))))
                        except FileNotFoundError:
                            pass
    except OSError:
        return None
    return hashlib.sha1(json.dumps(sorted(found)).encode()).hexdigest()[:12]


def image_identity(image: dict, path: str | None = None) -> str:
    """What the engine derives from an image is kept under (its overview, statistics): 12 hex digits
    of its stamp (as inspected, else as it is now) with how the inspection placed it (to_frame, each level's shape,
    origin and pixel size), its type and its channels' names. A rewrite of the image, or an inspection that reads it
    differently, makes another one."""
    stamp = image.get("stamp") or (image_stamp(path, image["key"]) if path else None)
    levels = [[lv.get("path"), lv.get("shape"), lv.get("origin"), lv.get("pixel")] for lv in image.get("levels") or []]
    content = [stamp, image.get("to_frame"), levels, image.get("dtype"), list(image.get("channels") or [])]
    return hashlib.sha1(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:12]


def frame_units(images: list[dict]) -> dict:
    """{frame, why} of the working frame when the helper did not say (an inspection made before it did): "pixel" when
    every usable image has one frame unit per pixel (within PIXEL_TOLERANCE), the helper's first sign (the
    images of the samples in pixels all had exactly 1.0, those in µm a fraction of it); "unknown" otherwise."""
    try:
        sizes = [_um_per_px(image) for image in images]
    except (KeyError, IndexError, TypeError):
        sizes = []
    if sizes and all(abs(size - 1.0) <= PIXEL_TOLERANCE for size in sizes):
        return {"frame": "pixel", "why": "every image has exactly one pixel per unit of the transcripts' coordinates"}
    return {"frame": "unknown", "why": ""}


def _shrink(plane: np.ndarray, max_px: int) -> np.ndarray:
    """A uint16 plane area-averaged down to at most max_px across (each pixel the mean of those it covers), the
    aspect kept; a plane already that small is returned as it is."""
    height, width = plane.shape
    if max(height, width) <= max_px:
        return plane
    from PIL import Image
    factor = max_px / max(height, width)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    small = Image.fromarray(plane.astype(np.float32)).resize(size, Image.Resampling.BOX)
    return np.clip(np.rint(np.asarray(small)), 0, 65535).astype("<u2")


def overview_channel(channels) -> int:
    """The channel the overview shows: the first nucleus stain (DAPI, Hoechst), as the stain picker chooses it, else the
    first (channel 0 was PolyT on MERSCOPE, a dim marker on CosMx)."""
    return next((index for index, name in enumerate(channels or []) if NUCLEUS_STAIN.search(str(name))), 0)


def _named(keys: list[str], longest: int = 40) -> str:
    """Element names for a one-line note: each cut to `longest` characters."""
    return ", ".join(key if len(key) <= longest else key[:longest - 1] + "…" for key in keys)


def open_notes(info: dict, points: dict | None, images: list[dict], image: dict | None) -> list[dict]:
    """What an open found that the page cannot see for itself, [{code, text: one plain line, detail: its ⓘ}], from the
    helper's inspection (`points` the transcripts used, None for a sample without; `images` those it can show,
    `image` the one it shows): no transcripts ("image_only"), coordinates that seem to be pixels
    ("pixel_frame"), several transcript elements of which one is used ("several_points"), z given as plane
    numbers ("z_index"), images left out ("skipped_images") or chosen without the sample saying so
    ("default_image"), and shapes whose names the app cannot use ("skipped_names")."""
    notes = []

    def note(code: str, text: str, detail: str):
        notes.append({"code": code, "text": text, "detail": detail})

    attrs = info.get("attrs") if isinstance(info.get("attrs"), dict) else {}
    if points is None:
        note("image_only", *NOTE_IMAGE_ONLY)
    units = info.get("units") if isinstance(info.get("units"), dict) else frame_units(images)
    if units.get("frame") == "pixel":
        why = str(units.get("why") or "").strip().rstrip(".")
        whose = "These transcripts seem" if points is not None else "This sample's coordinates seem"
        note("pixel_frame", f"{whose} to be in pixels, not micrometres: sizes shown in µm are pixels here.",
             (f"Why: {why}. " if why else "") + "The app takes the coordinates as micrometres, so the default box, the "
             "size limits and every area are in pixels for this sample.")
    if points is not None:
        elements = info.get("points_elements") or [p.get("key") for p in info.get("points") or []]
        elements = [str(key) for key in elements if key]
        choice = info.get("points_choice")        # what chose it: asked, attrs, name, or the first one listed
        decided = choice != "first" if choice else \
            points["key"] in (attrs.get("transcripts_dataframe"), attrs.get("transcripts"), "transcripts")
        if len(elements) > 1 and not decided:
            others = [key for key in elements if key != points["key"]]
            note("several_points", f"This sample has {len(elements)} transcript elements: this app uses "
                                   f"{points['key']} only.",
                 f"Left out: {', '.join(others)}. The sample does not say which element holds its transcripts (its "
                 "transcripts_dataframe attribute), so the app takes the first one listed.")
        z = info.get("z") if isinstance(info.get("z"), dict) else {}
        if z.get("kind") == "index":
            note("z_index", "The z values are plane numbers, not micrometres.",
                 "Proseg and the 3-D view read z as micrometres, so depths here count planes.")
    skipped = [s for s in info.get("skipped_images") or [] if isinstance(s, dict) and s.get("key")]
    for entry in info.get("images") or []:        # an inspection that does not list them: from the images it left out
        if entry.get("key") not in {s["key"] for s in skipped} and ("error" in entry or not entry.get("to_frame")):
            skipped.append({"key": entry["key"], "reason": entry.get("error") or
                            "it has no coordinate system in common with " +
                            ("the transcripts" if points is not None else "the image shown")})
    if skipped:
        keys = [str(s["key"]) for s in skipped]
        note("skipped_images", f"The image {_named(keys)} is not used here." if len(keys) == 1 else
                               f"{len(keys)} images are not used here: {_named(keys)}.",
             "\n".join(str(s.get("reason") or "") if str(s["key"]) in str(s.get("reason") or "") else
                       f"{s['key']}: {s.get('reason') or 'it cannot be placed'}" for s in skipped))
    if image is not None and len(images) > 1 and image["key"] != attrs.get("cell_segmentation_image"):
        note("default_image", f"This sample has {len(images)} images and does not say which one to show: this app shows "
                              f"{image['key']}.",
             "Stain segmentation lets you choose the image. Proseg runs show this one. A sample names its own with "
             "its cell_segmentation_image attribute.")
    refused = [str(s.get("key")) for s in info.get("shapes") or [] if not KEY.fullmatch(str(s.get("key")))]
    if refused:
        note("skipped_names", f"The layer {_named(refused)} has a name this app cannot use." if len(refused) == 1 else
                              f"{len(refused)} layers have names this app cannot use: {_named(refused)}.",
             f"Not used: {', '.join(refused)}. {NOTE_NAMES}")
    # The helper's own (what it read in the data, such as a cell column's "no cell" mark), after these; one of a code
    # said here is the helper's.
    given = [{"code": str(n.get("code") or ""), "text": n["text"].strip(), "detail": str(n.get("detail") or "").strip()}
             for n in info.get("notes") or [] if isinstance(n, dict) and isinstance(n.get("text"), str) and n["text"].strip()]
    return [n for n in notes if n["code"] not in {g["code"] for g in given}] + given


def is_spatialdata(path: Path) -> bool:
    """A zarr group with SpatialData elements (zarr v3 zarr.json or v2 .zgroup, plus an element folder)."""
    try:
        if not ((path / "zarr.json").is_file() or (path / ".zgroup").is_file()):
            return False
        return any((path / kind).is_dir() for kind in ELEMENT_KINDS)
    except OSError:
        return False


# ---- what a save of this app left in a sample, read from the element's own files (no helper) --------------------------
def shapes_provenance(path, key: str | None) -> dict | None:
    """The record a stain save wrote on the shapes it made (their attrs under the first of PROVENANCE_KEYS there is:
    run, result_revision, filters, …), from the element's zarr.json (v3) or .zattrs (v2); None when there is none or
    it cannot be read."""
    if not isinstance(key, str) or not KEY.fullmatch(key) or not path:
        return None
    folder = Path(path) / "shapes" / key
    for name in ("zarr.json", ".zattrs"):
        try:
            data = json.loads((folder / name).read_text())
        except (OSError, ValueError):
            continue
        attrs = data.get("attributes") if name == "zarr.json" and isinstance(data, dict) else data
        if not isinstance(attrs, dict):
            return None
        return next((attrs[name] for name in PROVENANCE_KEYS if isinstance(attrs.get(name), dict)), None)
    return None


def table_run(path, key: str | None) -> str | None:
    """The run a proseg save's table names (tables/<key>/uns/<the first of PROVENANCE_KEYS there is>/run, a string
    anndata wrote), or None."""
    if not isinstance(key, str) or not KEY.fullmatch(key) or not path:
        return None
    uns = Path(path) / "tables" / key / "uns"
    record = next((uns / name for name in PROVENANCE_KEYS if (uns / name).is_dir()), None)
    if record is None:
        return None
    try:
        import zarr
        value = zarr.open_array(store=str(record / "run"), mode="r")[()]
        value = value.item() if hasattr(value, "item") else value
        return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        return None


def unfinished_writes(path) -> set[tuple[str, str]]:
    """The elements a save into the sample at `path` wrote and never committed: its journal still says "writing" (the
    save was cut off; the next save into the sample rolls them back, zarr_bridge/helper.py _recover). Read from the
    sample's own backup folders (in either backup folder name, BACKUP_FOLDERS), by the helper's naming; nothing is
    touched."""
    try:
        from ..zarr_bridge.helper import _backup_folder
        store = Path(path).resolve()
    except (OSError, ImportError, RuntimeError, TypeError):
        return set()
    folders = []
    for name in BACKUP_FOLDERS:
        try:
            folders += list((store.parent / name).iterdir())
        except OSError:
            continue
    mine, found = _backup_folder(store), set()
    for folder in folders:
        if not mine.fullmatch(folder.name):
            continue
        try:
            journal = json.loads((folder / "journal.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(journal, dict) and journal.get("state") == "writing" and \
                journal.get("path", str(store)) == str(store):
            found |= {(str(item[0]), str(item[1])) for item in journal.get("written") or []
                      if isinstance(item, (list, tuple)) and len(item) == 2}
    return found


class HelperFailed(SampleError):
    pass


class ZarrSamples:
    def __init__(self, config: dict):
        section = config.get("zarr", {})
        self.python = section.get("python") or ""
        self.browse_start = str(Path(section.get("browse_start") or "~").expanduser())
        self.cache = Path(config["paths"]["cache_dir"]) / "zarr"
        self.registry = Path(config["paths"]["data_dir"]) / "samples.json"
        self.detail_um = float(config["proseg"]["max_region_um"])
        self.lock = threading.RLock()
        self.details: dict[str, dict] = {}
        self.building: dict[str, threading.Lock] = {}  # one detail build per sample, also shared by open and save
        self.arrays: dict[tuple, object] = {}
        self.outlines: dict[tuple, dict] = {}
        self.outline_stamps: dict[tuple, str | None] = {}   # the element stamp each of `outlines` was read at
        self.exporting: dict[tuple, threading.Lock] = {}
        # API mode (bind): the sample's name as the notebook gives it and the transcripts it names, by sample id.
        self.bound_names: dict[str, str] = {}
        self.bound_points: dict[str, str] = {}
        self.queries = ShapeQueries()      # the picker's box queries, answered off the engine's GIL
        try:     # every store ever saved into leaves a save lock (tests' and fixtures' copies too): drop stale ones
            from ..zarr_bridge.helper import prune_locks
            prune_locks()
        except Exception as exc:
            print(f"SoPA companion: could not prune stale zarr save locks: {exc}", flush=True)
        try:
            self.recent = [e for e in json.loads(self.registry.read_text()) if isinstance(e, dict) and "id" in e]
        except (OSError, ValueError):
            self.recent = []

    @property
    def available(self) -> bool:
        return bool(self.python) and Path(self.python).is_file()

    def close(self):
        self.queries.close()

    # ---- registry ---------------------------------------------------------------------------------------------
    # var/samples.json: every sample opened, the last first, RECENT_KEPT of them. The open dialog lists the first
    # RECENT_SHOWN that the user opened; the rest stay known by id, so their detail, layers and images still answer
    # (past 30 samples the detail said "open it again" while the layer routes opened it again, which put it
    # back on top). A sample a route opened again for one of its runs (`hidden`) is known without being listed.
    def list(self) -> list[dict]:
        with self.lock:
            return [dict(e) for e in self.recent if not e.get("hidden")][:RECENT_SHOWN]

    def known(self, sid: str) -> bool:
        with self.lock:
            return any(e["id"] == sid for e in self.recent)

    def _remember(self, detail: dict, shown: bool = True):
        with self.lock:
            entry = {"id": detail["id"], "name": detail["name"], "path": detail["path"], "kind": "zarr",
                     "opened": time.strftime("%Y-%m-%dT%H:%M:%S")}
            held = next((e for e in self.recent if e["id"] == detail["id"]), None)
            if not shown and held is not None:           # known already: its place on the list stays
                held.update(name=detail["name"], path=detail["path"])
            else:
                if not shown:
                    entry["hidden"] = True
                self.recent = [entry] + [e for e in self.recent if e["id"] != detail["id"]][:RECENT_KEPT - 1]
            _atomic_write(self.registry, json.dumps(self.recent, indent=1).encode())

    def forget(self, sid: str):
        with self.lock:
            self.recent = [e for e in self.recent if e["id"] != sid]
            _atomic_write(self.registry, json.dumps(self.recent, indent=1).encode())
            self.details.pop(sid, None)

    def path(self, sid: str) -> str:
        entry = self.entry(sid)
        if entry is None:
            raise SampleError("Unknown sample: open it again", 404)
        return entry["path"]

    def entry(self, sid: str) -> dict | None:
        """The registry's entry of a known sample ({id, name, path, ...}), listed or not; None for an unknown one."""
        with self.lock:
            entry = next((e for e in self.recent if e["id"] == sid), None)
            return dict(entry) if entry is not None else None

    # ---- browsing ---------------------------------------------------------------------------------------------
    def browse(self, path: str | None) -> dict:
        if not path:
            with self.lock:
                last = self.recent[0]["path"] if self.recent else None
            path = str(Path(last).parent) if last else self.browse_start
        folder = Path(path).expanduser()
        try:
            folder = folder.resolve()
            names = sorted(os.listdir(folder), key=str.lower)
        except FileNotFoundError:
            raise SampleError(f"{path} does not exist", 404) from None
        except NotADirectoryError:
            raise SampleError(f"{path} is not a folder") from None
        except OSError as exc:
            raise SampleError(f"Cannot read {path}: {exc.strerror or exc}") from None
        entries = []
        for name in names:
            if name.startswith("."):
                continue
            child = folder / name
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            entries.append({"name": name, "path": str(child), "kind": "zarr" if is_spatialdata(child) else "dir"})
        entries.sort(key=lambda e: (e["kind"] != "zarr", e["name"].lower()))
        shortcuts, seen = [], set()
        for label, target in [("Home", str(Path.home())), *[(f"Recent: {Path(e['path']).parent.name}", str(Path(e["path"]).parent)) for e in self.list()[:4]]]:
            if target not in seen and Path(target).is_dir():
                seen.add(target)
                shortcuts.append({"label": label, "path": target})
        return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None,
                "is_zarr": is_spatialdata(folder), "shortcuts": shortcuts, "entries": entries[:2000]}

    # ---- helper (sopa environment) ----------------------------------------------------------------------------
    def helper(self, command: str, args: dict, progress=None, should_stop=None, timeout: float = 3600) -> dict:
        """Run zarr_bridge/helper.py in the sopa environment; progress lines go to `progress(text, fraction)` (and never
        into the stderr lines kept for a failure's message, whoever listens); `should_stop()` returning True kills it."""
        if not self.available:
            raise SampleError("The sopa environment for SpatialData is not configured ([zarr] python)", 503)
        # The app by its package name where this interpreter needs it, whatever its folder is called; no display.
        env = children.env(children.kind(self.python))
        env.setdefault("OMP_NUM_THREADS", "16")
        # Its own session: a terminal's hang-up or Ctrl+C (sent to the whole foreground group) never cuts a save's
        # journalled write short, which is meant to finish or roll back by itself. A read-only
        # command goes with the engine instead (parent-death signal; this thread waits for it, so it outlives it).
        launcher = ["setpriv", "--pdeathsig", "TERM"] if command != "save" and shutil.which("setpriv") else []
        # The arguments go on stdin ("-"): one command-line argument is limited to 128 KiB, which a stain save's
        # provenance (every point of its removal areas) can pass, and the save could then never start. Only where it
        # writes (out, output) stays on the command line: the start's check for a save still at work finds it there
        # (leftovers.named). Written by a thread of its own, so a helper that never reads them is still stopped in time.
        given = json.dumps(args)
        named = [str(args[key]) for key in ("output", "out") if args.get(key)]
        process = subprocess.Popen([*launcher, self.python, "-m", f"{PACKAGE}.zarr_bridge.helper", command, "-", *named],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                   cwd=children.app_path(), **portable.new_session())
        if command != "save":
            portable.tie(process)          # macOS and Windows: as setpriv's death signal; a save is never tied

        def feed():
            try:
                with process.stdin:
                    process.stdin.write(given)
            except OSError:          # it ended (or was stopped) before reading them all: its exit says what happened
                pass

        threading.Thread(target=feed, daemon=True).start()
        tail: list[str] = []

        def read_progress():
            for line in process.stderr:
                line = line.rstrip()
                try:
                    message = json.loads(line)
                except ValueError:
                    message = None
                if isinstance(message, dict) and "progress" in message:
                    if progress:
                        progress(message["progress"], message.get("fraction"))
                    continue
                if line:
                    tail.append(line)
                    del tail[:-30]

        reader = threading.Thread(target=read_progress, daemon=True)
        reader.start()
        output: list[str] = []
        collector = threading.Thread(target=lambda: output.append(process.stdout.read()), daemon=True)
        collector.start()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if (should_stop and should_stop()) or time.monotonic() > deadline:
                process.kill()
                process.wait()
                raise HelperFailed("Stopped" if should_stop and should_stop() else f"{command} took too long", 409)
            time.sleep(.1)
        collector.join(5)
        reader.join(5)
        lines = [l for l in "".join(output).splitlines() if l.strip()]
        try:
            result = json.loads(lines[-1]) if lines else {}
        except ValueError:
            result = {}
        if process.returncode != 0 or "error" in result:
            message = result.get("error") or (tail[-1] if tail else f"helper exited with code {process.returncode}")
            # The page gets its one line; the engine log keeps what the helper printed (its traceback).
            print(f"SoPA companion: helper {command} failed: {message}" + "".join(f"\n  {line}" for line in tail),
                  flush=True)
            # A name clash the helper found itself (a save into names taken since the engine's own check) answers as
            # the engine's does: 409 with the names taken and the sample's names now, so the dialog can offer Replace.
            if isinstance(result.get("clash"), list):
                raise HelperFailed(message, 409, error=message, clash=result["clash"],
                                   existing=list(result.get("existing") or []))
            raise HelperFailed(message, 422)
        return result

    # ---- opening / detail ---------------------------------------------------------------------------------------
    def open(self, path: str, recent: bool = True) -> dict:
        """The sample at `path`, inspected again when it changed (POST /samples/open), then remembered: first on the
        open dialog's list, or with `recent` False (a run's sample opened again for it) known without being listed.
        Only an open the route can send is remembered (a detail holding NaN answered 500, and the sample
        was in Recent all the same); anything unexpected is a plain SampleError, its traceback in the engine log."""
        folder = Path(str(path or "")).expanduser()
        try:
            folder = folder.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SampleError(f"{path} does not exist", 404) from None
        if not is_spatialdata(folder):
            raise SampleError("This folder is not a SpatialData zarr (no zarr.json/.zgroup with points, shapes or images)")
        sid = sample_id(str(folder))
        try:
            with self._detail_lock(sid):
                detail = self._build(sid, str(folder), refresh=True)
            json.dumps(detail, allow_nan=False)             # what the route sends (strict JSON)
        except SampleError:
            with self.lock:
                self.details.pop(sid, None)
            raise
        except Exception:
            with self.lock:
                self.details.pop(sid, None)
            print(f"SoPA companion: opening {folder} failed:\n{traceback.format_exc()}", flush=True)
            raise SampleError("This sample could not be opened: the engine ran into a problem (the engine log has "
                              "it)") from None
        self._remember(detail, shown=recent)
        return detail

    def bind(self, path: str, *, name: str | None = None, points_key: str | None = None) -> dict:
        """API mode (bind: engine/session.py calls it on its own thread): the session's one sample, opened as Open
        does (inspected, its detail kept), named as the notebook gives it (`name`: a view is called view.zarr, so the
        folder never names it), with the transcripts the notebook names (`points_key`, the helper's inspect asked for
        it; None: the helper's own choice), and the only sample this engine knows (a registry of one: the mixing
        check finds it by its path there). Answers the sample's detail."""
        folder = Path(str(path or "")).expanduser()
        try:
            folder = folder.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SampleError(f"{path} does not exist", 404) from None
        sid = sample_id(str(folder))
        with self.lock:
            if name:
                self.bound_names[sid] = str(name)
            else:
                self.bound_names.pop(sid, None)
            if points_key:
                self.bound_points[sid] = str(points_key)
            else:
                self.bound_points.pop(sid, None)
        detail = self.open(str(folder), recent=True)
        with self.lock:
            self.recent = [e for e in self.recent if e["id"] == sid]
            _atomic_write(self.registry, json.dumps(self.recent, indent=1).encode())
        return detail

    def detail(self, sid: str) -> dict:
        with self.lock:
            if sid in self.details:
                return self.details[sid]
        with self._detail_lock(sid):
            with self.lock:
                if sid in self.details:
                    return self.details[sid]
            return self._build(sid, self.path(sid), refresh=False)

    def _detail_lock(self, sid):
        with self.lock:
            return self.building.setdefault(sid, threading.Lock())

    def _build(self, sid: str, path: str, refresh: bool) -> dict:
        """The sample detail (docs/ARCHITECTURE.md, "Opening a sample") from the helper's inspection, kept as
        inspect.json and made again on an open when the zarr changed since, or when it was made before the helper said
        what an open finds (its `units`: the notes)."""
        folder = self.cache / sid
        inspect_path = folder / "inspect.json"
        info = None
        with self.lock:
            asked = self.bound_points.get(sid)
        try:
            info = json.loads(inspect_path.read_text())
            if refresh and (abs(info.get("mtime", 0) - self._mtime(path)) > 1e-3 or "units" not in info or "areas" not in info):
                info = None
            elif asked and info.get("points_key") != asked or self._unstamped(info) or self._images_changed(path, info):
                info = None
        except (OSError, ValueError):
            info = None
        if info is None:
            info = self.helper("inspect", {"path": path, **({"points_key": asked} if asked else {})}, timeout=600)
            self._stamp_images(path, info)
            _atomic_write(inspect_path, json.dumps(info).encode())
            self._forget_arrays(path)
        attrs = info.get("attrs") if isinstance(info.get("attrs"), dict) else {}
        listed = [p for p in info.get("points") or [] if isinstance(p, dict) and p.get("key")]
        points_key = info.get("points_key")        # None: the sample has no transcripts
        points = next((p for p in listed if p["key"] == points_key), None) if points_key else None
        if points_key and points is None:
            raise SampleError(f"The sample's transcripts ({points_key}) could not be read")
        if points is not None and not points.get("count"):
            raise SampleError("The sample's transcripts are empty")
        images = [i for i in info.get("images") or [] if "error" not in i and i.get("to_frame")]
        image = self._default_image(images, attrs, info.get("default_image"))
        if points is None and image is None:
            raise SampleError("This sample has no transcripts and no image this app can show")
        options = self._prior_options(info, points)
        from .stain_levels import offered
        with self.lock:
            name = self.bound_names.get(sid) or info["name"]
        detail = {"schema": 2, "id": sid, "kind": "zarr", "name": name, "path": path,
                  "points_key": points["key"] if points is not None else None,
                  "points": [{k: p.get(k) for k in ("key", "count", "columns")} for p in listed],
                  "transcripts": points["count"] if points is not None else None,
                  "extent": points["extent"] if points is not None else
                  _extent(info.get("extent")) or self._image_extent(image),          # the images' together
                  "image_key": image["key"] if image else None,
                  "images": [{"key": i["key"], "channels": i["channels"], "levels": len(i["levels"]),
                              "um_per_px": _um_per_px(i), "scales": offered(i)} for i in images],
                  "shapes": [{**{k: s.get(k) for k in ("key", "count", "geometry", "median_area_um2")}, "extent": _extent(s.get("extent"))}
                             for s in info.get("shapes") or [] if s.get("to_frame")],
                  "tables": info.get("tables") or [], "labels": info.get("labels") or [],
                  "prior_options": options, "suggested_prior": self._suggestion(info, points, options),
                  "limits": {"detail_um": self.detail_um}, "z_range": points.get("z_range") if points is not None else None,
                  # The tools it opens in (the first screen's): proseg needs transcripts, stain segmentation an image.
                  "tools": ["proseg"] * (points is not None) + ["stain"] * (image is not None),
                  "notes": open_notes(info, points, images, image)}
        detail = _finite(detail)                   # a NaN anywhere (an element of empty geometries) is null
        x0, y0, x1, y1 = detail["extent"]
        detail.update(width_um=x1, height_um=y1)
        detail["channels"], detail["overview"] = [], None
        if image:
            try:
                detail["overview"] = self._overview(sid, path, image)
                detail["channels"] = self._stats(sid, path, image)
            except Exception as exc:    # the sample is still usable without its image
                detail["image_error"] = f"{type(exc).__name__}: {exc}"
                print(f"SoPA companion: the image {image['key']} of {path} could not be read: {detail['image_error']}",
                      flush=True)
                detail["notes"].append({
                    "code": "image_unreadable", "text": f"The image {image['key']} could not be read: it is not shown.",
                    "detail": ("Proseg runs work without it. " if points is not None else "") + "The engine log has the error."})
        with self.lock:
            self.details[sid] = detail
        return detail

    @staticmethod
    def _default_image(images: list[dict], attrs: dict, chosen: str | None = None) -> dict | None:
        """The image runs show and the stain picker starts from: the helper's choice (`chosen`, its default_image) when
        it is one of `images`, else by the same rule: the one the sample names (cell_segmentation_image), else the
        first that is not RGB (an H&E listed first became the overview, and its red channel the stain
        picker's nucleus channel), else the first."""
        named = next((i for i in images if i["key"] == chosen), None) if chosen else None
        named = named or next((i for i in images if i["key"] == attrs.get("cell_segmentation_image")), None)
        return named or next((i for i in images if not _rgb(i)), images[0] if images else None)

    def _image_extent(self, image: dict) -> list[float]:
        """The working-frame box an image's full resolution covers: the extent of a sample without transcripts."""
        level = image["levels"][0]
        x0, y0, sx, sy = self._geometry(image, level)
        height, width = level["shape"][-2:]
        xs, ys = (x0, x0 + sx * width), (y0, y0 + sy * height)
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    @staticmethod
    def _mtime(path: str) -> float:
        stamps = []
        for name in ("zarr.json", ".zattrs", *ELEMENT_KINDS):
            try:
                stamps.append(os.stat(Path(path) / name).st_mtime)
            except OSError:
                pass
        return max(stamps) if stamps else 0.0

    @staticmethod
    def _images_changed(path: str, info: dict) -> bool:
        """Whether an image the inspection stamped was written again since (image_stamp; _mtime does not see an image's
        own metadata: a transformation rewritten alone kept the old geometry). An image whose
        folder does not answer now says nothing."""
        for image in info.get("images") or []:
            recorded = image.get("stamp") if isinstance(image, dict) else None
            if recorded and image.get("key"):
                now = image_stamp(path, image["key"])
                if now is not None and now != recorded:
                    return True
        return False

    @staticmethod
    def _unstamped(info: dict) -> bool:
        """An inspection made before its images were stamped (an image with no `stamp` at all): inspected once more,
        never stamped as the image is now, which would keep the geometry of an image rewritten before this version
        (its transformation alone) for good."""
        return any(isinstance(image, dict) and image.get("key") and "stamp" not in image
                   for image in info.get("images") or [])

    @staticmethod
    def _stamp_images(path: str, info: dict):
        """Stamp a new inspection's images that the helper did not stamp, as they are now (the inspection has just read
        them). One whose folder does not answer is stamped None: inspected by this version, nothing to compare."""
        for image in info.get("images") or []:
            if isinstance(image, dict) and image.get("key") and not image.get("stamp"):
                image["stamp"] = image_stamp(path, image["key"])

    def _forget_arrays(self, path: str):
        """The sample was inspected again: arrays opened with its old metadata go (a rewritten level read through one
        raised 'cannot reshape')."""
        with self.lock:
            self.arrays = {key: array for key, array in self.arrays.items() if key[0] != path}

    @staticmethod
    def _prior_options(info: dict, points: dict | None) -> dict:
        """What proseg can start its cells from: shapes elements of more than one polygon (or circle) that the routes
        serve (_usable), and the transcripts' own columns; nothing for a sample without transcripts."""
        if points is None:
            return {"cells": [], "nuclei": []}
        polygons = [s for s in info.get("shapes") or [] if s.get("to_frame") and (s.get("count") or 0) > 1 and _usable(s)
                    and any(g in ("Polygon", "MultiPolygon") or g == "Point" for g in s.get("geometry") or {})]
        def shapes(prefer):
            # The app's own saves last (the helper's app_save: a stain save may cover one box only), as the helper's
            # suggestion ranks them; then the layers named for the part, Xenium's first.
            ranked = sorted(polygons, key=lambda s: (bool(s.get("app_save")), prefer not in s["key"].lower(),
                                                     "xenium" not in s["key"].lower(), s["key"]))
            return [{"kind": "shapes", "key": s["key"], "count": s["count"],
                     "label": f"{s['key']} ({s['count']:,} shapes)"} for s in ranked]
        columns = points.get("prior_columns", {})
        cells = shapes("cell") + [
            {"kind": "column", "column": c["column"], "unassigned": c["unassigned"], "fraction": c["assigned_fraction"],
             "label": f"Transcripts' {c['column']} column"
                      + (f" ({c['assigned_fraction']:.0%} assigned)" if c["assigned_fraction"] is not None else "")}
            for c in columns.get("cells", [])]
        nuclei = shapes("nucle") + [
            {"kind": "column", "column": c["column"], "nuclear": c["nuclear"], "fraction": c["nuclear_fraction"],
             "label": f"Transcripts' {c['column']} flag"
                      + (" (no transcript flagged)" if c["nuclear_fraction"] == 0 else
                         f" ({c['nuclear_fraction']:.0%} flagged)" if c["nuclear_fraction"] is not None else "")}
            for c in columns.get("nuclei", [])]
        return {"cells": cells, "nuclei": nuclei}

    @staticmethod
    def _suggestion(info: dict, points: dict | None, options: dict) -> dict:
        """The helper's suggested prior when the picker offers what it names; else the helper's own rule
        (_suggest_prior) again over the shapes the picker offers (a name the routes refuse, sorted first, was
        the pre-selected prior, then its outlines answered 404), or, should that rule not take them, without the part
        it cannot offer."""
        empty = {"cells": None, "nuclei": None}
        suggested = info.get("suggested_prior") if isinstance(info.get("suggested_prior"), dict) else empty
        if points is None:
            return empty
        offered = {part: {o["key"] for o in options[part] if o["kind"] == "shapes"} for part in ("cells", "nuclei")}
        missing = [part for part in ("cells", "nuclei") if isinstance(suggested.get(part), dict)
                   and suggested[part].get("kind") == "shapes" and suggested[part].get("key") not in offered[part]]
        if not missing:
            return {**empty, **suggested}
        try:
            from ..zarr_bridge.helper import _suggest_prior
            return _suggest_prior([s for s in info.get("shapes") or [] if _usable(s)], points)
        except Exception as exc:                      # noqa: BLE001 - the helper's rule is a nicety here
            print(f"SoPA companion: the suggested prior could not be chosen again: {type(exc).__name__}: {exc}", flush=True)
            return {**empty, **{part: spec for part, spec in suggested.items() if part not in missing}}

    def validate_prior(self, sid: str, prior) -> dict:
        """The prior as the helper takes it, checked against what the sample offers."""
        if not isinstance(prior, dict):
            raise SampleError("Choose a prior (cells and/or nuclei)")
        options = self.detail(sid)["prior_options"]
        result = {}
        for part in ("cells", "nuclei"):
            spec = prior.get(part)
            if not spec:
                result[part] = None
                continue
            if not isinstance(spec, dict):
                raise SampleError(f"Unknown {part} prior")
            match = next((o for o in options[part] if o["kind"] == spec.get("kind") and
                          o.get("key", o.get("column")) == spec.get("key", spec.get("column"))), None)
            if match is None:
                raise SampleError(f"The sample has no {part} prior '{spec.get('key') or spec.get('column')}'")
            result[part] = {k: v for k, v in match.items() if k in ("kind", "key", "column", "unassigned", "nuclear")}
            # An explicitly chosen token wins over the inferred one (e.g. integer ids whose background is not -1/0).
            for token in ("unassigned", "nuclear"):
                if match["kind"] == "column" and token in spec and token in result[part]:
                    value = spec[token]
                    if not isinstance(value, (str, int)) or len(str(value)) > 100:
                        raise SampleError(f"Invalid {token} value for the {part} prior")
                    result[part][token] = str(value)
        if not result["cells"] and not result["nuclei"]:
            raise SampleError("Choose a prior: proseg needs cells or nuclei to start from")
        return result

    # ---- images -----------------------------------------------------------------------------------------------
    def _image_info(self, sid: str, key: str | None = None) -> dict:
        self.detail(sid)       # (re)inspects when needed
        info = json.loads((self.cache / sid / "inspect.json").read_text())
        images = [i for i in info["images"] if "error" not in i and i.get("to_frame")]
        image = next((i for i in images if i["key"] == key), None) if key else None
        if image is None:
            detail = self.detail(sid)
            image = next((i for i in images if i["key"] == detail.get("image_key")), None)
        if image is None:
            raise SampleError("This sample has no usable image", 404)
        return image

    def _array(self, path: str, level: dict):
        """The level's zarr array, kept open, by its path and shape as inspected (another shape: opened again)."""
        import zarr
        key = (path, level["path"], json.dumps(level.get("shape")))
        with self.lock:
            array = self.arrays.get(key)
        if array is None:
            array = zarr.open_array(store=str(Path(path) / level["path"]), mode="r")
            with self.lock:
                self.arrays[key] = array
        return array

    @staticmethod
    def _geometry(image: dict, level: dict) -> tuple[float, float, float, float]:
        """(x0, y0, sx, sy): frame position of the level's pixel (0, 0) corner and µm per pixel along x / y."""
        matrix = np.asarray(image["to_frame"], float)
        ox, oy = level["origin"]
        px, py = level["pixel"]
        return (float(matrix[0, 0] * ox + matrix[0, 2]), float(matrix[1, 1] * oy + matrix[1, 2]),
                float(matrix[0, 0] * px), float(matrix[1, 1] * py))

    @staticmethod
    def _plane(array, level: dict, channel: int, rows: slice, cols: slice) -> np.ndarray:
        dims = level["dims"]
        index = tuple(channel if d == "c" else rows if d == "y" else cols if d == "x" else 0 for d in dims)
        return np.asarray(array[index])

    def _prune_derived(self, sid: str, prefix: str, kept: set[str]):
        """What was derived from an image under another identity goes (<prefix>-<identity>.<ext>, and the name it had
        before identities: <prefix>.<ext>), all but `kept`."""
        mine = re.compile(re.escape(prefix) + r"(?:-[0-9a-f]{12})?\.(?:png|json)")
        try:
            for path in (self.cache / sid).iterdir():
                if mine.fullmatch(path.name) and path.name not in kept:
                    path.unlink(missing_ok=True)
        except OSError:
            pass

    def _overview(self, sid: str, path: str, image: dict) -> dict:
        """The whole sample's thumbnail (overview-<image>-c<channel>-<identity>.png and its .json): the image's nucleus
        stain (overview_channel), from the largest pyramid level within OVERVIEW_PX; made again for another identity
        (image_identity: the image written again, or inspected differently), the old one going. Its meta carries the
        identity (`identity`): the page's overview URL can carry it, so a browser never shows the old pixels."""
        channel = overview_channel(image["channels"])
        identity = image_identity(image, path)
        target = self.cache / sid / f"overview-{image['key']}-c{channel}-{identity}.png"
        meta_path = target.with_suffix(".json")
        if target.exists() and meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except (OSError, ValueError):
                pass
        levels = image["levels"]
        level = next((lv for lv in levels if max(lv["shape"][-2:]) <= OVERVIEW_PX), levels[-1])
        array = self._array(path, level)
        plane = self._plane(array, level, channel, slice(None), slice(None))
        x0, y0, sx, sy = self._geometry(image, level)
        _atomic_write(target, _png(_stretch(_u16(plane, _display_scale(plane)))))
        meta = {"x0": x0, "y0": y0, "um_per_px": sx, "um_per_px_y": sy, "width": int(plane.shape[1]), "height": int(plane.shape[0]),
                "x1": x0 + sx * plane.shape[1], "y1": y0 + sy * plane.shape[0], "image": image["key"], "channel": channel,
                "identity": identity}
        _atomic_write(meta_path, json.dumps(meta).encode())
        self._prune_derived(sid, f"overview-{image['key']}-c{channel}", {target.name, meta_path.name})
        return meta

    def overview_path(self, sid: str) -> Path:
        detail = self.detail(sid)
        if not detail.get("overview"):
            raise SampleError("This sample has no image", 404)
        overview = detail["overview"]
        stamp = f"-{overview['identity']}" if overview.get("identity") else ""
        return self.cache / sid / f"overview-{overview['image']}-c{overview.get('channel', 0)}{stamp}.png"

    @staticmethod
    def _stats_level(image: dict) -> dict:
        levels = image["levels"]
        return next((lv for lv in levels if max(lv["shape"][-2:]) <= STATS_PX), levels[-1])

    def _stats_path(self, sid: str, path: str, image: dict) -> Path:
        return self.cache / sid / f"stats-{image['key']}-{image_identity(image, path)}.json"

    def _stats(self, sid: str, path: str, image: dict) -> list[dict]:
        """Each channel's statistics as the page draws it (_channel_stats of _u16), from the largest pyramid level within
        STATS_PX; a float image's also carry its display `scale` (_display_scale on that level: its views and crops are
        drawn times it). Kept under the image's identity (stats-<image>-<identity>.json, the others going), and worked out
        again when kept from before they had a scale, or for other channels than the image's."""
        target = self._stats_path(sid, path, image)
        level = self._stats_level(image)
        dtype = image.get("dtype")
        try:                                      # as the helper read it, else as its array says
            scaled = np.dtype(dtype if isinstance(dtype, str) else self._array(path, level).dtype).kind == "f"
        except TypeError:
            scaled = np.dtype(self._array(path, level).dtype).kind == "f"
        try:
            stats = json.loads(target.read_text())
            if [entry.get("name") for entry in stats] == list(image["channels"]) and \
                    (not scaled or all("scale" in entry for entry in stats)):
                return stats
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        array = self._array(path, level)
        stats = []
        for index, name in enumerate(image["channels"]):
            plane = self._plane(array, level, index, slice(None), slice(None))
            scale = _display_scale(plane)
            stats.append({"name": name, **_channel_stats(_u16(plane, scale)), **({"scale": scale} if scaled else {})})
        _atomic_write(target, json.dumps(stats).encode())
        self._prune_derived(sid, f"stats-{image['key']}", {target.name})
        return stats

    def _scales(self, sid: str, path: str, image: dict, array) -> list[float | None]:
        """Each channel's display scale (_stats) for planes read from `array` (one of the image's levels): None for an
        image that is not float, which is drawn as stored."""
        if np.dtype(array.dtype).kind != "f":
            return [None] * len(image["channels"])
        return [entry.get("scale") for entry in self._stats(sid, path, image)]

    def _window(self, image: dict, level: dict, box) -> tuple[int, int, int, int]:
        x0, y0, sx, sy = self._geometry(image, level)
        height, width = level["shape"][-2:]
        c0 = max(0, int(math.floor((box[0] - x0) / sx)))
        r0 = max(0, int(math.floor((box[1] - y0) / sy)))
        c1 = min(width, int(math.ceil((box[2] - x0) / sx)))
        r1 = min(height, int(math.ceil((box[3] - y0) / sy)))
        return c0, r0, c1, r1

    def crop_images(self, sid: str, box, destination: Path, key: str | None = None) -> dict:
        """Every channel over the box at full resolution: 8-bit PNG + raw uint16 (zstd) + contrast statistics, in the
        same layout as Xenium bundle crops (image.json)."""
        path = self.path(sid)
        image = self._image_info(sid, key)
        level = image["levels"][0]
        c0, r0, c1, r1 = self._window(image, level, box)
        if c1 <= c0 or r1 <= r0:
            return {"channels": []}
        array = self._array(path, level)
        scales = self._scales(sid, path, image, array)
        channels = []
        for index, name in enumerate(image["channels"]):
            plane = _u16(self._plane(array, level, index, slice(r0, r1), slice(c0, c1)), scales[index])
            file = f"c{index}.png"
            _atomic_write(destination / file, _png(_stretch(plane)))
            _atomic_write(destination / f"c{index}.u16.zst", _zstd(plane.tobytes(), 3))
            channels.append({"name": name, "file": file, "file16": f"c{index}.u16", **_channel_stats(plane)})
        x0, y0, sx, sy = self._geometry(image, level)
        meta = {"x0": x0 + c0 * sx, "y0": y0 + r0 * sy, "um_per_px": sx, "um_per_px_y": sy,
                "width": c1 - c0, "height": r1 - r0, "level": 0, "image": image["key"], "channels": channels}
        _atomic_write(destination / "image.json", json.dumps(meta).encode())
        return meta

    def read_view(self, sid: str, key: str | None, channel: int, box, max_px: int = 2048) -> tuple[np.ndarray, dict]:
        """One channel over the box at max_px pixels across (the page asks for the box's size in screen pixels): read
        from the coarsest pyramid level that is still at least as fine as that, then area-averaged down to it, so a
        view is never coarser than the screen. Taking the finest level within max_px instead could land a level step
        (2x) coarser than the screen, and a whole-tissue view looked as if it had not loaded."""
        path = self.path(sid)
        image = self._image_info(sid, key)
        if not 0 <= channel < len(image["channels"]):
            raise SampleError("Unknown channel", 404)
        want = max(box[2] - box[0], box[3] - box[1]) / max(1, max_px)      # micrometres per screen pixel
        chosen = image["levels"][0]
        for level in image["levels"]:                                      # finest first
            _, _, sx, sy = self._geometry(image, level)
            if max(sx, sy) > want * 1.001:
                break
            chosen = level
        c0, r0, c1, r1 = self._window(image, chosen, box)
        if c1 <= c0 or r1 <= r0:
            raise SampleError("The box is outside the image", 404)
        array = self._array(path, chosen)
        plane = _u16(self._plane(array, chosen, channel, slice(r0, r1), slice(c0, c1)),
                     self._scales(sid, path, image, array)[channel])
        plane = _shrink(plane, max_px)
        x0, y0, sx, sy = self._geometry(image, chosen)
        return plane, {
            "x0": x0 + c0 * sx, "y0": y0 + r0 * sy, "x1": x0 + c1 * sx, "y1": y0 + r1 * sy,
            "width": plane.shape[1], "height": plane.shape[0], "level": image["levels"].index(chosen)}

    # ---- shapes -----------------------------------------------------------------------------------------------
    def element_stamp(self, sid: str, key: str) -> str | None:
        """A shapes element as it is on disk now: 12 hex digits of its folder's and files' identity, size and mtime, which
        change whenever the element is written again (an app save, a notebook). None when its folder cannot be read
        now (the sample share is an automount that can fail for a moment), or of a sample not known: callers then keep
        what they have."""
        try:
            folder = Path(self.path(sid)) / "shapes" / key
            found = [("", *self._identity(folder.stat()))]
            with os.scandir(folder) as entries:
                found += [(e.name, *self._identity(e.stat())) for e in entries]
        except (OSError, SampleError):
            return None
        return hashlib.sha1(json.dumps(sorted(found)).encode()).hexdigest()[:12]

    def table_stamp(self, sid: str, key: str) -> str | None:
        """A table as it is on disk now (tables/<key>), as element_stamp says a shapes element: its folder's, its
        entries' and its counts' (X's) identity, size and mtime, so a table written again, or its counts alone, is
        another version (Compare keeps a table's own counts by it). None when its folder cannot be read now."""
        try:
            folder = Path(self.path(sid)) / "tables" / key
            found = [("", *self._identity(folder.stat()))]
            with os.scandir(folder) as entries:
                found += [(e.name, *self._identity(e.stat())) for e in entries]
            if (folder / "X").is_dir():
                with os.scandir(folder / "X") as entries:
                    found += [(f"X/{e.name}", *self._identity(e.stat())) for e in entries]
        except (OSError, SampleError):
            return None
        return hashlib.sha1(json.dumps(sorted(found)).encode()).hexdigest()[:12]

    def _readable_stamp(self, sid: str, key: str, read=None) -> str:
        """element_stamp (or `read`: table_stamp) for files kept by it (the layers' outlines, Compare's counts): read
        again once after STAMP_RETRY_S when the element's folder did not answer, then SampleError 503 in plain words.
        Nothing is ever kept under a stamp that could not be read, nor are the real version's files pruned because of
        one."""
        for attempt in range(2):
            stamp = (read or self.element_stamp)(sid, key)
            if stamp is not None:
                return stamp
            if not attempt:
                time.sleep(STAMP_RETRY_S)
        raise SampleError(f"The files of '{key}' could not be read just now (the sample's folder did not answer): try "
                          "again in a moment", 503)

    @staticmethod
    def _identity(info: os.stat_result) -> tuple:
        return info.st_ino, info.st_size, info.st_mtime_ns

    def _export_stale(self, sid: str, key: str, base: Path, stamp: str | None) -> bool:
        """Whether an export no longer matches its element: written again since (its stamp, kept beside the export as
        <base>.stamp), or, for an export made before stamps were kept, an element newer than it."""
        if stamp is None or not Path(str(base) + ".json").exists():
            return False
        try:
            return Path(str(base) + ".stamp").read_text().strip() != stamp
        except OSError:
            pass
        try:
            folder = Path(self.path(sid)) / "shapes" / key
            newest = max([folder.stat().st_mtime_ns, *(e.stat().st_mtime_ns for e in os.scandir(folder))])
            return newest > Path(str(base) + ".json").stat().st_mtime_ns
        except (OSError, ValueError):
            return False

    @staticmethod
    @contextmanager
    def _held(guard: threading.Lock, should_stop=None):
        """Hold an export's guard. With `should_stop`, waiting for another caller's export of the same element ends once
        it says stop (HelperFailed "Stopped", 409): a comparison replaced or the engine closing never stays stuck behind
        it."""
        if should_stop is None:
            guard.acquire()
        else:
            while not guard.acquire(timeout=0.25):
                if should_stop():
                    raise HelperFailed("Stopped", 409)
        try:
            yield
        finally:
            guard.release()

    def _drop_export(self, sid: str, key: str, base: Path):
        with self.lock:
            self.outlines.pop((sid, key), None)
            self.outline_stamps.pop((sid, key), None)
        for suffix in (".json", ".bin", ".stamp"):
            try:
                Path(str(base) + suffix).unlink()
            except OSError:
                pass

    def _shapes_base(self, sid: str, key: str, should_stop=None) -> Path:
        """Where a shapes element's outlines are exported (<base>.bin + <base>.json, its stamp in <base>.stamp): exported
        by the helper on first use (once at a time per element), and again once the element was written again since
        (_export_stale). The stamp is read again after an export: an element written while it was exported (a notebook,
        another engine) is exported once more, then refused (409), so an export never stands under a stamp it is not
        of. `should_stop`: the export and the wait for another's stop with it."""
        if not KEY.fullmatch(key):
            raise SampleError("Unknown shapes element", 404)
        if key not in {s["key"] for s in self.detail(sid)["shapes"]}:
            raise SampleError(f"The sample has no shapes element '{key}'", 404)
        with self.lock:
            guard = self.exporting.setdefault((sid, key), threading.Lock())
        base = self.cache / sid / "shapes" / key
        with self._held(guard, should_stop):
            stamp = self.element_stamp(sid, key)
            if self._export_stale(sid, key, base, stamp):
                self._drop_export(sid, key, base)
            for attempt in range(2):
                if Path(str(base) + ".bin").exists() and Path(str(base) + ".json").exists():
                    break
                self.helper("shapes", {"path": self.path(sid), "key": key, "points_key": self.detail(sid)["points_key"],
                                       "out": str(base)}, should_stop=should_stop, timeout=900)
                after = self.element_stamp(sid, key)
                if stamp is None or after is None or after == stamp:
                    break
                self._drop_export(sid, key, base)        # written again meanwhile: either version may be in it
                if attempt:
                    raise SampleError(f"'{key}' was written again while its outlines were read: try again", 409)
                stamp = after
            if stamp is not None and not Path(str(base) + ".stamp").exists():
                _atomic_write(Path(str(base) + ".stamp"), stamp.encode())
        return base

    def _outlines(self, sid: str, key: str, should_stop=None) -> dict:
        """A shapes element's exported outlines as arrays ({ids, bounds, offsets, owner, coords}: load_outlines; the ids
        are the element's own, in its order), the OUTLINES_KEPT most recently used kept (a dense whole sample: ~50 MB
        per element; runs read them once to build their prior shapes, the picker reads them in its workers, Compare
        matches its counts against them); read again once the element was written again since. `should_stop` (Compare):
        an export it starts, or waits for, stops with it (HelperFailed "Stopped", 409)."""
        cache_key = (sid, key)
        stamp = self.element_stamp(sid, key) if KEY.fullmatch(key) else None
        with self.lock:
            if cache_key in self.outlines:
                self.outlines[cache_key] = self.outlines.pop(cache_key)       # most recently used last
                kept = self.outline_stamps.get(cache_key)
                if stamp is None or kept is None or kept == stamp:
                    return self.outlines[cache_key]
        base = self._shapes_base(sid, key, should_stop)
        try:                                         # the stamp the export was made at (_shapes_base kept it)
            stamp = Path(str(base) + ".stamp").read_text().strip() or stamp
        except OSError:
            pass
        value = load_outlines(base)
        with self.lock:
            self.outlines[cache_key], self.outline_stamps[cache_key] = value, stamp
            while len(self.outlines) > OUTLINES_KEPT:
                oldest = next(iter(self.outlines))
                self.outlines.pop(oldest)
                self.outline_stamps.pop(oldest, None)
        return value

    def outline_stamp(self, sid: str, key: str) -> str | None:
        """The element stamp the outlines _outlines last gave for `key` were exported at (None: not read yet)."""
        with self.lock:
            return self.outline_stamps.get((sid, key))

    def shapes_json(self, sid: str, key: str, box=None, limit: int = 20000) -> memoryview:
        """shapes() as the response body (JSON, bytes-like), built in a worker process (engine/shape_query.py): a
        whole sample's picker preview never holds the engine's GIL."""
        for attempt in range(2):
            base = self._shapes_base(sid, key)
            try:
                return self.queries.query(base, key, box, limit)
            except FileNotFoundError:
                if attempt:                 # the export was replaced in between (a save into this sample): once more
                    raise
            except TimeoutError:
                raise SampleError("The outlines took too long to read: try again", 503) from None
            except Unavailable as exc:
                raise SampleError(str(exc), 503) from None
            except WorkerError as exc:
                raise SampleError(f"Could not read the outlines: {exc}", 500) from None
        raise AssertionError("unreachable")

    def shapes(self, sid: str, key: str, box=None, limit: int = 20000) -> dict:
        """{key, total, truncated, polygons: [{id, poly}]}: the selection of every prior source (vectors.hits/gather)."""
        data = self._outlines(sid, key)
        found, total = hits(data["bounds"], box, limit)
        edges, xy = gather(data, found)
        ids, xy = data["ids"], xy.tolist()
        polygons = [{"id": ids[owner], "poly": xy[a:b]}
                    for owner, a, b in zip(data["owner"][found].tolist(), edges[:-1].tolist(), edges[1:].tolist())]
        return {"key": key, "total": total, "truncated": total > len(polygons), "polygons": polygons}

    # ---- segmentation layers and Compare ----------------------------------------------------------------------------
    def segmentations(self, sid: str) -> list[dict]:
        """GET /samples/{id}/segmentations: every cell-like shapes element of the sample (_cell_like: polygons, more than
        one shape, never region_of_interest), Xenium's first: {key, name, source, model, table, coverage (describe_segmentation),
        count_by, count_table (whose count Compare reads for it: count_source), count, extent (µm, the working frame),
        version (element_stamp: its outlines' URL carries it; UNREAD when its folder did not answer twice,
        STAMP_RETRY_S apart: never a version anything is kept under)}."""
        detail = self.detail(sid)
        made_by = self._made_by(sid)
        layers = [s for s in detail["shapes"] if self._cell_like(s)]
        stamps = {s["key"]: self.element_stamp(sid, s["key"]) for s in layers}
        if None in stamps.values():                      # a folder that did not answer: once more, after a moment
            time.sleep(STAMP_RETRY_S)
            stamps.update({key: self.element_stamp(sid, key) for key, stamp in stamps.items() if stamp is None})
        listed = [{"key": s["key"], **describe_segmentation(s["key"], made_by), **count_source(s["key"], made_by),
                   "count": s["count"], "extent": s.get("extent"), "version": stamps[s["key"]] or UNREAD}
                  for s in layers]
        return sorted(listed, key=lambda e: (SOURCES.index(e["source"]), e["key"].lower()))

    @staticmethod
    def _cell_like(shape: dict) -> bool:
        """A segmentation: polygons, more than one, in an element the routes serve (_usable: not region_of_interest,
        however many pieces it has; a name KEY takes; outlines somewhere)."""
        return (shape.get("count") or 0) > 1 and any(g in SEGMENTATION_GEOMETRY for g in shape.get("geometry") or {}) \
            and _usable(shape)

    def _segmentation(self, sid: str, key: str) -> dict:
        """The detail's entry of a cell-like shapes element (404 otherwise)."""
        found = next((s for s in self.detail(sid)["shapes"] if s["key"] == key), None) if KEY.fullmatch(key) else None
        if found is None or not self._cell_like(found):
            raise SampleError(f"The sample has no segmentation '{key}'", 404)
        return found

    def _made_by(self, sid: str) -> dict | None:
        """The helper's `segmentations` (what made each element), cached until the zarr changes (_mtime); None when the
        helper could not say: every element is then named by its key. The provenance keys it was read with, and those
        that say a table holds proseg's own count, are in the stamp: a cache of the previous version, which knew one of
        them, is read again."""
        from ..zarr_bridge.helper import PROSEG_TABLE_KEYS
        path = self.path(sid)
        cached = self.cache / sid / "segmentations.json"
        stamp = [self._mtime(path), list(PROVENANCE_KEYS), list(PROSEG_TABLE_KEYS)]
        with self.lock:
            guard = self.exporting.setdefault(("made_by", sid), threading.Lock())
        with guard:
            try:
                found = json.loads(cached.read_text())
                if found.get("stamp") == stamp:
                    return found
            except (OSError, ValueError):
                pass
            try:
                found = self.helper("segmentations", {"path": path}, timeout=300)
            except SampleError as exc:
                print(f"SoPA companion: could not read what made the segmentations of {sid}: {exc}", flush=True)
                return None
            _atomic_write(cached, json.dumps({**found, "stamp": stamp}).encode())
            return found

    def saved_from(self, sid: str, run_id: str) -> list[str]:
        """The sample's layers a save of run `run_id` made, read from the zarr's own record (helper `segmentations`):
        a stain save's shapes (their attrs name the run), a proseg save's table (its uns record; the helper's `run`,
        else read here) and the shapes it annotates. A save cut off before it committed is not one (its
        layers go with the next save into the sample: unfinished_writes). [] when there is none, or when it cannot say
        (an older run whose save the last session finished is never saved twice)."""
        made = self._made_by(sid) or {}
        found = [s["key"] for s in made.get("shapes") or [] if (s.get("app") or {}).get("run") == run_id]
        for table in made.get("tables") or []:
            run = table.get("run")
            if run is None and table.get("proseg"):
                run = self._table_run(sid, table.get("key"))
            if run == run_id:
                found += [*[str(r) for r in table.get("region") or []], table["key"]]
        if found:
            cut_off = {name for _, name in unfinished_writes(self.path(sid))}
            found = [key for key in found if key not in cut_off]
        return list(dict.fromkeys(found))

    def saved_provenance(self, sid: str, key: str) -> dict | None:
        """A stain save's record on the shapes it made (shapes_provenance: run, result_revision, filters, …): an
        older run is "Saved already" only when that is the stain run's result as it is now (Runs._saved_already)."""
        return shapes_provenance(self.path(sid), key)

    def _table_run(self, sid: str, key: str | None) -> str | None:
        return table_run(self.path(sid), key)

    def segmentation_outlines(self, sid: str, key: str) -> tuple[Path, str]:
        """GET /samples/{id}/segmentations/{key}/outlines: the element's every outline (working frame) as the prior
        vectors binary with the cells layer only (engine/vectors.py, the page's decodeVectors), zstd-compressed, built
        once per element stamp from its export (cache/zarr/<sid>/segmentations/<key>-<stamp>.v1.bin.zst; earlier
        stamps' files go): (file, stamp). The stamp is always a real one (_readable_stamp: SampleError 503 when the
        element's folder does not answer), never UNREAD."""
        self._segmentation(sid, key)
        folder = self.cache / sid / "segmentations"
        for attempt in range(2):
            stamp = self._readable_stamp(sid, key)
            target = folder / f"{key}-{stamp}.v{VECTORS_VERSION}.bin.zst"
            if target.exists():
                return target, stamp
            with self.lock:
                guard = self.exporting.setdefault(("outlines", sid, key), threading.Lock())
            with guard:
                if target.exists():
                    return target, stamp
                data = self._outlines(sid, key)
                if self.outline_stamp(sid, key) not in (None, stamp):    # written again in between: never kept
                    continue                                              # under a stamp it is not of
                layer = select(data, None, len(data["bounds"]))
                _atomic_write(target, _zstd(encode({"cells": layer, "nuclei": None}, {"source": {"cells": key}}), 3))
                mine = _outline_files(key)
                for old in folder.iterdir():
                    if old != target and mine.fullmatch(old.name):
                        old.unlink(missing_ok=True)
                return target, stamp
        raise SampleError(f"'{key}' was written again while its outlines were read: try again", 409)

    def counts(self, sid: str, key: str, box=None, progress=None, should_stop=None, cached_only: bool = False,
               rule: dict | None = None, count_by: str = "outlines", table: str | None = None) -> Path | None:
        """Compare's counts of a shapes element (helper `count`: the .npz layout is there), each by its own count
        (count_source): with `table` (`count_by` "proseg" or "xenium") that table's own, its X, named by the table and
        its version (_table_count: a table written again is read again); else (`count_by` "outlines") the transcripts
        inside its outlines under the run's counting rule `rule` {min_qv, exclude} (count_rule): transcripts of quality
        min_qv or more, none whose gene name the pattern `exclude` matches (None, or either one missing: quality 20,
        real genes only). Cached per count and element stamp: counts/<key>-<rule>-<stamp>.npz for the whole element (it
        serves any box), else counts/<key>-<rule>-<stamp>-<x0>_<y0>_<x1>_<y1>.npz for the outlines whose centre lies in
        `box` (a cached box holding `box` serves too), so counts under different rules, or of different tables, never
        share a file. Counted once at a time per file; files of another stamp or counted the old way go, and of this
        rule's box files those beyond the COUNTS_KEPT newest. `cached_only`: None instead of counting. SampleError: 422
        for a rule or a count that is not one, 503 when the element's or the table's folder does not answer
        (_readable_stamp)."""
        self._segmentation(sid, key)
        if count_by == "outlines" and table is None:
            min_qv, exclude, name = self.count_rule(rule)
            asked, version = {"min_qv": min_qv, "exclude": exclude}, None
        elif count_by in COUNT_BY and count_by != "outlines" and isinstance(table, str) and KEY.fullmatch(table):
            name, version = self._table_count(sid, count_by, table)
            asked = {"table": table, "by": count_by}
        else:
            raise SampleError("A column is counted by its own table, proseg's or Xenium's, or by its outlines")
        stamp = self._readable_stamp(sid, key)
        folder = self.cache / sid / "counts"
        prefix = f"{key}-{name}-{stamp}"
        whole = folder / f"{prefix}.npz"
        if whole.exists():
            return whole
        target = whole
        if box is not None:
            box = [float(v) for v in box]
            for found, held in self._count_boxes(folder, prefix):
                if held[0] <= box[0] and held[1] <= box[1] and held[2] >= box[2] and held[3] >= box[3]:
                    return found
            target = folder / f"{prefix}-{'_'.join(f'{v:g}' for v in box)}.npz"
        if cached_only:
            return None
        with self.lock:
            guard = self.exporting.setdefault(("counts", str(target)), threading.Lock())
        with self._held(guard, should_stop):
            if not target.exists():
                self.helper("count", {"path": self.path(sid), "key": key, "points_key": self.detail(sid)["points_key"],
                                      "out": str(target), "box": box, **asked},
                            progress=progress, should_stop=should_stop, timeout=3600)
                # The helper reads the element in its own process: written again meanwhile (a notebook, another
                # engine), its counts may be of the other version. Never kept under this stamp; nor a table's.
                if self.element_stamp(sid, key) not in (None, stamp):
                    target.unlink(missing_ok=True)
                    raise SampleError(f"'{key}' was written again while it was counted: compare again", 409)
                if table is not None and self.table_stamp(sid, table) not in (None, version):
                    target.unlink(missing_ok=True)
                    raise SampleError(f"'{table}' was written again while it was read: compare again", 409)
                self._prune_counts(folder, key, prefix, stamp, name)
        return target

    def _table_count(self, sid: str, count_by: str, table: str) -> tuple[str, str]:
        """(the name a table's own counts are kept under, the table's version: table_stamp): t<8 hex digits of whose
        count, the table and its version>-v<COUNT_VERSION>, read again once when its folder does not answer, else
        SampleError 503 (_readable_stamp). Never a counting rule's name (those start with qv)."""
        from ..zarr_bridge.helper import COUNT_VERSION
        version = self._readable_stamp(sid, table, self.table_stamp)
        digest = hashlib.sha1(f"{count_by}|{table}|{version}".encode()).hexdigest()[:8]
        return f"t{digest}-v{COUNT_VERSION}", version

    @staticmethod
    def count_rule(rule: dict | None) -> tuple[float, str | None, str]:
        """(min_qv, exclude, the rule's name: the helper's count_rule) of a counting rule {min_qv?, exclude?} (None, or
        either missing: the default's); SampleError 422 in plain words for one that is not a rule."""
        from ..zarr_bridge.helper import COUNT_MIN_QV, HelperError, count_rule
        if rule is None:
            rule = {}
        if not isinstance(rule, dict):
            raise SampleError("A counting rule is {min_qv, exclude}")
        min_qv = COUNT_MIN_QV if rule.get("min_qv") is None else rule["min_qv"]
        exclude = rule.get("exclude")
        try:
            name = count_rule(min_qv, exclude)
        except HelperError as exc:
            raise SampleError(str(exc)) from None
        return float(min_qv) + 0.0, exclude, name

    @staticmethod
    def _count_boxes(folder: Path, prefix: str) -> list[tuple[Path, list[float]]]:
        """The cached box files of an element's counts (prefix: <key>-<rule>-<stamp>), newest first, with their boxes."""
        number = r"(-?[0-9.]+(?:e[+-]?[0-9]+)?)"
        pattern = re.compile(re.escape(prefix) + "-" + "_".join([number] * 4) + r"\.npz")
        found = []
        try:
            for path in folder.iterdir():
                match = pattern.fullmatch(path.name)
                if match:
                    found.append((path.stat().st_mtime, path, [float(v) for v in match.groups()]))
        except (OSError, ValueError):
            return []
        return [(path, box) for _, path, box in sorted(found, key=lambda f: -f[0])]

    def _prune_counts(self, folder: Path, key: str, prefix: str, stamp: str, name: str = ""):
        """After a count at `stamp` (a real one) under `name`: the element's count files that can never serve again go
        (another stamp, counted the old way: another COUNT_VERSION, or, after a table's own count, another table or
        version of one: an element reads one table), and of this count's (`prefix`) box files those beyond the
        COUNTS_KEPT newest. Counts under the other rules of this version stay: runs with other input rules compare too."""
        from ..zarr_bridge.helper import COUNT_VERSION
        mine = _count_files(key)
        try:
            for path in folder.iterdir():
                match = mine.fullmatch(path.name)
                if match and (match.group(3) != stamp or int(match.group(2)) != COUNT_VERSION
                              or name.startswith("t") and match.group(1).startswith("t") and match.group(1) != name):
                    path.unlink(missing_ok=True)            # another stamp, counted the old way, another table's
            for path, _ in self._count_boxes(folder, prefix)[COUNTS_KEPT:]:
                path.unlink(missing_ok=True)
        except OSError:
            pass

    def tissue(self, sid: str, key: str | None = None) -> dict:
        """{source, polygons, shapes, area_um2}: sopa's `region_of_interest` when the sample has one, else sopa's tissue
        detection run in memory by the helper on the image's lowest level; cached here, never written into the sample.
        `shapes` keeps each polygon's holes ([exterior, hole, …]) for the runs; `polygons` is every ring, for drawing.
        Stain runs and training patches stay inside it when "Tissue only" is on."""
        from ..zarr_bridge.tissue import DETECTION_VERSION
        image = self._image_info(sid, key)
        detail = self.detail(sid)
        roi = ROI in {s["key"] for s in detail["shapes"]}
        name = hashlib.sha1(image["key"].encode()).hexdigest()[:12]
        cached = self.cache / sid / f"tissue-{name}.json"
        # The way it was found is in the stamp: a sample's tissue found an earlier way (no tissue on H&E) is found again.
        stamp = [self._mtime(self.path(sid)), image["key"], roi, DETECTION_VERSION]
        try:
            found = json.loads(cached.read_text())
            if found.get("stamp") != stamp:
                raise ValueError("stale")
        except (OSError, ValueError):
            args = {"path": self.path(sid), "image": image}
            if roi:
                args["points_key"] = detail["points_key"]
            found = self.helper("tissue", args, timeout=600)
            _atomic_write(cached, json.dumps({**found, "stamp": stamp}).encode())
        shapes = [[r for r in polygon if len(r) >= 3] for polygon in found["polygons"]]
        shapes = [s for s in shapes if s]
        source = ("region_of_interest" if roi else "detected") if shapes else "none"
        return {"source": source, "polygons": [r for s in shapes for r in s], "shapes": shapes,
                "area_um2": float(found.get("area_um2") or 0.)}

    # ---- proseg input and results -----------------------------------------------------------------------------
    def prepare(self, sid: str, region: dict, prior: dict, destination: Path, progress=None, should_stop=None) -> dict:
        detail = self.detail(sid)
        return self.helper("prepare", {"path": self.path(sid), "points_key": detail["points_key"], "region": region,
                                       "prior": prior, "out": str(destination)},
                           progress=progress, should_stop=should_stop, timeout=4 * 3600)

    def existing_elements(self, sid: str) -> list[str]:
        return self.elements_of(self.path(sid))

    @staticmethod
    def elements_of(path) -> list[str]:
        """Every element name of the zarr at `path` (a linked element's folder counts), [] for one that cannot be read."""
        root = Path(path)
        names = set()
        for kind in ELEMENT_KINDS:
            try:
                names.update(p.name for p in (root / kind).iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                continue
        return sorted(names)

    def save_defaults(self, sid: str, kind: str = "proseg", method: str = "cellpose", also=()) -> dict:
        """The names a save is offered: the next free ones, compared ignoring case (a case twin is refused at the save),
        against the sample's element names and `also` (API mode: the notebook's object's and its
        zarr's); `existing` lists them all."""
        existing = sorted(set(self.existing_elements(sid)) | {str(name) for name in also})
        folded = {s.casefold() for s in existing}
        if kind == "stain":
            prefix = "instanseg" if method == "instanseg" else "cellpose"
            n = 1
            while f"{prefix}_boundaries_{n}".casefold() in folded:
                n += 1
            return {"shapes_key": f"{prefix}_boundaries_{n}", "table_key": None, "existing": existing}
        n = 1
        while True:
            suffix = "" if n == 1 else f"_{n}"
            shapes_key, table_key = f"proseg_boundaries{suffix}", f"proseg_table{suffix}"
            if shapes_key.casefold() not in folded and table_key.casefold() not in folded:
                return {"shapes_key": shapes_key, "table_key": table_key, "existing": existing}
            n += 1

    def save(self, sid: str, output: Path, shapes_key: str, table_key: str, set_default: bool, overwrite: bool,
             provenance: dict, progress=None, exclude=None, min_area_um2=None, min_brightness=None, job=None, keep=None,
             deliver: dict | None = None) -> dict:
        """`exclude`: lasso rings (frame µm) whose cells, by centre, stay out of a stain run's saved shapes;
        `min_area_um2`: cells smaller than this stay out too, and with `min_brightness` ({channel, value}) cells dimmer
        than it in the image of the run's `job` (its job.json; engine/stain_intensity.py). New stain saves pass `keep`,
        a bool .npy file in resolved-outline order; the helper checks its length before writing anything.
        `deliver` (API mode): where the helper writes instead of the sample alone, {path: the object's zarr or None,
        view: the engine's view when that is not it, frame_from, stage_to, consolidate} (Runs._delivery_where); the
        answer then also says `staged`, `written_to` and `view_written`."""
        detail = self.detail(sid)
        args = {"path": self.path(sid), "points_key": detail["points_key"], "output": str(output),
                "shapes_key": shapes_key, "table_key": table_key, "set_default": set_default,
                "overwrite": overwrite, "provenance": provenance, "kind": provenance.get("kind", "proseg"),
                "exclude": exclude or [], "min_area_um2": min_area_um2,
                "min_brightness": min_brightness, "job": job, "keep": str(keep) if keep is not None else None}
        if deliver is not None:
            args.update({key: deliver.get(key) for key in ("path", "view", "frame_from", "stage_to", "consolidate")})
        result = self.helper("save", args, progress=progress, timeout=4 * 3600)
        with self._detail_lock(sid):
            with self.lock:      # the sample changed: re-inspect on next open, forget cached outlines of replaced layers
                self.details.pop(sid, None)
                for key in (shapes_key,):
                    self.outlines.pop((sid, key), None)
                    self.outline_stamps.pop((sid, key), None)
            for name in ("inspect.json", "segmentations.json"):
                try:
                    (self.cache / sid / name).unlink()
                except OSError:
                    pass
            for suffix in (".bin", ".json", ".stamp"):
                try:
                    Path(str(self.cache / sid / "shapes" / shapes_key) + suffix).unlink()
                except OSError:
                    pass
            # Its outlines for the layers and its counts for Compare are kept by element stamp (a new one now): the old go.
            for folder, pattern in (("segmentations", _outline_files(shapes_key)), ("counts", _count_files(shapes_key))):
                try:
                    for path in (self.cache / sid / folder).iterdir():
                        if pattern.fullmatch(path.name):
                            path.unlink(missing_ok=True)
                except OSError:
                    pass
        return result
