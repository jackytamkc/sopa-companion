"""Image runs. Only the helper imports Cellpose / sopa; the engine serves committed outline arrays."""
from __future__ import annotations

from datetime import datetime
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np

from .. import portable
from . import cellpose_settings, children, stain_levels, stain_rules, vectors
from .stain_exclusions import meet
from .runs import ACTIVE, ONE_RUN, Run, RunError, atomic_text, now, region_spec

# The methods by the names their refusals give (StainRuns.offered).
METHOD_NAMES = {"cellpose": "Cellpose", "instanseg": "InstanSeg"}


class StainRuns:
    def __init__(self, runs, config, geometry_python=None):
        self.runs = runs
        # The sopa environment's interpreter ([stain] python); unset, stain segmentation is unavailable (as the other
        # engines treat an unset interpreter), never someone else's path.
        self.python = config.get("python", "")
        self.module = config.get("helper_module", "sopa_companion.zarr_bridge.stain")
        self.patch_export_module = config.get("patch_export_module", "sopa_companion.zarr_bridge.stain_patches")
        # InstanSeg is dormant: it is not ready for users, so the engine offers it only while the internal switch
        # [stain] instanseg_enabled is true. Off (the default) health lists Cellpose alone, InstanSeg's runs and model
        # list are refused (offered) and [stain] instanseg_python is never started; its code stays as it is.
        self.instanseg_enabled = config.get("instanseg_enabled", False)
        # InstanSeg's own environment ([stain] instanseg_python; load_config expands "~"): unset, InstanSeg is unavailable.
        self.instanseg_python = os.path.expanduser(config.get("instanseg_python") or "")
        self.instanseg_module = config.get("instanseg_module", "sopa_companion.zarr_bridge.stain_instanseg")
        self.instanseg_models = self.runs.root.parent / "instanseg-models"
        # A computer with one graphics card, unless [stain] gpu says another (counted within the engine's
        # CUDA_VISIBLE_DEVICES: engine/children.py gpu).
        self.gpu = config.get("gpu", 0)
        self.tile = config.get("tile_pixels", 1024)
        self.overlap = config.get("overlap_pixels", 128)
        self._models = None
        self._models_until = 0
        self._models_lock = threading.Lock()
        # The list's "learned at" labels, worked out once per refresh of the list (they read the exported sets'
        # folders), outside _models_lock: (the list they label, the labelled list).
        self._labelled = None
        self._labels_lock = threading.Lock()
        self._instanseg_models = None
        from .stain_worker import StainWorker
        self.worker = StainWorker(config.get("patch_python", geometry_python or self.python),
                                  config.get("patch_module", "sopa_companion.zarr_bridge.stain_patches"))
        from .stain_patches import StainPatches
        self.patches = StainPatches(self, config)
        from .stain_exclusions import StainExclusions
        self.exclusions = StainExclusions(self)
        if type(self.instanseg_enabled) is not bool:
            raise ValueError("stain.instanseg_enabled must be true or false")
        if type(self.gpu) is not int or self.gpu < 0:
            raise ValueError("stain.gpu must be a non-negative integer")
        if type(self.tile) is not int or not 64 <= self.tile <= 4096:
            raise ValueError("stain.tile_pixels must be between 64 and 4096")
        if type(self.overlap) is not int or not 0 < self.overlap < self.tile / 2:
            raise ValueError("stain.overlap_pixels must be positive and less than half the tile size")

    @property
    def available(self):
        return bool(self.python and Path(self.python).is_file())

    @property
    def methods(self):
        """The methods this engine offers, in the picker's order: Cellpose, and InstanSeg while it is switched on."""
        return ("cellpose", "instanseg") if self.instanseg_enabled else ("cellpose",)

    def offered(self, method):
        """`method` when this engine offers it, else refused before anything is made or started: "Choose Cellpose"
        (422), the methods it offers by name."""
        if not isinstance(method, str) or method not in self.methods:
            raise RunError("Choose " + " or ".join(METHOD_NAMES[m] for m in self.methods))
        return method

    def method_available(self, method):
        return self.available and method in self.methods and (method == "cellpose" or
                                                              Path(self.instanseg_python).is_file())

    def helper_env(self, cache, *pythons):
        """A stain helper's environment (engine/children.py: the app on PYTHONPATH as the helper's environment needs it,
        on the desktop the app alone, as before; no desktop), [stain] gpu as its only GPU, small thread pools, its
        caches in `cache`; on a Mac, torch's operations that Apple's GPU lacks run on the CPU instead of failing
        (PYTORCH_ENABLE_MPS_FALLBACK, nothing elsewhere). `pythons`: the interpreters it runs ([stain] python by
        default; an InstanSeg run's helper starts [stain] instanseg_python too, which inherits this)."""
        same = all(children.kind(python) == "same" for python in (pythons or (self.python,)))
        gpu = self.gpu if type(self.gpu) is int else None
        env = children.env("same" if same else "other", gpu=gpu, inherit=False)
        if gpu is None:                    # a test's "": the helpers see no GPU
            env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   NUMBA_CACHE_DIR=str(cache / "numba"), MPLCONFIGDIR=str(cache / "matplotlib"),
                   XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache / "torch"), PYTHONDONTWRITEBYTECODE="1",
                   PYTORCH_ENABLE_MPS_FALLBACK="1")
        for key in ("INSTANSEG_BIOIMAGEIO_PATH", "INSTANSEG_MODEL_PATH", "INSTANSEG_TORCHSCRIPT_PATH"):
            env[key] = str(self.instanseg_models)
        return env

    def run_pythons(self, settings):
        """The interpreters a run's helper starts: [stain] python, and InstanSeg's for an InstanSeg run."""
        method = (settings or {}).get("method", "cellpose")
        return (self.python, self.instanseg_python) if method == "instanseg" else (self.python,)

    def models(self, method="cellpose", labels=True):
        """The models offered for `method` (copies). Cellpose's list is refreshed every 60 s; `labels`: with each custom
        model's "learned at" label (_model_labels), worked out once per refresh. Start and the GUI settings ask without
        labels: they never wait for the sets' folders to be read."""
        self.offered(method)
        with self._models_lock:
            if method == "instanseg":
                if self._instanseg_models is None:
                    try:
                        result = subprocess.run([self.instanseg_python, "-m", self.instanseg_module, "--models"],
                            env=self.helper_env(self.runs.root.parent / "stain-cache", self.instanseg_python),
                            cwd=children.app_path(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=60, check=True, **portable.no_window())
                        found = json.loads(result.stdout)["models"]
                        if not isinstance(found, list) or not found:
                            raise ValueError("No models")
                    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                        raise RunError("The model list could not be loaded. Try opening the picker again.", 503) from exc
                    self._instanseg_models = found
                return [dict(model) for model in self._instanseg_models]
            if self._models is None or time.monotonic() >= self._models_until:
                try:
                    result = subprocess.run([self.python, "-m", self.module, "--models"],
                        env=self.helper_env(self.runs.root.parent / "stain-cache"), cwd=children.app_path(),
                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=True,
                        **portable.no_window())
                    found = json.loads(result.stdout)["models"]
                    if not isinstance(found, list) or not found:
                        raise ValueError("No models")
                except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                    raise RunError("The model list could not be loaded. Try opening the picker again.", 503) from exc
                self._models, self._models_until = found, time.monotonic() + 60
            found = self._models
        if not labels:
            return [dict(model) for model in found]
        with self._labels_lock:
            with self._models_lock:
                labelled = self._labelled
            if labelled is None or labelled[0] is not found:
                labelled = (found, self._model_labels(found))
                with self._models_lock:
                    if self._models is found:
                        self._labelled = labelled
        return [dict(model) for model in labelled[1]]

    def _model_labels(self, models):
        """A model saved by the GUI is associated by name with its patch set, never by hashing its weights. A custom one
        so found carries the level its patches were cut at: `resolution` {level, factor} as the set recorded them (the
        models route sends it; the picker adds "learned at scale1": stain.mjs modelLabel) and `label`. The sets are read
        oldest first, so the newest set naming a model gives its level; one that cannot be read is passed over."""
        learned = {}
        try:
            sources = self.patches.sources()
        except (OSError, ValueError, TypeError):
            sources = []
        for source in sources:
            try:
                settings = cellpose_settings.gui_settings([source], models)
                name = (settings or {}).get("settings", {}).get("model")
                if name:
                    # The level the patches were cut at; a set exported before Resolution was cut at scale0 (level 0).
                    learned[name] = {k: settings["resolution"][k] for k in ("level", "factor")}
            except (OSError, ValueError, TypeError):
                continue
        result = []
        for model in models:
            entry = dict(model)
            resolution = learned.get(model["name"])
            if model.get("kind") == "custom" and resolution is not None:
                level = stain_levels.resolution_name(resolution) or stain_levels.level_name(0)
                entry.update(resolution=resolution, label=f"{model['name']} · learned at {level}")
            result.append(entry)
        return result

    def gui_settings(self, sample):
        """What the Cellpose GUI saved in this sample's exported patch sets, wherever they were exported (the newest
        file wins)."""
        sources = self.patches.sources(sample)
        detail = self.runs.zarr.detail(sample)
        found = cellpose_settings.gui_settings(sources, self.models(labels=False), images=detail.get("images", []))
        return {"found": False} if found is None else {"found": True, **found}

    def settings(self, detail, value, empty=False):
        if not isinstance(value, dict):
            raise RunError("Choose an image and its channels")
        if type(value.get("tissue_only", True)) is not bool:
            raise RunError("Choose whether to stay inside the tissue")
        k = value.get("level", 0)
        if type(k) is not int or k < 0:
            raise RunError("Choose a resolution this image has")
        return {**self._settings(detail, value, empty), "tissue_only": value.get("tissue_only", True), "level": k}

    def _settings(self, detail, value, empty):
        method = self.offered(value.get("method", "cellpose"))
        image = next((i for i in detail.get("images", []) if i["key"] == value.get("image")), None)
        if image is None:
            raise RunError("Choose an image from this sample")
        if method == "instanseg":
            name = value.get("model", "fluorescence_nuclei_and_cells")
            if not any(m["name"] == name for m in self.models(method)):
                raise RunError("Choose a model from the list")
            channels = value.get("channels")
            if (not isinstance(channels, list) or not channels or
                    any(not isinstance(c, str) or c not in image["channels"] for c in channels) or
                    len(set(channels)) != len(channels)):
                raise RunError("Choose channels from this image")
            if name == "single_channel_nuclei" and len(channels) != 1:
                raise RunError("Choose one channel for this model")
            if name == "brightfield_nuclei" and len(channels) != 3:
                raise RunError("Choose red, green and blue channels in that order")
            dual = name == "fluorescence_nuclei_and_cells"
            output = value.get("output", "cells" if dual else "nuclei")
            resolve = value.get("resolve_cells", False)
            if output not in ("cells", "nuclei") or not dual and output != "nuclei":
                raise RunError("Choose cells or nuclei for this model")
            if type(resolve) is not bool or resolve and not dual:
                raise RunError("Choose whether to match cells and nuclei")
            return dict(method=method, model=name, model_kind="builtin", image=image["key"],
                        channels=channels, output=output, resolve_cells=resolve)
        nucleus, cyto = value.get("nucleus"), value.get("cytoplasm")
        if nucleus not in image["channels"] or (cyto is not None and cyto not in image["channels"]):
            raise RunError("Choose channels from this image")
        if nucleus == cyto:
            raise RunError("Choose different channels for the nucleus and cell body")
        name = value.get("model", "cpsam")
        if not isinstance(name, str) or name in (".", "..") or any(c in name for c in "/\\\0"):
            raise RunError("Choose a model from the list")
        # The list without its "learned at" labels: Start never waits for the exported sets' folders to be read.
        listed = [] if empty else self.models(labels=False)
        model = {"kind": "builtin"} if empty else next((m for m in listed if m["name"] == name), None)
        if model is None:
            raise RunError("Choose a model from the list")
        result = {"method": "cellpose", "model": name, "model_kind": model["kind"], "image": image["key"],
                  "nucleus": nucleus, "cytoplasm": cyto}
        # GUI-trained models from InstanSeg patches use the same first three image channels during segmentation.
        if "channels" in value:
            channels = value["channels"]
            if (not isinstance(channels, list) or not 1 <= len(channels) <= 3 or
                    any(not isinstance(c, str) or c not in image["channels"] for c in channels) or
                    len(set(channels)) != len(channels)):
                raise RunError("Choose one to three channels from this image")
            result["channels"] = channels
        if model["kind"] == "custom":
            result["model_path"] = model["path"]
        try:
            result.update(cellpose_settings.validate(value))
        except cellpose_settings.SettingsError as exc:
            raise RunError(str(exc)) from None
        return result

    def prepare(self, sample, region, settings, kind="stain", models=True):
        """Every check of a new run or patch item, and its helper job, in memory: create() writes them, the patch
        preview only asks the patch helper for boxes (engine/stain_patches.py). `models=False` (the preview): the boxes
        do not depend on the model, so the model list is not consulted (it can take seconds to refresh)."""
        runs = self.runs
        patches = kind == "patches"
        start = settings.get("start_from", "model") if isinstance(settings, dict) else "model"
        count, size = (settings.get("count", 8), settings.get("size", 256)) if isinstance(settings, dict) else (8, 256)
        if patches and (start not in ("model", "empty") or type(count) is not int or not 1 <= count <= 64 or
                        type(size) is not int or not 64 <= size <= 4096):
            raise RunError("Choose Model outlines or Empty, 1–64 patches and a size of 64 to 4096 pixels")
        empty = patches and start == "empty"
        if not empty:
            # A method this engine does not offer (InstanSeg while it is switched off) is refused as a choice (422)
            # before the software it would need is looked for (503).
            method = self.offered(settings.get("method", "cellpose") if isinstance(settings, dict) else "cellpose")
            if not self.method_available(method):
                raise RunError("The image segmentation software is unavailable", 503)
        if runs.zarr is None or not runs.zarr.known(sample):
            raise RunError("Open a SpatialData sample first")
        detail = runs.zarr.detail(sample)
        settings = self.settings(detail, {**settings, "method": "cellpose"} if empty else settings,
                                 empty=empty or not models)
        spec = region_spec(region, detail["extent"])
        image = runs.zarr._image_info(sample, settings["image"])
        level = image["levels"][0]
        if any(n > 1 and d not in ("c", "x", "y") for d, n in zip(level["dims"], level["shape"])):
            raise RunError("Choose a two-dimensional image")
        if level["dims"][-2:] != ["y", "x"]:
            raise RunError("This image must store rows and columns last")
        resolution = next((s for s in stain_levels.offered(image) if s["level"] == settings["level"]), None)
        if resolution is None:
            raise RunError("Choose a resolution this image has")
        settings["resolution"] = resolution
        if settings["method"] == "instanseg":
            _, _, sx, sy = runs.zarr._geometry(image, level)
            if not all(math.isfinite(s) and s > 0 for s in (sx, sy)) or not math.isclose(sx, sy, rel_tol=1e-5):
                raise RunError("InstanSeg needs an image with equal pixel sizes along both axes")
            settings["pixel_size"] = math.sqrt(resolution["um_per_px"] * resolution["um_per_px_y"])
        level = image["levels"][settings["level"]]
        window = runs.zarr._window(image, level, spec["bbox"])
        if window[2] <= window[0] or window[3] <= window[1]:
            raise RunError("The region is outside this image")
        across = min(window[2] - window[0], window[3] - window[1])
        if across < stain_levels.MIN_ACROSS:
            raise RunError(f"Resolution: the region is only {across} pixels across at "
                           f"{stain_levels.level_name(settings['level'])}")
        tissue = {"source": "off", "shapes": [], "area_um2": None}
        if settings["tissue_only"]:
            tissue = runs.zarr.tissue(sample, image["key"])
            tissue.setdefault("shapes", [[ring] for ring in tissue.get("polygons", [])])
            if tissue["source"] == "none":
                raise RunError("No tissue was found in this sample. Untick Tissue only to use it anyway.")
            x0, y0, x1, y1 = spec["bbox"]
            outline = spec["points"] if spec.get("kind") == "polygon" else [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            tissue["shapes"] = [s for s in tissue["shapes"] if meet(s[0], outline)]     # holes travel with them
            if not tissue["shapes"]:
                raise RunError("This region has no tissue. Untick Tissue only to use it anyway.")
        job = {"path": detail["path"], "image": image, "region": spec, "settings": settings,
               "out": None, "tile_pixels": self.tile, "overlap_pixels": self.overlap,
               "tissue": tissue["shapes"] or None, "stitch": "own"}
        if settings["method"] == "instanseg":
            job.update(instanseg_python=self.instanseg_python, instanseg_module=self.instanseg_module)
        return {"detail": detail, "spec": spec, "image": image, "settings": settings, "tissue": tissue, "job": job,
                "start_from": start, "count": count, "size": size, "window": window}

    def create(self, sample, region, settings, kind="stain"):
        """A stain run or Training patches item, for Runs.create, which refused it already while another run is open
        (check_open) before this prepares it; the registry's checks are made again where it is registered."""
        runs = self.runs
        patches = kind == "patches"
        ready = self.prepare(sample, region, settings, kind)
        # Boxes the picker previewed are checked before anything exists, then kept as the item's first draft.
        first = self.patches.first(ready, settings.get("boxes")) if patches else None
        detail, spec, image, tissue, settings = (ready[k] for k in ("detail", "spec", "image", "tissue", "settings"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{kind}-{re.sub(r'[^A-Za-z0-9-]+', '-', detail['name'])[:40].strip('-')}"
        with runs.lock:
            # Under the registry's lock, where the run is registered: a close that began
            # meanwhile, or a run registered meanwhile (any but a test square's; the older runs are kept apart
            # from the registry, in Runs.older, so they never count).
            if runs.closing:
                raise RunError("The engine is shutting down", 503)
            if runs.one_run and any(not r.meta.get("qc_test") for r in runs.runs.values()):
                raise RunError(ONE_RUN, 409)
            if not patches and sum(r.meta["state"] in ACTIVE for r in runs.runs.values()) >= runs.max_active:
                raise RunError("Finish or stop another run first", 409)
            run_id, n = base, 1
            while run_id in runs.runs or (runs.root / run_id).exists():
                n += 1
                run_id = f"{base}-{n}"
            directory = runs.root / run_id
            directory.mkdir()
            box = spec["bbox"]
            meta = {"id": run_id, "kind": kind, "sample": sample, "sample_name": detail["name"],
                    "session_model": 2, "sample_tools": detail.get("tools", ["proseg", "stain"] if detail.get("transcripts") else ["stain"]),
                    "covered": None,
                    "sample_kind": "zarr", "sample_path": detail["path"], "region": spec, "box": box,
                    "mode": "tiled" if max(box[2]-box[0], box[3]-box[1]) > runs.max_region else "detail",
                    "stain": settings, "state": "paused", "paused": True, "stain_started": False,
                    "tissue": {"source": tissue["source"], "area_um2": tissue["area_um2"], "tiles_skipped": None},
                    "created": now(), "message": "",
                    "outputs": [], "zarr_save": None, "progress": {"done": 0, "total": 0, "revision": 0},
                    "options": {}, "prior": None,
                    "exclusions": {"version": 0, "areas_version": 0, "digest": stain_rules.digest({}),
                                   "count": 0, "cells": 0, "kept": 0, "removed": 0, "in_areas": 0,
                                   "size": None, "by_size": 0, "brightness": {}, "by_brightness": 0, "error": None},
                    "image": {"source": "sample", "image": image["key"],
                              "channels": runs.zarr._stats(sample, detail["path"], image)}}
            if any(isinstance(note, dict) and note.get("code") == "pixel_frame" for note in detail.get("notes") or []):
                # Its transcripts seem to be in pixels (zarr_samples.open_notes): sizes in µm are pixels there, so
                # the run's page makes no µm claim either (stain-patches.mjs patchSizeNote), as the picker does not.
                meta["pixel_frame"] = True
            run = Run(directory, meta)
            if patches:
                meta.update(state="done", paused=False, start_from=ready["start_from"],
                            patch_count=ready["count"], patch_size=ready["size"])
            run.ack = {"state": "paused", "tiles": 0}
            (directory / "stain").mkdir()
            (directory / "output").mkdir()
            atomic_text(directory / "stain" / "job.json", json.dumps({**ready["job"], "out": str(directory)}))
            if first is not None:
                self.patches.save(run, {**self.patches.draft(run), **first, "version": 1, "revision": 0})
            run.save()
            runs.runs[run_id] = run
        runs.changed()
        return run.public()

    def write_control(self, run):
        atomic_text(run.dir / "stain" / "control.json",
                    json.dumps({"paused": run.meta.get("paused", False), "stop": run.meta.get("stain_stop", False)}))

    def job_file(self, run, join=False):
        """The job file this run's helper reads, written again at each launch from stain/job.json as this engine runs
        it now: the run's folder where it is now (the job names it by its full path, from when the run was made, and a
        data folder moved since, as an app folder moves with its var/, would send the helper to the old place) and,
        for an InstanSeg run, this engine's InstanSeg environment and module (an earlier version's job names its own
        package's). A join gets join.json, the same job marked join_only."""
        path = run.dir / "stain" / "job.json"
        job = {**json.loads(path.read_text()), "out": str(run.dir)}
        if (job.get("settings") or {}).get("method") == "instanseg":
            job.update(instanseg_python=self.instanseg_python, instanseg_module=self.instanseg_module)
        if join:
            path, job = path.with_name("join.json"), {**job, "join_only": True}
        atomic_text(path, json.dumps(job))
        return path

    def launch(self, run, join=False):
        run.stop_requested = False
        self.write_control(run)
        run.worker = threading.Thread(target=self.execute, args=(run, join), name=f"stain-{run.meta['id']}", daemon=True)
        self.runs.changed()
        run.worker.start()

    def execute(self, run, join=False):
        runs, meta = self.runs, run.meta
        process = None
        try:
            if not join:
                if not runs.zarr.known(meta["sample"]):
                    runs.zarr.open(meta["sample_path"], recent=False)
                info = runs.zarr._image_info(meta["sample"], meta["stain"]["image"])
                image = {"source": "sample", "image": info["key"],
                         "channels": runs.zarr._stats(meta["sample"], meta["sample_path"], info)}
                runs._set(run, image=image, state="running", started=meta.get("started") or now(), message="")
            env = self.helper_env(run.dir / "stain/cache", *self.run_pythons(meta["stain"]))
            job_path = self.job_file(run, join)
            command = [self.python, "-m", self.module, str(job_path)]
            launcher = children.helper_prefix("stain")       # ends with the engine: its tiles are its own
            with open(run.dir / "stain.log", "a") as log:
                log.write("$ " + " ".join(command) + "\n")
                log.flush()
                with run.lock:
                    if run.stop_requested:
                        raise RuntimeError("Stopped before the helper started")
                    process = run.process = subprocess.Popen(launcher + command, env=env, cwd=run.dir,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                        **portable.no_window())
                    portable.tie(process)                    # macOS and Windows: setpriv's death signal there
                runs._set(run, pid=process.pid)
                ended = False
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = json.loads(line)
                    kind = event.get("ev")
                    if kind == "hello":
                        runs._set(run, stain_runtime=event, message="",
                                  stain={**meta["stain"], **{k: event[k] for k in
                                         ("model_path", "model_sha256") if k in event}})
                    elif kind == "progress":
                        runs._set(run, message=event["message"])
                    elif kind in ("plan", "tile", "done"):
                        # Files are committed before their event. The persisted revision exposes only complete tiles.
                        progress = {**event["progress"], "revision": meta["progress"]["revision"] + 1}
                        runs._set(run, progress=progress, summary=event.get("summary", meta.get("summary")),
                                  last_tile=event.get("tile", meta.get("last_tile")))
                        if kind in ("plan", "done"):
                            runs._set(run, covered=event.get("covered") if kind == "done" else None)
                        if kind == "plan" and "tiles_skipped" in event:
                            runs._set(run, tissue={**(meta.get("tissue") or {}), "tiles_skipped": event["tiles_skipped"]},
                                      tiling=event.get("tiling"))
                        # The helper's summary counts every cell: removed areas still apply to new outlines.
                        runs._set(run, summary_model=None)
                        try:
                            self.exclusions.refresh(run)
                        except Exception as exc:
                            runs.note(f"stain cells refresh {meta['id']} failed: {exc}", meta["id"])
                            runs._set(run, exclusions={**(meta.get("exclusions") or {}),
                                      "error": "The cells' values could not be read. Try again in a moment."})
                        if kind == "done":
                            ended = True
                    elif kind == "state":
                        with run.lock:
                            run.ack = {"state": event["state"], "tiles": event["done"]}
                        runs._set(run, state="finishing" if meta.get("stain_stop") else event["state"], message="")
                    elif kind == "error":
                        raise RuntimeError(event["message"])
                    elif kind == "fallback":
                        runs._set(run, stain_fallback=event,
                                  stain_runtime={**(meta.get("stain_runtime") or {}), "device": event["device"]})
                code = process.wait()
                if code or not ended:
                    raise RuntimeError(f"Image helper ended without a result (exit {code}); see the Log")
            complete = meta["progress"].get("complete", False)
            defaults, message = None, ""
            try:
                if not runs.zarr.known(meta["sample"]):
                    runs.zarr.open(meta["sample_path"], recent=False)
                # With "Make it the sample's segmentation" only for a run over the whole sample (Runs._whole).
                defaults = runs._save_defaults(run)
            except Exception as exc:
                message = f"Shapes are ready, but the sample's save names could not be read: {exc}"
            runs._set(run, state="done" if complete or meta.get("finish_requested") else "stopped",
                      paused=False, finished=now(), outputs=runs._outputs(run), message=message,
                      save_defaults=defaults)
        except Exception as exc:
            if not run.stop_requested:          # noted before anyone sees it failed
                runs.note(f"stain run {meta['id']} failed: {exc}", meta["id"])
            runs._set(run, state="stopped" if run.stop_requested else "failed", message=str(exc), finished=now())
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout:
                    process.stdout.close()
            run.process = None

    def join(self, run):
        """Join an ended run's committed tiles, without segmenting another tile. Wait until its shapes are ready."""
        with run.lock:
            if run.meta.get("kind") != "stain":
                raise RunError("This is not an image run", 404)
            if run.meta["progress"].get("resolved"):
                return run.public()
            if run.meta["state"] in ACTIVE or (run.worker and run.worker.is_alive()):
                raise RunError("Wait for the helper to stop", 409)
            if not run.meta["progress"].get("done"):
                raise RunError("The run has no cells yet", 409)
            run.meta.update(state="finishing", paused=False, stain_stop=True, finish_requested=True, covered=None)
            run.save()
            self.launch(run, join=True)
            worker = run.worker
        worker.join()
        if not run.meta["progress"].get("resolved"):
            raise RunError(run.meta.get("message") or "The cells could not be joined. Try again.", 409)
        return run.public()

    def control(self, run, action):
        with run.lock:
            meta, state = run.meta, run.meta["state"]
            if action == "resume":
                # A run whose method is switched off since it was made (a test engine keeps its runs) segments nothing
                # more: its helper would start that method's environment.
                self.offered((meta.get("stain") or {}).get("method", "cellpose"))
            if action == "resume" and state == "paused" and not meta.get("stain_started", True):
                if self.runs.closing:
                    raise RunError("The engine is shutting down", 503)
                meta.pop("paused_away", None)
                meta.update(state="preparing", paused=False, stain_started=True)
                run.ack = None
                run.save()
                self.launch(run)
                return run.public()
            if state == "paused" and not meta.get("stain_started", True) and action in ("stop", "abort", "finish"):
                meta.pop("paused_away", None)
                meta.update(state="stopped", paused=False)
                run.ack = None
                run.save()
                run.touch()
                self.runs.changed()
                return run.public()
            if action == "resume" and state in ("stopped", "failed", "done") and not meta["progress"].get("complete"):
                if (meta.get("zarr_save") or {}).get("state") == "saving":
                    raise RunError("Wait for the shapes to finish saving", 409)
                if self.runs.closing:
                    raise RunError("The engine is shutting down", 503)
                if run.worker and run.worker.is_alive():
                    raise RunError("Wait for the helper to stop", 409)
                with self.runs.lock:
                    if sum(r.meta["state"] in ACTIVE for r in self.runs.runs.values()) >= self.runs.max_active:
                        raise RunError("Finish or stop another run first", 409)
                    meta.pop("paused_away", None)
                    meta.update(state="preparing", paused=False, stain_started=True,
                                stain_stop=False, finish_requested=False, message="", covered=None)
                    meta["progress"] = {**meta["progress"], "resolved": False,
                                        "revision": meta["progress"]["revision"] + 1}
                    if (meta.get("zarr_save") or {}).get("state") in ("saved", "queued"):
                        meta["zarr_save"] = {**meta["zarr_save"], "state": "outdated",
                                            "message": "Saved shapes are from an earlier result. Save again to include the new cells."}
                run.ack = None
                run.save()
                self.launch(run)
                return run.public()
            if state not in ACTIVE:
                raise RunError("This run has ended", 409)
            if meta.get("stain_stop") and action in ("pause", "resume"):
                raise RunError("Wait for the helper to stop", 409)
            if action in ("resume", "stop", "finish", "abort"):
                meta.pop("paused_away", None)        # carried on or ended: the pause while no page showed it is over
            if action in ("pause", "resume"):
                meta["paused"] = action == "pause"
            elif action in ("stop", "finish"):
                meta.update(stain_stop=True, paused=False, state="finishing", finish_requested=action == "finish")
            elif action == "abort":
                run.stop_requested = True
                if run.process:
                    run.process.terminate()
            else:
                raise RunError("Choose Pause, Resume or Stop for this run")
            self.write_control(run)
            run.save()
            run.touch()
        self.runs.changed()
        return run.public()

    @staticmethod
    def reviewable(run):
        meta = run.meta
        return (meta["state"] == "paused" and meta.get("paused") and (run.ack or {}).get("state") == "paused" or
                meta["state"] in ("done", "stopped", "failed"))

    def cell_bytes(self, run, set=None, channel=None):
        return self.exclusions.cell_bytes(run, set, channel)

    def outline_bytes(self, run, box=None):
        """Stable k + 1 ids over the whole current set; only removed areas are omitted from these vectors."""
        with run.lock:
            progress, state = dict(run.meta["progress"]), self.exclusions.load(run)
        final = progress.get("resolved", False)
        layout = self.exclusions.layout(run, progress)
        records = json.loads((run.dir / "stain/tiles.json").read_text()) if not final and box is not None and layout else []
        parts, total = [], 0
        for i, (file, base, count) in enumerate(layout):
            if records:
                tile = records[i]["box"]
                if tile[2] < box[0] or tile[0] > box[2] or tile[3] < box[1] or tile[1] > box[3]:
                    continue
            mapped = {k: np.load(file / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                      for k in ("bounds", "offsets", "coords")} if final else None
            with nullcontext(mapped) if final else np.load(file, allow_pickle=False) as data:
                if len(data["bounds"]) != count:
                    raise RunError("The cells do not match their outlines. Reopen the run to try again.", 409)
                found, n = vectors.hits(data["bounds"], box, count)
                removed = self.exclusions.area_mask(run, file, state)
                if removed is not None and found.size:
                    found = found[~removed[found]]
                    n = len(found)
                total += n
                if found.size:
                    edges, xy = vectors.gather({"offsets": data["offsets"], "coords": data["coords"]}, found)
                    parts.append((edges, xy, (found + base + 1).astype(np.uint32)))
        offsets, xy, ids = [0], [], []
        for edges, coords, labels in parts:
            offsets.extend((edges[1:] + offsets[-1]).tolist())
            xy.append(coords)
            ids.append(labels)
        layer = vectors._layer(offsets, np.concatenate(xy) if xy else [], np.concatenate(ids) if ids else [], None)
        method = "InstanSeg" if run.meta["stain"].get("method") == "instanseg" else "Cellpose"
        with run.lock:
            if self.exclusions.set_id(run) != self.exclusions.set_id(run, progress) or self.exclusions.load(run)["areas_version"] != state["areas_version"]:
                raise RunError("The cells changed. Reload their outlines and try again.", 409)
        return vectors.encode({"cells": layer}, {"source": {"cells": method}, "revision": progress["revision"],
                              "resolved": final, "total": total, "set": self.exclusions.set_id(run, progress),
                              "areas_version": state["areas_version"]})
