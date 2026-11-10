"""Patch sets, cheap folder watches and the explicit desktop hand-off.

A helper call (boxes, a proposal, an export, a set's inspection) never runs under the run's lock: it takes seconds, a
model's export minutes, and every list of the runs (the page's pushes, GET /runs) waits for each run's lock. Under the
lock the call's inputs are read (snapshot: a copy of the run's meta, its boxes' version, its cells'
revision, its removed areas) and its answer is kept; an answer to inputs that changed meanwhile is refused with 409, as
boxes changed in another window always were.

An export goes into a folder the user chose (engine/patch_folders.py). Nothing in a folder the user chose is read or
written under a run's lock, except the export's last step: renames in that folder and its manifest.
"""
import errno
import json
import os
import sys
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

import psutil

from .. import portable
from .runs import RunError, atomic_text, now, user_cache
from . import children, stain_levels
from .patch_folders import (META, OPEN_DIR, PatchFolders, Source, legacy, read_small, valid_manifest, words)
from .zarr_samples import ZarrSamples

# The Cellpose GUI's own packages (Cellpose's `gui` extra): sopa's sopa[cellpose] installs Cellpose without them, and a
# GUI started without them closes at once. Its interpreter is asked for them before it starts (StainPatches.gui_lacking).
GUI_PROBE = ("import importlib.util as u, json\n"
             "lacking = [m for m in ('qtpy', 'pyqtgraph', 'superqt') if u.find_spec(m) is None]\n"
             "if not any(u.find_spec(m) for m in ('PyQt6', 'PyQt5', 'PySide6', 'PySide2')):\n"
             "    lacking.append('PyQt6')\n"
             "print(json.dumps(lacking))\n")
GUI_PROBE_TIMEOUT_S = 60.0
GUI_LOG_TAIL = 4096                     # bytes of this launch's part of cellpose.log read for why a GUI closed
GUI_ERROR_S = 600.0                     # a GUI that closed with an error is named this long after the engine saw it close
# An export that runs the model (start_from "model"): its process is stopped after EXPORT_FIRST_S without its first patch
# (the model's start included), then after EXPORT_PATCH_S without the next one. Per patch, never in all: a long export
# on the CPU (Cellpose without a GPU: about a minute a 256-pixel patch) goes on while each patch comes.
EXPORT_FIRST_S = 900.0
EXPORT_PATCH_S = 900.0
# The line of a GUI's log that says why it closed: the last that reads as an error (an exception's, Qt's), else its last.
GUI_ERROR_LINE = re.compile(r"Error\b|error:|Exception\b|[Cc]ould not|[Ff]ailed|[Cc]annot|[Aa]bort|[Ss]egmentation fault")


def patches_root(runs) -> Path:
    """Where sets of the earlier layout are found (<root>/<run>/<set>/, each exported into a set folder of its own): an
    API session's patches_dir (Runs.patches_dir), which outlives the session's own folders; else
    <data_dir>/stain-patches. A patches_dir is made when missing and must be a folder of this user's, not a link
    (ValueError otherwise: the engine does not start)."""
    if not runs.patches_dir:
        return runs.root.parent / "stain-patches"
    folder = Path(runs.patches_dir).expanduser()
    if not folder.is_absolute():
        raise ValueError("patches_dir must be a full folder path")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        info = folder.lstat()
    except OSError as exc:
        raise ValueError(f"patches_dir could not be made: {exc}") from None
    if not stat.S_ISDIR(info.st_mode) or not portable.mine(info):
        raise ValueError(f"patches_dir must be a folder of yours, not a link: {folder}")
    return folder


class StainPatches:
    def __init__(self, stain, config):
        self.stain, self.runs = stain, stain.runs
        self.root = patches_root(self.runs)
        # Where the Cellpose GUI opens (display): the app's own Chrome window, when it has one (desktop/window.py
        # AppWindow); else whether a page is open now, in any browser (server/app.py sets both).
        self.window = None
        self.page_here = None
        self.launcher = config.get("gui_module", "sopa_companion.zarr_bridge.stain_gui")
        self.processes, self.cache = {}, {}
        self.gui_ready: set[str] = set()       # interpreters found with the GUI's packages (gui_lacking)
        # Each folder's GUI this engine started: where its launch's part of cellpose.log begins, and when the engine
        # first saw it close with an error ((pid, monotonic time): gui_state).
        self.log_starts: dict[str, int] = {}
        self.exits: dict[str, tuple[int, float]] = {}
        self.exporting: set[str] = set()       # runs whose patches are being exported (one export per run at a time)
        self.export_progress: dict[str, dict] = {}   # their patches written so far: {done, total} (status "exporting")
        self.lock = threading.RLock()
        self.folders = PatchFolders(self)
        # Where the helper writes an export before the engine copies it into the chosen folder: emptied at each start
        # (one engine per data folder), so an export a crash cut short leaves nothing behind.
        self.staging = self.runs.root.parent / "stain-staging"
        shutil.rmtree(self.staging, ignore_errors=True)

    def require(self, run, ready=False):
        self.runs.get(run.meta["id"])
        if run.meta.get("kind") not in ("stain", "patches"):
            raise RunError("This is not an image run", 404)
        if ready and run.meta["kind"] == "stain" and (not self.stain.reviewable(run) or not run.meta["progress"].get("done")):
            raise RunError("Pause the run and wait for its cells before making patches", 409)
        if ready and self.runs.closing:
            raise RunError("The engine is shutting down", 503)

    def draft(self, run):
        path = run.dir / "patches.json"
        if path.exists():
            draft = json.loads(path.read_text())
            return {"level": 0, "factor": 1., "um_per_px": draft.get("run_pixel_to_sample", [0, 0, 1, 1])[2], **draft}
        job = json.loads((run.dir / "stain/job.json").read_text())
        image = job["image"]
        k, level, factor = stain_levels.job_level(job)
        ox, oy, sx, sy = ZarrSamples._geometry(image, level)
        x, y, _, _ = ZarrSamples._window(ZarrSamples, image, level, job["region"]["bbox"])
        return {"schema": 1, "run": run.meta["id"], "version": 0,
                "level": k, "factor": factor, "um_per_px": sx,
                "size": run.meta.get("patch_size", 256), "count": run.meta.get("patch_count", 8),
                "start_from": run.meta.get("start_from", "run"), "patches": [], "set": None,
                "run_pixel_to_sample": [ox+x*sx, oy+y*sy, sx, sy]}

    def save(self, run, value):
        atomic_text(run.dir / "patches.json", json.dumps(value))
        return value

    def version(self, draft, body):
        if type(body.get("version")) is not int or body["version"] != draft["version"]:
            raise RunError("The boxes changed. Reload them and try again", 409, reason="changed")

    def exclusions(self, run):
        return [ring.tolist() if hasattr(ring, "tolist") else ring for ring in self.stain.exclusions.areas(run)]

    def snapshot(self, run):
        """What a helper call reads of the run, copied under its lock: its folder and its meta as they are now (the call
        sends the meta while the run's own may change)."""
        with run.lock:
            meta = json.loads(json.dumps(run.meta))
            state = self.stain.exclusions.load(run)
            meta["exclusions"] = {**(meta.get("exclusions") or {}), **state,
                                  "digest": self.stain.exclusions.digest(state)}
            return SimpleNamespace(dir=run.dir, meta=meta)

    def unchanged(self, run, draft, revision, exclusions, digest=None):
        """Under the run's lock, after a helper call made without it: refuse its answer if the boxes, the run's cells or
        its removed areas changed meanwhile (another window, a resume), as for boxes changed in another window."""
        if (self.draft(run)["version"] != draft["version"] or run.meta["progress"]["revision"] != revision or
                self.exclusions(run) != exclusions or
                digest is not None and self.stain.exclusions.digest(self.stain.exclusions.load(run)) != digest):
            raise RunError("The boxes changed. Reload them and try again", 409, reason="changed")

    def change(self, run, body, propose=False):
        with run.lock:
            self.require(run, ready=True)
            draft = self.draft(run)
            self.version(draft, body)
            size, count = body.get("size", draft["size"]), body.get("count", draft["count"])
            if type(size) is not int or not 64 <= size <= 4096 or type(count) is not int or not 1 <= count <= 64:
                raise RunError("Choose 1–64 patches and a size between 64 and 4096 pixels")
            revision, exclusions, state = run.meta["progress"]["revision"], self.exclusions(run), self.snapshot(run)
            digest = self.stain.exclusions.digest(self.stain.exclusions.load(run))
        if propose:
            result = self.stain.worker.call(state, "propose", count=count, size=size, exclusions=exclusions)
        elif body.get("boxes") == []:
            result = {"patches": [], "message": ""}
        else:
            result = self.stain.worker.call(state, "boxes", boxes=body.get("boxes"), exclusions=exclusions)
        with run.lock:
            self.require(run, ready=True)
            self.unchanged(run, draft, revision, exclusions, digest)
            self.save(run, {**draft, **result, "version": draft["version"]+1, "size": size, "count": count,
                            "revision": revision, "exclusions_digest": digest})
        return self.status(run)

    def ahead(self, ready, **values):
        """The helper's boxes for a patch item not created yet (StainRuns.prepare): its job travels with the call, and
        no run folder is read or written. The helper is shared; one started here works and logs in the helpers' cache
        folder."""
        if self.runs.closing:
            raise RunError("The engine is shutting down", 503)
        folder = self.runs.root.parent / "stain-cache"
        folder.mkdir(exist_ok=True)
        meta = {"kind": "patches", "start_from": ready["start_from"], "progress": {"done": 0, "total": 0, "revision": 0}}
        return self.stain.worker.call(SimpleNamespace(dir=folder, meta=meta), "preview", job=ready["job"], **values)

    def preview(self, sample, region, settings):
        """Training patches before Start: the boxes an item created from the same region and
        settings starts with, in its frame. Refused for the same reasons as creating it."""
        ready = self.stain.prepare(sample, region, settings, kind="patches", models=False)
        found = self.ahead(ready, count=ready["count"], size=ready["size"])
        result = {k: found[k] for k in ("patches", "run_pixel_to_sample", "message")}
        result.update({k: ready["settings"]["resolution"][k] for k in ("level", "factor", "um_per_px")})
        # The boxes' frame, so the page keeps moved boxes inside the region: the window's size, and a lasso in pixels.
        x0, y0, x1, y1 = ready["window"]
        result["extent"] = [x1 - x0, y1 - y0]
        if ready["spec"].get("kind") == "polygon":
            ox, oy, sx, sy = found["run_pixel_to_sample"]
            result["outline"] = [[(x - ox) / sx, (y - oy) / sy] for x, y in ready["spec"]["points"]]
        return result

    def first(self, ready, boxes):
        """The first draft of an item created with the picker's boxes, checked as the panel's box edits are; None when
        none were sent. Runs before the item exists, so invalid boxes create nothing."""
        if boxes is None:
            return None
        return {"patches": [], "message": ""} if boxes == [] else self.ahead(ready, boxes=boxes)

    def selected(self, run, name, draft=None) -> Source:
        """Set `name` of this run as the panel reads it (patch_folders.Source): in the folder it was exported into
        (draft["exports"]), its manifest in that folder's records; a set exported before that (missing from the map)
        in its own set folder of the earlier layout. A manifest that is gone or not valid gives manifest None."""
        if not isinstance(name, str) or not re.fullmatch(r"[a-f0-9]{32}", name):
            raise RunError("Export a patch set first")
        if draft is None:
            with run.lock:
                draft = self.draft(run)
        exports = draft.get("exports") if isinstance(draft.get("exports"), dict) else {}
        if isinstance(exports.get(name), str):
            folder = Path(exports[name])
            meta, path, flat = folder / META, folder / META / "sets" / f"{name}.json", True
        else:
            folder = self.root / run.meta["id"] / name
            meta, path, flat = folder, folder / "patches.json", False
        try:
            manifest = valid_manifest(read_small(path), set=name)
        except (OSError, ValueError):
            manifest = None
        return Source(folder, manifest, meta, flat)

    def folder_listing(self, run, path=None):
        """The export dialog's view of a folder (engine/patch_folders.py PatchFolders.listing)."""
        self.require(run)
        if self.runs.closing:
            raise RunError("The engine is shutting down", 503)
        return self.folders.listing(run, path)

    def sources(self, sample=None) -> list:
        """Every exported set this engine can find, oldest first by its manifest's time and each once: a set folder of
        the earlier layout (a Path), or a set in a chosen folder ({folder, manifest, set, run}); of `sample` only when
        given. For the models' "learned at" labels and the picker's Use settings from Cellpose."""
        found = {}
        for when, folder, manifest in legacy(self.root):
            if sample is None or manifest.get("sample") == sample:
                name = manifest.get("set") if isinstance(manifest.get("set"), str) else str(folder)
                found.setdefault(name, (when, folder))
        for when, item in self.folders.sources(sample):
            found.setdefault(item["set"], (when, item))
        return [item for _, item in sorted(found.values(), key=lambda pair: pair[0])]

    def gui_cache(self) -> Path:
        """Where the Cellpose GUI starts and keeps its caches, never in the patch folder (nothing there may be
        imported or read as a cache): the helpers' cache folder on the desktop; in a notebook, the user's cache (the
        detached GUI outlives the session's folders)."""
        folder = user_cache() / "stain-gui" if self.folders.notebook() else self.runs.root.parent / "stain-cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def gui_lacking(self) -> list[str]:
        """The Cellpose GUI's packages its interpreter lacks, asked in the environment the GUI starts in: [] when it has
        them all, or when it cannot be asked (the GUI then starts as before, and gui_state says why it closed). An
        interpreter found complete is not asked again (an install adds packages; it does not take them away)."""
        python = str(self.stain.python)
        if python in self.gui_ready:
            return []
        cache = self.gui_cache()
        try:
            done = subprocess.run([python, "-c", GUI_PROBE], cwd=cache, env={**self.stain.helper_env(cache),
                                  "PYTHONSAFEPATH": "1"}, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=GUI_PROBE_TIMEOUT_S, **portable.no_window())
            lacking = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            lacking = None
        if not isinstance(lacking, list):
            return []
        if not lacking:
            self.gui_ready.add(python)
        return [str(name) for name in lacking]

    def gui_state(self, source: Source) -> tuple[bool, dict | None]:
        """(whether a GUI is open on the set's folder, why the GUI this engine started there closed with an error):
        {pid, code, reason} for GUI_ERROR_S after the engine first saw it close, else None (it runs, it closed as usual,
        or another engine started it). One look at the process answers both."""
        record = self.gui_record(source)            # read before the manager's lock: the folder may be slow
        key = str(source.folder)
        with self.lock:
            running = self.running(source, record)
            process = self.processes.get(key)
            code = process.poll() if process is not None else None
            if not code:
                return running, None
            if self.exits.get(key, (None,))[0] != process.pid:
                self.exits[key] = (process.pid, time.monotonic())
            seen, start = self.exits[key][1], self.log_starts.get(key, 0)
        if time.monotonic() - seen > GUI_ERROR_S:
            return running, None
        return running, {"pid": process.pid, "code": code, "reason": self.gui_reason(source, start, code)}

    def gui_reason(self, source: Source, start: int, code: int) -> str:
        """Why a GUI closed, from its launch's part of the set's cellpose.log (from `start`, the log's size when it
        started; at most its last GUI_LOG_TAIL bytes, opened without following a link): the last line that reads as an
        error, else the last line; for a GUI a signal ended that wrote nothing, the signal's name. Printable
        characters only, at most 300 of them."""
        lines = []
        try:
            meta = self.records(source)
        except RunError:
            meta = None
        if meta is not None:
            try:
                fd = os.open("cellpose.log", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=meta)
                try:
                    info = os.fstat(fd)
                    if stat.S_ISREG(info.st_mode) and info.st_size > start:
                        os.lseek(fd, max(start, info.st_size - GUI_LOG_TAIL), os.SEEK_SET)
                        text = os.read(fd, GUI_LOG_TAIL).decode("utf-8", "replace")
                        lines = ["".join(c if c.isprintable() else " " for c in line).strip() for line in text.splitlines()]
                        lines = [line for line in lines if line]
                finally:
                    os.close(fd)
            except OSError:
                pass
            finally:
                os.close(meta)
        said = [line for line in lines if GUI_ERROR_LINE.search(line)] or lines
        if said:
            return said[-1][:300]
        if code < 0:
            try:
                return signal.Signals(-code).name
            except ValueError:
                pass
        return ""

    def export(self, run, body):
        """The boxes' patches into the folder `body["folder"]` (checked first, outside every lock): the helper writes
        them into the app's staging folder, the engine copies them into the folder under hidden part names and then,
        under the run's lock, renames them into place and writes the export's manifest (engine/patch_folders.py)."""
        if portable.WINDOWS:              # its folders are opened by handle (engine/patch_folders.py), which Windows lacks
            raise RunError("Exporting training patches is not available on Windows yet.", 501)
        target = self.folders.check(body.get("folder"))
        with run.lock:
            self.require(run, ready=True)
            draft = self.draft(run)
            self.version(draft, body)
            if run.meta.get("kind") == "patches" and run.meta.get("start_from") == "model":
                # An item whose boxes the model outlines exports only with a method this engine offers (a test
                # engine keeps its items): its export runs that method's model, in that method's environment.
                self.stain.offered((run.meta.get("stain") or {}).get("method", "cellpose"))
            revision, exclusions, state = run.meta["progress"]["revision"], self.exclusions(run), self.snapshot(run)
            digest = self.stain.exclusions.digest(self.stain.exclusions.load(run))
            key = self.folders.key(run, draft)
            total = len(draft["patches"])
            with self.lock:                 # the export runs without the run's lock: never two of one run at once
                if state.meta["id"] in self.exporting:
                    raise RunError("These patches are being exported already. Wait for it to finish", 409)
                self.exporting.add(state.meta["id"])
                self.export_progress[state.meta["id"]] = {"done": 0, "total": total}
        name, stem, staging = uuid.uuid4().hex, None, None
        try:
            sample_name = str(state.meta.get("sample_name") or state.meta.get("sample") or "")
            stem = self.folders.reserve(target.real, sample_name, len(draft["patches"]))
            staging = self.staging / name
            staging.mkdir(parents=True)
            manifest = {**{k: v for k, v in draft.items() if k != "exports"}, "set": name, "created": now(),
                        "revision": revision, "exclusions_digest": digest, "stem": stem, "sample_name": sample_name,
                        "layout": "flat"}
            runtime = {}
            if state.meta.get("kind") == "patches" and state.meta["start_from"] == "model":
                runtime = self.model_export(run, state, manifest, staging, target.real, exclusions)
            else:
                self.stain.worker.call(state, "export", manifest=manifest, staging=str(staging),
                                       target=str(target.real), exclusions=exclusions)
            try:
                written = valid_manifest(read_small(staging / "patches.json"), set=name, stem=stem)
            except (OSError, ValueError):
                written = None
            if written is None:
                raise RunError("The patches could not be exported. Details are in the Log.", 503)
            with self.folders.opened(target.real) as folder:
                folder.copy_in(staging, written, name)          # the slow part, before the run's lock
                with run.lock:
                    self.require(run, ready=True)
                    self.unchanged(run, draft, revision, exclusions, digest)
                    folder.publish()
                    folder.write_manifest(name, written)
                    exports = draft.get("exports") if isinstance(draft.get("exports"), dict) else {}
                    try:
                        self.save(run, {**draft, "set": name, "exports": {**exports, name: str(target.real)},
                                        "version": draft["version"]+1})
                    except BaseException:
                        folder.undo()
                        raise
                    if runtime:             # what the model's process reported of itself (its device, a fallback)
                        run.meta.update(runtime)
                        run.save()
            self.folders.used(target.real, key)
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            if stem is not None:
                self.folders.release(target.real, stem)
            with self.lock:
                self.exporting.discard(state.meta["id"])
                self.export_progress.pop(state.meta["id"], None)
        return self.status(run)

    def model_export(self, run, state, manifest, staging, target, exclusions) -> dict:
        """Export with the run's model in its own process (the sopa environment, on the GPU), from `state` (snapshot):
        {stain_runtime, stain_fallback} as it reported them, for export() to keep under the run's lock. Each patch it
        writes is said ("progress": status's "exporting"); killed after EXPORT_FIRST_S without its first patch, then
        EXPORT_PATCH_S without the next, or by an engine that closes (Runs.close kills run.process)."""
        request = staging / "request.json"
        atomic_text(request, json.dumps(dict(root=str(state.dir), meta=state.meta, manifest=manifest,
                                             staging=str(staging), target=str(target), exclusions=exclusions)))
        launcher = children.helper_prefix("export")      # ends with the engine: it writes only the patch set
        command = [*launcher, self.stain.python, "-B", "-m", self.stain.patch_export_module, "--export", str(request)]
        env = self.stain.helper_env(state.dir / "stain/cache", *self.stain.run_pythons(state.meta.get("stain")))
        error, done, runtime = None, False, {}
        with (state.dir / "stain-patches.log").open("a") as log:
            process = subprocess.Popen(command, env=env, cwd=state.dir, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=log, text=True, **portable.new_session())
            portable.tie(process)                        # macOS and Windows: setpriv's death signal there
            run.process = process
            def stop():
                try:
                    portable.signal_group(process.pid, portable.SIGKILL)
                except ProcessLookupError:
                    pass
            def watch(seconds):
                timer = threading.Timer(seconds, stop)
                timer.daemon = True
                timer.start()
                return timer
            timer = watch(EXPORT_FIRST_S)
            if self.runs.closing:
                stop()
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    if event["ev"] == "hello":
                        runtime["stain_runtime"] = event
                    elif event["ev"] == "progress":            # a patch written: the next has its own time
                        with self.lock:
                            self.export_progress[state.meta["id"]] = {"done": int(event["done"]),
                                                                      "total": int(event["total"])}
                        timer.cancel()
                        timer = watch(EXPORT_PATCH_S)
                    elif event["ev"] == "fallback":
                        runtime["stain_fallback"] = event
                    elif event["ev"] == "error":
                        error = event["message"]
                    elif event["ev"] == "done":
                        done = True
                if process.wait() or not done:
                    raise RunError(error or "The patches could not be exported. Details are in the Log.", 503)
            finally:
                timer.cancel()
                if process.poll() is None:
                    stop()
                    process.wait()
                process.stdout.close()
                run.process = None
        request.unlink()
        return runtime

    def status(self, run):
        started = time.perf_counter()
        with run.lock:
            self.require(run)
            draft = self.draft(run)
            result = {**draft, "exported": None}
            state = self.snapshot(run) if draft["set"] else None
        with self.lock:                  # an export on its way: its patches written so far (the dialog says so)
            progress = self.export_progress.get(run.meta["id"])
            result["exporting"] = dict(progress) if progress is not None else None
        if draft["set"]:
            source = self.selected(run, draft["set"], draft)
            if source.manifest is None:
                # Its folder or its records were moved or deleted: the panel says so, and Export patches still works.
                result["exported"] = {"set": draft["set"], "folder": str(source.folder), "missing": True,
                                      "patches": [], "states": [], "gui_running": False, "corrected": 0, "cells": 0}
                result["poll_ms"] = (time.perf_counter()-started)*1000
                return result
            folder, manifest = source.folder, source.manifest
            signature = []
            for patch in manifest["patches"]:
                for key in ("image", "mask"):
                    path = folder / patch[key]
                    try:
                        st = path.stat()
                        signature.append((st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
                    except OSError:
                        signature.append(None)
            cache = (run.meta["id"], draft["set"])
            with self.lock:
                cached = self.cache.get(cache)
            if cached is None or cached[0] != signature or cached[2] < time.monotonic():
                # Neither lock is held here: an inspection reads every patch's mask, which can take long.
                states = self.stain.worker.call(state, "inspect", folder=str(folder), manifest=manifest)
                # A GUI save may be observed part-way through: retry failures even if stat stays unchanged.
                until = time.monotonic()+2 if any(p["state"] == "missing" for p in states) else float("inf")
                with self.lock:
                    self.cache[cache] = (signature, states, until)
            else:
                states = cached[1]
            running, failed = self.gui_state(source)
            result["exported"] = {**manifest, "folder": str(folder), "states": states,
                                  "gui_running": running, "gui_error": failed,
                                  "corrected": sum(p["state"] == "corrected" for p in states),
                                  "cells": sum(p["cells"] or 0 for p in states if p["state"] == "corrected")}
        result["poll_ms"] = (time.perf_counter()-started)*1000
        return result

    @staticmethod
    def gui_record(source: Source):
        """The detached GUI's record in the set's records ({pid, started}), or None."""
        try:
            record = read_small(source.meta / "gui.json")
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def running(self, source: Source, record) -> bool:
        """Under the manager's lock: whether a GUI is open on the set's folder, this engine's own or the one its record
        names (still that process, by its start time); one GUI per folder, whichever of its sets opened it. Once this
        engine's own has closed, the record may name another engine's, opened on the folder since."""
        process = self.processes.get(str(source.folder))
        if process is not None and process.poll() is None:
            return True
        try:
            process = psutil.Process(record["pid"])
            return process.create_time() == record["started"] and process.status() != psutil.STATUS_ZOMBIE
        except (TypeError, KeyError, ValueError, psutil.Error):
            return False

    def gui_running(self, source: Source) -> bool:
        record = self.gui_record(source)            # read before the manager's lock: the folder may be slow
        with self.lock:
            return self.running(source, record)

    def display(self):
        # The app's own Chrome window when there is one (browser = "chrome"), which may have been opened after a
        # headless engine started; else a page open in any browser, on the desktop this engine was started from (the
        # launcher's, or a notebook kernel's that has a display). The engine never reads which browser shows the page.
        if os.environ.get("PLH_TAG") or os.environ.get("PLH_DISPLAY"):
            return {}
        env = None
        try:
            if self.window and self.window():
                env = self.window.found.environ()
                if not env.get("DISPLAY"):
                    # Chrome rewrites its process title over its environment block, so /proc often shows no DISPLAY.
                    # The window is open, so the engine's own desktop (the launcher's) is the one the user looks at.
                    env = os.environ
        except Exception:
            env = None
        try:
            if env is None and self.page_here and self.page_here():
                env = os.environ
        except Exception:
            env = None
        if env is None:
            return {}
        display = env.get("DISPLAY", "")
        number = re.fullmatch(r"(?:[^:]*):(\d+)(?:\.\d+)?", display)
        if not number or 240 <= int(number[1]) <= 249 or env.get("PLH_DISPLAY") or env.get("PLH_TAG"):
            return {}
        return {k: env[k] for k in ("DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR") if env.get(k)}

    def gui_desktop(self) -> dict | None:
        """The desktop the Cellpose GUI opens on: on Linux the X11 display's variables (display()); on macOS and
        Windows, which need none, {} while a page of this engine is open on a desktop session of this computer's own
        (server/app.py page_here: not over SSH); None when there is none (the page shows the command instead)."""
        if sys.platform.startswith("linux"):
            return self.display() or None
        if os.environ.get("PLH_TAG") or os.environ.get("PLH_DISPLAY"):
            return None
        try:
            return {} if self.page_here and self.page_here() else None
        except Exception:
            return None

    def open_gui(self, run, body):
        """The Cellpose GUI on the set's folder, detached: started in its own cache folder with PYTHONSAFEPATH, so
        nothing in the patch folder is imported or read as a cache; its log and record in the set's records, opened
        without following a link. Without a desktop, the command to start it by hand."""
        with run.lock:
            self.require(run)
            draft = self.draft(run)
        source = self.selected(run, body.get("set"), draft)
        if source.manifest is None:
            raise RunError(words("gone", folder=source.folder), 404)
        patches = source.manifest["patches"]
        first = source.folder / patches[0]["image"] if patches else None
        try:
            if first is None or not stat.S_ISREG(os.lstat(first).st_mode):
                raise RunError("The first patch image is missing")
        except OSError:
            raise RunError("The first patch image is missing") from None
        command = [self.stain.python, "-m", "sopa_companion.zarr_bridge.stain_gui", str(first)]
        gpu = children.gpu(self.stain.gpu) if type(self.stain.gpu) is int else str(self.stain.gpu)
        # The app on PYTHONPATH only where the GUI's interpreter needs it (an installed copy's own environment has it).
        entry = children.app_entry(children.kind(self.stain.python))
        shell = ["env", *([f"PYTHONPATH={entry}"] if entry else []), "PYTHONSAFEPATH=1",
                 f"CUDA_VISIBLE_DEVICES={gpu}", "CUDA_DEVICE_ORDER=PCI_BUS_ID", *command]
        fallback = {"folder": str(source.folder), "command": shlex.join(shell), "running": False}
        # Only a private, server-side test configuration may substitute a fake launcher.
        fake = self.launcher == "sopa_companion.tests.fake_stain_gui"
        desktop = self.gui_desktop()
        if desktop is not None and not fake and not self.gui_running(source):
            # Without its own packages the GUI would close at once and the panel would say nothing: its interpreter is
            # asked first, outside the manager's lock (it takes a moment). The panel keeps the command to copy.
            lacking = self.gui_lacking()
            if lacking:
                install = shlex.join([str(self.stain.python), "-m", "pip", "install", "cellpose[gui]"])
                raise RunError(words("gui_missing", python=self.stain.python, command=install), 503, command=install)
        record = self.gui_record(source)            # read after the question: another engine may have opened one since
        with self.lock:
            if self.running(source, record):
                return {**fallback, "running": True}
            if desktop is None and not fake:
                return fallback
            cache = self.gui_cache()
            env = self.stain.helper_env(cache)
            env.update(desktop or {})            # (a test's fake GUI has no desktop)
            env["PYTHONSAFEPATH"] = "1"          # its working folder is never on its import path
            env["PYTHONUNBUFFERED"] = "1"        # its output and its errors reach its log in the order they happen
            command[2] = self.launcher
            meta = self.records(source)
            try:
                log = self.open_log(meta)
                try:
                    start = os.fstat(log).st_size        # this launch's part of the log begins here (gui_reason)
                    process = subprocess.Popen(command, cwd=cache, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=log, close_fds=True, **portable.new_session())
                finally:
                    os.close(log)
                self.processes[str(source.folder)] = process
                self.log_starts[str(source.folder)] = start
                try:
                    started = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    started = None
                if started is not None:
                    self.write_record(meta, {"pid": process.pid, "started": started})
            except OSError as exc:
                raise RunError(f"The Cellpose GUI could not be started: {exc.strerror or exc}", 503) from None
            finally:
                os.close(meta)
            return {**fallback, "running": True}

    @staticmethod
    def records(source: Source) -> int:
        """A handle to the set's records folder, opened without following a link (in a chosen folder: from the folder
        itself down); 404 with the words for a set no longer there when it cannot be opened so."""
        try:
            if not source.flat:
                return os.open(source.meta, OPEN_DIR)
            folder = os.open(source.folder, OPEN_DIR)
            try:
                return os.open(META, OPEN_DIR, dir_fd=folder)
            finally:
                os.close(folder)
        except OSError:
            raise RunError(words("gone", folder=source.folder), 404) from None

    @staticmethod
    def open_log(meta: int) -> int:
        """The GUI's log in the records (`meta`, a handle), opened for appending: a plain file only, never written
        through a link. It is opened without waiting, so a FIFO put there is refused at once: with no reader, a plain
        open would wait for one while the manager's lock is held. OSError otherwise, in words the panel can show."""
        refusal = "its log, cellpose.log, is not a plain file"
        try:
            fd = os.open("cellpose.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
                         | os.O_CLOEXEC, 0o666, dir_fd=meta)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENXIO):      # a link; a FIFO with no reader, a socket or a device
                raise OSError(exc.errno, refusal) from None
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):        # a FIFO someone reads, or a device
                raise OSError(errno.EINVAL, refusal)
            os.set_blocking(fd, True)
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def write_record(meta: int, record: dict) -> None:
        """gui.json in the records (`meta`, a handle), through a new hidden file of its own and a rename: a file or a
        link of that name is replaced, never written through."""
        temporary = f".gui.json.{secrets.token_hex(8)}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666, dir_fd=meta)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream)
            os.replace(temporary, "gui.json", src_dir_fd=meta, dst_dir_fd=meta)
        except OSError:
            try:
                os.unlink(temporary, dir_fd=meta)
            except OSError:
                pass

    def forget(self, run):
        # Exported sets outlive the run: the detached GUI may still be using them.
        with self.lock:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != run.meta["id"]}
