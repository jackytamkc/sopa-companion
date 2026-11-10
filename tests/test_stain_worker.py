"""The patch helper and the runs' locks: the long-lived helper outlives the request thread that started it, no patch
call holds the run's lock while the helper works, and stain segmentation names no interpreter of its own. Helper
doubles only: no sopa environment, no Cellpose."""
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import wait_for
from test_stain import SETTINGS, Images, ended
from sopa_companion.engine.runs import RunError, Runs
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.stain_worker import StainWorker

ECHO = """#!{python}
import json, os, sys
for line in sys.stdin:                       # the patch helper's protocol: one request, one answer, until stdin closes
    print(json.dumps({{"ok": True, "result": {{"pid": os.getpid(), "action": json.loads(line)["action"]}}}}), flush=True)
"""


def test_the_patch_helper_outlives_the_request_thread_that_started_it(tmp_path):
    """The helper was started under setpriv --pdeathsig from whichever request thread called first; the server retires
    idle request threads, and the signal follows the thread, so the helper died under the next calls (an export failed
    with "The patches could not be read"). It now lives until close(), or until its stdin closes."""
    helper = tmp_path / "helper"
    helper.write_text(ECHO.format(python=sys.executable))
    helper.chmod(0o755)
    worker = StainWorker(str(helper))
    run = SimpleNamespace(dir=tmp_path, meta={})
    first = []
    thread = threading.Thread(target=lambda: first.append(worker.call(run, "inspect")))
    thread.start()
    thread.join()                                           # the thread that started it is gone
    time.sleep(0.5)
    assert worker.process.poll() is None, "the helper died with the thread that started it"
    assert worker.call(run, "export") == {"pid": first[0]["pid"], "action": "export"}     # the same helper answers
    process = worker.process
    worker.close()
    assert worker.process is None and process.wait(timeout=5) == 0           # ended by its stdin closing


@pytest.fixture
def stain_run(config, tmp_path):
    config["stain"].update(python=sys.executable, helper_module="sopa_companion.tests.fake_stain", tile_pixels=128,
                           overlap_pixels=16)
    runs = Runs(config, Samples(config), Images(tmp_path / "unwritten.zarr"))
    run = runs.get(runs.create("image-sample", kind="stain", region={"kind": "full"}, stain=SETTINGS)["id"])
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: ended(run))
    yield runs, run
    runs.close(save=False)


def answers_soon(fn, within=1.0) -> bool:
    """Whether fn() returns within `within` seconds (asked from another thread: a lock held elsewhere never hangs the
    test)."""
    done = threading.Event()
    threading.Thread(target=lambda: (fn(), done.set()), daemon=True).start()
    return done.wait(within)


def test_no_patch_call_holds_the_runs_lock_while_the_helper_works(stain_run, monkeypatch, chosen):
    """A proposal and an export held the run's lock across the helper call (an export with model outlines
    takes minutes), and every list of the runs (the page's pushes, GET /runs) waited for it. Now the runs list at once
    meanwhile; an answer to boxes that changed meanwhile is refused (409) and its export left nothing behind; and one
    export of a run at a time."""
    runs, run = stain_run
    patches = runs.stain.patches
    working, release, sent = threading.Event(), threading.Event(), []
    folder = chosen / "patches"

    def call(state, action, **values):
        if action == "inspect":
            return [{"id": p["id"], "state": "untouched", "cells": 1} for p in values["manifest"]["patches"]]
        sent.append(state)
        working.set()
        assert release.wait(10)
        if action == "export":
            manifest, staging, written = values["manifest"], Path(values["staging"]), []
            for patch in manifest["patches"]:              # the files and names the patch helper writes
                name = f"{manifest['stem']}-{patch['id']:02d}"
                (staging / f"{name}.tif").write_text("")
                (staging / f"{name}_seg.npy").write_text("")
                written.append({**patch, "image": f"{name}.tif", "mask": f"{name}_seg.npy", "export_hash": "0" * 64,
                                "cells": 0})
            (staging / "patches.json").write_text(json.dumps({**manifest, "patches": written}))
            return {}
        return {"patches": [{"id": 1, "box": [0, 0, 64, 64]}], "message": ""}
    monkeypatch.setattr(runs.stain.worker, "call", call)

    version = patches.status(run)["version"]
    proposed = []
    thread = threading.Thread(target=lambda: proposed.append(patches.change(run, {"version": version}, propose=True)))
    thread.start()
    assert working.wait(10)
    assert answers_soon(runs.list) and answers_soon(run.public)             # while the helper works
    release.set()
    thread.join(10)
    assert proposed[0]["version"] == version + 1 and proposed[0]["patches"] == [{"id": 1, "box": [0, 0, 64, 64]}]
    # A copy, taken under the lock: the run's meta with its exclusions' whole state (the areas the helper leaves out),
    # as a snapshot of the run takes it now.
    assert sent[0] is not run and sent[0].meta == patches.snapshot(run).meta

    working.clear()
    release.clear()
    exported = []

    def export():
        try:
            exported.append(patches.export(run, {"version": version + 1, "folder": str(folder)}))
        except RunError as exc:
            exported.append(exc)
    thread = threading.Thread(target=export)
    thread.start()
    assert working.wait(10)
    assert answers_soon(runs.list)
    with pytest.raises(RunError, match="being exported already"):
        patches.export(run, {"version": version + 1, "folder": str(folder)})
    with pytest.raises(RunError, match="have been exported"):                # nor is the run deleted under it
        runs.delete(run.meta["id"])
    patches.change(run, {"version": version + 1, "boxes": []})              # the boxes change meanwhile
    release.set()
    thread.join(10)
    assert isinstance(exported[0], RunError) and exported[0].status == 409
    # Its staging went, no part file is left in the folder, and no set was made.
    assert not list(patches.staging.iterdir()) and not list(folder.rglob("*.part"))
    assert not list((folder / ".sopa-companion" / "sets").glob("*.json"))
    assert patches.status(run)["set"] is None and patches.exporting == set()
    exported.clear()
    exported.append(patches.export(run, {"version": version + 2, "folder": str(folder)}))   # the next one works
    assert exported[0]["set"] and exported[0]["exported"]["states"] == []


def test_stain_segmentation_names_no_interpreter_of_its_own(config):
    """Without [stain] python the default was one computer's own interpreter path; unset, stain
    segmentation is unavailable, as the other engines treat an unset interpreter."""
    config["stain"].pop("python", None)
    runs = Runs(config, Samples(config))
    try:
        assert runs.stain.python == "" and runs.stain.available is False
        assert runs.stain.method_available("cellpose") is False
    finally:
        runs.close(save=False)
    assert "/home/" not in (Path(__file__).resolve().parents[1] / "engine" / "stain.py").read_text()


def test_the_patch_helpers_own_lines_are_in_the_stain_runs_log(stain_run):
    """A patch call that failed answered "The patches could not be read", and what the patch helper printed
    (its traceback) was in no Log: the stain run's Log carries it now."""
    from sopa_companion.engine import runlog
    runs, run = stain_run
    (run.dir / "stain-patches.log").write_text("Traceback (most recent call last):\nValueError: Test model failed\n")
    out = runlog.Text()
    runlog.run_part(out, SimpleNamespace(runs=runs), run.meta["id"])
    assert "Patch helper" in out.text() and "ValueError: Test model failed" in out.text()
