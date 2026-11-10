"""The mixing check through the API and the WebSocket, with the fake trainer (tests/fake_qc_trainer.py): learning a
sample (GET/POST /samples/{id}/qc, /health), a run's switch (on by default, remembered, handed to the analysis
scheduler), the start line (proseg's count at iteration 0, only for a start from cells alone) and each frame's
comparison with its own start block (the arrow, retention, the areas' tags), the headline over live and single frames,
per-cell bytes, the series, the previous run, trials, the page's events and an engine restart, the count and the
constants every summary names, the earlier proseg cells only from a table of proseg's own count, with the GPU worker's
results handed over as the analysis scheduler hands them (engine/analysis.py `_deliver`); and once end to end through
the real scheduler and the fake worker process, which runs the worker's own numpy code for ops qc and qc_start."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from sopa_companion.analysis.counts_file import BASIS
from sopa_companion.engine import qc_constants as K
from sopa_companion.engine import qc_rules as R
from test_qc_models import FAKE, FAKE_PROBE, IDLE, Registry, write_table

BASE = "http://127.0.0.1:8799"
NCELLS = 50


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


CELLS_ONLY = {"cells": {"kind": "shapes", "key": "xenium_cell_boundaries"}, "nuclei": None}
PAIRED = {"cells": {"kind": "shapes", "key": "xenium_cell_boundaries"},
          "nuclei": {"kind": "shapes", "key": "xenium_nucleus_boundaries"}}


def write_sample(root: Path, proseg: bool = True) -> dict:
    """A sample with Xenium's cell table and an earlier proseg run's saved cells ("table"; `proseg` False: a table
    that is not proseg's own count, as a Xenium sample's own "table")."""
    path = root / "samples" / "sample-a-crop.zarr"
    write_table(path, "xenium_table")
    write_table(path, "table", proseg=proseg)
    from sopa_companion.engine.zarr_samples import sample_id
    return {"id": sample_id(str(path.resolve())), "name": "sample-a-crop", "path": str(path.resolve()), "kind": "zarr",
            "opened": "2026-10-17T12:00:00"}


def write_run(data: Path, run_id: str, sample: dict, created: str, frames: int = 8, prior=None, region=None,
              times=None, burnin: int | None = None, mode: str = "detail"):
    """A finished run on a zarr sample as Runs finds it on disk (no proseg needed). `times`: each frame's "t" (the
    run clock, s); without it the frames carry no time. `burnin`: the schedule's burn-in iterations (frame s is then
    iteration s, in burn-in before it); without it frame s is iteration 10 s, in burn-in for s < 3. `mode`: "tiled"
    for a whole-sample run (100 µm area bins). `prior`: how its cells started, Xenium's cells alone by default (a
    start line: qc_rules.start_of "cells"). Its proseg-live kept proseg's own counts (info.json's "counts": the engine
    reads no counts reason for it), though none of its frames is counted here."""
    folder = data / "runs" / run_id
    (folder / "live" / "frames").mkdir(parents=True)
    region = region or {"kind": "box", "box": [0, 0, 400, 300], "bbox": [0, 0, 400, 300]}
    prior = prior if prior is not None else CELLS_ONLY
    meta = {"id": run_id, "sample": sample["id"], "sample_name": sample["name"], "box": region["bbox"], "region": region,
            "mode": mode, "prior": prior, "sample_kind": "zarr", "sample_path": sample["path"], "schedule": "open",
            "schedule_params": {"label": "Until I stop", "indefinite": True}, "options": {}, "proseg_args": [],
            "threads": 2, "speed": "fast", "state": "done", "message": "", "created": created, "live_values": {},
            "outputs": [], "paused": False, "start_paused": False, "zarr_save": None}
    (folder / "run.json").write_text(json.dumps(meta))
    info = {"cells": NCELLS, "genes": 7, "prior": {"file": "prior.bin.zst", "nx": 10, "ny": 10, "nz": 4, "x0": 0, "y0": 0,
                                                   "voxel": 2.0}, "z": {"grid_origin": 0, "grid_spacing": 0.25},
            "counts": {"mode": "request", "format": "PLC1", "version": 1, "e_scale": 32, "start": True,
                       "start_every": 1, "spool": None, "where": 1},
            **({"schedule": {"burnin": burnin}} if burnin is not None else {})}
    (folder / "live" / "info.json").write_text(json.dumps(info))
    (folder / "live" / "transcripts.bin.gz").write_bytes(gzip.compress(b""))
    lines = [json.dumps({"seq": s, "iter": 10 * s if burnin is None else s,
                         "phase": "burnin" if s < (3 if burnin is None else burnin) else "sampling",
                         "file": f"frames/{s:06d}.bin.zst", "live": {"cell-compactness": 0.04 if s < 5 else 0.02},
                         **({"t": times[s]} if times is not None else {})})
             for s in range(frames)]
    (folder / "live" / "frames.jsonl").write_text("\n".join(lines) + "\n")
    return folder


class Scheduler:
    """The analysis scheduler's side of the provider contract (engine/analysis.py), recorded instead of run."""
    available = True

    def __init__(self):
        self.calls, self.changed, self.start, self.unavailable, self.requested = [], 0, "pending", None, None

    def configure_qc(self, run_id, enabled=None, seq=None):
        self.calls.append((run_id, enabled, seq))
        return {}

    def describe_qc(self, run_id):
        return {"start": self.start, "status": {"state": "idle"}, "backfill": 0, "unavailable": self.unavailable,
                "requested": self.requested}

    def qc_changed(self):
        self.changed += 1


@pytest.fixture
def world(config, tmp_path):
    config["proseg"]["keep_unsaved"] = True
    data = Path(config["paths"]["data_dir"])
    data.mkdir(parents=True, exist_ok=True)
    sample = write_sample(tmp_path)
    (data / "samples.json").write_text(json.dumps([sample]))
    created = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    write_run(data, "run-a", sample, created.isoformat())
    write_run(data, "run-b", sample, (created + timedelta(hours=1)).isoformat())
    write_run(data, "run-nuclei", sample, (created + timedelta(hours=2)).isoformat(),
              prior={"cells": None, "nuclei": {"kind": "shapes", "key": "xenium_nucleus_boundaries"}})
    write_run(data, "run-paired", sample, (created + timedelta(hours=3)).isoformat(), prior=PAIRED)
    from sopa_companion.engine.config import merge
    # These tests exercise the check's machinery with it switched on by default and learning on open; the shipped,
    # opt-in default (off, learned when switched on) is test_the_check_is_opt_in_by_default's.
    cfg = merge(config, {"qc": {"python": sys.executable, "trainer_module": FAKE, "probe_module": FAKE_PROBE,
                                "gpu_quiet_s": 0, "enabled_by_default": True, "learn": "open"},
                         "analysis": {"python": str(tmp_path / "no-rapids" / "python")}})   # no GPU worker here
    return cfg, sample


def make(cfg):
    from sopa_companion.server import create_app
    application = create_app(cfg, token="t" * 40)
    application.state.qc_models.probe = lambda: dict(IDLE)        # GPU 0 idle (no nvidia-smi in the tests)
    application.state.qc.analysis = Scheduler()
    return application


@pytest.fixture
def app(world):
    cfg, sample = world
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        yield application, client, sample


def deliver(qc, run_id: str, seq: int, judged: int, mixed: int, single=False, median_tx: float = 240.0,
            start_block=None):
    """A frame's result as the scheduler hands it over: the worker has written <seq>.qc.bin (one byte per cell); its
    summary says it is on proseg's averaged count and carries the frame's own start block when given."""
    import zstandard
    state = qc._state(run_id)
    state.dir.mkdir(parents=True, exist_ok=True)
    values = np.full(NCELLS, 255, np.uint8)
    values[:judged] = 20
    values[:mixed] = 200
    (state.dir / f"{seq:06d}.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes()))
    frame = qc.runs.get(run_id).frames[seq]
    qc.qc_result({"kind": "frame", "run": run_id, "seq": seq, "model": state.key, "dir": str(state.dir),
                  "source": "request" if single else "live", "single": single,
                  "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                  "summary": {"kind": "frame", "seq": seq, "judged": judged, "mixed": mixed, "ncells": NCELLS,
                              "median_tx": median_tx, "ms": {"score": 1.5}, "basis": "proseg", "count": "averaged",
                              **({"start_block": start_block} if start_block is not None else {})},
                  "bins": None, "cells": str(state.dir / f"{seq:06d}.qc.bin"), "start": None})
    return values


def deliver_start(qc, run_id: str, judged: int, mixed: int):
    import zstandard
    state = qc._state(run_id)
    state.dir.mkdir(parents=True, exist_ok=True)
    values = np.full(NCELLS, 255, np.uint8)
    values[:judged] = 30
    values[:mixed] = 180
    (state.dir / "start.bin").write_bytes(zstandard.ZstdCompressor().compress(values.tobytes() + b"later values"))
    qc.qc_result({"kind": "start", "run": run_id, "seq": None, "model": state.key, "dir": str(state.dir),
                  "source": "start", "single": True, "summary": {"kind": "start", "judged": judged, "mixed": mixed,
                                                                 "median_tx": 386.0, "ncells": NCELLS},
                  "bins": None, "cells": str(state.dir / "start.bin"), "start": str(state.dir / "start.npz")})
    return values


def ready_run(client, run_id="run-a"):
    client.get(f"/api/v1/runs/{run_id}/qc", headers=auth())          # the switch is on: its sample is learned
    wait_for(lambda: client.get(f"/api/v1/runs/{run_id}/qc", headers=auth()).json()["state"] in ("starting", "ready"),
             timeout=30)


# ---- samples --------------------------------------------------------------------------------------------------------
def test_health_and_learning_a_sample(app):
    application, client, sample = app
    health = client.get("/api/v1/health", headers=auth()).json()
    # No analysis worker here: Clusters & UMAP cannot run, so nothing is scored, whatever the check's probe answers
    # (its fake: ready, at once or a moment later).
    assert health["qc"]["available"] is False and health["qc"]["reason"] == "analysis"
    assert health["qc"]["state"] in ("checking", "ready")
    # "New run with this setting" says why it cannot start a run against the engine's own limit ([proseg]).
    assert health["max_active_runs"] == application.state.runs.max_active >= 1
    url = f"/api/v1/samples/{sample['id']}/qc"
    view = client.get(url, headers=auth()).json()
    assert view["state"] == "off" and view["model"] is None and view["estimate_s"] > 0 and view["table"] == "xenium_table"
    assert client.get(url, headers={"Authorization": "Bearer " + "t" * 40}).status_code == 200      # GET: no Origin
    assert client.post(url, json={"learn": True}, headers={"Authorization": "Bearer " + "t" * 40}).status_code == 403
    assert client.post(url, json={"learn": "yes"}, headers=auth()).status_code == 422
    assert client.get("/api/v1/samples/z000000000000/qc", headers=auth()).status_code == 404
    assert client.post("/api/v1/samples/z000000000000/qc", json={"learn": True}, headers=auth()).status_code == 404
    assert client.post(url, json={"learn": True}, headers=auth()).json()["state"] in ("waiting", "learning")
    wait_for(lambda: client.get(url, headers=auth()).json()["state"] == "ready", timeout=30)
    view = client.get(url, headers=auth()).json()
    assert view["model"]["cells"] == 50 and view["model"]["table"] == "xenium_table" and len(view["model"]["id"]) == 32
    assert "notebook_start_per100" not in view and view["progress"] is None
    assert application.state.qc.analysis.changed == 1                        # the scheduler was told at once


def test_opening_a_sample_with_the_check_on_learns_it(app, monkeypatch):
    application, client, sample = app
    qc = application.state.qc
    asked = []
    monkeypatch.setattr(application.state.qc_models, "ensure", lambda sid, retry=False: asked.append(sid))
    qc.sample_opened({"id": sample["id"], "kind": "zarr"})
    assert asked == [sample["id"]]
    qc.prefs = {"enabled": False}                                     # the check was switched off last time
    qc.sample_opened({"id": sample["id"], "kind": "zarr"})
    assert asked == [sample["id"]]


# ---- API mode: shared models, the engine's own switch ---------------------------------------------------------------
def test_shared_models_serve_the_next_session_and_the_switch_stays_each_engines_own(world, tmp_path):
    """[qc] models_dir (a notebook's model_cache): models are learned into it and the next session with a fresh data dir
    finds them ready; the switch (prefs.json) stays in each engine's own <data_dir>/qc-models, so a new session starts
    at enabled_by_default, never at another session's last choice."""
    from sopa_companion.engine import qc as Q
    from sopa_companion.engine.config import merge
    cfg, sample = world
    shared = tmp_path / "model-cache"
    first = merge(cfg, {"qc": {"models_dir": str(shared)}})
    application = make(first)
    url = f"/api/v1/samples/{sample['id']}/qc"
    with TestClient(application, base_url=BASE) as client:
        assert application.state.qc_models.root == shared
        client.post(url, json={"learn": True}, headers=auth())
        wait_for(lambda: client.get(url, headers=auth()).json()["state"] == "ready", timeout=30)
        model = client.get(url, headers=auth()).json()["model"]["id"]
        assert (shared / sample["id"] / model / "model.json").is_file()
        assert client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth()).status_code == 200
    own = Path(cfg["paths"]["data_dir"]) / "qc-models"
    assert json.loads((own / "prefs.json").read_text()) == {"enabled": False}
    assert not (shared / "prefs.json").exists() and not (own / sample["id"]).exists()
    second = merge(first, {"paths": {"data_dir": str(tmp_path / "next" / "data"),
                                     "state_dir": str(tmp_path / "next" / "state")}})
    models = Q.QcModels(second, Registry(sample), probe=lambda: dict(IDLE))
    try:
        assert models.describe(sample["id"])["state"] == "ready" and models.model(sample["id"])["id"] == model
        assert Q.QcManager(second, None, models).new_runs() is True        # enabled_by_default, not the last choice
    finally:
        models.close()


def test_a_view_linking_a_zarrs_table_shares_that_zarrs_model(tmp_path, monkeypatch):
    """A model key names the zarr that really holds the learned table (qc.table_home): an API session's view, whose
    tables are links into the caller's zarr, gets the key and the model folder the zarr opened in place gets, so either
    finds the other's model (one fit: about two minutes on GPU 0). A desktop sample, with no
    links, keeps its key."""
    import hashlib
    from sopa_companion.analysis import qc_trainer as T
    from sopa_companion.engine import qc as Q
    from sopa_companion.engine.zarr_samples import sample_id
    original = (tmp_path / "samples" / "a.zarr").resolve()
    for key in ("xenium_table", "table"):
        write_table(original, key)
    view = tmp_path / "session" / "view.zarr"
    (view / "tables").mkdir(parents=True)
    for folder in (view, view / "tables"):
        (folder / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    for key in ("xenium_table", "table"):
        (view / "tables" / key).symlink_to(original / "tables" / key, target_is_directory=True)
    assert Q.table_home(view, "xenium_table") == original and Q.table_home(original, "xenium_table") == original
    recipe, versions = T.recipe(), {"scviva-tools": "0.1.7", "scvi-tools": "1.5.1", "torch": "2.13.0"}
    info = Q.table_info(original, "xenium_table")
    others = {"table": Q.QcModels._fingerprint(Q.table_info(original, "table"))}
    key = Q.model_key(original, info, recipe, versions, others)
    assert Q.model_key(view, Q.table_info(view, "xenium_table"), recipe, versions, others) == key
    # The key as the desktop has always made it.
    legacy = {"zarr": str(original), "table": "xenium_table", "fingerprint": Q.QcModels._fingerprint(info),
              "others": others, "recipe": recipe, "versions": versions}
    assert key == hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:32]
    # Learned through the view into a shared models_dir: the folder is the zarr's, and the zarr in place finds it.
    entries = [{"id": sample_id(str(view)), "name": "view", "path": str(view)},
               {"id": sample_id(str(original)), "name": "a", "path": str(original)}]
    log = tmp_path / "launches.jsonl"
    monkeypatch.setenv("FAKE_QC_LOG", str(log))
    monkeypatch.delenv("FAKE_QC_MODE", raising=False)

    def config(name):
        return {"paths": {"data_dir": str(tmp_path / name / "data"), "state_dir": str(tmp_path / name / "state")},
                "qc": {"python": sys.executable, "trainer_module": FAKE, "probe_module": FAKE_PROBE,
                       "models_dir": str(tmp_path / "models")}}
    through_view = Q.QcModels(config("view-session"), Registry(*entries), probe=lambda: dict(IDLE))
    try:
        through_view.ensure(entries[0]["id"])
        wait_for(lambda: through_view.describe(entries[0]["id"])["state"] == "ready", timeout=30)
        folder = Path(through_view.model(entries[0]["id"])["dir"])
        assert folder.parent == tmp_path / "models" / entries[1]["id"]
    finally:
        through_view.close()
    in_place = Q.QcModels(config("in-place-session"), Registry(*entries), probe=lambda: dict(IDLE))
    try:
        assert in_place.describe(entries[1]["id"])["state"] == "ready"
        assert in_place.model(entries[1]["id"])["dir"] == str(folder)
        assert len(log.read_text().splitlines()) == 1                         # one fit for both
    finally:
        in_place.close()


def test_a_zarrs_models_and_refused_tables_stay_in_its_own_folder_whatever_a_session_calls_it(tmp_path, monkeypatch):
    """The models folder is the sample id of the zarr that holds the tables (qc.table_home), never the id a session
    gave its sample: a session that names the zarr otherwise finds the model learned through a view. A table the
    trainer refused through the view is remembered there too (tables.json), so the other session never asks again
    and the view leaves no folder of its own."""
    from sopa_companion.engine import qc as Q
    from sopa_companion.engine.zarr_samples import sample_id
    original = (tmp_path / "samples" / "a.zarr").resolve()
    for key in ("xenium_table", "table"):
        write_table(original, key)
    view = tmp_path / "session" / "view.zarr"
    (view / "tables").mkdir(parents=True)
    for folder in (view, view / "tables"):
        (folder / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    for key in ("xenium_table", "table"):
        (view / "tables" / key).symlink_to(original / "tables" / key, target_is_directory=True)
    log = tmp_path / "launches.jsonl"
    monkeypatch.setenv("FAKE_QC_LOG", str(log))
    monkeypatch.delenv("FAKE_QC_MODE", raising=False)
    monkeypatch.setenv("FAKE_QC_NOT_COUNTS", "xenium_table")
    models, home = tmp_path / "models", tmp_path / "models" / sample_id(str(original))

    def session(name, entry):
        config = {"paths": {"data_dir": str(tmp_path / name / "data"), "state_dir": str(tmp_path / name / "state")},
                  "qc": {"python": sys.executable, "trainer_module": FAKE, "probe_module": FAKE_PROBE,
                         "models_dir": str(models)}}
        return Q.QcModels(config, Registry(entry), probe=lambda: dict(IDLE))
    through_view = session("view-session", {"id": "view-session-sample", "name": "view", "path": str(view)})
    try:
        through_view.ensure("view-session-sample")
        wait_for(lambda: through_view.describe("view-session-sample")["state"] == "ready", timeout=30)
        assert through_view.describe("view-session-sample")["table"] == "table"
        assert Path(through_view.model("view-session-sample")["dir"]).parent == home
    finally:
        through_view.close()
    assert json.loads((home / "tables.json").read_text())["xenium_table"]["reason"] == "not_counts"
    assert [p.name for p in models.iterdir() if p.name != ".fit.lock"] == [home.name]    # none under the view's id
    other = session("other-session", {"id": "another-name", "name": "a", "path": str(original)})
    try:
        assert other.describe("another-name")["state"] == "ready"
        assert Path(other.model("another-name")["dir"]).parent == home
        assert [json.loads(line)["job"]["table"] for line in log.read_text().splitlines()] == ["xenium_table", "table"]
    finally:
        other.close()


# ---- a run ----------------------------------------------------------------------------------------------------------
def test_the_check_is_opt_in_by_default(world, monkeypatch):
    """The shipped default (config.toml): a new run's check is off and nothing is learned until a user switches it on;
    switching it on learns the run's sample."""
    from sopa_companion.engine.config import merge
    cfg, sample = world
    application = make(merge(cfg, {"qc": {"enabled_by_default": False, "learn": "start"}}))
    with TestClient(application, base_url=BASE) as client:
        asked = []
        monkeypatch.setattr(application.state.qc_models, "ensure", lambda sid, retry=False: asked.append(sid))
        new_runs = lambda: client.get(f"/api/v1/samples/{sample['id']}/qc", headers=auth()).json()["new_runs"]
        assert new_runs() is False                                          # the picker says nothing of the check
        application.state.qc.sample_opened({"id": sample["id"], "kind": "zarr"})
        assert asked == []                                                  # opening a sample learns nothing
        assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["enabled"] is False
        assert asked == []                                                  # a run starts with the check off
        assert client.post("/api/v1/runs/run-a/qc", json={"enabled": True}, headers=auth()).status_code == 200
        assert set(asked) == {sample["id"]}                                 # switching it on learns the sample (ensure is idempotent)
        assert new_runs() is True                                           # remembered: new runs start with it on
        assert client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth()).status_code == 200
        assert new_runs() is False


def test_a_run_saved_while_the_check_was_on_by_default_opens_with_it_off(world, monkeypatch):
    """Runs saved while the check was on by default kept qc/settings.json {"enabled": true, "n": 0} (no "v"): nobody chose
    that, so opening such a run follows today's default (off): nothing is learned or scored, and the file is brought up
    to date once. A switch someone changed back then keeps their choice; a current file is read as it is."""
    from sopa_companion.engine.config import merge
    cfg, sample = world
    application = make(merge(cfg, {"qc": {"enabled_by_default": False, "learn": "start"}}))
    runs = Path(cfg["paths"]["data_dir"]) / "runs"
    settings = lambda run_id: runs / run_id / "qc" / "settings.json"
    for run_id, stored in (("run-a", {"enabled": True, "n": 0}), ("run-b", {"enabled": True, "n": 2}),
                           ("run-nuclei", {"v": 2, "enabled": True, "n": 0})):
        settings(run_id).parent.mkdir(parents=True, exist_ok=True)
        settings(run_id).write_text(json.dumps(stored))
    with TestClient(application, base_url=BASE) as client:
        asked = []
        monkeypatch.setattr(application.state.qc_models, "ensure", lambda sid, retry=False: asked.append(sid))
        scheduler = application.state.qc.analysis
        view = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()
        assert view["enabled"] is False and view["state"] == "off" and view["switch_n"] == 0
        assert asked == [] and ("run-a", True, None) not in scheduler.calls     # nothing learned, nothing scored
        assert json.loads(settings("run-a").read_text()) == {"v": 2, "enabled": False, "n": 0}
        assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["enabled"] is False
        assert client.get("/api/v1/runs/run-b/qc", headers=auth()).json()["enabled"] is True   # a choice made back then
        assert json.loads(settings("run-b").read_text()) == {"v": 2, "enabled": True, "n": 2}
        assert client.get("/api/v1/runs/run-nuclei/qc", headers=auth()).json()["enabled"] is True   # current: as stored
        assert set(asked) == {sample["id"]}                                     # only the runs switched on learn it
    from sopa_companion.engine.qc import stored_switch
    assert stored_switch({"enabled": True}, False) == (False, 0, False)
    assert stored_switch({"enabled": True, "n": True}, True) == (True, 0, False)     # a bool is no count
    assert stored_switch({"v": 2, "enabled": False, "n": 3}, True) == (False, 3, True)
    with pytest.raises(KeyError):
        stored_switch({"n": 1}, False)


def test_a_runs_check_from_learning_to_results(app):
    application, client, sample = app
    qc, scheduler = application.state.qc, application.state.qc.analysis
    url = "/api/v1/runs/run-a/qc"
    first = client.get(url, headers=auth()).json()
    assert first["enabled"] is True and first["state"] in ("waiting", "learning", "starting")
    assert first["thresholds"] == {"mixed_at": 0.3, "judge_min": 100} and first["start_source"] == "xenium"
    assert first["start_kind"] == "cells"                                  # Xenium's cells alone: a start line
    assert scheduler.calls == [("run-a", True, None)]                     # the scheduler's switch follows ours
    ready_run(client)
    view = client.get(url, headers=auth()).json()
    assert view["state"] == "starting" and view["start"] is None and view["seqs"] == []
    assert view["scoring"] == {"state": "idle"} and view["backfill"] == 0
    model = view["model"]["id"]
    folder = Path(application.state.qc_models.root) / sample["id"] / model
    assert qc.qc_model(application.state.runs.get("run-a")) == {"id": model, "path": str(folder),
                                                                 "deep": str(folder / "deep.npz")}
    assert client.get(url + "/start", headers=auth()).status_code == 404

    # The start line: Xenium's cells, before anything moved (start.json is written here).
    start_values = deliver_start(qc, "run-a", judged=40, mixed=4)
    folder = application.state.runs.get("run-a").dir / "qc" / f"{model[:8]}.{BASIS}"     # a folder per count basis
    assert folder.name.endswith(".pc1") and qc._state("run-a").dir == folder
    assert json.loads((folder / "start.json").read_text())["per100"] == 10.0
    assert client.get(url, headers=auth()).json()["state"] == "ready"
    start = client.get(url + "/start", headers=auth()).json()
    assert start["per100"] == 10.0 and start["judged"] == 40 and start["median_tx"] == 386.0 and start["cells"] == NCELLS
    assert start["start_source"] == "xenium" and start["model"] == model
    cells = client.get(url + "/start/cells", headers=auth({"Accept-Encoding": "zstd"}))
    assert cells.content == start_values.tobytes()                                 # the per-cell values only

    # Live frames: smoothed (a region's weight on the newest, 0.2: measured on proseg's count), the shown integer with
    # hysteresis; <seq>.qc.json written.
    for seq, judged, mixed in ((3, 40, 4), (4, 40, 8), (5, 40, 8)):
        deliver(qc, "run-a", seq, judged, mixed)
    summaries = [client.get(f"{url}/{s}", headers=auth()).json() for s in (3, 4, 5)]
    assert [s["per100"] for s in summaries] == [10.0, 12.0, 13.6]
    assert [s["raw_per100"] for s in summaries] == [10.0, 20.0, 20.0]
    assert [s["shown"] for s in summaries] == [10, 12, 14] and not any(s["single"] for s in summaries)
    last = summaries[-1]
    assert last["iter"] == 50 and last["phase"] == "sampling" and last["model"] == model
    assert last["start"] == {"per100": 10.0, "judged": 40, "mixed": 4, "median_tx": 386.0}
    assert last["median_tx"] == 240.0 and last["areas"] == [] and last["suggestion"] is None
    assert last["batch_per100"] is not None and len(json.dumps(last)) < 8192
    assert (last["basis"], last["count"]) == ("proseg", "averaged")       # proseg's own count, as the worker says
    assert last["constants"] == {"mode": "region", "id": "proseg-2026-10-31", "measured": True, "kept": []}
    assert json.loads((folder / "000005.qc.json").read_text()) == last          # the scheduler's "done" marker
    view = client.get(url, headers=auth()).json()
    assert view["latest_seq"] == 5 and view["seqs"] == [3, 4, 5] and view["batch_per100"] == last["batch_per100"]
    assert view["batch_table"] == "table"                                # the earlier proseg cells' table
    assert qc.qc_result({"kind": "frame", "run": "run-a", "seq": 6, "model": "another-model", "summary": {"judged": 9}}) \
        is None and 6 not in client.get(url, headers=auth()).json()["seqs"]    # another yardstick: not kept

    # Per-cell bytes: zstd as written, or inflated; the model id in the URL must be this run's. Only a URL that names
    # the model and the count may be cached for good: without the model the same URL gives another model's bytes after
    # a relearn, without the count those of the frame counted again (a pause's, the hold's, a finish's or the end's
    # count).
    values = deliver(qc, "run-a", 6, 30, 3)
    packed = client.get(f"{url}/6/cells?model={model}&count=averaged", headers=auth({"Accept-Encoding": "zstd"}))
    plain = client.get(f"{url}/6/cells", headers=auth({"Accept-Encoding": "identity"}))
    modelled = client.get(f"{url}/6/cells?model={model}", headers=auth())
    assert plain.content == modelled.content == values.tobytes() and plain.headers.get("content-encoding") is None
    assert packed.status_code == 200 and "immutable" in packed.headers["cache-control"]
    for sent in (plain, modelled):
        assert "immutable" not in sent.headers["cache-control"] and "no-cache" in sent.headers["cache-control"]
    for count in ("sampled", "e"):                                        # read on the averaged count: not on another
        refused = client.get(f"{url}/6/cells?model={model}&count={count}", headers=auth())
        assert refused.status_code == 404 and refused.json()["detail"] == "This iteration's count changed"
    assert client.get(f"{url}/6/cells?model=0123", headers=auth()).status_code == 404
    start_cells = client.get(f"{url}/start/cells", headers=auth())
    pinned = client.get(f"{url}/start/cells?model={model}", headers=auth())
    assert start_cells.content == pinned.content == start_values.tobytes()
    assert "no-cache" in start_cells.headers["cache-control"] and "immutable" in pinned.headers["cache-control"]
    assert client.get(f"{url}/start/cells?model=0123", headers=auth()).status_code == 404
    assert client.get(f"{url}/7/cells", headers=auth()).status_code == 404
    assert client.get(f"{url}/99", headers=auth()).status_code == 404
    assert client.get(f"{url}/6/bins", headers=auth()).status_code == 404

    # Scrubbing back (POST {seq}): the scheduler is asked; that frame comes back single, never smoothed.
    assert client.post(url, json={"seq": 1}, headers=auth()).status_code == 200
    assert scheduler.calls[-1] == ("run-a", True, 1)
    deliver(qc, "run-a", 1, 20, 10, single=True)
    single = client.get(f"{url}/1", headers=auth()).json()
    assert single["single"] is True and single["per100"] == 50.0 and single["shown"] == 50
    for bad, status in (({"seq": 999}, 404), ({"seq": "last"}, 422), ({"enabled": "yes"}, 422)):
        assert client.post(url, json=bad, headers=auth()).status_code == status
    assert scheduler.calls[-1] == ("run-a", True, 1)                              # refused: nothing handed over

    # The series (the line over the run), thinned.
    series = client.get(url + "/series?points=3", headers=auth()).json()
    assert series["model"] == model and series["start"]["per100"] == 10.0
    assert [p["seq"] for p in series["points"]] == [1, 4, 6]
    assert client.get(url + "/series", headers=auth()).json()["points"][-1]["seq"] == 6

    # The switch: off for this run (and the scheduler's), remembered for the next ones.
    assert client.get("/api/v1/runs/run-b/qc", headers=auth()).json()["enabled"] is True
    assert client.post(url, json={"enabled": False}, headers=auth()).json()["state"] == "off"
    assert scheduler.calls[-1] == ("run-a", False, None)
    assert json.loads((application.state.runs.get("run-a").dir / "qc" / "settings.json").read_text()) == {
        "v": 2, "enabled": False, "n": 1}
    assert client.get(url, headers=auth()).json()["switch_n"] == 1, "the switch's count: pages keep the newest"
    assert client.post(url, json={"enabled": False}, headers=auth()).json()["switch_n"] == 1, "no change: same count"
    assert qc.prefs == {"enabled": False}
    assert client.get("/api/v1/runs/run-b/qc", headers=auth()).json()["enabled"] is True   # its own switch was saved
    assert client.get("/api/v1/runs/run-nuclei/qc", headers=auth()).json()["state"] == "off"   # a new one: remembered
    assert client.post(url, json={"enabled": True}, headers=auth()).json()["state"] == "ready"


def test_the_headline_is_smoothed_over_the_runs_clock(world):
    """The shown integer's 0.3 margin was measured with a region's smoothing, 0.2 per second of the run clock (on
    proseg's count): results 0.1 s apart (a crop) move the headline a little each, the first after a pause replaces
    most of it, and an engine restart goes on from where it was."""
    cfg, sample = world
    data = Path(cfg["paths"]["data_dir"])
    write_run(data, "run-t", sample, "2026-10-17T15:00:00+00:00", times=[0.0, 0.1, 0.2, 0.3, 0.4, 10.4, 10.5, 10.6])
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-t")
        for seq, mixed in ((0, 10), (1, 30), (2, 30)):
            deliver(qc, "run-t", seq, 100, mixed)
        url = "/api/v1/runs/run-t/qc"
        w = K.value("region", "headline_weight")
        step = 1 - (1 - w) ** 0.1                                                 # 0.1 s: weight 0.022
        first = 10 * (1 - step) + 30 * step
        second = first * (1 - step) + 30 * step
        summaries = [client.get(f"{url}/{s}", headers=auth()).json() for s in (0, 1, 2)]
        assert [s["per100"] for s in summaries] == [10.0, round(first, 1), round(second, 1)] == [10.0, 10.4, 10.9]
        assert [s["shown"] for s in summaries] == [10, 10, 11]                     # (0.2 per result: 10, 14, 17)
    application = make(cfg)                                                        # an engine restart
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-t")
        deliver(qc, "run-t", 5, 100, 50)                                           # 10 s later: most of it new
        after = client.get("/api/v1/runs/run-t/qc/5", headers=auth()).json()
        weight = 1 - (1 - w) ** 10.2
        assert after["per100"] == round(second * (1 - weight) + 50 * weight, 1) == 46.0 and after["shown"] == 46


def test_the_line_keeps_the_whole_run_however_long_it_goes(world, monkeypatch):
    """GET /qc/series is the line over the whole run: it starts at the first scored frame however many results the run
    has had (it lost its start past RESULTS_KEPT results, ~4,000: an hour of a long crop run), also after an engine
    restart; summaries no longer held in memory are read from disk; the newest are held without any scan."""
    from sopa_companion.engine import qc as Q
    monkeypatch.setattr(Q, "RESULTS_KEPT", 20)
    cfg, sample = world
    data = Path(cfg["paths"]["data_dir"])
    write_run(data, "run-long", sample, "2026-10-17T16:00:00+00:00", frames=70, times=[0.1 * s for s in range(70)])
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-long")
        for seq in range(5, 70):                                        # live, newest first ...
            deliver(qc, "run-long", seq, 40, 4 + seq % 3)
        for seq in range(5):                                            # ... then backfill (single) of the first ones
            deliver(qc, "run-long", seq, 40, 8, single=True)
        url = "/api/v1/runs/run-long/qc"
        series = client.get(url + "/series?points=600", headers=auth()).json()["points"]
        assert [p["seq"] for p in series] == list(range(70)) and series[0]["single"] is True
        assert client.get(url, headers=auth()).json()["seqs"] == list(range(70))
        state = qc._state("run-long")
        assert len(state.results) == 20 and set(state.results) >= set(range(5))   # the newest kept: the backfill too
        early = client.get(f"{url}/10", headers=auth()).json()                      # from disk
        assert early["seq"] == 10 and early["per100"] is not None and 10 not in state.results
    application = make(cfg)                                             # an engine restart
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        series = client.get("/api/v1/runs/run-long/qc/series?points=600", headers=auth()).json()["points"]
        assert [p["seq"] for p in series] == list(range(70)) and len(qc._state("run-long").results) == 20
        thin = client.get("/api/v1/runs/run-long/qc/series?points=10", headers=auth()).json()["points"]
        assert len(thin) == 10 and thin[0]["seq"] == 0 and thin[-1]["seq"] == 69


def test_the_worker_memory_numbers_reach_the_summary(world):
    """The GPU worker's pool after a pass ({pool_used_mb, pool_total_mb}) goes into the frame's summary: the harness
    budgets the check's own share of GPU 1 with it."""
    cfg, sample = world
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client)
        state = qc._state("run-a")
        frame = qc.runs.get("run-a").frames[3]
        qc.qc_result({"kind": "frame", "run": "run-a", "seq": 3, "model": state.key, "dir": str(state.dir),
                      "source": "live", "single": False, "frame": {k: frame.get(k) for k in ("seq", "iter", "phase")},
                      "summary": {"judged": 40, "mixed": 4, "gpu": {"pool_used_mb": 812.5, "pool_total_mb": 1650.0,
                                                                  "allocator": "rmm", "note": "x", "bad": True}}})
        summary = client.get("/api/v1/runs/run-a/qc/3", headers=auth()).json()
        assert summary["gpu"] == {"pool_used_mb": 812.5, "pool_total_mb": 1650.0, "allocator": "rmm"}


def test_closing_stops_a_fit_before_the_save_dialog(world, monkeypatch):
    """The engine stops the mixing check first when it closes: the save dialog waits up to 10 minutes, and a fit
    must neither go on nor start on GPU 0 (shared with the user's notebook) meanwhile."""
    import psutil
    from sopa_companion.server import app as server
    cfg, sample = world
    log = Path(cfg["paths"]["data_dir"]).parent / "launches.jsonl"
    monkeypatch.setenv("FAKE_QC_MODE", "hang")
    monkeypatch.setenv("FAKE_QC_LOG", str(log))
    seen = {}

    def dialog(unsaved, timeout_s=None, **kw):
        pid = json.loads(log.read_text().splitlines()[0])["pid"]
        seen.update(closed=application.state.qc_models.closed,
                    trainer_gone=not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE)
        return True
    monkeypatch.setattr(server, "ask_to_save", dialog)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        assert client.post(f"/api/v1/samples/{sample['id']}/qc", json={"learn": True}, headers=auth()).status_code == 200
        wait_for(lambda: log.exists() and application.state.qc_models.describe(sample["id"])["state"] == "learning")
        application.state.runs.pending = lambda: [{"id": "run-a", "state": "running"}]   # a run to offer for saving
    assert seen == {"closed": True, "trainer_gone": True}


def test_a_frame_request_needs_clusters_and_the_umap(world):
    cfg, sample = world
    from sopa_companion.server import create_app
    application = create_app(cfg, token="t" * 40)                 # the real scheduler, without its environment
    application.state.qc_models.probe = lambda: dict(IDLE)
    with TestClient(application, base_url=BASE) as client:
        ready_run(client)
        response = client.post("/api/v1/runs/run-a/qc", json={"seq": 1}, headers=auth())
        detail = response.json()["detail"]
        assert response.status_code == 503 and "Clusters and the UMAP" in detail, detail
        assert detail == ("The mixing check needs Clusters and the UMAP, which can't run here: the python set under "
                          f"[analysis] ({cfg['analysis']['python']}) is not there")
        assert client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth()).status_code == 200


def test_trials_without_a_previous_run(app):
    application, client, sample = app
    qc = application.state.qc
    ready_run(client, "run-a")
    ready_run(client, "run-b")
    for seq, mixed in ((2, 4), (3, 6)):
        deliver(qc, "run-a", seq, 40, mixed)
    assert client.get("/api/v1/runs/run-b/qc/previous", headers=auth()).status_code == 404

    url = "/api/v1/runs/run-a/qc/trial"
    for bad in ({"rule": "r4", "flag": "cell-compactness"}, {"rule": "R4", "flag": "Cell compactness"},
                {"rule": "R4", "flag": "cell-compactness", "area": "a1"}, {"rule": "R4", "flag": "x", "to": [1]},
                {"rule": "R4", "flag": "voxel-size", "to": 2.0}):              # not a Live knob: never a trial
        assert client.post(url, json=bad, headers=auth()).status_code == 422, bad
    run = application.state.runs.get("run-a")
    body = {"rule": "R4", "area": "A", "flag": "cell-compactness", "from": 0.04, "to": 0.02}
    assert client.post(url, json=body, headers=auth()).status_code == 409    # a finished run is not sampling
    run.meta["state"] = "running"
    frames = list(run.frames)
    run.frames = frames[:4]                                               # Try it at iteration 30 (frame 3) ...
    trial = client.post(url, json=body, headers=auth()).json()
    assert trial["id"] == 1 and trial["flag"] == "cell-compactness" and trial["verdict"] is None
    assert trial["asked_seq"] == 3 and trial["start_seq"] is None
    run.frames = frames                                                   # ... proseg applies it from frame 5
    trial = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["trial"]
    assert trial["start_seq"] == 5 and trial["start_iter"] == 50          # the first frame whose live value is 0.02
    assert client.post(url, json={"rule": "R2", "flag": "cell-compactness", "to": 0.04}, headers=auth()).status_code == 409
    end = "/api/v1/runs/run-a/qc/trial/end"
    assert client.post(end, json={"outcome": "maybe"}, headers=auth()).status_code == 422
    assert client.post(end, json={"outcome": "undone"}, headers=auth()).json()["outcome"] == "undone"
    assert client.post(end, json={"outcome": "kept"}, headers=auth()).status_code == 404
    assert client.post(end, json={"outcome": "kept", "rule": "U"}, headers=auth()).status_code == 404   # no Undo card
    assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["trial"] is None
    lines = [json.loads(l) for l in (qc._state("run-a").dir / "trials.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["start", "end"] and lines[1]["start_seq"] == 5 and lines[1]["area"] == "A"


def test_runs_without_a_start_line_or_a_table(app):
    """A start with nuclei has no start line: nuclei only, and cells with their nuclei too (proseg's count at
    iteration 0 holds only the nuclei there); describe says how the run started (start_kind) and is ready without
    waiting for a start line."""
    application, client, sample = app
    qc = application.state.qc
    ready_run(client, "run-nuclei")
    view = client.get("/api/v1/runs/run-nuclei/qc", headers=auth()).json()
    assert view["state"] == "ready" and view["start_source"] == "nuclei" and view["start_kind"] == "nuclei"
    deliver_start(qc, "run-nuclei", 40, 4)                               # the scheduler scores the prior anyway ...
    assert client.get("/api/v1/runs/run-nuclei/qc", headers=auth()).json()["start"] is None   # ... it is not shown
    assert client.get("/api/v1/runs/run-nuclei/qc/start", headers=auth()).status_code == 404
    ready_run(client, "run-paired")
    view = client.get("/api/v1/runs/run-paired/qc", headers=auth()).json()
    assert view["state"] == "ready" and view["start"] is None and view["start_source"] == "xenium"
    assert view["start_kind"] == "paired"
    deliver_start(qc, "run-paired", 40, 4)
    # Its start line's routes refuse it for good, as Go deeper's start line does: never "no start line yet".
    for run_id, kind, words in (("run-paired", "paired", "This run started from cells with their nuclei: it has no start line"),
                                ("run-nuclei", "nuclei", "This run started from nuclei only: it has no start line")):
        assert qc.log_info(run_id)["start_kind"] == kind, "the run log says it has nothing to compare with"
        for path in ("start", "start/cells", "start/bins"):
            answer = client.get(f"/api/v1/runs/{run_id}/qc/{path}", headers=auth())
            assert answer.status_code == 404, path
            assert {k: answer.json().get(k) for k in ("detail", "reason", "start_kind")} == \
                {"detail": words, "reason": "no_start", "start_kind": kind}, (run_id, path)
    assert client.get("/api/v1/runs/run-paired/qc", headers=auth()).json()["start"] is None
    application.state.qc.analysis.start = "none"                         # a run proseg exported no prior for
    ready_run(client, "run-b")
    assert client.get("/api/v1/runs/run-b/qc", headers=auth()).json()["state"] == "ready"
    application.state.qc.analysis.unavailable = "no_transcripts"         # made before live analysis: nothing scored
    view = client.get("/api/v1/runs/run-b/qc", headers=auth()).json()
    assert (view["state"], view["reason"]) == ("unavailable", "no_transcripts")
    application.state.qc.analysis.unavailable = None
    assert client.get("/api/v1/runs/missing/qc", headers=auth()).status_code == 404
    # A run on a Xenium bundle (made before SpatialData samples): no cell table to learn from.
    runs = application.state.runs
    legacy = client.post("/api/v1/runs", json={"sample": application.state.samples.list()[0]["id"], "box": [50, 50, 150, 150]},
                         headers=auth()).json()
    view = client.get(f"/api/v1/runs/{legacy['id']}/qc", headers=auth()).json()
    assert view["state"] == "unavailable" and view["reason"] == "no_table"
    assert qc.qc_model(runs.get(legacy["id"])) is None
    client.post(f"/api/v1/runs/{legacy['id']}/control", json={"action": "abort"}, headers=auth())
    wait_for(lambda: runs.get(legacy["id"]).meta["state"] in ("stopped", "done", "failed"))
    # A deleted run is forgotten (its qc folder went with it).
    assert "run-nuclei" in qc.states
    assert client.delete("/api/v1/runs/run-nuclei", headers=auth()).status_code == 200
    assert client.get("/api/v1/runs/run-nuclei/qc", headers=auth()).status_code == 404 and "run-nuclei" not in qc.states


def test_a_sample_the_check_cannot_learn_says_why_also_with_the_switch_off(world, monkeypatch):
    """A sample whose only counts table is log-normalised says so at once, with no wait for
    GPU 0, and a table called "cells" is found. A run on the first says why even with its switch off ("sample_reason":
    the off panel has more to say than "turn it on"). The sample's own state and reason are what Compare's 409 names
    (engine/compare.py reads them from QcModels.describe) once the check failed or cannot run: then waiting cannot help."""
    from sopa_companion.engine.config import merge
    from sopa_companion.engine.zarr_samples import sample_id
    from test_qc_models import write_counts
    cfg, sample = world
    data = Path(cfg["paths"]["data_dir"])
    root = Path(sample["path"]).parent
    write_table(root / "lognorm.zarr", "xenium_table")
    write_counts(root / "lognorm.zarr", "xenium_table", np.log1p(np.arange(1, 9000) / 7.0))
    write_table(root / "cells.zarr", "cells")
    extra = [{"id": sample_id(str((root / f"{name}.zarr").resolve())), "name": name,
              "path": str((root / f"{name}.zarr").resolve()), "kind": "zarr", "opened": "2026-10-22T12:00:00"}
             for name in ("lognorm", "cells")]
    (data / "samples.json").write_text(json.dumps(json.loads((data / "samples.json").read_text()) + extra))
    write_run(data, "run-lognorm", extra[0], "2026-10-22T12:00:00+00:00")
    monkeypatch.setenv("FAKE_QC_MODE", "error:not_settled")
    application = make(merge(cfg, {"qc": {"enabled_by_default": False, "learn": "start"}}))
    with TestClient(application, base_url=BASE) as client:
        models = application.state.qc_models
        lognorm = client.get(f"/api/v1/samples/{extra[0]['id']}/qc", headers=auth()).json()
        assert (lognorm["state"], lognorm["reason"]) == ("unavailable", "not_counts")
        found = client.get(f"/api/v1/samples/{extra[1]['id']}/qc", headers=auth()).json()
        assert (found["state"], found["table"]) == ("off", "cells")
        view = client.get("/api/v1/runs/run-lognorm/qc", headers=auth()).json()
        assert (view["enabled"], view["state"], view["reason"]) == (False, "off", None)
        assert (view["sample_state"], view["sample_reason"]) == ("unavailable", "not_counts")
        assert models.describe(sample["id"])["state"] == "off"              # not asked for yet: turning it on may help
        client.post("/api/v1/runs/run-a/qc", json={"enabled": True}, headers=auth())
        wait_for(lambda: client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["state"] == "failed", timeout=30)
        assert (models.describe(sample["id"])["state"], models.describe(sample["id"])["reason"]) == ("failed",
                                                                                                    "not_settled")
        assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["sample_reason"] == "not_settled"


def test_the_page_gets_qc_events(app):
    application, client, sample = app
    qc = application.state.qc
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": "run-a", "since": 0})
        ready_run(client)
        deliver_start(qc, "run-a", 40, 4)
        for seq in (3, 4):
            deliver(qc, "run-a", seq, 40, 6)
        qc.qc_result({"kind": "error", "run": "run-a", "seq": 5, "model": qc._state("run-a").key,
                      "error": "ValueError: only 12 cells have >= 20 transcripts"})
        seen, deadline = [], 200
        while deadline and not (any(e.get("state") == "error" for e in seen)
                                and {e.get("seq") for e in seen if e.get("state") == "ready"} >= {3, 4}):
            event = ws.receive_json()
            deadline -= 1
            if event["ev"] == "qc":
                seen.append(event)
    samples = [e for e in seen if "sample" in e]
    assert samples and samples[-1]["sample"] == sample["id"] and samples[-1]["state"] == "ready"
    assert {e["state"] for e in samples} <= {"waiting", "learning", "ready"}
    assert all(set(e) >= {"progress", "eta_s", "stage"} for e in samples)
    (start,) = [e for e in seen if e.get("state") == "start"]
    assert start["run"] == "run-a" and start["summary"]["per100"] == 10.0
    ready = [e for e in seen if e.get("state") == "ready" and "run" in e]
    assert [e["seq"] for e in ready] == [3, 4] and ready[0]["summary"]["shown"] == 15 and ready[0]["iter"] == 30
    (error,) = [e for e in seen if e.get("state") == "error"]
    assert error["run"] == "run-a" and error["seq"] == 5 and "12 cells" in error["error"]


def test_a_switch_flipped_elsewhere_reaches_the_pages_watching_the_run(app):
    """Another window or the API flips a run's switch: every page watching the run hears it ({state: "switch",
    enabled, n}), on the first look after a watch (a flip between the page's GET /runs/{id}/qc and its watch is never
    lost) and once per change; `n` counts the switch's changes (GET /runs/{id}/qc says switch_n), so a page keeps the
    newer of an answer and a push whatever order they arrive in."""
    application, client, sample = app
    qc = application.state.qc
    switches = lambda events: [e for e in events if e.get("run") == "run-a" and e.get("state") == "switch"]
    cursor = {}
    assert switches(qc.events(cursor, "run-a", 0.0)) == [{"ev": "qc", "run": "run-a", "state": "switch",
                                                           "enabled": True, "n": 0}], "first look: what it is now"
    assert switches(qc.events(cursor, "run-a", 0.5)) == [], "then only changes"
    assert client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth()).status_code == 200
    assert switches(qc.events(cursor, "run-a", 1.0)) == [{"ev": "qc", "run": "run-a", "state": "switch",
                                                           "enabled": False, "n": 1}]
    assert switches(qc.events(cursor, "run-a", 2.0)) == [], "once per change"
    client.post("/api/v1/runs/run-a/qc", json={"enabled": True}, headers=auth())
    assert [(e["enabled"], e["n"]) for e in switches(qc.events(cursor, "run-a", 3.0))] == [(True, 2)]
    assert [e["run"] for e in qc.events(cursor, "run-b", 4.0) if e.get("state") == "switch"] == ["run-b"], \
        "another run watched: its own first look"
    # Over the socket: a page watching the run hears the API's change (the watch is taken once the run is pushed).
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": "run-a", "since": 0})
        heard, pushed = [], False
        for _ in range(200):                                  # the first look after the watch: as it is now
            event = ws.receive_json()
            heard += switches([event]) if event["ev"] == "qc" else []
            pushed = pushed or (event["ev"] == "run" and (event.get("run") or {}).get("id") == "run-a")
            if pushed and heard:
                break
        assert heard == [{"ev": "qc", "run": "run-a", "state": "switch", "enabled": True, "n": 2}]
        client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth())
        heard = []
        for _ in range(200):
            event = ws.receive_json()
            heard += switches([event]) if event["ev"] == "qc" else []
            if heard:
                break
    assert heard == [{"ev": "qc", "run": "run-a", "state": "switch", "enabled": False, "n": 3}]


def test_the_run_says_which_frame_waits_for_its_sample_to_learn(app):
    """A frame asked for while the sample learns is kept by the scheduler (scored once the model is ready): the run's
    answer names it, so the page can say it was asked for."""
    application, client, sample = app
    assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["requested"] is None
    application.state.qc.analysis.requested = 4
    assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["requested"] == 4


def test_a_frame_asked_for_is_named_as_the_counted_frame_it_snaps_to_even_once_the_scheduler_took_it(app):
    """POST /runs/{id}/qc {seq}: the reply names the counted frame the scheduler snapped the frame to (configure_qc's
    own reply), also when its thread, just woken, took the request before the reply's describe looked (describe_qc then
    says none waits): DESIGN 2.2, the reply carries its seq, as POST /runs/{id}/analysis does."""
    application, client, sample = app
    scheduler = application.state.qc.analysis
    scheduler.configure_qc = lambda run_id, enabled=None, seq=None: {"requested": 2} if seq is not None else {}
    scheduler.requested = None                                 # taken already: nothing waits
    reply = client.post("/api/v1/runs/run-a/qc", json={"seq": 3}, headers=auth())
    assert reply.status_code == 200 and reply.json()["requested"] == 2
    assert client.post("/api/v1/runs/run-a/qc", json={"enabled": True}, headers=auth()).json()["requested"] is None


def test_the_check_resumes_after_an_engine_restart(world):
    cfg, sample = world
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client)
        deliver_start(qc, "run-a", 40, 4)
        for seq, mixed in ((3, 4), (4, 12)):                                # 10, then 30 in 100: a region's weight,
            deliver(qc, "run-a", seq, 40, mixed)                            # 0.2, makes it 14, then 17.2
        assert client.get("/api/v1/runs/run-a/qc/4", headers=auth()).json()["per100"] == 14.0
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        view = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()
        assert view["state"] == "ready" and view["seqs"] == [3, 4] and view["start"]["per100"] == 10.0
        assert client.get("/api/v1/runs/run-a/qc/4", headers=auth()).json()["shown"] == 14
        assert qc.analysis.calls == [("run-a", True, None)]                 # the new scheduler gets the switch again
        deliver(qc, "run-a", 5, 40, 12)                                     # smoothing continues from state.json
        assert client.get("/api/v1/runs/run-a/qc/5", headers=auth()).json()["per100"] == 17.2
        assert len(json.loads(Path(qc._state("run-a").dir / "state.json").read_text())["ema"]) == 2


# ---- flagged areas (engine/qc_areas.py through QcManager) --------------------------------------------------------------
NY_B, NX_B = 12, 16                                          # the bins of the area tests: 50 µm, 16 x 12
P1, P2 = [(r, c) for r in range(1, 4) for c in range(1, 4)], [(r, c) for r in range(7, 10) for c in range(10, 13)]


def bins_table(hot=(), start_hot=None, judged=20, mixed=2, hot_mixed=12):
    """A frame's per-bin sums as the worker's bins file holds them (worker.write_bins): judged and mixed per bin (10 %
    mixed, 60 % in the `hot` bins), and, when `start_hot` (the start's hot bins) is given, its own start block per bin
    (judged_start, mixed_start) and its cells paired with it."""
    def plane(cells):
        values = np.full((NY_B, NX_B), float(mixed))
        for r, c in cells:
            values[r, c] = hot_mixed
        return values.ravel()
    now, n = plane(hot), NY_B * NX_B
    table = {"grid": np.array([0.0, 0.0, 50.0, NX_B, NY_B]), "bin": np.arange(n, dtype=np.int32),
             "cells": np.full(n, judged + 5, np.int32), "judged": np.full(n, judged, np.int32),
             "mixed": now.astype(np.int32), "tx": np.full(n, 300.0 * (judged + 5), np.float32)}
    if start_hot is not None:
        start = plane(start_hot)
        table.update(paired=np.full(n, judged, np.int32), paired_mixed_now=now.astype(np.int32),
                     paired_mixed_start=start.astype(np.int32), n10=np.maximum(now - start, 0).astype(np.int32),
                     n01=np.maximum(start - now, 0).astype(np.int32), tx_start=np.full(n, 250.0 * (judged + 5), np.float32),
                     judged_start=np.full(n, judged, np.int32), mixed_start=start.astype(np.int32))
    return table


def deliver_bins(qc, run_id: str, seq: int, table: dict, single=False):
    state = qc._state(run_id)
    frame = qc.runs.get(run_id).frames[seq]
    qc.qc_result({"kind": "frame", "run": run_id, "seq": seq, "model": state.key, "dir": str(state.dir),
                  "source": "request" if single else "live", "single": single,
                  "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                  "summary": {"kind": "frame", "seq": seq, "judged": int(table["judged"].sum()),
                              "mixed": int(table["mixed"].sum()), "ncells": NCELLS, "median_tx": 300.0},
                  "bins": table, "cells": None, "start": None})


def deliver_start_bins(qc, run_id: str, hot=P1):
    """The start line as the scheduler hands it over: the worker has written start.bins.npz (kept: an engine restart
    reads it back for the tags)."""
    state = qc._state(run_id)
    table = bins_table(hot)
    state.dir.mkdir(parents=True, exist_ok=True)
    np.savez(state.dir / "start.bins.npz", **table)
    qc.qc_result({"kind": "start", "run": run_id, "seq": None, "model": state.key, "dir": str(state.dir),
                  "source": "start", "single": True, "bins": table, "cells": None, "start": None,
                  "summary": {"kind": "start", "judged": int(table["judged"].sum()), "mixed": int(table["mixed"].sum()),
                              "median_tx": 386.0, "ncells": NCELLS}})


def area_world(world, run_id="run-areas", frames=45):
    """A run whose frame s is iteration s, t = 0.5 s, burn-in 10: areas are shown from iteration 30."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), run_id, sample, "2026-10-17T18:00:00+00:00", frames=frames,
              times=[0.5 * s for s in range(frames)], burnin=10)
    return cfg


def test_areas_are_flagged_steadily_from_burn_in_plus_20(world):
    """The start line's own areas (served while cells form), then the run's: smoothed, from burn-in + 20 iterations
    with letters A, B by rank, tagged against the start, entering only after 2 s, the set drawn logged when it changes
    (drawn.jsonl: the letters of single snapshots), each outline stored once and served with every summary, and the
    per-bin map for the overview layer. What only the page's area list and stage read is no longer made:
    no spots left out, no count listed, no events, no whole number per area."""
    cfg = area_world(world)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-areas")
        url = "/api/v1/runs/run-areas/qc"
        deliver_bins(qc, "run-areas", 0, bins_table(P1))      # scored before the start line: no paired sums
        deliver_start_bins(qc, "run-areas")
        start = client.get(url + "/start", headers=auth()).json()
        (own,) = start["areas"]
        assert (own["L"], own["tag"]) == ("A", "already") and len(own["outline"]) >= 4
        assert not {"listed", "more_spots"} & set(start) and "shown" not in own
        assert own["per100"] == own["start_per100"] and start["grid"] == {"x0": 0.0, "y0": 0.0, "bin": 50.0, "nx": 16, "ny": 12}
        assert all(0 <= x <= 300 and 0 <= y <= 300 for x, y in own["outline"])        # around P1 (top left)
        # The overview grid: each bin's 3 x 3 window share (sum M / sum J), then its own judged cells (planes: 2).
        grid = client.get(url + "/start/bins", headers=auth())
        n = NX_B * NY_B
        assert grid.status_code == 200 and len(grid.content) == 2 * n and grid.content[2 * NX_B + 2] == 150
        assert grid.content[0] == 56 and grid.content[n + 2 * NX_B + 2] == 20     # P1's corner: 18 of 80 judged
        assert json.loads(grid.headers["x-bins"]) == {"x0": 0.0, "y0": 0.0, "bin": 50.0, "nx": 16, "ny": 12, "planes": 2}
        view = client.get(url, headers=auth()).json()
        assert "area_from_iter" not in view
        # The start line as describe and the series give it carries its areas too (the page's paused prior).
        assert [(a["L"], a["outline"]) for a in view["start"]["areas"]] == [("A", own["outline"])]
        assert view["start"]["per100"] == start["per100"] and "listed" not in view["start"]
        assert [a["L"] for a in client.get(url + "/series", headers=auth()).json()["start"]["areas"]] == ["A"]
        for seq in range(1, 40):
            if seq != 33:                                     # one frame the live passes skipped (scored later)
                deliver_bins(qc, "run-areas", seq, bins_table(P1 + (P2 if seq >= 31 else []), start_hot=P1))
        summaries = {s: client.get(f"{url}/{s}", headers=auth()).json() for s in range(40) if s != 33}
        for s in range(30):                                   # cells still forming: the start line's areas are shown
            early = summaries[s]
            assert early["area_view"] == "start" and "areas_from" not in early, s
            assert [(a["L"], a["tag"], a["outline"]) for a in early["areas"]] == [("A", "already", own["outline"])], s
            assert json.loads((qc._state("run-areas").dir / f"{s:06d}.qc.json").read_text())["areas"] == []   # compact
        # Burn-in + 20: the run's own areas. The start line's A is still there: it keeps its letter.
        assert [a["L"] for a in summaries[30]["areas"]] == ["A"] and summaries[30]["area_view"] == "smoothed"
        assert summaries[30]["areas"][0]["tag"] == "already" and "areas_from" not in summaries[30]
        for s in (31, 32, 34):                                # P2 is hot from frame 31: shown 2 s later (frame 35)
            assert [a["L"] for a in summaries[s]["areas"]] == ["A"]
        assert sorted(a["L"] for a in summaries[35]["areas"]) == ["A", "B"]
        assert not any({"area_events", "more_spots", "listed", "noise"} & set(summaries[s]) for s in summaries)
        # What was drawn, logged only when it changed: nothing before the start line, its A from frame 1, the run's own
        # A from frame 30 (cells formed), B joining at frame 35.
        drawn = [json.loads(line) for line in (qc._state("run-areas").dir / "drawn.jsonl").read_text().splitlines()]
        logged = [(d["seq"], d["start"], sorted(L for L, _ in d["areas"])) for d in drawn]
        assert logged[:4] == [(0, True, []), (1, True, ["A"]), (30, False, ["A"]), (35, False, ["A", "B"])]
        tags = {a["L"]: a["tag"] for a in summaries[39]["areas"]}
        assert tags == {"A": "already", "B": "rose"}           # paired sums only from the frames after the start line
        late = {a["L"]: a for a in summaries[39]["areas"]}
        assert late["B"]["start_per100"] < late["B"]["per100"] and late["B"]["judged"] >= 120
        assert late["A"]["depth"] == 300 and late["A"]["depth_start"] == 250 and late["A"]["cells"] > late["A"]["judged"]
        assert all(10 * 50 - 100 <= x <= 13 * 50 + 100 and 7 * 50 - 100 <= y <= 10 * 50 + 100 for x, y in late["B"]["outline"])
        assert all("shown" not in a for a in summaries[39]["areas"]) and len(json.dumps(summaries[39])) < 8192
        # Stored compactly: each outline once in shapes.jsonl; summaries name their shape (served with the outline).
        folder = qc._state("run-areas").dir
        stored = json.loads((folder / "000039.qc.json").read_text())
        assert all("outline" not in a and len(a["shape"]) == 12 for a in stored["areas"])
        shapes = [json.loads(line) for line in (folder / "shapes.jsonl").read_text().splitlines()]
        assert len({s["id"] for s in shapes}) == len(shapes) and len(shapes) <= 6
        assert {s["id"]: s["outline"] for s in shapes}[late["B"]["shape"]] == late["B"]["outline"]
        series = (folder / "series.jsonl").read_text()
        assert '"outline"' not in series and len(series.splitlines()) == 39       # no area's outline stored there
        # The overview layer's per-bin share: smoothed for live results.
        overview = client.get(f"{url}/39/bins", headers=auth({"Accept-Encoding": "identity"}))
        planes = np.frombuffer(overview.content, np.uint8).reshape(2, NY_B, NX_B)
        values, judged = planes
        assert overview.status_code == 200 and json.loads(overview.headers["x-bins"])["nx"] == NX_B
        assert json.loads(overview.headers["x-bins"])["planes"] == 2 and summaries[39]["grid"]["planes"] == 2
        assert values[2, 2] in (149, 150) and values[0, 0] in (55, 56) and 30 < values[8, 11] < 149   # P2 warming up
        assert (judged == 20).all()
        packed = client.get(f"{url}/39/bins?model={qc._state('run-areas').key}", headers=auth({"Accept-Encoding": "zstd"}))
        assert packed.headers.get("content-encoding") == "zstd" and "immutable" in packed.headers["cache-control"]
        assert client.get(f"{url}/39/bins?model=0123", headers=auth()).status_code == 404
        assert client.get(f"{url}/33/bins", headers=auth()).status_code == 404       # not scored


def test_a_scrubbed_frame_shows_its_own_areas_never_a_later_ones(world):
    cfg = area_world(world)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-areas")
        deliver_start_bins(qc, "run-areas")
        url = "/api/v1/runs/run-areas/qc"
        for seq in range(1, 40):
            if seq != 33:
                deliver_bins(qc, "run-areas", seq, bins_table(P1 + (P2 if seq >= 31 else []), start_hot=P1))
        assert [a["L"] for a in client.get(f"{url}/32", headers=auth()).json()["areas"]] == ["A"]   # as it was then
        forming = client.get(f"{url}/20", headers=auth()).json()     # cells still forming: the start line's areas
        assert forming["area_view"] == "start" and [a["tag"] for a in forming["areas"]] == ["already"]
        # Frame 33, asked for after frame 39: judged alone ("single"), the run's areas untouched. Its area at P2 is no
        # place the page showed at frame 33 (A was at P1 then): it takes the first letter that names nothing there,
        # never A, which names P1 (by rank it would have been A).
        deliver_bins(qc, "run-areas", 33, bins_table(P2, start_hot=P1), single=True)
        single = client.get(f"{url}/33", headers=auth()).json()
        assert single["single"] is True and single["area_view"] == "single"
        assert [(a["L"], a["tag"]) for a in single["areas"]] == [("B", "rose")]       # P2 alone there
        # A frame asked for where the page showed areas keeps their letters for the same places.
        deliver_bins(qc, "run-areas", 36, bins_table(P2 + P1, start_hot=P1), single=True)   # scored again, alone
        again = client.get(f"{url}/36", headers=auth()).json()
        assert again["area_view"] == "single" and {a["L"]: a["tag"] for a in again["areas"]} == {"A": "already", "B": "rose"}
        assert client.get(f"{url}/33/bins", headers=auth()).status_code == 200
        deliver_bins(qc, "run-areas", 40, bins_table(P1 + P2, start_hot=P1))
        after = client.get(f"{url}/40", headers=auth()).json()
        assert sorted(a["L"] for a in after["areas"]) == ["A", "B"] and after["area_view"] == "smoothed"
        # Early single frames have no areas of their own either: the start line's are served with them.
        deliver_bins(qc, "run-areas", 41, bins_table(P1, start_hot=P1))
        deliver_bins(qc, "run-areas", 12, bins_table(P2, start_hot=P1), single=True)
        early = client.get(f"{url}/12", headers=auth()).json()
        assert early["area_view"] == "start" and [a["L"] for a in early["areas"]] == ["A"] and early["single"] is True


def test_the_areas_resume_after_an_engine_restart(world):
    cfg = area_world(world)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-areas")
        deliver_start_bins(qc, "run-areas")
        for seq in range(1, 38):
            deliver_bins(qc, "run-areas", seq, bins_table(P1 + (P2 if seq >= 20 else []), start_hot=P1))
        last = client.get("/api/v1/runs/run-areas/qc/37", headers=auth()).json()["areas"]
        before, tags = {a["L"]: a["shape"] for a in last}, {a["L"]: a["tag"] for a in last}
        assert sorted(before) == ["A", "B"] and sorted(tags.values()) == ["already", "rose"]
        folder = qc._state("run-areas").dir
    saved = json.loads((folder / "state.json").read_text())       # written at close: the last result's smoothing
    assert saved["last_live"] == 37 and saved["areas"]["seq"] == 37 and saved["areas"]["kind"] == "detail"
    assert sorted(a["L"] for a in saved["areas"]["drawn"]) == ["A", "B"] and saved["areas"]["revealed"] is True
    assert saved["areas"]["flag"] and set(saved["areas"]["sm"]["keys"]) == {"J", "M", "B", "N1", "S1", "n10", "n01",
                                                                             "J0", "M0"}
    assert not list(folder.glob("*.npz.*")) and not (folder / "areas.state.npz").exists()
    application = make(cfg)                                    # an engine restart
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        start = client.get("/api/v1/runs/run-areas/qc/start", headers=auth()).json()
        assert [a["L"] for a in start["areas"]] == ["A"] and start["areas"][0]["outline"]
        old = client.get("/api/v1/runs/run-areas/qc/36", headers=auth()).json()     # from disk, with its outlines
        assert sorted(a["L"] for a in old["areas"]) == ["A", "B"] and all(a["outline"] for a in old["areas"])
        deliver_bins(qc, "run-areas", 38, bins_table(P1 + P2, start_hot=P1))
        after = client.get("/api/v1/runs/run-areas/qc/38", headers=auth()).json()
        assert {a["L"]: a["shape"] for a in after["areas"]} == before and after["area_view"] == "smoothed"
        assert {a["L"]: a["tag"] for a in after["areas"]} == tags          # paired with the start line read back


def test_a_whole_samples_summary_with_eight_areas_stays_under_8_kb(world):
    """A whole-sample (tiled) run draws up to 8 areas on 100 µm bins: each outline gets at most 44 points
    (qc_areas.OUTLINE_BUDGET over the cap), so the summary the page gets stays <= 8 KB however ragged the areas are,
    with 5-digit µm coordinates; every area carries its own noise share (the run's, and each area's whole number, no
    longer)."""
    cfg, sample = world
    nx, ny, frames = 64, 40, 30
    write_run(Path(cfg["paths"]["data_dir"]), "run-tiled", sample, "2026-10-17T19:00:00+00:00", frames=frames,
              times=[float(s) for s in range(frames)], burnin=0, mode="tiled")
    rng = np.random.default_rng(11)
    holes = (np.arange(ny)[:, None] % 2 == 0) & (np.arange(nx)[None, :] % 2 == 0)
    tissue = ~holes                                           # ragged tissue: outlines follow the bins holding cells
    judged = np.where(tissue, 30, 0)
    mixed = np.where(tissue, 3, 0)
    for k in range(8):                                        # 8 blobs, far apart
        r0, c0 = 2 + 20 * (k // 4), 2 + 16 * (k % 4)
        blob = (rng.random((10, 10)) < 0.6) & tissue[r0:r0 + 10, c0:c0 + 10]
        mixed[r0:r0 + 10, c0:c0 + 10][blob] = 16
    n = nx * ny
    table = {"grid": np.array([10050.0, 9030.0, 100.0, nx, ny]), "bin": np.arange(n, dtype=np.int32),
             "cells": np.where(tissue, judged + 5, 0).ravel().astype(np.int32), "judged": judged.ravel().astype(np.int32),
             "mixed": mixed.ravel().astype(np.int32), "tx": np.full(n, 9000.0, np.float32),
             "assigned": np.full(n, 10000.0, np.float32), "foreground": np.full(n, 8500.0, np.float32)}
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-tiled")
        for seq in range(20, frames):                         # areas are shown from iteration 0 + 20
            deliver_bins(qc, "run-tiled", seq, table)
        served = client.get(f"/api/v1/runs/run-tiled/qc/{frames - 1}", headers=auth()).json()
    areas = served["areas"]
    assert len(areas) == 8 and served["area_view"] == "smoothed"
    points = [len(a["outline"]) for a in areas]
    assert sum(p == 44 for p in points) >= 5 and min(points) >= 4    # the budget binds on these ragged areas
    assert all(10050 <= x <= 10050 + 100 * nx and 9030 <= y <= 9030 + 100 * ny for a in areas for x, y in a["outline"])
    # As sent (compact JSON), with room for the worker's ms and gpu numbers this fake leaves out (each area with its
    # own comparison and edge, the summary with its coverage, change and provenance); even with spaces.
    from sopa_companion.engine.qc import SUMMARY_BYTES
    assert len(json.dumps(served, separators=(",", ":"))) <= SUMMARY_BYTES - 400
    assert len(json.dumps(served)) <= SUMMARY_BYTES + 512
    assert all(set(a["local"]) == {"judged", "paired", "matched_change", "threshold", "enough", "direction"}
               for a in areas)
    assert all(a["edge"] is True for a in areas)               # ragged tissue: every area meets empty bins
    assert all(a["noise"] == pytest.approx(0.15, abs=1e-3) and not {"noise_run", "shown"} & set(a) for a in areas)
    assert "noise" not in served


def test_state_json_is_written_at_most_once_a_second_and_at_close(world):
    """state.json (the headline's and the areas' smoothing) after a live result: at once for the first one and when
    the drawn areas change, otherwise at most once per STATE_SAVE_S of the engine's clock; the engine's close writes
    the newest."""
    cfg = area_world(world, "run-save")
    application = make(cfg)
    now = [100.0]
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        qc.clock = lambda: now[0]
        ready_run(client, "run-save")
        deliver_start_bins(qc, "run-save")
        folder = qc._state("run-save").dir

        def saved():
            return json.loads((folder / "state.json").read_text())["last_live"]
        deliver_bins(qc, "run-save", 31, bins_table(P1, start_hot=P1))
        assert saved() == 31                                   # the first live result (the areas are revealed)
        deliver_bins(qc, "run-save", 32, bins_table(P1, start_hot=P1))
        assert saved() == 31                                   # the same areas, within a second: not yet
        now[0] += 1.0
        deliver_bins(qc, "run-save", 33, bins_table(P1, start_hot=P1))
        assert saved() == 33
        for seq in range(34, 44):                              # P2 enters 2 s after it shows: written at once
            deliver_bins(qc, "run-save", seq, bins_table(P1 + P2, start_hot=P1))
            if len(client.get(f"/api/v1/runs/run-save/qc/{seq}", headers=auth()).json()["areas"]) == 2:
                break
        assert saved() == seq and seq > 34
        last = seq + 3
        for seq in range(seq + 1, last + 1):                   # (B may still grow a bin: then written at once)
            deliver_bins(qc, "run-save", seq, bins_table(P1 + P2, start_hot=P1))
    assert saved() == last                                     # the engine's close writes the newest


def test_areas_name_the_cell_groups_that_meet_there(world, tmp_path):
    """With Clusters & UMAP on, an area names the clusters of its frame's own UMAP result that hold >= 15 % of its
    cells (largest first, by their first two top genes), the cells placed by their centroid in that frame; without a
    UMAP result for the frame, no groups (and no key)."""
    import zstandard
    cfg = area_world(world, "run-groups")
    folder = Path(cfg["paths"]["data_dir"]) / "runs" / "run-groups"
    frames = [json.loads(line) for line in (folder / "live" / "frames.jsonl").read_text().splitlines()]
    cx = np.full(NCELLS, np.nan, np.float32)
    cy = np.full(NCELLS, np.nan, np.float32)
    groups = np.full(NCELLS, -1, np.int16)
    cx[:30], cy[:30] = np.linspace(60, 190, 30), np.linspace(190, 60, 30)    # inside P1 (bins 1-3)
    groups[:20], groups[20:27], groups[27:30] = 3, 7, 9                       # 67 %, 23 %, 10 % of area A's cells
    cx[30:], cy[30:], groups[30:] = 700.0, 500.0, 1                           # far from any area
    stats = np.zeros((7, NCELLS), np.uint32)
    stats[5], stats[6] = cx.view(np.uint32), cy.view(np.uint32)
    for f in frames:
        f["cells_file"] = f"frames/{f['seq']:06d}.cells.zst"
        (folder / "live" / f["cells_file"]).write_bytes(zstandard.ZstdCompressor().compress(stats.tobytes()))
    (folder / "live" / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    cells_bin = tmp_path / "000040.cells.bin"
    cells_bin.write_bytes(groups.tobytes() + np.zeros((NCELLS, 3), np.float32).tobytes())

    locked, generations = [], []

    class WithUmap(Scheduler):
        def generation(self, run_id):
            return 2                                         # the run's Clusters & UMAP settings changed twice

        def summary(self, run_id, seq, generation=None):
            generations.append(("summary", generation))
            if seq != 40:
                raise KeyError(seq)
            # Read without the run's lock: page requests and pushes never wait on these files.
            import threading
            lock, got = application.state.qc._state("run-groups").lock, []
            probe = threading.Thread(target=lambda: got.append(lock.acquire(timeout=1.0)) or (got[-1] and lock.release()))
            probe.start()
            probe.join()
            locked.append(not got[0])
            return {"clusters": [{"id": 3, "label": "ALB · APOA2 · CYP2E1",
                                  "top": [{"gene": "ALB"}, {"gene": "APOA2"}, {"gene": "CYP2E1"}]},
                                 {"id": 7, "label": "PTPRC", "top": [{"gene": "PTPRC"}]}]}

        def cells_path(self, run_id, seq, generation=None):
            generations.append(("cells", generation))
            if seq != 40:
                raise KeyError(seq)
            return cells_bin
    application = make(cfg)
    application.state.qc.analysis = WithUmap()
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-groups")
        deliver_start_bins(qc, "run-groups")
        for seq in range(30, 41):
            deliver_bins(qc, "run-groups", seq, bins_table(P1, start_hot=P1))
        with_umap = client.get("/api/v1/runs/run-groups/qc/40", headers=auth()).json()["areas"]
        without = client.get("/api/v1/runs/run-groups/qc/39", headers=auth()).json()["areas"]
    assert [a["L"] for a in with_umap] == ["A"] and with_umap[0]["clusters"] == [
        {"id": 3, "label": "ALB · APOA2"}, {"id": 7, "label": "PTPRC"}]               # 9 holds 10 %: left out
    assert [a["L"] for a in without] == ["A"] and "clusters" not in without[0]
    assert locked == [False], "the cluster files are read outside the run's lock"
    # Both files of one settings generation, the run's current one: never one's labels with another's names.
    assert ("summary", 2) in generations and ("cells", 2) in generations
    assert {generation for _, generation in generations} == {2}
    stored = json.loads((application.state.qc._state("run-groups").dir / "000040.qc.json").read_text())
    assert stored["areas"][0]["clusters"] and stored["ms"]["clusters"] >= 0


def test_a_restart_while_cells_form_keeps_the_start_lines_areas_on_screen(world):
    """An engine restart before the run's own areas are shown: the tracker resumes from state.json and the page is
    still shown the start line's areas (no event at the next result); at burn-in + 20 the run's area keeps its letter.
    A state.json from before the areas (no "areas" entry) starts the areas afresh."""
    cfg = area_world(world, "run-forming")
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-forming")
        deliver_start_bins(qc, "run-forming")
        for seq in range(1, 21):
            deliver_bins(qc, "run-forming", seq, bins_table(P1, start_hot=P1))
        folder = qc._state("run-forming").dir
    saved = json.loads((folder / "state.json").read_text())
    assert saved["last_live"] == 20 and saved["areas"]["revealed"] is False and saved["areas"]["drawn"]
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        url = "/api/v1/runs/run-forming/qc"
        deliver_bins(qc, "run-forming", 21, bins_table(P1, start_hot=P1))
        forming = client.get(f"{url}/21", headers=auth()).json()
        assert forming["area_view"] == "start" and [a["L"] for a in forming["areas"]] == ["A"]
        for seq in range(22, 31):
            deliver_bins(qc, "run-forming", seq, bins_table(P1, start_hot=P1))
        shown = client.get(f"{url}/30", headers=auth()).json()
        assert shown["area_view"] == "smoothed" and [a["L"] for a in shown["areas"]] == ["A"]
    saved = json.loads((folder / "state.json").read_text())
    (folder / "state.json").write_text(json.dumps({k: v for k, v in saved.items() if k != "areas"}))
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        assert qc._state("run-forming").tracker is None and qc._state("run-forming").ema == saved["ema"]
        # A fresh tracker decides nothing before 2 results over 2 s: meanwhile the areas the page showed last (drawn.jsonl)
        # stay, with this result's numbers, never blanked; then the tracker's own, with the same letters.
        for seq in range(31, 36):
            deliver_bins(qc, "run-forming", seq, bins_table(P1, start_hot=P1))
        url = "/api/v1/runs/run-forming/qc"
        first = client.get(f"{url}/31", headers=auth()).json()
        assert first["area_view"] == "smoothed" and first["areas_carried"] is True and [a["L"] for a in first["areas"]] == ["A"]
        warm = client.get(f"{url}/35", headers=auth()).json()
        assert warm["area_view"] == "smoothed" and "areas_carried" not in warm and [a["L"] for a in warm["areas"]] == ["A"]
        letters = {s: [a["L"] for a in client.get(f"{url}/{s}", headers=auth()).json()["areas"]] for s in range(31, 36)}
        assert all(found == ["A"] for found in letters.values())


def test_a_check_that_starts_late_waits_for_the_enter_rule(world):
    """A check that starts after the areas are shown (switched on, or a model that arrived late) commits nothing on
    one result: for the enter rule's 2 results over 2 s it shows no areas ("pending": the page says nothing about
    areas yet), then the areas that held, once, the start line's letters kept where its areas still are."""
    cfg = area_world(world, "run-late")
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-late")
        deliver_start_bins(qc, "run-late")
        url = "/api/v1/runs/run-late/qc"
        for seq in range(34, 41):                               # 0.5 s apart: warm at frame 38 (2 s after 34)
            deliver_bins(qc, "run-late", seq, bins_table(P1 + P2, start_hot=P1))
        got = {s: client.get(f"{url}/{s}", headers=auth()).json() for s in range(34, 41)}
        for s in range(34, 38):
            assert got[s]["single"] is False and got[s]["area_view"] == "pending" and got[s]["areas"] == [], s
        assert got[38]["area_view"] == "smoothed" and sorted(a["L"] for a in got[38]["areas"]) == ["A", "B"]
        assert {a["L"]: a["tag"] for a in got[38]["areas"]} == {"A": "already", "B": "rose"}   # A: the start line's place
        assert sorted(a["L"] for a in got[40]["areas"]) == ["A", "B"]


def test_pausing_on_the_newest_frame_keeps_the_live_letters_and_number(world):
    """Pause (or Step) shows the run's newest frame before a live pass scored it, so the page asks for it and the
    scheduler hands it over as a request (single). The newest frame of a live history continues that history: the same
    letters for the same places and the headline's margin, never a snapshot judged alone (letters by rank, the raw
    number): pressing Pause renames nothing and flips no number. A frame asked for behind the live edge is judged alone,
    its areas named as the page showed them there, its number with the margin against the live result before it."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-pause", sample, "2026-10-17T18:00:00+00:00", frames=51,
              times=[0.5 * s for s in range(51)], burnin=10)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-pause")
        deliver_start_bins(qc, "run-pause", hot=P1)                   # the start line: its area A at P1
        for seq in range(1, 50):                                      # P1 cools from 25 on, P2 is hot from then
            deliver_bins(qc, "run-pause", seq, bins_table(P1 if seq < 25 else P2, start_hot=P1))
        url = "/api/v1/runs/run-pause/qc"
        live = client.get(f"{url}/49", headers=auth()).json()
        places = {a["L"]: a["tag"] for a in live["areas"]}
        assert live["area_view"] == "smoothed" and "B" in places and live["shown"] == 12      # 474 of 3,840: 12.3
        # Paused on frame 50 (the newest), asked for: raw 13.05 in 100 (501 of 3,840) would read 13 alone.
        deliver_bins(qc, "run-pause", 50, bins_table(P2, start_hot=P1, hot_mixed=15), single=True)
        paused = client.get(f"{url}/50", headers=auth()).json()
        assert paused["single"] is False and paused["area_view"] == "smoothed", "it continues the live history"
        assert {a["L"]: a["tag"] for a in paused["areas"]} == places and paused["shown"] == 12
        assert paused["raw_per100"] == 13.0 and 12.3 < paused["per100"] < 12.8
        assert qc._state("run-pause").last_live == 50
        # Behind the live edge: judged alone, the page's letters for the same places, the margin against frame 39's:
        # raw 12.6 in 100 (483 of 3,840) reads 12 there (13 alone).
        deliver_bins(qc, "run-pause", 40, bins_table(P2, start_hot=P1, hot_mixed=13), single=True)
        behind = client.get(f"{url}/40", headers=auth()).json()
        assert behind["single"] is True and behind["area_view"] == "single"
        drawn_then = {a["L"] for a in client.get(f"{url}/39", headers=auth()).json()["areas"]}
        assert {a["L"] for a in behind["areas"]} <= drawn_then | {"C"} and "B" in {a["L"] for a in behind["areas"]}
        assert behind["shown"] == 12 and behind["raw_per100"] == 12.6
        assert qc._state("run-pause").last_live == 50, "a frame behind the live edge never enters the history"


def test_a_live_result_without_per_bin_sums_keeps_the_areas_on_screen(world):
    """A live result whose bins file could not be read keeps the areas the page shows (their numbers on the smoothed
    sums, "areas_carried") instead of blanking them for one result; the next result goes on as before."""
    cfg = area_world(world, "run-nobins")
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-nobins")
        deliver_start_bins(qc, "run-nobins")
        for seq in range(1, 38):
            deliver_bins(qc, "run-nobins", seq, bins_table(P1 + P2, start_hot=P1))
        url = "/api/v1/runs/run-nobins/qc"
        before = client.get(f"{url}/37", headers=auth()).json()
        deliver(qc, "run-nobins", 38, 3840, 500)                      # no per-bin sums this time
        lost = client.get(f"{url}/38", headers=auth()).json()
        assert lost["area_view"] == "smoothed" and lost["areas_carried"] is True
        assert [(a["L"], a["shape"]) for a in lost["areas"]] == [(a["L"], a["shape"]) for a in before["areas"]]
        assert lost["grid"]["planes"] == 2
        assert client.get(f"{url}/38/bins", headers=auth()).status_code == 200
        deliver_bins(qc, "run-nobins", 39, bins_table(P1 + P2, start_hot=P1))
        after = client.get(f"{url}/39", headers=auth()).json()
        assert "areas_carried" not in after and [a["L"] for a in after["areas"]] == [a["L"] for a in before["areas"]]


def test_area_rows_hold_steady(world):
    """Two near-equal areas keep their rows: one moves above the other only with 1.25 x its extra mixed cells (the
    order the summary lists them, which the cards read). No area carries a whole number of its own any more: that was
    for its chip on the page, which draws no area; state.json keeps the order only."""
    cfg = area_world(world, "run-steady", frames=60)
    P3 = [(r, c) for r in range(7, 10) for c in range(3, 6)]

    def table(a, b):
        t = bins_table(P2 + P3, start_hot=P1)
        mixed = np.full((NY_B, NX_B), 2.0)
        for cells, h in ((P2, a), (P3, b)):
            for r, c in cells:
                mixed[r, c] = h
        t["mixed"] = mixed.ravel()
        return t
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-steady")
        deliver_start_bins(qc, "run-steady")
        for seq in range(1, 60):                                      # each area's share swings 20.3 <-> 20.7
            deliver_bins(qc, "run-steady", seq, table(*((8.433, 7.233) if seq % 2 else (7.233, 8.433))))
        url = "/api/v1/runs/run-steady/qc"
        got = [client.get(f"{url}/{s}", headers=auth()).json() for s in range(40, 60)]
    rounded = [{a["L"]: round(a["per100"]) for a in g["areas"]} for g in got]
    assert len({tuple(sorted(r.items())) for r in rounded}) == 2, "their shares swap places every result"
    assert all(len(g["areas"]) == 2 and not any("shown" in a for a in g["areas"]) for g in got)
    assert len({tuple(a["L"] for a in g["areas"]) for g in got}) == 1, "the rows keep their order"
    letters = sorted(a["L"] for a in got[-1]["areas"])
    saved = json.loads((qc._state("run-steady").dir / "state.json").read_text())
    assert sorted(saved["area_order"]) == letters and "area_shown" not in saved


def test_a_torn_line_in_shapes_jsonl_costs_one_outline_once(world):
    """The engine killed while appending to shapes.jsonl leaves a torn last line: the next session cuts it off (the
    next outline appended is whole again) and remembers the outline it held as lost, so summaries naming it are served
    without it and never make the whole file be read again."""
    cfg = area_world(world, "run-torn")
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-torn")
        deliver_start_bins(qc, "run-torn")
        for seq in range(1, 40):
            deliver_bins(qc, "run-torn", seq, bins_table(P1 + (P2 if seq >= 31 else []), start_hot=P1))
        folder = qc._state("run-torn").dir
        lost = client.get("/api/v1/runs/run-torn/qc/39", headers=auth()).json()["areas"][-1]["shape"]
    lines = (folder / "shapes.jsonl").read_text().splitlines(keepends=True)
    keep = [line for line in lines if lost not in line]
    (folder / "shapes.jsonl").write_text("".join(keep) + '{"id":"%s","outline":[[1,2],[3' % lost)   # torn mid-line
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        url = "/api/v1/runs/run-torn/qc"
        served = client.get(f"{url}/39", headers=auth()).json()["areas"]
        assert {a["shape"]: a["outline"] for a in served}[lost] == []
        assert (folder / "shapes.jsonl").read_text().endswith("\n"), "the torn line is cut off"
        reads = []
        load = type(qc)._load_shapes
        qc._load_shapes = lambda state: (reads.append(1), load(state))
        for _ in range(3):
            client.get(f"{url}/39", headers=auth())
        assert reads == [], "a lost outline is not looked for again"
        for seq in range(40, 44):                                     # P2 drops: a new outline is appended, whole
            deliver_bins(qc, "run-torn", seq, bins_table(P1 + P2[:6], start_hot=P1))
    shapes = [json.loads(line) for line in (folder / "shapes.jsonl").read_text().splitlines()]
    assert all("id" in s and "outline" in s for s in shapes)


def test_a_new_watch_hears_only_errors_no_result_answered_and_every_result_is_named(app):
    """A page that watches a run again (reopened, reconnected) hears the start line, the newest results and only the
    errors no later result answered (an old failure would say "the graphics worker restarts" for good on a finished
    run). Within a watch, results past PUSHED_SEQS a tick are named in one "seqs" event, so the page knows they exist."""
    application, client, sample = app
    qc = application.state.qc
    ready_run(client)
    deliver_start(qc, "run-a", 40, 4)
    key = qc._state("run-a").key
    qc.qc_result({"kind": "error", "run": "run-a", "seq": 2, "model": key, "error": "worker exited"})
    for seq in (3, 4):
        deliver(qc, "run-a", seq, 40, 6)
    qc.qc_result({"kind": "error", "run": "run-a", "seq": 5, "model": key, "error": "worker exited"})
    events = qc.events({}, "run-a", 0.0)
    assert [e["seq"] for e in events if e.get("state") == "error"] == [5], "2 was answered by results 3 and 4"
    cursor = {}
    qc.events(cursor, "run-a", 0.0)
    for seq in range(5, 8):
        deliver(qc, "run-a", seq, 40, 6)
    named = qc.events(cursor, "run-a", 1.0)
    assert [e["seqs"] for e in named if e.get("state") == "seqs"] == [] and \
        [e["seq"] for e in named if e.get("state") == "ready"] == [5, 6, 7]
    application.state.runs.get("run-a").frames.extend(
        [{**application.state.runs.get("run-a").frames[-1], "seq": s, "iter": 10 * s} for s in range(8, 16)])
    for seq in range(8, 15):
        deliver(qc, "run-a", seq, 40, 6)
    named = qc.events(cursor, "run-a", 2.0)
    assert [e["seqs"] for e in named if e.get("state") == "seqs"] == [[8, 9, 10]]
    assert [e["seq"] for e in named if e.get("state") == "ready"] == [11, 12, 13, 14]


def test_areas_are_shown_from_the_first_formed_frame_plus_20_without_a_schedule(world):
    """Without the schedule's burn-in in info.json, areas are shown from the first frame past burn-in (upscale,
    sampling, recorded or hill-climb) + 20 iterations."""
    cfg, sample = world
    folder = write_run(Path(cfg["paths"]["data_dir"]), "run-climb", sample, "2026-10-17T18:00:00+00:00", frames=12)
    frames = [json.loads(line) for line in (folder / "live" / "frames.jsonl").read_text().splitlines()]
    for f in frames:                                              # burn-in to iteration 40, then hill-climb
        f["phase"] = "burnin" if f["seq"] < 5 else "hillclimb"
    (folder / "live" / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-climb")
        deliver_start_bins(qc, "run-climb")
        for seq in (6, 7):                                        # iterations 60 and 70
            deliver_bins(qc, "run-climb", seq, bins_table(P1, start_hot=P1))
        view = [client.get(f"/api/v1/runs/run-climb/qc/{s}", headers=auth()).json()["area_view"] for s in (6, 7)]
        assert view[0] == "start" and view[1] != "start"


def test_runs_from_nuclei_have_no_start_areas_and_no_tags(app):
    """A start with nuclei (nuclei only, or cells with their nuclei) has no start line: no start areas (none served
    while cells form either), its areas are compared with no start whatever per-bin sums a result carries (no "rose" or
    "already", no start share or depth, no comparison of an area's own), and the worker's whole-run pairs make no
    "rose" for the whole tissue. A run from cells alone pairs its areas with each result's own start sums only, never
    with the start line's bins: a result without them tags nothing against the start."""
    application, client, sample = app
    qc = application.state.qc
    for run_id in ("run-nuclei", "run-paired"):
        ready_run(client, run_id)
        deliver_start_bins(qc, run_id)
        assert client.get(f"/api/v1/runs/{run_id}/qc/start/bins", headers=auth()).status_code == 404
        early = deliver_counts(qc, run_id, 3, bins=matched_bins(bins_table(P1, start_hot=P1)), judged=40, mixed=4,
                               ncells=NCELLS)                 # iteration 30: cells still form until 50
        assert early["area_view"] == "start" and early["areas"] == [], run_id
        assert qc._start_area_public(qc._state(run_id)) == {"areas": [], "grid": None}
        for seq in (5, 6, 7):                                 # frames carry no time here: a second apart
            deliver_bins(qc, run_id, seq, matched_bins(bins_table(P1, start_hot=P1)))
        summary = client.get(f"/api/v1/runs/{run_id}/qc/7", headers=auth()).json()
        assert [(a["L"], a["tag"], a["start_per100"]) for a in summary["areas"]] == [("A", "similar", None)], run_id
        assert summary["areas"][0]["local"]["paired"] is None and summary["areas"][0]["depth_start"] is None
    # The worker's whole-run pairs (a result without per-bin sums) rose clearly: R2's "rose" for a run from cells
    # alone, none for one whose start has nuclei.
    ready_run(client, "run-b")
    pairs = {"paired": {"n": 900, "n10": 200, "n01": 20, "mixed_now": 300, "mixed_start": 100}}
    for run_id, rose in (("run-b", True), ("run-paired", False)):
        run, state = application.state.runs.get(run_id), qc._state(run_id)
        assert qc._whole(run, state, None, pairs)["rose"] is rose, run_id
    deliver_start_bins(qc, "run-b")                           # from cells alone, its start line hot where P1 is
    for seq in (5, 6, 7):                                     # P1 hot now too, but no start sums of its own
        deliver_bins(qc, "run-b", seq, bins_table(P1))
    own = client.get("/api/v1/runs/run-b/qc/7", headers=auth()).json()
    assert [(a["L"], a["tag"], a["start_per100"]) for a in own["areas"]] == [("A", "similar", None)]
    deliver(qc, "run-a", 3, 40, 4)                            # no per-bin sums (an older worker): no areas
    plain = client.get("/api/v1/runs/run-a/qc/3", headers=auth()).json()
    assert plain["areas"] == [] and plain["area_view"] == "none" and plain["grid"] is None
    assert client.get("/api/v1/runs/run-a/qc/3/bins", headers=auth()).status_code == 404


# ---- what a summary rests on, and every comparison on the depth-matched change ---------------------------------------------
def deliver_counts(qc, run_id: str, seq: int, single=False, bins=None, **summary):
    """A frame's result with the worker's full summary (`summary`: judged, mixed, ncells, empty, under20, under100,
    tx_sum, matched, ...)."""
    state = qc._state(run_id)
    frame = qc.runs.get(run_id).frames[seq]
    qc.qc_result({"kind": "frame", "run": run_id, "seq": seq, "model": state.key, "dir": str(state.dir),
                  "source": "request" if single else "live", "single": single,
                  "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                  "summary": {"kind": "frame", "seq": seq, "median_tx": 300.0, **summary}, "bins": bins, "cells": None,
                  "start": None})
    return qc.summary(run_id, seq)


def matched(paired, now, start):
    """The worker's like-for-like change (`matched`), deterministic: no spread, no seeds, all paired cells used."""
    return {"paired": paired, "used": paired, "per100_now": now, "per100_start": start, "change": now - start,
            "sd": None, "seeds": 0}


def deliver_sampled(qc, run_id: str, seq: int, own: dict, averaged: dict, single=False, bins=None, **summary):
    """A frame's result read on the count proseg saves (f: counted for a pause, the hold, a finish or the end) as the
    scheduler hands it over, its record's count "sampled": f's own numbers (`own`: judged, mixed, tx_sum, median_tx),
    the frame's averaged count beside them (`averaged`, e: what the live path reads) and the rest on e as the worker
    gives it (`summary`: matched, start_block; the per-bin sums `bins`)."""
    state = qc._state(run_id)
    frame = qc.runs.get(run_id).frames[seq]
    qc.qc_result({"kind": "frame", "run": run_id, "seq": seq, "model": state.key, "dir": str(state.dir),
                  "source": "request" if single else "live", "single": single, "count": "sampled",
                  "frame": {k: frame.get(k) for k in ("seq", "iter", "phase", "t", "live", "voxel")},
                  "summary": {"kind": "frame", "seq": seq, "median_tx": 300.0, "basis": "proseg", "count": "sampled",
                              **summary, **own, "averaged": averaged},
                  "bins": bins, "cells": None, "start": None})
    return qc.summary(run_id, seq)


def test_every_summary_says_what_it_rests_on_and_compares_on_the_matched_change(world):
    """Coverage over all the run's cells (judged, empty, under 20, 20-99), retention against the frame's own start
    block, the worker's depth-matched change as it gave it, the comparison with the start (the raw change
    unrounded, against the start block; the arrow's direction on the smoothed matched change against the size-aware
    threshold) and the model's provenance in every summary; the start line's coverage in describe and GET /qc/start;
    series points carry the matched change."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-cov", sample, "2026-10-19T08:00:00+00:00", frames=40,
              times=[float(s) for s in range(40)], burnin=0)        # compared with the start from iteration 20
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-cov")
        state = qc._state("run-cov")
        state.dir.mkdir(parents=True, exist_ok=True)
        qc.qc_result({"kind": "start", "run": "run-cov", "seq": None, "model": state.key, "dir": str(state.dir),
                      "source": "start", "single": True, "bins": None, "cells": None, "start": None,
                      "summary": {"kind": "start", "judged": 3200, "mixed": 369, "median_tx": 380.0, "ncells": 3439,
                                  "empty": 0, "under20": 20, "under100": 219, "tx_sum": 1200000}})
        url = "/api/v1/runs/run-cov/qc"
        start = client.get(url + "/start", headers=auth()).json()
        assert start["coverage"] == {"cells": 3439, "judged": 3200, "empty": 0, "under20": 20, "under100": 219,
                                     "judged_share": round(3200 / 3439, 4), "enough": True}
        assert start["tx"] == 1200000 and client.get(url, headers=auth()).json()["start"]["coverage"]["judged"] == 3200
        block = {"per100": 11.53, "judged": 3200, "mixed": 369, "tx_sum": 1200000.0, "median_tx": 380.0, "iteration": 0}
        got = {s: deliver_counts(qc, "run-cov", s, judged=3309, mixed=378, ncells=3439, empty=0, under20=5, under100=125,
                                 tx_sum=1476000, matched=matched(3170, 10.5, 11.6), start_block=block)
               for s in range(23, 30)}
        view = client.get(url, headers=auth()).json()
        series = client.get(url + "/series", headers=auth()).json()["points"]
    last = got[29]
    assert last["coverage"] == {"cells": 3439, "judged": 3309, "empty": 0, "under20": 5, "under100": 125,
                                "judged_share": round(3309 / 3439, 4), "enough": True}
    assert sum(last["coverage"][k] for k in ("judged", "empty", "under20", "under100")) == last["coverage"]["cells"]
    assert last["retention"] == {"ratio": 1.23, "start_tx": 1200000, "tx": 1476000}
    assert last["matched"] == matched(3170, 10.5, 11.6) | {"change": pytest.approx(-1.1)}
    change = last["change"]
    assert change["raw"] == pytest.approx(100 * 378 / 3309 - 100 * 369 / 3200, abs=1e-4)      # unrounded
    assert change["matched"] == pytest.approx(-1.1, abs=1e-3) and change["threshold"] == 1.0  # a region of 3,439 cells
    # The model's own threshold, a region's: below 2.3 in 100 two learnings can read it with opposite signs.
    assert change["direction"] == "fewer" and change["meaningful"] and change["model_threshold"] == 2.3
    assert "prev_change" not in last and last["provenance"]["table"] == "xenium_table"
    assert view["model"]["provenance"] == application.state.qc_models.describe(sample["id"])["model"]["provenance"]
    assert view["model"]["deep"] is True and view["detail"]["state"] == "ready"
    assert [p["matched_change"] for p in series] == [got[s]["change"]["matched"] for s in range(23, 30)]
    assert got[23]["change"]["matched"] == pytest.approx(-1.1, abs=1e-3)             # the first live result: its own
    logged = application.state.qc.log_info("run-cov")                               # for the Log button
    assert logged["seq"] == 29 and logged["change"] == change and logged["coverage"] == last["coverage"]
    assert logged["retention"] == last["retention"] and logged["constants"] == last["constants"]
    assert last["start"]["per100"] == 11.5                              # the start line shown: qc_start's, fixed
    assert logged["provenance"]["table"] == "xenium_table" and application.state.qc.log_info("run-b") is None


def test_a_number_resting_on_few_cells_or_transcript_loss_gets_no_arrow(world):
    """Three cells, two empty, one judged: the share is 0 in 100, but coverage says one in three was judged
    and there is no verdict (too few cells). Every cell thinned to half (a whole sample, no boundary moved): the raw
    share falls 12.85 -> 9.71 in 100, the matched change reads +0.14: "about the same". A real fall in the matched
    change is "fewer"; while cells form nothing is compared."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-thin", sample, "2026-10-19T09:00:00+00:00", frames=40,
              times=[float(s) for s in range(40)], burnin=10)       # compared with the start from iteration 30
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-thin")
        deliver_start(qc, "run-thin", 40, 4)
        few = deliver_counts(qc, "run-thin", 33, single=True, judged=1, mixed=0, ncells=3, empty=2, under20=0, under100=0,
                             tx_sum=200, matched=matched(1, 0.0, 0.0))
        assert few["per100"] == 0.0 and few["coverage"]["judged_share"] == 0.3333 and few["coverage"]["enough"] is False
        assert few["change"]["direction"] is None and few["change"]["enough"] is False
        assert few["retention"] is None                          # the start line counted no transcripts: nothing to compare
        application.state.runs.get("run-thin").info["cells"] = 41190            # a whole sample's worth of cells
        thin = deliver_counts(qc, "run-thin", 34, single=True, judged=30868, mixed=2996, ncells=41190, empty=121,
                              under20=1371, under100=8830, tx_sum=6000000,
                              matched=matched(32384, 9.84, 9.70))           # +0.14 at the same depth
        assert thin["change"]["direction"] == "same" and not thin["change"]["meaningful"] and thin["change"]["threshold"] == 0.3
        real = deliver_counts(qc, "run-thin", 35, single=True, judged=37219, mixed=4918, ncells=40070, empty=1, under20=275,
                              under100=2575, tx_sum=12000000, matched=matched(35999, 11.42, 13.26))
        assert real["change"]["direction"] == "fewer" and real["change"]["meaningful"]
        forming = deliver_counts(qc, "run-thin", 12, single=True, judged=30000, mixed=1000, ncells=40070,
                                 matched=matched(30000, 3.3, 13.0))
        assert forming["change"]["direction"] is None and forming["change"]["forming"] is True
        none = deliver_counts(qc, "run-thin", 37, single=True, judged=3000, mixed=300, ncells=3439,
                              matched={"paired": 5, "used": 0.0, "per100_now": None, "per100_start": None,
                                       "change": None, "sd": None, "seeds": 0})       # paired, none still judged
        assert none["matched"] == {"paired": 5, "used": 0.0, "per100_now": None, "per100_start": None, "change": None,
                                   "sd": None, "seeds": 0}                            # as the worker gave it
        assert none["change"]["matched"] is None and none["change"]["direction"] is None
        used = deliver_counts(qc, "run-thin", 38, single=True, judged=3000, mixed=300, ncells=3439,
                              matched={**matched(3100, 9.0, 11.0), "used": 380.0})    # 380 compared: too few to say
        assert used["change"]["matched"] == pytest.approx(-2.0) and used["change"]["enough"] is False
        old = deliver_counts(qc, "run-thin", 36, single=True, judged=3000, mixed=300, ncells=3439)   # an older worker
        assert old["matched"] is None and old["change"]["matched"] is None and old["change"]["direction"] is None
        assert old["coverage"] == {"cells": 3439, "judged": 3000, "empty": None, "under20": None, "under100": None,
                                   "judged_share": round(3000 / 3439, 4), "enough": True}


def matched_bins(table, rise=None):
    """Per-bin sums with the worker's depth-matched sums: the paired cells as mixed now and at the start, or with
    `rise` {bin: extra mixed now} at the matched depth."""
    out = {**table, "matched_paired": np.asarray(table["paired"], np.float32),
           "matched_mixed_now": np.asarray(table["paired_mixed_start"], np.float32).copy(),
           "matched_mixed_start": np.asarray(table["paired_mixed_start"], np.float32)}
    for b, extra in (rise or {}).items():
        out["matched_mixed_now"][b] += extra
    return out


def test_an_area_got_more_mixed_only_by_its_own_depth_matched_change(world):
    """For areas: an area whose paired cells look more mixed only because they hold more transcripts
    (its matched change none) is not "got more mixed" (tag "already", where the start line already flagged it: its R1
    card says the cause is not known); the same area with the rise at the same depth is "rose". Each area carries its
    own comparison (judged, paired, matched change, its own threshold, enough, direction) and whether it touches the
    tissue's edge; a region's border is a cut, not an edge."""
    cfg = area_world(world, "run-own", frames=60)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-own")
        deliver_start_bins(qc, "run-own", hot=P1)
        hot = [r * NX_B + c for r, c in P2]
        for seq in range(1, 40):                                  # P2 rose in the paired cells' plain counts only
            table = bins_table(P1 + (P2 if seq >= 31 else []), start_hot=P1)
            deliver_bins(qc, "run-own", seq, matched_bins(table))
        flat = {a["L"]: a for a in qc.summary("run-own", 39)["areas"]}
        for seq in range(40, 60):                                 # now also at the same depth
            table = bins_table(P1 + P2, start_hot=P1)
            deliver_bins(qc, "run-own", seq, matched_bins(table, rise={b: 10.0 for b in hot}))
        rose = {a["L"]: a for a in qc.summary("run-own", 59)["areas"]}
    b = next(L for L, a in rose.items() if a["start_per100"] is not None and a["per100"] > a["start_per100"] + 10)
    assert flat[b]["tag"] != "rose" and flat[b]["local"]["matched_change"] == 0.0 and flat[b]["local"]["direction"] == "same"
    assert rose[b]["tag"] == "rose" and rose[b]["local"]["direction"] == "more" and rose[b]["local"]["enough"]
    local = rose[b]["local"]
    assert local["paired"] >= 400 and local["judged"] == rose[b]["judged"]
    share = (local["matched_change"] + rose[b]["start_per100"]) / 100       # its matched share now (paired = judged)
    assert local["threshold"] == pytest.approx(R.local_threshold(share, local["paired"]), abs=0.02)
    assert set(rose[b]["local"]) == {"judged", "paired", "matched_change", "threshold", "enough", "direction"}
    assert all(a["edge"] is False for a in rose.values())       # a box region: its border is a cut through tissue


def test_areas_at_the_tissues_edge_are_listed_after_the_others_unless_they_rose(world):
    """Windows at the tissue's edge look more mixed by themselves (18 % vs 12 % at the start of a whole
    sample): of two areas mixed from the start, the one at the edge (empty background beside it) is listed after the
    one inside, even with more extra mixed cells; its card stays (the cause is not known), edge True."""
    cfg = area_world(world, "run-edge")
    P3 = [(r, c) for r in range(7, 10) for c in range(3, 6)]
    application = make(cfg)

    def table(hot_mixed_p2, hot_mixed_p3):
        t = bins_table(P2 + P3, start_hot=P2 + P3)
        empty = [r * NX_B + c for r in range(5, 12) for c in range(13, 16)]      # background right of P2
        for name in ("cells", "judged", "mixed", "paired", "paired_mixed_now", "paired_mixed_start", "n10", "n01",
                     "tx", "tx_start", "judged_start", "mixed_start"):
            t[name] = np.asarray(t[name]).copy()
            t[name][empty] = 0
        for hot, cells in ((hot_mixed_p2, P2), (hot_mixed_p3, P3)):
            for (r, c) in cells:
                for name in ("mixed", "paired_mixed_now", "paired_mixed_start", "mixed_start"):
                    t[name][r * NX_B + c] = hot
        return matched_bins(t)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-edge")
        state = qc._state("run-edge")
        state.dir.mkdir(parents=True, exist_ok=True)
        start = table(15, 12)
        np.savez(state.dir / "start.bins.npz", **start)
        qc.qc_result({"kind": "start", "run": "run-edge", "seq": None, "model": state.key, "dir": str(state.dir),
                      "source": "start", "single": True, "bins": start, "cells": None, "start": None,
                      "summary": {"kind": "start", "judged": int(start["judged"].sum()), "mixed": int(start["mixed"].sum()),
                                  "median_tx": 386.0, "ncells": NCELLS}})
        for seq in range(1, 45):
            deliver_bins(qc, "run-edge", seq, table(15, 12))      # the edge area (P2) has more extra mixed cells
        areas = qc.summary("run-edge", 44)["areas"]
        starting = qc.start_info("run-edge")["areas"]               # the start line's own areas: listed the same way
        deliver_bins(qc, "run-edge", 40, table(15, 12), single=True)
        single = qc.summary("run-edge", 40)["areas"]                # a snapshot judged alone too
    assert [a["edge"] for a in starting] == [False, True] and [a["edge"] for a in single] == [False, True]
    by = {a["L"]: a for a in areas}
    assert len(areas) == 2 and {a["tag"] for a in areas} == {"already"}
    edge = next(a for a in areas if a["edge"])
    inside = next(a for a in areas if not a["edge"])
    assert areas[0] is inside or areas[0]["L"] == inside["L"]
    assert edge["per100"] > inside["per100"] and by[edge["L"]]["edge"] is True


def test_the_matched_smoothing_resumes_after_an_engine_restart(world):
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-ema", sample, "2026-10-19T10:00:00+00:00", frames=12,
              times=[float(s) for s in range(12)], burnin=2)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-ema")
        deliver_start(qc, "run-ema", 40, 4)
        for seq in (3, 4):
            deliver_counts(qc, "run-ema", seq, judged=4000, mixed=400, ncells=4500, matched=matched(4000, 10.0, 10.0))
        folder = qc._state("run-ema").dir
    saved = json.loads((folder / "state.json").read_text())
    assert saved["ema_matched"] == pytest.approx([4000.0, 400.0, 400.0])
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-ema")
        after = deliver_counts(qc, "run-ema", 5, judged=4000, mixed=400, ncells=4500, matched=matched(4000, 8.0, 10.0))
    weight = 1 - (1 - K.value("region", "headline_weight")) ** 1.0      # 1 s after the last live result: 0.2
    assert after["change"]["matched"] == pytest.approx(-2.0 * weight, abs=1e-3)   # smoothed on from before


def test_the_change_carries_the_two_numbers_it_compares(world):
    """Every change block says the two numbers its matched change compares (per100_now, per100_start: of the cells
    compared now and at the start, how many per 100 look mixed, at the matched depth), from the same counts: a live
    result's from the smoothed like-for-like counts (ema_matched), as its arrow is; its own change (`own.change`) and a
    frame judged alone from the frame's own; matched is their difference (each kept to 4 decimals); none without a
    matched change."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-two", sample, "2026-10-19T10:00:00+00:00", frames=12,
              times=[float(s) for s in range(12)], burnin=2)
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-two")
        deliver_start(qc, "run-two", 40, 4)
        first = deliver_counts(qc, "run-two", 3, judged=4000, mixed=400, ncells=4500, matched=matched(4000, 10.0, 12.0))
        second = deliver_counts(qc, "run-two", 4, judged=4000, mixed=400, ncells=4500, matched=matched(4000, 7.0, 12.5))
        ema = list(qc._state("run-two").ema_matched)
        alone = deliver_counts(qc, "run-two", 6, single=True, judged=4000, mixed=400, ncells=4500,
                               matched=matched(4000, 9.25, 11.75))
        none = deliver_counts(qc, "run-two", 7, single=True, judged=4000, mixed=400, ncells=4500)
    numbers = lambda change: (change["per100_now"], change["per100_start"], change["matched"])  # noqa: E731
    assert numbers(first["change"]) == numbers(first["own"]["change"]) == (10.0, 12.0, -2.0)    # the first: its own
    smoothed = (100.0 * ema[1] / ema[0], 100.0 * ema[2] / ema[0])
    assert ema[0] == 4000.0 and smoothed[0] < 10.0 and smoothed != (7.0, 12.5)
    live = second["change"]
    assert (live["per100_now"], live["per100_start"]) == (round(smoothed[0], 4), round(smoothed[1], 4))
    assert live["matched"] == pytest.approx(live["per100_now"] - live["per100_start"], abs=2e-4)
    assert numbers(second["own"]["change"]) == (7.0, 12.5, -5.5)                 # the frame's own, not smoothed
    assert numbers(alone["change"]) == numbers(alone["own"]["change"]) == (9.25, 11.75, -2.5)
    assert numbers(none["change"]) == numbers(none["own"]["change"]) == (None, None, None)


# ---- a frame read on the count proseg saves (f) ---------------------------------------------------------------------
F_OWN = {"judged": 10000, "mixed": 879, "tx_sum": 650000.0, "median_tx": 320.0}      # 8.79 in 100 on f (8.8 shown)
E_SUMS = {"judged": 10000, "mixed": 800, "tx_sum": 600000.0, "median_tx": 300.0}     # 8 in 100 on e
# 8.46 in 100 on f (8.5 at one decimal: its whole number 8, rounded once), fewer judged cells than a verdict needs (300)
F_FEW = {"judged": 260, "mixed": 22, "tx_sum": 17000.0, "median_tx": 320.0}


def twin_runs(world, *run_ids, frames=45):
    """Runs on disk whose frame s is iteration s, 0.5 s apart, burn-in 10, their knobs on every frame (areas and cards
    from iteration 30), each fed the same live history: the start line's areas at P1, then frames 1 .. 39 at 8 in
    100 (e), fewer mixed like for like than at their own start blocks (10 in 100), P1 and P2 hot, P2's cells grown.
    Each run grows as its results come (its frames up to the newest result) and samples; returns (application,
    client, the per-bin sums fed, the start block, {run: its frames})."""
    from test_qc_trials import around, growth_table, knob_run, with_matched
    cfg = None
    for run_id in run_ids:
        cfg = knob_run(world, run_id, frames)
    application = make(cfg)
    return application, with_matched(growth_table(P1 + P2, P1, grown=around(P2))), block(10000, 1000, 800000.0)


def grow(application, run_id: str, frames: list, seq: int):
    """The run's frames up to `seq`: it is the newest (the run's live edge)."""
    application.state.runs.get(run_id).frames = frames[:seq + 1]


def live_history(application, client, run_id: str, table, start) -> list:
    qc = application.state.qc
    ready_run(client, run_id)
    run = application.state.runs.get(run_id)
    frames = list(run.frames)
    run.meta["state"] = "running"
    deliver_start_bins(qc, run_id)
    for s in range(1, 40):
        grow(application, run_id, frames, s)
        deliver_counts(qc, run_id, s, bins=table, matched=matched(10000, 8.0, 10.0), start_block=start, **E_SUMS)
    return frames


def test_a_frame_read_on_the_count_proseg_saves_shows_its_own_numbers_and_the_rest_reads_e(world):
    """Decision A: a frame proseg counted for a pause, the hold, a finish or the end is read on f, the count it saves
    (its record's count "sampled"). Its summary shows f's own numbers: per100 the raw share and its whole number with
    no margin (8.79 after a live 8 reads 9), single, and its own change, on e against its own start block, neither
    smoothed nor held (the word held before it let go: it is the run's newest frame). The live path reads its averaged
    count exactly as an averaged live result of the same bins would: the smoothing, the area tracker's sums, the
    areas, the card and retention (e's RNA) are those of its averaged twin. Every summary carries the frame's own
    numbers (`own`)."""
    application, table, start = twin_runs(world, "run-f", "run-e")
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        frames = {run_id: live_history(application, client, run_id, table, start) for run_id in ("run-f", "run-e")}
        before = qc.summary("run-f", 39)
        for run_id in frames:                                 # frame 40, the newest: like for like now more mixed
            grow(application, run_id, frames[run_id], 40)
        turned = matched(10000, 12.0, 10.0)
        f40 = deliver_sampled(qc, "run-f", 40, F_OWN, E_SUMS, bins=table, matched=turned, start_block=start)
        e40 = deliver_counts(qc, "run-e", 40, bins=table, matched=turned, start_block=start, **E_SUMS)
        states = {run_id: qc._state(run_id) for run_id in frames}
        sums = {run_id: {k: np.asarray(v).tolist() for k, v in states[run_id].tracker.sm.items()} for run_id in states}
        released = dict(states["run-f"].steady.verdicts["change"])
        for run_id in frames:
            application.state.runs.get(run_id).meta["state"] = "done"
    assert before["shown"] == 8 and before["own"] == {"per100": 8.0, "shown": 8, "change": before["own"]["change"]}
    assert (f40["count"], f40["single"], f40["judged"], f40["mixed"]) == ("sampled", True, 10000, 879)
    assert f40["per100"] == f40["raw_per100"] == 8.8 and f40["shown"] == 9, "its own: no margin against the live 8"
    assert (e40["count"], e40["single"], e40["shown"]) == ("averaged", False, 8)
    assert f40["own"] == {"per100": 8.79, "shown": 9, "change": f40["change"]} and e40["own"]["per100"] == 8.0
    own = f40["change"]
    assert own["raw"] == pytest.approx(8.0 - 10.0) and own["matched"] == pytest.approx(2.0) and "held" not in own
    assert own["direction"] == "more" and own["model_threshold"] == K.value("region", "matched_model")
    assert e40["own"]["change"] == own and e40["change"]["direction"] == "fewer"     # its twin's live word: smoothed
    assert released == {"shown": "more", "pending": None, "since": None, "count": 0, "differs": None,
                        "numbers": {"per100_now": 12.0, "per100_start": 10.0}}       # its own word, its numbers
    # The live path read e: its averaged twin's smoothing, areas' sums, areas, cards and retention.
    for key in ("ema", "ema_matched", "ema_start", "shown", "last_live"):
        assert getattr(states["run-f"], key) == getattr(states["run-e"], key), key
    assert sums["run-f"] == sums["run-e"] and states["run-f"].last_live == 40
    assert f40["areas"] and f40["suggestion"] is not None
    for key in ("areas", "area_view", "suggestion", "more_ideas", "trial", "retention"):
        assert f40.get(key) == e40.get(key), key
    assert f40["retention"] == {"ratio": 0.75, "start_tx": 800000, "tx": 600000}


def test_a_frame_counted_again_keeps_its_areas_and_cards_and_shows_its_own_numbers(world):
    """A token's count of the newest frame (a request line: averaged, live, with its areas and a card), then the
    hold's count of the same frame (its line now holds f): the hold's summary shows f's own numbers (its whole number
    rounded once: 8.46 reads 8), and the areas, area view, card, other ideas and trial the frame had (they read e,
    which a count again does not change: the card shown before the hold line is still shown after it), its
    <seq>.qc.grid as it was; its change is e's against its own start block, with e's support (f judges fewer cells
    here than a verdict needs, e enough); the smoothing and the areas' sums are left alone, and the page hears of the
    frame again (its count changed)."""
    application, table, start = twin_runs(world, "run-hold")
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        frames = live_history(application, client, "run-hold", table, start)
        grow(application, "run-hold", frames, 40)
        asked = deliver_counts(qc, "run-hold", 40, bins=table, matched=matched(10000, 8.0, 10.0), start_block=start,
                               **E_SUMS)
        state = qc._state("run-hold")
        grid = (state.dir / "000040.qc.grid").read_bytes()
        kept = (list(state.ema), list(state.ema_matched), state.last_live, state.live_count,
                {k: np.asarray(v).tolist() for k, v in state.tracker.sm.items()})
        cursor = {}
        qc.events(cursor, "run-hold", 0.0)
        held = deliver_sampled(qc, "run-hold", 40, F_FEW, E_SUMS, bins=table, matched=matched(10000, 8.0, 10.0),
                               start_block=start)
        pushed = qc.events(cursor, "run-hold", 1.0)
        points = client.get("/api/v1/runs/run-hold/qc/series", headers=auth()).json()["points"]
        after = (list(state.ema), list(state.ema_matched), state.last_live, state.live_count,
                 {k: np.asarray(v).tolist() for k, v in state.tracker.sm.items()})
        application.state.runs.get("run-hold").meta["state"] = "done"
    assert asked["count"] == "averaged" and asked["suggestion"] is not None and asked["areas"]
    assert (held["count"], held["single"], held["per100"], held["shown"], held["mixed"]) == \
        ("sampled", True, 8.5, 8, 22)
    assert held["own"] == {"per100": 8.4615, "shown": 8, "change": held["change"]}
    change = held["change"]                                   # e's share (8 in 100) less its start block's (10)
    assert change["raw"] == pytest.approx(8.0 - 10.0) and change["matched"] == pytest.approx(-2.0)
    assert "held" not in change
    assert change["enough"] is True and change["direction"] == "fewer", "e's 10,000 judged cells, not f's 260"
    for key in ("areas", "area_view", "suggestion", "more_ideas", "trial"):
        assert held.get(key) == asked.get(key), key
    assert (state.dir / "000040.qc.grid").read_bytes() == grid and after == kept
    assert [(e["seq"], e["summary"]["count"]) for e in pushed if e.get("state") == "ready"] == [(40, "sampled")]
    assert [(p["seq"], p["count"]) for p in points[-2:]] == [(39, "averaged"), (40, "sampled")]


def test_an_older_frame_read_on_the_count_proseg_saves_leaves_the_live_word_alone(world):
    """A frame read on f that is not the run's newest (a pause's count scored once the run went on: backfill, or a
    frame asked for while scrubbing back) shows its own numbers and its own change, while the word the live results
    hold stays as it is: only the run's newest frame lets it go (the run rests there)."""
    application, table, start = twin_runs(world, "run-old")
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-old")
        run = application.state.runs.get("run-old")
        frames = list(run.frames)
        run.meta["state"] = "running"
        deliver_start_bins(qc, "run-old")
        for s in [*range(1, 33), *range(34, 40)]:            # frame 33: a pause's count, scored after the run went on
            grow(application, "run-old", frames, s)
            deliver_counts(qc, "run-old", s, bins=table, matched=matched(10000, 8.0, 10.0), start_block=start,
                           **E_SUMS)
        state = qc._state("run-old")
        word = dict(state.steady.verdicts["change"])
        old = deliver_sampled(qc, "run-old", 33, F_OWN, E_SUMS, bins=table, matched=matched(10000, 12.0, 10.0),
                              start_block=start)
        after = dict(state.steady.verdicts["change"])
        run.meta["state"] = "done"
    assert word["shown"] == "fewer" and after == word, "the run plays on past this frame: its live word stays"
    assert (old["count"], old["single"], old["shown"]) == ("sampled", True, 9)
    assert old["change"]["direction"] == "more" and "held" not in old["change"]
    assert old["own"]["change"] == old["change"]


def test_a_frame_read_on_f_without_its_averaged_count_is_never_live(world):
    """A result on f that carries no averaged count beside it has nothing the live path could smooth with: even as the
    run's newest frame it is judged alone, and the smoothing, the shown integer and the live history stay as they
    were (f never goes down the live path)."""
    application, table, start = twin_runs(world, "run-bare")
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        frames = live_history(application, client, "run-bare", table, start)
        grow(application, "run-bare", frames, 40)
        state = qc._state("run-bare")
        kept = (list(state.ema), list(state.ema_matched), state.shown, state.last_live, state.live_count)
        bare = deliver_sampled(qc, "run-bare", 40, F_OWN, None, bins=table, matched=matched(10000, 8.0, 10.0),
                               start_block=start)
        after = (list(state.ema), list(state.ema_matched), state.shown, state.last_live, state.live_count)
        application.state.runs.get("run-bare").meta["state"] = "done"
    assert after == kept and (bare["count"], bare["single"], bare["shown"]) == ("sampled", True, 9)


def test_every_whole_number_is_rounded_once_from_the_unrounded_share(world):
    """The headline's whole number is rounded once, from the unrounded share, never from per100's one decimal: live,
    a smoothed share of 8.77 after a shown 8 stays 8 (not yet 0.3 past 8.5; its per100, 8.8, is); a frame judged alone
    with no live result before it reads 8 at 8.46 (per100 8.5), and so does its own whole number."""
    cfg, sample = world
    data = Path(cfg["paths"]["data_dir"])
    write_run(data, "run-r", sample, "2026-10-17T15:00:00+00:00", frames=3, times=[0.0, 1.0, 2.0])
    write_run(data, "run-s", sample, "2026-10-17T15:30:00+00:00", frames=3, times=[0.0, 1.0, 2.0])
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-r")
        ready_run(client, "run-s")
        weight = 1 - (1 - K.value("region", "headline_weight")) ** (1.0 / K.value("region", "headline_ref_s"))
        mixed = round((8.0 + 0.77 / weight) * 100)                # 1 s after the shown 8: the smoothed share 8.77
        deliver(qc, "run-r", 0, 10000, 800)
        deliver(qc, "run-r", 1, 10000, mixed)
        live = qc.summary("run-r", 1)
        deliver(qc, "run-s", 2, 5000, 423, single=True)
        alone = qc.summary("run-s", 2)
    smoothed = 8.0 * (1 - weight) + mixed / 100 * weight
    assert 8.75 < smoothed < 8.8 and (live["single"], live["per100"], live["shown"]) == (False, 8.8, 8)
    assert (alone["single"], alone["per100"], alone["shown"]) == (True, 8.5, 8)
    assert (alone["own"]["per100"], alone["own"]["shown"]) == (8.46, 8)


def test_a_frames_bytes_are_served_on_the_count_it_is_read_on(app):
    """GET /qc/{seq}/cells?count=: "sampled" serves <seq>.qc.bin while the frame's summary is on f; "averaged" its
    <seq>.qc.bin while the summary is averaged and <seq>.qc.averaged.bin (the worker's bytes on e) while it is sampled;
    any other count, or that file missing, 404 "This iteration's count changed" (the caller asks the check again).
    Cached for good only with both the model and the count named."""
    import zstandard
    application, client, sample = app
    qc = application.state.qc
    ready_run(client)
    model = qc._state("run-a").key
    url = "/api/v1/runs/run-a/qc"
    averaged = deliver(qc, "run-a", 3, 30, 3)
    folder = qc._state("run-a").dir
    on_f, on_e = np.full(NCELLS, 20, np.uint8), np.full(NCELLS, 200, np.uint8)
    (folder / "000004.qc.bin").write_bytes(zstandard.ZstdCompressor().compress(on_f.tobytes()))
    (folder / "000004.qc.averaged.bin").write_bytes(zstandard.ZstdCompressor().compress(on_e.tobytes()))
    deliver_sampled(qc, "run-a", 4, {"judged": 30, "mixed": 0}, {"judged": 30, "mixed": 30, "tx_sum": 1.0})

    def got(seq, query=""):
        return client.get(f"{url}/{seq}/cells{query}", headers=auth({"Accept-Encoding": "identity"}))
    for seq, query, values in ((4, f"?model={model}&count=sampled", on_f), (4, f"?model={model}&count=averaged", on_e),
                               (3, f"?model={model}&count=averaged", averaged)):
        sent = got(seq, query)
        assert sent.content == values.tobytes() and "immutable" in sent.headers["cache-control"], query
    for seq, query, values in ((4, "", on_f), (4, f"?model={model}", on_f), (4, "?count=averaged", on_e)):
        sent = got(seq, query)
        assert sent.content == values.tobytes() and "no-cache" in sent.headers["cache-control"], query
    for seq, query in ((3, "?count=sampled"), (4, "?count=e"), (3, f"?model={model}&count=sampled")):
        refused = got(seq, query)
        assert refused.status_code == 404 and refused.json()["detail"] == "This iteration's count changed", query
    (folder / "000004.qc.averaged.bin").unlink()
    refused = got(4, "?count=averaged")
    assert refused.status_code == 404 and refused.json()["detail"] == "This iteration's count changed"
    assert got(5, "?count=averaged").status_code == 404 and got(4, "?model=0123&count=sampled").status_code == 404
    # Frame 3 counted again (the hold's line, with f, its count's file there) while its result is still the averaged
    # one: it is being scored again, and the worker rewrites its bytes before the new result is in, so no count is
    # served for it until then (named by no count, its bytes as they are, never pinned). With that file gone (it is
    # never scored again) the result kept stands and is served as it is.
    run = application.state.runs.get("run-a")
    counted = run.live / "frames" / "000003.counts.zst"
    counted.parent.mkdir(parents=True, exist_ok=True)
    counted.write_bytes(b"PLC1")
    with run.lock:
        run.frames[3].update(counts_file="frames/000003.counts.zst", counts={"seq": 3, "reason": "hold", "has_f": True})
    for query in (f"?model={model}&count=averaged", "?count=averaged", f"?model={model}&count=sampled"):
        refused = got(3, query)
        assert refused.status_code == 404 and refused.json()["detail"] == "This iteration's count changed", query
    sent = got(3, "")
    assert sent.content == averaged.tobytes() and "no-cache" in sent.headers["cache-control"]
    counted.unlink()
    sent = got(3, f"?model={model}&count=averaged")
    assert sent.content == averaged.tobytes() and "immutable" in sent.headers["cache-control"]


# ---- the start: proseg's count at iteration 0 shown, each frame compared with its own start block --------------------
def block(judged, mixed, tx_sum, median_tx=300.0, iteration=0):
    """A frame's start block as the worker gives it (`start_block`): the cells proseg started with, their RNA kept as
    proseg keeps it at the frame, scored at the block's own depth."""
    return {"per100": round(100.0 * mixed / judged, 2), "judged": judged, "mixed": mixed, "tx_sum": tx_sum,
            "median_tx": median_tx, "iteration": iteration}


def test_the_arrow_and_retention_compare_each_frame_with_its_own_start_block(world):
    """The start line shown is proseg's count at iteration 0 (qc_start: 10 in 100), fixed for the whole run. Each frame
    is compared with its own start block instead: the raw change is the smoothed share minus the block's share smoothed
    the same way (never the start line's), retention is proseg's count now over the block's (unsmoothed, per frame), a
    frame judged alone uses its own block and leaves the smoothing alone, a frame without a block (or with one of more
    mixed than judged cells) compares nothing, and an engine restart goes on smoothing the block from where it was."""
    cfg, sample = world
    write_run(Path(cfg["paths"]["data_dir"]), "run-block", sample, "2026-10-30T08:00:00+00:00", frames=12,
              times=[float(s) for s in range(12)], burnin=2)
    url = "/api/v1/runs/run-block/qc"
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-block")
        deliver_start(qc, "run-block", 40, 4)                            # iteration 0: 10 in 100
        counts = {"judged": 4000, "mixed": 400, "ncells": 4500, "tx_sum": 600000.0}       # 10 in 100 now
        first = deliver_counts(qc, "run-block", 3, start_block=block(4000, 480, 800000.0), **counts)
        second = deliver_counts(qc, "run-block", 4, start_block=block(4000, 320, 750000.0), **counts)
        alone = deliver_counts(qc, "run-block", 1, single=True, start_block=block(4000, 200, 700000.0), **counts)
        torn = deliver_counts(qc, "run-block", 2, single=True, start_block=block(4000, 4400, 700000.0), **counts)
        bare = deliver_counts(qc, "run-block", 5, **counts)              # a frame without a start block
        series = client.get(url + "/series", headers=auth()).json()
        folder = qc._state("run-block").dir
    assert first["start"] == second["start"] == {"per100": 10.0, "judged": 40, "mixed": 4, "median_tx": 386.0}
    assert series["start"]["per100"] == 10.0                             # the chart's start point: iteration 0's
    assert first["change"]["raw"] == pytest.approx(10.0 - 12.0)          # against its block, 12 in 100
    assert first["retention"] == {"ratio": 0.75, "start_tx": 800000, "tx": 600000}
    # 1 s later (a region's weight, 0.2): the block smoothed to 480 x 0.8 + 320 x 0.2 = 448 of 4000 (11.2 in 100), as
    # the headline would be; retention its own.
    assert second["change"]["raw"] == pytest.approx(10.0 - 11.2, abs=1e-9) and second["retention"]["ratio"] == 0.8
    assert alone["single"] and alone["change"]["raw"] == pytest.approx(10.0 - 5.0) and alone["retention"]["ratio"] == 0.8571
    assert bare["change"]["raw"] is None and bare["retention"] is None and bare["start"]["per100"] == 10.0
    assert torn["change"]["raw"] is None and torn["retention"] is None    # more mixed than judged: no block at all
    assert json.loads((folder / "state.json").read_text())["ema_start"] == pytest.approx([4000.0, 448.0])
    application = make(cfg)                                              # an engine restart
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-block")
        after = deliver_counts(qc, "run-block", 6, start_block=block(4000, 600, 800000.0), **counts)
    assert after["change"]["raw"] == pytest.approx(10.0 - 11.96)         # 448 x 0.8 + 600 x 0.2 of 4000: smoothed on


def test_a_start_with_nuclei_compares_nothing_with_a_start(app):
    """Cells with their nuclei (a Xenium bundle's own start): no start line, so whatever a result carries, no raw
    change, no like-for-like change, no verdict, no retention (the worker's numbers kept as it gave them), and no
    "forming" note while cells form (a run from cells alone has one then)."""
    application, client, sample = app
    qc = application.state.qc
    early = {}
    for run_id in ("run-paired", "run-b"):
        ready_run(client, run_id)
        early[run_id] = deliver_counts(qc, run_id, 3, judged=4000, mixed=400, ncells=4500)["change"]   # iteration 30
    assert "forming" not in early["run-paired"] and early["run-b"]["forming"] is True      # cells form until 50
    assert early["run-paired"]["direction"] is None and early["run-b"]["direction"] is None
    got = deliver_counts(qc, "run-paired", 5, judged=4000, mixed=400, ncells=4500, tx_sum=600000.0,
                         start_block=block(4000, 480, 800000.0), matched=matched(4000, 10.0, 12.0))
    assert got["start"] is None and got["retention"] is None
    change = got["change"]
    assert change["raw"] is None and change["matched"] is None and change["direction"] is None and not change["meaningful"]
    assert "forming" not in change and got["matched"]["per100_start"] == 12.0
    assert client.get("/api/v1/runs/run-paired/qc", headers=auth()).json()["start_kind"] == "paired"


def test_every_summary_names_its_count_and_the_constants_it_was_judged_with(app, monkeypatch):
    """basis as the worker says it (None when it does not), the count the engine asked it to read ("averaged" unless a
    sampled frame's), and the constants of the run's mode: a dense region, a tiled one, a whole sample (20,000 cells or
    more); measured true for a mode re-measured on proseg's count (regions and tiled regions), false until it is (whole
    samples: the page's caveat), and the constants kept from the outlines' measurement there (`kept`: the caveats)."""
    application, client, sample = app
    qc = application.state.qc
    ready_run(client)
    run = application.state.runs.get("run-a")
    plain = deliver_counts(qc, "run-a", 3, judged=40, mixed=4, ncells=NCELLS)
    assert (plain["basis"], plain["count"]) == (None, "averaged")
    assert plain["constants"] == {"mode": "region", "id": K.CONSTANTS_ID, "measured": True, "kept": []}
    said = deliver_counts(qc, "run-a", 4, judged=40, mixed=4, ncells=NCELLS, basis="proseg", count="averaged")
    assert (said["basis"], said["count"]) == ("proseg", "averaged")
    run.meta["mode"] = "tiled"
    tiled = deliver_counts(qc, "run-a", 5, judged=40, mixed=4, ncells=NCELLS)
    assert tiled["constants"] == {"mode": "tiled", "id": K.CONSTANTS_ID, "measured": True,
                                  "kept": ["glow", "where_words"]}
    run.info["cells"] = 41190
    whole = deliver_counts(qc, "run-a", 6, judged=40, mixed=4, ncells=41190)
    assert whole["constants"] == {"mode": "whole", "id": K.CONSTANTS_ID, "measured": False,
                                  "kept": ["glow", "where_words", "start_band", "min_counts", "like_depth"]}
    monkeypatch.setitem(K.MEASURED, "whole", True)                     # once its own repeat runs measure it
    assert deliver_counts(qc, "run-a", 7, judged=40, mixed=4, ncells=41190)["constants"]["measured"] is True


def test_a_run_on_tiles_is_tiled_and_a_dense_run_of_a_whole_samples_size_keeps_its_50_um_areas(app):
    """A run is tiled by its own mode or by a prior proseg reads in tiles (info.json prior format "tiles"; the rule
    AnalysisManager.qc_grid sizes its bins with): its summaries name the tiled constants and its areas are 100 µm. A
    dense run of 20,000 cells or more is judged as a whole sample, but its areas keep the dense constants: its bins
    are 50 µm."""
    application, client, sample = app
    qc = application.state.qc
    ready_run(client)
    run = application.state.runs.get("run-a")
    dense = run.info["prior"]
    assert run.meta["mode"] == "detail" and qc._area_params(run)["bin_um"] == 50.0
    run.info["prior"] = {**dense, "format": "tiles"}
    assert deliver_counts(qc, "run-a", 3, judged=40, mixed=4, ncells=NCELLS)["constants"]["mode"] == "tiled"
    assert (qc._area_params(run)["mode"], qc._area_params(run)["bin_um"]) == ("tiled", 100.0)
    run.info["prior"] = dense
    run.info["cells"] = 41190
    assert deliver_counts(qc, "run-a", 4, judged=40, mixed=4, ncells=41190)["constants"]["mode"] == "whole"
    assert (qc._area_params(run)["mode"], qc._area_params(run)["bin_um"]) == ("whole", 50.0)


@pytest.mark.parametrize("mode", ["tiled", "whole"])
def test_a_tiled_region_and_a_whole_sample_are_judged_with_their_own_modes_constants(world, monkeypatch, mode):
    """Every mode holds the same values today, so here the run's own mode is given values of its own, as a calibration
    may: a tiled region's numbers (and a whole sample's: 20,000 cells or more) follow them, never a dense region's. The
    headline's smoothing and its whole number's margin, the minimum support (coverage, the arrow's verdict), the
    arrow's threshold and the start line's coverage (describe, GET /qc/start and the page's start event)."""
    from sopa_companion.engine import qc_constants as K
    for key, value in (("headline_weight", 0.25), ("shown_margin", 0.05), ("local_min_judged", 5000),
                       ("matched_fixed", 7.0)):
        monkeypatch.setitem(K.TABLE[mode], key, value)
    cfg, sample = world
    folder = write_run(Path(cfg["paths"]["data_dir"]), "run-mode", sample, "2026-10-30T09:00:00+00:00", frames=12,
                       times=[float(s) for s in range(12)], burnin=2, mode="tiled")
    if mode == "whole":
        info = json.loads((folder / "live" / "info.json").read_text())
        (folder / "live" / "info.json").write_text(json.dumps({**info, "cells": 41190}))
    url = "/api/v1/runs/run-mode/qc"
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client, "run-mode")
        state = qc._state("run-mode")
        state.dir.mkdir(parents=True, exist_ok=True)
        qc.qc_result({"kind": "start", "run": "run-mode", "seq": None, "model": state.key, "dir": str(state.dir),
                      "source": "start", "single": True, "bins": None, "cells": None, "start": None,
                      "summary": {"kind": "start", "judged": 4000, "mixed": 400, "median_tx": 380.0, "ncells": 4500}})
        common = {"judged": 4000, "ncells": 4500, "tx_sum": 600000.0, "matched": matched(4000, 10.0, 11.0),
                  "start_block": block(4000, 440, 700000.0)}
        first = deliver_counts(qc, "run-mode", 3, mixed=400, **common)
        second = deliver_counts(qc, "run-mode", 4, mixed=496, **common)
        start = client.get(url + "/start", headers=auth()).json()
        described = client.get(url, headers=auth()).json()["start"]
        (pushed,) = [e["summary"] for e in qc.events({}, "run-mode", 0.0) if e.get("state") == "start"]
    assert first["constants"]["mode"] == second["constants"]["mode"] == mode
    # 1 s later the newest weighs 0.25: (0.75 x 400 + 0.25 x 496) / 4000 = 10.6 in 100 (a region's 0.5 gives 11.2),
    # 0.1 past 10.5: the whole number moves with a margin of 0.05 (a region's 0.3 keeps it at 10).
    assert first["per100"] == 10.0 and second["per100"] == pytest.approx(10.6)
    assert first["shown"] == 10 and second["shown"] == 11
    assert second["coverage"]["enough"] is False and second["change"]["enough"] is False     # 4,000 judged < 5,000
    assert second["change"]["threshold"] == 7.0           # the mode's floor (a region's is 1.0, a whole sample's 0.3)
    assert start["coverage"]["enough"] is False and described["coverage"]["enough"] is False
    assert pushed["coverage"]["enough"] is False and pushed["per100"] == 10.0


def test_the_earlier_proseg_cells_come_only_from_a_table_of_prosegs_own_count(world):
    """The earlier line ("Proseg's cells saved in this sample") reads the model's batch.parquet only while the table it
    scored holds proseg's own count in the sample (its uns: proseg's run record): a model learned before that rule may
    have scored a Xenium sample's own "table" (Xenium Ranger's outline re-count), never shown as proseg's cells."""
    import shutil
    cfg, sample = world
    application = make(cfg)
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client)
        deliver(qc, "run-a", 3, 40, 4)
        assert client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["batch_per100"] is not None
        model_dir = Path(qc._state("run-a").model_dir)
    shutil.rmtree(Path(sample["path"]) / "tables" / "table" / "uns")        # the same table, no proseg record
    application = make(cfg)                                              # read again in a new session
    with TestClient(application, base_url=BASE) as client:
        qc = application.state.qc
        ready_run(client)
        assert (model_dir / "batch.parquet").exists() and Path(qc._state("run-a").model_dir) == model_dir
        deliver(qc, "run-a", 4, 40, 4)
        view = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()
        assert view["batch_per100"] is None and view["batch_table"] is None
        assert client.get("/api/v1/runs/run-a/qc/4", headers=auth()).json()["batch_per100"] is None


# ---- end to end: the real analysis scheduler and the fake worker process -------------------------------------------------
class OneRun:
    def __init__(self, run):
        self.run = run

    def get(self, run_id):
        if run_id != self.run.meta["id"]:
            raise KeyError(run_id)
        return self.run

    def list(self):
        return [dict(self.run.meta)]


class Counting:
    """A QcManager whose results are counted (how many the scheduler handed over)."""

    def __init__(self, manager):
        self.manager, self.records = manager, []

    def qc_model(self, run):
        return self.manager.qc_model(run)

    def qc_result(self, record):
        self.records.append(record)
        return self.manager.qc_result(record)


def test_results_flow_from_the_analysis_scheduler(tmp_path, monkeypatch):
    from sopa_companion.engine import qc as Q
    from sopa_companion.engine.analysis import AnalysisManager
    from test_analysis import JUDGE, disk_run
    tests = str(Path(__file__).resolve().parent)
    monkeypatch.setenv("PYTHONPATH", tests + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.delenv("FAKE_QC_MODE", raising=False)
    sample = write_sample(tmp_path)
    run = disk_run(tmp_path / "runs" / "r1", frames=4)                   # frames, transcripts, prior, cell statistics
    run.meta.update(state="done", sample_kind="zarr", sample=sample["id"], sample_path=sample["path"],
                    sample_name=sample["name"], created="2026-10-17T12:00:00+00:00",
                    prior={"cells": {"kind": "shapes", "key": "xenium_cell_boundaries"}, "nuclei": None})
    runs = OneRun(run)
    config = {"paths": {"data_dir": str(tmp_path / "var"), "state_dir": str(tmp_path / "state")},
              "analysis": {"python": sys.executable, "worker_module": "fake_analysis_worker"},
              "qc": {"python": sys.executable, "trainer_module": FAKE, "probe_module": FAKE_PROBE, "gpu_quiet_s": 0,
                     "enabled_by_default": True, "learn": "open",
                     "judge_min_counts": JUDGE, "fit_min_counts": 5, "bin_um_detail": 8}}

    def session():
        models = Q.QcModels(config, Registry(sample), probe=lambda: dict(IDLE))
        analysis = AnalysisManager(config, runs)
        manager = Q.QcManager(config, runs, models, analysis=analysis)
        counting = Counting(manager)
        analysis.attach_qc(counting)
        return models, analysis, manager, counting

    models, analysis, qc, counting = session()
    try:
        assert qc.describe("r1")["state"] in ("waiting", "learning")      # the switch is on: the sample is learned
        wait_for(lambda: qc.describe("r1")["seqs"] == [0, 1, 2, 3] and qc.describe("r1")["start"] is not None, timeout=60)
    finally:
        analysis.close()
        analysis.thread.join(5)
        models.close()
    key = qc._state("r1").key
    order = [(r["kind"], r["seq"]) for r in counting.records]
    assert order == [("frame", 3), ("start", None), ("frame", 0), ("frame", 2), ("frame", 1)], order
    # One folder per model and count basis, the scheduler's (the worker's files) and this manager's alike
    # (engine/analysis.py qc_dir): results scored on another count are never read.
    folder = run.dir / "qc" / f"{key[:8]}.{BASIS}"
    assert qc._state("r1").dir == folder == analysis.qc_dir(run, key), "the scheduler's and the check's folder"
    names = {p.name for p in folder.iterdir()}
    assert {f"{s:06d}.qc.json" for s in range(4)} | {f"{s:06d}.qc.bin" for s in range(4)} <= names
    assert {"start.json", "start.bin", "start.npz", "series.jsonl", "state.json"} <= names
    worker = {r["seq"]: r["summary"] for r in counting.records}
    for seq in range(4):                                                 # the worker's numbers, this manager's headline
        mine = qc.summary("r1", seq)
        assert (mine["judged"], mine["mixed"]) == (worker[seq]["judged"], worker[seq]["mixed"]) and mine["judged"] > 0
        assert mine["single"] is True and mine["model"] == key      # frame 3, proseg's end: its own numbers (sampled)
        assert mine["count"] == ("sampled" if seq == 3 else "averaged")
    assert qc.summary("r1", 3)["per100"] == round(100 * worker[3]["mixed"] / worker[3]["judged"], 1)
    assert qc.describe("r1")["start"]["judged"] == worker[None]["judged"]
    # Another session: the scheduler finds every frame and the start line done (this manager's files): nothing again.
    models, analysis, qc, counting = session()
    try:
        assert qc.describe("r1")["seqs"] == [0, 1, 2, 3]
        wait_for(lambda: analysis.describe_qc("r1")["start"] == "done" and analysis.describe_qc("r1")["seqs"] == [0, 1, 2, 3],
                 timeout=20)
        import time
        time.sleep(0.5)
        assert counting.records == []
    finally:
        analysis.close()
        analysis.thread.join(5)
        models.close()
