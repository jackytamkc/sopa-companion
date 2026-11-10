"""Engine robustness (2026-10-17 baseline backlog): the picker's shape queries off the event loop, run phases with a
time-to-first-frame estimate, deleted runs' transcript indexes and leftover folders, exact ports for non-default
engines, and stale zarr save locks."""
from __future__ import annotations

import fcntl
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient

from conftest import HERE, wait_for

BASE = "http://127.0.0.1:8799"
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def auth(extra=None):
    return {"Authorization": "Bearer " + "t" * 40, "Origin": BASE, **(extra or {})}


# ---- shapes: synthetic exports in the helper's layout ---------------------------------------------------------------
def write_outlines(base: Path, polygons: list[np.ndarray], ids: list[str]):
    """What zarr_bridge/helper.py `shapes` writes: <base>.bin (bounds, offsets, owner, coords) + <base>.json."""
    base.parent.mkdir(parents=True, exist_ok=True)
    coords = np.concatenate(polygons).astype(np.float32) if polygons else np.zeros((0, 2), np.float32)
    offsets = np.zeros(len(polygons) + 1, np.uint32)
    np.cumsum([len(p) for p in polygons], out=offsets[1:])
    bounds = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polygons], np.float32).reshape(-1, 4)
    owner = np.arange(len(polygons), dtype=np.uint32)
    Path(f"{base}.bin").write_bytes(b"".join(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
                                             for a in (bounds, offsets, owner, coords)))
    Path(f"{base}.json").write_text(json.dumps({"schema": 1, "parts": len(polygons), "vertices": int(len(coords)), "ids": ids}))


def synthetic_polygons(n: int, vertices: int, seed: int = 0, size: float = 6000.0):
    """n jittered rings (cell-sized, ~15 µm) scattered over a size x size µm sample, at Xenium-like float32 values."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(20, size - 20, (n, 2))
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    rings = []
    for c in centres:
        radius = rng.uniform(4, 9) * (1 + 0.15 * rng.standard_normal(vertices))
        rings.append(np.stack([c[0] + radius * np.cos(angles), c[1] + radius * np.sin(angles)], 1).astype(np.float32))
    return rings, [f"cell{k:06d}-{seed}" for k in range(n)]


def old_body(base: Path, key: str, box, limit: int) -> bytes:
    """The response as the engine built it before (zarr.shapes + Starlette's JSONResponse rendering)."""
    from sopa_companion.engine.shape_query import load_outlines
    data = load_outlines(base)
    bounds = data["bounds"]
    if box is not None:
        x0, y0, x1, y1 = box
        hits = np.flatnonzero((bounds[:, 2] >= x0) & (bounds[:, 0] <= x1) & (bounds[:, 3] >= y0) & (bounds[:, 1] <= y1))
    else:
        hits = np.arange(len(bounds))
    total = int(hits.size)
    hits = hits[:max(0, int(limit))]
    offsets, coords, owner, ids = data["offsets"], data["coords"], data["owner"], data["ids"]
    polygons = [{"id": ids[owner[i]], "poly": np.round(coords[offsets[i]:offsets[i + 1]], 2).tolist()} for i in hits.tolist()]
    value = {"key": key, "total": total, "truncated": total > len(polygons), "polygons": polygons}
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode()


def test_shape_queries_answer_in_workers_with_the_old_bytes(tmp_path):
    from sopa_companion.engine.shape_query import ShapeQueries
    rings, ids = synthetic_polygons(3000, 24, seed=1, size=2000)
    rings.append(np.array([[50.004, -0.001]], np.float32))       # degenerate: one vertex, rounding to -0.0
    ids.append("odd-ünïcode")
    base = tmp_path / "shapes" / "cells"
    write_outlines(base, rings, ids)
    queries = ShapeQueries()
    try:
        for box, limit in [((100, 100, 900, 700), 20000), ((0, 0, 2000, 2000), 20000), ((0, 0, 2000, 2000), 500),
                           ((5000, 5000, 6000, 6000), 20000), (None, 400_000), ((100, 100, 900, 700), 0),
                           ((-10.5, 33.25, 400.125, 1999.0), 7)]:
            assert queries.query(base, "cells", box, limit) == old_body(base, "cells", box, limit), (box, limit)
        workers = [w.pid for w in queries.idle]
        assert workers and os.getpid() not in workers
        assert all(open(f"/proc/{pid}/cmdline").read().split("\0")[1:3] == ["-m", "sopa_companion.engine.shape_query"]
                   for pid in workers)
        assert all(len(os.listdir(f"/proc/{pid}/task")) == 1 for pid in workers)    # one BLAS thread: no pool of 64

        replaced, replaced_ids = synthetic_polygons(10, 8, seed=2, size=2000)        # a new export of the same key
        time.sleep(0.01)
        write_outlines(base, replaced, replaced_ids)
        assert queries.query(base, "cells", None, 20000) == old_body(base, "cells", None, 20000)
        Path(f"{base}.bin").unlink()
        with pytest.raises(FileNotFoundError):
            queries.query(base, "cells", None, 20000)
    finally:
        queries.close()
    for pid in workers:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_a_dead_shapes_worker_is_replaced(tmp_path):
    from sopa_companion.engine.shape_query import ShapeQueries
    rings, ids = synthetic_polygons(50, 8, seed=3, size=500)
    write_outlines(tmp_path / "cells", rings, ids)
    queries = ShapeQueries(workers=1)
    try:
        first = queries.query(tmp_path / "cells", "cells", None, 100)
        [worker] = queries.idle
        worker.kill()
        worker.wait()
        assert queries.query(tmp_path / "cells", "cells", None, 100) == first
        assert queries.idle and queries.idle[0].pid != worker.pid
    finally:
        queries.close()


def test_workers_on_descriptors_past_1024_answer(tmp_path):
    """select() refused pipes numbered 1024 or more with ValueError, which leaked the
    slot; two such leaks parked every later picker request for good. poll() has no such limit."""
    from sopa_companion.engine.shape_query import ShapeQueries
    rings, ids = synthetic_polygons(50, 8, seed=6, size=500)
    write_outlines(tmp_path / "cells", rings, ids)
    queries = ShapeQueries(workers=2)
    hold = [open(os.devnull) for _ in range(1100)]          # an engine with many sockets and files open
    try:
        expected = old_body(tmp_path / "cells", "cells", None, 100)
        for _ in range(3):
            assert queries.query(tmp_path / "cells", "cells", None, 100) == expected
        assert all(w.stdout.fileno() >= 1024 for w in queries.idle)
        assert queries.out == 0
    finally:
        for f in hold:
            f.close()
        queries.close()


def test_any_failure_frees_the_slot_and_stops_the_worker(tmp_path, monkeypatch):
    """An exception that is not an OSError (ValueError, MemoryError for a huge body)
    used to skip the cleanup: the slot stayed taken and the worker ran on outside the pool."""
    from sopa_companion.engine import shape_query
    rings, ids = synthetic_polygons(50, 8, seed=7, size=500)
    write_outlines(tmp_path / "cells", rings, ids)
    queries = shape_query.ShapeQueries(workers=2)
    real_read = shape_query._read
    try:
        first = queries.query(tmp_path / "cells", "cells", None, 100)
        used = list(queries.idle)                        # the first failure takes this worker, the second a new one
        real_spawn = queries._spawn
        monkeypatch.setattr(queries, "_spawn", lambda: used.append(real_spawn()) or used[-1])
        for error in (ValueError("filedescriptor out of range in select()"), MemoryError()):
            def failing(fd, size, deadline, error=error):
                raise error
            monkeypatch.setattr(shape_query, "_read", failing)
            with pytest.raises(type(error)):
                queries.query(tmp_path / "cells", "cells", None, 100)
            assert queries.out == 0, "the slot is free again"
        monkeypatch.setattr(shape_query, "_read", real_read)
        assert len(used) == 2 and all(w.poll() is not None for w in used), "the failed workers were stopped"
        assert not queries.idle
        done = []
        thread = threading.Thread(target=lambda: done.append(queries.query(tmp_path / "cells", "cells", None, 100)),
                                  daemon=True)
        thread.start()
        thread.join(30)
        assert done and done[0] == first, "the next request is answered (before: it waited forever)"
        children = [p for p in psutil.Process().children() if "sopa_companion.engine.shape_query" in " ".join(p.cmdline())]
        assert len(children) <= 2 and queries.out == 0
    finally:
        monkeypatch.setattr(shape_query, "_read", real_read)
        queries.close()


def test_a_request_waits_for_a_busy_pool_only_until_its_deadline_and_a_closed_pool_says_so(tmp_path):
    from sopa_companion.engine.shape_query import ShapeQueries, Unavailable
    rings, ids = synthetic_polygons(20, 8, seed=8, size=500)
    write_outlines(tmp_path / "cells", rings, ids)
    queries = ShapeQueries(workers=1)
    try:
        queries.out = 1                                     # the one worker is out with a long request
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            queries.query(tmp_path / "cells", "cells", None, 100, timeout=0.3)
        assert 0.25 <= time.monotonic() - started < 5
        queries.out = 0
        assert queries.query(tmp_path / "cells", "cells", None, 100)
    finally:
        queries.close()
    with pytest.raises(Unavailable):
        queries.query(tmp_path / "cells", "cells", None, 100)


def test_shapes_errors_map_to_their_status(config, tmp_path, monkeypatch):
    """A closed pool (shutdown) is 503, a worker failure 500; neither is a 'Storage error'."""
    from sopa_companion.engine.shape_query import Unavailable, WorkerError
    from sopa_companion.engine.zarr_samples import ZarrSamples
    from sopa_companion.engine.samples import SampleError
    zarr = ZarrSamples(config)
    try:
        monkeypatch.setattr(zarr, "_shapes_base", lambda sid, key: tmp_path / key)
        for error, status in ((Unavailable("The engine is shutting down"), 503), (WorkerError("boom"), 500),
                              (TimeoutError("slow"), 503)):
            def failing(*args, error=error, **kwargs):
                raise error
            monkeypatch.setattr(zarr.queries, "query", failing)
            with pytest.raises(SampleError) as caught:
                zarr.shapes_json("s", "cells")
            assert caught.value.status == status
    finally:
        zarr.close()


# ---- shapes: the engine stays responsive (a real engine process, health probed while the picker queries) ------------
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def engine_toml(path: Path, port: int, data: Path, state: Path) -> Path:
    path.write_text(f'[server]\nport = {port}\ntoken = "{"t" * 40}"\n'
                    f'[paths]\ndata_dir = "{data}"\nstate_dir = "{state}"\n'
                    '[desktop]\nexit_when_closed = false\nprofile_dir = ""\n')
    return path


def register_sample(data: Path, sid: str, layers: dict[str, tuple[list, list]], extent: float):
    """A SpatialData sample as the engine remembers it once opened (registry, inspection, exported outlines), so
    no sopa environment is needed."""
    folder = data / "cache" / "zarr" / sid
    folder.mkdir(parents=True)
    (data / "samples.json").write_text(json.dumps([{"id": sid, "name": "synthetic", "path": str(data / "synthetic.zarr"),
                                                    "kind": "zarr", "opened": "2026-10-17T00:00:00"}]))
    shapes = [{"key": key, "count": len(rings), "geometry": ["Polygon"], "extent": [0, 0, extent, extent],
               "to_frame": IDENTITY} for key, (rings, _) in layers.items()]
    (folder / "inspect.json").write_text(json.dumps({
        "name": "synthetic", "points_key": "transcripts", "images": [], "shapes": shapes, "tables": [], "labels": [],
        "attrs": {}, "points": [{"key": "transcripts", "count": 1000, "columns": ["x", "y"], "extent": [0, 0, extent, extent]}],
        "suggested_prior": {"cells": {"kind": "shapes", "key": "cells"}, "nuclei": {"kind": "shapes", "key": "nuclei"}}}))
    for key, (rings, ids) in layers.items():
        write_outlines(folder / "shapes" / key, rings, ids)


@pytest.fixture
def engine_process(tmp_path):
    """`python -m sopa_companion serve` on a free port with a synthetic whole-sample-sized shapes prior."""
    port = free_port()
    data, state = tmp_path / "var", tmp_path / "state"
    layers = {"cells": synthetic_polygons(40_000, 28, seed=4), "nuclei": synthetic_polygons(40_000, 24, seed=5)}
    register_sample(data, "zsynthetic0001", layers, 6000.0)
    config = engine_toml(tmp_path / "engine.toml", port, data, state)
    log = open(tmp_path / "engine.log", "wb")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = str(HERE.parents[1])
    process = subprocess.Popen([sys.executable, "-m", "sopa_companion", "--config", str(config), "serve", "--port", str(port)],
                               cwd=str(HERE.parents[1]), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", headers={"Authorization": "Bearer " + "t" * 40},
                          timeout=60)
    try:
        wait_for(lambda: process.poll() is not None or _answers(client), timeout=30, interval=0.1)
        assert process.poll() is None, (tmp_path / "engine.log").read_text()[-2000:]
        yield process, client, port
    finally:
        client.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()


def _answers(client) -> bool:
    try:
        return client.get("/health").status_code == 200
    except Exception:
        return False


def test_health_answers_while_the_picker_queries_a_whole_sample(engine_process):
    """The 2026-10-17 baseline: health max 0.95-1.5 s during the picker's shape queries on two whole samples. 40,000
    outlines per layer here: the old engine encoded each 20,000-polygon response on the event loop for about a
    second."""
    import httpx
    process, client, port = engine_process
    health: list[tuple[float, float]] = []
    stop = threading.Event()

    def probe():
        with httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", headers={"Authorization": "Bearer " + "t" * 40},
                          timeout=60) as probe_client:
            while not stop.is_set():
                started = time.perf_counter()
                assert probe_client.get("/health").status_code == 200
                health.append((time.monotonic(), (time.perf_counter() - started) * 1000))
                time.sleep(0.01)

    def picker(box, out):
        def one(key):
            response = client.get(f"/samples/zsynthetic0001/shapes/{key}", params={"box": ",".join(map(str, box)),
                                                                                  "limit": 20000})
            out[key] = (response.status_code, len(response.content), response.headers.get("content-type"))
        threads = [threading.Thread(target=one, args=(key,)) for key in ("cells", "nuclei")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    prober = threading.Thread(target=probe, daemon=True)
    prober.start()
    try:
        wait_for(lambda: len(health) >= 5)
        started = time.monotonic()
        results = []
        for box in ([0, 0, 6000, 6000], [0, 0, 5000, 6000], [500, 0, 6000, 5500], [0, 0, 6000, 6000]):
            out: dict = {}
            picker(box, out)
            results.append(out)
        ended = time.monotonic()
    finally:
        stop.set()
        prober.join(10)
    during = [ms for t, ms in health if t >= started and t - ms / 1000 <= ended]
    for out in results:
        for key, (status, size, kind) in out.items():
            assert status == 200 and kind == "application/json" and size > 5_000_000, (key, status, size)
    body = client.get("/samples/zsynthetic0001/shapes/cells", params={"box": "0,0,6000,6000", "limit": 20000}).json()
    assert body["total"] == 40_000 and body["truncated"] and len(body["polygons"]) == 20_000
    print(f"health during the picker's queries: n={len(during)}, max {max(during):.1f} ms, "
          f"median {sorted(during)[len(during) // 2]:.1f} ms over {ended - started:.2f} s")
    assert len(during) >= 10 and max(during) < 400, during      # old engine: up to 2.3 s here; now ~10 ms

    # The workers end with the engine, even a killed one (their stdin closes with it).
    workers = [p for p in psutil.Process(process.pid).children(recursive=True) if "sopa_companion.engine.shape_query" in p.cmdline()]
    assert workers
    process.send_signal(signal.SIGKILL)
    process.wait()
    gone, alive = psutil.wait_procs(workers, timeout=10)
    assert not alive


# ---- run phase and the time-to-first-frame estimate -------------------------------------------------------------------
def test_the_estimate_matches_the_runs_it_was_fitted_on():
    from sopa_companion.engine.runs import load_seconds, prepare_seconds, region_fraction
    measured = {3_120_307: (4.5, 4.8), 40_462_202: (38.0, 39.5), 81_208_356: (79.8, 86.9)}   # crop, two whole samples
    for transcripts, (prepare, load) in measured.items():
        assert prepare_seconds(transcripts, transcripts) == pytest.approx(prepare, rel=0.12)
        assert load_seconds(transcripts) == pytest.approx(load, rel=0.12)
    extent = [0, 0, 200, 100]
    assert region_fraction({"kind": "full", "bbox": extent}, extent) == 1
    assert region_fraction({"kind": "box", "bbox": [0, 0, 50, 100]}, extent) == 0.25
    assert region_fraction({"kind": "polygon", "points": [[0, 0], [100, 0], [0, 100]], "bbox": [0, 0, 100, 100]},
                           extent) == 0.25


def make_runs(config):
    from sopa_companion.engine.runs import Runs
    from sopa_companion.engine.samples import Samples
    return Runs(config, Samples(config))


def test_phase_and_eta_follow_a_run_from_preparing_to_done(config, monkeypatch):
    from sopa_companion.engine import runs as runs_module
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "2.5")
    monkeypatch.setattr(runs_module, "LOAD_S", (30.0, 0.0))      # an estimate that outlasts the fake's loading
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    created = runs.create(sample, [50, 50, 150, 150])
    assert created["phase"] in ("preparing", "loading") and isinstance(created["eta_s"], int) and 0 < created["eta_s"] <= 33
    runs.control(created["id"], "resume")                          # every run starts paused on its prior
    run = runs.get(created["id"])
    assert run.expected == {"sample": 4000, "region": 333}         # 100 x 100 of the bundle's 400 x 300 µm
    seen, etas = [], []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        public = run.public()
        if not seen or seen[-1] != public["phase"]:
            seen.append(public["phase"])
        if public["phase"] not in ("preparing", "loading"):
            break
        if public["phase"] == "loading":
            etas.append((public["eta_s"], run.version))
        time.sleep(0.05)
    assert seen[-2:] == ["loading", "sampling"] and set(seen[:-1]) <= {"preparing", "loading"}, seen
    values = [eta for eta, _ in etas]
    assert all(isinstance(v, int) for v in values) and values == sorted(values, reverse=True) and values[0] - values[-1] >= 1
    assert etas[-1][1] - etas[0][1] >= 2                            # sent again every second while loading
    assert run.public()["eta_s"] is None
    assert runs.control(run.meta["id"], "pause")["phase"] == "paused"
    assert runs.control(run.meta["id"], "finish")["phase"] == "finishing"
    wait_for(lambda: run.meta["state"] == "done")
    assert run.public()["phase"] == "done" and run.public()["eta_s"] is None


def test_a_run_started_paused_is_loading_until_its_first_frame(config, monkeypatch):
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "1.5")
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    run = runs.get(runs.create(sample, [50, 50, 150, 150])["id"])       # every run starts paused on its prior
    wait_for(lambda: run.meta["state"] == "paused")
    assert not run.frames and run.public()["phase"] == "loading"
    run.spawned_at -= 3600                                           # far past the estimate: no number is better than 0
    assert run.public()["eta_s"] is None
    wait_for(lambda: len(run.frames) >= 1)
    assert run.public()["phase"] == "paused"
    runs.control(run.meta["id"], "abort")
    wait_for(lambda: run.meta["state"] == "stopped")
    assert run.public()["phase"] == "stopped"


def test_finish_while_loading_stays_loading_with_its_estimate(config, monkeypatch):
    """Finish & save before the first frame said 'finishing' with no estimate for as
    long as proseg still read its input; it is loading until frame 0, then finishes."""
    from sopa_companion.engine import runs as runs_module
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "1.5")
    monkeypatch.setattr(runs_module, "LOAD_S", (30.0, 0.0))
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    run = runs.get(runs.create(sample, [50, 50, 150, 150])["id"])
    wait_for(lambda: run.public()["phase"] == "loading")
    public = runs.control(run.meta["id"], "finish")
    assert public["state"] == "finishing" and public["phase"] == "loading" and isinstance(public["eta_s"], int)
    wait_for(lambda: run.meta["state"] == "done", timeout=30)
    assert run.frames and run.public()["phase"] == "done"


def test_the_countdown_goes_out_once_a_second_only_while_there_is_one(config):
    """A SpatialData sample's preparation ticks through the helper's poll (once a
    second at most), and nothing is re-sent once the estimate has passed."""
    from sopa_companion.engine.runs import Run, RunError
    runs = make_runs(config)
    run = Run(runs.root / "ticking", {"id": "ticking", "state": "preparing", "sample": "s", "region": {}, "prior": {},
                                      "mode": "tiled", "box": [0, 0, 1, 1]})
    run.expected, run.started_at = {"sample": 10_000_000, "region": 10_000_000}, time.monotonic()
    (runs.root / "ticking").mkdir()
    versions, polls = [], []

    class Zarr:
        def prepare(self, sid, region, prior, destination, progress=None, should_stop=None):
            deadline = time.monotonic() + 2.3
            while time.monotonic() < deadline:
                polls.append(should_stop())
                versions.append(run.version)
                time.sleep(0.1)
            return {"transcripts": 0}
    runs.zarr = Zarr()
    with pytest.raises(RunError, match="no transcripts"):
        runs._prepare_zarr(run)
    assert not any(polls) and len(polls) >= 20
    ticks = len(set(versions)) - 1                      # the first message counts too: the rest are the ticks
    assert 2 <= ticks <= 4, versions
    run.meta["state"], run.spawned_at = "paused", time.monotonic() - 3600     # loading, far past the estimate
    before = run.version
    run.ticked = 0
    run.tick()
    assert run.public()["phase"] == "loading" and run.public()["eta_s"] is None and run.version == before


def test_the_page_gets_the_countdown_over_the_websocket(config, monkeypatch):
    from sopa_companion.server import create_app
    monkeypatch.setenv("FAKE_PROSEG_LOAD_S", "3")
    with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
        sample = client.app.state.samples.list()[0]["id"]
        run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
        client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
        ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
        loading = []
        with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
            ws.send_json({"op": "watch", "run": run["id"], "since": 0})
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                message = ws.receive_json()
                if message["ev"] != "run":
                    continue
                if message["run"]["phase"] == "loading":
                    loading.append(message["run"]["eta_s"])
                elif loading and message["run"]["phase"] == "sampling":
                    assert message["run"]["eta_s"] is None
                    break
        assert len(loading) >= 2 and all(isinstance(v, int) or v is None for v in loading), loading
        client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())


# ---- deleting runs: the transcript index and what NTFS leaves -----------------------------------------------------------
def mapped(path: str) -> bool:
    return os.path.realpath(path) in Path("/proc/self/maps").read_text()


def test_deleting_a_run_unmaps_its_transcript_index(config):
    from sopa_companion.server import create_app
    with TestClient(create_app(config, token="t" * 40), base_url=BASE) as client:
        runs = client.app.state.runs
        sample = runs.samples.list()[0]["id"]
        run = client.post("/api/v1/runs", json={"sample": sample, "box": [50, 50, 150, 150]}, headers=auth()).json()
        client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "resume"}, headers=auth())
        wait_for(lambda: len(runs.get(run["id"]).frames) >= 2)
        folder = runs.get(run["id"]).dir
        found = client.get(f"/api/v1/runs/{run['id']}/transcripts/view", params={"box": "0,0,5,5"}, headers=auth())
        assert found.status_code == 200 and found.headers["x-count"] == "3"
        sorted_file = str(folder / "live" / "transcripts.sorted.bin")
        assert mapped(sorted_file) and run["id"] in runs.indexes
        client.post(f"/api/v1/runs/{run['id']}/control", json={"action": "abort"}, headers=auth())
        wait_for(lambda: runs.get(run["id"]).meta["state"] == "stopped")
        assert client.delete(f"/api/v1/runs/{run['id']}", headers=auth()).status_code == 200
        assert not mapped(sorted_file) and not runs.indexes and not folder.exists()
        assert client.get(f"/api/v1/runs/{run['id']}/transcripts/view", params={"box": "0,0,5,5"},
                          headers=auth()).status_code == 404


def test_closing_the_engine_unmaps_every_index(config):
    runs = make_runs(config)
    sample = runs.samples.list()[0]["id"]
    run = runs.get(runs.create(sample, [50, 50, 150, 150])["id"])
    runs.control(run.meta["id"], "resume")
    wait_for(lambda: len(run.frames) >= 2)
    runs.transcript_index(run.meta["id"]).query((0, 0, 5, 5), 10)
    assert mapped(str(run.live / "transcripts.sorted.bin"))
    runs.close(save=False)
    assert not mapped(str(run.live / "transcripts.sorted.bin")) and not runs.indexes


def test_holders_finds_open_files_mappings_and_working_folders(tmp_path):
    from sopa_companion.engine.leftovers import holders
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    (tmp_path / "free").mkdir()
    opened = open(tmp_path / "a" / "open.bin", "wb")
    (tmp_path / "b" / "mapped.bin").write_bytes(b"x" * 4096)
    with open(tmp_path / "b" / "mapped.bin", "rb") as stream:
        view = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    child = subprocess.Popen(["sleep", "30"], cwd=tmp_path / "c")
    try:
        found = holders([tmp_path])
        assert {str(tmp_path / "a" / "open.bin"), str(tmp_path / "b" / "mapped.bin"), str(tmp_path / "c")} <= found
        assert not holders([tmp_path / "free"])
    finally:
        opened.close()
        view.close()
        child.kill()
        child.wait()
    assert not holders([tmp_path])


def test_the_engine_start_sweeps_what_deleted_runs_left(config, tmp_path):
    root = Path(config["paths"]["data_dir"]) / "runs"
    (root / "emptied" / "live" / "frames").mkdir(parents=True)                  # a deleted run, once released
    (root / "hidden" / "live").mkdir(parents=True)
    (root / "hidden" / "live" / ".fuse_hidden0000a1").write_bytes(b"x")         # ... whose holder is gone
    (root / "held" / "live").mkdir(parents=True)
    (root / "held" / "live" / ".fuse_hidden0000a2").write_bytes(b"x")
    (root / "odd").mkdir()
    (root / "odd" / "notes.txt").write_text("not a run's leftover")
    kept = root / "20261017-000000-kept"
    (kept / "live").mkdir(parents=True)
    (kept / "run.json").write_text(json.dumps({"id": kept.name, "state": "done", "created": "2026-10-17T00:00:00+00:00"}))
    (kept / "live" / ".fuse_hidden0000a3").write_bytes(b"x")
    holder = open(root / "held" / "live" / ".fuse_hidden0000a2", "rb")
    try:
        runs = make_runs(config)
    finally:
        holder.close()
    assert sorted(p.name for p in root.iterdir()) == sorted(["held", "odd", kept.name])
    assert (root / "held" / "live" / ".fuse_hidden0000a2").exists() and (root / "odd" / "notes.txt").exists()
    assert not (kept / "live" / ".fuse_hidden0000a3").exists() and kept.name in runs.older_runs   # an older run: listed
    make_runs(config)                                                             # released now: the next start
    assert sorted(p.name for p in root.iterdir()) == sorted(["odd", kept.name])


def test_what_builders_left_in_deleted_runs_goes_and_unknown_folders_are_reported(config, capsys):
    """An index or vectors build still writing when its run was deleted or discarded
    (engine exit included) left regular files in a folder without run.json, which the sweep then kept forever and
    reported nowhere."""
    root = Path(config["paths"]["data_dir"]) / "runs"
    (root / "indexed" / "live").mkdir(parents=True)
    (root / "indexed" / "live" / "transcripts.sorted.bin").write_bytes(b"x" * 64)
    (root / "indexed" / "live" / "transcripts.grid.json").write_text("{}")
    (root / "interrupted" / "live").mkdir(parents=True)
    (root / "interrupted" / "live" / ".transcripts.sorted.bin.12345").write_bytes(b"x" * 64)   # killed mid-write
    (root / "vectors" / "xenium").mkdir(parents=True)
    (root / "vectors" / "xenium" / ".vectors.v1.bin.zst.12345.678").write_bytes(b"x")
    (root / "vectors" / "xenium" / "boundaries.json").write_text("{}")
    (root / "odd" / "input").mkdir(parents=True)
    (root / "odd" / "input" / "transcripts.parquet").write_bytes(b"x" * 3_000_000)          # a run's own data: kept
    kept = root / "20261017-000000-kept"                                            # a saved run an engine died in
    (kept / "xenium").mkdir(parents=True)
    (kept / "live").mkdir()
    (kept / "run.json").write_text(json.dumps({"id": kept.name, "state": "done", "created": "2026-10-17T00:00:00+00:00"}))
    (kept / "xenium" / "vectors.v1.bin.zst").write_bytes(b"whole")
    (kept / "xenium" / ".vectors.v1.bin.zst.4242.140000").write_bytes(b"torn")
    (kept / "live" / ".transcripts.sorted.bin.4242").write_bytes(b"torn")
    runs = make_runs(config)
    assert sorted(p.name for p in root.iterdir()) == sorted(["odd", kept.name]) and kept.name in runs.older_runs
    assert sorted(p.name for p in (kept / "xenium").iterdir()) == ["vectors.v1.bin.zst"]
    assert not any((kept / "live").iterdir())
    out = capsys.readouterr().out
    assert "removed 3 folder(s) and 2 file(s)" in out and "kept 1 folder(s)" in out and "odd (3 MB)" in out, out


def test_holders_sees_a_held_file_that_was_unlinked(tmp_path):
    """fd targets of unlinked files end in ' (deleted)', as map paths do."""
    from sopa_companion.engine.leftovers import holders
    path = tmp_path / ".fuse_hidden0000b1"
    path.write_bytes(b"x")
    with open(path, "rb"):
        path.unlink()
        assert str(path) in holders([tmp_path])


def test_a_folder_still_held_after_a_delete_goes_once_it_is_released(config, tmp_path):
    runs = make_runs(config)
    folder = Path(config["paths"]["data_dir"]) / "runs" / "deleted"
    stuck = folder / "live" / "frames"
    stuck.mkdir(parents=True)
    (stuck / "000001.bin.gz").write_bytes(b"x")
    stuck.chmod(0o500)                       # rmtree cannot empty it: what a file held on NTFS does to the folder
    held = {"calls": 0}

    def holders(folders):
        held["calls"] += 1
        if held["calls"] == 1:
            return {str(stuck / "000001.bin.gz")}                                 # still held at the first retry
        stuck.chmod(0o700)                                                        # let go
        return set()

    runs.held = holders
    try:
        runs._remove(folder, delays=(0.05, 0.05, 0.05))
        assert folder.exists()
        wait_for(lambda: not folder.exists(), timeout=10)
        assert held["calls"] == 2
    finally:
        if stuck.exists():
            stuck.chmod(0o700)


# ---- ports --------------------------------------------------------------------------------------------------------------
def test_only_the_default_port_falls_back():
    from sopa_companion.cli import candidate_ports, listen
    assert candidate_ports(8777) == list(range(8777, 8877))
    assert candidate_ports(8821) == [8821] and candidate_ports(8810) == [8810]
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"Port {port} on 127.0.0.1 is in use"):
            listen("127.0.0.1", port)
    sock = listen("127.0.0.1", port)          # free again: exactly that port
    try:
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


def test_an_engine_on_a_busy_port_refuses_to_start(tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        config = engine_toml(tmp_path / "engine.toml", port, tmp_path / "var", tmp_path / "state")
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        env["PYTHONPATH"] = str(HERE.parents[1])
        result = subprocess.run([sys.executable, "-m", "sopa_companion", "--config", str(config), "serve", "--port", str(port)],
                                cwd=str(HERE.parents[1]), env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and f"Port {port} on 127.0.0.1 is in use" in result.stderr, result.stderr[-2000:]
    assert not (tmp_path / "state" / "run" / "server.json").exists()


# ---- zarr save locks ----------------------------------------------------------------------------------------------------
def test_stale_save_locks_are_pruned_and_held_or_recent_ones_kept(tmp_path, monkeypatch):
    from sopa_companion.zarr_bridge import helper
    folder = Path(os.environ["SOPA_COMPANION_ZARR_LOCK_DIR"])
    assert [p.parent for p in helper._lock_paths(tmp_path / "a.zarr")] == [folder]   # the tests' private folder alone
    old = time.time() - 2 * helper.LOCK_STALE_S
    stale, held, recent = (helper._lock_paths(tmp_path / name)[0] for name in ("stale.zarr", "held.zarr", "recent.zarr"))
    for path in (stale, held, recent):
        path.write_text("")
    for path in (stale, held):
        os.utime(path, (old, old))
    with open(held) as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert helper.prune_locks() == [stale.name]
    assert not stale.exists() and held.exists() and recent.exists() and os.stat(held).st_mtime == pytest.approx(old)
    assert sorted(helper.prune_locks(older_than_s=0)) == sorted([held.name, recent.name])
    monkeypatch.setenv("SOPA_COMPANION_ZARR_LOCK_DIR", str(tmp_path / "never-saved"))
    assert helper.prune_locks() == [] and not (tmp_path / "never-saved").exists()      # pruning creates nothing


def test_a_lock_pruned_between_open_and_flock_is_taken_again(tmp_path, monkeypatch):
    from sopa_companion.zarr_bridge import helper
    [target] = helper._lock_paths(tmp_path / "s.zarr")
    opened = []

    def opening(path, mode="r", *args, **kwargs):
        handle = open(path, mode, *args, **kwargs)
        if not opened:
            os.unlink(path)                    # prune_locks got in between our open and our flock
        opened.append(os.fstat(handle.fileno()).st_ino)
        return handle

    monkeypatch.setattr(helper, "open", opening, raising=False)
    with helper._store_lock(tmp_path / "s.zarr"):
        assert len(opened) == 2 and os.stat(target).st_ino == opened[-1]
        with open(target) as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


LOCKER = """
import os, sys, time
sys.path.insert(0, {root!r})
from sopa_companion.zarr_bridge import helper
helper.progress = lambda *a, **k: None
marker = {marker!r}
for _ in range({rounds}):
    with helper._store_lock({store!r}):
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)       # fails if another holder is inside
        time.sleep(0.0005)
        os.close(fd)
        os.unlink(marker)
print("ok")
"""

PRUNER = """
import sys, time
sys.path.insert(0, {root!r})
from sopa_companion.zarr_bridge import helper
deadline = time.monotonic() + 20
pruned = 0
while time.monotonic() < deadline and not __import__("os").path.exists({stop!r}):
    pruned += len(helper.prune_locks(older_than_s=0))
print(pruned)
"""


def test_save_locks_exclude_each_other_while_being_pruned(tmp_path):
    """Two savers of one store and an aggressive pruner: never two inside, and the pruner did unlink files."""
    root, store = str(HERE.parents[1]), str(tmp_path / "s.zarr")
    marker, stop = str(tmp_path / "inside"), str(tmp_path / "stop")
    lockers = [subprocess.Popen([sys.executable, "-c", LOCKER.format(root=root, marker=marker, rounds=300, store=store)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    pruner = subprocess.Popen([sys.executable, "-c", PRUNER.format(root=root, stop=stop)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
    try:
        outputs = [p.communicate(timeout=60) for p in lockers]
    finally:
        Path(stop).write_text("")
        pruned, _ = pruner.communicate(timeout=60)
    assert all(p.returncode == 0 and out.strip() == "ok" for p, (out, _) in zip(lockers, outputs)), outputs
    assert int(pruned) > 0
