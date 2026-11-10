"""Go deeper through the API and the WebSocket (server/app.py): the answer
states from the real check (fake trainer), 404/422/503, the kept answer by GET (immutable only with ?model=), the
history, DELETE, POST /samples/{id}/qc {detail: true}, the "deeper" events of the watched run, and a deleted run
taking its answers with it. The analysis scheduler is recorded instead of run (tests/test_deeper_scheduler.py runs it).
Answers and the frames' bytes live in the folder of results on proseg's own count (engine/analysis.py qc_folder)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import wait_for
from sopa_companion.engine.deeper import qc_folder
from test_qc_api import BASE, NCELLS, auth, deliver, make, ready_run, world  # noqa: F401  (world: a fixture)


def delivered(qc, run_id, seq, judged, mixed):
    """test_qc_api's deliver (the frame's result handed to the check), its bytes where Go deeper reads the worker's
    files of proseg's count (the check's own folder is that one once both keep their results there)."""
    values = deliver(qc, run_id, seq, judged, mixed)
    state = qc._state(run_id)
    made, read = state.dir / f"{seq:06d}.qc.bin", qc_folder(qc.runs.get(run_id).dir, state.key) / f"{seq:06d}.qc.bin"
    if made.resolve() != read.resolve():
        read.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(made, read)
    return values


class Scheduler:
    """The analysis scheduler's Go deeper side, recorded."""
    available = True

    def __init__(self):
        self.calls = []

    def request_deeper(self, run_id, seq, cell, follow=None):
        self.calls.append(("request", run_id, seq, cell, follow))

    def follow_deeper(self, run_id, cell):
        self.calls.append(("follow", run_id, cell))

    def stop_deeper(self, run_id):
        self.calls.append(("stop", run_id))


@pytest.fixture
def deep_app(world):  # noqa: F811
    """The app with its check on (test_qc_api's world), Go deeper's scheduler recorded, and the model's details under
    the test's control: control["deep"] (deep.npz kept or not) and control["detail"] (the details job's state)."""
    cfg, sample = world
    application = make(cfg)
    deeper = application.state.deeper
    deeper.analysis = Scheduler()
    control = {"deep": True, "detail": {"state": "none"}}
    qc, describe = application.state.qc, application.state.qc.describe

    def described(run_id):
        view = describe(run_id)
        if isinstance(view.get("model"), dict):
            view["model"]["deep"] = control["deep"]
            if not control["deep"]:                           # what the worker would read: no details either
                for found in Path(view["model"].get("dir") or application.state.qc_models.root).rglob("deep.npz"):
                    found.unlink()
        view["detail"] = control["detail"]
        return view
    qc.describe = described
    with TestClient(application, base_url=BASE) as client:
        yield application, client, sample, control


def post(client, run_id, body, status=200):
    reply = client.post(f"/api/v1/runs/{run_id}/qc/deeper", json=body, headers=auth())
    assert reply.status_code == status, reply.text
    return reply.json()


def test_go_deeper_through_the_api(deep_app):
    application, client, sample, control = deep_app
    deeper, scheduler = application.state.deeper, application.state.deeper.analysis
    first = post(client, "run-a", {"seq": 3, "cell": 7})
    assert first["state"] in ("waiting", "learning", "working"), "its check is on: the sample is being learned"
    ready_run(client)
    model = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["model"]["id"]
    # The details: not kept (a sample learned before Go deeper existed), being learned, failed.
    control.update(deep=False)
    found = post(client, "run-a", {"seq": 3, "cell": 7})
    assert (found["state"], found["model"], found["deep"]) == ("needs_detail", model, False)
    assert found["estimate_s"] > 0
    control["detail"] = {"state": "learning", "progress": {"step": 1, "of": 3}, "eta_s": 40}
    found = post(client, "run-a", {"seq": 3, "cell": 7})
    assert (found["state"], found["progress"], found["eta_s"]) == ("detail", {"step": 1, "of": 3}, 40)
    control["detail"] = {"state": "failed", "reason": "detail_mismatch"}
    assert post(client, "run-a", {"seq": 3, "cell": 7})["reason"] == "detail_mismatch"
    assert not [c for c in scheduler.calls if c[0] == "request"] or first["state"] == "working"
    scheduler.calls.clear()
    # The details are there: the cell is worked out (its answer comes by WebSocket), then kept.
    control.update(deep=True, detail={"state": "ready"})
    found = post(client, "run-a", {"seq": 3, "cell": 7, "follow": True})
    assert found == {"seq": 3, "cell": 7, "model": model, "deep": True, "state": "working"}
    assert scheduler.calls == [("request", "run-a", 3, 7, True)]
    url = "/api/v1/runs/run-a/qc/deeper/3/7"
    assert client.get(url, headers=auth()).status_code == 404, "not worked out yet"
    result = {"v": 1, "run": "run-a", "seq": 3, "iter": 30, "cell": "c7", "model": model, "state": "ready",
              "transcripts": 682, "code": 116, "parts": [0.54, 0.459, 0.001], "ms": 1.2}
    deeper.deeper_result({"run": "run-a", "seq": 3, "iter": 30, "cell": 7, "model": model, "explicit": True,
                          "result": result})
    found = post(client, "run-a", {"seq": 3, "cell": 7})
    assert found["state"] == "ready" and found["result"] == result and found["iter"] == 30
    # Kept for good only when the URL names the model and the count (the harness's and the tests' route: a frame
    # counted again, a pause's, the hold's, a finish's or the end's count, has another answer); another count than the
    # frame's: 404.
    plain = client.get(url, headers=auth())
    modelled = client.get(f"{url}?model={model}", headers=auth())
    pinned = client.get(f"{url}?model={model}&count=averaged", headers=auth())
    assert plain.json() == modelled.json() == pinned.json() == result
    assert "no-cache" in plain.headers["cache-control"] and "no-cache" in modelled.headers["cache-control"]
    assert "immutable" in pinned.headers["cache-control"]
    refused = client.get(f"{url}?model={model}&count=sampled", headers=auth())
    assert refused.status_code == 404 and refused.json()["detail"] == "This iteration's count changed"
    assert client.get(f"{url}?model=0123", headers=auth()).status_code == 404
    assert client.get("/api/v1/runs/run-a/qc/deeper/start/7", headers=auth()).status_code == 404
    folder = application.state.runs.get("run-a").dir / "qc" / f"{model[:8]}.pc1" / "deeper"
    wait_for(lambda: (folder / "000003-7.json").exists())
    # Refused requests change nothing.
    calls = list(scheduler.calls)
    for run_id, body, status in (("nope", {"seq": 1, "cell": 1}, 404), ("run-a", {"seq": 99, "cell": 1}, 404),
                                 ("run-a", {"seq": 1, "cell": NCELLS}, 404), ("run-a", {"seq": 1}, 422),
                                 ("run-a", {"seq": "last", "cell": 1}, 422), ("run-a", [1, 2], 422),
                                 ("run-a", {"seq": 1, "cell": 1, "follow": 1}, 422)):
        post(client, run_id, body, status)
    ready_run(client, "run-nuclei")
    ready_run(client, "run-paired")
    # No start line but for a start from cells alone (run-a's): from nuclei only, or Xenium's cells with their nuclei.
    for run_id, kind in (("run-nuclei", "nuclei"), ("run-paired", "paired")):
        refused = post(client, run_id, {"seq": "start", "cell": 1}, 404)
        assert (refused["reason"], refused["start_kind"]) == ("no_start", kind)
        assert refused["detail"].endswith("it has no start line")
    scheduler.available = False
    post(client, "run-a", {"seq": 1, "cell": 1}, 503)
    # Where Clusters and the UMAP cannot run, the refusal says so, in the scheduler's words (unavailable_why).
    scheduler.unavailable_why = lambda: "this Python environment lacks igraph"
    refused = post(client, "run-a", {"seq": 1, "cell": 1}, 503)
    assert refused["detail"] == "Go deeper needs Clusters and the UMAP, which can't run here: this Python environment " \
                                "lacks igraph"
    assert refused["reason"] == "analysis"          # the window's own words for it (any other 503 is a failure)
    del scheduler.unavailable_why
    scheduler.available = True
    assert scheduler.calls == calls
    assert client.post("/api/v1/runs/run-a/qc/deeper", json={"seq": 1, "cell": 1},
                       headers={"Authorization": "Bearer " + "t" * 40}).status_code == 403   # Origin rules hold
    # The check switched off: Go deeper says so and asks nothing.
    client.post("/api/v1/runs/run-a/qc", json={"enabled": False}, headers=auth())
    assert post(client, "run-a", {"seq": 3, "cell": 7})["state"] == "off"
    client.post("/api/v1/runs/run-a/qc", json={"enabled": True}, headers=auth())
    # The window closed.
    assert client.delete("/api/v1/runs/run-a/qc/deeper", headers=auth()).status_code == 204
    assert scheduler.calls[-1] == ("stop", "run-a")
    assert client.delete("/api/v1/runs/nope/qc/deeper", headers=auth()).status_code == 404


def test_a_frame_answered_as_another_is_never_cached_for_good(deep_app):
    """The kept answer of a frame without proseg's count of its own is the counted frame's it snaps to, and is never sent
    as immutable, even with ?model=: its own count can come after its record (a request frame's, a pause's), and then
    the same URL answers with the frame's own, which is."""
    application, client, sample, control = deep_app
    deeper, scheduler = application.state.deeper, application.state.deeper.analysis
    ready_run(client)
    model = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["model"]["id"]

    def result(seq):
        return {"v": 1, "run": "run-a", "seq": seq, "iter": seq * 10, "cell": "c7", "model": model, "state": "ready",
                "transcripts": 682, "code": 116 + seq, "parts": [0.54, 0.459, 0.001], "ms": 1.2}

    def keep(seq):
        deeper.deeper_result({"run": "run-a", "seq": seq, "iter": seq * 10, "cell": 7, "model": model, "explicit": True,
                              "result": result(seq)})
    keep(3)
    scheduler.snap_deeper = lambda run_id, seq, kept=None: 3 if seq == 4 else seq      # frame 4 not counted yet
    url = f"/api/v1/runs/run-a/qc/deeper/4/7?model={model}&count=averaged"
    snapped = client.get(url, headers=auth())
    assert snapped.json() == result(3) and snapped.headers["cache-control"] == "private, no-cache"
    own = client.get(f"/api/v1/runs/run-a/qc/deeper/3/7?model={model}&count=averaged", headers=auth())
    assert own.json() == result(3) and own.headers["cache-control"] == "private, max-age=31536000, immutable"
    # Frame 4 counted since: the same URL waits for its own answer, then gives it, for good.
    scheduler.snap_deeper = lambda run_id, seq, kept=None: seq
    assert client.get(url, headers=auth()).status_code == 404
    keep(4)
    found = client.get(url, headers=auth())
    assert found.json() == result(4) and found.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_the_history_and_learning_the_details(deep_app, monkeypatch):
    application, client, sample, control = deep_app
    qc = application.state.qc
    ready_run(client)
    assert client.get("/api/v1/runs/run-a/qc/deeper/history?cell=0", headers=auth()).json() == \
        {"model": qc._state("run-a").key, "start": None, "points": []}, "nothing scored yet"
    for seq, judged, mixed in ((3, 40, 4), (5, 40, 8)):
        delivered(qc, "run-a", seq, judged, mixed)             # cell 0 is mixed (byte 200), cell 30 judged (20)
    found = client.get("/api/v1/runs/run-a/qc/deeper/history?cell=0", headers=auth()).json()
    assert found == {"model": qc._state("run-a").key, "start": None, "points": [[30, 200], [50, 200]]}
    assert client.get("/api/v1/runs/run-a/qc/deeper/history?cell=30&points=1", headers=auth()).json()["points"][-1] \
        == [50, 20]
    for bad in ("cell=-1", f"cell={NCELLS}"):
        assert client.get(f"/api/v1/runs/run-a/qc/deeper/history?{bad}", headers=auth()).status_code == 404
    assert client.get("/api/v1/runs/run-a/qc/deeper/history", headers=auth()).status_code == 422
    # POST /samples/{id}/qc {detail: true}: the window's "Learn the details" (and its "Try again").
    asked = []
    monkeypatch.setattr(application.state.qc_models, "ensure",
                        lambda sid, retry=False, detail=False: asked.append((sid, retry, detail)) or {"sample": sid})
    url = f"/api/v1/samples/{sample['id']}/qc"
    assert client.post(url, json={"detail": True}, headers=auth()).json() == {"sample": sample["id"]}
    assert client.post(url, json={"learn": True}, headers=auth()).status_code == 200
    assert asked == [(sample["id"], True, True), (sample["id"], True, False)]
    assert client.post(url, json={"detail": "yes"}, headers=auth()).status_code == 422 and len(asked) == 2


def test_pages_hear_go_deeper_answers_over_the_websocket(deep_app):
    application, client, sample, control = deep_app
    deeper = application.state.deeper
    ready_run(client)
    model = client.get("/api/v1/runs/run-a/qc", headers=auth()).json()["model"]["id"]
    ws_headers = {"Host": "127.0.0.1:8799", "Origin": BASE, "Authorization": "Bearer " + "t" * 40}
    with client.websocket_connect("/api/v1/ws", headers=ws_headers) as ws:
        ws.send_json({"op": "watch", "run": "run-a", "since": 0})
        for _ in range(400):                                  # the watch is taken once the run is pushed
            event = ws.receive_json()
            if event["ev"] == "run" and (event.get("run") or {}).get("id") == "run-a":
                break
        wait_for(lambda: deeper.watchers.get("run-a") == 1)
        assert post(client, "run-a", {"seq": 4, "cell": 9, "follow": True})["state"] == "working"
        deeper.deeper_result({"run": "run-a", "seq": 4, "iter": 40, "cell": 9, "model": model, "explicit": True,
                              "result": {"v": 1, "state": "ready", "cell": "c9", "ms": 0.9}})
        deeper.deeper_result({"run": "run-a", "seq": 5, "iter": 50, "cell": 9, "model": model, "explicit": False,
                              "error": "the GPU worker stopped", "reason": "worker"})
        heard = []
        for _ in range(400):
            event = ws.receive_json()
            if event["ev"] == "deeper" and event.get("run") == "run-a":
                heard.append(event)
            if len(heard) == 2:
                break
    answer, failure = heard
    assert (answer["seq"], answer["iter"], answer["cell"], answer["model"], answer["explicit"]) == (4, 40, 9, model, True)
    assert answer["result"]["cell"] == "c9" and answer["worker_ms"] == 0.9 and answer["latency_ms"] >= 0
    assert failure == {"ev": "deeper", "run": "run-a", "seq": 5, "cell": 9, "state": "error", "error": "worker",
                       "explicit": False}
    wait_for(lambda: "run-a" not in deeper.watchers)          # the page went: the run is unwatched


def test_a_deleted_run_takes_its_answers_with_it(deep_app):
    application, client, sample, control = deep_app
    deeper = application.state.deeper
    ready_run(client, "run-b")
    model = client.get("/api/v1/runs/run-b/qc", headers=auth()).json()["model"]["id"]
    deeper.deeper_result({"run": "run-b", "seq": 2, "iter": 20, "cell": 1, "model": model, "explicit": True,
                          "result": {"v": 1, "state": "ready"}})
    folder = application.state.runs.get("run-b").dir
    wait_for(lambda: (folder / "qc" / f"{model[:8]}.pc1" / "deeper" / "000002-1.json").exists())
    assert client.delete("/api/v1/runs/run-b", headers=auth()).status_code == 200
    assert not folder.exists() and all(key[0] != "run-b" for key in deeper.cache)
    assert deeper.analysis.calls[-1] == ("stop", "run-b")
    assert client.get("/api/v1/runs/run-b/qc/deeper/2/1", headers=auth()).status_code == 404
